"""The portfolio manager: a Responses-API call wrapped in a promise it cannot break.

Every path through `select` returns a real, kernel-approved `TradeProposal` and a
`Selection` describing how it was chosen. The model is given a menu and a hard time
budget; if it is slow, errors, returns invalid JSON, or points at an index that is
not on the menu, the deterministic ranker decides instead and `fell_back` is set.
There is no retry loop and no raise — §6.3 makes the model advisory, never
load-bearing, and the way you keep that promise is by making the failure path a
first-class return value rather than an exception someone upstream has to remember
to catch.

**The client is injected, never constructed here at import time.** That is what
lets the whole entry path be unit-tested with a fake that returns a canned response
(valid, invalid, or raising) and never opens a socket — the test suite must not be
able to reach OpenAI any more than it can reach Alpaca.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from vigil.agent.prompt import menu_prompt, system_prompt
from vigil.agent.schema import PMSelection, strict_json_schema
from vigil.config import AgentConfig, RiskConfig, StrategyConfig
from vigil.domain import PortfolioState, TradeProposal
from vigil.logging import get_logger

log = get_logger(__name__)

# The deterministic chooser the caller owns. The manager never ranks itself — it
# calls back into the same `_rank` the worker uses, so "the model fell back" and
# "the model was off" produce byte-identical trades. A fallback that ranked
# differently would be a second strategy nobody had tested.
Fallback = Callable[[list[TradeProposal]], TradeProposal]


class _Tokens(TypedDict):
    """The five usage counters, typed so `**tokens` into `Selection` type-checks.

    A plain `dict[str, int | None]` unpacked with `**` lets mypy believe it could
    fill `Selection`'s bool fields too; a TypedDict pins each key to its field.
    """

    input_tokens: int | None
    cached_tokens: int | None
    cache_write_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None


class _ResponsesClient(Protocol):
    """The sliver of the OpenAI async client this module touches.

    Declared as a Protocol so a test double satisfies it structurally — there is
    no base class to import and no `AsyncOpenAI` instance required to type-check.
    """

    @property
    def responses(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class Selection:
    """The chosen trade plus everything `llm_memos` needs to stay honest.

    `fell_back` is the headline reliability number (§6.3): it is set whenever the
    deterministic path decided, whether because the model was disabled, timed out,
    errored, or answered out of contract. `cached_tokens` is the headline cost
    number (§6.2): it is how the prompt-ordering discipline is *measured* rather
    than assumed.
    """

    proposal: TradeProposal
    fell_back: bool
    model: str
    effort: str
    memo: str = ""
    confidence: float | None = None
    stood_down: bool = False
    latency_ms: int | None = None
    input_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None


class PortfolioManager:
    """Selects one proposal from a pre-approved menu, or defers to the ranker."""

    def __init__(self, client: _ResponsesClient, config: AgentConfig) -> None:
        self._client = client
        self._config = config

    async def select(
        self,
        candidates: list[TradeProposal],
        state: PortfolioState,
        *,
        regime: str,
        cold_start: bool,
        risk: RiskConfig,
        strategy: StrategyConfig,
        fallback: Fallback,
        effort: str | None = None,
    ) -> Selection:
        """Ask the model to pick; return the pick or the deterministic default.

        `candidates` is assumed non-empty and already kernel-approved — the caller
        does not invoke the model when the menu is empty, because there is nothing
        to choose and a call would only cost latency.
        """
        effort = effort or self._config.entry_effort
        default = fallback(candidates)  # computed up front so every failure path is trivial

        try:
            parsed, usage, latency_ms = await self._ask(
                candidates, state, regime=regime, cold_start=cold_start,
                risk=risk, strategy=strategy, effort=effort,
            )
        except Exception as exc:  # noqa: BLE001 — deliberately total: any failure is a fallback
            # The model having a bad minute must never stop the desk. Logged, not
            # raised, and the deterministic trade goes ahead.
            log.warning("agent.fell_back", reason=type(exc).__name__, detail=str(exc)[:200])
            return Selection(
                proposal=default, fell_back=True,
                model=self._config.model, effort=effort,
                memo="deterministic fallback (model unavailable)",
            )

        return self._resolve(parsed, candidates, default, usage, latency_ms, effort)

    # ----------------------------------------------------------------------- #
    # The call, and the parse. Separated so `select` reads as pure control flow.
    # ----------------------------------------------------------------------- #

    async def _ask(
        self,
        candidates: list[TradeProposal],
        state: PortfolioState,
        *,
        regime: str,
        cold_start: bool,
        risk: RiskConfig,
        strategy: StrategyConfig,
        effort: str,
    ) -> tuple[PMSelection, Any, int]:
        """One Responses call. Returns (parsed output, usage, latency_ms).

        Raises on anything — timeout, transport error, or output that does not
        parse against the schema. `select` turns every one of those into a
        fallback, so this method is free to be strict.
        """
        # System prompt first (cacheable), volatile market state last (§6.2). The
        # `input` list order is the byte order the cache keys on, so this ordering
        # is load-bearing, not stylistic.
        sys_text = system_prompt(risk=risk, strategy=strategy)
        user_text = menu_prompt(candidates, state, regime=regime, cold_start=cold_start)

        started = time.monotonic()
        response = await self._client.responses.create(
            model=self._config.model,
            reasoning={"effort": effort},
            input=[
                {"role": "system", "content": sys_text},
                {"role": "user", "content": user_text},
            ],
            text=strict_json_schema(PMSelection, "pm_selection"),
            max_output_tokens=self._config.max_output_tokens,
            timeout=self._config.timeout_seconds,
        )
        latency_ms = int((time.monotonic() - started) * 1000)

        # `output_text` aggregates the model's text output; strict mode guarantees
        # it is JSON matching the schema, but we validate anyway rather than trust
        # the mode — a validation here is cheaper than a bad index at the broker.
        parsed = PMSelection.model_validate_json(response.output_text)
        return parsed, getattr(response, "usage", None), latency_ms

    def _resolve(
        self,
        parsed: PMSelection,
        candidates: list[TradeProposal],
        default: TradeProposal,
        usage: Any,
        latency_ms: int,
        effort: str,
    ) -> Selection:
        """Turn a valid model answer into a Selection, guarding the index.

        A stand-down is honoured by returning the deterministic default but marking
        `stood_down` — the entry cycle reads that flag and enters nothing. An
        out-of-range index is treated as a malformed answer: the ranker decides and
        we record a fallback, because a model pointing off the end of the menu is
        exactly the kind of quiet error strict mode does not catch.
        """
        tokens = self._tokens(usage)
        confidence = _clamp01(parsed.confidence)

        if parsed.stand_down:
            return Selection(
                proposal=default, fell_back=False, stood_down=True,
                model=self._config.model, effort=effort, memo=parsed.memo,
                confidence=confidence, latency_ms=latency_ms, **tokens,
            )

        if not 0 <= parsed.choice < len(candidates):
            log.warning("agent.bad_index", choice=parsed.choice, menu=len(candidates))
            return Selection(
                proposal=default, fell_back=True,
                model=self._config.model, effort=effort,
                memo="deterministic fallback (index out of range)",
                confidence=confidence, latency_ms=latency_ms, **tokens,
            )

        chosen = candidates[parsed.choice]
        log.info(
            "agent.selected", choice=parsed.choice, structure=chosen.structure.value,
            underlying=chosen.underlying, confidence=confidence,
        )
        return Selection(
            proposal=chosen, fell_back=False,
            model=self._config.model, effort=effort, memo=parsed.memo,
            confidence=confidence, latency_ms=latency_ms, **tokens,
        )

    @staticmethod
    def _tokens(usage: Any) -> _Tokens:
        """Pull the token counters off a usage object, tolerating its absence.

        Every field is optional because a fallback has no usage and a future SDK
        might reshape the object — a missing counter should read as "unknown", not
        crash the cycle that was only trying to journal it.
        """
        if usage is None:
            return _Tokens(
                input_tokens=None, cached_tokens=None, cache_write_tokens=None,
                output_tokens=None, reasoning_tokens=None,
            )
        in_details = getattr(usage, "input_tokens_details", None)
        out_details = getattr(usage, "output_tokens_details", None)
        return _Tokens(
            input_tokens=getattr(usage, "input_tokens", None),
            cached_tokens=getattr(in_details, "cached_tokens", None),
            cache_write_tokens=getattr(in_details, "cache_write_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            reasoning_tokens=getattr(out_details, "reasoning_tokens", None),
        )


def _clamp01(value: float) -> float:
    """Confidence is contractually in [0,1]; clamp rather than trust the model."""
    return max(0.0, min(1.0, float(value)))


def build_client(config: AgentConfig | None = None) -> tuple[Any, AgentConfig] | None:
    """The OpenAI async client and config, or `None` when the model is off/unkeyed.

    Shared by `build_manager` (selection) and the journal's social draft (prose),
    so the enable/key decision and the client construction live in exactly one
    place. `None` means "run the deterministic path" — the honest shape, reached
    identically whether the operator disabled the model or never set a key.

    The `openai` import is local so nothing pays for it unless the model is on.
    """
    config = config or _load_config()
    if not config.enabled:
        return None
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        log.info("agent.disabled", reason="no OPENAI_API_KEY — deterministic path only")
        return None

    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=api_key), config


def build_manager(config: AgentConfig | None = None) -> PortfolioManager | None:
    """Construct a portfolio manager, or `None` on the deterministic path."""
    built = build_client(config)
    return None if built is None else PortfolioManager(*built)


def _load_config() -> AgentConfig:
    # Indirect so `build_manager()` with no args uses the cached config, while a
    # test can pass an explicit one without reaching the filesystem.
    from vigil.config import agent_config

    return agent_config()


__all__ = ["PortfolioManager", "Selection", "build_manager", "build_client", "Fallback"]
