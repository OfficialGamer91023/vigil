"""The portfolio manager — with a fake OpenAI client, never the network.

The suite's rule that no test may reach Alpaca applies to OpenAI just as hard
(conftest strips the key autouse). Every test here injects a double whose
`responses.create` returns a canned answer or raises, so the manager's control
flow — pick, stand down, out-of-range, error, invalid JSON — is exercised without
a socket. The one invariant under all of them: `select` always returns a real,
kernel-approved proposal, and never raises.
"""

from __future__ import annotations

from typing import Any

import openai

from vigil.agent.manager import PortfolioManager, build_manager
from vigil.agent.schema import PMSelection, strict_json_schema
from vigil.config import AgentConfig, risk_config, strategy_config

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class _InDetails:
    def __init__(self, cached: int, write: int) -> None:
        self.cached_tokens = cached
        self.cache_write_tokens = write


class _OutDetails:
    def __init__(self, reasoning: int) -> None:
        self.reasoning_tokens = reasoning


class _Usage:
    def __init__(self) -> None:
        self.input_tokens = 1000
        self.output_tokens = 40
        self.input_tokens_details = _InDetails(cached=800, write=200)
        self.output_tokens_details = _OutDetails(reasoning=10)


class _Response:
    def __init__(self, text: str, usage: Any | None) -> None:
        self.output_text = text
        self.usage = usage


class _Responses:
    """Records the kwargs it was called with, then returns/raises as configured."""

    def __init__(self, *, text: str | None = None, error: Exception | None = None) -> None:
        self._text = text
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        assert self._text is not None
        return _Response(self._text, _Usage())


class _Client:
    def __init__(self, responses: _Responses) -> None:
        self.responses = responses


def _config() -> AgentConfig:
    return AgentConfig(
        model="gpt-5.5-test", entry_effort="low", review_effort="high",
        timeout_seconds=5.0, max_output_tokens=100, enabled=True,
    )


def _manager(*, text: str | None = None, error: Exception | None = None) -> PortfolioManager:
    return PortfolioManager(_Client(_Responses(text=text, error=error)), _config())


async def _select(manager: PortfolioManager, candidates, state, **kw):
    """`select` with the real config objects and a fallback that returns the LAST
    candidate — chosen so a test can tell a model pick from a fallback at a glance.
    """
    return await manager.select(
        candidates, state,
        regime="trend_up", cold_start=False,
        risk=risk_config(), strategy=strategy_config(),
        fallback=lambda cs: cs[-1],
        **kw,
    )


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #

async def test_a_valid_pick_returns_the_chosen_candidate(
    put_credit_spread, iron_condor, flat_book
):
    body = PMSelection(
        stand_down=False, choice=0, confidence=0.8, memo="best credit/width"
    ).model_dump_json()
    manager = _manager(text=body)

    sel = await _select(manager, [put_credit_spread, iron_condor], flat_book)

    assert sel.proposal is put_credit_spread      # index 0, not the fallback (index -1)
    assert sel.fell_back is False
    assert sel.stood_down is False
    assert sel.memo == "best credit/width"
    assert sel.confidence == 0.8


async def test_token_usage_is_captured_for_the_memo(put_credit_spread, iron_condor, flat_book):
    body = PMSelection(stand_down=False, choice=1, confidence=0.5, memo="x").model_dump_json()
    sel = await _select(_manager(text=body), [put_credit_spread, iron_condor], flat_book)

    # The counters map straight onto the llm_memos columns (§6.2).
    assert sel.input_tokens == 1000
    assert sel.cached_tokens == 800
    assert sel.cache_write_tokens == 200
    assert sel.output_tokens == 40
    assert sel.reasoning_tokens == 10
    assert sel.latency_ms is not None


# --------------------------------------------------------------------------- #
# Every failure mode resolves to a real trade, never an exception
# --------------------------------------------------------------------------- #

async def test_stand_down_enters_nothing_but_does_not_fall_back(
    put_credit_spread, iron_condor, flat_book
):
    body = PMSelection(
        stand_down=True, choice=0, confidence=0.1, memo="nothing worth it"
    ).model_dump_json()
    sel = await _select(_manager(text=body), [put_credit_spread, iron_condor], flat_book)

    assert sel.stood_down is True
    assert sel.fell_back is False          # a deliberate decline is not a failure
    assert sel.memo == "nothing worth it"


async def test_an_out_of_range_index_falls_back_to_the_ranker(
    put_credit_spread, iron_condor, flat_book
):
    body = PMSelection(
        stand_down=False, choice=9, confidence=0.9, memo="off the end"
    ).model_dump_json()
    sel = await _select(_manager(text=body), [put_credit_spread, iron_condor], flat_book)

    assert sel.fell_back is True
    assert sel.proposal is iron_condor     # the fallback: last candidate
    assert sel.stood_down is False


async def test_a_timeout_falls_back(put_credit_spread, iron_condor, flat_book):
    manager = _manager(error=openai.APITimeoutError(request=None))  # type: ignore[arg-type]
    sel = await _select(manager, [put_credit_spread, iron_condor], flat_book)

    assert sel.fell_back is True
    assert sel.proposal is iron_condor


async def test_invalid_json_falls_back(put_credit_spread, iron_condor, flat_book):
    manager = _manager(text="not json at all")
    sel = await _select(manager, [put_credit_spread, iron_condor], flat_book)

    assert sel.fell_back is True
    assert sel.proposal is iron_condor


async def test_confidence_is_clamped_into_the_unit_interval(
    put_credit_spread, iron_condor, flat_book
):
    body = PMSelection(
        stand_down=False, choice=0, confidence=9.9, memo="overconfident"
    ).model_dump_json()
    sel = await _select(_manager(text=body), [put_credit_spread, iron_condor], flat_book)
    assert sel.confidence == 1.0


# --------------------------------------------------------------------------- #
# The request itself: cache-friendly ordering and the strict contract
# --------------------------------------------------------------------------- #

async def test_prompt_puts_stable_system_first_and_volatile_state_last(
    put_credit_spread, iron_condor, flat_book
):
    """§6.2: the prefix must be byte-identical across cycles or nothing caches.

    So the system message may not contain any market state, and the volatile menu
    must live in the trailing user message.
    """
    body = PMSelection(stand_down=False, choice=0, confidence=0.5, memo="x").model_dump_json()
    manager = _manager(text=body)
    await _select(manager, [put_credit_spread, iron_condor], flat_book)

    call = manager._client.responses.calls[0]  # type: ignore[attr-defined]
    messages = call["input"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    # The regime read and the menu change every cycle — they must live in the
    # trailing user message, never in the cached system prefix. (The role text's
    # fixed "$100,000 account" is a constant and caches fine, so it is not the
    # thing to check; the *volatile* regime is.)
    assert "trend_up" not in messages[0]["content"]
    assert "put_credit_spread" not in messages[0]["content"]
    assert "trend_up" in messages[1]["content"]
    assert "put_credit_spread" in messages[1]["content"]
    # Effort is passed as the cost dial, not a model swap.
    assert call["reasoning"] == {"effort": "low"}


async def test_the_output_schema_is_hardened_for_strict_mode():
    fmt = strict_json_schema(PMSelection, "pm_selection")["format"]
    assert fmt["strict"] is True
    assert fmt["schema"]["additionalProperties"] is False
    # Strict mode requires *every* property in `required`, not just the mandatory ones.
    assert set(fmt["schema"]["required"]) == {"stand_down", "choice", "confidence", "memo"}


# --------------------------------------------------------------------------- #
# The switch
# --------------------------------------------------------------------------- #

def test_build_manager_is_none_without_a_key(monkeypatch):
    """No key -> the deterministic path, not a crash (§6.3)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert build_manager(_config()) is None


def test_build_manager_is_none_when_disabled(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-real")
    disabled = AgentConfig(
        model="gpt-5.5", entry_effort="low", review_effort="high",
        timeout_seconds=5.0, max_output_tokens=100, enabled=False,
    )
    assert build_manager(disabled) is None
