"""Prompt construction, ordered for the cache (§6.2).

**Prompt ordering is the whole game here.** OpenAI's prompt caching is automatic
and prefix-based — there are no breakpoints to declare, so the only lever is byte
order. Anything identical across the day's ~30 entry cycles must come *first* and
anything that changes every cycle must come *last*, or the cached prefix ends at
the first differing byte and every call pays full freight.

So this module builds the prompt in two halves:

- `system_prompt()` — the role, the rules, and the strategy config. Byte-identical
  for the life of the process (the config is `lru_cache`d and does not change under
  a running worker), so it is the cacheable prefix.
- `menu_prompt()` — the regime read, the book, and the candidate menu. Different
  every cycle, so it is the volatile suffix and is never allowed to leak into the
  prefix.

The menu is rendered from the *already-approved* proposals. The model is told, in
those words, that it is choosing from a pre-gated set and cannot add to it — the
guardrail is stated in the prompt as well as enforced in code, because a model
that understands the boundary argues with it less.
"""

from __future__ import annotations

from decimal import Decimal

from vigil.config import RiskConfig, StrategyConfig
from vigil.domain import PortfolioState, TradeProposal

# The stable half of the prefix. No market data, no timestamps, nothing that moves
# cycle to cycle — that is what makes it cache. Written as a module constant rather
# than an f-string so it cannot accidentally interpolate a volatile value.
_ROLE = """\
You are the portfolio manager of Vigil, an autonomous options desk trading a \
$100,000 Alpaca paper account over a six-session tournament. You do not place \
orders and you do not size trades. A deterministic risk kernel has already \
rejected everything unsafe; a deterministic strategy engine has already sized \
what remains. Your one job each cycle is to choose AT MOST ONE structure from a \
menu of candidates the kernel has already approved, or to stand down.

Rules you must reason within:
- You may only pick a candidate from the menu, by its index, or stand down. You \
cannot propose a different structure, strike, width, or size — those are fixed.
- Standing down is a legitimate and common choice. A legal menu is not a reason \
to trade. Prefer standing down when nothing on the menu has a clear edge.
- This is a tournament, not a career: six sessions, paper money, rank-based \
scoring. Favour the candidate with the best compensation per dollar of risk \
(credit as a share of width), and break ties toward the smaller absolute risk.
- confidence is your conviction in [0,1]. It can only ever reduce activity, never \
increase it. If unsure, lower it and consider standing down.
- Your memo is stored verbatim with the trade. Keep it to one or two sentences of \
genuine rationale grounded in the menu and the regime — not boilerplate."""


def system_prompt(*, risk: RiskConfig, strategy: StrategyConfig) -> str:
    """The cacheable prefix: role + the config the desk runs on.

    The config is folded in as plain text because the model reasons better when it
    can see the thresholds its menu was built against — but it is context, not
    instruction: the kernel already applied every one of these numbers, so the
    model cannot violate them even by ignoring them here.
    """
    cfg = (
        "Desk parameters (already enforced by the kernel — shown for context):\n"
        f"- risk per trade: {risk.max_risk_per_trade_pct:%} of equity (hard cap)\n"
        f"- min credit: {risk.min_credit_pct_of_width:%} of width\n"
        f"- max open structures: {risk.max_open_structures}, "
        f"max per underlying: {risk.max_structures_per_underlying}\n"
        f"- portfolio dollar-delta cap: {risk.max_dollar_delta_pct:%} of equity\n"
        f"- short-strike delta target: {strategy.short_delta_target} "
        f"(band {strategy.short_delta_min}-{strategy.short_delta_max})\n"
        f"- days to expiry: {strategy.min_dte}-{strategy.max_dte}\n"
        f"- profit target: {strategy.profit_target_pct:%} of max profit, "
        f"resting at the broker from entry"
    )
    return f"{_ROLE}\n\n{cfg}"


def _fmt_candidate(index: int, p: TradeProposal) -> str:
    """One menu line. Compact but complete enough to choose on."""
    credit = f"{p.credit_pct_of_width:.1%} of width"
    # max_profit is Infinity for long-only structures; say so rather than print it.
    mp = "uncapped" if p.max_profit == Decimal("Infinity") else f"${p.max_profit:.0f}"
    return (
        f"[{index}] {p.structure.value} {p.underlying} exp {p.expiry} "
        f"x{p.contracts} | credit {credit} | max loss ${p.max_loss:.0f} | "
        f"max profit {mp} | $delta {p.dollar_delta:+.0f} | {p.rationale}"
    )


def menu_prompt(
    candidates: list[TradeProposal],
    state: PortfolioState,
    *,
    regime: str,
    cold_start: bool,
) -> str:
    """The volatile suffix: the book, the regime, and the menu. Never cached.

    Kept strictly after the system prompt in the request so the differing bytes
    start as late as possible — everything above this survives in the cache.
    """
    book = (
        f"Book now: equity ${state.equity:.0f}, open structures "
        f"{len(state.open_structures)}, open risk ${state.open_risk:.0f}, "
        f"net $delta {state.net_dollar_delta:+.0f}, day P&L ${state.day_pnl:+.0f}."
    )
    regime_line = f"Regime: {regime}" + (" (cold-start proxy)" if cold_start else "")
    menu = "\n".join(_fmt_candidate(i, p) for i, p in enumerate(candidates))
    return (
        f"{regime_line}\n{book}\n\n"
        f"Approved candidate menu ({len(candidates)}):\n{menu}\n\n"
        "Choose at most one by index, or stand down. Respond with the required "
        "JSON only."
    )
