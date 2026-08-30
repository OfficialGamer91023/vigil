"""The escalation ladder (PLAN §4.7).

A tournament strategy prices the fact that it is in a tournament; a career one
ignores it. By the back half of the six sessions the agent knows its own P&L and
roughly how many sessions remain, and it tilts two dials accordingly:

- the **convexity share** of the risk budget — the only convex payoff in the book,
  and §4.7 argues convexity is what a rank-based sprint actually pays for; and
- the **core risk per trade**, *within Gate 2's ceiling and never above it*.

Everything here is a pure function of `(sessions_left, day P&L)` against a config
table. It holds no state, touches no network, and reaches no LLM — the ladder is
deterministic config, not a decision the model gets to argue with. It sits in
`strategy/`, not `risk/`, precisely because it may only ever size *down* from the
kernel's limits; it is not itself a gate and does not touch the frozen kernel.

**The clamp is the load-bearing invariant.** The §4.7 table's aggressive rungs
name a 2.5% core figure, but Gate 2's row, the sentence under that same table, and
PRIMER §383 all state the ladder never widens the ceiling. `effective_core_risk_pct`
is where those two are reconciled: the rung is the *request*, Gate 2's threshold is
the *cap*, and `min` of the two is what sizing actually gets. Escalation therefore
expresses itself as more convexity, not as a bigger per-trade loss than the kernel
would ever approve — and Gate 2 re-checks the result regardless, so a mis-tuned
rung is caught, not trusted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from vigil.config import LadderConfig, LadderRung


def sessions_left(today: date, session_dates: Sequence[date]) -> int:
    """How many tournament sessions remain, counting today if it is one.

    A plain count of calendar dates on or after `today`, so on the final session
    this is 1 (there is still a session to trade), not 0. Robust to a `today` that
    has fallen off the end of the calendar — the count is simply 0 — and to a
    `today` that is not itself a session date (a weekend the scheduler somehow
    woke on), because it counts *remaining* dates rather than indexing a position.
    """
    return sum(1 for d in session_dates if d >= today)


@dataclass(frozen=True, slots=True)
class ResolvedRung:
    """The rung the ladder selected, plus the inputs that chose it — carried so a
    single journal line explains *why* this size, not merely what it was."""

    convexity_share: Decimal
    core_risk_pct: Decimal
    sessions_left: int
    ahead: bool
    label: str


def _matches(when: str, ahead: bool) -> bool:
    """Does a rung's P&L condition apply in the current standing?

    `any` is the catch-all a decision table needs at the bottom so there is always
    exactly one answer; `behind` is the escalation trigger; `ahead` protects a
    lead. The membership is validated at config-load time, so an unknown value
    here is a loader bug, not a silent False.
    """
    if when == "any":
        return True
    if when == "ahead":
        return ahead
    return not ahead  # "behind"


def resolve_rung(
    today: date, day_pnl_pct: Decimal, config: LadderConfig
) -> ResolvedRung:
    """Select the ladder rung for today's standing — **first match wins.**

    The rungs are read top to bottom exactly as PLAN §4.7's table is written, and
    the first whose session floor *and* P&L condition both hold is taken. Order is
    the whole semantics: the early `min_sessions_left: 4, when: any` rung claims
    days 1–3 before any escalation rung can, and the trailing `when: any` rung is
    the guaranteed catch-all (an "ahead" session in the back half falls through the
    `behind`-only escalation rungs onto it, and so does the degenerate past-the-
    calendar case). That guarantees a total function: every `(sessions, P&L)` pair
    resolves to exactly one rung.

    "Ahead" is `day P&L ≥ target`; the target is a config proxy for "the day is
    already good enough that protecting the gain beats pressing it", because the
    free tier cannot show us the rivals a literal tournament rank would need.
    """
    left = sessions_left(today, config.session_dates)
    ahead = day_pnl_pct >= config.day_pnl_target_pct

    rung: LadderRung | None = next(
        (
            r
            for r in config.rungs
            if left >= r.min_sessions_left and _matches(r.when, ahead)
        ),
        None,
    )
    if rung is None:
        # Unreachable while the config carries a trailing `when: any, min: 0` rung,
        # which the loader enforces. Failing loud beats sizing off a silent default:
        # a budget of "whatever" is exactly the class of bug the kernel exists to
        # make impossible, so surface it here rather than let it flow into sizing.
        raise ValueError(
            f"no ladder rung matched sessions_left={left} ahead={ahead}; the config "
            f"is missing a catch-all rung (min_sessions_left: 0, when: any)."
        )

    return ResolvedRung(
        convexity_share=rung.convexity_share,
        core_risk_pct=rung.core_risk_pct,
        sessions_left=left,
        ahead=ahead,
        label=(
            f"{left} left / {'ahead' if ahead else 'behind'} "
            f"→ core {rung.core_risk_pct:.1%}, convexity {rung.convexity_share:.0%}"
        ),
    )


def effective_core_risk_pct(rung: ResolvedRung, gate2_ceiling: Decimal) -> Decimal:
    """Core risk the ladder is *allowed* to size to: the rung, capped at Gate 2.

    This is the one place the §4.7 table's aggressive rung and the "never above the
    ceiling" rule meet. The rung is a request; Gate 2's `max_risk_per_trade_pct` is
    the cap; sizing gets the smaller. Keeping the clamp here — rather than trusting
    every call site to remember it — is what makes the invariant a property of the
    ladder itself, and it is asserted directly in the tests.
    """
    return min(rung.core_risk_pct, gate2_ceiling)
