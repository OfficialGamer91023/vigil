"""Position management (PLAN §4.6). **Runs before any entry logic, always.**

Protecting capital outranks deploying it, so the cycle is
`sense -> reconcile -> manage -> think -> gate -> act` and this module is the
`manage` step.

**Deterministic, and deliberately so.** CLAUDE.md: *the LLM is never on the path
for closing a position.* A model that is slow, rate-limited or creative must not
be able to delay an exit, so nothing here consults one. The decision functions are
pure over `(structure, spot, now, config)` — the same discipline as the risk
kernel, for the same reason: they have to be testable without a broker and
impossible to talk out of a stop.

**The three exits (§4.4.1), and the one that is deliberately absent.**

1. The resting GTC profit target at 50% — placed at entry (§2.6), so it is not a
   sweep decision at all. The sweep's job is to notice when one is *missing*.
2. **Short-strike breach** with more than `breach_exit_min_minutes_left` to run.
3. The **15:40 time stop** on anything expiring today.

There is **no mark-based stop**. §4.4.1 derives why: a `2x credit` stop needs an
80% win rate to break even while touch probability at a 0.16-delta short strike is
~32%, and max loss is already bounded on entry by Gate 2. It bounded nothing and
converted recoverable drawdowns into realized losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from vigil.clock import ET, MARKET_CLOSE
from vigil.config import StrategyConfig, strategy_config
from vigil.domain import OpenStructure, Structure
from vigil.execution.pricing import (
    debit_profit_target_price,
    premium_multiple_target_price,
    profit_target_price,
)


class Action(StrEnum):
    HOLD = "hold"
    CLOSE_BREACH = "close_breach"
    CLOSE_TIME_STOP = "close_time_stop"
    REPLACE_TARGET = "replace_target"


@dataclass(frozen=True, slots=True)
class ManagementDecision:
    """What to do about one open structure, and why.

    Carries the reason as text because this lands in the journal and in the demo:
    "closed 3 structures" is not a story, "closed SPY 761P — spot 760.40 breached
    the short strike with 94 minutes left" is.
    """

    structure: OpenStructure
    action: Action
    reason: str = ""

    @property
    def closes(self) -> bool:
        return self.action in (Action.CLOSE_BREACH, Action.CLOSE_TIME_STOP)


def minutes_to_close(now: datetime) -> int:
    """Minutes until the closing bell, in US/Eastern. Negative after the close.

    Converts before reading any wall-clock field, for the same reason Gate 11
    does: 16:00 is an Eastern fact, and a UTC-aware timestamp of the same instant
    would answer this question four hours wrong.
    """
    et = now.astimezone(ET)
    close = datetime.combine(et.date(), MARKET_CLOSE, tzinfo=ET)
    return int((close - et).total_seconds() // 60)


def resting_target_price(
    structure: OpenStructure, config: StrategyConfig | None = None
) -> Decimal | None:
    """The price to re-rest a lost profit target at (§2.6 repair), or ``None``.

    Reproduces exactly the three branches ``router._rest_profit_target`` runs at
    entry — a credit structure buys back at ``target_pct`` of its credit, a
    long-only sleeve exits at a multiple of the premium, a debit spread sells back
    *above* what it paid — but keyed on the structure's **reconciled** package
    credit rather than the entry fill, because the entry ``Order`` is not what the
    manage sweep has in front of it.

    Two properties make re-resting off the reconciled credit honest rather than a
    guess, which is what the earlier "warn only" repair was rightly cautious about:

    - It rests a **closing** order. A credit a cent off the true fill can only make
      the exit fire slightly early or slightly late; it can lower realized risk and
      never raise it, and max loss stays bounded by Gate 2 either way.
    - It is a **real order the next reconcile reads back from the broker**, so a
      repair can never "look repaired" while being absent — the failure mode the
      original docstring warned about.

    Returns ``None`` — and the caller keeps the §2.6 alarm — when the opening
    credit is unknown (``net_credit == 0``: an adopted position we never priced) or
    the shape cannot be mapped to a target without guessing. A wrong exit is worse
    than none; *no* exit that we admit to is better than a fabricated one.
    """
    cfg = config or strategy_config()
    credit = structure.net_credit
    if credit == 0:
        return None

    # Long-only is read from the legs, not the enum, so it stays right for any
    # long-premium shape added later — the same derivation `TradeProposal` uses.
    long_only = bool(structure.legs) and not any(leg.is_short for leg in structure.legs)
    if structure.structure is Structure.LONG_STRANGLE or long_only:
        # Max profit is unbounded, so "50% of max profit" is undefined — exit at a
        # multiple of the premium paid instead.
        return premium_multiple_target_price(credit, cfg.convexity_profit_target_multiple)

    if credit > 0:
        # Any credit structure — vertical or condor: buy it back at `target_pct`.
        return profit_target_price(credit, cfg.profit_target_pct)

    if structure.structure is Structure.DEBIT_SPREAD and len(structure.strikes) >= 2:
        # A debit spread exits *above* cost. Width is the strike span; a single
        # strike (or none) means we cannot size the target and must not guess.
        width = max(structure.strikes) - min(structure.strikes)
        if width <= 0:
            return None
        return debit_profit_target_price(credit, width, cfg.profit_target_pct)

    return None


def decide(
    structure: OpenStructure,
    *,
    spot: Decimal,
    now: datetime,
    config: StrategyConfig | None = None,
) -> ManagementDecision:
    """The management verdict for one structure. Pure; no network, no LLM.

    Order matters and is not arbitrary — the time stop is checked **first**.
    Auto-exercise makes an expiring position the one thing that must never be held
    (§1.1: anything ITM by $0.01 is exercised), so nothing may outrank it. A
    breached 0DTE structure at 15:41 is closed because it expires today, not
    because it is breached, and the journal should say so.
    """
    cfg = config or strategy_config()
    et = now.astimezone(ET)
    left = minutes_to_close(now)

    if structure.expiry <= et.date() and et.time() >= cfg.time_stop:
        return ManagementDecision(
            structure, Action.CLOSE_TIME_STOP,
            f"expires {structure.expiry}; {et:%H:%M} ET is past the "
            f"{cfg.time_stop:%H:%M} flatten — auto-exercise is not a strategy",
        )

    # Breach is a **thesis-invalidation** signal, not a P&L signal, which is
    # precisely why it survives §4.4.1's deletion of the mark-based stop: it says
    # the trade's premise is wrong, not merely that it is currently losing.
    if structure.has_short_legs and structure.is_breached(spot):
        if left > cfg.breach_exit_min_minutes_left:
            side = "put" if structure.short_put_strikes and spot <= max(
                structure.short_put_strikes) else "call"
            return ManagementDecision(
                structure, Action.CLOSE_BREACH,
                f"spot {spot} breached the short {side} strike with {left} min "
                f"left (> {cfg.breach_exit_min_minutes_left})",
            )
        # Inside the last half hour, closing a breached short-dated structure
        # means crossing a spread at peak gamma into a book that is about to be
        # flattened anyway. Hold and let the time stop do it.
        return ManagementDecision(
            structure, Action.HOLD,
            f"breached but only {left} min left; the {cfg.time_stop:%H:%M} time "
            f"stop handles it more cheaply than crossing the spread now",
        )

    # §2.6: an open structure with no live resting exit is a reconciliation
    # defect, not a style choice. Checked last so a structure that is about to be
    # closed is not first given an exit order it will never use.
    if not structure.has_resting_target:
        return ManagementDecision(
            structure, Action.REPLACE_TARGET,
            "no resting profit-target order at the broker (§2.6 defect)",
        )

    return ManagementDecision(structure, Action.HOLD, "within thesis, target resting")


def sweep(
    structures: tuple[OpenStructure, ...],
    *,
    spots: dict[str, Decimal],
    now: datetime,
    config: StrategyConfig | None = None,
) -> tuple[ManagementDecision, ...]:
    """Decide for every open structure. Closes are ordered first.

    Ordering matters to the executor rather than to the decision: when the sweep
    is only part-way through and the process dies, the closes are the actions we
    most want to have already happened.

    A structure whose underlying has no spot is **held**, not closed. A missing
    quote is a data gap, and closing a position because we could not price its
    underlying would turn a feed hiccup into realized losses.
    """
    out: list[ManagementDecision] = []
    for s in structures:
        spot = spots.get(s.underlying)
        if spot is None:
            out.append(ManagementDecision(
                s, Action.HOLD, f"no spot for {s.underlying}; holding rather than "
                                f"acting on a data gap"))
            continue
        out.append(decide(s, spot=spot, now=now, config=config))
    return tuple(sorted(out, key=lambda d: not d.closes))


def is_flatten_time(now: datetime, config: StrategyConfig | None = None) -> bool:
    """Has the hard flatten passed? Eastern, always."""
    cfg = config or strategy_config()
    return now.astimezone(ET).time() >= cfg.time_stop


def stale_entry_cutoff(now: datetime) -> datetime:
    """Entry orders older than this are stale and should be cancelled.

    The price ladder submits `day` orders (§2.5), so the broker expires them at
    the close. This is the *intra-session* rule: an entry working for more than
    one ladder's worth of time is one whose regime read no longer applies, and
    the reconcile step cancels it rather than letting it fill on a thesis that has
    since expired.
    """
    from vigil.execution.pricing import MAX_RUNGS, RUNG_WAIT_SECONDS

    return now - timedelta(seconds=MAX_RUNGS * RUNG_WAIT_SECONDS * 3)
