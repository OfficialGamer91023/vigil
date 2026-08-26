"""The cron table (§2.4), as a pure function of the clock.

**Why this is not `arq`, and why that is a correction rather than a shortcut.**

PLAN §10 specifies `arq` for the worker — "async-native, Redis-backed, has cron
built in. Runs the loop *and* the slow LLM jobs." But CLAUDE.md hard rule #6 and
§2.2 both state, in those words, that *the trading loop must run correctly with
`api`, `web` and Redis all stopped*. arq's cron lives in Redis. With Redis down
an arq worker does not run a degraded loop; it does not run at all.

Those two cannot both hold, so the hard rule wins:

- **The trading loop is scheduled in-process**, by this module and `runner.py`,
  with no Redis dependency and no broker between the clock and a cycle. Redis
  going down costs a chain cache and a dashboard feed, not a position — which is
  exactly what §2.2 says the optional layers are allowed to cost.
- **arq keeps the queued slow LLM jobs**, where it earns its place and where
  Redis being unavailable means falling back to the deterministic path (§6.3)
  rather than stopping.

Keeping the schedule pure — `datetime -> set of due cycles` — is what makes the
whole cron table testable without waiting for 15:40 to come round.
"""

from __future__ import annotations

from datetime import datetime, time
from enum import StrEnum

from vigil.clock import ET


class CycleKind(StrEnum):
    PREMARKET = "premarket"
    OPEN = "open"
    MANAGE = "manage"
    ENTRY = "entry"
    FLATTEN = "flatten"
    POSTCLOSE = "postclose"


PREMARKET_AT = time(8, 45)
OPEN_AT = time(9, 45)
FLATTEN_AT = time(15, 40)
POSTCLOSE_AT = time(16, 15)

MANAGE_FROM, MANAGE_TO = time(9, 35), time(15, 55)
ENTRY_FROM, ENTRY_TO = time(10, 30), time(14, 30)

MANAGE_EVERY_MIN = 15
# §4.6: "short gamma on a 15-minute poll is not risk management, it is sampling."
MANAGE_EVERY_MIN_ZERO_DTE = 5
ENTRY_EVERY_MIN = 30


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def due(
    now: datetime,
    *,
    holding_zero_dte: bool = False,
    is_trading_day: bool = True,
) -> frozenset[CycleKind]:
    """Which cycles are due at `now`. Pure, and Eastern regardless of input zone.

    `holding_zero_dte` tightens the manage sweep from 15 minutes to 5 (§4.6).
    That is a **fact about the book**, not about the clock, so it is passed in —
    the schedule stays a pure function and the caller owns the state.

    `is_trading_day` comes from Alpaca's calendar, never from a weekday check:
    `vigil.clock.is_market_open` says in its own docstring that it does not know
    about holidays, and a holiday the schedule mistook for a session would run a
    full entry cycle into a closed market.
    """
    et = now.astimezone(ET)
    if not is_trading_day:
        return frozenset()

    t = et.time()
    m = _minutes(t)
    out: set[CycleKind] = set()

    if t == PREMARKET_AT:
        out.add(CycleKind.PREMARKET)
    if t == OPEN_AT:
        out.add(CycleKind.OPEN)
    if t == FLATTEN_AT:
        out.add(CycleKind.FLATTEN)
    if t == POSTCLOSE_AT:
        out.add(CycleKind.POSTCLOSE)

    step = MANAGE_EVERY_MIN_ZERO_DTE if holding_zero_dte else MANAGE_EVERY_MIN
    # Anchored to the window's start rather than to process start, so a worker
    # restarted at 10:07 does not spend the day sweeping at :07, :22, :37 while
    # the journal claims a 15-minute cadence.
    in_manage_window = _minutes(MANAGE_FROM) <= m <= _minutes(MANAGE_TO)
    if in_manage_window and (m - _minutes(MANAGE_FROM)) % step == 0:
        out.add(CycleKind.MANAGE)

    in_entry_window = _minutes(ENTRY_FROM) <= m <= _minutes(ENTRY_TO)
    if in_entry_window and (m - _minutes(ENTRY_FROM)) % ENTRY_EVERY_MIN == 0:
        out.add(CycleKind.ENTRY)
        # **§2.3's ordering invariant, enforced structurally.** "Management of
        # existing positions always runs before new-entry logic — protecting
        # capital outranks deploying it."
        #
        # The two cadences never coincide on their own: manage is anchored to
        # 09:35 in 15-minute steps and entry to 10:30 in 30-minute steps, so
        # manage lands on :05/:20/:35/:50 and entry on :00/:30. Left alone, an
        # entry at 11:00 would reason from a book last swept at 10:50 — ten
        # minutes of unreconciled drift, at a moment we are about to add risk.
        # Emitting MANAGE alongside ENTRY makes the invariant a property of the
        # schedule rather than a convention the runner has to remember.
        out.add(CycleKind.MANAGE)

    return frozenset(out)


def next_due_after(
    now: datetime, *, holding_zero_dte: bool = False, horizon_minutes: int = 24 * 60
) -> datetime | None:
    """The next minute at which anything is due. `None` if nothing within horizon.

    Used by the runner to sleep until the next boundary rather than waking every
    second: on a laptop that is a battery choice, on a VM it is the difference
    between a process that looks idle and one that looks like a spin loop.
    """
    from datetime import timedelta

    et = now.astimezone(ET).replace(second=0, microsecond=0)
    for i in range(1, horizon_minutes + 1):
        candidate = et + timedelta(minutes=i)
        if due(candidate, holding_zero_dte=holding_zero_dte):
            return candidate
    return None
