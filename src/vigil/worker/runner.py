"""The in-process scheduler: the loop that actually runs the agent.

**Why not arq.** PLAN §10 specifies arq, but CLAUDE.md hard rule #6 and §2.2 both
require the trading loop to run correctly with `api`, `web` and Redis all
stopped — and arq's cron lives in Redis. With Redis down an arq worker does not
run a degraded loop; it does not run at all. `worker/schedule.py` documents the
resolution: the loop is scheduled here, in process, with no broker between the
clock and a cycle. arq keeps the queued slow LLM jobs, where Redis being
unavailable means falling back to the deterministic path rather than stopping.

**What this module is responsible for, and what it deliberately is not.**

It owns *when*. `schedule.due()` owns *which*, and `sessions.run_cycle` owns
*what*. Keeping those three apart is what lets the whole cron table be tested
without waiting for 15:40 to come round.

It does **not** own retries of a failed broker call, which belong in the call, nor
recovery of a crashed process, which belongs to the container's restart policy.
What it does own is refusing to let one failed cycle end the trading day.

    make worker      (or: python -m vigil.worker.runner)
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from datetime import datetime, timedelta

from vigil.clock import ET, now_et
from vigil.logging import get_logger
from vigil.worker.broker import Broker
from vigil.worker.schedule import CycleKind, due
from vigil.worker.sessions import CycleResult, run_cycle

log = get_logger(__name__)

# Cycles run in this order when several fall due in the same minute. The schedule
# emits MANAGE alongside every ENTRY precisely so the book is swept before risk is
# added (§2.3), and that guarantee is only real if the runner honours the order.
CYCLE_ORDER: tuple[CycleKind, ...] = (
    CycleKind.PREMARKET,
    CycleKind.OPEN,
    CycleKind.FLATTEN,   # before MANAGE: at 15:40 the flatten *is* the management
    CycleKind.MANAGE,
    CycleKind.ENTRY,
    CycleKind.POSTCLOSE,
)


async def is_trading_day(broker: Broker, day: datetime) -> bool:
    """Ask Alpaca's calendar, never a weekday check.

    `clock.is_market_open` says in its own docstring that it does not know about
    holidays, and a holiday the schedule mistook for a session would run a full
    entry cycle into a closed market — placing orders that sit unfilled until the
    next open, on a regime read a day stale.

    A calendar lookup that *fails* returns `False`. The asymmetry is deliberate:
    the cost of skipping a real session is one missed day of entries, and the cost
    of trading a day that is not a session is an unmanaged position.
    """
    from alpaca.trading.requests import GetCalendarRequest

    try:
        calendar = await asyncio.to_thread(
            broker.client.get_calendar,
            GetCalendarRequest(start=day.date(), end=day.date()),
        )
    except Exception as exc:  # noqa: BLE001 — the fallback is the whole point
        log.warning("calendar.unavailable", error=str(exc), assuming="closed")
        return False
    return any(getattr(entry, "date", None) == day.date() for entry in calendar)


async def holding_zero_dte(broker: Broker, day: datetime) -> bool:
    """Does the book hold anything expiring today?

    Drives the manage cadence from 15 minutes to 5 (§4.6: *"short gamma on a
    15-minute poll is not risk management, it is sampling"*). Read from the broker
    each minute rather than cached, because the answer changes the moment an entry
    fills — and caching it would keep sweeping every 15 minutes through the exact
    session where 5 was the point.

    A failed read returns `True`: the tighter cadence is the safe error.
    """
    try:
        structures = await broker.structures()
    except Exception as exc:  # noqa: BLE001
        log.warning("positions.unavailable", error=str(exc), assuming="holding 0DTE")
        return True
    return any(s.expiry <= day.date() for s in structures)


class Runner:
    """The loop. Sleeps to the next boundary; never spins."""

    def __init__(self, broker: Broker | None = None) -> None:
        self.broker = broker or Broker()
        self._stop = asyncio.Event()
        # Guards against running the same minute twice, which a clock that ticks
        # backwards over a leap second or an NTP correction can otherwise cause.
        self._last_minute: datetime | None = None

    def stop(self) -> None:
        self._stop.set()

    async def run_once(self, at: datetime | None = None) -> list[CycleResult]:
        """Run whatever is due at `at`. Returns what ran, in execution order."""
        now = (at or now_et()).astimezone(ET).replace(second=0, microsecond=0)

        trading = await is_trading_day(self.broker, now)
        zero_dte = await holding_zero_dte(self.broker, now) if trading else False
        kinds = due(now, holding_zero_dte=zero_dte, is_trading_day=trading)
        if not kinds:
            return []

        results: list[CycleResult] = []
        for kind in CYCLE_ORDER:
            if kind not in kinds:
                continue
            try:
                results.append(await run_cycle(kind, broker=self.broker))
            except Exception as exc:  # noqa: BLE001
                # **One failed cycle must not end the trading day.** A manage
                # sweep that throws at 11:05 has to leave the 11:20 sweep — and
                # the 15:40 flatten — still scheduled. `run_cycle` has already
                # journalled the failure and left `finished_at` NULL, so nothing
                # is being hidden here; the loop simply continues.
                log.error(
                    "cycle.unhandled", cycle=kind.value,
                    error=str(exc), error_type=type(exc).__name__,
                )
        return results

    async def run_forever(self) -> None:
        """Wake once a minute, run what is due, sleep again.

        A minute tick rather than `next_due_after`: the next-due calculation
        depends on `holding_zero_dte`, which changes when an entry fills *during*
        a sleep. Sleeping to a boundary computed before that fill would keep the
        15-minute cadence through the session where §4.6 asks for 5. A wakeup that
        finds nothing due costs one comparison, so the loop pays a trivial price
        for being unable to schedule itself into the wrong cadence.
        """
        log.info("runner.start", tz=str(ET))
        while not self._stop.is_set():
            now = now_et().replace(second=0, microsecond=0)
            if now != self._last_minute:
                self._last_minute = now
                await self.run_once(now)

            # Sleep to the top of the next minute, not a flat 60s, so the loop
            # cannot drift past a boundary after a cycle that took 40 seconds.
            nxt = (now + timedelta(minutes=1))
            delay = max((nxt - now_et()).total_seconds(), 1.0)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
        log.info("runner.stop")


async def main() -> int:
    runner = Runner()

    # SIGTERM is what Docker sends on `stop`, and the default action is immediate
    # death. Handling it lets an in-flight cycle finish rather than being killed
    # between submitting an entry and resting its profit target — which is
    # precisely the state §2.6 calls a reconciliation defect.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, runner.stop)

    await runner.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
