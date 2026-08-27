"""The scheduler loop: ordering, failure isolation, and the calendar.

No network anywhere — the runner talks only to `Broker`, so a stub with the same
three methods exercises the real loop. That seam is the reason `worker/broker.py`
exists as a separate module.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from vigil.worker import runner as R
from vigil.worker.runner import CYCLE_ORDER, Runner
from vigil.worker.schedule import CycleKind
from vigil.worker.sessions import CycleResult

ET = R.ET


class StubBroker:
    """Only what the runner asks of a broker, and nothing more."""

    def __init__(self, *, structures=(), calendar_raises=False, positions_raise=False):
        self._structures = structures
        self.calendar_raises = calendar_raises
        self.positions_raise = positions_raise

    async def structures(self):
        if self.positions_raise:
            raise ConnectionError("broker unreachable")
        return self._structures

    @property
    def client(self):
        raise AssertionError("the loop must not reach the SDK in tests")


class FakeStructure:
    def __init__(self, expiry: date) -> None:
        self.expiry = expiry


@pytest.fixture
def ran(monkeypatch):
    """Record which cycles ran, in order, instead of running them."""
    order: list[CycleKind] = []

    async def _fake_run_cycle(kind, *, broker=None):
        order.append(kind)
        return CycleResult(kind=kind)

    monkeypatch.setattr(R, "run_cycle", _fake_run_cycle)
    return order


@pytest.fixture
def open_market(monkeypatch):
    monkeypatch.setattr(R, "is_trading_day", _always(True))
    return monkeypatch


def _always(value):
    async def _f(*_args, **_kwargs):
        return value
    return _f


async def test_entry_minute_sweeps_before_it_enters(ran, open_market):
    """§2.3's ordering invariant, end to end.

    The schedule emits MANAGE alongside every ENTRY; this asserts the runner then
    actually *runs* them in that order. Protecting capital outranks deploying it,
    and an entry reasoning from a book last swept ten minutes ago is exactly the
    drift that ordering exists to remove.
    """
    open_market.setattr(R, "holding_zero_dte", _always(False))
    await Runner(StubBroker()).run_once(datetime(2026, 8, 28, 11, 0, tzinfo=ET))
    assert ran == [CycleKind.MANAGE, CycleKind.ENTRY]


async def test_flatten_outranks_the_manage_sweep_at_1540(ran, open_market):
    """At 15:40 the flatten *is* the management, and it must not queue behind it."""
    open_market.setattr(R, "holding_zero_dte", _always(True))
    await Runner(StubBroker()).run_once(datetime(2026, 8, 28, 15, 40, tzinfo=ET))
    assert ran.index(CycleKind.FLATTEN) < ran.index(CycleKind.MANAGE)


async def test_nothing_runs_on_a_holiday(ran, monkeypatch):
    """A weekday check would have run a full entry cycle into a closed market."""
    monkeypatch.setattr(R, "is_trading_day", _always(False))
    monkeypatch.setattr(R, "holding_zero_dte", _always(False))
    await Runner(StubBroker()).run_once(datetime(2026, 8, 28, 11, 0, tzinfo=ET))
    assert ran == []


async def test_one_failing_cycle_does_not_cancel_the_rest(monkeypatch, open_market):
    """A manage sweep that throws at 11:00 must leave the 11:00 entry scheduled.

    The stronger form of this guarantee — that the 15:40 flatten still fires after
    a mid-session failure — is the same property one loop iteration up.
    """
    ran: list[CycleKind] = []

    async def _flaky(kind, *, broker=None):
        ran.append(kind)
        if kind is CycleKind.MANAGE:
            raise RuntimeError("sweep exploded")
        return CycleResult(kind=kind)

    monkeypatch.setattr(R, "run_cycle", _flaky)
    open_market.setattr(R, "holding_zero_dte", _always(False))

    results = await Runner(StubBroker()).run_once(datetime(2026, 8, 28, 11, 0, tzinfo=ET))
    assert ran == [CycleKind.MANAGE, CycleKind.ENTRY]
    # The failure is dropped from the results, not from the schedule.
    assert [r.kind for r in results] == [CycleKind.ENTRY]


async def test_zero_dte_holding_tightens_the_sweep_to_five_minutes(ran, open_market):
    """§4.6: a 15-minute poll on short gamma is sampling, not risk management."""
    open_market.setattr(R, "holding_zero_dte", _always(True))
    # 10:40 is on the 5-minute grid from 09:35 but not on the 15-minute one.
    await Runner(StubBroker()).run_once(datetime(2026, 8, 28, 10, 40, tzinfo=ET))
    assert ran == [CycleKind.MANAGE]


async def test_no_zero_dte_leaves_the_sweep_at_fifteen(ran, open_market):
    open_market.setattr(R, "holding_zero_dte", _always(False))
    await Runner(StubBroker()).run_once(datetime(2026, 8, 28, 10, 40, tzinfo=ET))
    assert ran == []


async def test_unreadable_positions_assume_zero_dte(open_market):
    """The tighter cadence is the safe error when the broker cannot be read."""
    assert await R.holding_zero_dte(
        StubBroker(positions_raise=True), datetime(2026, 8, 28, 11, 0, tzinfo=ET)
    )


async def test_holding_zero_dte_ignores_later_expiries():
    broker = StubBroker(structures=(FakeStructure(date(2026, 9, 4)),))
    assert not await R.holding_zero_dte(broker, datetime(2026, 8, 28, 11, 0, tzinfo=ET))


async def test_holding_zero_dte_sees_todays_expiry():
    broker = StubBroker(structures=(FakeStructure(date(2026, 8, 28)),))
    assert await R.holding_zero_dte(broker, datetime(2026, 8, 28, 11, 0, tzinfo=ET))


def test_cycle_order_covers_every_cycle_kind():
    """A new cycle kind that nobody added to CYCLE_ORDER would never run.

    Silently. `due()` would return it and the runner's `if kind not in kinds`
    loop would simply never reach it — the worst kind of scheduling bug, because
    nothing errors.
    """
    assert set(CYCLE_ORDER) == set(CycleKind)
