"""The cron table (§2.4) as a pure function.

Pure so the whole schedule is testable without waiting for 15:40 to come round —
and so the one question that matters can be asked directly: **does the flatten
fire?** A scheduler bug that skipped 15:40 would not show up until a 0DTE
position auto-exercised over the weekend.
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from vigil.worker.schedule import CycleKind, due, next_due_after

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def at(hh: int, mm: int, *, tz: ZoneInfo = ET) -> datetime:
    return datetime(2026, 8, 26, hh, mm, tzinfo=ET).astimezone(tz)


# --------------------------------------------------------------------------- #
# The fixed points
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("hh", "mm", "kind"),
    [
        (8, 45, CycleKind.PREMARKET),
        (9, 45, CycleKind.OPEN),
        (15, 40, CycleKind.FLATTEN),
        (16, 15, CycleKind.POSTCLOSE),
    ],
)
def test_the_fixed_cycles_fire_at_their_appointed_minute(hh, mm, kind) -> None:
    assert kind in due(at(hh, mm))


def test_the_flatten_fires_at_1540_exactly() -> None:
    """Auto-exercise (§1.1) makes this the one cycle that cannot be missed."""
    assert CycleKind.FLATTEN in due(at(15, 40))
    assert CycleKind.FLATTEN not in due(at(15, 39))
    assert CycleKind.FLATTEN not in due(at(15, 41))


# --------------------------------------------------------------------------- #
# The repeating windows
# --------------------------------------------------------------------------- #

def test_manage_runs_every_fifteen_minutes_in_its_window() -> None:
    assert CycleKind.MANAGE in due(at(9, 35))
    assert CycleKind.MANAGE in due(at(9, 50))
    assert CycleKind.MANAGE in due(at(10, 5))
    assert CycleKind.MANAGE not in due(at(9, 40))


def test_manage_tightens_to_five_minutes_while_holding_zero_dte() -> None:
    """§4.6: short gamma on a 15-minute poll is sampling, not risk management."""
    assert CycleKind.MANAGE not in due(at(9, 40))
    assert CycleKind.MANAGE in due(at(9, 40), holding_zero_dte=True)


def test_manage_does_not_run_outside_its_window() -> None:
    assert CycleKind.MANAGE not in due(at(9, 20))
    assert CycleKind.MANAGE not in due(at(16, 5))


def test_entry_runs_every_thirty_minutes_between_1030_and_1430() -> None:
    assert CycleKind.ENTRY in due(at(10, 30))
    assert CycleKind.ENTRY in due(at(11, 0))
    assert CycleKind.ENTRY in due(at(14, 30))
    assert CycleKind.ENTRY not in due(at(10, 45))


def test_no_entry_after_the_window_closes() -> None:
    """Gate 11 would refuse it anyway; not scheduling it saves the round trip."""
    assert CycleKind.ENTRY not in due(at(15, 0))
    assert CycleKind.ENTRY not in due(at(10, 0))


def test_every_entry_is_preceded_by_a_management_sweep() -> None:
    """§2.3: management always runs before new-entry logic.

    The two cadences never coincide by themselves — manage is anchored to 09:35
    in 15-minute steps (:05/:20/:35/:50) and entry to 10:30 in 30s (:00/:30). An
    entry at 11:00 would otherwise reason from a book last swept at 10:50: ten
    minutes of unreconciled drift, at exactly the moment we are about to add
    risk. The schedule emits MANAGE with every ENTRY so the invariant holds
    structurally rather than by convention.
    """
    for hh, mm in ((10, 30), (11, 0), (11, 30), (14, 30)):
        kinds = due(at(hh, mm))
        assert CycleKind.ENTRY in kinds
        assert CycleKind.MANAGE in kinds, f"entry at {hh}:{mm:02d} had no manage sweep"


def test_manage_still_runs_on_its_own_cadence_between_entries() -> None:
    assert due(at(10, 50)) == frozenset({CycleKind.MANAGE})


# --------------------------------------------------------------------------- #
# Timezone and calendar discipline
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tz", [ET, UTC, ZoneInfo("Asia/Karachi")])
def test_the_schedule_is_eastern_whatever_zone_it_is_handed(tz) -> None:
    """Same instant, three zones. The worker will run in a UTC container."""
    assert CycleKind.FLATTEN in due(at(15, 40, tz=tz))


def test_a_non_trading_day_schedules_nothing() -> None:
    """Holidays come from Alpaca's calendar, never a weekday check — clock.py
    says in its own docstring that it does not know about holidays, and a
    holiday mistaken for a session would run a full entry cycle into a closed
    market."""
    assert due(at(11, 0), is_trading_day=False) == frozenset()


# --------------------------------------------------------------------------- #
# Sleeping to the next boundary
# --------------------------------------------------------------------------- #

def test_next_due_finds_the_following_boundary() -> None:
    nxt = next_due_after(at(10, 31))
    assert nxt is not None
    assert nxt.astimezone(ET).time() == time(10, 35)


def test_next_due_crosses_into_the_premarket_of_the_following_day() -> None:
    nxt = next_due_after(at(20, 0))
    assert nxt is not None
    assert nxt.astimezone(ET).time() == time(8, 45)
    assert nxt.astimezone(ET).date().day == 27
