"""The clock-drift guard (`vigil.clock_guard`) — the local-time analogue of the
account lock. The comparison is pure and tested here without a network; the wiring
that puts it on the startup path is proved in `test_sessions.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from vigil.clock import ET
from vigil.clock_guard import (
    MAX_CLOCK_DRIFT,
    ClockDriftError,
    assert_clock_synced,
    broker_now,
    verify_clock,
)

# A fixed instant, expressed two ways, so "same moment, different zone" is testable.
INSTANT_UTC = datetime(2026, 8, 31, 15, 0, tzinfo=UTC)          # 11:00 ET
INSTANT_ET = INSTANT_UTC.astimezone(ET)


class _FakeClockClient:
    """Stands in for alpaca-py's TradingClient: only `get_clock().timestamp`."""

    def __init__(self, timestamp: datetime) -> None:
        self._timestamp = timestamp

    def get_clock(self) -> SimpleNamespace:
        return SimpleNamespace(timestamp=self._timestamp)


# --------------------------------------------------------------------------- #
# assert_clock_synced — the pure comparison
# --------------------------------------------------------------------------- #

def test_identical_clocks_have_zero_drift() -> None:
    assert assert_clock_synced(INSTANT_ET, INSTANT_UTC) == 0.0


def test_the_same_instant_in_two_zones_is_zero_drift() -> None:
    """The whole point of requiring aware datetimes: ET-local vs UTC-broker for the
    same moment must read as *synced*, not as a four-hour skew."""
    assert assert_clock_synced(INSTANT_ET, INSTANT_UTC, tolerance=timedelta(0)) == 0.0


def test_drift_within_tolerance_passes_and_returns_the_gap() -> None:
    local = INSTANT_UTC + timedelta(seconds=30)
    assert assert_clock_synced(local, INSTANT_UTC) == 30.0


def test_drift_exactly_at_the_boundary_passes() -> None:
    """60s is the tolerance, and the check is strictly *greater than*, so a clock
    exactly one minute off is still accepted — the refusal is for clocks past it."""
    local = INSTANT_UTC + MAX_CLOCK_DRIFT
    assert assert_clock_synced(local, INSTANT_UTC) == 60.0


def test_drift_over_tolerance_refuses() -> None:
    local = INSTANT_UTC + timedelta(seconds=61)
    with pytest.raises(ClockDriftError, match="CLOCK DRIFT"):
        assert_clock_synced(local, INSTANT_UTC)


def test_the_refusal_is_symmetric_in_direction() -> None:
    """A local clock behind the broker is as dangerous as one ahead — both raise."""
    ahead = INSTANT_UTC + timedelta(minutes=5)
    behind = INSTANT_UTC - timedelta(minutes=5)
    for local in (ahead, behind):
        with pytest.raises(ClockDriftError):
            assert_clock_synced(local, INSTANT_UTC)


@pytest.mark.parametrize(
    "local, broker",
    [
        (datetime(2026, 8, 31, 11, 0), INSTANT_UTC),   # naive local
        (INSTANT_ET, datetime(2026, 8, 31, 15, 0)),    # naive broker
    ],
)
def test_a_naive_timestamp_is_refused_not_assumed(local: datetime, broker: datetime) -> None:
    """A naive value would silently assume the machine's zone — the exact failure
    this guard exists to catch — so it is an error, never coerced."""
    with pytest.raises(ClockDriftError, match="naive"):
        assert_clock_synced(local, broker)


# --------------------------------------------------------------------------- #
# broker_now / verify_clock — the reader and the startup wrapper
# --------------------------------------------------------------------------- #

def test_broker_now_reads_the_clock_timestamp() -> None:
    assert broker_now(_FakeClockClient(INSTANT_UTC)) == INSTANT_UTC


def test_verify_clock_passes_when_the_local_clock_matches() -> None:
    drift = verify_clock(client=_FakeClockClient(INSTANT_UTC), local=INSTANT_ET)
    assert drift == 0.0


def test_verify_clock_refuses_a_skewed_host() -> None:
    skewed_local = INSTANT_ET + timedelta(minutes=2)
    with pytest.raises(ClockDriftError, match="CLOCK DRIFT"):
        verify_clock(client=_FakeClockClient(INSTANT_UTC), local=skewed_local)
