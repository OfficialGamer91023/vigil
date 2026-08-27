"""`session_closes` — the window realized vol is measured over. No network.

This file exists because of a bug that produced no error, no warning and a
perfectly plausible number. `session_closes` used to end in `[-BARS_PER_SESSION:]`
— the last 78 bars — and its docstring claimed that trim kept an overnight gap out
of an intraday volatility estimate. It did the opposite. 78 is the length of a
*finished* session, so mid-morning the slice reached back across the overnight
gap into yesterday and picked up thin extended-hours prints on the way.

The damage is downstream and silent. That number is ranked against
`signals.history.rv_history`, which `backfill_vrp.py` builds **per session,
regular hours only**. Rank a gap-inflated estimate against a distribution built
without gaps and it sits systematically too high; the cold-start proxy inverts the
realized-vol percentile, so too high reads as *"vol is expensive, do not sell"*.
The agent stands down on days it should trade, and every log line explains itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from vigil.data import bars as bars_mod

ET = ZoneInfo("America/New_York")


@dataclass
class FakeBar:
    timestamp: datetime
    close: float


class FakeBarsClient:
    """Returns one canned series regardless of the request."""

    def __init__(self, series: list[FakeBar]) -> None:
        self.series = series

    def get_stock_bars(self, req: object) -> dict[str, list[FakeBar]]:
        return {"SPY": self.series}


def bar(day: date, hour: int, minute: int, close: float) -> FakeBar:
    return FakeBar(datetime(day.year, day.month, day.day, hour, minute, tzinfo=ET), close)


def session(day: date, *, count: int, start: float = 100.0) -> list[FakeBar]:
    """`count` regular-hours 5-minute bars from 09:30, drifting a cent each bar."""
    out = []
    for i in range(count):
        minutes = 570 + i * 5          # 570 = 09:30 in minutes
        out.append(bar(day, minutes // 60, minutes % 60, start + i * 0.01))
    return out


@pytest.fixture
def served(monkeypatch):
    def _install(series: list[FakeBar]) -> None:
        monkeypatch.setattr(bars_mod, "stock_data_client", lambda: FakeBarsClient(series))
    return _install


YESTERDAY, TODAY = date(2026, 8, 26), date(2026, 8, 27)


def test_a_partial_today_never_borrows_bars_from_yesterday(served) -> None:
    """The original bug: at 09:45 today has 3 bars, so `[-78:]` was 75 of yesterday.

    A full yesterday plus a thin today is exactly the shape that produced a
    "today's realized vol" that described yesterday.
    """
    served(session(YESTERDAY, count=78, start=100.0) + session(TODAY, count=3, start=200.0))
    result = bars_mod.session_closes("SPY")

    # Yesterday, whole and unmixed — not a blend, and not three bars of today.
    assert result.date == YESTERDAY
    assert len(result.closes) == 78
    assert max(result.closes) < 200.0


def test_a_complete_today_wins_over_yesterday(served) -> None:
    served(session(YESTERDAY, count=78, start=100.0) + session(TODAY, count=40, start=200.0))
    result = bars_mod.session_closes("SPY")

    assert result.date == TODAY
    assert len(result.closes) == 40
    assert min(result.closes) >= 200.0


def test_extended_hours_prints_are_excluded(served) -> None:
    """Matching `backfill_vrp.py` exactly, because one is ranked against the other.

    A pre-market print at 04:00 and an after-hours print at 19:55 are thin on IEX
    and their gaps are not intraday movement. If the live estimate keeps them and
    the historical series drops them, the two are not comparable.
    """
    served(
        [bar(TODAY, 4, 0, 900.0)]
        + session(TODAY, count=40, start=200.0)
        + [bar(TODAY, 19, 55, 0.5)]
    )
    result = bars_mod.session_closes("SPY")

    assert len(result.closes) == 40
    assert 900.0 not in result.closes and 0.5 not in result.closes


def test_the_16_00_bar_is_excluded_on_the_close_side(served) -> None:
    """`[09:30, 16:00)` — half-open, the same interval the backfill uses."""
    served(session(TODAY, count=40, start=200.0) + [bar(TODAY, 16, 0, 777.0)])
    assert 777.0 not in bars_mod.session_closes("SPY").closes


def test_too_few_bars_anywhere_returns_an_empty_result_not_a_stub(served) -> None:
    """Below `MIN_BARS` there is nothing measurable, and saying so is the point.

    An undersized series handed to `realized_vol` comes back `None`, which the
    router now turns into a stand-down. Returning a short list here rather than
    an empty one would only move the same decision somewhere less visible.
    """
    served(session(TODAY, count=4, start=200.0) + session(YESTERDAY, count=5, start=100.0))
    result = bars_mod.session_closes("SPY")

    assert result.date is None
    assert result.closes == []
    assert not result


def test_closes_come_back_in_chronological_order(served) -> None:
    """sqrt-of-time scaling is only valid on returns taken in order."""
    served(session(TODAY, count=30, start=200.0))
    closes = bars_mod.session_closes("SPY").closes
    assert closes == sorted(closes)      # the canned series drifts upward
