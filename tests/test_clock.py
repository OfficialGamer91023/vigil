"""The ET trading clock — guards against the machine-timezone bug A1 exposed."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from vigil.clock import ET, is_expiry_live, is_market_open, today_et


def test_trading_date_follows_new_york_not_the_machine() -> None:
    """On a UTC+5 machine it can be 26 Aug locally while New York is still on 25 Aug.

    Labelling the 26 Aug expiry 0DTE a day early would arm the 15:40 flatten on the
    wrong session, so this must track ET regardless of where the process runs.
    """
    karachi_evening = datetime(2026, 8, 26, 1, 37, tzinfo=ZoneInfo("Asia/Karachi"))
    assert karachi_evening.astimezone(ET).date().isoformat() == "2026-08-25"
    # today_et() is derived from the same conversion, so it cannot disagree.
    assert today_et() == datetime.now(tz=ET).date()


def test_market_hours_boundaries_are_half_open() -> None:
    def at(h: int, m: int) -> datetime:
        return datetime(2026, 8, 26, h, m, tzinfo=ET)      # a Wednesday

    assert is_market_open(at(9, 30)) is True               # open is inclusive
    assert is_market_open(at(15, 59)) is True
    assert is_market_open(at(16, 0)) is False              # close is exclusive
    assert is_market_open(at(9, 29)) is False


def test_weekends_are_closed() -> None:
    saturday = datetime(2026, 8, 29, 11, 0, tzinfo=ET)
    assert is_market_open(saturday) is False


def test_todays_expiry_is_live_during_the_session_and_dead_after_the_close() -> None:
    """0DTE is the contract we want intraday — and worthless the second it settles."""
    from datetime import date

    from vigil.clock import is_expiry_live

    today = date(2026, 8, 26)
    midday = datetime(2026, 8, 26, 11, 0, tzinfo=ET)
    after = datetime(2026, 8, 26, 16, 37, tzinfo=ET)

    assert is_expiry_live(today, midday) is True
    assert is_expiry_live(today, after) is False
    # Tomorrow's expiry survives the close; yesterday's never comes back.
    assert is_expiry_live(date(2026, 8, 27), after) is True
    assert is_expiry_live(date(2026, 8, 25), midday) is False


# --------------------------------------------------------------------------- #
# B5 — the helpers reason in Eastern regardless of the caller's zone
# --------------------------------------------------------------------------- #

def test_is_market_open_agrees_across_zones_for_one_instant() -> None:
    """The worker will run in a container whose clock reasons in UTC. The same
    instant must not be 'open' in one zone and 'closed' in another."""
    et_now = datetime(2026, 8, 26, 11, 0, tzinfo=ET)
    for zone in ("UTC", "Asia/Karachi", "Europe/London"):
        assert is_market_open(et_now.astimezone(ZoneInfo(zone))) is True


def test_a_utc_timestamp_before_the_open_still_reads_as_closed() -> None:
    """13:00 UTC is 09:00 ET — pre-market. Read as a bare wall clock it looks
    like mid-session."""
    et_now = datetime(2026, 8, 26, 9, 0, tzinfo=ET)
    assert is_market_open(et_now.astimezone(ZoneInfo("UTC"))) is False


def test_expiry_liveness_agrees_across_zones() -> None:
    """The case that motivated `clock.py`: a UTC+5 machine reported tomorrow's
    expiry as 0DTE because its own date had already rolled over."""
    late_et = datetime(2026, 8, 26, 20, 0, tzinfo=ET)  # 01:00 next day in UTC
    expiry = date(2026, 8, 27)
    assert is_expiry_live(expiry, late_et) is True
    assert is_expiry_live(expiry, late_et.astimezone(ZoneInfo("UTC"))) is True


def test_the_helpers_refuse_a_naive_timestamp() -> None:
    """`astimezone` on a naive datetime assumes the machine's local zone, which
    is the bug this module exists to prevent."""
    naive = datetime(2026, 8, 26, 11, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        is_market_open(naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        is_expiry_live(date(2026, 8, 27), naive)
