"""Trading-calendar time. Never call `date.today()` anywhere else.

CLAUDE.md: timestamps are stored UTC, trading logic reasons in **US/Eastern**.
`date.today()` reads the *machine's* local date, so on a machine east of New York
it rolls over hours early — a chain expiring tomorrow in ET gets labelled 0DTE,
and 0DTE is what drives the 15:40 hard flatten. Found by A1 on a UTC+5 machine
reporting the 26 Aug expiry as 0DTE while it was still 25 Aug in New York.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# Regular US equity/option session.
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
# §4.5 / hard rule: nothing 0DTE may be held past this. Auto-exercise is the reason.
HARD_FLATTEN = time(15, 40)


def now_et() -> datetime:
    """Current time in US/Eastern, timezone-aware."""
    return datetime.now(tz=ET)


def today_et() -> date:
    """The current *trading* date. Use this instead of `date.today()`."""
    return now_et().date()


def to_et(at: datetime | None) -> datetime:
    """Coerce a caller's timestamp into US/Eastern, refusing naive input.

    `astimezone` on a *naive* datetime silently assumes the machine's local zone,
    which is the same bug this module exists to prevent — so a naive argument is
    an error, not something to guess at. Aware timestamps in any zone convert
    correctly, which is what lets a UTC container call these helpers safely.
    """
    if at is None:
        return now_et()
    if at.tzinfo is None or at.tzinfo.utcoffset(at) is None:
        raise ValueError(f"expected a timezone-aware datetime, got naive {at!r}")
    return at.astimezone(ET)


def is_market_open(at: datetime | None = None) -> bool:
    """Weekday regular-hours check. Does not know about holidays.

    Holidays are deliberately out of scope here: the agent gets its authoritative
    answer from Alpaca's clock/calendar endpoint. This is for labelling and local
    reasoning, and it says so rather than pretending to be a calendar.
    """
    t = to_et(at)
    if t.weekday() >= 5:
        return False
    return MARKET_OPEN <= t.time() < MARKET_CLOSE


def is_expiry_live(expiry: date, at: datetime | None = None) -> bool:
    """Can this expiry still be traded right now?

    Today's expiry is the 0DTE contract we actively want during the session, but
    the moment the session closes it is dead — settled, auto-exercised if ITM, and
    quoted with null greeks. Treating a dead expiry as a candidate is how a chain
    scan silently fills up with untradeable contracts.
    """
    t = to_et(at)
    if expiry < t.date():
        return False
    # Same-day expiry stays live right up to the closing bell, then settles.
    return not (expiry == t.date() and t.time() >= MARKET_CLOSE)
