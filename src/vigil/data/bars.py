"""Underlying bars — the source of every *signal* (§1.2).

Signals come from the **underlying**, never from the option chain: the free tier
serves an indicative options feed with 15-minute-delayed trades, so anything
derived from it as a *signal* would be reasoning about a market that has moved.
Option data is used only for pricing, greeks and liquidity checks.

Two series, two purposes, and they are not interchangeable:

- **Daily closes** feed the EMA trend score, which is a multi-day statement.
- **5-minute closes** feed realized volatility, which is a statement about *today's*
  tape. `realized_vol` annualizes from a per-5-minute standard deviation, so
  handing it daily bars would misstate the number by roughly √78.

Extracted from `scripts/dry_run.py`, where these lived as private helpers. The
session runner needs exactly the same two series, and two copies of a market-data
fetch is two places for the feed, the lookback or the bar size to drift apart.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, time, timedelta
from decimal import Decimal

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from vigil.clock import ET, now_et
from vigil.data.alpaca_client import stock_data_client
from vigil.data.chain import STOCK_FEED
from vigil.signals.vol import MIN_BARS

# The regular session. Bars outside it are excluded for the same reason
# `scripts/backfill_vrp.py` excludes them: pre- and post-market prints on IEX are
# thin, and their gaps are not intraday movement. The two constructions have to
# match exactly, because one is ranked against the other.
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)


def daily_closes(symbol: str, *, days: int = 60) -> list[float]:
    """The last `days` daily closes, oldest first.

    The request window is widened by 1.6x because `days` counts *sessions* and the
    API takes calendar dates: 60 sessions spans about 84 calendar days once
    weekends and holidays are in it, and asking for 60 calendar days would quietly
    return a 43-session series to a percentile that needs 60.
    """
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=now_et() - timedelta(days=int(days * 1.6)),
        end=now_et(),
        feed=STOCK_FEED,
    )
    return [float(b.close) for b in stock_data_client().get_stock_bars(req)[symbol]]


@dataclass(frozen=True, slots=True)
class SessionBars:
    """One session's 5-minute closes, and **which** session they are.

    The date is carried rather than assumed because `session_closes` legitimately
    returns *yesterday* early in the morning, and a realized-vol number whose
    session is unknown cannot be reported honestly — the difference between "vol
    is calm today" and "vol was calm yesterday" is the whole signal at 09:45.
    """

    date: date | None
    closes: list[float]

    def __bool__(self) -> bool:
        return bool(self.closes)


def session_closes(symbol: str, *, sessions: int = 3) -> SessionBars:
    """The most recent *complete-enough* session's 5-minute closes, oldest first.

    **This used to be `bars[-78:]`, and that was wrong in a way the old docstring
    denied.** Slicing the last 78 bars claims to take "one session's worth", but
    78 is the length of a *finished* session. Ask at 09:45 and today has three
    bars, so the window is 75 bars of yesterday, the overnight gap, and three bars
    of today — a number describing yesterday's tape, labelled as today's, at the
    exact cycle that sets the day's plan. Ask at 12:00 and it is still a
    gap-spanning blend.

    That matters because this number is not read on its own: it is ranked against
    `signals.history.rv_history`, which `backfill_vrp.py` builds **per session,
    regular hours only**. A live estimate that folds in an overnight gap and a
    handful of thin extended-hours prints is ranked against a distribution that
    contains neither, so it sits systematically too high — and since the
    cold-start proxy *inverts* the realized-vol percentile, systematically too
    high reads as "vol is expensive, do not sell premium." The router stands down
    on days it should trade.

    So: group by Eastern session date, keep regular hours only, and return the
    most recent session that actually has enough bars to measure. Early in the
    morning that is yesterday, reported as yesterday.
    """
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        # Calendar days, so the window survives a weekend.
        start=now_et() - timedelta(days=sessions * 3),
        end=now_et(),
        feed=STOCK_FEED,
    )
    bars = stock_data_client().get_stock_bars(req)[symbol]

    by_session: dict[date, list[float]] = defaultdict(list)
    for bar in bars:
        ts = bar.timestamp.astimezone(ET)
        if not (REGULAR_OPEN <= ts.time() < REGULAR_CLOSE):
            continue
        by_session[ts.date()].append(float(bar.close))

    # Newest first, and take the first session with enough bars for `realized_vol`
    # to accept. Falling back a session beats returning a stub that `realized_vol`
    # rejects, because a rejected estimate becomes `rv_annual = None` downstream —
    # and the router treats "cannot measure" as a stand-down.
    for day in sorted(by_session, reverse=True):
        if len(by_session[day]) >= MIN_BARS:
            return SessionBars(day, by_session[day])
    return SessionBars(None, [])


def prev_close_and_open(closes: list[float], spot: Decimal) -> tuple[Decimal, Decimal]:
    """`(prev_close, session_open)` for the gap test, degrading to spot.

    With fewer than two daily bars there is no previous close to compare against,
    and the honest answer is a gap of zero — returning spot for both makes
    `gap_pct` evaluate to 0 rather than to a number invented from one bar.
    """
    if not closes:
        return spot, spot
    if len(closes) == 1:
        return Decimal(str(closes[0])), Decimal(str(closes[0]))
    return Decimal(str(closes[-2])), Decimal(str(closes[-1]))
