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

from datetime import timedelta
from decimal import Decimal

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from vigil.clock import now_et
from vigil.data.alpaca_client import stock_data_client
from vigil.data.chain import STOCK_FEED
from vigil.signals.vol import BARS_PER_SESSION


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


def session_closes(symbol: str, *, sessions: int = 3) -> list[float]:
    """Recent 5-minute closes, trimmed to the most recent session's worth.

    Trimmed because realized vol is a statement about today's tape, not about the
    average of the last week — and a window spanning an overnight gap would fold
    that gap into an intraday volatility estimate.
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
    return [float(b.close) for b in bars][-BARS_PER_SESSION:]


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
