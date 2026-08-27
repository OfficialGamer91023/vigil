"""Backfill the realized-volatility series that the VRP percentile needs.

PLAN.md §4.3.1 defines the regime router's VRP input as:

    rv_annual = stdev(5-min log returns) * sqrt(78 * 252)   # 78 bars per session
    vrp_raw   = iv_atm - rv_annual
    VRP       = percentile of vrp_raw within the trailing 60 sessions

An honest limit, stated rather than hidden: **only the RV half is backfillable.**
`iv_atm` comes from the option chain snapshot, and the free tier serves no
historical implied volatility — option bars carry OHLCV, not IV. So this script
produces the 60-session RV series, and the IV leg accumulates forward one session
at a time. That is exactly the cold-start case §4.3.1 already anticipates: until
60 sessions of `vrp_raw` exist, the router uses the absolute `vrp_raw > 0` test
**and logs that it is doing so**.

Run:  uv run python scripts/backfill_vrp.py [SYMBOL ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from vigil.config import universe_config
from vigil.data.alpaca_client import stock_data_client
from vigil.data.chain import STOCK_FEED
from vigil.settings import REPO_ROOT
from vigil.signals.vol import realized_vol

# The primary universe comes from config/universe.yaml, not from a literal here:
# three scripts repeating ["SPY", "QQQ"] is three places to forget when the
# universe changes. Pass symbols on the command line to override.
DEFAULT_UNIVERSE = list(universe_config().primary)
SESSIONS = 60
ET = ZoneInfo("America/New_York")
OUT_DIR = REPO_ROOT / "data" / "raw"



def backfill(underlying: str) -> dict[str, float]:
    # Calendar days, generously over-fetched: 60 *trading* sessions is ~84
    # calendar days once weekends and holidays are removed.
    end = datetime.now(tz=ET)
    start = end - timedelta(days=int(SESSIONS * 1.6))

    req = StockBarsRequest(
        symbol_or_symbols=underlying,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=STOCK_FEED,
    )
    bars = stock_data_client().get_stock_bars(req)

    by_session: dict[str, list[float]] = defaultdict(list)
    for bar in bars[underlying]:
        ts = bar.timestamp.astimezone(ET)
        # Regular session only. Overnight and pre-market bars are thin on IEX and
        # would inflate RV with gaps that are not intraday movement.
        if (ts.hour, ts.minute) < (9, 30) or (ts.hour, ts.minute) >= (16, 0):
            continue
        by_session[ts.date().isoformat()].append(float(bar.close))

    series: dict[str, float] = {}
    for session, closes in by_session.items():
        rv = realized_vol(closes)
        if rv is not None:
            series[session] = round(rv, 6)

    # Keep the most recent SESSIONS entries — the trailing window §4.3.1 specifies.
    return dict(sorted(series.items())[-SESSIONS:])


def main() -> int:
    universe = sys.argv[1:] or DEFAULT_UNIVERSE
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = True

    for underlying in universe:
        series = backfill(underlying)
        path: Path = OUT_DIR / f"rv_{underlying.lower()}.json"
        path.write_text(json.dumps(series, indent=2))

        values = list(series.values())
        print(f"\n=== {underlying} — {len(values)} sessions -> {path.relative_to(REPO_ROOT)} ===")
        if not values:
            print("  no bars returned")
            ok = False
            continue

        ordered = sorted(values)
        print(f"  latest   {list(series)[-1]}  rv_annual = {values[-1]:.4f}")
        med = ordered[len(ordered) // 2]
        print(f"  min/med/max  {ordered[0]:.4f} / {med:.4f} / {ordered[-1]:.4f}")
        if len(values) < SESSIONS:
            print(f"  short of {SESSIONS} sessions — router stays on the cold-start path (§4.3.1)")
            ok = False

    print("\nNote: the IV leg of vrp_raw is NOT backfillable on the free tier.")
    print("It accumulates one session at a time; until 60 exist, §4.3.1 cold start applies.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
