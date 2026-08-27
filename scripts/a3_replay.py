"""A3 (PLAN §1.3): is the gate stack permissive enough to fire on a normal day?

**The question.** STRESS stand-down, the VRP sell floor, the overnight-gap veto,
the daily halt, the macro blackout and the time windows all compound. Nobody has
checked what fraction of days survive all of them. §14 lists "the gate stack is so
restrictive the agent rarely trades" as a top project risk, and §1.3 sets the
target at **>= 50% of sessions producing at least one permitted entry**. A silent
zero-trade agent is the worst possible demo.

**What this can and cannot measure — stated rather than hidden.**

Replayable, because it derives from underlying bars we can fetch historically:
  - the regime router in full (trend, gap, VRP percentile, stand-downs)
  - Gate 11, time windows, at every 30-minute entry slot the cron would hit
  - Gate 3 / 4, which on a flat book are structurally passes

NOT replayable, because they need a live option chain that no longer exists:
  - Gate 8 (liquidity) and Gate 9 (credit quality)

So this is **an upper bound on permissiveness**: it answers "how often does the
router even offer a candidate, and is there a legal time to place it?" A2 answers
the credit question separately, against a live chain. Reporting a single blended
number would imply a completeness this cannot have.

Run:  make a3        (or: uv run python scripts/a3_replay.py [SYMBOL ...])
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from vigil.config import RegimeConfig, regime_config, risk_config, universe_config
from vigil.data.alpaca_client import stock_data_client
from vigil.data.chain import STOCK_FEED
from vigil.signals.regime import MarketSnapshot, classify
from vigil.signals.vol import realized_vol

ET = ZoneInfo("America/New_York")
# The primary universe comes from config/universe.yaml, not from a literal here:
# three scripts repeating ["SPY", "QQQ"] is three places to forget when the
# universe changes. Pass symbols on the command line to override.
DEFAULT_UNIVERSE = list(universe_config().primary)
SESSIONS = 60
# The §2.4 cron: entry evaluation every 30 minutes, 10:30-14:30 ET.
ENTRY_SLOTS = [time(10, 30), time(11, 0), time(11, 30), time(12, 0),
               time(12, 30), time(13, 0), time(13, 30), time(14, 0), time(14, 30)]


@dataclass(frozen=True, slots=True)
class Session:
    date: str
    closes: list[float]        # 5-min closes, regular hours
    prev_close: float
    session_open: float


def _load_sessions(underlying: str) -> list[Session]:
    """Group 5-minute bars into regular-hours sessions, oldest first."""
    end = datetime.now(tz=ET)
    req = StockBarsRequest(
        symbol_or_symbols=underlying,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=end - timedelta(days=int(SESSIONS * 1.8)),
        end=end,
        feed=STOCK_FEED,
    )
    by_day: dict[str, list[float]] = {}
    for bar in stock_data_client().get_stock_bars(req)[underlying]:
        ts = bar.timestamp.astimezone(ET)
        if (ts.hour, ts.minute) < (9, 30) or (ts.hour, ts.minute) >= (16, 0):
            continue
        by_day.setdefault(ts.date().isoformat(), []).append(float(bar.close))

    days = sorted(by_day)
    out: list[Session] = []
    for i, day in enumerate(days):
        closes = by_day[day]
        if len(closes) < 20:          # half-day or an outage; not a normal session
            continue
        prev = by_day[days[i - 1]][-1] if i else closes[0]
        out.append(Session(date=day, closes=closes, prev_close=prev,
                           session_open=closes[0]))
    return out[-SESSIONS:]


def _time_window_open(slot: time, cfg: RegimeConfig) -> bool:
    """Gate 11 for an entry slot, on a 0-2 DTE mandate.

    Reimplemented rather than imported because Gate 11 takes a full proposal and
    a KernelContext, and building 540 synthetic proposals to ask a question about
    the clock would obscure what is being measured. The thresholds come from the
    same config the gate reads.
    """
    r = risk_config()
    if slot < time(9, 30) or slot >= time(16, 0):
        return False
    if slot < (datetime.combine(datetime.today(), time(9, 30))
               + timedelta(minutes=r.no_entry_first_minutes)).time():
        return False
    if slot > (datetime.combine(datetime.today(), time(16, 0))
               - timedelta(minutes=r.no_entry_last_minutes)).time():
        return False
    # 0DTE is the default expiry at 0-2 DTE, so the cutoff binds.
    return slot < r.zero_dte_entry_cutoff


def replay(underlying: str) -> tuple[int, int, Counter[str]]:
    sessions = _load_sessions(underlying)
    cfg = regime_config()
    regimes: Counter[str] = Counter()
    fired = 0

    # The trailing distributions grow as the replay walks forward — reasoning from
    # a distribution that includes the future would flatter every verdict.
    rv_series: list[float] = []
    daily_closes: list[float] = []

    for s in sessions:
        rv = realized_vol(s.closes)
        if rv is None:
            continue
        daily_closes.append(s.closes[-1])

        snap = MarketSnapshot(
            underlying=underlying,
            spot=Decimal(str(s.closes[-1])),
            prev_close=Decimal(str(s.prev_close)),
            session_open=Decimal(str(s.session_open)),
            daily_closes=list(daily_closes),
            # No historical IV on the free tier (CLI_NOTES §2). Leaving it at 0
            # would fake a VRP; instead vrp_history stays empty, the router takes
            # the documented cold-start path, and rv_history drives it — which is
            # exactly what Day 1 will do.
            iv_atm=0.0,
            rv_annual=rv,
            vrp_history=[],
            iv_history=[],
            rv_history=list(rv_series),
        )
        v = classify(snap, cfg)
        regimes[v.regime.value] += 1

        slots_open = sum(1 for slot in ENTRY_SLOTS if _time_window_open(slot, cfg))
        if v.structure is not None and v.size_multiplier > 0 and slots_open > 0:
            fired += 1

        rv_series.append(rv)

    return fired, len(sessions), regimes


def main() -> int:
    universe = sys.argv[1:] or DEFAULT_UNIVERSE
    print("A3 — gate-stack permissiveness replay")
    print("Measures the ROUTER and the time windows. Gates 8/9 need a live chain")
    print("and are answered by A2 (`make a2`) instead. This is an upper bound.\n")

    worst = 1.0
    for underlying in universe:
        fired, total, regimes = replay(underlying)
        if not total:
            print(f"{underlying}: no sessions returned")
            return 1
        rate = fired / total
        worst = min(worst, rate)
        print(f"=== {underlying} — {fired}/{total} sessions would permit an entry "
              f"({rate:.0%}) ===")
        for regime, n in regimes.most_common():
            bar = "#" * round(40 * n / total)
            print(f"  {regime:<10} {n:>3} ({n / total:>4.0%}) {bar}")
        print()

    target = 0.50
    if worst >= target:
        print(f"PASS — every symbol clears the {target:.0%} target (§1.3).")
        return 0
    print(f"FAIL — below the {target:.0%} target. A gate is miscalibrated, and the")
    print("       regime histogram above names which stand-down is doing the damage.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
