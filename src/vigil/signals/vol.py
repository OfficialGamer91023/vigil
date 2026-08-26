"""Realized volatility and the VRP percentile (PLAN §4.3.1).

Two bugs this module exists to prevent:

**Bug 1 — units.** ATM IV from the chain is *annualized*. A per-5-minute realized
vol is not. Subtracting them is meaningless, so `realized_vol` annualizes before
anything compares the two.

**Bug 2 — the sign test degenerates.** Short-dated IV sits above trailing RV on
nearly every session, so `vrp_raw > 0` is true almost always and the STRESS regime
never fires. `vrp_percentile` ranks instead of subtracting.
"""

from __future__ import annotations

import math
import statistics

# 78 five-minute bars in a 6.5-hour session; 252 trading days a year.
BARS_PER_SESSION = 78
TRADING_DAYS = 252
# Below this many bars a session's stdev is too noisy to trust (half-days, outages).
MIN_BARS = 10


def realized_vol(closes: list[float]) -> float | None:
    """Annualized realized volatility from one session's 5-minute closes.

    Log returns rather than percent changes: they are additive across bars, which
    is what makes the sqrt-of-time scaling valid.
    """
    if len(closes) < MIN_BARS:
        return None
    rets = [
        math.log(b / a) for a, b in zip(closes, closes[1:], strict=False) if a > 0 and b > 0
    ]
    if len(rets) < MIN_BARS:
        return None
    return statistics.stdev(rets) * math.sqrt(BARS_PER_SESSION * TRADING_DAYS)


def vrp_percentile(vrp_raw: float, history: list[float]) -> float | None:
    """Where `vrp_raw` sits within its own trailing distribution, in [0, 1].

    Returns None when history is too short to rank against — the caller must then
    take the §4.3.1 cold-start path *and log that it is doing so*, rather than
    silently reasoning from an empty distribution.
    """
    if len(history) < 2:
        return None
    below = sum(1 for h in history if h < vrp_raw)
    return below / len(history)
