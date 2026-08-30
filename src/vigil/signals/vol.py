"""Realized volatility for the VRP calculation (PLAN §4.3.1).

The bug this module exists to prevent — **units.** ATM IV from the chain is
*annualized*. A per-5-minute realized vol is not. Subtracting them is meaningless,
so `realized_vol` annualizes before anything compares the two.

The second §4.3.1 bug — the `vrp_raw > 0` sign test degenerates because short-dated
IV sits above trailing RV nearly every session — is handled downstream: the regime
router ranks `vrp_raw` within its trailing distribution via
`indicators.percentile_rank`, not by subtracting. (An earlier `vrp_percentile`
helper here duplicated that ranking and nothing called it; it was removed.)
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
