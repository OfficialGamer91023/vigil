"""EMA and gap. Five lines each, which is exactly why there is no TA-Lib here.

PLAN §10 rejects a technical-analysis dependency: importing a library to compute
an exponential moving average would be a dependency we could not justify in one
sentence, and it would hide the one parameter that matters (the smoothing factor).
"""

from __future__ import annotations

from decimal import Decimal


def ema(values: list[float], period: int) -> float | None:
    """Exponential moving average, seeded with a simple average of the first window.

    alpha = 2/(period+1) is the conventional smoothing factor: it makes the EMA's
    centre of mass match an SMA of the same period, which is why EMA(9) and SMA(9)
    are comparable at all. Seeding with an SMA rather than the first value avoids
    a long warm-up bias in the early output.
    """
    if len(values) < period or period < 1:
        return None
    alpha = 2.0 / (period + 1)
    out = sum(values[:period]) / period
    for v in values[period:]:
        out = alpha * v + (1 - alpha) * out
    return out


def trend_score(closes: list[float], fast: int, slow: int) -> float | None:
    """(EMA_fast − EMA_slow) / EMA_slow.

    Normalised by the slow EMA so the number is a *fraction*, comparable across
    a $765 SPY and a $200 IWM. A raw dollar difference would silently make the
    trend threshold mean something different for every underlying.
    """
    f, s = ema(closes, fast), ema(closes, slow)
    if f is None or s is None or s == 0:
        return None
    return (f - s) / s


def gap_pct(open_price: Decimal, prev_close: Decimal) -> Decimal:
    """Overnight gap as a signed fraction of the previous close."""
    if prev_close <= 0:
        return Decimal(0)
    return (open_price - prev_close) / prev_close


def percentile_rank(value: float, history: list[float]) -> float | None:
    """Fraction of `history` strictly below `value`, in [0, 1].

    Returns None when the history is too short to rank against, so the caller
    must take an explicit cold-start path rather than silently ranking against
    an empty distribution (§4.3.1).
    """
    if len(history) < 2:
        return None
    return sum(1 for h in history if h < value) / len(history)
