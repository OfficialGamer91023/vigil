"""Realized vol — §4.3.1 Bug 1 (units), as tests.

Bug 2 (the `vrp_raw > 0` sign test degenerating) is covered where the ranking now
lives: `percentile_rank` in `test_iv_seed.py`.
"""

from __future__ import annotations

import math

import pytest

from vigil.signals.vol import BARS_PER_SESSION, TRADING_DAYS, realized_vol


def test_realized_vol_is_annualized_not_per_bar() -> None:
    """Bug 1: a per-5-minute RV is not comparable to an annualized IV."""
    # A deterministic series: alternating +1%/-1% log moves.
    step = 0.01
    closes = [100.0]
    for i in range(40):
        closes.append(closes[-1] * math.exp(step if i % 2 == 0 else -step))

    rv = realized_vol(closes)
    assert rv is not None
    # stdev of an alternating +/-step series is very close to `step`, scaled up by
    # sqrt(78 * 252). If someone deletes the annualization, this drops ~140x.
    expected = step * math.sqrt(BARS_PER_SESSION * TRADING_DAYS)
    assert rv == pytest.approx(expected, rel=0.05)
    assert rv > 1.0, "annualized 1%-per-5min vol must be a large number, not 0.01"


def test_flat_session_has_zero_vol() -> None:
    assert realized_vol([100.0] * 40) == 0.0


def test_too_few_bars_returns_none_rather_than_a_noisy_number() -> None:
    assert realized_vol([100.0, 101.0, 100.5]) is None
    assert realized_vol([]) is None
