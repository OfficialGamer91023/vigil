"""Local Black-Scholes greeks — the §1.3 A1 fallback.

These matter more than a normal unit test. A1 failed, so **every delta the agent
trades on comes from this module**: strike selection, Gate 7's dollar delta, the
IV percentile and the VRP input. There is no feed value left to cross-check
against at runtime, which means the only thing standing between a solver bug and
a mis-sized position is this file.

Two classes of test, and the second is the important one:

  * that the maths is right — round-trip, parity, monotonicity;
  * that the solver **refuses** rather than guesses. A wrong delta is far worse
    than a missing one, because a missing one is skipped by `pick_by_delta` while
    a wrong one gets selected on, sized against and submitted.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from vigil.data.chain import _with_greeks
from vigil.data.greeks import (
    ModelGreeks,
    bs_delta,
    bs_price,
    implied_vol,
    solve,
    year_fraction,
)

ET = ZoneInfo("America/New_York")

# 11:00 ET on the 28th: mid-session, five hours before the 0DTE contracts expire.
NOW = datetime(2026, 8, 28, 11, 0, tzinfo=ET)
ZERO_DTE = date(2026, 8, 28)
TWO_DTE = date(2026, 8, 31)
SPOT = 771.71
RATE = 0.04


def _t(expiry: date) -> float:
    return year_fraction(expiry, NOW)


# --------------------------------------------------------------------------- #
# The maths
# --------------------------------------------------------------------------- #

def test_round_trip_reproduces_the_observed_price() -> None:
    """The solved sigma, priced back through the model, returns the input price."""
    t = _t(TWO_DTE)
    sigma = implied_vol(price=1.80, spot=SPOT, strike=765.0, t=t, rate=RATE, is_put=True)
    assert sigma is not None
    priced = bs_price(spot=SPOT, strike=765.0, t=t, sigma=sigma, rate=RATE, is_put=True)
    assert priced == pytest.approx(1.80, abs=1e-4)


def test_put_call_delta_parity() -> None:
    """call delta - put delta == 1 for the same strike, expiry and vol.

    The sharpest available check on the sign conventions in `bs_delta`: it fails
    loudly if the put branch is wrong, which is the branch this agent lives on.
    """
    args = {"spot": SPOT, "strike": 765.0, "t": _t(TWO_DTE), "sigma": 0.15, "rate": RATE}
    assert bs_delta(**args, is_put=False) - bs_delta(**args, is_put=True) == pytest.approx(1.0)


def test_put_delta_is_negative_and_monotone_in_strike() -> None:
    """Higher put strike -> more negative delta. Selection walks this ordering."""
    deltas = [
        bs_delta(spot=SPOT, strike=k, t=_t(TWO_DTE), sigma=0.15, rate=RATE, is_put=True)
        for k in (755.0, 760.0, 765.0, 770.0)
    ]
    assert all(d < 0 for d in deltas)
    assert deltas == sorted(deltas, reverse=True)


def test_atm_call_delta_is_about_half() -> None:
    d = bs_delta(spot=SPOT, strike=SPOT, t=_t(TWO_DTE), sigma=0.15, rate=RATE, is_put=False)
    assert 0.50 <= d <= 0.53


def test_solves_the_selection_band_at_zero_dte() -> None:
    """The load-bearing case: a realistic 0DTE chain must yield 0.15-0.20 deltas.

    0DTE is where the textbook Newton solver falls apart (vega -> 0), and 0.16
    delta is the strike §4.5 sells. If this band is unreachable the agent has
    nothing to trade, so it is asserted directly rather than inferred from
    coverage percentages.
    """
    quotes = {770.0: 1.60, 768.0: 0.85, 765.0: 0.30, 760.0: 0.08}
    solved = {
        k: solve(price=p, spot=SPOT, strike=k, expiry=ZERO_DTE, rate=RATE, is_put=True, now=NOW)
        for k, p in quotes.items()
    }
    assert all(m is not None for m in solved.values())
    # Implied vols must be plausible for 0DTE index-ETF options, not bracket ends.
    assert all(0.05 < m.iv < 1.5 for m in solved.values() if m is not None)
    # And the chain must span the band selection picks from.
    assert any(0.10 <= abs(m.delta) <= 0.30 for m in solved.values() if m is not None)


# --------------------------------------------------------------------------- #
# Refusing to guess
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("label", "price", "spot", "strike"),
    [
        ("zero price", 0.0, SPOT, 765.0),
        ("negative price", -1.0, SPOT, 765.0),
        # A put quoted below its own intrinsic value: no volatility produces this,
        # so it is a broken quote on the derived feed, not cheap optionality.
        ("below intrinsic", 0.50, 700.0, 765.0),
        # Above the maximum possible payoff (the discounted strike).
        ("above upper bound", 800.0, SPOT, 765.0),
    ],
)
def test_unsolvable_prices_return_none(
    label: str, price: float, spot: float, strike: float
) -> None:
    assert implied_vol(
        price=price, spot=spot, strike=strike, t=_t(TWO_DTE), rate=RATE, is_put=True
    ) is None, label


def test_expired_contract_returns_none() -> None:
    """T <= 0 has no implied vol. Guarded before the solver, not inside it."""
    assert implied_vol(price=1.0, spot=SPOT, strike=765.0, t=0.0, rate=RATE, is_put=True) is None
    expired = solve(
        price=1.0, spot=SPOT, strike=765.0, expiry=date(2026, 8, 27),
        rate=RATE, is_put=True, now=NOW,
    )
    assert expired is None


# --------------------------------------------------------------------------- #
# Time to expiry
# --------------------------------------------------------------------------- #

def test_year_fraction_is_intraday_not_integer_days() -> None:
    """The bug this guards: integer DTE makes every 0DTE contract T = 0.

    That would make the pricing formula degenerate for the majority of what this
    agent trades, and the failure would look like "the solver returns None a lot"
    rather than like a units bug.
    """
    hours = year_fraction(ZERO_DTE, NOW) * 365 * 24
    assert hours == pytest.approx(5.0)


def test_year_fraction_runs_to_the_closing_bell() -> None:
    """Options stop trading at 16:00 ET, not at midnight."""
    at_the_bell = datetime(2026, 8, 28, 16, 0, tzinfo=ET)
    assert year_fraction(ZERO_DTE, at_the_bell) == 0.0
    assert year_fraction(ZERO_DTE, at_the_bell - timedelta(hours=1)) > 0.0


def test_year_fraction_never_goes_negative() -> None:
    assert year_fraction(date(2026, 8, 20), NOW) == 0.0


def test_year_fraction_refuses_naive_datetimes() -> None:
    """Inherited from `clock.to_et`: a naive timestamp is an error, not a guess."""
    with pytest.raises(ValueError, match="timezone-aware"):
        year_fraction(ZERO_DTE, datetime(2026, 8, 28, 11, 0))


# --------------------------------------------------------------------------- #
# Wiring into the chain
# --------------------------------------------------------------------------- #

def _attach(contract):
    from decimal import Decimal

    return _with_greeks(contract, spot=Decimal(str(SPOT)), now=NOW, rate=RATE)


def test_feed_greeks_win_over_the_model(make_contract) -> None:
    """The fallback is a fallback. If OPRA is ever entitled, it goes quiet by itself."""
    c = _attach(make_contract("SPY260831P00765000", delta=-0.163, iv=0.15, bid=1.78, ask=1.82))
    assert c.computed is None
    assert c.delta == -0.163
    assert c.greeks_are_modelled is False


def test_model_fills_in_when_the_feed_omits_greeks(make_contract) -> None:
    """What actually happens on every contract today."""
    c = _attach(make_contract("SPY260831P00765000", bid=1.78, ask=1.82))
    assert isinstance(c.computed, ModelGreeks)
    assert c.greeks_are_modelled is True
    assert c.delta is not None and -0.40 < c.delta < -0.10
    assert c.iv is not None and 0.05 < c.iv < 1.0


def test_no_quote_means_no_delta(make_contract) -> None:
    """No two-sided quote, nothing to invert — and deliberately no reach for the
    last trade, which is 15 minutes stale on the indicative feed."""
    c = _attach(make_contract("SPY260831P00765000"))
    assert c.computed is None
    assert c.delta is None


def test_partial_feed_data_is_completed_not_overridden(make_contract) -> None:
    """Greeks present but IV missing: keep the fed delta, model only the gap."""
    c = _attach(make_contract("SPY260831P00765000", delta=-0.163, bid=1.78, ask=1.82))
    assert c.delta == -0.163          # feed
    assert c.iv is not None           # modelled
    assert c.greeks_are_modelled is False
