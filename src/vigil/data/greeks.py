"""Black-Scholes greeks, computed locally. The PLAN §1.3 A1 fallback.

**Why this module exists.** A1 — "the options feed populates `greeks` and
`implied_volatility`" — was measured on 28 Aug 2026 and **failed**. Alpaca serves
those analytics only on the OPRA feed, which needs a paid Algo Trader Plus
subscription; the free indicative feed omits both fields from the wire payload
entirely (not null — absent). Confirmed at the raw HTTP level, on every available
feed value, pre-market and mid-session, so it is a permanent property of the free
tier rather than a staleness artifact.

Roughly 60% of the design reads delta: strike selection (§4.5), Gate 7's dollar
delta, the IV percentile and the VRP input. So we model it, from the one thing
the indicative feed does deliver reliably — a live two-sided quote.

**The method.** Invert Black-Scholes for the volatility that reproduces the
observed mid price, then evaluate delta at that volatility. Selection only needs
contracts to be *comparable*, and two contracts priced by one model on the same
inputs are exactly that.

**Why bisection rather than Newton-Raphson.** The textbook IV solver is Newton,
stepping by `price_error / vega`. Vega vanishes as expiry approaches, so at 0DTE —
the contracts this agent trades most — Newton divides by something near zero, and
the step explodes. Bisection needs only that price be monotone in volatility
(it always is) and cannot diverge. It costs ~23 iterations of a handful of flops
on one contract, which is free at our volumes.

**Why no new dependency.** The normal CDF is `erf`, from the standard library.
`py_vollib` drags in scipy for a one-line function and is already on the §12
rejected bench.

**What this module will not do: return a number it cannot justify.** Every
failure path — expired contract, zero or crossed quote, a price outside the
model's no-arbitrage bounds, a solve that does not converge — returns `None`.
`None` then propagates exactly as a feed-omitted greek does: `pick_by_delta`
skips the contract and the kernel rejects any proposal missing the field. A
fabricated delta would instead sail through Gate 7 with full confidence, which is
the one outcome worse than not trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import erf, exp, log, sqrt

from vigil.clock import ET, MARKET_CLOSE, to_et

# Bisection brackets. 0.5% annualised is below any listed option that trades;
# 500% is above even an earnings-print meme name. A price that implies something
# outside this range is a broken quote, not extreme volatility.
SIGMA_LOW = 0.005
SIGMA_HIGH = 5.0
# Halving 0.005 -> 5.0 down to a 1e-6 bracket takes ~23 iterations. The cap is
# slack enough that reaching it means something is genuinely wrong.
MAX_ITERATIONS = 100
SIGMA_TOLERANCE = 1e-6

_SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


@dataclass(frozen=True, slots=True)
class ModelGreeks:
    """Greeks derived here rather than received from the feed.

    A distinct type, not a bare tuple, so that everything downstream — the
    journal especially — can tell a modelled delta from a quoted one. When the
    agent explains a Gate 7 rejection after the fact, "which delta was that?" is
    the first question worth being able to answer.
    """

    iv: float
    delta: float


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def year_fraction(expiry: date, now: datetime | None = None) -> float:
    """Time to expiry in years, with intraday precision.

    **Integer days would break this outright.** `(expiry - today).days` is 0 for
    every 0DTE contract, which drives `T = 0`, which makes the pricing formula
    degenerate and the IV solve unsolvable — for precisely the contracts the
    agent trades most. So the clock runs in seconds, and it runs to 16:00 ET on
    the expiry date rather than to midnight, because that is when the option
    stops trading.

    Calendar time, not a 252-day trading year. The choice matters much less than
    it appears: sigma and T reach the price almost entirely through the total
    variance `sigma^2 * T`, so a T biased high solves to a sigma biased low, and
    the delta that comes out the far side is nearly unchanged. What matters is
    that every contract uses the *same* convention, which is what makes them
    comparable.
    """
    expiry_at = datetime.combine(expiry, MARKET_CLOSE, tzinfo=ET)
    remaining = (expiry_at - to_et(now)).total_seconds()
    return max(0.0, remaining) / _SECONDS_PER_YEAR


def _d1(spot: float, strike: float, t: float, sigma: float, rate: float) -> float:
    return (log(spot / strike) + (rate + 0.5 * sigma * sigma) * t) / (sigma * sqrt(t))


def bs_price(
    *, spot: float, strike: float, t: float, sigma: float, rate: float, is_put: bool
) -> float:
    """Black-Scholes price of a European option.

    **No dividend term.** SPY and QQQ pay quarterly, going ex-dividend on the
    third Friday of Mar/Jun/Sep/Dec — so no 0-2 DTE window in this competition
    (through 4 Sep 2026) contains one. Add a `q` term before reusing this at
    longer tenors.

    **American exercise is ignored.** For short-dated contracts with no dividend
    in the window, early exercise is never optimal, so the American premium over
    the European price is zero to well past the precision that matters here.
    """
    if t <= 0.0 or sigma <= 0.0:
        # Degenerate but not an error: with no time or no volatility the option
        # is worth exactly its intrinsic value. Callers guard against reaching
        # here; the branch exists so the math returns rather than raising.
        return max(0.0, (strike - spot) if is_put else (spot - strike))

    d1 = _d1(spot, strike, t, sigma, rate)
    d2 = d1 - sigma * sqrt(t)
    discounted_strike = strike * exp(-rate * t)
    if is_put:
        return discounted_strike * norm_cdf(-d2) - spot * norm_cdf(-d1)
    return spot * norm_cdf(d1) - discounted_strike * norm_cdf(d2)


def bs_delta(
    *, spot: float, strike: float, t: float, sigma: float, rate: float, is_put: bool
) -> float:
    """dPrice/dSpot. `N(d1)` for a call; `N(d1) - 1` for a put, hence negative."""
    if t <= 0.0 or sigma <= 0.0:
        # At expiry delta is a step function: fully long/short the underlying if
        # in the money, zero if out. No smoothing — a made-up interior value here
        # would be a guess wearing a number's clothes.
        if is_put:
            return -1.0 if spot < strike else 0.0
        return 1.0 if spot > strike else 0.0

    d1 = _d1(spot, strike, t, sigma, rate)
    return norm_cdf(d1) - 1.0 if is_put else norm_cdf(d1)


def implied_vol(
    *, price: float, spot: float, strike: float, t: float, rate: float, is_put: bool
) -> float | None:
    """The volatility that reproduces `price`, or `None` if there isn't one.

    `None` is a real answer here, not an error path to be smoothed over — see the
    module docstring.
    """
    if price <= 0.0 or t <= 0.0 or spot <= 0.0 or strike <= 0.0:
        return None

    # No-arbitrage bounds, checked *before* the solver runs. A quote below
    # intrinsic value or above the maximum possible payoff corresponds to no
    # volatility at all — it is a broken quote, and on a derived feed those do
    # occur. Bisection would otherwise happily converge on a bracket endpoint and
    # hand back 0.5% or 500% with complete confidence.
    discounted_strike = strike * exp(-rate * t)
    if is_put:
        lower, upper = max(0.0, discounted_strike - spot), discounted_strike
    else:
        lower, upper = max(0.0, spot - discounted_strike), spot
    if not lower < price < upper:
        return None

    def error(sigma: float) -> float:
        priced = bs_price(spot=spot, strike=strike, t=t, sigma=sigma, rate=rate, is_put=is_put)
        return priced - price

    # Price is strictly increasing in sigma, so the bracket is valid exactly when
    # the error changes sign across it. Testing that up front is what turns a
    # silent wrong answer into an honest None.
    low, high = SIGMA_LOW, SIGMA_HIGH
    if error(low) > 0.0 or error(high) < 0.0:
        return None

    for _ in range(MAX_ITERATIONS):
        mid = 0.5 * (low + high)
        if error(mid) < 0.0:
            low = mid
        else:
            high = mid
        if high - low < SIGMA_TOLERANCE:
            return 0.5 * (low + high)
    return None


def solve(
    *,
    price: float,
    spot: float,
    strike: float,
    expiry: date,
    rate: float,
    is_put: bool,
    now: datetime | None = None,
) -> ModelGreeks | None:
    """IV and delta implied by one observed option price. `None` if not solvable.

    The single entry point callers should use: it keeps the two steps consistent
    by construction, since the delta is always evaluated at the sigma that was
    just solved and at the same `t` used to solve it.
    """
    t = year_fraction(expiry, now)
    sigma = implied_vol(price=price, spot=spot, strike=strike, t=t, rate=rate, is_put=is_put)
    if sigma is None:
        return None
    delta = bs_delta(spot=spot, strike=strike, t=t, sigma=sigma, rate=rate, is_put=is_put)
    return ModelGreeks(iv=sigma, delta=delta)
