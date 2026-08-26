"""The convexity sleeve: the debit-spread builder and the regime dispatch.

Two failures are pinned here, and they are different in kind.

The first is a **silent inversion**: the old dispatch mapped every structure that
was not a put credit spread onto `is_put=False` and built a *call credit spread*,
so a CHEAP-VOL session — the router's own signal that volatility is cheap and
worth buying — sold premium instead. Nothing raised; the trade was simply the
opposite of the one the regime asked for.

The second is **arithmetic sign**. A debit structure pays premium rather than
collecting it, so max loss, the ladder direction and the profit target all invert.
Reusing the credit-path formulas produces orders that never fill, or worse, a
resting exit priced below cost.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from vigil.domain import Regime, Structure
from vigil.signals.regime import RegimeVerdict
from vigil.strategy.candidates import (
    build_debit_spread,
    build_for_regime,
    build_long_strangle,
)

EXPIRY = date(2026, 8, 28)
SPOT = Decimal("765.00")
BUDGET = Decimal(2000)
DELTA_BUDGET = Decimal(5000)


@pytest.fixture
def chain(make_contract):
    """A small two-sided chain with a realistic delta gradient around the money.

    Built through the same JSON parsing path a live snapshot takes, so a builder
    that works here works against the feed.
    """
    out = []
    # Calls: delta falls as the strike rises above spot.
    for strike, delta, bid, ask in (
        (765, 0.52, "3.10", "3.14"),
        (766, 0.35, "2.20", "2.24"),
        (767, 0.29, "1.70", "1.74"),
        (768, 0.22, "1.20", "1.24"),
        (769, 0.16, "0.80", "0.84"),
        (770, 0.11, "0.50", "0.54"),
    ):
        out.append(make_contract(f"SPY260828C00{strike}000", delta=delta, bid=bid, ask=ask))
    # Puts: |delta| falls as the strike drops below spot.
    for strike, delta, bid, ask in (
        (765, -0.48, "3.00", "3.04"),
        (764, -0.35, "2.20", "2.24"),
        (763, -0.29, "1.70", "1.74"),
        (762, -0.22, "1.20", "1.24"),
        (761, -0.16, "0.80", "0.84"),
        (760, -0.11, "0.50", "0.54"),
    ):
        out.append(make_contract(f"SPY260828P00{strike}000", delta=delta, bid=bid, ask=ask))
    return out


def _oi(chain) -> dict[str, int]:
    return {c.occ.raw: 5_000 for c in chain}


# --------------------------------------------------------------------------- #
# The builder
# --------------------------------------------------------------------------- #

def test_the_debit_spread_pays_premium_rather_than_collecting_it(chain) -> None:
    """`net_credit` is negative for a debit package. That sign is what every
    downstream branch reads to tell the two structures apart."""
    p = build_debit_spread(
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY, is_put=False,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain),
    )
    assert p is not None
    assert p.structure is Structure.DEBIT_SPREAD
    assert p.net_credit < 0
    assert not p.is_credit


def test_the_long_leg_is_nearer_the_money_than_the_short_leg(chain) -> None:
    """The defining geometry. Reversed, this is a credit spread wearing the
    convexity sleeve's name."""
    p = build_debit_spread(
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY, is_put=False,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain),
    )
    assert p is not None
    long_leg = next(leg for leg in p.legs if not leg.is_short)
    short_leg = next(leg for leg in p.legs if leg.is_short)
    assert long_leg.occ.strike < short_leg.occ.strike, "call debit spread buys the lower strike"
    assert abs(long_leg.delta) > abs(short_leg.delta)


def test_max_loss_is_the_debit_paid_not_the_width(chain) -> None:
    """A debit spread cannot lose more than it cost. Using `width − credit` here
    would overstate the loss and silently undersize the sleeve."""
    p = build_debit_spread(
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY, is_put=False,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain),
    )
    assert p is not None
    assert p.max_loss_per_contract == abs(p.net_credit) * 100
    assert p.max_profit_per_contract == (p.width - abs(p.net_credit)) * 100


def test_the_limit_price_stays_a_positive_magnitude(chain) -> None:
    """Alpaca's mleg ticket takes a positive package price; direction lives on
    the legs. A negative debit here reads to Gate 12 as 200% off mid."""
    p = build_debit_spread(
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY, is_put=False,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain),
    )
    assert p is not None and p.limit_price > 0


def test_a_debit_at_or_above_the_width_is_declined(make_contract) -> None:
    """Paying a full width for a spread whose max payoff *is* that width cannot
    profit under any outcome — max profit would be zero or negative.

    Priced deliberately: the $766 long asks $2.24 while the $767 short bids only
    $1.10, so the package costs $1.14 against $1.00 of width.
    """
    broken = [
        make_contract("SPY260828C00766000", delta=0.35, bid="2.20", ask="2.24"),
        make_contract("SPY260828C00767000", delta=0.29, bid="1.10", ask="1.14"),
    ]
    assert build_debit_spread(
        broken, underlying="SPY", spot=SPOT, expiry=EXPIRY, is_put=False,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET,
        open_interest=_oi(broken),
    ) is None


# --------------------------------------------------------------------------- #
# The dispatch — the silent inversion
# --------------------------------------------------------------------------- #

def _verdict(regime: Regime, structure: Structure | None, trend: float | None) -> RegimeVerdict:
    return RegimeVerdict(regime=regime, structure=structure, reason="test", trend=trend)


def test_cheap_vol_never_produces_a_credit_structure(chain) -> None:
    """**The regression this module exists for.** CHEAP-VOL asks to buy movement.
    The old dispatch built a call credit spread — short premium — instead."""
    p = build_for_regime(
        _verdict(Regime.CHEAP_VOL, Structure.DEBIT_SPREAD, trend=0.004),
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain),
    )
    assert p is not None
    assert p.structure is Structure.DEBIT_SPREAD
    assert not p.is_credit, "CHEAP-VOL sold premium — the inversion is back"


def test_the_debit_spread_follows_the_trend(chain) -> None:
    """Buy calls into strength, puts into weakness."""
    up = build_for_regime(
        _verdict(Regime.CHEAP_VOL, Structure.DEBIT_SPREAD, trend=0.004),
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain))
    down = build_for_regime(
        _verdict(Regime.CHEAP_VOL, Structure.DEBIT_SPREAD, trend=-0.004),
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain))
    assert up is not None and down is not None
    assert not up.legs[0].occ.is_put
    assert down.legs[0].occ.is_put


def test_a_trendless_cheap_vol_session_declines_rather_than_guessing(chain) -> None:
    """A debit spread with no directional read is a coin flip with a premium
    attached."""
    assert build_for_regime(
        _verdict(Regime.CHEAP_VOL, Structure.DEBIT_SPREAD, trend=None),
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET,
        open_interest=_oi(chain)) is None


def test_the_sleeve_takes_a_share_of_the_budget_not_all_of_it(chain) -> None:
    """§4.5: 20-25%. A debit spread can lose its entire premium without the
    underlying doing anything unusual."""
    from vigil.config import strategy_config

    p = build_for_regime(
        _verdict(Regime.CHEAP_VOL, Structure.DEBIT_SPREAD, trend=0.004),
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain))
    assert p is not None
    assert p.max_loss <= BUDGET * strategy_config().convexity_risk_share


def test_a_stand_down_verdict_builds_nothing(chain) -> None:
    assert build_for_regime(
        _verdict(Regime.STRESS, None, trend=0.0),
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET,
        open_interest=_oi(chain)) is None


@pytest.mark.parametrize(
    ("structure", "expected"),
    [
        (Structure.PUT_CREDIT_SPREAD, Structure.PUT_CREDIT_SPREAD),
        (Structure.CALL_CREDIT_SPREAD, Structure.CALL_CREDIT_SPREAD),
        (Structure.IRON_CONDOR, Structure.IRON_CONDOR),
        (Structure.DEBIT_SPREAD, Structure.DEBIT_SPREAD),
    ],
)
def test_every_structure_routes_to_its_own_builder(chain, structure, expected) -> None:
    """Exhaustiveness, stated as a test rather than trusted to a `match`.

    If a new `Structure` member is added without a branch, this parametrisation
    is where it should be noticed — not in a live session, where it would be
    routed to whichever builder happens to accept its arguments.
    """
    p = build_for_regime(
        _verdict(Regime.CHOP, structure, trend=0.004),
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain))
    assert p is not None, f"{structure} built nothing"
    assert p.structure is expected


def test_the_dispatch_covers_every_member_of_the_structure_enum() -> None:
    """A guard against the enum growing past the dispatch."""
    import inspect

    from vigil.strategy import candidates

    src = inspect.getsource(candidates.build_for_regime)
    missing = [s.name for s in Structure if f"Structure.{s.name}" not in src]
    assert not missing, f"build_for_regime has no branch for: {missing}"


# --------------------------------------------------------------------------- #
# The long strangle — the sleeve that Gate 7 does not crush
# --------------------------------------------------------------------------- #

def test_the_strangle_is_delta_neutral_enough_that_gate_7_stops_binding(chain) -> None:
    """**The whole reason this structure exists.** A directional debit spread
    carries ~$4,590 of dollar delta per contract against Gate 7's $5,000 budget
    for the entire book, so one contract consumes 92% of it."""
    strangle = build_long_strangle(
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain))
    spread = build_debit_spread(
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY, is_put=False,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain))
    assert strangle is not None and spread is not None

    per_contract = abs(strangle.dollar_delta) / strangle.contracts
    assert per_contract < abs(spread.dollar_delta) / spread.contracts / 10


def test_the_strangle_buys_both_sides(chain) -> None:
    p = build_long_strangle(
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain))
    assert p is not None
    assert p.structure is Structure.LONG_STRANGLE
    assert {leg.occ.is_put for leg in p.legs} == {True, False}
    assert all(not leg.is_short for leg in p.legs), "a strangle has no short leg"
    assert p.is_long_only


def test_both_strangle_legs_are_out_of_the_money(chain) -> None:
    """An ITM leg is intrinsic value, not convexity — it costs far more and buys
    less movement per dollar."""
    p = build_long_strangle(
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain))
    assert p is not None
    call = next(leg for leg in p.legs if not leg.occ.is_put)
    put = next(leg for leg in p.legs if leg.occ.is_put)
    assert call.occ.strike > SPOT
    assert put.occ.strike < SPOT


def test_max_loss_is_the_premium_and_max_profit_is_unbounded(chain) -> None:
    """With no short leg there is nothing to be assigned on, so the cheque we
    wrote is the whole exposure — and nothing caps the upside."""
    p = build_long_strangle(
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain))
    assert p is not None
    assert p.max_loss_per_contract == abs(p.net_credit) * 100
    assert p.max_profit_per_contract == Decimal("Infinity")
    assert p.max_loss.is_finite(), "Gate 1 requires a finite, computable max loss"


def test_the_strangle_declares_no_width(chain) -> None:
    """Width is meaningless across a call and a put — the distance between the
    strikes is the range the underlying must escape, not the edge of a payoff."""
    p = build_long_strangle(
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain))
    assert p is not None and p.width == 0


def test_cheap_vol_now_routes_to_the_strangle(chain) -> None:
    p = build_for_regime(
        _verdict(Regime.CHEAP_VOL, Structure.LONG_STRANGLE, trend=0.004),
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain))
    assert p is not None
    assert p.structure is Structure.LONG_STRANGLE
    assert not p.is_credit


def test_the_strangle_needs_no_trend_read(chain) -> None:
    """Unlike the debit spread: CHEAP-VOL is a claim about magnitude, so a
    trendless session is not a reason to decline."""
    p = build_for_regime(
        _verdict(Regime.CHEAP_VOL, Structure.LONG_STRANGLE, trend=None),
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain))
    assert p is not None


def test_the_sleeve_now_deploys_a_real_share_of_its_budget(chain) -> None:
    """The regression that motivated the whole structure change.

    The debit-spread sleeve deployed $54 of $440 — 12%, and six times smaller
    than the 5% allocation PLAN §12 rejected as decoration. A hedge that cannot
    pay is not a hedge.
    """
    from vigil.config import strategy_config

    sleeve = BUDGET * strategy_config().convexity_risk_share
    p = build_for_regime(
        _verdict(Regime.CHEAP_VOL, Structure.LONG_STRANGLE, trend=0.004),
        chain, underlying="SPY", spot=SPOT, expiry=EXPIRY,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain))
    assert p is not None
    assert p.max_loss <= sleeve, "the sleeve must still respect its own budget"
    assert p.max_loss / sleeve > Decimal("0.5"), (
        f"sleeve deployed only {p.max_loss / sleeve:.0%} of ${sleeve} — decoration again"
    )
