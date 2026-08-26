"""The regime router (§4.3) — especially the two bugs §4.3.1 exists to prevent."""

from __future__ import annotations

from decimal import Decimal

from vigil.domain import Regime, Structure
from vigil.signals.regime import MarketSnapshot, classify


def snap(**kw: object) -> MarketSnapshot:
    """A calm, mid-percentile baseline. Each test perturbs exactly one input."""
    base: dict[str, object] = dict(
        underlying="SPY",
        spot=Decimal("765.85"),
        prev_close=Decimal("765.00"),
        session_open=Decimal("765.20"),
        # Flat tape: EMA fast and slow coincide, so trend ~ 0 -> CHOP.
        daily_closes=[700.0] * 40,
        iv_atm=0.18,
        rv_annual=0.12,
        vrp_history=[i / 1000 for i in range(60)],   # 0.000 .. 0.059
        iv_history=[0.10 + i / 500 for i in range(60)],
    )
    base.update(kw)
    return MarketSnapshot(**base)  # type: ignore[arg-type]


def test_flat_tape_with_healthy_vrp_is_chop_and_wants_a_condor() -> None:
    v = classify(snap())
    assert v.regime is Regime.CHOP
    assert v.structure is Structure.IRON_CONDOR


def test_uptrend_sells_puts_and_downtrend_sells_calls() -> None:
    up = classify(snap(daily_closes=[700.0 + i for i in range(40)]))
    assert up.regime is Regime.TREND_UP
    assert up.structure is Structure.PUT_CREDIT_SPREAD

    down = classify(snap(daily_closes=[700.0 - i for i in range(40)]))
    assert down.regime is Regime.TREND_DOWN
    assert down.structure is Structure.CALL_CREDIT_SPREAD


def test_a_large_overnight_gap_stands_us_down_entirely() -> None:
    """Short vol into a moving market is how accounts die."""
    v = classify(snap(session_open=Decimal("755.00")))     # -1.3% gap
    assert v.regime is Regime.STRESS
    assert v.structure is None
    assert v.size_multiplier == Decimal(0)


def test_bottom_decile_vrp_fires_stress_at_reduced_size() -> None:
    """The whole reason VRP is a percentile: this branch must be reachable."""
    v = classify(snap(iv_atm=0.1201, rv_annual=0.12))   # vrp_raw ~0.0001 -> low pct
    assert v.regime is Regime.STRESS
    assert v.size_multiplier == Decimal("0.5")


def test_the_sign_test_would_have_missed_that_entirely() -> None:
    """§4.3.1 Bug 2, as an executable claim.

    `vrp_raw > 0` is TRUE in the STRESS case above — a sign test would have called
    it a normal selling day. Only the percentile separates them.
    """
    stressed = snap(iv_atm=0.1201, rv_annual=0.12)
    normal = snap(iv_atm=0.18, rv_annual=0.12)
    assert stressed.vrp_raw > 0 and normal.vrp_raw > 0        # sign test: identical
    assert classify(stressed).regime is not classify(normal).regime


def test_cheap_iv_funds_the_convexity_sleeve() -> None:
    v = classify(snap(iv_atm=0.05, rv_annual=0.02, iv_history=[0.10 + i / 100 for i in range(60)]))
    assert v.regime is Regime.CHEAP_VOL
    # A long strangle, not a debit spread: "vol is cheap" is a claim about
    # magnitude, and Gate 7 caps a directional structure at ~1 contract anyway.
    assert v.structure is Structure.LONG_STRANGLE


def test_cold_start_is_flagged_not_hidden() -> None:
    """Until 60 sessions exist the router uses the weaker absolute test — and says so."""
    v = classify(snap(vrp_history=[]))
    assert v.cold_start is True

    warm = classify(snap())
    assert warm.cold_start is False


def test_vrp_below_the_sell_floor_declines_to_sell_premium() -> None:
    """Between the stress decile (<=10%) and the sell floor (<40%): stand down.

    Not the same branch as bottom-decile STRESS, which still trades a far-OTM
    condor at half size. Here we are simply not paid enough to be short premium.
    """
    v = classify(snap(iv_atm=0.14, rv_annual=0.12))     # vrp percentile ~35%
    assert v.vrp_pct is not None and 0.10 < v.vrp_pct < 0.40
    assert v.size_multiplier == Decimal(0)
    assert v.structure is None


def test_every_verdict_carries_the_numbers_it_reasoned_from() -> None:
    """A regime call that cannot be audited is an opinion."""
    v = classify(snap())
    assert v.vrp_pct is not None and v.trend is not None
    assert v.reason
