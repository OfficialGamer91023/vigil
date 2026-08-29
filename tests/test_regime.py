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


# --------------------------------------------------------------------------- #
# Option 1 — the cold-start vrp_raw override of the sell floor (§4.3.1)
# --------------------------------------------------------------------------- #

def test_rich_vrp_raw_overrides_the_cold_start_sell_floor_at_reduced_size() -> None:
    """A real IV−RV measurement beats the proxy that benched us.

    Cold start, the RV proxy lands the VRP percentile in the sell-floor zone
    (10–40%), but the real vrp_raw shows premium demonstrably rich — so the router
    trades at reduced size rather than standing down.
    """
    calm = [0.10 + i / 1000 for i in range(60)]     # 0.100 .. 0.159
    # rv=0.14 ranks high in `calm` -> proxy VRP ~33% (sell-floor zone).
    # iv=0.20 -> vrp_raw=0.06, well over the 0.03 rich floor.
    v = classify(snap(vrp_history=[], iv_atm=0.20, rv_annual=0.14, rv_history=calm))
    assert v.cold_start is True
    assert v.vrp_pct is not None and 0.10 < v.vrp_pct < 0.40
    assert v.structure is not None                  # not benched
    assert v.size_multiplier == Decimal("0.5")      # but at reduced size
    assert "override" in v.reason


def test_the_override_does_not_fire_when_vrp_raw_is_not_rich() -> None:
    """Same benching proxy, but the real measurement does not confirm rich premium."""
    calm = [0.10 + i / 1000 for i in range(60)]
    # iv=0.15 -> vrp_raw=0.01, below the 0.03 floor: no override, stand down.
    v = classify(snap(vrp_history=[], iv_atm=0.15, rv_annual=0.14, rv_history=calm))
    assert v.cold_start is True
    assert v.size_multiplier == Decimal(0)
    assert v.structure is None


def test_the_override_never_touches_a_warm_percentile() -> None:
    """A real VRP distribution is trusted as-is — the override is cold-start only.

    The default snap carries a full `vrp_history`, so even with a rich vrp_raw the
    sell-floor stand-down stands: we are not second-guessing a real percentile.
    """
    # iv=0.14, rv=0.12 -> vrp_raw=0.02 (< floor anyway) and warm vrp_pct ~35%.
    v = classify(snap(iv_atm=0.14, rv_annual=0.12))
    assert v.cold_start is False
    assert v.size_multiplier == Decimal(0)
    assert v.structure is None


def test_the_override_cannot_rescue_a_stress_or_gap_veto() -> None:
    """The stress decile and the gap veto sit above the sell floor and are final."""
    # A large gap: STRESS, size 0, regardless of how rich vrp_raw is.
    v = classify(snap(vrp_history=[], iv_atm=0.30, rv_annual=0.14,
                       rv_history=[0.10 + i / 1000 for i in range(60)],
                       session_open=Decimal("755.00")))
    assert v.regime is Regime.STRESS
    assert v.size_multiplier == Decimal(0)


def test_every_verdict_carries_the_numbers_it_reasoned_from() -> None:
    """A regime call that cannot be audited is an opinion."""
    v = classify(snap())
    assert v.vrp_pct is not None and v.trend is not None
    assert v.reason


# --------------------------------------------------------------------------- #
# Cold start — the safety regime must be reachable on day one
# --------------------------------------------------------------------------- #

def test_cold_start_ranks_realized_vol_when_the_vrp_series_is_empty() -> None:
    """§4.3.1's own fallback (`vrp_raw > 0`) is true nearly every session, so
    STRESS would never fire and the router would collapse to 'always sell'.

    Realized vol is backfillable where historical IV is not, and it is the moving
    half of `iv − rv`: a session in the top decile of RV lands in the bottom
    decile of VRP, which is exactly when we want to stand down.
    """
    calm = [0.05 + i / 1000 for i in range(60)]   # 0.050 .. 0.109
    v = classify(snap(vrp_history=[], rv_annual=0.30, rv_history=calm))
    assert v.cold_start is True
    assert v.regime is Regime.STRESS, "a violently moving session did not stand down"


def test_cold_start_still_sells_premium_on_a_quiet_session() -> None:
    """The proxy must not simply refuse to trade — a gate that never passes is as
    broken as one that never fires (§5.2)."""
    calm = [0.05 + i / 1000 for i in range(60)]
    v = classify(snap(vrp_history=[], rv_annual=0.051, rv_history=calm))
    assert v.cold_start is True
    assert v.regime is not Regime.STRESS
    assert v.structure is not None


def test_cold_start_falls_back_to_the_sign_test_with_no_backfill_at_all() -> None:
    """Degenerate, and honestly labelled — but better than an exception at 09:31."""
    v = classify(snap(vrp_history=[], rv_history=[]))
    assert v.cold_start is True
    assert v.structure is not None


def test_a_warm_vrp_series_ignores_the_realized_vol_proxy() -> None:
    """Once the real distribution exists it wins; the proxy is cold start only."""
    v = classify(snap(rv_history=[0.05 + i / 1000 for i in range(60)]))
    assert v.cold_start is False


# --------------------------------------------------------------------------- #
# Unmeasurable realized vol must fail CLOSED, not open
# --------------------------------------------------------------------------- #

def test_unmeasurable_realized_vol_stands_down_rather_than_selling() -> None:
    """The regression this pins is a fail-*open*, which is the dangerous direction.

    Callers used to spell "could not measure realized vol" as `rv_annual = 0.0`.
    Zero is not a missing number — it is the calmest market that has ever existed,
    so `vrp_raw` collapsed to `iv_atm`, the realized-vol percentile ranked at the
    very bottom, and the cold-start proxy *inverts* that into a VRP percentile of
    100%. A half-day, an IEX outage or the first ten minutes of a session then
    read as "premium has never been richer" and the router sold into it at full
    size. A missing measurement became maximum conviction.
    """
    calm = [0.05 + i / 1000 for i in range(60)]
    v = classify(snap(vrp_history=[], rv_annual=None, rv_history=calm))

    assert v.regime is Regime.STRESS
    assert v.structure is None                    # no structure at all, not a small one
    assert v.size_multiplier == Decimal(0)
    assert "unmeasurable" in v.reason


def test_unmeasurable_realized_vol_leaves_vrp_raw_uncomputable() -> None:
    """`None` propagates rather than being papered over with a zero."""
    assert snap(rv_annual=None).vrp_raw is None
    assert snap(rv_annual=0.12).vrp_raw == 0.18 - 0.12


def test_the_stand_down_outranks_a_tradeable_looking_gap() -> None:
    """Ordering: with no VRP read there is no regime read, so nothing else applies."""
    v = classify(snap(vrp_history=[], rv_annual=None, rv_history=[], iv_atm=0.30))
    assert v.structure is None
