"""The IV-history seed (§4.3.1, Option 3) — the CHEAP_VOL bootstrap.

Pure function, so these are pure tests. The behaviours that matter: it must not
let CHEAP_VOL fire on day one (when we genuinely cannot know vol is cheap), it
must let it fire once a real IV drop is visible against the accumulated series, it
must step aside entirely once real data is sufficient, and it must fail *closed*
(a short, unrankable series) when it has no realized-vol shape to build a band on.
"""

from __future__ import annotations

from vigil.signals.indicators import percentile_rank
from vigil.signals.iv_seed import build_iv_history

# A realized-vol distribution with real dispersion, 0.10 .. 0.218.
RV = [0.10 + i / 500 for i in range(60)]
CHEAP_VOL_THRESHOLD = 0.33   # config: regime.cheap_vol_iv_pct


def test_day_one_ranks_today_near_the_median_so_cheap_vol_stays_off() -> None:
    """With no real history the band centres on today, so today cannot look cheap.

    This is the safety property: on the first session we have no basis to call vol
    cheap, and the seed must not manufacture one and fire the convexity sleeve.
    """
    hist = build_iv_history([], iv_atm=0.18, rv_history=RV, min_sessions=20)
    assert len(hist) == 20
    pct = percentile_rank(0.18, hist)
    assert pct is not None and pct > CHEAP_VOL_THRESHOLD


def test_a_real_iv_drop_ranks_low_and_would_fire_cheap_vol() -> None:
    """Once real sessions accrue, an IV well below them ranks cheap — as it should."""
    real = [0.20, 0.21, 0.19, 0.20, 0.22]
    hist = build_iv_history(real, iv_atm=0.10, rv_history=RV, min_sessions=20)
    pct = percentile_rank(0.10, hist)
    assert pct is not None and pct <= CHEAP_VOL_THRESHOLD


def test_enough_real_history_drops_the_seed_entirely() -> None:
    """The synthetic band is scaffolding — removed once real data can stand alone."""
    real = [0.2] * 20
    hist = build_iv_history(real, iv_atm=0.10, rv_history=RV, min_sessions=20)
    assert hist == real            # untouched: no synthetic points appended


def test_no_realized_vol_shape_fails_closed() -> None:
    """No band to build -> a short series -> percentile_rank returns None -> off.

    Failing closed is the point: a convexity sleeve fired on a fabricated
    distribution bleeds theta, so 'cannot rank' must beat 'rank confidently'.
    """
    hist = build_iv_history([0.2], iv_atm=0.1, rv_history=[], min_sessions=20)
    assert hist == [0.2]
    assert percentile_rank(0.1, hist) is None


def test_a_zero_iv_anchor_seeds_nothing() -> None:
    """An absent ATM IV (passed as 0.0 by `sense`) cannot anchor a band."""
    assert build_iv_history([], iv_atm=0.0, rv_history=RV, min_sessions=20) == []


def test_the_real_observations_are_kept_and_come_last() -> None:
    """Real values are authoritative and chronologically most recent."""
    real = [0.19, 0.20]
    hist = build_iv_history(real, iv_atm=0.18, rv_history=RV, min_sessions=20)
    assert hist[-2:] == real       # real observations preserved at the recent end
    assert len(hist) == 20


def test_the_band_centres_on_the_mean_of_real_not_on_today() -> None:
    """Anchoring on measured history (not today) is what lets today rank at all.

    If the band re-centred on today every call, today would sit at the median
    forever and cheap vol would never be detectable. Anchored on the real mean,
    a today far below that mean ranks low.
    """
    real = [0.30, 0.31, 0.29]      # mean ~0.30
    low = percentile_rank(0.12, build_iv_history(real, iv_atm=0.12, rv_history=RV, min_sessions=20))
    assert low is not None and low < 0.10
