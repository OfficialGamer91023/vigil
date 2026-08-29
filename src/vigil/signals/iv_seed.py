"""Bootstrapping the IV history the CHEAP_VOL regime needs (§4.3.1, Option 3).

The problem this exists to solve: the free tier serves **no historical implied
volatility**, so `iv_history` starts empty, `percentile_rank` returns `None`, and
the CHEAP_VOL branch — the one that funds the convexity sleeve — can never fire.
The router is permanently blind to cheap vol.

The fix, in two halves:

1. **Accumulate forward.** The worker already writes `iv_atm` to `market_snapshots`
   every cycle, so a real daily IV series accrues on its own. The caller reads it
   back (`daily_iv_history`) and passes it in as `real`.
2. **Seed the rest.** Until enough real sessions exist, pad the series with a
   synthetic band so a percentile means something. The band is *anchored to the
   measured IV level* and *shaped by the realized-vol distribution* we already
   backfill — the two data sources we actually have — and is dropped entirely once
   real data reaches `min_sessions`.

**Why this is honest rather than a fabrication.** The synthetic points are not
claimed to be past observations; they are a plausible dispersion backdrop so that
today's *real* IV can be ranked at all, and every verdict built on them is still
flagged `cold_start`. The convexity sleeve pays premium, so a spurious CHEAP_VOL
bleeds theta — which is exactly why the band is anchored to the real IV level (it
cannot drift far from reality) and why too-few points yields no ranking at all
rather than a confident one.
"""

from __future__ import annotations

from collections.abc import Sequence


def build_iv_history(
    real: Sequence[float],
    *,
    iv_atm: float,
    rv_history: Sequence[float],
    min_sessions: int,
) -> list[float]:
    """A rankable IV series: real observations, padded with a synthetic band.

    `real` is the accumulated daily IV, oldest first. `iv_atm` is today's measured
    IV, used only as the anchor when no real history exists yet. `rv_history` lends
    the band its *shape* — the relative dispersion of realized vol, which is the
    one distribution we can actually backfill.

    Returns `real` unchanged once it is long enough (the seed has done its job and
    real data is authoritative), and returns it unpadded — accepting a short,
    possibly unrankable series — when there is no usable RV shape to build a band
    from. A short series is the safe failure: `percentile_rank` returns `None`,
    CHEAP_VOL stays offline, and the convexity sleeve does not fire on noise.
    """
    real = list(real)
    if len(real) >= min_sessions:
        return real

    # Anchor on the mean of what we have actually measured — stable across
    # sessions (unlike re-centering on today, which would rank today at the median
    # every day and never detect cheap vol) and grounded in real observation. Day
    # one, with nothing measured yet, falls back to today's IV.
    anchor = (sum(real) / len(real)) if real else iv_atm
    if anchor <= 0:
        return real  # a zero/absent IV anchor cannot seed anything meaningful

    rv = [x for x in rv_history if x > 0]
    if len(rv) < 2:
        return real  # no shape to borrow — leave it short and let CHEAP_VOL stay off

    mean_rv = sum(rv) / len(rv)
    if mean_rv <= 0:
        return real

    # Multiplicative shape: each session's RV relative to the average RV, applied
    # to the anchor. Mean factor is 1, so the band centres on the anchor and
    # carries realized vol's dispersion — a plausible spread of IV levels.
    band = [anchor * (x / mean_rv) for x in rv]

    pad_needed = min_sessions - len(real)
    if len(band) > pad_needed:
        # Even stride so the pad preserves the band's full dispersion rather than
        # an arbitrary oldest slice of it.
        step = len(band) / pad_needed
        band = [band[int(i * step)] for i in range(pad_needed)]

    # Synthetic backdrop first (oldest), real observations last (recent and
    # authoritative), matching the chronological convention every history uses.
    return band + real
