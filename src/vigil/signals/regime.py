"""The regime router (§4.3). Deterministic — the LLM interprets it, never computes it.

Two bugs this module is built to avoid, both from §4.3.1:

1. **Units.** ATM IV is annualized; a per-5-minute realized vol is not. They are
   compared only after `signals.vol.realized_vol` has annualized the latter.
2. **A degenerate sign test.** `IV - RV > 0` is true on nearly every session, so
   STRESS would never fire and the router would collapse to "always sell". VRP is
   therefore a **percentile within its own trailing distribution**, which is a
   statement that can actually be false today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from vigil.config import RegimeConfig, regime_config
from vigil.domain import Regime, Structure
from vigil.signals.indicators import gap_pct, percentile_rank, trend_score


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Everything the router reads. All of it derived from the **underlying** (§1.2).

    Option data never reaches this object except as `iv_atm`, which is used for
    pricing context rather than as a signal in its own right.
    """

    underlying: str
    spot: Decimal
    prev_close: Decimal
    session_open: Decimal
    daily_closes: list[float]
    iv_atm: float
    # **Optional, and `None` is not the same as `0.0`.** Realized vol is
    # unmeasurable often enough to matter — a half day, an IEX outage, the first
    # ten minutes of a session. Typing it `float` forced every caller to spell
    # that case `rv or 0.0`, which is the single most dangerous value it could
    # take: `vrp_raw` collapses to `iv_atm`, the realized-vol percentile ranks at
    # the very bottom, the cold-start proxy inverts that to 100%, and the router
    # reads "premium has never been richer — sell." A missing measurement became
    # maximum conviction. `None` makes it a stand-down instead.
    rv_annual: float | None
    # Trailing distributions. Short history is not an error — it triggers the
    # documented cold-start path rather than a silent guess.
    vrp_history: list[float]
    iv_history: list[float]
    # Trailing realized vol, from `scripts/backfill_vrp.py`. Backfillable where
    # `vrp_history` is not, which is what makes the cold-start path usable —
    # see `_cold_start_vrp_pct`.
    rv_history: list[float] = field(default_factory=list)

    @property
    def vrp_raw(self) -> float | None:
        """IV minus realized vol, both annualized. Bug 1 is fixed upstream of here.

        `None` when realized vol could not be measured: the difference is not
        computable, and inventing a zero for the missing half would make it look
        like it was.
        """
        if self.rv_annual is None:
            return None
        return self.iv_atm - self.rv_annual


@dataclass(frozen=True, slots=True)
class RegimeVerdict:
    regime: Regime
    structure: Structure | None
    reason: str
    # True when the VRP percentile could not be computed and the absolute
    # fallback was used. Every cycle that reasons this way must say so (§4.3.1).
    cold_start: bool = False
    size_multiplier: Decimal = Decimal(1)
    trend: float | None = None
    vrp_pct: float | None = None
    iv_pct: float | None = None


def _cold_start_vrp_pct(snap: MarketSnapshot) -> float | None:
    """Stand in for the VRP percentile before 60 sessions of `vrp_raw` exist.

    **Why a stand-in is needed at all.** `vrp_raw = iv_atm − rv_annual`, and the
    free tier serves no historical implied volatility, so the true series can only
    accumulate forward one session at a time. On day one there is nothing to rank
    against.

    **Why not §4.3.1's documented fallback.** That fallback is the absolute test
    `vrp_raw > 0` — and §4.3.1 spends a page explaining that short-dated IV sits
    above trailing RV on nearly every session, so the test is true almost always,
    **STRESS never fires, and the router collapses to "always sell."** Using it
    means starting the competition with the safety regime disabled.

    **The proxy.** Realized vol *is* backfillable, and it is the moving half of
    the subtraction: with IV roughly stable session to session, `vrp_raw` falls
    precisely when `rv_annual` rises. So rank today's realized vol within its own
    trailing distribution and invert it — a session whose RV is in the top decile
    lands at a VRP percentile in the bottom decile, and STRESS fires.

    Read plainly, that says *"the market is moving unusually fast today, so do not
    sell premium into it"* — which is what STRESS is for, and arguably a more
    direct statement of it than the VRP formulation. It is still a proxy, and the
    verdict is still flagged `cold_start`, because the assumption that IV is
    stable is exactly the thing that fails on the days this matters most.

    **When the proxy is unavailable too.** The proxy *is* today's realized vol, so
    without it there is no proxy — and §4.3.1's own absolute test (`vrp_raw > 0`)
    needs the same missing number. Returning `None` says "cannot assess", which
    the router turns into a stand-down. A deliberate fail-closed: the alternative
    reading, *"no evidence vol is expensive, therefore sell"*, is how a data
    outage becomes a position.
    """
    if snap.rv_annual is None:
        return None
    if snap.rv_history:
        rv_pct = percentile_rank(snap.rv_annual, snap.rv_history)
        if rv_pct is not None:
            # High realized vol -> low VRP percentile. The inversion is the point.
            return 1.0 - rv_pct
    # No backfill, but today's RV is measured. Fall back to §4.3.1's absolute
    # test, degenerate as it is — a weak signal honestly labelled beats an
    # exception at 09:31.
    raw = snap.vrp_raw
    return 1.0 if raw is not None and raw > 0 else 0.0


def classify(snap: MarketSnapshot, cfg: RegimeConfig | None = None) -> RegimeVerdict:
    """Map a market snapshot to a regime and the structure that regime wants."""
    c = cfg or regime_config()

    trend = trend_score(snap.daily_closes, c.ema_fast, c.ema_slow)
    raw = snap.vrp_raw
    vrp_pct = percentile_rank(raw, snap.vrp_history) if raw is not None else None
    iv_pct = percentile_rank(snap.iv_atm, snap.iv_history)
    gap = gap_pct(snap.session_open, snap.prev_close)

    cold_start = vrp_pct is None
    if cold_start:
        vrp_pct = _cold_start_vrp_pct(snap)

    def verdict(
        regime: Regime,
        structure: Structure | None,
        reason: str,
        size_multiplier: Decimal = Decimal(1),
    ) -> RegimeVerdict:
        """Closes over the measured values so every branch reports the same context.

        A local closure rather than a dict splat: `**kwargs` would erase the types
        and let a renamed field fail at runtime instead of at type-check time.
        """
        return RegimeVerdict(
            regime=regime, structure=structure, reason=reason,
            cold_start=cold_start, size_multiplier=size_multiplier,
            trend=trend, vrp_pct=vrp_pct, iv_pct=iv_pct,
        )

    # **Unmeasurable comes before everything, including the gap test.** If the VRP
    # input is missing there is no regime read at all, and a router that answered
    # anyway would be guessing with the confidence of a measurement.
    if vrp_pct is None:
        return verdict(
            Regime.STRESS, None,
            "realized vol unmeasurable this cycle — no VRP read, standing down",
            size_multiplier=Decimal(0))

    # STRESS next: it is a veto, and it must be able to override a setup that
    # otherwise looks tradeable. Short vol into a moving market is how accounts die.
    if abs(gap) > c.stress_gap_pct:
        return verdict(
            Regime.STRESS, None,
            f"overnight gap {gap:.2%} exceeds {c.stress_gap_pct:.2%} — standing down",
            size_multiplier=Decimal(0))
    if Decimal(str(vrp_pct)) <= c.vrp_stress_pct:
        return verdict(
            Regime.STRESS, Structure.IRON_CONDOR,
            f"VRP percentile {vrp_pct:.0%} in the bottom decile — vol is cheap "
            f"relative to recent pricing; far-OTM condor at reduced size only",
            size_multiplier=Decimal("0.5"))

    # CHEAP-VOL funds the convexity sleeve rather than selling into it. The
    # structure is a long strangle, not a directional debit spread: "vol is
    # cheap" is a claim about *magnitude*, and a spread would additionally
    # require being right about direction — which Gate 7 then caps at roughly one
    # contract anyway (see strategy.candidates.build_long_strangle).
    if iv_pct is not None and Decimal(str(iv_pct)) <= c.cheap_vol_iv_pct:
        return verdict(
            Regime.CHEAP_VOL, Structure.LONG_STRANGLE,
            f"IV percentile {iv_pct:.0%} in the bottom third — buy movement while "
            f"it is on sale")

    # Below the sell floor we are not paid enough to be short premium.
    if Decimal(str(vrp_pct)) < c.vrp_sell_floor_pct:
        return verdict(
            Regime.STRESS, None,
            f"VRP percentile {vrp_pct:.0%} below the {c.vrp_sell_floor_pct:.0%} "
            f"sell floor — not paid enough to be short premium",
            size_multiplier=Decimal(0))

    if trend is None:
        return verdict(
            Regime.CHOP, Structure.IRON_CONDOR,
            "insufficient history for a trend read — defaulting to direction-agnostic")

    t = Decimal(str(trend))
    if t > c.trend_threshold:
        return verdict(
            Regime.TREND_UP, Structure.PUT_CREDIT_SPREAD,
            f"trend +{trend:.2%} above threshold — sell fear below the trend")
    if t < -c.trend_threshold:
        return verdict(
            Regime.TREND_DOWN, Structure.CALL_CREDIT_SPREAD,
            f"trend {trend:.2%} below threshold — sell greed above the trend")

    return verdict(
        Regime.CHOP, Structure.IRON_CONDOR,
        f"trend {trend:+.2%} inside +/-{c.trend_threshold:.2%} — max theta, "
        f"direction-agnostic")
