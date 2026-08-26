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

from dataclasses import dataclass
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
    rv_annual: float
    # Trailing distributions. Short history is not an error — it triggers the
    # documented cold-start path rather than a silent guess.
    vrp_history: list[float]
    iv_history: list[float]

    @property
    def vrp_raw(self) -> float:
        """IV minus realized vol, both annualized. Bug 1 is fixed upstream of here."""
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


def classify(snap: MarketSnapshot, cfg: RegimeConfig | None = None) -> RegimeVerdict:
    """Map a market snapshot to a regime and the structure that regime wants."""
    c = cfg or regime_config()

    trend = trend_score(snap.daily_closes, c.ema_fast, c.ema_slow)
    vrp_pct = percentile_rank(snap.vrp_raw, snap.vrp_history)
    iv_pct = percentile_rank(snap.iv_atm, snap.iv_history)
    gap = gap_pct(snap.session_open, snap.prev_close)

    cold_start = vrp_pct is None
    if cold_start:
        # §4.3.1's documented fallback: until 60 sessions exist, use the absolute
        # sign test — and record that we did, because it is a weaker statement.
        vrp_pct = 1.0 if snap.vrp_raw > 0 else 0.0

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

    # STRESS first: it is a veto, and it must be able to override a setup that
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
