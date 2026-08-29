"""The broken-wing (delta-skewed) condor — the trend structure that resolves B-1.

B-1 is an arithmetic fact, not a bug: a vertical's conservative credit divided by
its width equals the risk-neutral breach probability N(−d₂), which is pinned below
the short strike's |delta|. At the 0.16 target the ceiling is ~14-16%, so no
vertical at 0-2 DTE can reach Gate 9's 18% credit floor — the router builds a
structure the kernel then always rejects.

The condor collects **two** credits against **one** width, so credit/width roughly
doubles and clears the floor. The two demonstrations that matter are pinned here:

1. On the *same chain*, the lone 0.16 vertical the router used to build fails Gate
   9, and the broken-wing condor passes. This is B-1, reproduced and then fixed.
2. The skew is directional: a bullish trend leans the package net-long, a bearish
   trend net-short — while both sides stay defined-risk, so it is never naked.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from tests.conftest import DEFAULT_EXPIRY
from vigil.config import risk_config
from vigil.domain import Regime, Structure
from vigil.risk.kernel import evaluate
from vigil.signals.regime import RegimeVerdict
from vigil.strategy.candidates import (
    build_broken_wing_condor,
    build_for_regime,
    build_vertical,
)

SPOT = Decimal("765.85")
BUDGET = Decimal(2000)
DELTA_BUDGET = Decimal(5000)

# A self-consistent $1-strike ladder around 765. Premiums are built so the
# difference between adjacent strikes equals the local delta — which is exactly
# why a lone 0.16 vertical collects ~13% of width here (below Gate 9's floor) and
# is not an artefact of hand-picked prices. Strike: (delta, bid, ask).
_CALLS = {
    765: (0.50, "2.34", "2.37"),
    766: (0.42, "1.84", "1.87"),
    767: (0.35, "1.42", "1.45"),
    768: (0.29, "1.07", "1.10"),
    769: (0.22, "0.78", "0.81"),
    770: (0.16, "0.56", "0.59"),
    771: (0.11, "0.40", "0.43"),
    772: (0.08, "0.29", "0.32"),
}
_PUTS = {
    765: (-0.50, "2.34", "2.37"),
    764: (-0.42, "1.84", "1.87"),
    763: (-0.35, "1.42", "1.45"),
    762: (-0.29, "1.07", "1.10"),
    761: (-0.22, "0.78", "0.81"),
    760: (-0.16, "0.56", "0.59"),
    759: (-0.11, "0.40", "0.43"),
    758: (-0.08, "0.29", "0.32"),
}


@pytest.fixture
def chain(make_contract):
    """Both sides of the ladder, built through the live JSON parsing path."""
    out = []
    for strike, (delta, bid, ask) in _CALLS.items():
        out.append(make_contract(f"SPY260827C00{strike}000", delta=delta, bid=bid, ask=ask))
    for strike, (delta, bid, ask) in _PUTS.items():
        out.append(make_contract(f"SPY260827P00{strike}000", delta=delta, bid=bid, ask=ask))
    return out


def _oi(chain) -> dict[str, int]:
    return {c.occ.raw: 5_000 for c in chain}


def _bwc(chain, *, trend: float):
    return build_broken_wing_condor(
        chain, underlying="SPY", spot=SPOT, expiry=DEFAULT_EXPIRY, trend=trend,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain),
    )


# --------------------------------------------------------------------------- #
# B-1: the vertical fails the floor the condor clears, on the same chain
# --------------------------------------------------------------------------- #

def test_the_lone_vertical_cannot_reach_the_credit_floor(chain, flat_book, ctx) -> None:
    """The structure the router used to build in a trend. Gate 9 rejects it — not
    because the chain is adversarial, but because credit/width < |delta| always."""
    vert = build_vertical(
        chain, underlying="SPY", spot=SPOT, expiry=DEFAULT_EXPIRY, is_put=True,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain))
    assert vert is not None
    assert vert.credit_pct_of_width < risk_config().min_credit_pct_of_width
    assert 9 in {v.number for v in evaluate(vert, flat_book, ctx).failures}


def test_the_broken_wing_condor_clears_the_floor_the_vertical_missed(
    chain, flat_book, ctx
) -> None:
    """Same chain, two credits: credit/width roughly doubles and clears 18%."""
    p = _bwc(chain, trend=0.01)
    assert p is not None
    assert p.structure is Structure.BROKEN_WING_CONDOR
    assert p.credit_pct_of_width >= risk_config().min_credit_pct_of_width
    assert 9 not in {v.number for v in evaluate(p, flat_book, ctx).failures}


def test_the_broken_wing_condor_passes_every_gate(chain, flat_book, ctx) -> None:
    """The whole point: a structure the kernel actually approves in a trend."""
    p = _bwc(chain, trend=0.01)
    assert p is not None
    d = evaluate(p, flat_book, ctx)
    assert d.approved, d.summary


# --------------------------------------------------------------------------- #
# The skew is directional, and defined-risk on both sides
# --------------------------------------------------------------------------- #

def test_a_bullish_trend_leans_the_package_net_long(chain) -> None:
    """Selling the put nearer the money tilts net delta positive — with the trend.
    A short put carries +delta, so the near side is what does the leaning."""
    p = _bwc(chain, trend=0.02)
    assert p is not None
    assert p.dollar_delta > 0
    # The near (put) short strike sits closer to spot than the far (call) short.
    short_put = next(leg for leg in p.legs if leg.occ.is_put and leg.is_short)
    short_call = next(leg for leg in p.legs if not leg.occ.is_put and leg.is_short)
    assert SPOT - short_put.occ.strike < short_call.occ.strike - SPOT


def test_a_bearish_trend_leans_the_package_net_short(chain) -> None:
    p = _bwc(chain, trend=-0.02)
    assert p is not None
    assert p.dollar_delta < 0
    short_put = next(leg for leg in p.legs if leg.occ.is_put and leg.is_short)
    short_call = next(leg for leg in p.legs if not leg.occ.is_put and leg.is_short)
    # Now the call is the near side, sold closer to spot than the put.
    assert short_call.occ.strike - SPOT < SPOT - short_put.occ.strike


def test_max_loss_is_one_width_not_two(chain) -> None:
    """A condor cannot finish below the put spread and above the call spread at
    once, so the exposure is one width minus the total credit — the same rule
    Gate 1's derived-width check enforces, which is why both wings stay equal."""
    p = _bwc(chain, trend=0.01)
    assert p is not None
    assert p.max_loss_per_contract == (p.width - p.net_credit) * 100
    # Every short leg is covered by a long leg on its own right — never naked.
    for is_put in (True, False):
        side = [leg for leg in p.legs if leg.occ.is_put == is_put]
        shorts = sum(leg.ratio_qty for leg in side if leg.is_short)
        longs = sum(leg.ratio_qty for leg in side if not leg.is_short)
        assert shorts == longs == 1


# --------------------------------------------------------------------------- #
# Regime dispatch
# --------------------------------------------------------------------------- #

def _verdict(structure, trend):
    return RegimeVerdict(
        regime=Regime.TREND_UP, structure=structure, reason="test", trend=trend)


def test_dispatch_routes_a_trend_verdict_to_the_condor(chain) -> None:
    p = build_for_regime(
        _verdict(Structure.BROKEN_WING_CONDOR, 0.02), chain,
        underlying="SPY", spot=SPOT, expiry=DEFAULT_EXPIRY,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain))
    assert p is not None and p.structure is Structure.BROKEN_WING_CONDOR


def test_dispatch_declines_a_broken_wing_without_a_direction(chain) -> None:
    """No trend read, nothing to skew toward — decline and let CHOP's symmetric
    condor handle a flat tape instead of building a worse one here."""
    p = build_for_regime(
        _verdict(Structure.BROKEN_WING_CONDOR, None), chain,
        underlying="SPY", spot=SPOT, expiry=DEFAULT_EXPIRY,
        risk_budget=BUDGET, remaining_delta_budget=DELTA_BUDGET, open_interest=_oi(chain))
    assert p is None
