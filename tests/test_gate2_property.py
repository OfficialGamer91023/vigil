"""Gate 2 as a property, not three examples (CLAUDE.md).

The invariant: **for any generated proposal and portfolio state, an approved
proposal never risks more than 2% of equity.** Example-based tests check the
cases we thought of; this checks the ones we did not.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.conftest import DEFAULT_EXPIRY, DEFAULT_NOW, make_leg
from vigil.config import risk_config
from vigil.domain import CONTRACT_MULTIPLIER, PortfolioState, Structure, TradeProposal
from vigil.risk.context import KernelContext
from vigil.risk.kernel import evaluate

# Money is Decimal, so the generators produce Decimals directly rather than
# floats — a float here would let binary rounding decide a boundary case.
equities = st.decimals(min_value=1_000, max_value=5_000_000, places=2)
credits = st.decimals(min_value="0.01", max_value="4.99", places=2)
widths = st.sampled_from([Decimal(1), Decimal(2), Decimal(5)])
counts = st.integers(min_value=1, max_value=500)


@st.composite
def proposals(draw: st.DrawFn) -> TradeProposal:
    width = draw(widths)
    # Credit must stay below width or the structure is not a credit spread at all.
    credit = draw(credits.filter(lambda c: c < width))
    n = draw(counts)
    short_delta = draw(st.floats(min_value=-0.45, max_value=-0.05))
    long_delta = draw(st.floats(min_value=-0.40, max_value=-0.01))
    return TradeProposal(
        structure=Structure.PUT_CREDIT_SPREAD,
        underlying="SPY",
        spot=draw(st.decimals(min_value=50, max_value=900, places=2)),
        expiry=DEFAULT_EXPIRY,
        legs=(
            make_leg("SPY260827P00761000", short=True, bid="0.50", ask="0.52", delta=short_delta),
            make_leg("SPY260827P00760000", bid="0.30", ask="0.32", delta=long_delta),
        ),
        contracts=n,
        net_credit=credit,
        width=width,
        client_order_id=f"vigil-prop-{draw(st.integers(0, 10**9))}",
        limit_price=credit,
    )


@st.composite
def books(draw: st.DrawFn) -> PortfolioState:
    equity = draw(equities)
    return PortfolioState(
        equity=equity,
        peak_equity=equity,          # at peak, so Gate 4 never masks the result
        day_pnl=Decimal(0),
    )


@given(p=proposals(), s=books())
@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
def test_no_approved_proposal_ever_exceeds_the_risk_budget(
    p: TradeProposal, s: PortfolioState
) -> None:
    decision = evaluate(p, s, KernelContext(now=DEFAULT_NOW))
    if decision.approved:
        assert p.max_loss <= risk_config().max_risk_per_trade_pct * s.equity


@given(p=proposals(), s=books())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_approval_always_implies_finite_positive_max_loss(
    p: TradeProposal, s: PortfolioState
) -> None:
    """Gate 1's promise: nothing approved can have unbounded downside."""
    decision = evaluate(p, s, KernelContext(now=DEFAULT_NOW))
    if decision.approved:
        assert p.max_loss.is_finite() and p.max_loss > 0
        # And max loss is exactly width-minus-credit per contract, times size.
        expected = (p.width - p.net_credit) * CONTRACT_MULTIPLIER * p.contracts
        assert p.max_loss == expected


@given(p=proposals(), s=books())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_every_gate_always_returns_a_verdict(p: TradeProposal, s: PortfolioState) -> None:
    """No gate may raise, and none may silently skip — the record must be complete."""
    decision = evaluate(p, s, KernelContext(now=DEFAULT_NOW))
    assert len(decision.verdicts) == 12
    assert [v.number for v in decision.verdicts] == list(range(1, 13))


def test_expiry_fixture_is_not_stale() -> None:
    """Guards the generators: a past expiry would make Gate 11 trivially true."""
    assert date(2026, 8, 26) < DEFAULT_EXPIRY
