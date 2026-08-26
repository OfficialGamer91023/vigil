"""Position sizing (§4.4.3). Width is fixed; contract count is the free variable."""

from __future__ import annotations

from decimal import Decimal

from vigil.strategy.sizing import (
    max_contracts_for_budget,
    max_contracts_for_dollar_delta,
    size_position,
)


def test_sizes_to_the_risk_budget_and_rounds_down() -> None:
    """$1 width, $0.20 credit -> $80 max loss per contract. $2,000 budget -> 25."""
    assert max_contracts_for_budget(Decimal(1), Decimal("0.20"), Decimal(2_000)) == 25


def test_rounding_is_always_down_never_nearest() -> None:
    """Rounding up would let sizing approve what Gate 2 then rejects — two
    components disagreeing about the same rule."""
    # $85 per contract into $2,000 is 23.5 -> 23, not 24.
    assert max_contracts_for_budget(Decimal(1), Decimal("0.15"), Decimal(2_000)) == 23


def test_a_credit_at_or_above_width_sizes_to_zero() -> None:
    """Not a bargain — an impossible structure, so refuse rather than divide by zero."""
    assert max_contracts_for_budget(Decimal(1), Decimal(1), Decimal(2_000)) == 0


def test_delta_neutral_structures_are_unconstrained_by_gate_seven() -> None:
    """Returning 0 here would silently ban iron condors."""
    assert max_contracts_for_dollar_delta(Decimal(0), Decimal(5_000)) > 1000


def test_the_binding_constraint_wins() -> None:
    """Gate 7 binds ~15x harder than Gate 2 on a directional SPY spread."""
    n = size_position(
        width=Decimal(1),
        net_credit=Decimal("0.20"),
        risk_budget=Decimal(2_000),        # would allow 25
        delta_per_contract=Decimal(3_829),  # ~$3.8k of dollar delta each
        remaining_delta_budget=Decimal(5_000),
    )
    assert n == 1


def test_the_regime_multiplier_can_only_reduce_size() -> None:
    """A STRESS session at 0.5x must never become 2x through a mistyped config."""
    kw = dict(
        width=Decimal(1), net_credit=Decimal("0.20"), risk_budget=Decimal(2_000),
        delta_per_contract=Decimal(0), remaining_delta_budget=Decimal(5_000),
    )
    full = size_position(**kw)                                  # type: ignore[arg-type]
    half = size_position(**kw, size_multiplier=Decimal("0.5"))  # type: ignore[arg-type]
    doubled = size_position(**kw, size_multiplier=Decimal(2))   # type: ignore[arg-type]
    assert half == full // 2
    assert doubled == full, "a multiplier above 1 must be clamped"


def test_no_delta_headroom_means_no_trade() -> None:
    assert size_position(
        width=Decimal(1), net_credit=Decimal("0.20"), risk_budget=Decimal(2_000),
        delta_per_contract=Decimal(3_829), remaining_delta_budget=Decimal(0),
    ) == 0


def test_conservative_credit_is_never_above_the_mid(put_credit_spread) -> None:
    """The asymmetry the builders rely on: judge pessimistically, open optimistically.

    `net_credit` (sell the bid, buy the ask) must always be <= `limit_price` (mid),
    or the ladder would open *below* the price the kernel approved.
    """
    short, long_ = put_credit_spread.legs
    assert (short.bid - long_.ask) <= (short.mid - long_.mid)


def test_a_debit_structure_risks_the_premium_paid_not_width_plus_premium() -> None:
    """`width − net_credit` with a negative credit gives `width + debit`, which
    overstates the loss and silently undersizes the convexity sleeve.

    A $1-wide spread bought for $0.40 risks $40 a contract, so a $440 budget buys
    11. Under the credit formula it would read $140 a contract and buy 3.
    """
    n = max_contracts_for_budget(
        width=Decimal(1), net_credit=Decimal("-0.40"), budget=Decimal(440)
    )
    assert n == 11


def test_debit_sizing_agrees_with_the_proposal_max_loss(debit_spread) -> None:
    """The sizing helper and `TradeProposal.max_loss_per_contract` must compute
    the same rule, or Gate 2 rejects what the sizer just approved."""
    budget = debit_spread.max_loss_per_contract * 7
    n = max_contracts_for_budget(
        width=debit_spread.width, net_credit=debit_spread.net_credit, budget=budget
    )
    assert n == 7
