"""Pricing a close: `package_mid` and `closing_limit`.

The asymmetry these encode is the point — entries concede nothing and closes
concede 10% — so the tests assert the *direction* of the concession, not just
that a number comes back.
"""

from __future__ import annotations

from decimal import Decimal

from vigil.execution.pricing import TICK, closing_limit, package_mid


def test_package_mid_signs_a_credit_structure_positive():
    """A put credit spread: short leg worth more than the long. Positive = credit."""
    value = package_mid([
        (Decimal("0.50"), Decimal("0.52"), True),    # short: +0.51
        (Decimal("0.30"), Decimal("0.32"), False),   # long:  -0.31
    ])
    assert value == Decimal("0.20")


def test_package_mid_signs_a_long_only_structure_negative():
    """A long strangle collects nothing, so its package value is negative."""
    value = package_mid([
        (Decimal("0.80"), Decimal("0.84"), False),
        (Decimal("0.80"), Decimal("0.84"), False),
    ])
    assert value == Decimal("-1.64")


def test_package_mid_matches_the_net_credit_convention(put_credit_spread):
    """The signed mid agrees with `TradeProposal.limit_price` on the same legs.

    This is the invariant that lets a structure's current value be compared
    against the credit it was opened for. If the two modules disagreed about the
    sign, every close would be priced as though the position were the opposite
    trade.
    """
    value = package_mid([
        (leg.bid, leg.ask, leg.is_short) for leg in put_credit_spread.legs
    ])
    assert value == put_credit_spread.limit_price


def test_closing_a_credit_structure_bids_above_the_mid():
    """We are buying the package back, so conceding means paying *more*."""
    assert closing_limit(Decimal("0.20")) == Decimal("0.22")


def test_closing_a_debit_structure_accepts_below_the_mid():
    """We are selling the package back, so conceding means accepting *less*."""
    assert closing_limit(Decimal("-1.64")) == Decimal("1.47")


def test_closing_limit_is_always_a_positive_magnitude():
    """Direction lives in `position_intent`, never in the sign of the price."""
    assert closing_limit(Decimal("-1.64")) > 0
    assert closing_limit(Decimal("0.20")) > 0


def test_closing_limit_never_returns_a_zero_order():
    """A $0.00 limit is not an order. A near-worthless package still gets a tick."""
    assert closing_limit(Decimal("0.001")) == TICK
    assert closing_limit(Decimal("0")) == TICK


def test_slippage_is_configurable_and_directional():
    aggressive = closing_limit(Decimal("1.00"), slippage_pct=Decimal("0.25"))
    gentle = closing_limit(Decimal("1.00"), slippage_pct=Decimal("0.05"))
    # Buying back: more aggressive means willing to pay more.
    assert aggressive > gentle
