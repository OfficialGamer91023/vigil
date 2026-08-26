"""Reconciliation: rebuilding structures from broker truth (§2.3).

The scenario these tests really encode is a **worker restart**. §2.6 leaves a
resting GTC exit at the broker so a position survives the process dying — but
that is only half a guarantee unless the worker, on restart, can find its way
back to a position it has no memory of. Every test below is "the process just
came up and all it has is the broker".
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from vigil.domain import Structure
from vigil.execution.reconcile import (
    BrokerPosition,
    RestingOrder,
    group_positions,
    structures_missing_targets,
)

SHORT_PUT = BrokerPosition("SPY260827P00761000", Decimal(-4), Decimal("0.50"))
LONG_PUT = BrokerPosition("SPY260827P00760000", Decimal(4), Decimal("0.30"))
SHORT_CALL = BrokerPosition("SPY260827C00770000", Decimal(-4), Decimal("0.50"))
LONG_CALL = BrokerPosition("SPY260827C00771000", Decimal(4), Decimal("0.30"))


def test_a_put_spread_is_rebuilt_from_two_positions() -> None:
    (s,) = group_positions([SHORT_PUT, LONG_PUT])
    assert s.underlying == "SPY"
    assert s.expiry == date(2026, 8, 27)
    assert s.structure is Structure.PUT_CREDIT_SPREAD
    assert s.short_put_strikes == (Decimal(761),)
    assert s.contracts == 4


def test_the_rebuilt_structure_can_be_closed() -> None:
    """The point of the whole module: legs carry enough to build a closing
    ticket without the TradeProposal that opened them."""
    (s,) = group_positions([SHORT_PUT, LONG_PUT])
    assert {leg.symbol for leg in s.legs} == {SHORT_PUT.symbol, LONG_PUT.symbol}
    assert [leg.is_short for leg in sorted(s.legs, key=lambda x: x.symbol)] == [False, True]


def test_max_loss_uses_the_same_width_rule_as_gate_1() -> None:
    """Width from the strikes, per right — so a reconciled structure and a
    proposed one report the same exposure and Gate 6 cannot disagree with itself
    depending on which path produced the number."""
    (s,) = group_positions([SHORT_PUT, LONG_PUT])
    # $1 wide, $0.20 credit, 4 contracts -> (1.00 - 0.20) * 100 * 4
    assert s.net_credit == Decimal("0.20")
    assert s.max_loss == Decimal(320)


def test_a_condor_is_one_structure_not_two_spreads() -> None:
    """Treating it as two would roughly double the book's apparent risk and
    trip Gate 5 and Gate 6 on a position that is within both."""
    structures = group_positions([SHORT_PUT, LONG_PUT, SHORT_CALL, LONG_CALL])
    assert len(structures) == 1
    s = structures[0]
    assert s.structure is Structure.IRON_CONDOR
    assert s.short_put_strikes and s.short_call_strikes
    # One width of risk, not two: (1.00 - 0.40) * 100 * 4
    assert s.max_loss == Decimal(240)


def test_a_long_strangle_is_recognised_and_carries_no_short_strikes() -> None:
    long_put = BrokerPosition("SPY260827P00761000", Decimal(2), Decimal("0.84"))
    long_call = BrokerPosition("SPY260827C00769000", Decimal(2), Decimal("0.84"))
    (s,) = group_positions([long_put, long_call])
    assert s.structure is Structure.LONG_STRANGLE
    assert not s.has_short_legs
    assert s.net_credit < 0
    # Max loss is the premium paid: 1.68 * 100 * 2
    assert s.max_loss == Decimal(336)


def test_different_expiries_are_different_structures() -> None:
    later = BrokerPosition("SPY260828P00761000", Decimal(-4), Decimal("0.50"))
    later_long = BrokerPosition("SPY260828P00760000", Decimal(4), Decimal("0.30"))
    assert len(group_positions([SHORT_PUT, LONG_PUT, later, later_long])) == 2


def test_different_underlyings_are_different_structures() -> None:
    qqq_short = BrokerPosition("QQQ260827P00500000", Decimal(-4), Decimal("0.50"))
    qqq_long = BrokerPosition("QQQ260827P00499000", Decimal(4), Decimal("0.30"))
    out = group_positions([SHORT_PUT, LONG_PUT, qqq_short, qqq_long])
    assert {s.underlying for s in out} == {"SPY", "QQQ"}


def test_a_non_option_position_is_skipped_not_folded_in() -> None:
    """An equity leg folded into an option structure would misstate its risk
    entirely, and this agent trades only defined-risk option structures."""
    out = group_positions([SHORT_PUT, LONG_PUT, BrokerPosition("SPY", Decimal(100), Decimal(765))])
    assert len(out) == 1
    assert all(len(s.legs) == 2 for s in out)


def test_a_zero_quantity_position_is_ignored() -> None:
    closed = BrokerPosition("SPY260827P00755000", Decimal(0), Decimal("0.10"))
    (s,) = group_positions([SHORT_PUT, LONG_PUT, closed])
    assert len(s.legs) == 2


# --------------------------------------------------------------------------- #
# The §2.6 defect: an open structure with no resting exit
# --------------------------------------------------------------------------- #

def test_a_matching_closing_order_marks_the_target_present() -> None:
    resting = RestingOrder("o1", frozenset({SHORT_PUT.symbol, LONG_PUT.symbol}), True)
    (s,) = group_positions([SHORT_PUT, LONG_PUT], resting=[resting])
    assert s.has_resting_target
    assert structures_missing_targets((s,)) == ()


def test_no_resting_order_is_reported_as_a_defect() -> None:
    (s,) = group_positions([SHORT_PUT, LONG_PUT], resting=[])
    assert not s.has_resting_target
    assert structures_missing_targets((s,)) == (s,)


def test_an_opening_order_does_not_count_as_an_exit() -> None:
    """A working entry ticket covers the same symbols. Counting it would report
    an exit that does not exist — the precise defect §2.6 names."""
    entry = RestingOrder("o2", frozenset({SHORT_PUT.symbol, LONG_PUT.symbol}), False)
    (s,) = group_positions([SHORT_PUT, LONG_PUT], resting=[entry])
    assert not s.has_resting_target


def test_a_partial_closing_order_does_not_count() -> None:
    """An order covering only one leg cannot close the structure, so reporting
    it as the exit would hide a half-protected position."""
    partial = RestingOrder("o3", frozenset({SHORT_PUT.symbol}), True)
    (s,) = group_positions([SHORT_PUT, LONG_PUT], resting=[partial])
    assert not s.has_resting_target


def test_an_empty_account_reconciles_to_nothing() -> None:
    assert group_positions([]) == ()
