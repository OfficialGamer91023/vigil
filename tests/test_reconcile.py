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

from vigil.domain import PortfolioState, Structure
from vigil.execution.reconcile import (
    BrokerPosition,
    RestingOrder,
    group_positions,
    refresh_deltas,
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


# --------------------------------------------------------------------------- #
# refresh_deltas — folding live chain deltas onto a reconciled book (D-1).
# `group_positions` cannot compute delta (no Greek on a broker position, no chain
# in scope), so it leaves a placeholder 0 and this fills it in once the sense step
# has a chain. Gate 7 reads the result, so getting the sign and the sum right is
# what makes the portfolio delta gate mean anything on a book of more than one.
# --------------------------------------------------------------------------- #

def test_refresh_writes_the_dollar_delta_the_proposal_formula_would() -> None:
    """Same ruler as `TradeProposal.dollar_delta`: Σ δ×100×spot×signed_qty × contracts.

    A short put's negative delta becomes a *positive* portfolio delta (selling a
    put is a bullish lean), so the two legs do not simply add.
    """
    (s,) = group_positions([SHORT_PUT, LONG_PUT])  # 4 contracts, $1-wide put spread
    refreshed, unpriced = refresh_deltas(
        (s,),
        {SHORT_PUT.symbol: -0.16, LONG_PUT.symbol: -0.12},
        {"SPY": Decimal(765)},
    )
    assert unpriced == ()
    # short: -0.16 × 100 × 765 × (−1) = +12240 ; long: -0.12 × 100 × 765 × (+1) = −9180
    # (12240 − 9180) × 4 contracts = 12240
    assert refreshed[0].dollar_delta == Decimal(12240)


def test_equal_deltas_on_the_two_legs_net_to_a_flat_structure() -> None:
    """The sign check, stated on its own: a short and a long leg of equal delta
    cancel, which is what "sell vol, not direction" looks like in this unit."""
    (s,) = group_positions([SHORT_PUT, LONG_PUT])
    refreshed, _ = refresh_deltas(
        (s,), {SHORT_PUT.symbol: -0.15, LONG_PUT.symbol: -0.15}, {"SPY": Decimal(765)}
    )
    assert refreshed[0].dollar_delta == Decimal(0)


def test_a_leg_off_the_sensed_chain_is_flagged_not_half_priced() -> None:
    """A partial refresh is worse than a gap: Gate 7 would count a half-priced
    structure as whole and understate the book. The structure keeps its placeholder
    and is returned as unpriced for the caller to surface."""
    (s,) = group_positions([SHORT_PUT, LONG_PUT])
    refreshed, unpriced = refresh_deltas(
        (s,), {SHORT_PUT.symbol: -0.16}, {"SPY": Decimal(765)}  # long leg missing
    )
    assert unpriced == (s,)
    assert refreshed[0].dollar_delta == Decimal(0)


def test_a_structure_on_an_unsensed_underlying_is_flagged() -> None:
    (s,) = group_positions([SHORT_PUT, LONG_PUT])
    refreshed, unpriced = refresh_deltas(
        (s,), {SHORT_PUT.symbol: -0.16, LONG_PUT.symbol: -0.12}, {}  # no SPY spot
    )
    assert unpriced == (s,)
    assert refreshed[0].dollar_delta == Decimal(0)


def test_the_portfolio_delta_sums_the_whole_book_not_one_trade() -> None:
    """The D-1 defect itself: `net_dollar_delta` summed placeholder zeros, so Gate 7
    silently evaluated a single trade. With deltas refreshed it sees every open
    structure — which is exactly when the gate is supposed to start refusing."""
    (spy,) = group_positions([SHORT_PUT, LONG_PUT])
    qqq_short = BrokerPosition("QQQ260827P00500000", Decimal(-4), Decimal("0.50"))
    qqq_long = BrokerPosition("QQQ260827P00499000", Decimal(4), Decimal("0.30"))
    (qqq,) = group_positions([qqq_short, qqq_long])

    refreshed, unpriced = refresh_deltas(
        (spy, qqq),
        {
            SHORT_PUT.symbol: -0.16, LONG_PUT.symbol: -0.12,
            qqq_short.symbol: -0.16, qqq_long.symbol: -0.12,
        },
        {"SPY": Decimal(765), "QQQ": Decimal(500)},
    )
    assert unpriced == ()

    before = PortfolioState(
        equity=Decimal(100_000), peak_equity=Decimal(100_000),
        day_pnl=Decimal(0), open_structures=(spy, qqq),  # the placeholder book
    )
    after = PortfolioState(
        equity=Decimal(100_000), peak_equity=Decimal(100_000),
        day_pnl=Decimal(0), open_structures=refreshed,
    )
    assert before.net_dollar_delta == Decimal(0)          # the bug: always zero
    assert after.net_dollar_delta != Decimal(0)           # now the book is visible
    assert after.net_dollar_delta == sum(
        (r.dollar_delta for r in refreshed), Decimal(0)
    )


def test_refresh_of_an_empty_book_is_empty() -> None:
    assert refresh_deltas((), {}, {}) == ((), ())
