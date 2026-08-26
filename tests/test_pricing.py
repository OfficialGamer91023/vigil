"""The price ladder (§2.5) and profit-target pricing (§2.6)."""

from __future__ import annotations

from decimal import Decimal

from vigil.execution.pricing import (
    TICK,
    credit_ladder,
    debit_ladder,
    debit_profit_target_price,
    max_debit_for_ratio,
    profit_target_price,
    round_to_tick,
)


def test_ladder_descends_from_target_to_floor() -> None:
    lad = credit_ladder(target_credit=Decimal("0.24"), min_credit=Decimal("0.18"))
    assert lad.rungs[0] == Decimal("0.24")
    assert lad.rungs[-1] == Decimal("0.18")
    # Monotonically conceding: we only ever ask for less, never more.
    assert list(lad.rungs) == sorted(lad.rungs, reverse=True)


def test_ladder_never_crosses_the_floor() -> None:
    """A missed trade is free; a bad fill is not. The floor is Gate 9's threshold."""
    lad = credit_ladder(target_credit=Decimal("0.30"), min_credit=Decimal("0.18"))
    assert all(r >= Decimal("0.18") for r in lad.rungs)


def test_ladder_collapses_when_target_is_already_at_the_floor() -> None:
    lad = credit_ladder(target_credit=Decimal("0.18"), min_credit=Decimal("0.18"))
    assert lad.rungs == (Decimal("0.18"),)


def test_ladder_deduplicates_rungs_that_land_on_the_same_tick() -> None:
    """A narrow span would otherwise burn 20 seconds resubmitting the same price."""
    lad = credit_ladder(target_credit=Decimal("0.19"), min_credit=Decimal("0.18"))
    assert len(set(lad.rungs)) == len(lad.rungs)


def test_rounding_never_shaves_the_credit_in_our_disfavour() -> None:
    """Rounding a credit down by a cent per rung is exactly what Gate 9 argues about."""
    assert round_to_tick(Decimal("0.1801")) == Decimal("0.19")
    assert round_to_tick(Decimal("0.1801"), favour_us=False) == Decimal("0.18")


def test_profit_target_is_the_debit_to_buy_it_back() -> None:
    """50% of max profit on a $0.20 credit means closing for $0.10."""
    assert profit_target_price(Decimal("0.20"), Decimal("0.50")) == Decimal("0.10")
    assert profit_target_price(Decimal("0.30"), Decimal("0.50")) == Decimal("0.15")


def test_profit_target_never_becomes_a_zero_limit() -> None:
    """A $0.00 limit is not an order; an order that cannot fill is decoration."""
    assert profit_target_price(Decimal("0.01"), Decimal("0.99")) == TICK


# --------------------------------------------------------------------------- #
# The debit ladder — the mirror, and the mirror is not cosmetic
# --------------------------------------------------------------------------- #

def test_the_debit_ladder_concedes_upward() -> None:
    """On a debit we are *paying*, so conceding means offering more. A credit
    ladder here would bid progressively less for something we want to own, which
    simply never fills."""
    lad = debit_ladder(target_debit=Decimal("0.40"), max_debit=Decimal("0.52"))
    assert list(lad.rungs) == sorted(lad.rungs)
    assert lad.rungs[0] == Decimal("0.40")


def test_the_debit_ladder_never_offers_above_its_ceiling() -> None:
    lad = debit_ladder(target_debit=Decimal("0.40"), max_debit=Decimal("0.52"))
    assert all(r <= Decimal("0.52") for r in lad.rungs)


def test_a_debit_already_at_the_ceiling_gets_a_single_rung() -> None:
    lad = debit_ladder(target_debit=Decimal("0.60"), max_debit=Decimal("0.52"))
    assert lad.rungs == (Decimal("0.52"),)


def test_the_debit_ceiling_comes_from_gate_9_rather_than_a_new_knob() -> None:
    """`D / (W − D) ≤ r` rearranges to `D ≤ rW/(1+r)`. Deriving the ceiling from
    the gate means the ladder can never concede to a price the kernel would then
    reject — the same discipline the credit floor already follows."""
    width, ratio = Decimal(1), Decimal("5.5")
    ceiling = max_debit_for_ratio(width, ratio)
    assert ceiling == ratio * width / (Decimal(1) + ratio)
    # At the ceiling the ratio is the limit, not past it. Compared at a
    # tolerance because Decimal division carries 28 significant digits and the
    # round-trip lands a few ulps short of exactly 5.5.
    assert abs(ceiling / (width - ceiling) - ratio) < Decimal("1e-20")


def test_the_debit_profit_target_sits_above_what_was_paid() -> None:
    """Max profit on a debit spread is `W − D`, so half of it means selling back
    for more than it cost. The credit formula subtracts instead of adding, and
    would rest an exit at a guaranteed loss."""
    entry = Decimal("0.40")
    target = debit_profit_target_price(entry, width=Decimal(1), target_pct=Decimal("0.50"))
    assert target > entry
    # 0.40 paid + 50% of the 0.60 remaining = 0.70.
    assert target == Decimal("0.70")


def test_the_credit_and_debit_targets_move_in_opposite_directions() -> None:
    """The single most confusable pair of formulas in the execution layer."""
    entry = Decimal("0.40")
    half = Decimal("0.50")
    assert profit_target_price(entry, half) < entry
    assert debit_profit_target_price(entry, Decimal(1), half) > entry
