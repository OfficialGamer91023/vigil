"""The price ladder (§2.5) and profit-target pricing (§2.6)."""

from __future__ import annotations

from decimal import Decimal

from vigil.execution.pricing import (
    TICK,
    credit_ladder,
    debit_ladder,
    debit_profit_target_price,
    natural_debit_ceiling,
    premium_multiple_target_price,
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


def test_the_debit_ceiling_is_the_natural_price() -> None:
    """§2.5 concedes *toward the natural*, and for a debit package the natural is
    what the builder already priced: buy the ask, sell the bid."""
    assert natural_debit_ceiling(Decimal("-0.54")) == Decimal("0.54")
    assert natural_debit_ceiling(Decimal("0.54")) == Decimal("0.54")


def test_the_natural_ceiling_is_tighter_than_the_old_gate_9_bound() -> None:
    """The ratio bound `D ≤ rW/(1+r)` was legal but far too loose: on a $1-wide
    spread it permitted paying $0.85 for a package whose natural price was $0.54
    — conceding 57% past the ask to chase a fill nobody was refusing."""
    width, ratio = Decimal(1), Decimal("5.5")
    old_bound = ratio * width / (Decimal(1) + ratio)
    natural = natural_debit_ceiling(Decimal("-0.54"))
    assert natural < old_bound
    assert old_bound > Decimal("0.84")


def test_the_ladder_never_offers_past_the_natural() -> None:
    lad = debit_ladder(target_debit=Decimal("0.50"),
                       max_debit=natural_debit_ceiling(Decimal("-0.54")))
    assert all(r <= Decimal("0.54") for r in lad.rungs)
    assert lad.rungs[0] == Decimal("0.50")


def test_an_unbounded_profit_structure_exits_on_a_premium_multiple() -> None:
    """"50% of max profit" has no meaning when max profit is infinite. A $1.68
    strangle at 2.0x rests its sell at $3.36 — a 100% gain on the premium."""
    assert premium_multiple_target_price(Decimal("-1.68"), Decimal(2)) == Decimal("3.36")


def test_the_strangle_target_is_not_the_debit_spread_formula_with_zero_width() -> None:
    """The accident this guards: with `width = 0` the spread formula returns
    exactly the premium — an order to sell the position back for what it cost.

    The clamp inside `debit_profit_target_price` stops that being an outright
    loss, but break-even is not a profit target; it is an exit that guarantees
    the sleeve can never pay.
    """
    premium, half = Decimal("1.68"), Decimal("0.50")
    wrong = debit_profit_target_price(premium, Decimal(0), half)
    right = premium_multiple_target_price(premium, Decimal(2))
    assert wrong == premium, "the spread formula degenerates to break-even"
    assert right > premium


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
