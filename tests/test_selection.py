"""Strike selection (§4.5: short delta 0.15-0.20)."""

from __future__ import annotations

from decimal import Decimal

from tests.conftest import MakeContract
from vigil.strategy.selection import pick_by_delta


def test_picks_the_strike_nearest_the_target_delta(make_contract: MakeContract) -> None:
    puts = [
        make_contract("SPY260828P00625000", delta=-0.08),
        make_contract("SPY260828P00630000", delta=-0.17),   # closest to 0.16
        make_contract("SPY260828P00635000", delta=-0.31),
    ]
    assert pick_by_delta(puts).occ.strike == Decimal("630")


def test_absolute_value_means_calls_work_too(make_contract: MakeContract) -> None:
    """Puts carry negative delta, calls positive. One comparison serves both."""
    calls = [
        make_contract("SPY260828C00645000", delta=0.15),
        make_contract("SPY260828C00650000", delta=0.40),
    ]
    assert pick_by_delta(calls).occ.strike == Decimal("645")


def test_contracts_without_a_delta_are_skipped_not_guessed(make_contract: MakeContract) -> None:
    """A missing delta must never be inferred — that smuggles a guess past the kernel."""
    puts = [
        make_contract("SPY260828P00625000"),                 # no greeks at all
        make_contract("SPY260828P00630000", delta=-0.30),
    ]
    assert pick_by_delta(puts).occ.strike == Decimal("630")


def test_returns_none_when_nothing_has_a_delta(make_contract: MakeContract) -> None:
    assert pick_by_delta([make_contract("SPY260828P00625000")]) is None
    assert pick_by_delta([]) is None
