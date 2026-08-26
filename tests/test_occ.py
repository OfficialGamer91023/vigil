"""OCC symbol parsing — the format every other module depends on reading correctly."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from vigil.data.occ import parse_occ


def test_parses_a_standard_put() -> None:
    occ = parse_occ("SPY260904P00640000")
    assert occ.underlying == "SPY"
    assert occ.expiry == date(2026, 9, 4)
    assert occ.is_put is True
    assert occ.right == "P"
    # Decimal, not float: 640.0 == Decimal("640") is True, but the *type* matters
    # downstream where money must never touch binary floating point.
    assert occ.strike == Decimal("640")
    assert isinstance(occ.strike, Decimal)


def test_parses_a_call_with_a_fractional_strike() -> None:
    occ = parse_occ("QQQ260828C00512500")
    assert occ.underlying == "QQQ"
    assert occ.is_put is False
    assert occ.strike == Decimal("512.5")


def test_root_length_is_not_assumed() -> None:
    """The classic bug: assuming a 3-character root. Parse from the right instead."""
    assert parse_occ("F260828C00012000").underlying == "F"
    assert parse_occ("GOOGL260828C00180000").underlying == "GOOGL"


def test_dte_counts_calendar_days() -> None:
    occ = parse_occ("SPY260828P00630000")
    assert occ.dte(date(2026, 8, 28)) == 0     # 0DTE
    assert occ.dte(date(2026, 8, 26)) == 2


@pytest.mark.parametrize("bad", ["SPY", "", "SPY260828X00630000"])
def test_rejects_malformed_symbols(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_occ(bad)
