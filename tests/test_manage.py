"""The management sweep (§4.6). Pure decisions, no broker, no LLM.

These tests encode the exit policy §4.4.1 argues for, including the exit it
argues *against*: there is no mark-based stop here, and a test asserting one
exists would be a regression, not a gap.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from vigil.domain import OpenStructure, PositionLeg, Structure
from vigil.execution.manage import (
    Action,
    decide,
    is_flatten_time,
    minutes_to_close,
    resting_target_price,
    sweep,
)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
TODAY = date(2026, 8, 26)
TOMORROW = date(2026, 8, 27)


def put_spread(
    *, expiry: date = TOMORROW, short: str = "761", target: bool = True
) -> OpenStructure:
    """A put credit spread short the 761 strike, with its resting exit live."""
    return OpenStructure(
        underlying="SPY",
        expiry=expiry,
        strikes=(Decimal(short), Decimal("760")),
        max_loss=Decimal(80),
        dollar_delta=Decimal(-3800),
        has_resting_target=target,
        structure=Structure.PUT_CREDIT_SPREAD,
        short_put_strikes=(Decimal(short),),
        net_credit=Decimal("0.20"),
        contracts=1,
    )


def condor(*, expiry: date = TOMORROW) -> OpenStructure:
    return OpenStructure(
        underlying="SPY", expiry=expiry,
        strikes=(Decimal("760"), Decimal("761"), Decimal("770"), Decimal("771")),
        max_loss=Decimal(640), dollar_delta=Decimal(0), has_resting_target=True,
        structure=Structure.IRON_CONDOR,
        short_put_strikes=(Decimal("761"),), short_call_strikes=(Decimal("770"),),
        net_credit=Decimal("0.20"), contracts=8,
    )


def strangle(*, expiry: date = TOMORROW) -> OpenStructure:
    """Long-only: nothing short, so nothing to breach."""
    return OpenStructure(
        underlying="SPY", expiry=expiry,
        strikes=(Decimal("761"), Decimal("769")),
        max_loss=Decimal(336), dollar_delta=Decimal(0), has_resting_target=True,
        structure=Structure.LONG_STRANGLE, net_credit=Decimal("-1.68"), contracts=2,
    )


def at(hh: int, mm: int, *, day: date = TODAY, tz: ZoneInfo = ET) -> datetime:
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=ET).astimezone(tz)


# --------------------------------------------------------------------------- #
# Breach — thesis invalidation, not a P&L stop
# --------------------------------------------------------------------------- #

def test_a_breached_short_strike_closes_when_there_is_time_to_act() -> None:
    d = decide(put_spread(), spot=Decimal("760.40"), now=at(11, 0))
    assert d.action is Action.CLOSE_BREACH
    assert "breached" in d.reason


def test_an_unbreached_structure_is_held() -> None:
    d = decide(put_spread(), spot=Decimal("765.00"), now=at(11, 0))
    assert d.action is Action.HOLD


def test_the_short_strike_itself_counts_as_breached() -> None:
    """Touching the strike is the event, not passing through it — the whole
    argument for using breach rather than a mark is that it is path-based."""
    assert decide(put_spread(), spot=Decimal("761"), now=at(11, 0)).action is Action.CLOSE_BREACH


def test_a_breach_inside_the_last_half_hour_is_held_for_the_time_stop() -> None:
    """Closing a breached short-dated structure at peak gamma means crossing a
    spread into a book the 15:40 flatten is about to close anyway."""
    d = decide(put_spread(), spot=Decimal("760.40"), now=at(15, 35))
    assert d.action is Action.HOLD
    assert "time stop" in d.reason


def test_a_condor_breaches_on_either_side() -> None:
    assert decide(condor(), spot=Decimal("760.50"), now=at(11, 0)).action is Action.CLOSE_BREACH
    assert decide(condor(), spot=Decimal("770.50"), now=at(11, 0)).action is Action.CLOSE_BREACH
    assert decide(condor(), spot=Decimal("765.00"), now=at(11, 0)).action is Action.HOLD


def test_a_long_only_structure_can_never_be_breached() -> None:
    """A strangle has no short leg. Testing spot against its strikes would close
    the position precisely when it started working."""
    for spot in ("755.00", "765.00", "775.00"):
        assert decide(strangle(), spot=Decimal(spot), now=at(11, 0)).action is Action.HOLD


# --------------------------------------------------------------------------- #
# Time stop — auto-exercise is the reason it outranks everything
# --------------------------------------------------------------------------- #

def test_anything_expiring_today_closes_after_the_flatten_time() -> None:
    d = decide(put_spread(expiry=TODAY), spot=Decimal("765.00"), now=at(15, 41))
    assert d.action is Action.CLOSE_TIME_STOP
    assert "auto-exercise" in d.reason


def test_a_later_expiry_is_untouched_by_the_flatten() -> None:
    d = decide(put_spread(expiry=TOMORROW), spot=Decimal("765.00"), now=at(15, 41))
    assert d.action is not Action.CLOSE_TIME_STOP


def test_the_time_stop_outranks_breach() -> None:
    """A breached 0DTE structure at 15:41 is closed because it expires today.
    Nothing may outrank auto-exercise, and the journal should say which rule fired."""
    d = decide(put_spread(expiry=TODAY), spot=Decimal("760.00"), now=at(15, 41))
    assert d.action is Action.CLOSE_TIME_STOP


def test_before_the_flatten_a_zero_dte_structure_is_managed_normally() -> None:
    d = decide(put_spread(expiry=TODAY), spot=Decimal("765.00"), now=at(11, 0))
    assert d.action is Action.HOLD


# --------------------------------------------------------------------------- #
# The missing resting target is a defect, not a preference
# --------------------------------------------------------------------------- #

def test_a_structure_without_a_resting_target_gets_one() -> None:
    d = decide(put_spread(target=False), spot=Decimal("765.00"), now=at(11, 0))
    assert d.action is Action.REPLACE_TARGET
    assert "§2.6" in d.reason


def test_a_structure_being_closed_is_not_first_given_an_exit_order() -> None:
    """Ordering: replacing the target on something we are about to close would
    submit an order that can never be used."""
    d = decide(put_spread(target=False), spot=Decimal("760.00"), now=at(11, 0))
    assert d.action is Action.CLOSE_BREACH


# --------------------------------------------------------------------------- #
# Timezone discipline — the same bug as B5, in a different module
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tz", [ET, UTC, ZoneInfo("Asia/Karachi")])
def test_the_sweep_reasons_in_eastern_whatever_zone_it_is_handed(tz) -> None:
    """15:41 ET is past the flatten. The same instant in UTC is 19:41, and read
    as a bare wall clock it would look like the middle of the night."""
    d = decide(put_spread(expiry=TODAY), spot=Decimal("765.00"), now=at(15, 41, tz=tz))
    assert d.action is Action.CLOSE_TIME_STOP


def test_minutes_to_close_is_measured_in_eastern() -> None:
    assert minutes_to_close(at(15, 30)) == 30
    assert minutes_to_close(at(15, 30, tz=UTC)) == 30


def test_is_flatten_time_agrees_across_zones() -> None:
    assert is_flatten_time(at(15, 41)) is True
    assert is_flatten_time(at(15, 41, tz=UTC)) is True
    assert is_flatten_time(at(15, 39)) is False


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #

def test_the_sweep_puts_closes_first() -> None:
    """If the process dies half-way through, the closes are what we most want to
    have already happened."""
    book = (put_spread(target=False), put_spread(short="770"), condor())
    out = sweep(book, spots={"SPY": Decimal("769.00")}, now=at(11, 0))
    actions = [d.action for d in out]
    assert actions[0] is Action.CLOSE_BREACH
    assert all(not d.closes for d in out[1:])


def test_a_missing_spot_holds_rather_than_closing() -> None:
    """A feed gap must not become realized losses."""
    out = sweep((put_spread(),), spots={}, now=at(11, 0))
    assert out[0].action is Action.HOLD
    assert "no spot" in out[0].reason


def test_the_sweep_covers_every_structure_exactly_once() -> None:
    book = (put_spread(), condor(), strangle())
    out = sweep(book, spots={"SPY": Decimal("765.00")}, now=at(11, 0))
    assert len(out) == len(book)
    assert {id(d.structure) for d in out} == {id(s) for s in book}


def test_there_is_no_mark_based_stop() -> None:
    """§4.4.1 deletes it: break-even needs an 80% win rate while touch probability
    is ~32%, and max loss is already bounded on entry by Gate 2.

    A deeply losing but unbreached structure is **held**. If this test ever fails
    because someone added a mark stop, read §4.4.1 before 'fixing' it.
    """
    losing = put_spread(short="750")          # short strike far from a 765 spot
    assert decide(losing, spot=Decimal("765.00"), now=at(11, 0)).action is Action.HOLD


# --------------------------------------------------------------------------- #
# resting_target_price — the §2.6 re-rest, priced the same way entry prices it
# --------------------------------------------------------------------------- #

def test_a_credit_spread_re_rests_at_the_50pct_target() -> None:
    """Buy a $0.20 credit spread back for $0.10 — 50% of max profit."""
    assert resting_target_price(put_spread()) == Decimal("0.10")


def test_a_condor_re_rests_off_its_two_credits() -> None:
    """A credit structure of any leg count uses the same buy-back rule."""
    assert resting_target_price(condor()) == Decimal("0.10")


def test_a_long_strangle_re_rests_at_a_premium_multiple() -> None:
    """Unbounded max profit → exit at a multiple of the premium, not a fraction of
    it. A $1.68 strangle at 2.0x rests a sell order at $3.36."""
    assert resting_target_price(strangle()) == Decimal("3.36")


def test_a_debit_spread_re_rests_above_what_it_paid() -> None:
    """A debit structure sells back *above* cost: $0.40 paid on a $1 width, 50% of
    the $0.60 max profit → 0.40 + 0.30 = $0.70."""
    debit = OpenStructure(
        underlying="SPY", expiry=TOMORROW,
        strikes=(Decimal("760"), Decimal("761")),
        max_loss=Decimal(40), dollar_delta=Decimal(0), has_resting_target=False,
        structure=Structure.DEBIT_SPREAD, net_credit=Decimal("-0.40"), contracts=1,
        legs=(
            PositionLeg(symbol="SPY260827C00760000", ratio_qty=1, is_short=False),
            PositionLeg(symbol="SPY260827C00761000", ratio_qty=1, is_short=True),
        ),
    )
    assert resting_target_price(debit) == Decimal("0.70")


def test_an_unknown_opening_credit_yields_no_target() -> None:
    """An adopted position we never priced has net_credit == 0. No honest target
    can be derived, so the caller keeps the §2.6 alarm rather than guessing."""
    from dataclasses import replace

    assert resting_target_price(replace(put_spread(), net_credit=Decimal(0))) is None
