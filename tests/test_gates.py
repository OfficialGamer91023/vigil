"""The twelve gates. The tests that matter most (CLAUDE.md).

Each test breaks exactly one thing against a baseline proposal that passes
everything, so a failure names its own cause.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from tests.conftest import DEFAULT_EXPIRY, DEFAULT_NOW, make_leg
from vigil.config import risk_config
from vigil.domain import (
    GateVerdict,
    KernelDecision,
    OpenStructure,
    PortfolioState,
    Structure,
    TradeProposal,
)
from vigil.risk.context import KernelContext
from vigil.risk.kernel import evaluate

ET = ZoneInfo("America/New_York")


def failures(p: TradeProposal, s: PortfolioState, x: KernelContext) -> set[int]:
    return {v.number for v in evaluate(p, s, x).failures}


def _verdict(decision: KernelDecision, number: int) -> GateVerdict:
    """One gate's verdict by number.

    Asserting on the specific gate rather than on `approved` is what keeps a test
    honest: a proposal can fail for a reason the test did not intend, and
    `not approved` would happily call that a pass.
    """
    return next(v for v in decision.verdicts if v.number == number)


# --- baseline -------------------------------------------------------------- #

def test_baseline_proposal_passes_every_gate(put_credit_spread, flat_book, ctx) -> None:
    d = evaluate(put_credit_spread, flat_book, ctx)
    assert d.approved, d.summary
    assert len(d.verdicts) == 12
    # Passes are recorded too — §5 requires the full record, not just rejections.
    assert all(v.passed for v in d.verdicts)


def test_halt_flag_outranks_the_gates(put_credit_spread, flat_book, ctx) -> None:
    d = evaluate(put_credit_spread, replace(flat_book, halted=True), ctx)
    assert not d.approved
    assert d.verdicts[0].number == 0 and "HALT" in d.verdicts[0].reason


# --- gate 1: defined risk -------------------------------------------------- #

def test_gate1_rejects_an_uncovered_short_put(put_credit_spread, flat_book, ctx) -> None:
    naked = replace(put_credit_spread, legs=(put_credit_spread.legs[0],))
    assert 1 in failures(naked, flat_book, ctx)


def test_gate1_will_not_net_a_short_call_against_a_long_put(
    flat_book, ctx, put_credit_spread
) -> None:
    """A short call is covered by a long call, never by a long put."""
    bad = replace(put_credit_spread, legs=(
        make_leg("SPY260827C00770000", short=True, delta=0.16),
        make_leg("SPY260827P00760000", short=False, delta=-0.11),
    ))
    assert 1 in failures(bad, flat_book, ctx)


# --- gate 2: per-trade risk ------------------------------------------------ #

def test_gate2_rejects_oversized_position(put_credit_spread, flat_book, ctx) -> None:
    # 8 contracts is 0.64% of equity; 30 contracts is 2.4% — over the 2% budget.
    assert 2 in failures(replace(put_credit_spread, contracts=30), flat_book, ctx)


def test_gate2_allows_exactly_the_budget(put_credit_spread, flat_book, ctx) -> None:
    # max loss per contract = (1.00 - 0.20) * 100 = $80. 25 contracts = $2,000 = 2.0%.
    assert 2 not in failures(replace(put_credit_spread, contracts=25), flat_book, ctx)
    assert 2 in failures(replace(put_credit_spread, contracts=26), flat_book, ctx)


# --- gates 3 & 4: portfolio stops ------------------------------------------ #

def test_gate3_daily_loss_halts_new_entries(put_credit_spread, flat_book, ctx) -> None:
    hurt = replace(flat_book, day_pnl=Decimal(-3_000))       # exactly -3%
    assert 3 in failures(put_credit_spread, hurt, ctx)


def test_gate4_drawdown_from_peak_not_from_start(put_credit_spread, ctx) -> None:
    """Drawdown is measured from the *peak*, so a profitable account can still halt."""
    s = PortfolioState(
        equity=Decimal(110_000), peak_equity=Decimal(120_000), day_pnl=Decimal(0)
    )
    assert s.equity > Decimal(100_000)            # up on the week
    assert 4 in failures(put_credit_spread, s, ctx)   # but -8.3% from peak


# --- gates 5 & 6: concurrency and concentration ---------------------------- #

def _open(underlying: str, strike: int, risk: str = "500") -> OpenStructure:
    return OpenStructure(
        underlying=underlying, expiry=DEFAULT_EXPIRY, strikes=(Decimal(strike),),
        max_loss=Decimal(risk), dollar_delta=Decimal(0),
    )


def test_gate5_caps_open_structures(put_credit_spread, flat_book, ctx) -> None:
    book = replace(flat_book, open_structures=tuple(
        _open("QQQ", 700 + i) for i in range(6)))
    assert 5 in failures(put_credit_spread, book, ctx)


def test_gate6_caps_structures_per_underlying(put_credit_spread, flat_book, ctx) -> None:
    book = replace(flat_book, open_structures=(_open("SPY", 750), _open("SPY", 755)))
    assert 6 in failures(put_credit_spread, book, ctx)


def test_gate6_caps_risk_share_in_one_name(put_credit_spread, flat_book, ctx) -> None:
    """One SPY structure plus this proposal would be >40% of open risk."""
    book = replace(flat_book, open_structures=(
        _open("SPY", 750, risk="600"), _open("QQQ", 700, risk="100")))
    assert 6 in failures(put_credit_spread, book, ctx)


# --- gate 7: dollar delta -------------------------------------------------- #

def test_gate7_is_measured_in_dollars_across_the_whole_book(
    put_credit_spread, flat_book, ctx
) -> None:
    # $5,000 is the limit at $100k. Existing book already sits at $4,900.
    book = replace(flat_book, open_structures=(
        OpenStructure("QQQ", DEFAULT_EXPIRY, (Decimal(700),),
                      Decimal(500), Decimal(4_900)),))
    assert 7 in failures(put_credit_spread, book, ctx)


def test_gate7_a_bare_delta_count_would_have_passed(
    put_credit_spread, flat_book, ctx
) -> None:
    """The §5.1 point: 0.4 contract-delta looks tiny, $30k of drift is not."""
    huge = replace(put_credit_spread, contracts=60)
    assert abs(sum(leg.delta * leg.signed_ratio for leg in huge.legs)) < 0.1   # "small"
    assert abs(huge.dollar_delta) > Decimal(5_000)                             # not small
    assert 7 in failures(huge, flat_book, ctx)


# --- gate 8: liquidity ----------------------------------------------------- #

def test_gate8_rejects_thin_open_interest(put_credit_spread, flat_book, ctx) -> None:
    thin = replace(put_credit_spread, legs=(
        make_leg("SPY260827P00761000", short=True, bid="0.50", ask="0.52", oi=10),
        put_credit_spread.legs[1]))
    assert 8 in failures(thin, flat_book, ctx)


def test_gate8_rejects_a_leg_with_no_bid(put_credit_spread, flat_book, ctx) -> None:
    nobid = replace(put_credit_spread, legs=(
        make_leg("SPY260827P00761000", short=True, bid="0.00", ask="0.52"),
        put_credit_spread.legs[1]))
    assert 8 in failures(nobid, flat_book, ctx)


def test_gate8_allows_a_wide_percentage_on_a_cheap_wing(
    put_credit_spread, flat_book, ctx
) -> None:
    """$0.05/$0.12 is 97% of mid — but it is also the tightest a cheap wing gets."""
    cheap = replace(put_credit_spread, legs=(
        put_credit_spread.legs[0],
        make_leg("SPY260827P00760000", bid="0.05", ask="0.12", delta=-0.03)))
    assert 8 not in failures(cheap, flat_book, ctx)


# --- gate 9: credit quality ------------------------------------------------ #

def test_gate9_rejects_credit_below_the_floor(put_credit_spread, flat_book, ctx) -> None:
    """The A2 measurement: $1-wide verticals priced 11-16.5% against an 18% floor."""
    thin = replace(put_credit_spread, net_credit=Decimal("0.145"), limit_price=Decimal("0.145"))
    assert thin.credit_pct_of_width < risk_config().min_credit_pct_of_width
    assert 9 in failures(thin, flat_book, ctx)


def test_gate9_rejects_a_steamroller_ratio(put_credit_spread, flat_book, ctx) -> None:
    # $0.10 credit on $1 width: 9:1 loss:profit, and below the credit floor too.
    bad = replace(put_credit_spread, net_credit=Decimal("0.10"), limit_price=Decimal("0.10"))
    assert 9 in failures(bad, flat_book, ctx)


# --- gate 10: events ------------------------------------------------------- #

def test_gate10_blocks_earnings_inside_contract_life(put_credit_spread, flat_book, ctx) -> None:
    x = KernelContext(now=ctx.now, earnings_dates={"SPY": (DEFAULT_EXPIRY,)})
    assert 10 in failures(put_credit_spread, flat_book, x)


def test_gate10_ignores_earnings_after_expiry(put_credit_spread, flat_book, ctx) -> None:
    x = KernelContext(now=ctx.now, earnings_dates={"SPY": (date(2026, 9, 30),)})
    assert 10 not in failures(put_credit_spread, flat_book, x)


def test_gate10_blocks_the_five_minutes_around_a_macro_print(
    put_credit_spread, flat_book, ctx
) -> None:
    x = KernelContext(now=ctx.now, macro_events=(ctx.now + timedelta(minutes=3),))
    assert 10 in failures(put_credit_spread, flat_book, x)


# --- gate 11: time windows ------------------------------------------------- #

@pytest.mark.parametrize(
    ("hh", "mm", "blocked"),
    [(9, 31, True),    # first 15 minutes
     (9, 46, False),
     (15, 41, True),   # last 20 minutes
     (15, 39, False),
     (8, 0, True),     # pre-market
     (16, 30, True)],  # after the close
)
def test_gate11_entry_windows(put_credit_spread, flat_book, hh, mm, blocked) -> None:
    x = KernelContext(now=datetime(2026, 8, 26, hh, mm, tzinfo=ET))
    assert (11 in failures(put_credit_spread, flat_book, x)) is blocked


def test_gate11_blocks_late_zero_dte_entry(put_credit_spread, flat_book) -> None:
    """0DTE after 14:30 is banned; the same structure 1 DTE out is fine."""
    at_1500 = datetime(2026, 8, 26, 15, 0, tzinfo=ET)
    zero_dte = replace(put_credit_spread, expiry=at_1500.date())
    assert 11 in failures(zero_dte, flat_book, KernelContext(now=at_1500))
    assert 11 not in failures(put_credit_spread, flat_book, KernelContext(now=at_1500))


# --- gate 12: idempotency and sanity --------------------------------------- #

def test_gate12_rejects_a_replayed_client_order_id(put_credit_spread, flat_book, ctx) -> None:
    """The classic autonomous-agent bug: retrying a call that actually succeeded."""
    book = replace(flat_book, known_client_order_ids=frozenset({"vigil-test-0001"}))
    assert 12 in failures(put_credit_spread, book, ctx)


def test_gate12_rejects_a_duplicate_structure(put_credit_spread, flat_book, ctx) -> None:
    book = replace(flat_book, open_structures=(OpenStructure(
        "SPY", DEFAULT_EXPIRY, (Decimal(760), Decimal(761)), Decimal(500), Decimal(0)),))
    assert 12 in failures(put_credit_spread, book, ctx)


def test_gate12_requires_leg_ratios_with_gcd_one(put_credit_spread, flat_book, ctx) -> None:
    """An Alpaca constraint: 2:2 must be expressed as 1:1 with size in qty."""
    doubled = replace(put_credit_spread, legs=(
        make_leg("SPY260827P00761000", short=True, bid="0.50", ask="0.52", ratio=2),
        make_leg("SPY260827P00760000", bid="0.30", ask="0.32", ratio=2)))
    assert 12 in failures(doubled, flat_book, ctx)


def test_gate12_rejects_a_limit_far_from_mid(
    put_credit_spread, flat_book, ctx
) -> None:
    # Package mid is $0.20; $0.50 is 150% away.
    assert 12 in failures(replace(put_credit_spread, limit_price=Decimal("0.50")), flat_book, ctx)


def test_gate12_rejects_symbols_absent_from_the_chain(put_credit_spread, flat_book, ctx) -> None:
    x = KernelContext(now=ctx.now, available_symbols=frozenset({"SPY260827P00761000"}))
    assert 12 in failures(put_credit_spread, flat_book, x)


def test_gate12_skips_the_chain_check_when_no_chain_supplied(
    put_credit_spread, flat_book, ctx
) -> None:
    """A missing input must not masquerade as a risk violation."""
    assert 12 not in failures(put_credit_spread, flat_book, ctx)


# --- the kernel reports every reason, not just the first ------------------- #

def test_a_bad_proposal_reports_all_of_its_failures(
    put_credit_spread, flat_book, ctx
) -> None:
    awful = replace(put_credit_spread, contracts=500, net_credit=Decimal("0.02"),
                    limit_price=Decimal("0.02"))
    d = evaluate(awful, flat_book, ctx)
    assert not d.approved
    assert {2, 9} <= {v.number for v in d.failures}
    assert len(d.verdicts) == 12


# --- findings worth pinning down ------------------------------------------- #

def test_gate6_cannot_block_the_first_trade_in_an_empty_book(
    put_credit_spread, flat_book, ctx
) -> None:
    """A 40% single-name cap is unsatisfiable below 3 structures — 1/N > 0.40.

    Enforcing it from the first trade would make the gate block the very book it
    exists to shape, and the agent would never open a position.
    """
    assert 6 not in failures(put_credit_spread, flat_book, ctx)
    one_open = replace(flat_book, open_structures=(_open("QQQ", 700, risk="100"),))
    assert 6 not in failures(put_credit_spread, one_open, ctx)


def test_gate7_binds_far_harder_than_gate2_on_directional_structures() -> None:
    """A measured design tension, recorded so it is not rediscovered mid-session.

    Gate 2 (2% of equity) sizes on *max loss*; Gate 7 (5% dollar delta) sizes on
    *directional exposure*. On a $1-wide put credit spread on a $765 underlying
    these disagree by more than an order of magnitude, and Gate 7 wins — so the
    2% risk budget is unreachable with directional structures alone.
    """

    book = PortfolioState(equity=Decimal(100_000), peak_equity=Decimal(100_000),
                          day_pnl=Decimal(0))
    x = KernelContext(now=DEFAULT_NOW)

    def spread(n: int) -> TradeProposal:
        return TradeProposal(
            structure=Structure.PUT_CREDIT_SPREAD,
            underlying="SPY", spot=Decimal("765.85"), expiry=DEFAULT_EXPIRY,
            legs=(make_leg("SPY260827P00761000", short=True, bid="0.50", ask="0.52", delta=-0.16),
                  make_leg("SPY260827P00760000", bid="0.30", ask="0.32", delta=-0.11)),
            contracts=n, net_credit=Decimal("0.20"), width=Decimal(1),
            client_order_id=f"vigil-tension-{n}", limit_price=Decimal("0.20"))

    # Gate 2 alone would allow 25 contracts...
    assert 2 not in failures(spread(25), book, x)
    # ...but Gate 7 rejects everything above 1.
    assert 7 not in failures(spread(1), book, x)
    assert 7 in failures(spread(2), book, x)

    # The ratio is the finding: ~15x tighter than the risk budget implies.
    assert spread(1).dollar_delta > Decimal(3_500)


def test_gate12_accepts_a_limit_at_the_package_mid(put_credit_spread, flat_book, ctx) -> None:
    """The builder opens at the mid (§2.5) while being *judged* on the conservative
    credit. Gate 12 compares the limit to the mid, so the two must agree."""
    from dataclasses import replace as _replace

    p = put_credit_spread
    package_mid = sum(
        (Decimal(leg.signed_ratio) * leg.mid for leg in p.legs), Decimal(0)
    ) * -1
    assert 12 not in failures(_replace(p, limit_price=package_mid), flat_book, ctx)


# --------------------------------------------------------------------------- #
# B4 — Gate 1 derives width from the strikes instead of believing the proposal
# --------------------------------------------------------------------------- #

def test_gate1_rejects_a_declared_width_the_strikes_do_not_support(
    put_credit_spread, flat_book, ctx
) -> None:
    """The one place the kernel could be *convinced*.

    `max_loss` is `(width − credit) × 100 × n`, so understating width understates
    the number Gate 2 sizes against. Before this check, a $5-wide spread declaring
    width $1 reported $82 of max loss instead of $482 and passed all twelve gates.
    """
    lying = replace(
        put_credit_spread,
        legs=(
            make_leg("SPY260827P00761000", short=True, bid="0.50", ask="0.52", delta=-0.16),
            make_leg("SPY260827P00756000", short=False, bid="0.30", ask="0.32", delta=-0.11),
        ),
        width=Decimal(1),
    )
    v = _verdict(evaluate(lying, flat_book, ctx), 1)
    assert not v.passed
    assert "width" in v.reason


def test_gate1_accepts_a_width_that_matches_the_strikes(
    put_credit_spread, flat_book, ctx
) -> None:
    assert _verdict(evaluate(put_credit_spread, flat_book, ctx), 1).passed


def test_gate1_measures_an_iron_condor_as_one_width_not_the_strike_span(
    iron_condor, flat_book, ctx
) -> None:
    """The gap between the put spread and the call spread is the profit zone, not
    a width — measuring across it would reject every condor ever built."""
    assert _verdict(evaluate(iron_condor, flat_book, ctx), 1).passed


def test_gate1_rejects_legs_spanning_multiple_expiries(
    put_credit_spread, flat_book, ctx
) -> None:
    """A calendar has no max loss derivable from strike distance, and
    `TradeProposal` carries one `expiry` that could only describe one of them."""
    mixed = replace(
        put_credit_spread,
        legs=(
            make_leg("SPY260827P00761000", short=True, bid="0.50", ask="0.52", delta=-0.16),
            make_leg("SPY260828P00760000", short=False, bid="0.30", ask="0.32", delta=-0.11),
        ),
    )
    v = _verdict(evaluate(mixed, flat_book, ctx), 1)
    assert not v.passed and "expiries" in v.reason


def test_gate1_rejects_legs_that_disagree_with_the_proposal_expiry(
    put_credit_spread, flat_book, ctx
) -> None:
    v = _verdict(evaluate(replace(put_credit_spread, expiry=date(2026, 8, 28)),
                          flat_book, ctx), 1)
    assert not v.passed and "expiry" in v.reason


# --------------------------------------------------------------------------- #
# B5 — Gate 11 reads wall-clock fields in US/Eastern, never the caller's zone
# --------------------------------------------------------------------------- #

def test_gate11_gives_the_same_verdict_for_the_same_instant_in_any_zone(
    put_credit_spread, flat_book
) -> None:
    """09:00 ET is pre-market. Expressed as 13:00 UTC it used to read as
    mid-session, and Gate 11 permitted a pre-market entry."""
    et_now = datetime(2026, 8, 26, 9, 0, tzinfo=ET)
    for zone in (UTC, ZoneInfo("Asia/Karachi"), ZoneInfo("Europe/London")):
        same_instant = et_now.astimezone(zone)
        v = _verdict(evaluate(put_credit_spread, flat_book,
                              KernelContext(now=same_instant)), 11)
        assert not v.passed, f"pre-market entry permitted when now was {zone}"


def test_gate11_still_permits_a_valid_session_time_expressed_in_utc(
    put_credit_spread, flat_book
) -> None:
    """The conversion must not simply reject everything non-ET."""
    et_now = datetime(2026, 8, 26, 11, 0, tzinfo=ET)
    v = _verdict(evaluate(put_credit_spread, flat_book,
                          KernelContext(now=et_now.astimezone(UTC))), 11)
    assert v.passed


def test_kernel_context_refuses_a_naive_timestamp() -> None:
    """`astimezone` on a naive datetime silently assumes the machine's zone —
    the same bug arriving through a different door. There is no safe default."""
    with pytest.raises(ValueError, match="timezone-aware"):
        KernelContext(now=datetime(2026, 8, 26, 11, 0))


# --------------------------------------------------------------------------- #
# Gate 10 — the macro blackout is asymmetric on purpose
# --------------------------------------------------------------------------- #

def test_gate10_blocks_entry_shortly_before_a_macro_print(
    put_credit_spread, flat_book
) -> None:
    now = datetime(2026, 8, 26, 11, 0, tzinfo=ET)
    ctx = KernelContext(now=now, macro_events=(now + timedelta(minutes=3),))
    v = _verdict(evaluate(put_credit_spread, flat_book, ctx), 10)
    assert not v.passed and "before" in v.reason


def test_gate10_keeps_blocking_after_the_print_when_the_tape_is_disorderly(
    put_credit_spread, flat_book
) -> None:
    """The window after a print is longer than the window before it: spreads
    widen, the indicative feed lags hardest, and the move keeps developing."""
    now = datetime(2026, 8, 26, 11, 0, tzinfo=ET)
    ctx = KernelContext(now=now, macro_events=(now - timedelta(minutes=10),))
    v = _verdict(evaluate(put_credit_spread, flat_book, ctx), 10)
    assert not v.passed and "after" in v.reason


def test_gate10_clears_once_the_blackout_has_elapsed(
    put_credit_spread, flat_book
) -> None:
    now = datetime(2026, 8, 26, 11, 0, tzinfo=ET)
    ctx = KernelContext(now=now, macro_events=(now - timedelta(minutes=30),))
    assert _verdict(evaluate(put_credit_spread, flat_book, ctx), 10).passed
