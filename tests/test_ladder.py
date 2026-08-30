"""The escalation ladder (§4.7) — the size regime as a pure decision table.

Two things these tests pin down. First, the *table* itself: the right rung for
each `(sessions_left, ahead/behind)` standing, read first-match-wins exactly as
PLAN §4.7 is written. Second, and more important, the *invariant* that keeps the
ladder honest — core risk per trade is never allowed above Gate 2's ceiling, no
matter which rung fires. The ladder may escalate the convexity mix freely; it may
not widen the per-trade loss the frozen kernel would approve.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from vigil.config import LadderConfig, LadderRung, ladder_config, risk_config
from vigil.strategy.ladder import (
    effective_core_risk_pct,
    resolve_rung,
    sessions_left,
)

# The real tournament calendar, so the tests move with the shipped config rather
# than a copy that could drift from it.
CAL = ladder_config().session_dates


# --------------------------------------------------------------------------- #
# sessions_left — a count of remaining calendar dates, today included
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "today, expected",
    [
        (date(2026, 8, 28), 6),   # opening session: all six remain
        (date(2026, 8, 31), 5),
        (date(2026, 9, 1), 4),
        (date(2026, 9, 2), 3),
        (date(2026, 9, 3), 2),
        (date(2026, 9, 4), 1),    # final session still counts as one left
        (date(2026, 9, 5), 0),    # past the calendar
        (date(2026, 8, 27), 6),   # a day before it starts
        (date(2026, 8, 29), 5),   # a weekend inside the window: counts forward dates
    ],
)
def test_sessions_left_counts_dates_on_or_after_today(today: date, expected: int) -> None:
    assert sessions_left(today, CAL) == expected


# --------------------------------------------------------------------------- #
# resolve_rung — the §4.7 table, first match wins
# --------------------------------------------------------------------------- #

BEHIND = Decimal("0.00")     # a flat day is below the +2% target → "behind"
AHEAD = Decimal("0.05")      # a +5% day clears the target → "ahead"


@pytest.mark.parametrize(
    "today, day_pnl, conv, core",
    [
        # ≥4 sessions left: the base rung, regardless of P&L (the `any` row claims
        # the early sessions before any escalation rung can).
        (date(2026, 8, 28), BEHIND, "0.20", "0.020"),
        (date(2026, 8, 28), AHEAD, "0.20", "0.020"),
        (date(2026, 9, 1), BEHIND, "0.20", "0.020"),
        # The escalation rungs — only when behind.
        (date(2026, 9, 2), BEHIND, "0.30", "0.020"),   # 3 left
        (date(2026, 9, 3), BEHIND, "0.40", "0.025"),   # 2 left
        (date(2026, 9, 4), BEHIND, "0.50", "0.025"),   # 1 left
        # Ahead in the back half: fall through the behind-only rungs to the
        # protective catch-all.
        (date(2026, 9, 2), AHEAD, "0.15", "0.015"),
        (date(2026, 9, 4), AHEAD, "0.15", "0.015"),
        # Past the calendar resolves to the catch-all rather than crashing.
        (date(2026, 9, 5), BEHIND, "0.15", "0.015"),
    ],
)
def test_resolve_rung_matches_the_plan_table(
    today: date, day_pnl: Decimal, conv: str, core: str
) -> None:
    rung = resolve_rung(today, day_pnl, ladder_config())
    assert rung.convexity_share == Decimal(conv)
    assert rung.core_risk_pct == Decimal(core)


def test_the_target_is_the_ahead_behind_boundary() -> None:
    """Exactly at the target counts as ahead (`>=`), one tick below is behind."""
    cfg = ladder_config()
    target = cfg.day_pnl_target_pct
    back_half = date(2026, 9, 3)   # 2 left, where ahead vs behind changes the rung

    at_target = resolve_rung(back_half, target, cfg)
    below = resolve_rung(back_half, target - Decimal("0.0001"), cfg)

    assert at_target.ahead is True
    assert at_target.convexity_share == Decimal("0.15")     # protective rung
    assert below.ahead is False
    assert below.convexity_share == Decimal("0.40")         # escalation rung


def test_the_label_explains_the_choice() -> None:
    """The journal line the worker records must state both inputs, not just size."""
    rung = resolve_rung(date(2026, 9, 3), BEHIND, ladder_config())
    assert "2 left" in rung.label
    assert "behind" in rung.label


# --------------------------------------------------------------------------- #
# The clamp — the one invariant that ties the ladder to the frozen kernel
# --------------------------------------------------------------------------- #

def test_the_aggressive_rung_is_clamped_to_the_gate2_ceiling() -> None:
    """§4.7's 2.5% rung is a *request*; Gate 2's 2.0% is the cap. The smaller wins."""
    ceiling = risk_config().max_risk_per_trade_pct
    assert ceiling == Decimal("0.02")
    rung = resolve_rung(date(2026, 9, 3), BEHIND, ladder_config())   # the 2.5% rung
    assert rung.core_risk_pct == Decimal("0.025")
    assert effective_core_risk_pct(rung, ceiling) == Decimal("0.02")


def test_a_rung_below_the_ceiling_is_left_untouched() -> None:
    """The protective 1.5% rung is under the cap, so the clamp is a no-op on it."""
    ceiling = risk_config().max_risk_per_trade_pct
    rung = resolve_rung(date(2026, 9, 3), AHEAD, ladder_config())    # the 1.5% rung
    assert effective_core_risk_pct(rung, ceiling) == Decimal("0.015")


@pytest.mark.parametrize("today", [CAL[0], *CAL, date(2026, 9, 5)])
@pytest.mark.parametrize("day_pnl", [Decimal("-0.10"), BEHIND, AHEAD, Decimal("0.30")])
def test_effective_core_never_exceeds_the_ceiling_for_any_standing(
    today: date, day_pnl: Decimal
) -> None:
    """The safety property in one line: across every reachable standing, the size
    the ladder hands to sizing is at or below Gate 2's per-trade ceiling. If a
    future config edit adds a rung above the cap, this fails rather than shipping
    a position the kernel would reject on every entry."""
    ceiling = risk_config().max_risk_per_trade_pct
    rung = resolve_rung(today, day_pnl, ladder_config())
    assert effective_core_risk_pct(rung, ceiling) <= ceiling


# --------------------------------------------------------------------------- #
# Loader validation — a malformed ladder fails at load, not at 09:31
# --------------------------------------------------------------------------- #

def _ladder_dict(rungs: list[dict]) -> dict:
    return {
        "ladder": {
            "session_dates": ["2026-09-04"],
            "day_pnl_target_pct": 0.02,
            "rungs": rungs,
        }
    }


CATCH_ALL = {"min_sessions_left": 0, "when": "any", "convexity_share": 0.15, "core_risk_pct": 0.015}


def test_an_unknown_when_is_rejected_at_load(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = {"min_sessions_left": 3, "when": "winning", "convexity_share": 0.3, "core_risk_pct": 0.02}
    monkeypatch.setattr("vigil.config._load", lambda name: _ladder_dict([bad, CATCH_ALL]))
    with pytest.raises(ValueError, match="when"):
        LadderConfig.load()


def test_a_missing_catch_all_rung_is_rejected_at_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a `min: 0, when: any` rung, some standing would match nothing —
    refuse the config rather than discover the hole while sizing a live trade."""
    only_escalation = {
        "min_sessions_left": 3, "when": "behind", "convexity_share": 0.3, "core_risk_pct": 0.02,
    }
    monkeypatch.setattr("vigil.config._load", lambda name: _ladder_dict([only_escalation]))
    with pytest.raises(ValueError, match="catch-all"):
        LadderConfig.load()


def test_session_dates_accept_both_date_objects_and_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """YAML yields a `date` for an unquoted value and a `str` for a quoted one;
    both must load, so the file's author is never surprised by their own quoting."""
    d = _ladder_dict([CATCH_ALL])
    d["ladder"]["session_dates"] = [date(2026, 9, 3), "2026-09-04"]
    monkeypatch.setattr("vigil.config._load", lambda name: d)
    cfg = LadderConfig.load()
    assert cfg.session_dates == (date(2026, 9, 3), date(2026, 9, 4))


def test_the_shipped_config_loads_and_has_the_expected_shape() -> None:
    """A smoke test over the real file: the rungs parse, and the last is the
    catch-all the resolver relies on."""
    cfg = ladder_config()
    assert len(cfg.rungs) == 5
    assert isinstance(cfg.rungs[0], LadderRung)
    assert cfg.rungs[-1].min_sessions_left == 0
    assert cfg.rungs[-1].when == "any"
