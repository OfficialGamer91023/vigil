"""The session report and the social draft.

Split by dependency: the shaping and prose logic is tested as pure functions over
a hand-built `DayReport` (no database, no model), and one integration test builds
a report from a **real Postgres** to prove the queries join correctly. The social
draft's model path uses an injected fake, never the network — same rule as the
agent tests.
"""

from __future__ import annotations

import os
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import text

from vigil.db.session import get_session
from vigil.journal.report import DayReport, LlmStats, build_report, render
from vigil.journal.social_draft import _template_draft, draft_post


def _report(**over: Any) -> DayReport:
    """A baseline report; each test overrides only the field it is about."""
    base: dict[str, Any] = dict(
        trading_date=date(2026, 8, 28),
        account_id="acct-xyz",
        equity=Decimal("100000"),
        opening_equity=Decimal("100000"),
        day_pnl=Decimal("0"),
        open_risk=Decimal("0"),
        net_dollar_delta=Decimal("0"),
        cycles=6, proposals=1, approved=0, orders=0, fills=0,
        gate_stats=[(9, "credit_quality", 0, 1), (11, "time_windows", 0, 1),
                    (1, "defined_risk", 1, 0)],
        open_structures=[],
        llm=None,
    )
    base.update(over)
    return DayReport(**base)


# --------------------------------------------------------------------------- #
# DayReport arithmetic
# --------------------------------------------------------------------------- #

def test_session_pnl_is_none_before_the_first_snapshot():
    assert _report(opening_equity=None).session_pnl is None


def test_session_pnl_is_the_move_since_the_open():
    r = _report(opening_equity=Decimal("100000"), equity=Decimal("100250"))
    assert r.session_pnl == Decimal("250")


def test_top_blocker_is_the_gate_that_rejected_the_most():
    r = _report(gate_stats=[(9, "credit_quality", 0, 5), (11, "time_windows", 0, 2)])
    assert r.top_blocker == ("credit_quality", 5)


def test_top_blocker_is_none_when_nothing_was_rejected():
    assert _report(gate_stats=[(1, "defined_risk", 3, 0)]).top_blocker is None


def test_llm_rates_are_none_without_calls():
    s = LlmStats(calls=0, fallbacks=0, input_tokens=0, cached_tokens=0, latest_memo=None)
    assert s.fallback_rate is None
    assert s.cache_hit_rate is None


def test_llm_rates_compute_from_totals():
    s = LlmStats(calls=4, fallbacks=1, input_tokens=1000, cached_tokens=750, latest_memo="hi")
    assert s.fallback_rate == 0.25
    assert s.cache_hit_rate == 0.75


# --------------------------------------------------------------------------- #
# render()
# --------------------------------------------------------------------------- #

def test_render_names_the_binding_gate_on_an_idle_day():
    out = render(_report())
    assert "credit_quality" in out
    assert "top blocker" in out


def test_render_shows_the_open_book_when_there_is_one():
    r = _report(open_structures=[("SPY", date(2026, 8, 28), Decimal("95"))])
    out = render(r)
    assert "open book" in out
    assert "SPY" in out


# --------------------------------------------------------------------------- #
# The social draft — deterministic template
# --------------------------------------------------------------------------- #

def test_template_reports_a_quiet_day_honestly():
    post = _template_draft(_report(proposals=0))
    assert "@lablabai" in post and "@AlpacaHQ" in post
    assert "stand down" in post.lower()


def test_template_reports_a_holdout_with_its_binding_gate():
    post = _template_draft(_report(proposals=1, approved=0))
    assert "credit_quality" in post
    assert "held the line" in post


def test_template_reports_fills_when_they_happened():
    post = _template_draft(_report(approved=1, fills=1))
    assert "fill" in post.lower()


async def test_draft_post_uses_the_template_when_the_model_is_off():
    # No client injected and no key (conftest strips it) -> the template path.
    draft = await draft_post(_report(proposals=0))
    assert draft.fell_back is True
    assert "@lablabai" in draft.text


async def test_draft_post_uses_the_model_when_available():
    class _Resp:
        output_text = "A crisp rewritten post. @lablabai @AlpacaHQ #buildinpublic"

    class _Responses:
        async def create(self, **_kw: Any) -> _Resp:
            return _Resp()

    class _Client:
        responses = _Responses()

    from vigil.config import AgentConfig

    cfg = AgentConfig(
        model="gpt-5.5-test", entry_effort="low", review_effort="high",
        timeout_seconds=5.0, max_output_tokens=100, enabled=True,
    )
    draft = await draft_post(_report(), client=_Client(), config=cfg)
    assert draft.fell_back is False
    assert draft.text.startswith("A crisp rewritten post")


async def test_draft_post_falls_back_to_template_on_model_error():
    class _Responses:
        async def create(self, **_kw: Any) -> Any:
            raise RuntimeError("model down")

    class _Client:
        responses = _Responses()

    from vigil.config import AgentConfig

    cfg = AgentConfig(
        model="gpt-5.5-test", entry_effort="low", review_effort="high",
        timeout_seconds=5.0, max_output_tokens=100, enabled=True,
    )
    draft = await draft_post(_report(proposals=0), client=_Client(), config=cfg)
    assert draft.fell_back is True
    assert "@lablabai" in draft.text


# --------------------------------------------------------------------------- #
# build_report against a real Postgres
# --------------------------------------------------------------------------- #

@pytest.mark.db
class TestBuildReportIntegration:
    @pytest.fixture(autouse=True)
    async def _fresh_engine(self):
        from vigil.db import session as session_module

        session_module.engine.cache_clear()
        session_module.session_factory.cache_clear()
        yield
        try:
            await session_module.engine().dispose()
        finally:
            session_module.engine.cache_clear()
            session_module.session_factory.cache_clear()

    @pytest.fixture(autouse=True)
    async def _require_db(self):
        try:
            async with get_session() as s:
                await s.execute(text("SELECT 1"))
        except Exception:
            pytest.skip(f"no Postgres at {os.getenv('DATABASE_URL', 'localhost/vigil')}")

    async def test_build_report_counts_proposals_scoped_to_the_session(self):
        """Seed one account/session/cycle with a proposal and assert the counts
        join through `cycle -> session` correctly rather than counting the world."""
        tag = f"test-{os.urandom(6).hex()}"
        async with get_session() as s:
            acct = (await s.execute(text(
                "INSERT INTO accounts (alpaca_account_id, starting_equity) "
                "VALUES (:a, 100000) RETURNING id"), {"a": tag})).scalar_one()
            sess = (await s.execute(text(
                "INSERT INTO sessions (account_id, trading_date, opening_equity) "
                "VALUES (:a, :d, 100000) RETURNING id"),
                {"a": acct, "d": date(2026, 8, 28)})).scalar_one()
            cyc = (await s.execute(text(
                "INSERT INTO cycles (session_id, kind) VALUES (:s, 'entry') RETURNING id"),
                {"s": sess})).scalar_one()
            await s.execute(text(
                "INSERT INTO proposals (cycle_id, structure_type, underlying, expiry, "
                "spot, contracts, net_credit, width, max_loss, dollar_delta, "
                "client_order_id, limit_price, approved) VALUES "
                "(:c, 'put_credit_spread', 'SPY', :d, 765, 1, 0.20, 1, 80, 100, "
                ":coid, 0.20, false)"),
                {"c": cyc, "d": date(2026, 8, 28), "coid": f"{tag}-1"})

        try:
            async with get_session() as s:
                report = await build_report(s, day=date(2026, 8, 28))
            assert report is not None
            assert report.proposals == 1
            assert report.approved == 0
            assert report.cycles == 1
        finally:
            async with get_session() as s:
                await s.execute(text("DELETE FROM accounts WHERE id = :a"), {"a": acct})
