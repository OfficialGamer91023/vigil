"""Auto post-close journaling (`vigil.journal.emit`).

The pieces it orchestrates — `build_report`, `render`, `draft_post` — are each
tested elsewhere; here we prove the seam: that a post-close pass writes both
artefacts to disk with the right names and content, that it survives a keyless
model by shipping the template, and that "no account yet" is a clean `None` rather
than a crash. No database and no socket: `build_report` is monkeypatched to a fixed
`DayReport` and the writes go to `tmp_path`.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from vigil.journal import emit
from vigil.journal.report import DayReport


def _report(**over: Any) -> DayReport:
    """A baseline report; each test overrides only the field it cares about."""
    base: dict[str, Any] = dict(
        trading_date=date(2026, 8, 31),
        account_id="acct-xyz",
        equity=Decimal("100250"),
        opening_equity=Decimal("100000"),
        day_pnl=Decimal("250"),
        open_risk=Decimal("400"),
        net_dollar_delta=Decimal("0"),
        cycles=6, proposals=2, approved=1, orders=1, fills=1,
        gate_stats=[(1, "defined_risk", 1, 0)],
        open_structures=[],
        llm=None,
    )
    base.update(over)
    return DayReport(**base)


# --------------------------------------------------------------------------- #
# write_daily_artifacts — pure filesystem
# --------------------------------------------------------------------------- #

def test_write_lands_both_files_named_by_date(tmp_path):
    report_path, social_path = emit.write_daily_artifacts(
        date(2026, 8, 31),
        report_text="the report",
        social_text="the post",
        base_dir=tmp_path,
    )
    assert report_path == tmp_path / "2026-08-31-report.txt"
    assert social_path == tmp_path / "2026-08-31-social.txt"
    assert report_path.read_text() == "the report"
    assert social_path.read_text() == "the post"


def test_write_creates_a_missing_base_dir(tmp_path):
    """The worker should not need someone to `mkdir` first — the dir is made on write."""
    nested = tmp_path / "var" / "journal"
    assert not nested.exists()
    emit.write_daily_artifacts(
        date(2026, 8, 31), report_text="r", social_text="s", base_dir=nested
    )
    assert nested.is_dir()


# --------------------------------------------------------------------------- #
# emit_session_journal — the orchestration
# --------------------------------------------------------------------------- #

class _FakeModelClient:
    """Drives `draft_post`'s model path with a canned response — no socket."""

    class _Resp:
        output_text = "  A model-polished post.  "

    class _Responses:
        async def create(self, **_kw: Any) -> _FakeModelClient._Resp:
            return _FakeModelClient._Resp()

    responses = _Responses()


async def test_emit_writes_both_artefacts_and_reports_where(tmp_path, monkeypatch):
    monkeypatch.setattr(emit, "build_report", _stub_build(_report()))

    # No client injected → draft_post takes the deterministic template path (no key),
    # which is exactly the shape the worker runs in without an OpenAI key.
    result = await emit.emit_session_journal(object(), base_dir=tmp_path)

    assert result is not None
    assert result.report_path.exists() and result.social_path.exists()
    assert result.report_path == tmp_path / "2026-08-31-report.txt"
    # The rendered report is real output, not empty.
    assert "Vigil session report" in result.report_path.read_text()
    # Template path → fell back, and the file holds the draft text verbatim.
    assert result.draft.fell_back is True
    assert result.social_path.read_text() == result.draft.text


async def test_emit_uses_the_injected_model_when_given_one(tmp_path, monkeypatch):
    monkeypatch.setattr(emit, "build_report", _stub_build(_report()))

    result = await emit.emit_session_journal(
        object(), base_dir=tmp_path, client=_FakeModelClient()
    )

    assert result is not None
    assert result.draft.fell_back is False
    # The model's text is what gets persisted, stripped.
    assert result.draft.text == "A model-polished post."
    assert result.social_path.read_text() == "A model-polished post."


async def test_emit_returns_none_when_there_is_no_account(tmp_path, monkeypatch):
    """No account row yet (worker never ran) is a clean no-op, not an exception —
    and nothing is written."""
    monkeypatch.setattr(emit, "build_report", _stub_build(None))

    result = await emit.emit_session_journal(object(), base_dir=tmp_path)

    assert result is None
    assert list(tmp_path.iterdir()) == []


def _stub_build(report: DayReport | None):
    """A stand-in for the async `build_report(db)` that ignores the db and returns
    a fixed report — keeps these tests network- and database-free."""

    async def _build(_db: Any, **_kw: Any) -> DayReport | None:
        return report

    return _build
