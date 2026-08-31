"""Produce the day's journal artefacts automatically at post-close.

`journal/report.py` and `journal/social_draft.py` already *build* the day's report
and the build-in-public draft — but only when a human runs the CLI. This module is
the seam that makes the post-close cycle do it on its own (§6.1): one call that
gathers the `DayReport`, renders it, drafts the post, and writes both to disk so
each session leaves a tangible deliverable for the submission.

**No trading logic lives here** — it reads the journal and writes text files, the
same blast radius as the CLI it automates. It is deliberately separate from the
`postclose` cycle so that cycle stays thin and the "how a journal is produced"
detail stays in the journal package. `postclose` calls this best-effort: closing
the books outranks narrating them, so a failure here must not undo the close.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from vigil.journal.report import DayReport, build_report, render
from vigil.journal.social_draft import SocialDraft, draft_post
from vigil.logging import get_logger
from vigil.settings import REPO_ROOT

log = get_logger(__name__)

#: Where the per-day artefacts land. Anchored on the repo root (not the CWD) so the
#: worker writes to the same place whether launched by compose, arq or a bare
#: `python -m`. Gitignored — these are regenerated output, not source.
JOURNAL_DIR: Path = REPO_ROOT / "var" / "journal"


@dataclass(frozen=True, slots=True)
class DailyJournal:
    """The result of one post-close journaling pass: what was produced and where."""

    report: DayReport
    draft: SocialDraft
    report_path: Path
    social_path: Path


def write_daily_artifacts(
    trading_date: date,
    *,
    report_text: str,
    social_text: str,
    base_dir: Path = JOURNAL_DIR,
) -> tuple[Path, Path]:
    """Write the rendered report and social draft to `base_dir`, return their paths.

    Pure filesystem I/O, no database and no model — split out so a test can drive it
    against a `tmp_path` and assert the files exist without standing anything up. The
    date prefixes the filenames so a week of sessions sorts and never collides.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    report_path = base_dir / f"{trading_date.isoformat()}-report.txt"
    social_path = base_dir / f"{trading_date.isoformat()}-social.txt"
    report_path.write_text(report_text)
    social_path.write_text(social_text)
    return report_path, social_path


async def emit_session_journal(
    db: AsyncSession,
    *,
    base_dir: Path = JOURNAL_DIR,
    client: Any | None = None,
) -> DailyJournal | None:
    """Build, render, draft and persist the day's journal. `None` if no account yet.

    `client` is threaded straight through to `draft_post` so a test can inject a
    fake model (or drive the deterministic template path) without opening a socket;
    left unset, `draft_post` decides — and returns the template whenever the model
    is disabled or unkeyed, so this works with no OpenAI key at all.
    """
    report = await build_report(db)
    if report is None:
        return None

    report_text = render(report)
    # draft_post never raises — any model failure returns the deterministic template
    # (§6.3), so the draft is always producible and this call has no failure path of
    # its own to guard.
    draft = await draft_post(report, client=client)

    report_path, social_path = write_daily_artifacts(
        report.trading_date,
        report_text=report_text,
        social_text=draft.text,
        base_dir=base_dir,
    )

    log.info(
        "journal.emitted",
        trading_date=report.trading_date.isoformat(),
        report_path=str(report_path),
        social_path=str(social_path),
        social_source="template" if draft.fell_back else draft.model,
        social_chars=draft.char_count,
    )
    return DailyJournal(
        report=report,
        draft=draft,
        report_path=report_path,
        social_path=social_path,
    )
