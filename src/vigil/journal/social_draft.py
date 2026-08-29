"""The daily build-in-public draft (§6.1) — the agent narrating its own journal.

`python -m vigil.journal.social_draft` reads the day's `DayReport` and produces a
short post a human then chooses to publish. **Text only, no side effects**: this
module never posts anything, and the LLM here — unlike the portfolio manager —
cannot touch a trade, so its blast radius is a paragraph.

Same fallback discipline as everywhere the model appears (§6.3): a deterministic
template is built from the numbers first, the model is asked to improve on it, and
*any* failure returns the template. The post is drafted from real figures either
way — the model rewrites the prose, it does not invent the facts. That ordering is
also what makes the honest claim possible: the draft cannot report a trade that
did not happen, because its inputs are the journalled counts, not the model's
imagination.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Any

from vigil.config import AgentConfig
from vigil.db.session import get_session
from vigil.journal.report import DayReport, build_report
from vigil.logging import get_logger

log = get_logger(__name__)

# The audience the submission commits to tagging (SUBMISSION.md). Kept as data so
# the template and the model prompt cannot disagree on the handles.
_TAGS = "@lablabai @AlpacaHQ"

_SYSTEM = """\
You write one short build-in-public post (at most ~700 characters) for an \
autonomous options-trading agent competing in a hackathon. Voice: technical, \
candid, a little dry — an engineer showing their work, not a hype account. \
You are given a factual draft built from the day's journal. Improve the prose, \
keep every number exactly as given, invent nothing, and keep the hashtags and \
handles. If the day was quiet (no trades), say so plainly — a risk kernel that \
correctly declined is a feature, not a failure. Return only the post text."""


@dataclass(frozen=True, slots=True)
class SocialDraft:
    """A post ready for a human to publish, and how it was produced."""

    text: str
    fell_back: bool
    model: str

    @property
    def char_count(self) -> int:
        return len(self.text)


def _template_draft(report: DayReport) -> str:
    """The deterministic post — real numbers, no model. Also the fallback.

    Deliberately publishable on its own. If the model never runs all week, this is
    what ships, so it has to stand without editing rather than read like a
    placeholder waiting to be improved.
    """
    pnl = report.session_pnl
    if report.fills > 0:
        headline = (
            f"Day on the desk: {report.fills} fill(s) from {report.approved} "
            f"kernel-approved structure(s)."
        )
    elif report.approved > 0:
        headline = (
            f"{report.approved} structure(s) cleared all 12 risk gates today; "
            f"none reached a fill."
        )
    elif report.proposals > 0:
        blocker = report.top_blocker
        why = f" Binding gate: {blocker[0]}." if blocker else ""
        headline = (
            f"Built {report.proposals} candidate(s), approved none — the kernel "
            f"held the line.{why}"
        )
    else:
        headline = "Quiet session — the regime read said stand down, so it did."

    pnl_line = "" if pnl is None else f" Session P&L ${pnl:+,.0f}."
    body = (
        f"Vigil — autonomous options agent, Alpaca paper.\n"
        f"{headline}{pnl_line}\n"
        f"{report.cycles} cycles · equity ${report.equity:,.0f} · "
        f"open risk ${report.open_risk:,.0f}.\n"
        f"The LLM proposes; the deterministic risk kernel disposes.\n"
        f"{_TAGS} #buildinpublic"
    )
    return body


async def _llm_rewrite(client: Any, config: AgentConfig, draft: str) -> str:
    """Ask the model to improve the draft. Raises on anything — the caller falls back.

    Plain-text output, not a strict schema: this is prose, and there is no field
    for the kernel to consume. High effort (`review_effort`) because it runs once
    a day, not thirty times, so judgement is worth more than latency here. `client`
    is typed `Any` because it is either an `AsyncOpenAI` or an injected test double
    and this module deliberately depends on neither concretely.
    """
    response = await client.responses.create(
        model=config.model,
        reasoning={"effort": config.review_effort},
        input=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Factual draft to improve:\n\n{draft}"},
        ],
        max_output_tokens=config.max_output_tokens,
        timeout=config.timeout_seconds,
    )
    text = str(response.output_text).strip()
    if not text:
        raise ValueError("model returned an empty post")
    return text


async def draft_post(
    report: DayReport,
    *,
    client: Any | None = None,
    config: AgentConfig | None = None,
) -> SocialDraft:
    """Draft the day's post: model-improved when available, template otherwise.

    `client`/`config` are injectable so a test drives the model path with a fake
    and never opens a socket. Left unset, the shared `build_client` decides — and
    returns `None` (template path) whenever the model is disabled or unkeyed.
    """
    template = _template_draft(report)

    if client is None:
        from vigil.agent import build_client

        built = build_client(config)
        if built is None:
            return SocialDraft(text=template, fell_back=True, model="deterministic")
        client, config = built
    elif config is None:
        from vigil.config import agent_config

        config = agent_config()

    try:
        text = await _llm_rewrite(client, config, template)
    except Exception as exc:  # noqa: BLE001 — a draft must always be producible
        log.warning("social.fell_back", reason=type(exc).__name__, detail=str(exc)[:200])
        return SocialDraft(text=template, fell_back=True, model=config.model)

    return SocialDraft(text=text, fell_back=False, model=config.model)


async def _amain() -> int:
    async with get_session() as db:
        report = await build_report(db)
    if report is None:
        print("no account in the journal yet — has the worker run?")
        return 1
    draft = await draft_post(report)
    source = "template" if draft.fell_back else draft.model
    print(f"--- build-in-public draft ({source}, {draft.char_count} chars) ---\n")
    print(draft.text)
    return 0


def main() -> int:
    argparse.ArgumentParser(description="Draft the day's build-in-public post").parse_args()
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
