"""The session report — `python -m vigil.journal.report [--today]` (CLAUDE.md).

A deterministic read over the journal: no model, no network, no broker. It answers
the questions a human asks at the end of a session — *did it trade, what did the
gates block, what is the book, and did the model actually run* — from the tables
the worker already wrote, so it works with the worker stopped.

**Two layers on purpose.** `build_report` gathers a `DayReport` (plain data) and
`render` turns it into text. Keeping them apart means the social draft (§6.1) can
reuse the exact same numbers the terminal shows without scraping a formatted
string — the post and the report can never disagree because they read one source.

The reads reuse `db/repositories/reporting.py` wherever it already has the query,
and add only session-scoped *counts* here. That split is deliberate: `reporting`
serves the API's fixed response shapes; a count scoped to "today's session" is a
report concern and lives with the report.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vigil.clock import ET, today_et
from vigil.db.models import Cycle, Fill, LlmMemo, Order, Proposal, Session
from vigil.db.repositories import reporting as R
from vigil.db.session import get_session


@dataclass(frozen=True, slots=True)
class LlmStats:
    """How the model behaved today — the two numbers §6 says to make observable.

    `fallback_rate` is reliability (how often the deterministic path decided) and
    `cache_hit_rate` is cost (how much of the prompt prefix actually cached). Both
    are `None` when the model made no calls, which reads as "not applicable" rather
    than a misleading zero.
    """

    calls: int
    fallbacks: int
    input_tokens: int
    cached_tokens: int
    latest_memo: str | None

    @property
    def fallback_rate(self) -> float | None:
        return None if self.calls == 0 else self.fallbacks / self.calls

    @property
    def cache_hit_rate(self) -> float | None:
        return None if self.input_tokens == 0 else self.cached_tokens / self.input_tokens


@dataclass(frozen=True, slots=True)
class DayReport:
    """Everything the terminal report and the social draft both read."""

    trading_date: date
    account_id: str
    equity: Decimal
    opening_equity: Decimal | None
    day_pnl: Decimal
    open_risk: Decimal
    net_dollar_delta: Decimal
    cycles: int
    proposals: int
    approved: int
    orders: int
    fills: int
    gate_stats: list[tuple[int, str, int, int]] = field(default_factory=list)
    open_structures: list[tuple[str, date, Decimal]] = field(default_factory=list)
    llm: LlmStats | None = None

    @property
    def session_pnl(self) -> Decimal | None:
        """Realised move since the open. `None` before the first snapshot exists."""
        if self.opening_equity is None:
            return None
        return self.equity - self.opening_equity

    @property
    def top_blocker(self) -> tuple[str, int] | None:
        """The gate that rejected the most candidates today — the binding constraint.

        The single most useful line in the whole report on an idle day: "why didn't
        it trade?" is answered by whichever gate failed most, and A2/A3 (§1.3) are
        exactly the hypotheses this number tests session by session.
        """
        blockers = [(name, bad) for _, name, _, bad in self.gate_stats if bad > 0]
        return max(blockers, key=lambda pair: pair[1]) if blockers else None


async def _session_for(db: AsyncSession, *, account_id: int, day: date | None) -> Session | None:
    """Today's session row, a named day's, or the most recent one."""
    if day is None:
        return await R.today_session(db, account_id=account_id)
    row: Session | None = await db.scalar(
        select(Session).where(
            Session.account_id == account_id, Session.trading_date == day
        )
    )
    return row


async def _counts(
    db: AsyncSession, session_id: int, trading_date: date
) -> tuple[int, int, int, int, int]:
    """(cycles, proposals, approved, orders, fills) for one session.

    Cycles and proposals scope through the `cycle -> session` foreign key the
    schema indexes. **Orders and fills cannot** — an `Order` links to a
    `structure`, not a cycle, so there is no session key to join on. They are
    scoped instead to the session's Eastern calendar day: `[00:00 ET, +24h)`
    converted to the UTC the `timestamptz` columns store. Counting in SQL rather
    than loading rows keeps the report to five integers, not a few hundred objects.
    """
    day_start = datetime.combine(trading_date, time.min, tzinfo=ET)
    day_end = day_start + timedelta(days=1)

    cycles = await db.scalar(
        select(func.count()).select_from(Cycle).where(Cycle.session_id == session_id)
    )
    proposals = await db.scalar(
        select(func.count())
        .select_from(Proposal)
        .join(Cycle, Cycle.id == Proposal.cycle_id)
        .where(Cycle.session_id == session_id)
    )
    approved = await db.scalar(
        select(func.count())
        .select_from(Proposal)
        .join(Cycle, Cycle.id == Proposal.cycle_id)
        .where(Cycle.session_id == session_id, Proposal.approved.is_(True))
    )
    orders = await db.scalar(
        select(func.count())
        .select_from(Order)
        .where(Order.submitted_at >= day_start, Order.submitted_at < day_end)
    )
    fills = await db.scalar(
        select(func.count())
        .select_from(Fill)
        .where(Fill.filled_at >= day_start, Fill.filled_at < day_end)
    )
    return (
        int(cycles or 0), int(proposals or 0), int(approved or 0),
        int(orders or 0), int(fills or 0),
    )


async def _llm_stats(db: AsyncSession, session_id: int) -> LlmStats:
    """Aggregate the day's `llm_memos` into the reliability and cost headlines."""
    row = (
        await db.execute(
            select(
                func.count(),
                func.count().filter(LlmMemo.fell_back.is_(True)),
                func.coalesce(func.sum(LlmMemo.input_tokens), 0),
                func.coalesce(func.sum(LlmMemo.cached_tokens), 0),
            )
            .join(Cycle, Cycle.id == LlmMemo.cycle_id)
            .where(Cycle.session_id == session_id)
        )
    ).one()
    latest = await db.scalar(
        select(LlmMemo.memo)
        .join(Cycle, Cycle.id == LlmMemo.cycle_id)
        .where(Cycle.session_id == session_id, LlmMemo.memo.isnot(None))
        .order_by(LlmMemo.created_at.desc())
        .limit(1)
    )
    calls, fallbacks, in_tok, cached = row
    return LlmStats(
        calls=int(calls), fallbacks=int(fallbacks),
        input_tokens=int(in_tok), cached_tokens=int(cached), latest_memo=latest,
    )


async def build_report(db: AsyncSession, *, day: date | None = None) -> DayReport | None:
    """Gather one session into a `DayReport`. `None` when there is no account yet.

    `day=None` means today (or the latest session if today has not opened) — the
    default the `--today` command wants.
    """
    account = await R.the_account(db)
    if account is None:
        return None

    session = await _session_for(db, account_id=account.id, day=day)
    equity_row = await R.latest_equity(db, account_id=account.id)
    gate_stats = await R.gate_stats(db)
    structures = await R.open_structures(db)
    open_risk = await R.open_risk(db)

    cycles = proposals = approved = orders = fills = 0
    llm: LlmStats | None = None
    if session is not None:
        cycles, proposals, approved, orders, fills = await _counts(
            db, session.id, session.trading_date
        )
        llm = await _llm_stats(db, session.id)

    return DayReport(
        trading_date=session.trading_date if session else today_et(),
        account_id=account.alpaca_account_id,
        equity=equity_row.equity if equity_row else Decimal(0),
        opening_equity=session.opening_equity if session else None,
        day_pnl=equity_row.day_pnl if equity_row else Decimal(0),
        open_risk=open_risk,
        net_dollar_delta=equity_row.net_dollar_delta if equity_row else Decimal(0),
        cycles=cycles, proposals=proposals, approved=approved, orders=orders, fills=fills,
        gate_stats=gate_stats,
        open_structures=[(s.underlying, s.expiry, s.max_loss) for s in structures],
        llm=llm,
    )


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def render(report: DayReport) -> str:
    """A plain-text session report. Deliberately not Rich: this is piped and grep'd.

    Kept to one screen. The ordering mirrors the desk's own priority — capital
    first (equity, P&L, open risk), then activity, then *why* (the binding gate),
    then the book, then the model.
    """
    lines: list[str] = []
    add = lines.append

    add(f"═══ Vigil session report · {report.trading_date} ═══")
    add(f"account {report.account_id}")
    add("")
    add(f"equity        ${report.equity:,.2f}")
    if report.session_pnl is not None:
        add(f"session P&L   ${report.session_pnl:+,.2f}")
    add(f"day P&L       ${report.day_pnl:+,.2f}")
    add(f"open risk     ${report.open_risk:,.2f}")
    add(f"net $delta    ${report.net_dollar_delta:+,.0f}")
    add("")
    add(
        f"activity      {report.cycles} cycles · {report.proposals} proposals · "
        f"{report.approved} approved · {report.orders} orders · {report.fills} fills"
    )
    blocker = report.top_blocker
    if blocker is not None:
        add(f"top blocker   gate '{blocker[0]}' rejected {blocker[1]} candidate(s)")
    elif report.proposals == 0:
        add("top blocker   — no candidates built (regime stand-down or no chain)")

    if report.open_structures:
        add("")
        add("open book:")
        for underlying, expiry, max_loss in report.open_structures:
            add(f"  {underlying} exp {expiry}  max loss ${max_loss:,.2f}")

    if report.llm is not None and report.llm.calls > 0:
        add("")
        add(
            f"model         {report.llm.calls} calls · "
            f"fallback {_pct(report.llm.fallback_rate)} · "
            f"cache hit {_pct(report.llm.cache_hit_rate)}"
        )
        if report.llm.latest_memo:
            add(f"latest memo   {report.llm.latest_memo}")

    if report.gate_stats:
        add("")
        add("gate ledger (passes / rejections):")
        for no, name, ok, bad in report.gate_stats:
            add(f"  {no:>2} {name:<24} {ok:>5} / {bad}")

    return "\n".join(lines)


async def _amain(day: date | None) -> int:
    async with get_session() as db:
        report = await build_report(db, day=day)
    if report is None:
        print("no account in the journal yet — has the worker run?")
        return 1
    print(render(report))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Vigil session report")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--today", action="store_true", help="today's session (the default)")
    group.add_argument("--date", type=date.fromisoformat, help="a specific YYYY-MM-DD session")
    args = parser.parse_args()
    # --today and the no-arg default are the same thing: day=None resolves to
    # today, falling back to the latest session if today has not opened yet.
    day = args.date if args.date else None
    return asyncio.run(_amain(day))


if __name__ == "__main__":
    raise SystemExit(main())
