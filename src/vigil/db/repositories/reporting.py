"""Journal **reads**, for the API (PLAN §7). Queries only — nothing here writes.

Separate from `journal.py` for the same reason the API is separate from the
worker: `journal.py` is the write path the trading loop depends on, and a read
helper added for a dashboard has no business living inside it where a careless
edit could reach a `session.add`. Every function here returns plain rows or
tuples; shaping them into a response body is `api/schemas.py`'s job.

**Eager loading is deliberate, not decorative.** SQLAlchemy's async session
cannot lazy-load a relationship — touching an unloaded attribute after the
awaited query raises `MissingGreenlet`, and it does so in the serializer, far
from the query that caused it. `selectinload` states up front what the response
will read.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from vigil.db.models import (
    Account,
    ControlFlag,
    Cycle,
    EquitySnapshot,
    GateVerdictRow,
    MarketSnapshotRow,
    OpenStructureRow,
    Order,
    Proposal,
    Session,
)


async def the_account(session: AsyncSession) -> Account | None:
    """The account row. There is exactly one (hard rule #7), or none before boot."""
    row: Account | None = await session.scalar(select(Account).order_by(Account.id).limit(1))
    return row


async def latest_equity(session: AsyncSession, *, account_id: int) -> EquitySnapshot | None:
    """The most recent equity snapshot — the API's view of "now".

    The API reads the **journal**, never the broker: a dashboard that called
    Alpaca would put the API on the trading path, which hard rule #6 forbids, and
    would burn the 200/min budget the worker needs. So "now" here means "as of the
    worker's last cycle", and `/health` reports that age so a stale number is
    visibly stale rather than quietly wrong.
    """
    row: EquitySnapshot | None = await session.scalar(
        select(EquitySnapshot)
        .where(EquitySnapshot.account_id == account_id)
        .order_by(EquitySnapshot.ts.desc())
        .limit(1)
    )
    return row


async def equity_curve(
    session: AsyncSession, *, account_id: int, since: datetime | None = None, limit: int = 2000
) -> list[EquitySnapshot]:
    """Equity points, oldest first, optionally from `since`.

    Ordered DESC in SQL and reversed in Python: the index on
    `(account_id, ts DESC)` makes "the most recent N" cheap, where an ASC order
    with a limit would take the *oldest* N and need a full scan to find the end.
    """
    stmt: Select[tuple[EquitySnapshot]] = (
        select(EquitySnapshot)
        .where(EquitySnapshot.account_id == account_id)
        .order_by(EquitySnapshot.ts.desc())
        .limit(limit)
    )
    if since is not None:
        stmt = stmt.where(EquitySnapshot.ts >= since)
    rows = list((await session.scalars(stmt)).all())
    return list(reversed(rows))


async def open_structures(session: AsyncSession) -> list[OpenStructureRow]:
    """Everything the book currently holds, newest first."""
    return list(
        (
            await session.scalars(
                select(OpenStructureRow)
                .where(OpenStructureRow.status == "open")
                .order_by(OpenStructureRow.opened_at.desc())
            )
        ).all()
    )


async def recent_cycles(session: AsyncSession, *, limit: int = 50) -> list[Cycle]:
    """The decision log, newest first. Summary only — no proposals joined."""
    return list(
        (
            await session.scalars(
                select(Cycle).order_by(Cycle.started_at.desc()).limit(limit)
            )
        ).all()
    )


async def cycle_detail(session: AsyncSession, cycle_id: int) -> Cycle | None:
    """One cycle with its proposals, their legs and **all twelve verdicts**.

    Passes included, because that is the whole design (§3): "did any of this ever
    actually fire?" has to be answerable from the table rather than from memory,
    and a detail view that showed only failures would quietly re-introduce the
    short-circuit the kernel deliberately does not do.
    """
    # `scalar()` is typed `Any`, so the annotation is what makes the return type
    # real rather than something mypy waves through.
    row: Cycle | None = await session.scalar(
        select(Cycle)
        .where(Cycle.id == cycle_id)
        .options(
            selectinload(Cycle.proposals).selectinload(Proposal.verdicts),
            selectinload(Cycle.proposals).selectinload(Proposal.legs),
        )
    )
    return row


async def gate_stats(session: AsyncSession) -> list[tuple[int, str, int, int]]:
    """`(gate_no, name, passes, failures)` per gate, ascending.

    Aggregated in SQL rather than by loading verdicts and counting in Python:
    twelve rows per proposal adds up fast, and `ix_gate_verdicts_gate_passed`
    exists for exactly this query.

    **A gate that never passes is as broken as one that never fires** (§5.2), so
    both counts are returned rather than a rejection tally — a rejection count
    alone cannot tell those two apart.
    """
    rows = await session.execute(
        select(
            GateVerdictRow.gate_no,
            func.min(GateVerdictRow.name),
            func.count().filter(GateVerdictRow.passed.is_(True)),
            func.count().filter(GateVerdictRow.passed.is_(False)),
        )
        .group_by(GateVerdictRow.gate_no)
        .order_by(GateVerdictRow.gate_no)
    )
    return [(no, name or "", int(ok), int(bad)) for no, name, ok, bad in rows]


async def control_flags(session: AsyncSession) -> list[ControlFlag]:
    return list((await session.scalars(select(ControlFlag).order_by(ControlFlag.name))).all())


async def today_session(session: AsyncSession, *, account_id: int) -> Session | None:
    row: Session | None = await session.scalar(
        select(Session)
        .where(Session.account_id == account_id)
        .order_by(Session.trading_date.desc())
        .limit(1)
    )
    return row


async def open_risk(session: AsyncSession) -> Decimal:
    """Total max loss across open structures — the number Gate 6 reasons about.

    `COALESCE` because `SUM` over zero rows is NULL, and an empty book should
    report `0`, not `null`. The dashboard would render "no risk" either way; the
    difference is whether a client has to know that.
    """
    total = await session.scalar(
        select(func.coalesce(func.sum(OpenStructureRow.max_loss), 0)).where(
            OpenStructureRow.status == "open"
        )
    )
    return Decimal(str(total or 0))


async def latest_market_snapshots(session: AsyncSession) -> list[MarketSnapshotRow]:
    """The most recent market read **per underlying** — what the router last saw.

    `DISTINCT ON` is Postgres-specific and is the reason this is one query rather
    than a window-function subquery: it keeps the first row of each `underlying`
    group after the `ORDER BY`, so ordering by `cycle_id DESC` inside the group
    yields the newest snapshot for each symbol directly. The leading `ORDER BY`
    columns must match the `DISTINCT ON` columns, which is why `underlying` is
    ordered first and the result is re-sorted for display afterwards.

    **Per underlying, deliberately.** `cycles.regime` holds a single value that
    the entry loop overwrites once per symbol, so it reports whichever underlying
    happened to be sensed last. This table is written once per symbol per cycle,
    so it is the only place the dashboard can honestly show SPY and QQQ as the
    separate reads they are.
    """
    rows = await session.scalars(
        select(MarketSnapshotRow)
        .distinct(MarketSnapshotRow.underlying)
        .order_by(MarketSnapshotRow.underlying, MarketSnapshotRow.cycle_id.desc())
    )
    return sorted(rows.all(), key=lambda r: r.underlying)


async def recent_orders(session: AsyncSession, *, limit: int = 50) -> list[Order]:
    """Every ticket the router sent, newest first.

    Entries, resting profit targets and closes all land here, distinguished by
    `intent` — so "did the §2.6 target actually get placed?" is answerable from
    the same table that records the entry it belongs to.
    """
    return list(
        (
            await session.scalars(
                select(Order).order_by(Order.submitted_at.desc(), Order.id.desc()).limit(limit)
            )
        ).all()
    )
