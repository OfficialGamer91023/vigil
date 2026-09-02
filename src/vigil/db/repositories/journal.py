"""Journal writes. **All database access lives here, never in strategy or risk.**

CLAUDE.md: strategy and risk code takes plain objects, not sessions — the kernel
has to stay pure and testable without a database. So the translation between the
frozen dataclasses in `vigil.domain` and the ORM rows in `vigil.db.models`
happens in exactly one place, and it is this one.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from vigil.data.occ import parse_occ
from vigil.db.models import (
    Account,
    ControlFlag,
    Cycle,
    EquitySnapshot,
    Fill,
    GateVerdictRow,
    LlmMemo,
    MarketSnapshotRow,
    OpenStructureRow,
    Order,
    Proposal,
    ProposalLeg,
    Session,
)
from vigil.domain import KernelDecision, OpenStructure, PortfolioState, TradeProposal


def _finite_or_none(value: Decimal) -> Decimal | None:
    """Map an unbounded max profit to SQL NULL.

    A long strangle's max profit is `Decimal("Infinity")` (see
    `TradeProposal.max_profit_per_contract`), and Postgres `NUMERIC` cannot store
    it. **NULL is the honest encoding**: it says "unbounded", where a large
    stand-in would say "we measured this" and be wrong. Every reader of
    `proposals.max_profit` therefore has to handle NULL, which is the point.
    """
    return value if value.is_finite() else None


async def record_proposal(
    session: AsyncSession,
    proposal: TradeProposal,
    decision: KernelDecision,
    *,
    cycle_id: int,
) -> Proposal:
    """Persist a proposal, its legs, and **every** gate verdict.

    Passes are stored alongside rejections because §5 requires the full record:
    "did any of this ever actually fire?" has to be answerable from the table
    rather than from memory, and a table holding only failures cannot answer it.

    Written in one transaction so a proposal can never exist with a partial set
    of verdicts — the unique constraint on `(proposal_id, gate_no)` guards the
    same invariant from the other side.
    """
    row = Proposal(
        cycle_id=cycle_id,
        structure_type=proposal.structure.value,
        underlying=proposal.underlying,
        expiry=proposal.expiry,
        spot=proposal.spot,
        contracts=proposal.contracts,
        net_credit=proposal.net_credit,
        width=proposal.width,
        max_loss=proposal.max_loss,
        max_profit=_finite_or_none(proposal.max_profit),
        dollar_delta=proposal.dollar_delta,
        client_order_id=proposal.client_order_id,
        limit_price=proposal.limit_price,
        regime=proposal.regime.value if proposal.regime else None,
        rationale=proposal.rationale,
        approved=decision.approved,
    )
    session.add(row)
    await session.flush()          # assigns row.id without ending the transaction

    for leg in proposal.legs:
        occ = parse_occ(leg.symbol)
        session.add(ProposalLeg(
            proposal_id=row.id,
            symbol=leg.symbol,
            ratio_qty=leg.ratio_qty,
            is_short=leg.is_short,
            strike=occ.strike,
            is_put=occ.is_put,
            bid=leg.bid,
            ask=leg.ask,
            delta=Decimal(str(leg.delta)),
            open_interest=leg.open_interest,
        ))

    for v in decision.verdicts:
        session.add(GateVerdictRow(
            proposal_id=row.id,
            gate_no=v.number,
            name=v.name,
            passed=v.passed,
            reason=v.reason or None,
            detail=dict(v.detail) or None,
        ))

    await session.flush()
    return row


# --------------------------------------------------------------------------- #
# Account / session / cycle — the spine the runner writes as it goes
# --------------------------------------------------------------------------- #

async def ensure_account(
    session: AsyncSession, *, alpaca_account_id: str, starting_equity: Decimal
) -> Account:
    """Get-or-create the one account row. Idempotent across restarts.

    `starting_equity` is written **only on creation**. It is the denominator of
    every "return since inception" figure in the report and the deck; refreshing
    it on each startup would silently rebase the competition's headline number
    every time the worker restarted.
    """
    existing = await session.scalar(
        select(Account).where(Account.alpaca_account_id == alpaca_account_id)
    )
    if existing is not None:
        return existing
    row = Account(alpaca_account_id=alpaca_account_id, starting_equity=starting_equity)
    session.add(row)
    await session.flush()
    return row


async def open_session(
    session: AsyncSession,
    *,
    account_id: int,
    trading_date: date,
    opening_equity: Decimal | None = None,
) -> Session:
    """Get-or-create today's session row.

    Get-or-create rather than insert because `uq_session_day` makes a second
    insert an error, and a worker restarted at 11:00 must rejoin the day it is
    already in — not crash, and not start a second one. Opening equity is
    likewise written once: the 09:45 reading is the day's baseline, and a restart
    at 14:00 must not redefine it as whatever the book happens to be worth then.
    """
    row = await session.scalar(
        select(Session).where(
            Session.account_id == account_id, Session.trading_date == trading_date
        )
    )
    if row is None:
        row = Session(
            account_id=account_id, trading_date=trading_date, opening_equity=opening_equity
        )
        session.add(row)
        await session.flush()
    elif row.opening_equity is None and opening_equity is not None:
        row.opening_equity = opening_equity
        await session.flush()
    return row


async def close_session(
    session: AsyncSession, *, session_id: int, closing_equity: Decimal
) -> None:
    await session.execute(
        update(Session).where(Session.id == session_id).values(closing_equity=closing_equity)
    )


async def start_cycle(
    session: AsyncSession, *, session_id: int, kind: str
) -> Cycle:
    """Open a cycle row. Written **before** the work, not after.

    A cycle that crashes half way through is exactly the one worth having a row
    for: `finished_at IS NULL` is then a live query for "did anything die
    mid-cycle?", which a write-on-completion design cannot answer at all.
    """
    row = Cycle(session_id=session_id, kind=kind)
    session.add(row)
    await session.flush()
    return row


async def finish_cycle(
    session: AsyncSession,
    *,
    cycle_id: int,
    regime: str | None = None,
    cold_start: bool = False,
    notes: str | None = None,
) -> None:
    await session.execute(
        update(Cycle)
        .where(Cycle.id == cycle_id)
        .values(
            finished_at=func.now(), regime=regime, cold_start=cold_start, notes=notes
        )
    )


# --------------------------------------------------------------------------- #
# What the router did — orders, fills, structures
# --------------------------------------------------------------------------- #

async def record_order(
    session: AsyncSession,
    *,
    client_order_id: str,
    intent: str,
    limit_price: Decimal,
    qty: int,
    status: str,
    broker_order_id: str | None = None,
    structure_id: int | None = None,
    rung: int | None = None,
) -> Order:
    """Persist one ticket. The unique constraint on `client_order_id` is the point.

    Deliberately **not** guarded with a get-or-create: hard rule #9 says a retry
    must raise an integrity error rather than double-fill, and swallowing the
    collision here would move that guarantee out of the database and into a
    convention. Callers that legitimately re-observe an order use
    `upsert_order_status` instead.
    """
    row = Order(
        client_order_id=client_order_id,
        broker_order_id=broker_order_id,
        structure_id=structure_id,
        intent=intent,
        limit_price=limit_price,
        qty=qty,
        status=status,
        rung=rung,
    )
    session.add(row)
    await session.flush()
    return row


async def upsert_order_status(
    session: AsyncSession, *, client_order_id: str, status: str,
    broker_order_id: str | None = None,
) -> None:
    """Update a known ticket's status. Silent no-op if we never recorded it.

    A no-op rather than an error because reconciliation reads the broker's open
    orders, which legitimately include tickets from a previous process that this
    database may not have (§2.3 adopts orphans rather than rejecting them).
    """
    values: dict[str, object] = {"status": status}
    if broker_order_id is not None:
        values["broker_order_id"] = broker_order_id
    await session.execute(
        update(Order).where(Order.client_order_id == client_order_id).values(**values)
    )


async def upsert_order_status_by_broker_id(
    session: AsyncSession, *, broker_order_id: str, status: str,
) -> None:
    """Update a ticket's status keyed on the **broker** order id. Silent no-op if unknown.

    The polling loops (`_await_close_filled`, `_await_target_settled`, the target
    live-confirmation) hold the broker order id they are polling, not the
    `client_order_id`, so this is the key they can write against without threading
    the client id down through every helper. The sibling `upsert_order_status`
    keys on the client id for the reconcile scan, where the SDK order carries both.

    Both exist because the whole §5.2 defect was that this was *never called* — the
    order row froze at its submit-time status (`pending_new`) forever. Persisting
    on every poll is what makes the journal reflect the broker rather than a
    one-shot guess taken the instant the order was accepted.
    """
    await session.execute(
        update(Order)
        .where(Order.broker_order_id == broker_order_id)
        .values(status=status)
    )


async def record_fill(
    session: AsyncSession,
    *,
    order_id: int,
    filled_qty: int,
    filled_avg_price: Decimal,
    partial: bool,
) -> Fill:
    row = Fill(
        order_id=order_id,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
        partial=partial,
    )
    session.add(row)
    await session.flush()
    return row


async def record_structure(
    session: AsyncSession,
    structure: OpenStructure,
    *,
    proposal_id: int | None = None,
) -> OpenStructureRow:
    """Register an open structure, or refresh the one already registered.

    Keyed on `(underlying, expiry, status='open')`, which mirrors exactly how
    `reconcile.group_positions` groups broker legs. Using the same key on both
    sides is what stops the registry and the reconciler from disagreeing about
    how many structures are open — the number Gate 5 and Gate 6 both read.
    """
    row = await session.scalar(
        select(OpenStructureRow).where(
            OpenStructureRow.underlying == structure.underlying,
            OpenStructureRow.expiry == structure.expiry,
            OpenStructureRow.status == "open",
        )
    )
    if row is None:
        row = OpenStructureRow(
            proposal_id=proposal_id,
            underlying=structure.underlying,
            expiry=structure.expiry,
            structure_type=structure.structure.value if structure.structure else None,
            contracts=structure.contracts,
            net_credit=structure.net_credit,
            max_loss=structure.max_loss,
            has_resting_target=structure.has_resting_target,
        )
        session.add(row)
    else:
        # Broker truth wins on every field it knows about (§2.3). A structure
        # partially closed at the broker has fewer contracts than we registered,
        # and reporting the stale number would overstate open risk.
        row.contracts = structure.contracts
        row.max_loss = structure.max_loss
        row.has_resting_target = structure.has_resting_target
        if proposal_id is not None:
            row.proposal_id = proposal_id
    await session.flush()
    return row


async def close_structure(
    session: AsyncSession, *, underlying: str, expiry: date, reason: str
) -> None:
    await session.execute(
        update(OpenStructureRow)
        .where(
            OpenStructureRow.underlying == underlying,
            OpenStructureRow.expiry == expiry,
            OpenStructureRow.status == "open",
        )
        .values(status="closed", closed_at=func.now(), close_reason=reason)
    )


async def mark_absent_structures_closed(
    session: AsyncSession, live: tuple[OpenStructure, ...], *, reason: str
) -> int:
    """Close registry rows for structures the broker no longer reports.

    This is how a resting GTC profit target gets journalled at all. It fills
    while the worker is asleep between sweeps, so no code path ever *observes*
    the close — the only evidence is the position's absence on the next
    reconcile. Without this the registry would accumulate phantom open rows and
    Gate 5 would refuse entries against positions that no longer exist.
    """
    live_keys = {(s.underlying, s.expiry) for s in live}
    rows = (
        await session.scalars(
            select(OpenStructureRow).where(OpenStructureRow.status == "open")
        )
    ).all()
    closed = 0
    for row in rows:
        if (row.underlying, row.expiry) not in live_keys:
            row.status = "closed"
            row.closed_at = datetime.now(UTC)
            row.close_reason = reason
            closed += 1
    await session.flush()
    return closed


async def open_structure_rows(session: AsyncSession) -> list[OpenStructureRow]:
    return list(
        (
            await session.scalars(
                select(OpenStructureRow).where(OpenStructureRow.status == "open")
            )
        ).all()
    )


async def open_structure_id(
    session: AsyncSession, *, underlying: str, expiry: date
) -> int | None:
    """The id of the open structure row for `(underlying, expiry)`, or ``None``.

    Keyed exactly like `record_structure`/`close_structure` — `(underlying,
    expiry, status='open')` — so the manage-path order writers can stamp
    `orders.structure_id` with the row reconcile already created, without holding
    an ORM object. Returns ``None`` when no open row exists (an order for a
    structure not yet registered), which the caller records as an unlinked order
    rather than inventing a row.
    """
    row_id: int | None = await session.scalar(
        select(OpenStructureRow.id).where(
            OpenStructureRow.underlying == underlying,
            OpenStructureRow.expiry == expiry,
            OpenStructureRow.status == "open",
        )
    )
    return row_id


async def ensure_open_structure(
    session: AsyncSession,
    *,
    underlying: str,
    expiry: date,
    structure_type: str | None,
    contracts: int,
    net_credit: Decimal,
    max_loss: Decimal,
    proposal_id: int | None = None,
) -> int:
    """Register the structure of a just-filled entry, returning its row id.

    Get-or-create on the same `(underlying, expiry, status='open')` key
    `record_structure` uses, so it is idempotent with the reconcile that will
    refresh this row from broker truth on the next cycle (§2.3). It exists because
    the entry flow records an `open` order and its resting `target` *before* any
    structure row is created — reconcile only mints one on the following cycle —
    so those two tickets had nothing to link to and were journalled orphaned
    (`structure_id` NULL). Registering on the fill both closes that gap and lets
    Gate 5/6 count the position the moment it exists rather than a cycle late.

    Takes primitives rather than an `OpenStructure` because a `TradeProposal` is
    not one, and constructing a full `OpenStructure` (split short strikes, dollar
    delta) here would duplicate `reconcile.group_positions` for no gain — reconcile
    fills those fields in from broker truth next cycle regardless.
    """
    row = await session.scalar(
        select(OpenStructureRow).where(
            OpenStructureRow.underlying == underlying,
            OpenStructureRow.expiry == expiry,
            OpenStructureRow.status == "open",
        )
    )
    if row is None:
        row = OpenStructureRow(
            proposal_id=proposal_id,
            underlying=underlying,
            expiry=expiry,
            structure_type=structure_type,
            contracts=contracts,
            net_credit=net_credit,
            max_loss=max_loss,
            has_resting_target=True,  # the router rests a target on every fill (§2.6)
        )
        session.add(row)
    else:
        # Already registered (a same-underlying/expiry refresh): keep the row, do
        # not clobber broker-reconciled fields. Only fill provenance we now know.
        if proposal_id is not None:
            row.proposal_id = proposal_id
    await session.flush()
    return row.id


# --------------------------------------------------------------------------- #
# Equity, market snapshots, control flags
# --------------------------------------------------------------------------- #

async def record_equity(
    session: AsyncSession,
    *,
    account_id: int,
    state: PortfolioState,
    net_dollar_delta: Decimal | None = None,
) -> EquitySnapshot:
    """One point on the curve the dashboard draws and the deck quotes.

    `net_dollar_delta` overrides the value computed from `state`. A cycle that read
    no option chain — `manage`, `postclose` — carries every open structure at a
    placeholder 0 delta (reconcile has no greeks), so recording `state`'s delta
    there would zero a live exposure on the desk. Those cycles pass the last
    refreshed value forward instead; only the entry cycle, which senses a chain,
    records the freshly computed delta.
    """
    row = EquitySnapshot(
        account_id=account_id,
        equity=state.equity,
        day_pnl=state.day_pnl,
        open_risk=state.open_risk,
        net_dollar_delta=(
            state.net_dollar_delta if net_dollar_delta is None else net_dollar_delta
        ),
    )
    session.add(row)
    await session.flush()
    return row


async def last_net_dollar_delta(
    session: AsyncSession, *, account_id: int
) -> Decimal | None:
    """The most recent snapshot's dollar-delta, to carry forward on a chain-less cycle.

    `None` before the first snapshot of all — the caller then records `state`'s
    value (0 for a flat or freshly-adopted book), which is the honest answer.
    """
    delta: Decimal | None = await session.scalar(
        select(EquitySnapshot.net_dollar_delta)
        .where(EquitySnapshot.account_id == account_id)
        .order_by(EquitySnapshot.ts.desc())
        .limit(1)
    )
    return delta


async def peak_equity(session: AsyncSession, *, account_id: int) -> Decimal | None:
    """The high-water mark Gate 4 measures drawdown against.

    Read from the journal rather than held in process memory, which is the whole
    reason the column exists: a worker restarted after a drawdown would otherwise
    take the *current* equity as its peak and compute a drawdown of zero —
    disarming the one gate whose job is to stop the unrecoverable day.
    """
    high: Decimal | None = await session.scalar(
        select(func.max(EquitySnapshot.equity)).where(
            EquitySnapshot.account_id == account_id
        )
    )
    return high


async def record_market_snapshot(
    session: AsyncSession,
    *,
    cycle_id: int,
    underlying: str,
    spot: Decimal,
    iv_atm: Decimal | None = None,
    rv_annual: Decimal | None = None,
    vrp_pct: Decimal | None = None,
    iv_pct: Decimal | None = None,
    trend: Decimal | None = None,
) -> MarketSnapshotRow:
    """What the router actually saw. Persisted so a decision can be re-derived."""
    row = MarketSnapshotRow(
        cycle_id=cycle_id,
        underlying=underlying,
        spot=spot,
        iv_atm=iv_atm,
        rv_annual=rv_annual,
        vrp_pct=vrp_pct,
        iv_pct=iv_pct,
        trend=trend,
    )
    session.add(row)
    await session.flush()
    return row


async def daily_iv_history(session: AsyncSession, underlying: str) -> list[float]:
    """One measured ATM IV per trading day, oldest first — the accumulated series.

    Option 3's "accumulate forward" half (§4.3.1). There is no historical IV on
    the free tier, but the worker writes `iv_atm` to `market_snapshots` every
    cycle, so a real daily series accrues on its own; this reads it back for the
    IV-percentile seed (`signals.iv_seed.build_iv_history`).

    `DISTINCT ON (trading_date)` with the matching leading `ORDER BY` collapses the
    many snapshots a day to the **first** one — the open read — which is the most
    consistent daily representative (every session is sensed at 09:45, not every
    session reaches an entry cycle). Re-sorted to chronological afterwards because
    `DISTINCT ON` dictates the primary sort key, not the caller's preferred one.
    """
    rows = await session.execute(
        select(Session.trading_date, MarketSnapshotRow.iv_atm)
        .distinct(Session.trading_date)
        .join(Cycle, Cycle.id == MarketSnapshotRow.cycle_id)
        .join(Session, Session.id == Cycle.session_id)
        .where(
            MarketSnapshotRow.underlying == underlying,
            MarketSnapshotRow.iv_atm.isnot(None),
        )
        .order_by(Session.trading_date, Cycle.started_at)
    )
    pairs = sorted(rows.all(), key=lambda r: r[0])
    return [float(iv) for _, iv in pairs if iv is not None]


async def is_flag_active(session: AsyncSession, name: str) -> bool:
    """Read a control flag. A missing row is inactive — the safe default.

    The API writes these rows; the worker reads them. That one-way channel is the
    narrowest thing satisfying hard rule #6: the worker keeps trading correctly
    with the API stopped, because a stopped API simply never sets a flag.
    """
    row = await session.scalar(select(ControlFlag).where(ControlFlag.name == name))
    return bool(row and row.active)


async def set_flag(
    session: AsyncSession,
    name: str,
    *,
    active: bool,
    set_by: str | None = None,
    reason: str | None = None,
) -> ControlFlag:
    row = await session.scalar(select(ControlFlag).where(ControlFlag.name == name))
    if row is None:
        row = ControlFlag(name=name, active=active, set_by=set_by, reason=reason)
        session.add(row)
    else:
        row.active = active
        row.set_by = set_by
        row.reason = reason
    await session.flush()
    return row


async def record_llm_memo(
    session: AsyncSession,
    *,
    cycle_id: int,
    model: str,
    proposal_id: int | None = None,
    effort: str | None = None,
    latency_ms: int | None = None,
    input_tokens: int | None = None,
    cached_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    output_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    memo: str | None = None,
    fell_back: bool = False,
) -> LlmMemo:
    """The model's reasoning and its cost, including whether it was used at all.

    `fell_back` is persisted on every call, not only on failures, because the
    fallback *rate* is the headline reliability number (§6.3) and a table that
    only recorded successes could not compute it.
    """
    row = LlmMemo(
        cycle_id=cycle_id,
        proposal_id=proposal_id,
        model=model,
        effort=effort,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        memo=memo,
        fell_back=fell_back,
    )
    session.add(row)
    await session.flush()
    return row
