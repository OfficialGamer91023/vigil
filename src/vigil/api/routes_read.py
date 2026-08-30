"""Read routes. Public, so a judge can simply visit the URL (§7).

Public is a decision, not an oversight: everything here is a *paper* account's
own decision log, and requiring a token to read it would mean the demo URL shows
a login box. Mutating routes are a different question entirely — see
`routes_control.py`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from vigil.api.deps import Db, db_session
from vigil.api.schemas import (
    ControlFlagOut,
    CycleDetailOut,
    CycleOut,
    EquityPoint,
    GateStat,
    Health,
    MarketSnapshotOut,
    OrderOut,
    StateOut,
    StructureOut,
)
from vigil.control import FLATTEN_FLAG, HALT_FLAG
from vigil.db.repositories import reporting as R

# Imported by name so a test can substitute it: the streaming generator runs
# after the response has started, outside the request's dependency scope, so it
# cannot use the `db_session` dependency and has to acquire sessions itself.
from vigil.db.session import get_session

router = APIRouter()


@router.get("/health", response_model=Health, tags=["ops"])
async def health(db: Db) -> Health:
    """Heartbeat. **The age of the last cycle is the answer**, not the 200.

    This service is designed to run while the worker is stopped, so a route that
    resolved would report "ok" for a completely dead agent. `last_cycle_age_seconds`
    is what a monitor should alert on: past roughly 20 minutes during a session,
    the loop is not turning.
    """
    cycles = await R.recent_cycles(db, limit=1)
    last = cycles[0] if cycles else None
    age = None
    if last is not None:
        # `started_at` is timestamptz, so it arrives aware; comparing against an
        # aware UTC `now` avoids the naive/aware TypeError that only shows up in
        # production where the column is populated.
        age = (datetime.now(UTC) - last.started_at).total_seconds()

    return Health(
        status="ok",
        database=True,
        last_cycle_at=last.started_at if last else None,
        last_cycle_kind=last.kind if last else None,
        last_cycle_age_seconds=age,
        halted=any(f.name == HALT_FLAG and f.active for f in await R.control_flags(db)),
        flatten_requested=any(
            f.name == FLATTEN_FLAG and f.active for f in await R.control_flags(db)
        ),
    )


@router.get("/api/state", response_model=StateOut, tags=["read"])
async def state(db: Db) -> StateOut:
    """Account, book, risk and flags — everything the desk page renders."""
    account = await R.the_account(db)
    flags = {f.name: f.active for f in await R.control_flags(db)}
    structures = await R.open_structures(db)

    snapshot = None
    session_row = None
    if account is not None:
        snapshot = await R.latest_equity(db, account_id=account.id)
        session_row = await R.today_session(db, account_id=account.id)

    return StateOut(
        account_id=account.alpaca_account_id if account else None,
        equity=snapshot.equity if snapshot else None,
        day_pnl=snapshot.day_pnl if snapshot else None,
        # From the structures table rather than the snapshot, so an empty book
        # reports 0 even before the worker has written its first equity row.
        open_risk=await R.open_risk(db),
        net_dollar_delta=snapshot.net_dollar_delta if snapshot else None,
        open_structures=[StructureOut.model_validate(s) for s in structures],
        halted=flags.get(HALT_FLAG, False),
        flatten_requested=flags.get(FLATTEN_FLAG, False),
        trading_date=session_row.trading_date if session_row else None,
        as_of=snapshot.ts if snapshot else None,
    )


@router.get("/api/equity", response_model=list[EquityPoint], tags=["read"])
async def equity(
    db: Db,
    since: datetime | None = Query(default=None, description="ISO-8601, inclusive"),
    limit: int = Query(default=2000, ge=1, le=10000),
) -> list[EquityPoint]:
    account = await R.the_account(db)
    if account is None:
        return []
    rows = await R.equity_curve(db, account_id=account.id, since=since, limit=limit)
    return [EquityPoint.model_validate(r) for r in rows]


@router.get("/api/cycles", response_model=list[CycleOut], tags=["read"])
async def cycles(db: Db, limit: int = Query(default=50, ge=1, le=500)) -> list[CycleOut]:
    return [CycleOut.model_validate(c) for c in await R.recent_cycles(db, limit=limit)]


@router.get("/api/cycles/{cycle_id}", response_model=CycleDetailOut, tags=["read"])
async def cycle(db: Db, cycle_id: int) -> CycleDetailOut:
    """One cycle with every proposal and **all twelve verdicts, passes included**.

    This route is the argument the whole project makes: the kernel does not
    short-circuit, so "was this trade allowed, and by what?" is answerable from
    the journal rather than from a log grep.
    """
    row = await R.cycle_detail(db, cycle_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No cycle {cycle_id}.")
    return CycleDetailOut.model_validate(row)


@router.get("/api/gates/stats", response_model=list[GateStat], tags=["read"])
async def gates(db: Db) -> list[GateStat]:
    return [
        GateStat(gate_no=no, name=name, passed=ok, failed=bad)
        for no, name, ok, bad in await R.gate_stats(db)
    ]


@router.get("/api/market", response_model=list[MarketSnapshotOut], tags=["read"])
async def market(db: Db) -> list[MarketSnapshotOut]:
    """The router's last read of each underlying — spot, trend, IV, RV, VRP.

    The signal side of the story. `/api/cycles` says what the agent *decided*;
    this says what it decided *from*, per symbol, which the single `regime`
    column on a cycle cannot express when the universe holds more than one name.
    """
    return [MarketSnapshotOut.model_validate(m) for m in await R.latest_market_snapshots(db)]


@router.get("/api/orders", response_model=list[OrderOut], tags=["read"])
async def orders(db: Db, limit: int = Query(default=50, ge=1, le=500)) -> list[OrderOut]:
    """Entries, resting targets and closes, newest first."""
    return [OrderOut.model_validate(o) for o in await R.recent_orders(db, limit=limit)]


@router.get("/api/control/flags", response_model=list[ControlFlagOut], tags=["read"])
async def flags(db: Db) -> list[ControlFlagOut]:
    """Reading the flags is public; **setting** them is not (`routes_control.py`)."""
    return [ControlFlagOut.model_validate(f) for f in await R.control_flags(db)]


# --------------------------------------------------------------------------- #
# SSE
# --------------------------------------------------------------------------- #

STREAM_POLL_SECONDS = 2.0


# Tripped by the ASGI lifespan on shutdown (`main.lifespan`). A bare `while True`
# generator never returns, so uvicorn's graceful shutdown — which waits on the
# lifespan, which cannot complete until every open response finishes — hangs for
# as long as a browser tab is left open. This event is the shutdown signal every
# open stream watches so it can return promptly instead of pinning the process.
_shutdown = asyncio.Event()


def signal_shutdown() -> None:
    """Ask every open SSE generator to finish and return. Called from the lifespan."""
    _shutdown.set()


@router.get("/api/stream", tags=["read"])
async def stream(request: Request) -> StreamingResponse:
    """Server-Sent Events: pushes a frame whenever a new cycle lands.

    **Polling the journal rather than subscribing to Redis pub/sub**, which is
    what PLAN §7 sketched. Redis is not wired yet, and more to the point the plan
    itself (§2.2) says the optional layers may cost a cache and a dashboard feed
    but never a position — so the demo URL should not be the one thing that needs
    Redis running. A 2-second poll of an indexed `ORDER BY started_at DESC LIMIT 1`
    is nothing next to a 5-minute cycle cadence. If Redis lands, this becomes a
    subscription and the wire format does not change.

    Each iteration opens its **own** session rather than holding one for the life
    of the stream. A connection held open for hours is a pooled connection nobody
    else can have, and `pool_size=5` means five such clients would stall the
    worker's writes.
    """

    async def events() -> AsyncIterator[str]:
        last_seen: int | None = None
        # A comment frame up front: it defeats proxy buffering and tells the
        # browser the stream is alive before the first real event, which on a
        # quiet market could otherwise be many minutes away.
        yield ": vigil stream open\n\n"
        # Two exit conditions, not one: `_shutdown` for a graceful app stop, and
        # `is_disconnected()` for the tab that simply went away. Without the second,
        # a closed tab would keep polling the database every 2s until the process
        # dies — a slow leak that only shows up under a demo full of reconnects.
        while not _shutdown.is_set():
            if await request.is_disconnected():
                break
            async with get_session() as db:
                rows = await R.recent_cycles(db, limit=1)
                if rows and rows[0].id != last_seen:
                    last_seen = rows[0].id
                    payload = CycleOut.model_validate(rows[0]).model_dump(mode="json")
                    yield f"event: cycle\ndata: {json.dumps(payload)}\n\n"
                else:
                    # A heartbeat comment, not an event: it keeps intermediaries
                    # from timing the connection out without the client having to
                    # filter no-op messages.
                    yield ": keepalive\n\n"
            # An interruptible sleep: wake the instant shutdown is signalled rather
            # than holding the lifespan open for up to a full poll interval. The
            # timeout is the normal path; the event set is the shutdown path.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(_shutdown.wait(), timeout=STREAM_POLL_SECONDS)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Nginx buffers proxied responses by default, which holds SSE frames
            # until the buffer fills — the stream looks dead, then bursts.
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["db_session", "router"]
