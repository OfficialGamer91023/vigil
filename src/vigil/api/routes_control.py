"""Control routes. **Every one of these requires the bearer token** (hard rule #10).

What these routes do and do not do is the whole point of §2.2. They write a row.
They do not call Alpaca, they do not cancel an order, they do not close a
position. The worker reads the row at the top of its next cycle and acts. That
one-way channel is what lets the worker keep trading correctly while this service
is stopped — a stopped API simply never sets a flag.

So `POST /api/control/flatten` is honest when it says the flag is set and the
worker will act next cycle. If you need positions gone *now* and the worker is
wedged, `make flatten` runs the guarded CLI path directly and does not involve
this service at all. Two paths, deliberately, because the fast one must not
depend on a web server being up.
"""

from __future__ import annotations

from fastapi import APIRouter, Body

from vigil.api.deps import Authed, Db
from vigil.api.schemas import ControlResult
from vigil.control import FLATTEN_FLAG, HALT_FLAG
from vigil.db.repositories import journal as J

# `dependencies=[Authed]` on the router, not on each route. Applying auth at the
# router means a route added later is authenticated by default — the failure mode
# of per-route decorators is a new endpoint that silently ships without one.
router = APIRouter(prefix="/api/control", tags=["control"], dependencies=[Authed])

NEXT_CYCLE = "the worker's next cycle"


@router.post("/halt", response_model=ControlResult)
async def halt(db: Db, reason: str | None = Body(default=None, embed=True)) -> ControlResult:
    """Stop **new entries**. Management keeps running.

    Deliberately not a full stop: a halt that also froze position management
    would leave open structures unattended through whatever caused the halt,
    which is worse than the condition itself. The 15:40 flatten still fires.
    """
    await J.set_flag(db, HALT_FLAG, active=True, set_by="api", reason=reason)
    return ControlResult(
        flag=HALT_FLAG, active=True, effective_from=NEXT_CYCLE,
        detail="New entries stopped. Position management and the 15:40 flatten continue.",
    )


@router.post("/unhalt", response_model=ControlResult)
async def unhalt(db: Db, reason: str | None = Body(default=None, embed=True)) -> ControlResult:
    """Resume entries — the documented human-in-the-loop resume path (§5.2).

    Gate 4's "full halt, human required" is an escape hatch, not a failure of
    autonomy, and an escape hatch with no way out is just a trap. This is the way
    out, and it is drilled rather than assumed.
    """
    await J.set_flag(db, HALT_FLAG, active=False, set_by="api", reason=reason)
    return ControlResult(
        flag=HALT_FLAG, active=False, effective_from=NEXT_CYCLE,
        detail="Entries permitted again, subject to the gates as usual.",
    )


@router.post("/flatten", response_model=ControlResult)
async def flatten(db: Db, reason: str | None = Body(default=None, embed=True)) -> ControlResult:
    """Request cancel-all + close-all. The **worker** performs it, not this service.

    Pre-empts whatever cycle was due rather than queueing behind it. The flag is
    left active afterwards on purpose: clearing it here would be this service
    asserting an outcome it cannot observe, and a flatten that half-completed
    would look finished.

    **The worker clears it, and only once it has seen an empty book.** Closes go
    out as limit orders (hard rule #5), so the cycle that submits them cannot
    confirm they filled; the flag survives until a later cycle reconciles against
    the broker and finds nothing left. That means a flatten whose closes never
    filled keeps pre-empting rather than quietly letting the agent trade again —
    and it means normal operation resumes on its own, without anyone having to
    remember to call `/unflatten`.
    """
    await J.set_flag(db, FLATTEN_FLAG, active=True, set_by="api", reason=reason)
    return ControlResult(
        flag=FLATTEN_FLAG, active=True, effective_from=NEXT_CYCLE,
        detail=(
            "Flatten requested; it pre-empts the next scheduled cycle. "
            "For an immediate close that does not depend on the worker, use `make flatten`."
        ),
    )


@router.post("/unflatten", response_model=ControlResult)
async def unflatten(db: Db, reason: str | None = Body(default=None, embed=True)) -> ControlResult:
    """Withdraw a flatten request that has not been acted on yet.

    Exists because the flag is sticky. Without this, a mis-click at 10:00 would
    keep pre-empting every cycle for the rest of the day.
    """
    await J.set_flag(db, FLATTEN_FLAG, active=False, set_by="api", reason=reason)
    return ControlResult(
        flag=FLATTEN_FLAG, active=False, effective_from=NEXT_CYCLE,
        detail="Flatten request withdrawn. Anything already closed stays closed.",
    )
