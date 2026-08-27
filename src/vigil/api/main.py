"""The API service. `uvicorn vigil.api.main:app`.

**It never trades** (hard rule #6). It reads the journal and writes control flags.
The worker must run correctly with this process, the frontend and Redis all
stopped — which is why nothing here imports the broker, the router or the kernel,
and why `tests/test_api_isolation.py` asserts that by walking the import graph.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from vigil.api import page, routes_control, routes_read
from vigil.api.deps import ControlTokenMissing
from vigil.logging import configure, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging on start; dispose the engine on stop.

    Disposing matters in a container: uvicorn's graceful shutdown waits on the
    ASGI lifespan, and pooled asyncpg connections left open hold server-side
    sessions until Postgres times them out. On a repeated `compose up` that is how
    a small `max_connections` gets exhausted by a service nobody is using.
    """
    configure()
    log.info("api.start")
    yield
    from vigil.db.session import engine

    await engine().dispose()
    log.info("api.stop")


app = FastAPI(
    title="Vigil",
    summary="Read the journal, write the control flags. Never trades.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(ControlTokenMissing)
async def _no_token(request: Request, exc: ControlTokenMissing) -> JSONResponse:
    """A misconfigured server is a 503, never a silently open route.

    `deps.require_control_token` already converts this to an HTTPException, so
    this handler is the backstop for any *future* code path that calls
    `control_token()` outside a dependency. Fail-closed twice rather than
    discover the gap later.
    """
    return JSONResponse(status_code=503, content={"detail": str(exc)})


app.include_router(routes_read.router)
app.include_router(routes_control.router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def desk() -> HTMLResponse:
    """The desk page (§8). One server-rendered page, no build step, no CDN.

    §11.1 cuts the Next.js dashboard by default: the submission requires *a
    working URL*, and a page that cannot fail is worth more on the day than a
    prettier one that has a deploy step. This is that page — it fetches
    `/api/state` and subscribes to `/api/stream`, and it is entirely
    self-contained so there is no third party between a judge and the demo.
    """
    return HTMLResponse(page.HTML)
