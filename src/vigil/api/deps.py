"""Dependencies: the database session, and the bearer guard on mutating routes.

**Hard rule #10 — mutating routes carry a bearer token from the first commit.**
Not "auth later". A public endpoint that can flatten a trading account is real
attack surface even on paper, and the version of this file that ships without it
is the version that gets deployed.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from vigil.db.session import get_session
from vigil.settings import REPO_ROOT


class ControlTokenMissing(RuntimeError):
    """Raised when a mutating route is reached with no token configured."""


def control_token() -> str:
    """The bearer token for mutating routes, from `API_CONTROL_TOKEN`.

    Read here rather than through `vigil.settings` on purpose. `load_settings()`
    asserts paper mode and demands Alpaca credentials, which this service has no
    business holding — it never constructs a broker client. Keeping the two apart
    is what lets the API run with an Alpaca-free environment.

    An unset or blank token raises rather than defaulting to "no auth required".
    Failing closed on a missing secret is the whole point of the setting: an
    operator who forgot to generate one should get 503 on `/api/control/flatten`,
    not an open control plane.
    """
    load_dotenv(REPO_ROOT / ".env")
    token = os.getenv("API_CONTROL_TOKEN", "").strip()
    if not token:
        raise ControlTokenMissing(
            "API_CONTROL_TOKEN is unset. Mutating control routes are refused "
            "rather than served without authentication (hard rule #10). "
            "Generate one with: openssl rand -hex 32"
        )
    return token


async def db_session() -> AsyncIterator[AsyncSession]:
    """One session per request, committed on success.

    Reuses the worker's factory, so the API cannot accidentally acquire different
    transaction semantics from the process it is reporting on.
    """
    async with get_session() as session:
        yield session


async def require_control_token(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """`Authorization: Bearer <token>` or nothing happens.

    `secrets.compare_digest` rather than `==`: string equality short-circuits on
    the first differing byte, so its runtime leaks how many leading characters
    were right. That turns a 256-bit token into something guessable one byte at a
    time. The cost of the constant-time compare is nil and the argument for `==`
    is nonexistent.

    The 503 branch matters as much as the 401. A missing server-side token is an
    operator error, not a client error, and answering 401 would tell the caller
    to go find a better credential when none could ever work.
    """
    try:
        expected = control_token()
    except ControlTokenMissing as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    scheme, _, presented = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not presented:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Expected an `Authorization: Bearer <token>` header.",
            # RFC 6750: tell the client which scheme to use rather than leaving it
            # to guess from a bare 401.
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not secrets.compare_digest(presented, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid control token."
        )


Db = Annotated[AsyncSession, Depends(db_session)]
Authed = Depends(require_control_token)
