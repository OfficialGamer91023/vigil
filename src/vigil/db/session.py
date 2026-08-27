"""Async engine and session factory.

**Async because the worker is async.** The trading loop is driven in-process by
`vigil.worker.schedule` (not by arq — see that module for why), and a blocking
database call inside a cycle would stall the event loop that is also waiting on
the broker. During a manage sweep that is the difference between a 15:40 flatten
firing at 15:40 and firing whenever the write finished.

The URL is read from `DATABASE_URL` and normalised to the `asyncpg` driver, so a
plain `postgresql://` copied from anywhere still works rather than failing with
SQLAlchemy's "dialect does not support async" error at the first query.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

DEFAULT_URL = "postgresql+asyncpg://localhost/vigil"


def database_url() -> str:
    """The async database URL, normalised.

    `postgresql://` and `postgres://` are both accepted and rewritten: they are
    what Postgres tooling, Docker and hosting providers hand you, and silently
    working with all three beats a stack trace at 09:31.
    """
    raw = os.getenv("DATABASE_URL", DEFAULT_URL)
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg://"):
        if raw.startswith(prefix):
            return raw
    if raw.startswith("postgres://"):
        return "postgresql+asyncpg://" + raw[len("postgres://"):]
    if raw.startswith("postgresql://"):
        return "postgresql+asyncpg://" + raw[len("postgresql://"):]
    return raw


@lru_cache(maxsize=1)
def engine() -> AsyncEngine:
    """One engine per process — it owns the connection pool.

    `pool_pre_ping` costs a round trip per checkout and buys back the case that
    actually happens on a small VM: the database restarts overnight, every
    pooled connection is dead, and without the ping the first cycle of the day
    fails instead of reconnecting.
    """
    return create_async_engine(database_url(), pool_pre_ping=True, pool_size=5, max_overflow=5)


@lru_cache(maxsize=1)
def session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine(), expire_on_commit=False, class_=AsyncSession)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """A session with commit-on-success, rollback-on-error.

    `expire_on_commit=False` on the factory is deliberate: without it, reading an
    attribute off an object after commit triggers a fresh SELECT, which inside an
    async context is a lazy-load on a closed session — the classic async
    SQLAlchemy footgun. Journal rows are read after commit constantly.
    """
    async with session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
