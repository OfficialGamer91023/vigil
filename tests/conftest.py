"""Shared fixtures.

Chain contracts are built from **raw API JSON**, not by hand-constructing SDK
models, so these tests exercise the same parsing path a live snapshot takes. A
fixture that bypassed parsing would pass while the real feed failed.

`make_contract` is a *factory fixture*: the fixture returns a function rather than
a value, so one test can build many differently-shaped contracts. conftest.py is
auto-discovered by pytest and must never be imported for its fixtures — the
module-level helpers below are plain functions and may be imported directly.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Protocol
from zoneinfo import ZoneInfo

import pytest
from alpaca.data.models.snapshots import OptionsSnapshot

from vigil.data.chain import Contract
from vigil.data.occ import parse_occ
from vigil.domain import Leg, PortfolioState, Structure, TradeProposal
from vigil.risk.context import KernelContext

_TS = datetime(2026, 8, 26, 14, 30, tzinfo=UTC).isoformat()

ET = ZoneInfo("America/New_York")
DEFAULT_EXPIRY = date(2026, 8, 27)
# 11:00 ET on a Wednesday: inside regular hours, past the opening window, well
# before the closing window. Chosen so Gate 11 is never the accidental failure.
DEFAULT_NOW = datetime(2026, 8, 26, 11, 0, tzinfo=ET)


# --------------------------------------------------------------------------- #
# Database safety — refuse to TRUNCATE anything but a dedicated test database
# --------------------------------------------------------------------------- #
#
# The DB-backed tests empty every journal table on the way in (`truncate_journal`)
# and once more when the run ends (`pytest_sessionfinish`). Pointed at the live desk
# database that is not a nuisance, it is data loss: on 2026-09-01 a verification run
# with DATABASE_URL aimed at the worker's Postgres wiped real trading history. So the
# suite runs *only* against a database whose name marks it disposable (ends in
# `_test`) and refuses everything else — a positive allowlist that fails closed, so
# an unrecognised database is treated as production, never the other way round.

_TEST_DB_URL = "postgresql+asyncpg://localhost/vigil_test"
_ALLOW_UNSAFE_DB_ENV = "VIGIL_TEST_ALLOW_UNSAFE_DB"


def _db_name(url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(url).path.lstrip("/")


def _database_is_disposable(url: str) -> bool:
    """Whether it is safe to TRUNCATE this database in the test suite.

    Safe means the name ends in `_test`, or the operator has set an explicit escape
    hatch. Everything else — including the production `vigil` database, whatever port
    it is on — is refused.
    """
    if os.environ.get(_ALLOW_UNSAFE_DB_ENV) == "1":
        return True
    return _db_name(url).endswith("_test")


def pytest_configure(config: object) -> None:
    """Abort the whole run *before* collection if the target DB is not disposable.

    Defaulting `DATABASE_URL` to `vigil_test` first keeps the common case (nothing
    set) safe with no ceremony; the guard then only ever fires when the suite has
    been pointed explicitly at a real database — the exact mistake worth stopping
    loudly, before a single table is truncated. This runs at one chokepoint so every
    TRUNCATE path (the per-file autouse fixtures and `pytest_sessionfinish`) is
    covered by it, not each independently.
    """
    os.environ.setdefault("DATABASE_URL", _TEST_DB_URL)

    from vigil.db.session import database_url

    url = database_url()
    if _database_is_disposable(url):
        return

    name = _db_name(url)
    looks_live = ":5433" in os.environ.get("DATABASE_URL", "") or name == "vigil"
    warning = (
        "  This is the LIVE desk database — running the suite here TRUNCATES every\n"
        "  journal table and destroys real trading history.\n"
        if looks_live
        else ""
    )
    raise pytest.UsageError(
        f"Refusing to run the destructive DB test suite against database {name!r}.\n"
        f"{warning}"
        f"  The suite only runs against a disposable database whose name ends in "
        f"'_test'.\n"
        f"  Use `make test` (targets vigil_test), or set DATABASE_URL to e.g. "
        f"{_TEST_DB_URL}.\n"
        f"  To override deliberately, set {_ALLOW_UNSAFE_DB_ENV}=1."
    )


# --------------------------------------------------------------------------- #
# Network isolation
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _no_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may reach OpenAI, the same way none may reach Alpaca.

    `build_manager()` reads OPENAI_API_KEY straight from the process environment,
    and any earlier test that called `load_settings()` (which runs `load_dotenv`)
    would have populated it for the rest of the pytest process. Stripping it here,
    autouse, makes the entry path build *no* client by default: a test that wants
    to exercise the model injects a fake `PortfolioManager` explicitly, and one
    that does not gets the deterministic path — which is exactly the production
    contract when the model is off (§6.3).
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


# --------------------------------------------------------------------------- #
# Chain contracts
# --------------------------------------------------------------------------- #

class MakeContract(Protocol):
    """The callable shape `make_contract` returns — keeps mypy meaningful in tests."""

    def __call__(
        self,
        symbol: str,
        *,
        delta: float | None = None,
        iv: float | None = None,
        bid: float | None = None,
        ask: float | None = None,
    ) -> Contract: ...


@pytest.fixture
def make_contract() -> MakeContract:
    def _make(
        symbol: str,
        *,
        delta: float | None = None,
        iv: float | None = None,
        bid: float | None = None,
        ask: float | None = None,
    ) -> Contract:
        raw: dict[str, object] = {}
        if delta is not None:
            # The feed sends all five greeks together; a partial object would not
            # be representative of anything the live endpoint returns.
            raw["greeks"] = {"delta": delta, "gamma": 0.01, "rho": 0.0, "theta": -0.5, "vega": 0.1}
        if iv is not None:
            raw["impliedVolatility"] = iv
        if bid is not None and ask is not None:
            raw["latestQuote"] = {"t": _TS, "bp": bid, "bs": 10, "ap": ask, "as": 10}
        return Contract(occ=parse_occ(symbol), snapshot=OptionsSnapshot(symbol, raw))

    return _make


# --------------------------------------------------------------------------- #
# Risk-kernel inputs
# --------------------------------------------------------------------------- #

def make_leg(
    symbol: str,
    *,
    short: bool = False,
    bid: str = "1.00",
    ask: str = "1.04",
    delta: float = -0.16,
    oi: int = 5000,
    ratio: int = 1,
) -> Leg:
    return Leg(
        occ=parse_occ(symbol),
        ratio_qty=ratio,
        is_short=short,
        bid=Decimal(bid),
        ask=Decimal(ask),
        delta=delta,
        open_interest=oi,
    )


@pytest.fixture
def put_credit_spread() -> TradeProposal:
    """SPY $1-wide put credit spread, 1 DTE, ~0.16 delta short, 20% credit.

    A baseline that passes all twelve gates, so each test can break exactly one
    thing and be certain the failure it sees is the one it caused.

    Sized at **1 contract**, and the reason is worth reading: Gate 2 would permit
    25 (max loss $80/contract against a $2,000 budget), but Gate 7 permits barely
    more than one. Net delta per spread is 0.16 - 0.11 = 0.05, and on a $765
    underlying that is 0.05 x 100 x 765.85 = ~$3,829 of dollar delta against a
    $5,000 portfolio limit. **Gate 7 binds ~15x harder than Gate 2 on directional
    structures** — see tests/test_gates.py::test_gate7_binds_far_harder_than_gate2.
    """
    return TradeProposal(
        structure=Structure.PUT_CREDIT_SPREAD,
        underlying="SPY",
        spot=Decimal("765.85"),
        expiry=DEFAULT_EXPIRY,
        legs=(
            make_leg("SPY260827P00761000", short=True, bid="0.50", ask="0.52", delta=-0.16),
            make_leg("SPY260827P00760000", short=False, bid="0.30", ask="0.32", delta=-0.11),
        ),
        contracts=1,
        net_credit=Decimal("0.20"),
        width=Decimal(1),
        client_order_id="vigil-test-0001",
        limit_price=Decimal("0.20"),
    )


@pytest.fixture
def flat_book() -> PortfolioState:
    """$100k, at its peak, flat on the day, nothing open."""
    return PortfolioState(
        equity=Decimal(100_000),
        peak_equity=Decimal(100_000),
        day_pnl=Decimal(0),
    )


@pytest.fixture
def ctx() -> KernelContext:
    return KernelContext(now=DEFAULT_NOW)


@pytest.fixture
def iron_condor() -> TradeProposal:
    """SPY $1-wide iron condor, 8 contracts, exactly delta-neutral.

    The multi-contract counterpart to `put_credit_spread`, and it has to be a
    condor to *be* multi-contract: Gate 7 binds at barely one contract on a
    directional structure, so any test that needs a partial fill (or any size at
    all) needs a book whose deltas cancel.

    Legs cancel by construction — short put −0.16 and short call +0.16 carry
    opposite signed ratios, as do the two longs — so `dollar_delta` is 0 and Gate
    7 is unconstrained. Max loss is **one** width minus the total credit, because
    the underlying cannot finish below the put spread and above the call spread
    at once: (1.00 − 0.20) × 100 × 8 = $640 against a $2,000 budget.
    """
    return TradeProposal(
        structure=Structure.IRON_CONDOR,
        underlying="SPY",
        spot=Decimal("765.85"),
        expiry=DEFAULT_EXPIRY,
        legs=(
            make_leg("SPY260827P00761000", short=True, bid="0.40", ask="0.42", delta=-0.16),
            make_leg("SPY260827P00760000", short=False, bid="0.28", ask="0.30", delta=-0.11),
            make_leg("SPY260827C00770000", short=True, bid="0.40", ask="0.42", delta=0.16),
            make_leg("SPY260827C00771000", short=False, bid="0.28", ask="0.30", delta=0.11),
        ),
        contracts=8,
        # Conservative: sell the bid, buy the ask -> (0.40 - 0.30) x 2 sides.
        net_credit=Decimal("0.20"),
        width=Decimal(1),
        client_order_id="vigil-test-cndr-0001",
        # The mid the ladder opens at: (0.41 - 0.29) x 2 sides.
        limit_price=Decimal("0.24"),
    )


@pytest.fixture
def debit_spread() -> TradeProposal:
    """SPY $1-wide call debit spread — the convexity sleeve's structure (§4.5).

    `net_credit` is **negative** because the package pays premium; `limit_price`
    stays a positive magnitude because that is what the broker takes. Max loss is
    the $0.41 debit, max profit the remaining $0.59 of width, so Gate 9's
    loss:profit ratio reads 0.69:1 and the credit floor does not apply.
    """
    return TradeProposal(
        structure=Structure.DEBIT_SPREAD,
        underlying="SPY",
        spot=Decimal("765.85"),
        expiry=DEFAULT_EXPIRY,
        legs=(
            make_leg("SPY260827C00766000", short=False, bid="0.60", ask="0.62", delta=0.35),
            make_leg("SPY260827C00767000", short=True, bid="0.21", ask="0.23", delta=0.29),
        ),
        contracts=1,
        # Conservative: pay the ask, sell the bid -> 0.62 - 0.21. Negative because
        # the package pays out.
        net_credit=Decimal("-0.41"),
        width=Decimal(1),
        client_order_id="vigil-test-debt-0001",
        # The mid: 0.61 - 0.22. A positive magnitude, per TradeProposal's contract.
        limit_price=Decimal("0.39"),
    )


@pytest.fixture
def long_strangle() -> TradeProposal:
    """SPY long strangle — the convexity sleeve (§4.5.1).

    Long-only, so `is_long_only` is True, `max_profit` is `Decimal("Infinity")`
    and `width` is 0. Gate 1 has a dedicated branch for exactly this shape: a
    call and a put have no width between them, and claiming one would have the
    risk arithmetic multiply by a fiction.

    Deltas cancel (+0.16 / −0.16), which is the entire reason the sleeve is a
    strangle rather than a debit spread — Gate 7 caps a directional structure at
    roughly one contract.
    """
    return TradeProposal(
        structure=Structure.LONG_STRANGLE,
        underlying="SPY",
        spot=Decimal("765.85"),
        expiry=DEFAULT_EXPIRY,
        legs=(
            make_leg("SPY260827C00769000", short=False, bid="0.80", ask="0.84", delta=0.16),
            make_leg("SPY260827P00761000", short=False, bid="0.80", ask="0.84", delta=-0.16),
        ),
        contracts=2,
        # Conservative: both asks. Negative because the package pays out.
        net_credit=Decimal("-1.68"),
        width=Decimal(0),
        client_order_id="vigil-test-strg-0001",
        limit_price=Decimal("1.64"),   # both mids
    )


# --------------------------------------------------------------------------- #
# The journal, between database-backed tests
# --------------------------------------------------------------------------- #

async def truncate_journal(session: object) -> None:
    """Empty every journal table. Run before each database-backed test.

    **The table list is derived from the mapped metadata, not hand-written.** It
    was spelled out as a literal in two separate files, so adding a table to
    `db/models.py` meant remembering to add it to two `TRUNCATE` strings that
    nothing checks — and a table missing from the list is one that silently
    accumulates rows across the whole suite. `sorted_tables` cannot fall behind
    the models, because it *is* the models.

    `alembic_version` is excluded deliberately: it is schema state, not journal
    state, and truncating it would make the next `alembic upgrade` try to replay
    a migration that has already run.
    """
    from sqlalchemy import text

    from vigil.db.models import Base
    from vigil.db.session import database_url

    # Defence in depth: `pytest_configure` already aborts the run against a
    # non-test database, but this is the function that actually destroys rows, so
    # it re-checks rather than trust that the guard ran (a stray direct import, a
    # future entry point that skips configure).
    assert _database_is_disposable(database_url()), (
        f"truncate_journal refused: {_db_name(database_url())!r} is not a test "
        f"database (name must end in '_test')"
    )

    names = ", ".join(
        t.name for t in Base.metadata.sorted_tables if t.name != "alembic_version"
    )
    await session.execute(  # type: ignore[attr-defined]
        text(f"TRUNCATE {names} RESTART IDENTITY CASCADE")
    )


def pytest_sessionfinish(session: object, exitstatus: object) -> None:
    """Leave the journal empty when the run ends.

    **Why this is a session hook and not a fixture teardown.** The per-test
    fixtures truncate on the way *in*, which is all test isolation needs — but it
    leaves the final test's rows sitting in the database after pytest exits. On a
    developer machine that database is also the one `make api` serves the desk
    page from, so a finished test run left the dashboard reporting a fake
    `acct-test` account at $110,000.

    That is worse than a cosmetic wrong number. `reporting.the_account()` takes
    the **lowest** account id on the assumption that hard rule #7 means there is
    exactly one; a leftover test account is created first, so once the real
    worker writes its own row the leftover shadows it permanently and the real
    account never appears on the page again.

    A fixture teardown cannot do this job: it would run after the engine-disposal
    fixture has already closed the pool, and opening a session then raises
    "attached to a different loop". A brand-new engine on a brand-new loop, once,
    after every test has finished, has no such ordering to lose.

    Failures here are swallowed on purpose. There may be no Postgres at all — the
    database tests skip in that case — and a cleanup problem must never turn a
    green run red.
    """
    import asyncio
    import contextlib

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from vigil.db.models import Base
    from vigil.db.session import database_url
    from vigil.db.session import engine as cached_engine
    from vigil.db.session import session_factory as cached_factory

    cached_engine.cache_clear()
    cached_factory.cache_clear()

    # Same allowlist as the fixtures: never TRUNCATE a database this suite does not
    # own, even in end-of-run cleanup. Normally unreachable (configure aborts first).
    if not _database_is_disposable(database_url()):
        return

    names = ", ".join(
        t.name for t in Base.metadata.sorted_tables if t.name != "alembic_version"
    )

    async def _wipe() -> None:
        eng = create_async_engine(database_url())
        try:
            async with eng.begin() as conn:
                await conn.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))
        finally:
            await eng.dispose()

    with contextlib.suppress(Exception):
        asyncio.run(_wipe())
