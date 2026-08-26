"""The journal, against a **real Postgres**.

Mocked out, these tests would prove nothing. Every guarantee below is a *database*
guarantee — a UNIQUE constraint, a NUMERIC type, a CHECK — and a fake that
answers the way we hoped would simply restate the hope. Hard rule #9 says
idempotency is a database constraint rather than application luck; the only way
to show that is to make the database refuse a duplicate.

Skipped, not failed, when no Postgres is reachable.
"""

from __future__ import annotations

import os
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from vigil.db.models import GateVerdictRow, Order, Proposal
from vigil.db.repositories.journal import record_proposal
from vigil.db.session import get_session
from vigil.domain import GateVerdict, KernelDecision

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
async def _fresh_engine():
    """Dispose the engine between tests.

    `vigil.db.session.engine()` is `lru_cache`d — one engine per process, which
    is right in production where there is one event loop for the lifetime of the
    worker. pytest-asyncio gives each test its **own** loop, so a cached engine
    hands the second test a pool of asyncpg connections bound to a loop that has
    already closed. The symptom is memorable: tests alternate pass/skip, because
    every other one poisons the pool for the next, with a stray
    `coroutine 'Connection._cancel' was never awaited` as the only clue.

    Disposing per test costs a reconnect and removes the whole class of problem.
    """
    from vigil.db import session as session_module

    session_module.engine.cache_clear()
    session_module.session_factory.cache_clear()
    yield
    try:
        await session_module.engine().dispose()
    finally:
        session_module.engine.cache_clear()
        session_module.session_factory.cache_clear()


async def _postgres_reachable() -> bool:
    try:
        async with get_session() as s:
            await s.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
async def _require_db():
    if not await _postgres_reachable():
        pytest.skip(f"no Postgres at {os.getenv('DATABASE_URL', 'localhost/vigil')}")


@pytest.fixture
async def cycle_id():
    """A throwaway account/session/cycle chain, rolled back after the test."""
    async with get_session() as s:
        acct = (await s.execute(text(
            "INSERT INTO accounts (alpaca_account_id, starting_equity) "
            "VALUES (:a, 100000) RETURNING id"), {"a": f"test-{os.urandom(6).hex()}"})).scalar_one()
        sess = (await s.execute(text(
            "INSERT INTO sessions (account_id, trading_date) VALUES (:a, :d) RETURNING id"),
            {"a": acct, "d": date(2026, 8, 26)})).scalar_one()
        cyc = (await s.execute(text(
            "INSERT INTO cycles (session_id, kind) VALUES (:s, 'entry') RETURNING id"),
            {"s": sess})).scalar_one()
    yield cyc
    async with get_session() as s:
        await s.execute(text("DELETE FROM accounts WHERE id = :a"), {"a": acct})


def _decision(approved: bool = True) -> KernelDecision:
    return KernelDecision(
        approved=approved,
        verdicts=tuple(
            GateVerdict(n, f"gate_{n}", passed=approved or n != 2,
                        reason="" if approved else "over budget",
                        detail={"observed": "1"})
            for n in range(1, 13)
        ),
    )


# --------------------------------------------------------------------------- #
# Every verdict is persisted, passes included
# --------------------------------------------------------------------------- #

async def test_all_twelve_verdicts_are_stored_including_passes(cycle_id, put_credit_spread):
    """§5: rejections are the evidence the risk system works, and a table holding
    only failures cannot answer 'did any of this ever fire?'"""
    async with get_session() as s:
        row = await record_proposal(s, put_credit_spread, _decision(), cycle_id=cycle_id)
        pid = row.id
    async with get_session() as s:
        verdicts = (await s.execute(
            select(GateVerdictRow).where(GateVerdictRow.proposal_id == pid))).scalars().all()
    assert len(verdicts) == 12
    assert all(v.passed for v in verdicts)


async def test_a_gate_cannot_be_recorded_twice_for_one_proposal(cycle_id, put_credit_spread):
    """The unique constraint is what makes 'twelve rows per proposal' literal
    rather than aspirational — a partial write cannot pass as a full evaluation."""
    async with get_session() as s:
        row = await record_proposal(s, put_credit_spread, _decision(), cycle_id=cycle_id)
        pid = row.id
    with pytest.raises(IntegrityError):
        async with get_session() as s:
            s.add(GateVerdictRow(proposal_id=pid, gate_no=1, name="dupe", passed=True))


async def test_a_rejected_proposal_is_stored_with_its_reason(cycle_id, put_credit_spread):
    async with get_session() as s:
        row = await record_proposal(s, put_credit_spread, _decision(approved=False),
                                    cycle_id=cycle_id)
        pid = row.id
    async with get_session() as s:
        p = (await s.execute(select(Proposal).where(Proposal.id == pid))).scalar_one()
        failed = (await s.execute(select(GateVerdictRow).where(
            GateVerdictRow.proposal_id == pid, GateVerdictRow.passed.is_(False)))).scalars().all()
    assert p.approved is False
    assert [v.gate_no for v in failed] == [2]
    assert failed[0].reason == "over budget"


# --------------------------------------------------------------------------- #
# Hard rule #9 — idempotency is a database constraint, not application luck
# --------------------------------------------------------------------------- #

async def test_a_duplicate_client_order_id_raises_rather_than_double_filling():
    """The whole point of hard rule #9. A retry after a timeout that actually
    succeeded must hit an integrity error, not place a second order."""
    coid = f"vigil-test-{os.urandom(6).hex()}"
    async with get_session() as s:
        s.add(Order(client_order_id=coid, intent="open", limit_price=Decimal("0.20"),
                    qty=1, status="new"))
    with pytest.raises(IntegrityError):
        async with get_session() as s:
            s.add(Order(client_order_id=coid, intent="open", limit_price=Decimal("0.20"),
                        qty=1, status="new"))
    async with get_session() as s:
        await s.execute(text("DELETE FROM orders WHERE client_order_id = :c"), {"c": coid})


async def test_ladder_rungs_do_not_collide(put_credit_spread):
    """B1's fix, checked at the database rather than the broker: three rungs
    sharing one id would fail here even if Alpaca allowed them."""
    from vigil.execution.mleg import rung_client_order_id

    base = f"vigil-test-{os.urandom(6).hex()}"
    async with get_session() as s:
        for rung in (1, 2, 3):
            s.add(Order(client_order_id=rung_client_order_id(base, rung), intent="open",
                        limit_price=Decimal("0.20"), qty=1, status="new", rung=rung))
    async with get_session() as s:
        n = (await s.execute(text(
            "SELECT count(*) FROM orders WHERE client_order_id LIKE :p"),
            {"p": f"{base}%"})).scalar_one()
        await s.execute(text("DELETE FROM orders WHERE client_order_id LIKE :p"),
                        {"p": f"{base}%"})
    assert n == 3


# --------------------------------------------------------------------------- #
# Unbounded max profit
# --------------------------------------------------------------------------- #

async def test_an_unbounded_max_profit_is_stored_as_null(cycle_id, long_strangle):
    """`Decimal("Infinity")` has no NUMERIC representation. NULL says 'unbounded';
    a large stand-in would say 'we measured this', and be wrong."""
    assert not long_strangle.max_profit.is_finite(), "fixture must actually be unbounded"

    async with get_session() as s:
        row = await record_proposal(s, long_strangle, _decision(), cycle_id=cycle_id)
        pid = row.id
    async with get_session() as s:
        p = (await s.execute(select(Proposal).where(Proposal.id == pid))).scalar_one()
    assert p.max_profit is None
    # Premium paid: 1.68 x 100 x 2 contracts.
    assert p.max_loss == Decimal("336.0000")


async def test_a_bounded_max_profit_is_stored_as_a_number(cycle_id, put_credit_spread):
    async with get_session() as s:
        row = await record_proposal(s, put_credit_spread, _decision(), cycle_id=cycle_id)
        pid = row.id
    async with get_session() as s:
        p = (await s.execute(select(Proposal).where(Proposal.id == pid))).scalar_one()
    assert p.max_profit == Decimal("20.0000")


async def test_money_round_trips_exactly_as_decimal(cycle_id, put_credit_spread):
    """NUMERIC, never float. A float column would round a credit and the journal
    would then disagree with the kernel about what the trade risked."""
    async with get_session() as s:
        row = await record_proposal(s, put_credit_spread, _decision(), cycle_id=cycle_id)
        pid = row.id
    async with get_session() as s:
        p = (await s.execute(select(Proposal).where(Proposal.id == pid))).scalar_one()
    assert isinstance(p.net_credit, Decimal)
    assert p.net_credit == Decimal("0.20")
