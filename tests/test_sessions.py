"""The session runners, against a **real Postgres** and a stub broker.

The broker is stubbed because no test may hit the Alpaca API; the database is
real because most of what a cycle does *is* database state — a structure marked
closed, a gate verdict persisted, a cycle row left unfinished after a crash — and
a mock of Postgres would only restate the hope.

Skipped, not failed, when no Postgres is reachable.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select, text

from vigil.db.models import Cycle, OpenStructureRow
from vigil.db.models import Order as OrderRow
from vigil.db.repositories import journal as J
from vigil.db.session import get_session
from vigil.domain import OpenStructure, PositionLeg, Structure
from vigil.worker import sessions as S
from vigil.worker.broker import AccountView
from vigil.worker.schedule import CycleKind

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
async def _fresh_engine():
    """Dispose the cached engine between tests — see tests/test_journal.py."""
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
async def _require_postgres():
    if not await _postgres_reachable():
        pytest.skip(f"no Postgres at {os.getenv('DATABASE_URL', 'localhost/vigil')}")


@pytest.fixture(autouse=True)
async def _clean_journal():
    """Truncate between tests. These assert on counts, so leftovers would lie."""
    async with get_session() as s:
        await s.execute(
            text(
                "TRUNCATE accounts, sessions, cycles, proposals, proposal_legs, "
                "gate_verdicts, structures, orders, fills, equity_snapshots, "
                "market_snapshots, llm_memos, control_flags RESTART IDENTITY CASCADE"
            )
        )
    yield


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #

class StubOrder:
    def __init__(self, client_order_id: str) -> None:
        self.id = f"broker-{client_order_id}"
        self.client_order_id = client_order_id
        self.status = "accepted"
        self.limit_price = "0.10"


class StubBroker:
    """Records what was asked of it; answers from fixtures. No network."""

    def __init__(
        self,
        *,
        structures: tuple[OpenStructure, ...] = (),
        equity: Decimal = Decimal(100_000),
        quotes: dict[str, tuple[Decimal, Decimal]] | None = None,
        spot: Decimal = Decimal("760.00"),
    ) -> None:
        self._structures = structures
        self._equity = equity
        self.quotes_by_symbol = quotes or {}
        self._spot = spot
        self.closes: list[tuple[OpenStructure, Decimal, str]] = []
        self.cancelled_all = False

    @property
    def client(self):
        return object()

    async def account(self) -> AccountView:
        return AccountView(
            account_id="acct-test", equity=self._equity,
            last_equity=self._equity, status="ACTIVE",
        )

    async def structures(self) -> tuple[OpenStructure, ...]:
        return self._structures

    async def spot(self, underlying: str) -> Decimal:
        return self._spot

    async def quotes(self, symbols):
        """Only the symbols this stub was given — omissions are the point.

        `fetch_quotes` omits symbols with no two-sided quote rather than
        zero-filling them, so a stub that invented a quote for every symbol
        asked would make the unpriceable-leg case untestable.
        """
        return {sym: self.quotes_by_symbol[sym] for sym in symbols
                if sym in self.quotes_by_symbol}

    async def submit_close(self, structure, limit_price, *, reason):
        self.closes.append((structure, limit_price, reason))
        return StubOrder(f"vigil-cls-{structure.underlying}-{len(self.closes)}")

    async def cancel_all_orders(self) -> None:
        self.cancelled_all = True


def make_structure(
    *,
    underlying: str = "SPY",
    expiry: date | None = None,
    short_put: Decimal = Decimal(755),
    has_target: bool = True,
) -> OpenStructure:
    exp = expiry or date.today()
    return OpenStructure(
        underlying=underlying,
        expiry=exp,
        strikes=(Decimal(754), short_put),
        max_loss=Decimal(800),
        dollar_delta=Decimal(0),
        has_resting_target=has_target,
        structure=Structure.PUT_CREDIT_SPREAD,
        short_put_strikes=(short_put,),
        net_credit=Decimal("0.20"),
        contracts=8,
        legs=(
            PositionLeg(symbol=f"{underlying}260828P00755000", ratio_qty=1, is_short=True),
            PositionLeg(symbol=f"{underlying}260828P00754000", ratio_qty=1, is_short=False),
        ),
    )


@pytest.fixture
def no_lock(monkeypatch):
    """The account assertion is tested in tests/test_account.py; stub it here."""
    monkeypatch.setattr(S, "verify_account", lambda **_kwargs: "acct-test")


async def _context(broker, db):
    return await S.open_context(broker, db)


# --------------------------------------------------------------------------- #
# Selection — pure, and the deterministic fallback the LLM will replace
# --------------------------------------------------------------------------- #

def test_rank_prefers_more_credit_per_dollar_of_width(put_credit_spread, iron_condor):
    """§6.3's deterministic fallback ranks by Gate 9's own quality measure."""
    from dataclasses import replace

    rich = replace(put_credit_spread, net_credit=Decimal("0.30"))   # 30% of width
    poor = replace(iron_condor, net_credit=Decimal("0.20"))          # 20% of width
    ranked = S._rank([(poor, None), (rich, None)])
    assert ranked[0] is rich


def test_rank_breaks_ties_on_the_smaller_bet(put_credit_spread):
    """Same edge, smaller bet wins — the conservative tiebreak, stated once."""
    from dataclasses import replace

    small = replace(put_credit_spread, contracts=1)
    large = replace(put_credit_spread, contracts=5, client_order_id="vigil-test-0002")
    ranked = S._rank([(large, None), (small, None)])
    assert ranked[0] is small


# --------------------------------------------------------------------------- #
# reconcile
# --------------------------------------------------------------------------- #

async def test_reconcile_registers_what_the_broker_reports(no_lock):
    broker = StubBroker(structures=(make_structure(),))
    async with get_session() as db:
        ctx = await _context(broker, db)
        result = S.CycleResult(kind=CycleKind.MANAGE)
        await S.reconcile(ctx, result)
        rows = await J.open_structure_rows(db)
    assert [r.underlying for r in rows] == ["SPY"]


async def test_a_vanished_structure_is_journalled_as_closed(no_lock):
    """The only evidence a resting GTC target filled is the position's absence.

    Nothing observes that close — it happens while the worker is asleep between
    sweeps (§2.6) — so without this the registry accumulates phantom open rows
    and Gate 5 refuses entries against positions that no longer exist.
    """
    structure = make_structure()
    async with get_session() as db:
        ctx = await _context(StubBroker(structures=(structure,)), db)
        await S.reconcile(ctx, S.CycleResult(kind=CycleKind.MANAGE))

    async with get_session() as db:
        ctx = await _context(StubBroker(structures=()), db)   # broker now flat
        result = S.CycleResult(kind=CycleKind.MANAGE)
        await S.reconcile(ctx, result)
        rows = (await db.scalars(select(OpenStructureRow))).all()

    assert [r.status for r in rows] == ["closed"]
    assert "closed while unobserved" in result.summary


async def test_a_structure_with_no_resting_exit_is_reported_as_a_defect(no_lock):
    """§2.6 calls this a defect in those words; the honest count is zero."""
    broker = StubBroker(structures=(make_structure(has_target=False),))
    async with get_session() as db:
        ctx = await _context(broker, db)
        result = S.CycleResult(kind=CycleKind.MANAGE)
        await S.reconcile(ctx, result)
    assert any("§2.6 DEFECT" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# manage
# --------------------------------------------------------------------------- #

async def test_a_breached_structure_is_closed_and_journalled(no_lock):
    """Spot through the short strike with hours left: close, and say why."""
    structure = make_structure(expiry=date.today() + timedelta(days=1))
    broker = StubBroker(
        structures=(structure,),
        spot=Decimal(750),                       # below the 755 short put
        quotes={
            structure.legs[0].symbol: (Decimal("5.00"), Decimal("5.10")),
            structure.legs[1].symbol: (Decimal("4.20"), Decimal("4.30")),
        },
    )
    async with get_session() as db:
        ctx = await _context(broker, db)
        result = S.CycleResult(kind=CycleKind.MANAGE)
        await S.run_manage(ctx, result)
        orders = (await db.scalars(select(OrderRow))).all()
        rows = (await db.scalars(select(OpenStructureRow))).all()

    assert len(broker.closes) == 1
    assert "breached" in broker.closes[0][2]
    assert [o.intent for o in orders] == ["close"]
    assert [r.status for r in rows] == ["closed"]
    assert result.closed == 1


async def test_an_unpriceable_leg_blocks_the_close_loudly(no_lock):
    """Hard rule #5: no market orders. So no quote means no order, and an alarm.

    The alternative — crossing with a market order because the position "must"
    close — is exactly the invitation §1.2 warns about on an indicative feed.
    """
    structure = make_structure(expiry=date.today() + timedelta(days=1))
    broker = StubBroker(
        structures=(structure,),
        spot=Decimal(750),
        quotes={structure.legs[0].symbol: (Decimal("5.00"), Decimal("5.10"))},  # one leg only
    )
    async with get_session() as db:
        ctx = await _context(broker, db)
        result = S.CycleResult(kind=CycleKind.MANAGE)
        await S.run_manage(ctx, result)

    assert broker.closes == []
    assert any("CANNOT PRICE" in w for w in result.warnings)
    assert result.closed == 0


async def test_a_healthy_structure_is_held(no_lock):
    structure = make_structure(expiry=date.today() + timedelta(days=1))
    broker = StubBroker(structures=(structure,), spot=Decimal(770))   # well above 755
    async with get_session() as db:
        ctx = await _context(broker, db)
        result = S.CycleResult(kind=CycleKind.MANAGE)
        await S.run_manage(ctx, result)
    assert broker.closes == []
    assert "1 held" in result.summary


# --------------------------------------------------------------------------- #
# flatten
# --------------------------------------------------------------------------- #

async def test_flatten_cancels_working_orders_before_closing(no_lock):
    """An entry filling at 15:41 would create a position after the flatten decided."""
    structure = make_structure(expiry=date.today())
    broker = StubBroker(
        structures=(structure,),
        quotes={
            structure.legs[0].symbol: (Decimal("0.10"), Decimal("0.12")),
            structure.legs[1].symbol: (Decimal("0.04"), Decimal("0.06")),
        },
    )
    async with get_session() as db:
        ctx = await _context(broker, db)
        result = S.CycleResult(kind=CycleKind.FLATTEN)
        await S.flatten(ctx, result)

    assert broker.cancelled_all
    assert len(broker.closes) == 1


async def test_flatten_leaves_later_expiries_alone(no_lock):
    """The 15:40 stop is about auto-exercise, not about being flat for its own sake."""
    structure = make_structure(expiry=date.today() + timedelta(days=2))
    broker = StubBroker(structures=(structure,))
    async with get_session() as db:
        ctx = await _context(broker, db)
        result = S.CycleResult(kind=CycleKind.FLATTEN)
        await S.flatten(ctx, result)
    assert broker.closes == []
    assert "nothing expiring today" in result.summary


async def test_an_incomplete_flatten_is_loud(no_lock):
    """A position that could not be closed at 15:40 is the worst thing in the book."""
    structure = make_structure(expiry=date.today())
    broker = StubBroker(structures=(structure,), quotes={})   # nothing priceable
    async with get_session() as db:
        ctx = await _context(broker, db)
        result = S.CycleResult(kind=CycleKind.FLATTEN)
        await S.flatten(ctx, result)
    assert any("FLATTEN INCOMPLETE" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# Control flags — the API's only channel to the worker
# --------------------------------------------------------------------------- #

async def test_the_halt_flag_stops_entries(no_lock):
    async with get_session() as db:
        ctx = await _context(StubBroker(), db)
        await J.set_flag(db, S.HALT_FLAG, active=True, set_by="test")
        result = S.CycleResult(kind=CycleKind.ENTRY)
        await S.run_entry(ctx, result)
    assert "HALT flag active" in result.summary


async def test_the_halt_flag_does_not_stop_management(no_lock):
    """A halt stops the agent taking risk *on*, never taking it *off*.

    Gate 3 makes the same distinction — it halts new entries while management
    keeps running — and a halt that closed that door too would turn a bad day
    into an unmanaged one.
    """
    structure = make_structure(expiry=date.today() + timedelta(days=1))
    broker = StubBroker(
        structures=(structure,),
        spot=Decimal(750),
        quotes={
            structure.legs[0].symbol: (Decimal("5.00"), Decimal("5.10")),
            structure.legs[1].symbol: (Decimal("4.20"), Decimal("4.30")),
        },
    )
    async with get_session() as db:
        ctx = await _context(broker, db)
        await J.set_flag(db, S.HALT_FLAG, active=True, set_by="test")
        result = S.CycleResult(kind=CycleKind.MANAGE)
        await S.run_manage(ctx, result)
    assert result.closed == 1


async def test_the_flatten_flag_preempts_any_cycle(no_lock, monkeypatch):
    """`/api/control/flatten` at 11:02 is not a request to finish the 11:02 sweep."""
    structure = make_structure(expiry=date.today())
    broker = StubBroker(
        structures=(structure,),
        quotes={
            structure.legs[0].symbol: (Decimal("0.10"), Decimal("0.12")),
            structure.legs[1].symbol: (Decimal("0.04"), Decimal("0.06")),
        },
    )
    async with get_session() as db:
        await J.set_flag(db, S.FLATTEN_FLAG, active=True, set_by="test")

    result = await S.run_cycle(CycleKind.ENTRY, broker=broker)
    assert result.kind is CycleKind.FLATTEN
    assert broker.cancelled_all


# --------------------------------------------------------------------------- #
# The cycle row itself
# --------------------------------------------------------------------------- #

async def test_a_completed_cycle_records_when_it_finished(no_lock):
    await S.run_cycle(CycleKind.PREMARKET, broker=StubBroker())
    async with get_session() as db:
        rows = (await db.scalars(select(Cycle))).all()
    assert len(rows) == 1
    assert rows[0].finished_at is not None
    assert rows[0].kind == "premarket"


async def test_a_crashed_cycle_leaves_finished_at_null(no_lock, monkeypatch):
    """`finished_at IS NULL` is the live query for "what died mid-cycle?".

    A write-on-completion design cannot answer that question at all, which is why
    the row is flushed before the work rather than after it.
    """
    async def _boom(ctx, result):
        raise RuntimeError("sense exploded")

    monkeypatch.setitem(S._CYCLES, CycleKind.PREMARKET, _boom)

    with pytest.raises(RuntimeError, match="sense exploded"):
        await S.run_cycle(CycleKind.PREMARKET, broker=StubBroker())

    async with get_session() as db:
        rows = (await db.scalars(select(Cycle))).all()
    assert len(rows) == 1
    assert rows[0].finished_at is None


async def test_a_restart_rejoins_todays_session_rather_than_starting_a_second(no_lock):
    """`uq_session_day` makes a second insert an error; a restart must not crash."""
    await S.run_cycle(CycleKind.PREMARKET, broker=StubBroker())
    await S.run_cycle(CycleKind.MANAGE, broker=StubBroker())

    async with get_session() as db:
        sessions = (await db.scalars(select(J.Session))).all()
        cycles = (await db.scalars(select(Cycle))).all()
    assert len(sessions) == 1
    assert len(cycles) == 2


async def test_opening_equity_is_not_rebased_by_a_restart(no_lock):
    """The 09:45 reading is the day's baseline. A 14:00 restart must not redefine it."""
    await S.run_cycle(CycleKind.PREMARKET, broker=StubBroker(equity=Decimal(100_000)))
    await S.run_cycle(CycleKind.MANAGE, broker=StubBroker(equity=Decimal(97_000)))

    async with get_session() as db:
        row = (await db.scalars(select(J.Session))).one()
    assert row.opening_equity == Decimal(100_000)


async def test_peak_equity_survives_a_restart(no_lock):
    """Gate 4 measures drawdown against the high-water mark, read from the journal.

    Held in process memory it would reset on every restart, computing a drawdown
    of zero and silently disarming the one gate whose job is the unrecoverable
    day.
    """
    await S.run_cycle(CycleKind.PREMARKET, broker=StubBroker(equity=Decimal(110_000)))

    async with get_session() as db:
        ctx = await _context(StubBroker(equity=Decimal(100_000)), db)
        state = await S._state(ctx, ())
    assert state.peak_equity == Decimal(110_000)
    assert state.drawdown_pct < 0
