"""The session runners, against a **real Postgres** and a stub broker.

The broker is stubbed because no test may hit the Alpaca API; the database is
real because most of what a cycle does *is* database state — a structure marked
closed, a gate verdict persisted, a cycle row left unfinished after a crash — and
a mock of Postgres would only restate the hope.

Skipped, not failed, when no Postgres is reachable.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select, text

from tests.conftest import truncate_journal
from vigil.clock import ET, today_et
from vigil.db.models import Cycle, OpenStructureRow
from vigil.db.models import Order as OrderRow
from vigil.db.repositories import journal as J
from vigil.db.session import get_session
from vigil.domain import OpenStructure, PositionLeg, Structure
from vigil.worker import sessions as S
from vigil.worker.broker import AccountView
from vigil.worker.schedule import CycleKind

# A fixed weekday mid-session moment. The breach-close path turns on
# `minutes_to_close(now)`, which reads the **live** wall clock through `now_et()`:
# after ~15:40 ET (and all weekend) the sweep correctly declines to cross a spread
# into the imminent flatten, so a test asserting a breach *close* must pin the
# clock to mid-session or it fails every afternoon the suite happens to run. 11:00
# on a Tuesday leaves 300 minutes to the 16:00 close — comfortably over the 30-min
# breach-exit floor — and is decoupled from `date.today()` so the result never
# depends on when or where the tests run.
MID_SESSION = datetime(2026, 9, 1, 11, 0, tzinfo=ET)

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
    """Truncate before every test. These assert on counts, so leftovers would lie.

    The *last* test's rows are cleaned by `pytest_sessionfinish` in conftest
    rather than here — a per-test teardown has to run after the engine-disposal
    fixture and lands on a closed event loop.
    """
    async with get_session() as s:
        await truncate_journal(s)
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
    """Neutralise the two startup guards here — each is tested in its own file
    (`test_account.py`, `test_clock_guard.py`) — so these session tests exercise
    session logic against a network-free fake broker whose `.client` is inert."""
    monkeypatch.setattr(S, "verify_account", lambda **_kwargs: "acct-test")
    monkeypatch.setattr(S, "verify_clock", lambda **_kwargs: 0.0)


async def _context(broker, db):
    return await S.open_context(broker, db)


async def test_open_context_refuses_a_drifted_clock(no_lock, monkeypatch):
    """The clock guard rides the startup path with the account lock's fail-closed
    contract: a drift failure aborts the cycle, it is not swallowed. Overrides the
    `no_lock` stub with one that raises to prove `open_context` actually calls it
    and lets the error propagate."""
    from vigil.clock_guard import ClockDriftError

    def _drifted(**_kwargs):
        raise ClockDriftError("CLOCK DRIFT (test)")

    monkeypatch.setattr(S, "verify_clock", _drifted)
    broker = StubBroker(structures=(make_structure(),))
    async with get_session() as db:
        with pytest.raises(ClockDriftError):
            await S.open_context(broker, db)


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

async def test_a_breached_structure_is_closed_and_journalled(no_lock, monkeypatch):
    """Spot through the short strike with hours left: close, and say why."""
    monkeypatch.setattr(S, "now_et", lambda: MID_SESSION)
    structure = make_structure(expiry=MID_SESSION.date() + timedelta(days=1))
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


async def test_an_unpriceable_leg_blocks_the_close_loudly(no_lock, monkeypatch):
    """Hard rule #5: no market orders. So no quote means no order, and an alarm.

    The alternative — crossing with a market order because the position "must"
    close — is exactly the invitation §1.2 warns about on an indicative feed.
    """
    monkeypatch.setattr(S, "now_et", lambda: MID_SESSION)
    structure = make_structure(expiry=MID_SESSION.date() + timedelta(days=1))
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
    structure = make_structure(expiry=today_et())
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
    structure = make_structure(expiry=today_et())
    broker = StubBroker(structures=(structure,), quotes={})   # nothing priceable
    async with get_session() as db:
        ctx = await _context(broker, db)
        result = S.CycleResult(kind=CycleKind.FLATTEN)
        await S.flatten(ctx, result)
    assert any("FLATTEN INCOMPLETE" in w for w in result.warnings)


async def test_a_requested_flatten_closes_later_expiries_too(no_lock):
    """`/api/control/flatten` says close-all; the 15:40 stop says close-expiring.

    The scheduled run deliberately leaves a Monday expiry alone — auto-exercise is
    what it exists to prevent, and a later expiry is a position we still want. An
    operator asking for a flatten is asking for something else entirely, and a
    flatten that left half the book on would be the more dangerous reading.
    """
    structure = make_structure(expiry=date.today() + timedelta(days=2))
    broker = StubBroker(
        structures=(structure,),
        quotes={
            structure.legs[0].symbol: (Decimal("0.10"), Decimal("0.12")),
            structure.legs[1].symbol: (Decimal("0.04"), Decimal("0.06")),
        },
    )
    async with get_session() as db:
        ctx = await _context(broker, db)
        result = S.CycleResult(kind=CycleKind.FLATTEN, flatten_requested=True)
        await S.flatten(ctx, result)

    assert len(broker.closes) == 1, "a requested flatten must take the whole book"
    assert "operator flatten requested" in broker.closes[0][2]


async def test_a_requested_flatten_keeps_the_flag_until_the_book_is_empty(no_lock):
    """Closes are limit orders, not fills — so this cycle cannot claim success.

    Hard rule #5 forbids the market order that would make a close immediate, so
    the honest state after submitting is "asked, not confirmed". Clearing the flag
    here would resume trading on the strength of an order that may never fill.
    """
    structure = make_structure(expiry=today_et())
    broker = StubBroker(
        structures=(structure,),
        quotes={
            structure.legs[0].symbol: (Decimal("0.10"), Decimal("0.12")),
            structure.legs[1].symbol: (Decimal("0.04"), Decimal("0.06")),
        },
    )
    async with get_session() as db:
        ctx = await _context(broker, db)
        await J.set_flag(db, S.FLATTEN_FLAG, active=True, set_by="test")
        result = S.CycleResult(kind=CycleKind.FLATTEN, flatten_requested=True)
        await S.flatten(ctx, result)
        assert await J.is_flag_active(db, S.FLATTEN_FLAG) is True
    assert "stays set until" in result.summary


async def test_a_requested_flatten_clears_the_flag_once_the_book_is_flat(no_lock):
    """The bug this exists for: the flag was never cleared by anything.

    Left set, it pre-empted **every** subsequent cycle — so a single mis-click on
    `/api/control/flatten` ended trading for the rest of the competition until a
    human remembered `/unflatten`. The next cycle after the closes fill has to
    reconcile against an empty broker and stand the agent back up on its own.
    """
    broker = StubBroker(structures=())          # the closes filled; nothing left
    async with get_session() as db:
        ctx = await _context(broker, db)
        await J.set_flag(db, S.FLATTEN_FLAG, active=True, set_by="test")
        result = S.CycleResult(kind=CycleKind.FLATTEN, flatten_requested=True)
        await S.flatten(ctx, result)
        assert await J.is_flag_active(db, S.FLATTEN_FLAG) is False
    assert "entries resume next cycle" in result.summary


async def test_the_scheduled_flatten_never_touches_the_flag(no_lock):
    """15:40 is the clock, not an operator. It has no business clearing a request.

    If someone sets FLATTEN at 15:39, the scheduled run must not be the thing that
    decides the request was satisfied — otherwise a daily stop would silently
    withdraw a human's instruction.
    """
    broker = StubBroker(structures=())
    async with get_session() as db:
        ctx = await _context(broker, db)
        await J.set_flag(db, S.FLATTEN_FLAG, active=True, set_by="test")
        result = S.CycleResult(kind=CycleKind.FLATTEN)   # flatten_requested stays False
        await S.flatten(ctx, result)
        assert await J.is_flag_active(db, S.FLATTEN_FLAG) is True


async def test_a_flatten_request_stands_the_agent_back_up_end_to_end(no_lock):
    """Two real cycles: the request pre-empts, then the agent resumes by itself."""
    async with get_session() as db:
        await J.set_flag(db, S.FLATTEN_FLAG, active=True, set_by="test")

    # Cycle one: an ENTRY was due, the flag pre-empts it, the book is already flat.
    first = await S.run_cycle(CycleKind.ENTRY, broker=StubBroker(structures=()))
    assert first.kind is CycleKind.FLATTEN
    assert first.flatten_requested is True

    # Cycle two: the flag is gone, so the cycle that was asked for actually runs.
    second = await S.run_cycle(CycleKind.MANAGE, broker=StubBroker(structures=()))
    assert second.kind is CycleKind.MANAGE, "the flag was never cleared"
    assert second.flatten_requested is False


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


async def test_the_halt_flag_does_not_stop_management(no_lock, monkeypatch):
    """A halt stops the agent taking risk *on*, never taking it *off*.

    Gate 3 makes the same distinction — it halts new entries while management
    keeps running — and a halt that closed that door too would turn a bad day
    into an unmanaged one.
    """
    monkeypatch.setattr(S, "now_et", lambda: MID_SESSION)
    structure = make_structure(expiry=MID_SESSION.date() + timedelta(days=1))
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
    structure = make_structure(expiry=today_et())
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
