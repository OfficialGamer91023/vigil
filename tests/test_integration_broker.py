"""Fake-broker integration: the real kernel and real ladder, a fake order client.

**Layer 1 of the integration harness.** Two existing test layers leave a gap this
one fills:

- `test_gates.py` calls the kernel directly — it proves each gate's logic, but not
  that the *submit path* actually consults the kernel, or with the right inputs.
- `test_sessions.py`'s `StubBroker` never runs the kernel at all — its
  `submit_entry` is itself a stub, so a wiring bug between the session runner and
  the kernel is invisible to it.

This drives `router.submit_entry` — the one place an order reaches a broker — through
the genuine article: all twelve gates run and the price ladder walks, against a
client that fabricates fills instead of calling Alpaca. That is what catches a bug a
pure gate test cannot — a gate fed the wrong context on the *binding* run (O-2), or a
fill that is never journalled (D-5) — and it is the harness a D-1-class portfolio-delta
regression needs (Layer 2 builds the sense→act path on top of it).
"""

from __future__ import annotations

import os
from datetime import date
from decimal import Decimal

import pytest

from tests.conftest import DEFAULT_NOW
from vigil.clock import today_et
from vigil.config import ladder_config, risk_config, strategy_config
from vigil.data.bars import SessionBars
from vigil.db.models import EquitySnapshot
from vigil.db.models import Fill as FillRow
from vigil.db.models import Order as OrderRow
from vigil.db.repositories import journal as J
from vigil.db.session import get_session
from vigil.domain import (
    OpenStructure,
    PortfolioState,
    PositionLeg,
    Structure,
    TradeProposal,
)
from vigil.execution.router import RiskKernelRejection, submit_entry
from vigil.risk.context import KernelContext
from vigil.worker import sense as sense_module
from vigil.worker import sessions as S
from vigil.worker.broker import AccountView
from vigil.worker.schedule import CycleKind

# --------------------------------------------------------------------------- #
# The fake order client + broker — the whole point of the file.
# --------------------------------------------------------------------------- #

class FakeOrder:
    """The handful of attributes the router reads off an SDK order object.

    Deliberately *not* an alpaca model: the router only ever touches `id`,
    `client_order_id`, `status`, `filled_qty` and `filled_avg_price`, so anything
    more would be fitting a mock to internals the code does not use.
    """

    def __init__(
        self,
        *,
        order_id: str,
        client_order_id: str,
        status: str,
        filled_qty: int,
        filled_avg_price: str | None,
        limit_price: str,
    ) -> None:
        self.id = order_id
        self.client_order_id = client_order_id
        self.status = status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.limit_price = limit_price


class FakeOrderClient:
    """Stands in for alpaca-py's `TradingClient`. No network, deterministic fills.

    The fill is resolved on `get_order_by_id`, not on `submit_order`, mirroring the
    real world: the broker accepts a working order first and reports the fill on a
    later poll. `fill_qty` is how many packages come back filled — the broker sets
    it to the proposal's contract count, so every Layer 1 fill is a full fill (the
    partial and no-fill ladder paths already have their own coverage in
    `test_router.py`).
    """

    def __init__(self, *, fill_qty: int, avg_price: str = "0.20") -> None:
        self._fill_qty = fill_qty
        self._avg = avg_price
        self.submitted: list[object] = []
        self.cancelled: list[str] = []
        self._seq = 0

    def _next_id(self) -> str:
        self._seq += 1
        return f"fake-{self._seq}"

    def submit_order(self, req: object) -> FakeOrder:
        # The request is a real `LimitOrderRequest` from `build_entry_order` /
        # `build_closing_order`; echo its own client_order_id and price back so the
        # journal records the ids the code actually generated.
        self.submitted.append(req)
        return FakeOrder(
            order_id=self._next_id(),
            client_order_id=str(getattr(req, "client_order_id", "")),
            # Accepted-but-unfilled: the router polls with get_order_by_id next.
            status="accepted",
            filled_qty=0,
            filled_avg_price=None,
            limit_price=str(getattr(req, "limit_price", "0")),
        )

    def get_order_by_id(self, order_id: str) -> FakeOrder:
        # Resolve the fill on the poll: the configured quantity comes back filled.
        return FakeOrder(
            order_id=order_id,
            client_order_id="",
            status="filled",
            filled_qty=self._fill_qty,
            filled_avg_price=self._avg,
            limit_price="0",
        )

    def cancel_order_by_id(self, order_id: str) -> None:
        self.cancelled.append(order_id)


class FakeBroker:
    """A broker whose `submit_entry` runs the *real* router against the fake client.

    Mirrors `worker.broker.Broker.submit_entry` exactly (hard rule #4: the kernel
    runs inside the submit path), minus the `to_thread` hop — the router is sync and
    a test has no event-loop starvation to avoid. `sleep` is stubbed to nothing so
    the ladder does not wait its real 60 seconds per rung.
    """

    def __init__(self, client: FakeOrderClient) -> None:
        self.client = client

    async def order_status(self, order_id: str) -> str:
        # The resting profit target the router places on every fill went live and is
        # working — the healthy §2.6 shape the target-confirmation poll expects.
        return "held"

    async def submit_entry(
        self,
        proposal: TradeProposal,
        state: PortfolioState,
        context: KernelContext,
        *,
        risk: object = None,
        strategy: object = None,
    ):
        # A full fill means every contract the proposal asked for.
        self.client._fill_qty = proposal.contracts
        return submit_entry(
            proposal, state, context,
            client=self.client, risk=risk, strategy=strategy,
            sleep=lambda _s: None,
        )


def _good_context(proposal: TradeProposal) -> KernelContext:
    """A context that passes gates 11 and 12: a frozen in-hours clock and a chain
    that lists exactly the proposal's own legs."""
    return KernelContext(
        now=DEFAULT_NOW,
        available_symbols=frozenset(leg.symbol for leg in proposal.legs),
    )


# --------------------------------------------------------------------------- #
# Failure injection — Gates 3, 4, 7 must block the order end-to-end (no DB).
#
# Each test breaks exactly one portfolio dimension and asserts two things: the
# kernel raised (so the proposal never bound) *and* nothing reached the client. The
# second half is the integration claim — a gate that "fails" but still lets an order
# through would pass a unit test and lose real money.
# --------------------------------------------------------------------------- #

async def test_gate3_daily_loss_stop_blocks_the_order_at_the_broker(put_credit_spread):
    cfg = risk_config()
    # A day P&L one point below the halt threshold — no ambiguity about which gate.
    equity = Decimal(100_000)
    state = PortfolioState(
        equity=equity,
        peak_equity=equity,
        day_pnl=equity * (cfg.daily_loss_halt_pct - Decimal("0.01")),
    )
    client = FakeOrderClient(fill_qty=1)
    broker = FakeBroker(client)

    with pytest.raises(RiskKernelRejection) as exc:
        await broker.submit_entry(put_credit_spread, state, _good_context(put_credit_spread))

    assert 3 in {v.number for v in exc.value.decision.failures}
    assert client.submitted == []  # nothing ever reached the broker


async def test_gate4_drawdown_halt_blocks_the_order_at_the_broker(put_credit_spread):
    cfg = risk_config()
    peak = Decimal(100_000)
    # Below peak by more than the drawdown limit, but flat on the *day* — so this is
    # Gate 4 firing, not Gate 3.
    equity = peak * (Decimal(1) + cfg.max_drawdown_pct - Decimal("0.01"))
    state = PortfolioState(equity=equity, peak_equity=peak, day_pnl=Decimal(0))
    client = FakeOrderClient(fill_qty=1)
    broker = FakeBroker(client)

    with pytest.raises(RiskKernelRejection) as exc:
        await broker.submit_entry(put_credit_spread, state, _good_context(put_credit_spread))

    assert 4 in {v.number for v in exc.value.decision.failures}
    assert client.submitted == []


async def test_gate7_portfolio_dollar_delta_blocks_the_order_at_the_broker(put_credit_spread):
    cfg = risk_config()
    equity = Decimal(100_000)
    limit = cfg.max_dollar_delta_pct * equity
    # An existing position on a *different* underlying carrying delta well past the
    # portfolio limit on its own. Different underlying so Gate 6 (concentration)
    # stays out of it — this is Gate 7 summing across the whole book, which is the
    # exact thing D-1 broke by leaving reconciled deltas at zero.
    heavy = OpenStructure(
        underlying="QQQ",
        expiry=put_credit_spread.expiry,
        strikes=(Decimal(500), Decimal(499)),
        max_loss=Decimal(800),
        dollar_delta=limit * Decimal(3),
        structure=Structure.PUT_CREDIT_SPREAD,
        contracts=1,
        legs=(
            PositionLeg(symbol="QQQ260827P00500000", ratio_qty=1, is_short=True),
            PositionLeg(symbol="QQQ260827P00499000", ratio_qty=1, is_short=False),
        ),
    )
    state = PortfolioState(
        equity=equity, peak_equity=equity, day_pnl=Decimal(0),
        open_structures=(heavy,),
    )
    client = FakeOrderClient(fill_qty=1)
    broker = FakeBroker(client)

    with pytest.raises(RiskKernelRejection) as exc:
        await broker.submit_entry(put_credit_spread, state, _good_context(put_credit_spread))

    assert 7 in {v.number for v in exc.value.decision.failures}
    assert client.submitted == []


# --------------------------------------------------------------------------- #
# Gate 12 on the binding run — the O-2 regression (no DB).
# --------------------------------------------------------------------------- #

async def test_gate12_rejects_a_strike_absent_from_the_binding_context(
    put_credit_spread, flat_book
):
    """A proposal whose short leg the chain never listed must be refused on the run
    that binds — proof the binding context actually carries the chain symbols. If
    `_submit` ever reverts to a bare context, `available_symbols` is empty, Gate 12
    skips, and this order would sail through."""
    legs = list(put_credit_spread.legs)
    only_the_long_leg = frozenset({legs[1].symbol})  # omit the short strike
    context = KernelContext(now=DEFAULT_NOW, available_symbols=only_the_long_leg)
    client = FakeOrderClient(fill_qty=1)
    broker = FakeBroker(client)

    with pytest.raises(RiskKernelRejection) as exc:
        await broker.submit_entry(put_credit_spread, flat_book, context)

    assert 12 in {v.number for v in exc.value.decision.failures}
    assert client.submitted == []


async def test_the_same_proposal_binds_when_both_strikes_are_present(put_credit_spread, flat_book):
    """The control for the test above: with the chain listing both legs, the very
    same proposal reaches the broker. Together they show Gate 12 is *consulted*, not
    merely always-passing or always-failing."""
    client = FakeOrderClient(fill_qty=1)
    broker = FakeBroker(client)

    result = await broker.submit_entry(
        put_credit_spread, flat_book, _good_context(put_credit_spread)
    )

    assert result.filled
    assert client.submitted, "an approved proposal must reach the broker"


# --------------------------------------------------------------------------- #
# D-5 — a fill is journalled to the `fills` table, through the real _submit (DB).
# --------------------------------------------------------------------------- #

async def _postgres_reachable() -> bool:
    try:
        async with get_session() as s:
            from sqlalchemy import text

            await s.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.fixture
async def db_ready():
    """A clean journal on a reachable Postgres, or skip. Non-autouse: only the DB
    tests request it, so the pure-kernel tests above run with no database.

    Truncates on the way **out** as well as in. The Layer 2 tests `commit` real
    rows (an entry cycle journals a market snapshot per underlying), and
    `test_journal.py` isolates by unique tags rather than truncation — so a global
    query there (`daily_iv_history("SPY")`) would otherwise pick up this file's
    committed SPY snapshots. Cleaning up after ourselves keeps the leak in-file."""
    from tests.conftest import truncate_journal
    from vigil.db import session as session_module

    session_module.engine.cache_clear()
    session_module.session_factory.cache_clear()
    if not await _postgres_reachable():
        pytest.skip(f"no Postgres at {os.getenv('DATABASE_URL', 'localhost/vigil')}")
    async with get_session() as s:
        await truncate_journal(s)
    yield
    try:
        async with get_session() as s:
            await truncate_journal(s)
        await session_module.engine().dispose()
    finally:
        session_module.engine.cache_clear()
        session_module.session_factory.cache_clear()


@pytest.mark.db
async def test_a_fill_is_recorded_to_the_fills_table(
    put_credit_spread, flat_book, db_ready, monkeypatch
):
    """D-5: `record_fill` was written but never called, so every fill silently
    dropped its `fills` row. Drive the real `_submit` end-to-end and assert the row
    is there, tied to the entry order and flagged as a full (non-partial) fill."""
    # Freeze the clock `_submit` reads so its binding context passes Gate 11
    # deterministically regardless of when the suite runs.
    monkeypatch.setattr(S, "now_et", lambda: DEFAULT_NOW)

    client = FakeOrderClient(fill_qty=1)
    broker = FakeBroker(client)

    async with get_session() as db:
        ctx = S.RunnerContext(
            broker=broker,
            db=db,
            account_row_id=1,
            session_row_id=1,
            trading_date=DEFAULT_NOW.date(),
            risk=risk_config(),
            strategy=strategy_config(),
            ladder=ladder_config(),
            universe=("SPY",),
            pm=None,
        )
        result = S.CycleResult(kind=CycleKind.ENTRY)
        await S._submit(
            ctx, put_credit_spread, flat_book, result,
            available_symbols=frozenset(leg.symbol for leg in put_credit_spread.legs),
        )
        await db.commit()

    # The fill row exists, joins back to the entry order, and is not a partial.
    async with get_session() as db:
        from sqlalchemy import select

        entry = (
            await db.execute(select(OrderRow).where(OrderRow.intent == "open"))
        ).scalar_one()
        fill = (await db.execute(select(FillRow))).scalar_one()
        assert fill.order_id == entry.id
        assert fill.filled_qty == put_credit_spread.contracts
        assert fill.partial is False
        assert result.submitted == 1


# --------------------------------------------------------------------------- #
# Layer 2 — sense → reconcile → refresh → record, through the real run_entry.
#
# This is the layer a D-1-class bug needs. D-1 was: `reconcile` builds every open
# structure with `dollar_delta = 0` (a broker position carries no Greek), and the
# sense step must fold the chain's live deltas back onto the book before Gate 7 sums
# it — or the portfolio-wide delta gate silently sees a single trade. Neither the
# gate unit tests nor the StubBroker sessions could catch that, because neither runs
# the sense→refresh wire against a chain. This does.
#
# Candidate building is neutered (`build_for_regime -> None`): the wire under test is
# upstream of it — refresh and the equity snapshot both happen before Phase 2 — and
# stubbing it keeps the test off the brittle business of synthesising a chain that
# also happens to produce a gate-passing trade. Layer 1 already owns the submit path.
# --------------------------------------------------------------------------- #

class SenseBroker:
    """A broker rich enough for a whole `run_entry`: it answers the sense reads and
    holds an existing book, but never submits (Phase 2 is stubbed out)."""

    def __init__(
        self,
        *,
        chain: list,
        structures: tuple,
        equity: Decimal = Decimal(100_000),
        spot: Decimal = Decimal(755),
    ) -> None:
        self.client = object()  # verify_account is monkeypatched; never dereferenced
        self._chain = chain
        self._structures = structures
        self._equity = equity
        self._spot = spot

    async def account(self) -> AccountView:
        return AccountView(
            account_id="acct-test", equity=self._equity,
            last_equity=self._equity, status="ACTIVE",
        )

    async def structures(self) -> tuple:
        return self._structures

    async def order_statuses(self) -> list[tuple[str, str, str]]:
        # No working orders to reconcile in these sense→reconcile tests.
        return []

    async def spot(self, underlying: str) -> Decimal:
        return self._spot

    async def chain(self, underlying, *, spot, max_dte, strike_window):
        return list(self._chain)

    async def open_interest(self, underlying, *, spot, max_dte):
        return {c.occ.raw: 5000 for c in self._chain}


def _existing_book() -> OpenStructure:
    """One reconciled put spread whose legs a sensed chain can price. `dollar_delta`
    starts at 0 — the reconcile-time value D-1 left in place — so a passing test
    proves the refresh, not the fixture."""
    return OpenStructure(
        underlying="SPY",
        expiry=date(2026, 8, 28),
        strikes=(Decimal(754), Decimal(755)),
        max_loss=Decimal(800),
        dollar_delta=Decimal(0),
        has_resting_target=True,  # no §2.6 defect → reconcile needs no quotes
        structure=Structure.PUT_CREDIT_SPREAD,
        short_put_strikes=(Decimal(755),),
        net_credit=Decimal("0.20"),
        contracts=8,
        legs=(
            PositionLeg(symbol="SPY260828P00755000", ratio_qty=1, is_short=True),
            PositionLeg(symbol="SPY260828P00754000", ratio_qty=1, is_short=False),
        ),
    )


def _patch_bars(monkeypatch, *, spot: float = 755.0) -> None:
    """Stub the three network/disk reads `sense` makes off the underlying, so the
    sense step runs with no Alpaca and no bar store."""
    monkeypatch.setattr(sense_module, "daily_closes", lambda *a, **k: [spot] * 60)
    session = SessionBars(today_et(), [spot + (i % 2) * 0.5 for i in range(30)])
    monkeypatch.setattr(sense_module, "session_closes", lambda *a, **k: session)
    monkeypatch.setattr(sense_module, "rv_history", lambda *a, **k: [])


async def _run_one_entry(broker, monkeypatch):
    """Set up the account/session/cycle rows the FKs require, then run one entry
    cycle. Returns the finished `CycleResult` and the account row id."""
    monkeypatch.setattr(S, "verify_account", lambda **_k: "acct-test")
    monkeypatch.setattr(S, "verify_clock", lambda **_k: 0.0)
    # Phase 2 off: the wire under test ends at `record_equity`, upstream of building.
    monkeypatch.setattr(S, "build_for_regime", lambda *a, **k: None)

    async with get_session() as db:
        ctx = await S.open_context(broker, db)
        ctx.universe = ("SPY",)  # one underlying, so the assertions are unambiguous
        cycle = await J.start_cycle(
            db, session_id=ctx.session_row_id, kind=CycleKind.ENTRY.value
        )
        result = S.CycleResult(kind=CycleKind.ENTRY, cycle_id=cycle.id)
        await S.run_entry(ctx, result)
        await db.commit()
        return result, ctx.account_row_id


async def _latest_equity_delta(account_id: int) -> Decimal:
    from sqlalchemy import select

    async with get_session() as db:
        rows = (
            await db.execute(
                select(EquitySnapshot)
                .where(EquitySnapshot.account_id == account_id)
                .order_by(EquitySnapshot.id.desc())
            )
        ).scalars().all()
        assert rows, "run_entry must record an equity snapshot"
        return rows[0].net_dollar_delta


@pytest.mark.db
async def test_reconciled_book_delta_is_refreshed_before_it_is_recorded(
    make_contract, db_ready, monkeypatch
):
    """D-1: with the book's legs present in the sensed chain, the equity snapshot the
    kernel's Gate 7 reads carries a *live* portfolio delta — not the zero `reconcile`
    left behind. If the refresh is ever unwired, this records 0 and fails."""
    _patch_bars(monkeypatch)
    chain = [
        make_contract("SPY260828P00755000", delta=-0.16, iv=0.20, bid=0.50, ask=0.52),
        make_contract("SPY260828P00754000", delta=-0.11, iv=0.20, bid=0.30, ask=0.32),
        make_contract("SPY260828P00750000", delta=-0.05, iv=0.20, bid=0.10, ask=0.12),
    ]
    broker = SenseBroker(chain=chain, structures=(_existing_book(),))

    result, account_id = await _run_one_entry(broker, monkeypatch)

    assert await _latest_equity_delta(account_id) != 0  # the D-1 claim
    assert result.regime is not None  # phase-1 regime was journalled for the symbol


@pytest.mark.db
async def test_delta_stays_zero_and_is_flagged_when_a_leg_is_off_chain(
    make_contract, db_ready, monkeypatch
):
    """The negative control: strip the book's legs out of the chain and the refresh
    has nothing to price it from — the delta stays 0 *and* the cycle says so, rather
    than passing a zero off as a real measurement. Proves the positive result above is
    data-driven, not a constant."""
    _patch_bars(monkeypatch)
    # A chain with a strike, so `sense` returns a view, but not the book's own legs.
    chain = [make_contract("SPY260828P00750000", delta=-0.05, iv=0.20, bid=0.10, ask=0.12)]
    broker = SenseBroker(chain=chain, structures=(_existing_book(),))

    result, account_id = await _run_one_entry(broker, monkeypatch)

    assert await _latest_equity_delta(account_id) == 0
    assert any("delta unresolved" in w for w in result.warnings)
