"""The async boundary around a synchronous SDK.

**Why this file exists.** `alpaca-py` is blocking, and the worker's cycle is
`async`. Calling `client.get_account()` directly inside a coroutine does not just
make *that* call slow — it stops the event loop, so nothing else in the process
runs until the HTTP round trip returns. During a manage sweep over six structures
that is the difference between a 15:40 flatten firing at 15:40 and firing whenever
the last blocking call happened to finish.

`asyncio.to_thread` hands each blocking call to a worker thread and awaits its
result, which is the standard way to use a synchronous library from async code.
It is not free (a thread hop per call), but broker latency is measured in tens of
milliseconds and thread-hand-off in microseconds, so the trade is not close.

**It is also the seam that makes the session runners testable.** Everything in
`sessions.py` talks to this class and nothing else talks to Alpaca, so a test
substitutes a fake `Broker` and exercises the real cycle logic with no network —
the same discipline the risk kernel gets from taking inert data.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.models import Order, Position, TradeAccount
from alpaca.trading.requests import GetOrdersRequest

from vigil.config import RiskConfig, StrategyConfig
from vigil.data.chain import (
    Contract,
    fetch_chain,
    fetch_open_interest,
    fetch_quotes,
    spot_price,
)
from vigil.domain import OpenStructure, PortfolioState, TradeProposal
from vigil.execution.reconcile import BrokerPosition, RestingOrder, group_positions
from vigil.execution.router import ExecutionResult, submit_close, submit_entry
from vigil.risk.context import KernelContext


@dataclass(frozen=True, slots=True)
class AccountView:
    """The account fields the portfolio gates need. Nothing else."""

    account_id: str
    equity: Decimal
    last_equity: Decimal
    status: str

    @property
    def day_pnl(self) -> Decimal:
        return self.equity - self.last_equity


# PEP 695 type-parameter syntax (Python 3.12): `[T]` declares the type variable
# inline, so it needs no module-level `TypeVar` and is scoped to this function.
def _model[T](result: T | dict[str, object] | str) -> T:
    """Narrow alpaca-py's `Model | dict | str` return types.

    The SDK returns raw dicts (and, for positions, bare symbol strings) when the
    client is in raw-data mode. Vigil never enables it, so anything but a model
    here means a misconfigured client — the same reasoning as
    `execution.router._as_order`, and worth failing loudly rather than
    duck-typing into an `AttributeError` three frames later.
    """
    if isinstance(result, dict | str):
        raise TypeError("TradingClient returned raw data; Vigil requires model objects")
    return result


def _dec(value: object, default: str = "0") -> Decimal:
    """Alpaca returns money as strings. Parse via `str`, never via `float`.

    `Decimal(float(x))` is the classic way to reintroduce binary rounding into a
    number that arrived as exact decimal text — `Decimal(0.1)` is not `0.1`.
    """
    if value is None:
        return Decimal(default)
    return Decimal(str(value))


class Broker:
    """Every broker call the worker makes, awaited off the event loop.

    Holds no trading state. It is a translator, not a cache: the point of §2.3 is
    that the broker is the source of truth, and a layer that remembered what it
    last saw would quietly become a second, staler source.
    """

    def __init__(self, client: TradingClient | None = None) -> None:
        # Constructed lazily so importing this module needs no credentials.
        self._client = client

    @property
    def client(self) -> TradingClient:
        if self._client is None:
            from vigil.data.alpaca_client import trading_client

            self._client = trading_client()
        return self._client

    # ---- reads ------------------------------------------------------------ #

    async def account(self) -> AccountView:
        acct: TradeAccount = _model(await asyncio.to_thread(self.client.get_account))
        equity = _dec(acct.equity)
        return AccountView(
            account_id=str(acct.id),
            equity=equity,
            # `last_equity` is the previous session's close, so `equity -
            # last_equity` is the day's P&L — the number Gate 3 halts on. It can
            # be absent on a brand-new account, where a day P&L of zero is the
            # honest answer rather than a fabricated one.
            last_equity=_dec(acct.last_equity, str(equity)),
            status=str(getattr(acct.status, "value", acct.status)),
        )

    async def positions(self) -> list[BrokerPosition]:
        raw = await asyncio.to_thread(self.client.get_all_positions)
        positions: list[Position] = [_model(p) for p in raw]
        return [
            BrokerPosition(
                symbol=p.symbol,
                qty=_dec(p.qty),
                avg_entry_price=_dec(p.avg_entry_price),
            )
            for p in positions
        ]

    async def open_orders(self) -> list[Order]:
        # `nested=True` returns an mleg package as one order with its legs
        # attached, rather than as four unrelated tickets — which is what
        # reconciliation needs to decide whether a *structure* has a resting exit.
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
        raw = await asyncio.to_thread(self.client.get_orders, req)
        return [_model(o) for o in raw]

    async def structures(self) -> tuple[OpenStructure, ...]:
        """Rebuild the book from broker truth, with resting-target flags attached.

        Positions and orders are fetched **concurrently**: they are independent
        reads, and `asyncio.gather` over two threads costs one round trip instead
        of two. On a 15-minute sweep that is cosmetic; inside the 15:40 flatten it
        is not.
        """
        positions, orders = await asyncio.gather(self.positions(), self.open_orders())
        return group_positions(positions, resting=[_as_resting(o) for o in orders])

    async def spot(self, underlying: str) -> Decimal:
        return await asyncio.to_thread(spot_price, underlying)

    async def chain(
        self, underlying: str, *, spot: Decimal, max_dte: int,
        strike_window: Decimal = Decimal(25),
    ) -> list[Contract]:
        return await asyncio.to_thread(
            fetch_chain, underlying, spot=spot, max_dte=max_dte,
            strike_window=strike_window,
        )

    async def open_interest(
        self, underlying: str, *, spot: Decimal, max_dte: int
    ) -> dict[str, int]:
        return await asyncio.to_thread(
            fetch_open_interest, underlying, spot=spot, max_dte=max_dte
        )

    async def quotes(self, symbols: Sequence[str]) -> dict[str, tuple[Decimal, Decimal]]:
        """Two-sided quotes for specific OCC symbols — how a close is priced.

        On this class rather than called directly from `sessions.py`, and that is
        not a style preference: **this class is the only seam through which the
        agent reaches the network**, and a session runner importing
        `fetch_quotes` puts a live API call inside cycle logic that is supposed
        to be testable without one. It was, briefly, and the test that caught it
        was passing by silently calling the real Alpaca endpoint.
        """
        return await asyncio.to_thread(fetch_quotes, list(symbols))

    # ---- writes. All of them route through execution/router.py (hard rule #4). #

    async def submit_entry(
        self,
        proposal: TradeProposal,
        state: PortfolioState,
        context: KernelContext,
        *,
        risk: RiskConfig | None = None,
        strategy: StrategyConfig | None = None,
    ) -> ExecutionResult:
        """Gate and submit. **The kernel runs inside `submit_entry`, not here.**

        Worth being explicit about: this method looks like a place one could add
        a "quick check" before calling through, and that is precisely how a second
        submit path is born. It forwards, and nothing else.
        """
        return await asyncio.to_thread(
            submit_entry, proposal, state, context,
            client=self.client, risk=risk, strategy=strategy,
        )

    async def submit_close(
        self, structure: OpenStructure, limit_price: Decimal, *, reason: str
    ) -> Order:
        return await asyncio.to_thread(
            submit_close, structure, limit_price, client=self.client, reason=reason
        )

    async def cancel_order(self, order_id: str) -> None:
        await asyncio.to_thread(self.client.cancel_order_by_id, order_id)

    async def cancel_all_orders(self) -> None:
        await asyncio.to_thread(self.client.cancel_orders)


def _as_resting(order: Order) -> RestingOrder:
    """Reduce an SDK order to what reconciliation asks of it.

    "Closing" is read from each leg's `position_intent` rather than from the
    order's side, because an mleg package has no single side — an iron condor's
    exit buys two legs and sells two. Falling back to the symbol set alone would
    mark an *entry* ticket as a resting exit and hide a §2.6 defect.
    """
    legs = getattr(order, "legs", None) or []
    symbols = {getattr(leg, "symbol", "") for leg in legs} if legs else set()
    if not symbols and order.symbol:
        symbols = {order.symbol}
    intents = [str(getattr(leg, "position_intent", "") or "") for leg in legs]
    if not intents:
        intents = [str(getattr(order, "position_intent", "") or "")]
    is_closing = any("close" in intent.lower() for intent in intents)
    return RestingOrder(
        order_id=str(order.id), symbols=frozenset(symbols), is_closing=is_closing
    )


def portfolio_state(
    account: AccountView,
    structures: tuple[OpenStructure, ...],
    *,
    peak_equity: Decimal | None = None,
    halted: bool = False,
    known_client_order_ids: frozenset[str] = frozenset(),
) -> PortfolioState:
    """Assemble the kernel's view of the book. Pure.

    `peak_equity` comes from the journal, not from this process. A worker
    restarted after a drawdown has no memory of the high-water mark, and
    defaulting to the current equity would compute a drawdown of zero — silently
    disarming Gate 4, the one gate whose whole job is the unrecoverable day. When
    the journal genuinely has no history, today's equity is the honest floor.
    """
    return PortfolioState(
        equity=account.equity,
        peak_equity=max(peak_equity or account.equity, account.equity),
        day_pnl=account.day_pnl,
        open_structures=structures,
        halted=halted,
        known_client_order_ids=known_client_order_ids,
    )


def expiries_today(structures: tuple[OpenStructure, ...], today: date) -> tuple[OpenStructure, ...]:
    """Structures expiring today — the ones auto-exercise makes non-negotiable."""
    return tuple(s for s in structures if s.expiry <= today)
