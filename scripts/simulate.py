"""Offline full-day simulator: the whole pipeline on synthetic data, no network.

`make dry-run` runs sense→gate against *live* quotes and stops before submit; the
test suite drives the pipeline offline but in slices (test_sessions.py stubs the
submit, test_integration_broker.py neuters the builder). Nothing runs one
continuous simulated day end to end — sense → regime → build → gate → submit →
fill → manage → **target fills** → flatten — and prints what happened. This does.

**How it stays honest.** It does *not* re-implement the agent. It subclasses the
real `worker.broker.Broker`, overriding only the four synthetic *reads*
(spot/chain/quotes/open_interest); `account`, `structures`, `submit_entry`,
`submit_close` are inherited and route through the real `execution/router.py`, the
real 12-gate kernel, the real price ladder and the real reconcile. The only fakes
are a `SimClient` (a duck-typed `TradingClient` holding a position book) and a
Black-Scholes chain priced by the project's own `data/greeks.py`. So a wiring bug
between the runner and the kernel shows up here exactly as it would in production.

Needs the journal database (like `make test`): it drives the real `run_cycle`,
which journals every cycle. Paper/synthetic only — it never touches Alpaca.

    make sim                      # a scripted trending day
    make sim ARGS="--regime chop" # force a chop day (iron condors)
    make sim ARGS="--seed 7"      # a different price path
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from types import SimpleNamespace

from vigil.clock import ET, today_et
from vigil.data.bars import SessionBars
from vigil.data.chain import Contract
from vigil.data.greeks import bs_price, year_fraction
from vigil.worker import sense as sense_module
from vigil.worker import sessions as S
from vigil.worker.broker import Broker
from vigil.worker.schedule import CycleKind

RATE = 0.045                         # the risk-free rate the greeks model assumes
HALF_SPREAD = Decimal("0.01")        # half the bid-ask, in dollars — tight enough for Gate 8
CONTRACT = Decimal(100)              # one option = 100 shares
UNIVERSE = ("SPY", "QQQ")

# A base implied vol per name and a strike grid. Higher IV lifts premium, which is
# what lets a 0-2 DTE credit structure clear the Gate 9 floor at all.
BASE_IV = {"SPY": 0.19, "QQQ": 0.24}
BASE_SPOT = {"SPY": Decimal("765"), "QQQ": Decimal("715")}


# --------------------------------------------------------------------------- #
# Synthetic chain — priced by the project's own Black-Scholes, so `sense`, the
# strike selector and the kernel see exactly the shape they expect.
# --------------------------------------------------------------------------- #

def _occ(underlying: str, expiry: date, is_put: bool, strike: Decimal) -> str:
    """Build an OCC symbol: underlying + YYMMDD + C/P + strike x1000 padded to 8."""
    right = "P" if is_put else "C"
    return f"{underlying}{expiry:%y%m%d}{right}{int(strike * 1000):08d}"


def _contract(
    underlying: str, expiry: date, is_put: bool, strike: Decimal,
    *, spot: Decimal, iv: float, now: datetime,
) -> Contract | None:
    """One synthetic contract: a BS mid, a tight quote around it, modelled greeks.

    Duck-typed snapshot rather than the alpaca `OptionsSnapshot` model — `Contract`
    only reads `.latest_quote.bid_price/.ask_price`, `.greeks` and
    `.implied_volatility` off it, and `worker.broker._model` accepts anything that
    is not a raw dict or string. The delta is attached by the real `solve` (below),
    the same code path the live feed's missing greeks take (A1).
    """
    t = year_fraction(expiry, now)
    price = Decimal(str(bs_price(
        spot=float(spot), strike=float(strike), t=t, sigma=iv, rate=RATE, is_put=is_put,
    )))
    if price <= 0:
        return None
    bid = max(price - HALF_SPREAD, Decimal("0.01"))
    ask = price + HALF_SPREAD
    snapshot = SimpleNamespace(
        latest_quote=SimpleNamespace(bid_price=float(bid), ask_price=float(ask)),
        greeks=None,                 # force the modelled path, exactly like the live feed
        implied_volatility=None,
    )
    from vigil.data.chain import (
        _with_greeks,  # noqa: PLC0415 — private BS-fallback helper, reused on purpose
    )

    return _with_greeks(
        Contract(occ=_parse(underlying, expiry, is_put, strike), snapshot=snapshot),
        spot=spot, now=now, rate=RATE,
    )


def _parse(underlying: str, expiry: date, is_put: bool, strike: Decimal):
    from vigil.data.occ import parse_occ  # noqa: PLC0415

    return parse_occ(_occ(underlying, expiry, is_put, strike))


def build_chain(
    underlying: str, *, spot: Decimal, expiries: list[date], iv: float, now: datetime,
    window: int = 12,
) -> list[Contract]:
    """A +/- `window`-dollar grid of $1 strikes, puts and calls, across `expiries`."""
    lo = int(spot) - window
    hi = int(spot) + window
    out: list[Contract] = []
    for expiry in expiries:
        for k in range(lo, hi + 1):
            for is_put in (True, False):
                c = _contract(underlying, expiry, is_put, Decimal(k),
                              spot=spot, iv=iv, now=now)
                if c is not None:
                    out.append(c)
    return out


def _mid(underlying: str, symbol: str, spot: Decimal, iv: float, now: datetime) -> Decimal:
    """Current mid for one OCC symbol — the mark used for fills and P&L."""
    occ = _parse_raw(symbol)
    t = year_fraction(occ.expiry, now)
    price = Decimal(str(bs_price(
        spot=float(spot), strike=float(occ.strike), t=t, sigma=iv, rate=RATE,
        is_put=occ.is_put,
    )))
    return max(price, Decimal("0.01"))


def _parse_raw(symbol: str):
    from vigil.data.occ import parse_occ  # noqa: PLC0415

    return parse_occ(symbol)


# --------------------------------------------------------------------------- #
# SimClient — a duck-typed TradingClient with a position book. No network.
# --------------------------------------------------------------------------- #

@dataclass
class _Pos:
    symbol: str
    qty: Decimal              # signed: negative = short
    avg: Decimal             # per-contract entry price


@dataclass
class _RestingOrder:
    id: str
    client_order_id: str
    legs: list                # each: symbol, position_intent
    limit_price: float


class SimClient:
    """Stands in for alpaca-py's TradingClient. Deterministic fills, a real book.

    Opens fill immediately (rung 1). A GTC close is a *resting* target — it is
    parked, not filled, until the harness decides the mark has reached it. A `day`
    close (management / flatten) fills at once. Every fill updates the book, so the
    real `Broker.structures()` reconstructs the same positions the agent will see.
    """

    def __init__(self) -> None:
        self.positions: dict[str, _Pos] = {}
        self.resting: dict[str, _RestingOrder] = {}
        self.realized = Decimal(0)
        self._n = 0
        self.mark: dict[str, Decimal] = {}   # symbol -> current mid, refreshed each step
        self._orders_by_id: dict[str, object] = {}   # so a poll echoes the submitted order

    # ---- the account read the portfolio gates need -------------------------- #
    def get_account(self):
        equity = Decimal(100_000) + self.realized + self._unrealized()
        return SimpleNamespace(
            id="SIM0000-0000-0000-0000-000000000000",
            equity=str(equity.quantize(Decimal("0.01"))),
            last_equity="100000",
            status="ACTIVE",
        )

    def _unrealized(self) -> Decimal:
        total = Decimal(0)
        for p in self.positions.values():
            m = self.mark.get(p.symbol, p.avg)
            total += (m - p.avg) * p.qty * CONTRACT
        return total

    def get_all_positions(self):
        return [
            SimpleNamespace(symbol=p.symbol, qty=str(p.qty), avg_entry_price=str(p.avg))
            for p in self.positions.values() if p.qty != 0
        ]

    def get_orders(self, req=None):
        # `nested=True` semantics: an mleg is one order carrying its legs.
        return [
            SimpleNamespace(
                id=o.id, client_order_id=o.client_order_id, symbol=None,
                legs=[SimpleNamespace(symbol=lg["symbol"],
                                      position_intent=SimpleNamespace(value=lg["intent"]))
                      for lg in o.legs],
                limit_price=o.limit_price, status=SimpleNamespace(value="new"),
                time_in_force=SimpleNamespace(value="gtc"), type=SimpleNamespace(value="limit"),
                order_class=SimpleNamespace(value="mleg"),
            )
            for o in self.resting.values()
        ]

    # ---- the order side the router drives ---------------------------------- #
    def submit_order(self, req):
        self._n += 1
        oid = f"sim-{self._n}"
        legs = list(req.legs)
        intents = [lg.position_intent.value for lg in legs]
        closing = any("close" in i for i in intents)
        tif = getattr(req.time_in_force, "value", str(req.time_in_force))

        if closing and tif == "gtc":
            # A resting profit target. Park it; do not fill.
            self.resting[oid] = _RestingOrder(
                id=oid, client_order_id=req.client_order_id,
                legs=[{"symbol": lg.symbol, "intent": lg.position_intent.value} for lg in legs],
                limit_price=float(req.limit_price),
            )
            order = _order(oid, req.client_order_id, "new", 0, None, req.limit_price)
        elif closing:
            # A management / flatten close — fill immediately against the book.
            self._apply_close(legs)
            order = _order(oid, req.client_order_id, "filled",
                           _pkg_qty(req), str(req.limit_price), req.limit_price)
        else:
            # An opening rung. Fill it fully and add the legs. A credit fills at a
            # negative avg (the O-1 convention); the router abs()es it.
            qty = _pkg_qty(req)
            self._apply_open(legs, qty)
            order = _order(oid, req.client_order_id, "filled", qty,
                           f"-{abs(float(req.limit_price)):.2f}", req.limit_price)

        self._orders_by_id[oid] = order
        return order

    def get_order_by_id(self, order_id: str):
        # The router polls after submitting; echo the exact order we returned, so
        # the coid and fill it journals are the ones it actually placed.
        return self._orders_by_id[order_id]

    def cancel_order_by_id(self, order_id: str) -> None:
        self.resting.pop(order_id, None)

    def cancel_orders(self) -> None:
        self.resting.clear()

    # ---- book mechanics ---------------------------------------------------- #
    def _apply_open(self, legs, qty: int) -> None:
        for lg in legs:
            sign = Decimal(-1) if "sell" in lg.position_intent.value else Decimal(1)
            dq = sign * Decimal(qty) * Decimal(getattr(lg, "ratio_qty", 1))
            price = self.mark.get(lg.symbol, Decimal("0.10"))
            self._add(lg.symbol, dq, price)

    def _apply_close(self, legs) -> None:
        for lg in legs:
            p = self.positions.get(lg.symbol)
            if p is None:
                continue
            m = self.mark.get(lg.symbol, p.avg)
            self.realized += (m - p.avg) * p.qty * CONTRACT
            self.positions.pop(lg.symbol, None)

    def _add(self, symbol: str, dq: Decimal, price: Decimal) -> None:
        p = self.positions.get(symbol)
        if p is None:
            self.positions[symbol] = _Pos(symbol=symbol, qty=dq, avg=price)
            return
        new_qty = p.qty + dq
        if new_qty == 0:
            self.realized += (price - p.avg) * -dq * CONTRACT
            self.positions.pop(symbol, None)
        else:
            p.qty = new_qty

    def fill_reached_targets(self) -> list[str]:
        """Fill any resting target whose structure can now be bought back at its
        limit. Returns the client_order_ids filled — the harness narrates them."""
        filled: list[str] = []
        for oid, o in list(self.resting.items()):
            debit = Decimal(0)
            priceable = True
            for lg in o.legs:
                m = self.mark.get(lg["symbol"])
                if m is None:
                    priceable = False
                    break
                # to close: buy back a short (+), sell a long (-)
                debit += m if "buy" in lg["intent"] else -m
            if not priceable or debit > Decimal(str(o.limit_price)):
                continue
            self._apply_close([SimpleNamespace(
                symbol=lg["symbol"],
                position_intent=SimpleNamespace(value=lg["intent"])) for lg in o.legs])
            self.resting.pop(oid, None)
            filled.append(o.client_order_id)
        return filled


def _order(oid, coid, status, filled_qty, filled_avg, limit):
    return SimpleNamespace(
        id=oid, client_order_id=coid, status=SimpleNamespace(value=status),
        filled_qty=str(filled_qty), filled_avg_price=filled_avg, limit_price=str(limit),
    )


def _pkg_qty(req) -> int:
    return int(getattr(req, "qty", 1) or 1)


# --------------------------------------------------------------------------- #
# SimBroker — the real Broker with the four network reads made synthetic.
# --------------------------------------------------------------------------- #

class SimBroker(Broker):
    def __init__(self, client: SimClient, world: World) -> None:
        super().__init__(client=client)
        self.world = world

    async def spot(self, underlying: str) -> Decimal:
        return self.world.spots[underlying]

    async def chain(self, underlying, *, spot, max_dte, strike_window=Decimal(12)):
        return self.world.chain(underlying)

    async def open_interest(self, underlying, *, spot, max_dte):
        # Deep, liquid OI everywhere — Gate 8/6 read this; the sim is not testing
        # a thin book (test_gates.py owns that).
        return {c.occ.raw: 8000 for c in self.world.chain(underlying)}

    async def quotes(self, symbols):
        out = {}
        for s in symbols:
            occ = _parse_raw(s)
            m = _mid(occ.underlying, s, self.world.spots[occ.underlying],
                     self.world.iv(occ.underlying), self.world.now)
            out[s] = (m - HALF_SPREAD, m + HALF_SPREAD)
        return out


# --------------------------------------------------------------------------- #
# World — the mutable market the harness advances between cycles.
# --------------------------------------------------------------------------- #

@dataclass
class World:
    now: datetime
    day: date
    spots: dict[str, Decimal]
    trend: Decimal                      # per-cycle drift applied to spots
    client: SimClient
    iv_scale: float = 1.0               # a vol crush (<1) collapses extrinsic value
    _chains: dict[str, list[Contract]] = field(default_factory=dict)

    def iv(self, underlying: str) -> float:
        return BASE_IV[underlying] * self.iv_scale

    def expiries(self) -> list[date]:
        return [self.day, self.day + timedelta(days=1), self.day + timedelta(days=2)]

    def rebuild(self) -> None:
        self._chains = {
            u: build_chain(u, spot=self.spots[u], expiries=self.expiries(),
                           iv=self.iv(u), now=self.now)
            for u in UNIVERSE
        }
        # Refresh every mark the client holds, plus every strike it might trade.
        self.client.mark = {
            c.occ.raw: (c.mid or Decimal("0.01"))
            for u in UNIVERSE for c in self._chains[u]
        }

    def chain(self, underlying: str) -> list[Contract]:
        return self._chains[underlying]

    def advance(self, minutes: int, *, drift: Decimal, iv_scale: float | None = None) -> None:
        self.now = self.now + timedelta(minutes=minutes)
        if iv_scale is not None:
            self.iv_scale = iv_scale
        for u in UNIVERSE:
            self.spots[u] = (self.spots[u] * (Decimal(1) + drift)).quantize(Decimal("0.01"))
        self.rebuild()


# --------------------------------------------------------------------------- #
# The trending / chop bar stubs `sense` needs off the underlying.
# --------------------------------------------------------------------------- #

def _install_bar_stubs(world: World, regime: str) -> None:
    """Stub the three underlying reads `sense` makes, so it runs with no bar store.

    A rising close series produces TREND_UP (broken-wing condor); a flat one
    produces CHOP (iron condor). This is the only place the regime is steered.
    """
    slope = {"trend_up": 1.0, "trend_down": -1.0, "chop": 0.0}.get(regime, 1.0)

    def daily_closes(underlying, *_a, **_k):
        base = float(world.spots[underlying])
        # 60 daily closes drifting into `base` at ~`slope` * 0.15%/day. Floats, like
        # the real reader — the indicators do float EMA math.
        return [base * (1 - slope * 0.0015 * (59 - i)) for i in range(60)]

    def session_closes(underlying, *_a, **_k):
        base = float(world.spots[underlying])
        pts = [base * (1 + slope * 0.0004 * i) for i in range(30)]
        return SessionBars(world.day, pts)

    sense_module.daily_closes = daily_closes            # type: ignore[assignment]
    sense_module.session_closes = session_closes        # type: ignore[assignment]
    sense_module.rv_history = lambda *_a, **_k: []       # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# The harness.
# --------------------------------------------------------------------------- #

class RealJournalRefused(SystemExit):
    """The target database holds the locked trading account — truncation refused."""


async def _guard_not_the_real_journal() -> None:
    """Refuse to run if this database holds the locked live account (hard rule #7).

    `make sim` truncates the whole journal for a clean slate. Against the local
    scratch DB that is harmless — but a `DATABASE_URL` pointed at the container's
    real journal (5433) would wipe the actual trading record. So before touching
    anything, look for an account row matching `config/account.lock`; its presence
    means we are aimed at the real journal, and we abort instead of truncating.

    The check runs in its own session so an unmigrated scratch DB (no `accounts`
    table) fails open — there is nothing real there to protect.
    """
    from sqlalchemy import text

    from vigil.account import read_lock
    from vigil.db.session import get_session

    locked = read_lock()
    if locked is None:                       # no lock written → nothing to protect
        return
    async with get_session() as db:
        try:
            hit = (await db.execute(
                text("SELECT 1 FROM accounts WHERE alpaca_account_id = :a LIMIT 1"),
                {"a": locked},
            )).first()
        except Exception:                    # noqa: BLE001 — no accounts table = scratch DB
            hit = None
    if hit is not None:
        raise RealJournalRefused(
            f"REFUSING TO RUN: this database holds the locked trading account "
            f"({locked}). `make sim` truncates the journal and must never touch the "
            f"real one. Unset DATABASE_URL to use the local scratch DB, or point it "
            f"at a throwaway database."
        )


async def _reset_journal() -> None:
    """Empty every journal table so each run is a clean slate (like the test suite).

    The table list is the mapped metadata, never a hand-kept literal — a table
    added to `db/models.py` is truncated here automatically. Guarded: it refuses
    to run against the locked trading account's journal (see the guard above).
    """
    from sqlalchemy import text

    from vigil.db.models import Base
    from vigil.db.session import get_session

    await _guard_not_the_real_journal()
    names = ", ".join(
        t.name for t in Base.metadata.sorted_tables if t.name != "alembic_version"
    )
    async with get_session() as db:
        await db.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))


def _hms(dt: datetime) -> str:
    return dt.strftime("%H:%M")


async def _run(kind: CycleKind, broker: SimBroker) -> S.CycleResult:
    # Point the clock the whole agent reads at the sim's `now`; keep the account
    # lock and clock guard out of the way (there is no real account to verify).
    S.now_et = lambda: broker.world.now                 # type: ignore[assignment]
    S.verify_account = lambda **_k: "SIM"               # type: ignore[assignment]
    S.verify_clock = lambda **_k: 0.0                   # type: ignore[assignment]
    return await S.run_cycle(kind, broker=broker)


def _print_cycle(world: World, kind: CycleKind, result: S.CycleResult, note: str = "") -> None:
    acct = world.client.get_account()
    equity = Decimal(acct.equity)
    pnl = equity - Decimal(100_000)
    spots = " ".join(f"{u} {world.spots[u]}" for u in UNIVERSE)
    tag = f"  ({note})" if note else ""
    print(f"\n[{_hms(world.now)}] {kind.value.upper():9} regime={result.regime or '-':10} "
          f"{spots}  equity {equity}  P&L {pnl:+}{tag}")
    print(f"    proposals={result.proposals} approved={result.approved} "
          f"submitted={result.submitted} closed={result.closed}")
    if result.notes:
        print(f"    · {result.summary}")
    for w in result.warnings:
        print(f"    ! {w}")


def _print_book(world: World) -> None:
    structs: dict[str, list[str]] = {}
    for p in world.client.positions.values():
        structs.setdefault(_parse_raw(p.symbol).underlying, []).append(
            f"{p.symbol} {p.qty:+}")
    if not any(structs.values()):
        print("    book: flat")
        return
    resting = len(world.client.resting)
    for u, legs in structs.items():
        print(f"    book {u}: {len(legs)} legs, {resting} resting target(s) total")


async def simulate(regime: str, seed: int) -> int:
    import logging
    import random

    from vigil.logging import configure

    # Quiet the per-cycle JSON logs (they go to stderr) so the narrative reads
    # clean; warnings — §2.6 defects, failed cycles — still surface.
    configure(level=logging.WARNING)

    rng = random.Random(seed)
    await _reset_journal()               # clean slate — no rows bleed between runs
    day = today_et()
    start = datetime.combine(day, time(9, 45), tzinfo=ET)
    client = SimClient()
    world = World(now=start, day=day, spots=dict(BASE_SPOT), trend=Decimal("0.001"),
                  client=client)
    world.rebuild()
    _install_bar_stubs(world, regime)
    broker = SimBroker(client, world)

    drift = {"trend_up": Decimal("0.0006"), "trend_down": Decimal("-0.0006"),
             "chop": Decimal("0.0001")}.get(regime, Decimal("0.0006"))

    print("=" * 72)
    print(f" VIGIL offline day simulator — regime={regime} seed={seed} date={day}")
    print(" synthetic chain (BS-priced), real kernel + ladder + reconcile, no network")
    print("=" * 72)

    # A scripted day, sequenced so each structure's target fills (or is flattened)
    # before the next entry — two live structures on one underlying+expiry would
    # merge at the broker into one un-exitable blob, which is a real constraint, not
    # something to paper over. (kind, minutes-to-advance, drift, iv_scale, note)
    plan = [
        (CycleKind.OPEN,     0,   drift,             1.0,  ""),
        (CycleKind.ENTRY,    45,  drift,             1.0,  "first entry"),
        (CycleKind.MANAGE,   5,   drift,             1.0,  "held"),
        # A volatility crush: extrinsic value collapses, so the resting 50% target
        # becomes reachable and fills — the §2.6 exit doing its job.
        (CycleKind.MANAGE,   90,  Decimal("0.0001"), 0.30, "vol crush → target fills"),
        # Vol normalises; the book is flat again, so a second entry is clean.
        (CycleKind.ENTRY,    60,  Decimal("0.0002"), 1.0,  "second entry"),
        (CycleKind.MANAGE,   5,   drift,             1.0,  "held"),
        (CycleKind.FLATTEN,  120, Decimal("0.0001"), 1.0,  "15:40 hard stop"),
        # Flatten cancels every working order first, so a surviving future-expiry
        # structure is momentarily left without its target — a §2.6 defect the next
        # sweep repairs by re-resting it (the auto-re-rest this branch added).
        (CycleKind.MANAGE,   10,  Decimal("0"),      1.0,  "re-rest the stripped target"),
        (CycleKind.POSTCLOSE, 25, Decimal("0"),      1.0,  "close the books"),
    ]

    for kind, mins, d, ivs, note in plan:
        if mins:
            # jitter the drift a little so no two runs are identical
            world.advance(mins, drift=d + Decimal(str(rng.uniform(-0.0002, 0.0002))),
                          iv_scale=ivs)
        # Between cycles the market moved — let any reached target fill before the
        # agent looks, so reconcile sees it vanish (the §2.6 "closed unobserved" path).
        for coid in client.fill_reached_targets():
            print(f"\n    ~ [{_hms(world.now)}] resting profit target FILLED at the broker: {coid}")
        result = await _run(kind, broker)
        _print_cycle(world, kind, result, note)
        _print_book(world)

    acct = client.get_account()
    pnl = Decimal(acct.equity) - Decimal(100_000)
    cents = Decimal("0.01")
    realized = client.realized.quantize(cents)
    open_legs = len([p for p in client.positions.values() if p.qty])
    print("\n" + "=" * 72)
    print(f" DONE  final equity {acct.equity}  day P&L {pnl.quantize(cents):+}  "
          f"realized {realized:+}  unrealized {(pnl - realized):+}  open legs {open_legs}")
    print("=" * 72)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline full-day pipeline simulator.")
    ap.add_argument("--regime", default="trend_up",
                    choices=["trend_up", "trend_down", "chop"],
                    help="steer the synthetic tape (default: trend_up)")
    ap.add_argument("--seed", type=int, default=1, help="price-path seed")
    args = ap.parse_args()
    return asyncio.run(simulate(args.regime, args.seed))


if __name__ == "__main__":
    raise SystemExit(main())
