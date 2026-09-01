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
import zlib
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from random import Random
from types import SimpleNamespace

from vigil.clock import ET, today_et
from vigil.data.bars import SessionBars
from vigil.data.chain import Contract
from vigil.data.greeks import bs_price, year_fraction
from vigil.execution.reconcile import structures_missing_targets
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

CENT = Decimal("0.01")


# --------------------------------------------------------------------------- #
# Frictions — the punches a real market throws that a clean BS chain does not.
#
# CLAUDE.md is explicit that paper P&L is optimistic: fills are generous, the
# options feed is indicative, ~10% of fills arrive partial, and spreads widen when
# you least want them to. A frictionless sim that prints a 100% win rate is worse
# than no sim, because it launders those risks into a green check. This models the
# four that move outcomes; everything here lives in the harness, never in `src/`.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class Frictions:
    """Per-run market frictions. `enabled=False` is the clean baseline the batch
    compares against — uniform 1¢ quotes, fills at the touch, no partials or gaps.

    `rng` drives the *stochastic* frictions (partials, gaps), so a `--seed` still
    reproduces a run to the cent. Spreads are deliberately **not** rng-driven: they
    are a deterministic function of the strike (via `crc32`), so the chain the
    kernel gates and the quotes the router fills against can never disagree within a
    cycle without threading a shared cache through every call site.
    """

    rng: Random
    enabled: bool = True
    base_half_spread: Decimal = Decimal("0.012")  # ATM half-spread, dollars
    moneyness_k: float = 7.0                        # how fast the spread widens OTM
    jitter_lo: float = 0.6
    jitter_hi: float = 1.5
    partial_prob: float = 0.10                     # §1.2: ~10% of fills are partial
    gap_prob: float = 0.20                         # a gap on ~1 in 5 cycle advances
    gap_size: Decimal = Decimal("0.006")           # ~0.6% shock, enough to breach

    def half_spread(self, strike: Decimal, spot: Decimal) -> Decimal:
        """Half the bid-ask for one contract: wider OTM, wider on illiquid strikes.

        The `crc32` term is a stable per-strike liquidity draw in [0,1) — the same
        strike always gets the same jitter, so bid/ask are reproducible and the
        kernel's liquidity view matches the fill engine's. Widening with moneyness
        is what makes a condor's long wings the legs most likely to trip Gate 8.
        """
        if not self.enabled:
            return HALF_SPREAD
        moneyness = abs(float(strike) - float(spot)) / float(spot)
        base = float(self.base_half_spread) * (1.0 + self.moneyness_k * moneyness)
        draw = (zlib.crc32(str(strike).encode()) & 0xFFFF) / 0xFFFF
        jitter = self.jitter_lo + draw * (self.jitter_hi - self.jitter_lo)
        return Decimal(str(round(max(base * jitter, 0.01), 2)))

    def fill_contracts(self, contracts: int) -> int:
        """How many of `contracts` actually fill — occasionally a partial (§1.2)."""
        if not self.enabled or contracts <= 1 or self.rng.random() >= self.partial_prob:
            return contracts
        return self.rng.randint(1, contracts - 1)

    def gap_drift(self) -> Decimal:
        """An occasional overnight-style jump layered on the cycle's normal drift.

        This is the friction that creates *losing* days: a gap can carry spot
        through a short strike between two cycles, which is exactly the breach the
        manage sweep must catch and close at a loss. A sim that only drifts smoothly
        never tests that path."""
        if not self.enabled or self.rng.random() >= self.gap_prob:
            return Decimal(0)
        return self.rng.choice((Decimal(1), Decimal(-1))) * self.gap_size


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
    *, spot: Decimal, iv: float, now: datetime, half_spread: Decimal,
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
    bid = max(price - half_spread, Decimal("0.01"))
    ask = price + half_spread
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
    frictions: Frictions, window: int = 12,
) -> list[Contract]:
    """A +/- `window`-dollar grid of $1 strikes, puts and calls, across `expiries`.

    Each strike's spread comes from `frictions`, so the wings are wider than the
    body — the shape a real chain has, and the reason Gate 8 has anything to reject.
    """
    lo = int(spot) - window
    hi = int(spot) + window
    out: list[Contract] = []
    for expiry in expiries:
        for k in range(lo, hi + 1):
            for is_put in (True, False):
                c = _contract(underlying, expiry, is_put, Decimal(k),
                              spot=spot, iv=iv, now=now,
                              half_spread=frictions.half_spread(Decimal(k), spot))
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

    def __init__(self, frictions: Frictions) -> None:
        self.frictions = frictions
        self.positions: dict[str, _Pos] = {}
        self.resting: dict[str, _RestingOrder] = {}
        self.realized = Decimal(0)
        self._n = 0
        self.mark: dict[str, Decimal] = {}   # symbol -> current mid, refreshed each step
        self.quotes: dict[str, tuple[Decimal, Decimal]] = {}   # symbol -> (bid, ask)
        self._orders_by_id: dict[str, object] = {}   # so a poll echoes the submitted order

    def _quote(self, symbol: str) -> tuple[Decimal, Decimal]:
        """Current (bid, ask). Falls back to a mid ± default spread off-grid."""
        q = self.quotes.get(symbol)
        if q is not None:
            return q
        m = self.mark.get(symbol, Decimal("0.10"))
        return (max(m - HALF_SPREAD, CENT), m + HALF_SPREAD)

    def _natural_credit(self, legs) -> Decimal:
        """Signed per-contract credit the market will actually pay *right now*.

        Sell the short legs at the bid, buy the long legs at the ask — the touch,
        i.e. the worst price, which is what you get when you cross to fill. Positive
        for a credit package, negative for a debit. The router's rung starts above
        this (at the mid) and concedes toward it, so this is the number that decides
        whether a given rung fills at all."""
        total = Decimal(0)
        for lg in legs:
            bid, ask = self._quote(lg.symbol)
            ratio = Decimal(getattr(lg, "ratio_qty", 1))
            total += (bid if "sell" in lg.position_intent.value else -ask) * ratio
        return total

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
            # An opening rung. It fills only if this rung's price is one the market
            # will actually pay (§2.5): a credit rung fills when we ask for no more
            # than the natural credit; a debit rung when we offer at least the
            # natural debit. Rung 1 sits at the mid, richer than the natural, so it
            # does NOT fill — the router cancels and concedes, exactly the ladder
            # walk the mid-fill sim never exercised. If no rung down to Gate 9's
            # floor is payable, the ladder exhausts into a (free) NO_FILL.
            requested = _pkg_qty(req)
            natural = self._natural_credit(legs)
            limit = Decimal(str(req.limit_price))
            fillable = limit <= natural if natural >= 0 else limit >= -natural
            if not fillable:
                order = _order(oid, req.client_order_id, "new", 0, None, req.limit_price)
            else:
                # ~10% of paper fills arrive partial (§1.2). A partial reports
                # `partially_filled`; the router rests a target sized to the fill and
                # cancels the remainder rather than legging into a second ticket.
                qty = self.frictions.fill_contracts(requested)
                self._apply_open(legs, qty)
                status = "filled" if qty >= requested else "partially_filled"
                order = _order(oid, req.client_order_id, status, qty,
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
            is_short = "sell" in lg.position_intent.value
            sign = Decimal(-1) if is_short else Decimal(1)
            dq = sign * Decimal(qty) * Decimal(getattr(lg, "ratio_qty", 1))
            bid, ask = self._quote(lg.symbol)
            # Fill each leg on the side that costs us: sell a short at the bid, buy a
            # long at the ask. Booking the entry at the touch (not the mid) is what
            # makes the round trip pay the bid-ask spread — the friction the old
            # mid-fill never charged, and the reason paper P&L reads optimistic.
            price = bid if is_short else ask
            self._add(lg.symbol, dq, price)

    def _apply_close(self, legs) -> None:
        for lg in legs:
            p = self.positions.get(lg.symbol)
            if p is None:
                continue
            bid, ask = self._quote(lg.symbol)
            # Unwind on the side that fills: buy a short back at the ask, sell a long
            # at the bid. Paired with the entry above, a flat round trip loses two
            # half-spreads — before the market has moved a cent.
            close_price = ask if p.qty < 0 else bid
            self.realized += (close_price - p.avg) * p.qty * CONTRACT
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
                q = self.quotes.get(lg["symbol"])
                if q is None:
                    priceable = False
                    break
                bid, ask = q
                # to close: buy back a short at the ask (+), sell a long at the bid
                # (-). Costing the buy-back at the ask (not the mid) is why a wider
                # spread makes the 50% target genuinely harder to reach.
                debit += ask if "buy" in lg["intent"] else -bid
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
        # Read the exact (bid, ask) the chain was built with this cycle — the same
        # numbers the kernel gated and the fill engine will book against — so a
        # management price and an entry fill can never see different markets. Only a
        # symbol that has drifted off the grid (after a gap) is recomputed.
        out = {}
        for s in symbols:
            q = self.world.client.quotes.get(s)
            if q is None:
                occ = _parse_raw(s)
                spot = self.world.spots[occ.underlying]
                m = _mid(occ.underlying, s, spot, self.world.iv(occ.underlying),
                         self.world.now)
                hs = self.world.frictions.half_spread(occ.strike, spot)
                q = (max(m - hs, CENT), m + hs)
            out[s] = q
        return out

    async def submit_entry(self, proposal, state, context, *, risk=None, strategy=None):
        """Same real router as production, with the ladder's fill-poll dwell skipped.

        `SimClient` fills rung 1 synchronously, so the router's real
        `RUNG_WAIT_SECONDS` wait between submit and poll is pure dead wall-clock —
        20s × two entries = 40s per simulated day, which makes a soak of hundreds of
        days infeasible. Injecting a no-op `sleep` removes only the wait: the same
        `execution.router.submit_entry` runs, gates, builds the ladder, submits and
        polls exactly as it will live. The 20s dwell is exercised for real by the
        worker and by `tests/`, not here.
        """
        import asyncio  # noqa: PLC0415

        from vigil.execution.router import (  # noqa: PLC0415
            submit_entry as _router_submit_entry,
        )

        return await asyncio.to_thread(
            _router_submit_entry, proposal, state, context,
            client=self.client, risk=risk, strategy=strategy,
            sleep=lambda _s: None,
        )


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
    frictions: Frictions
    iv_scale: float = 1.0               # a vol crush (<1) collapses extrinsic value
    _chains: dict[str, list[Contract]] = field(default_factory=dict)

    def iv(self, underlying: str) -> float:
        return BASE_IV[underlying] * self.iv_scale

    def expiries(self) -> list[date]:
        return [self.day, self.day + timedelta(days=1), self.day + timedelta(days=2)]

    def rebuild(self) -> None:
        self._chains = {
            u: build_chain(u, spot=self.spots[u], expiries=self.expiries(),
                           iv=self.iv(u), now=self.now, frictions=self.frictions)
            for u in UNIVERSE
        }
        # Refresh the client's marks (mids, for mark-to-market) and its quote cache
        # (bid/ask, for fills) from the chain just built, so the fill engine and the
        # kernel are looking at one market.
        self.client.mark = {}
        self.client.quotes = {}
        for u in UNIVERSE:
            for c in self._chains[u]:
                mid = c.mid or CENT
                hs = self.frictions.half_spread(c.occ.strike, self.spots[u])
                self.client.mark[c.occ.raw] = mid
                self.client.quotes[c.occ.raw] = (max(mid - hs, CENT), mid + hs)

    def chain(self, underlying: str) -> list[Contract]:
        return self._chains[underlying]

    def advance(self, minutes: int, *, drift: Decimal, iv_scale: float | None = None) -> None:
        self.now = self.now + timedelta(minutes=minutes)
        if iv_scale is not None:
            self.iv_scale = iv_scale
        # A gap rides on top of the cycle's normal drift ~1 in 5 advances. This is
        # what carries spot through a short strike between cycles, forcing the
        # breach-exit and turning some days red — the whole point of the friction.
        total_drift = drift + self.frictions.gap_drift()
        for u in UNIVERSE:
            self.spots[u] = (self.spots[u] * (Decimal(1) + total_drift)).quantize(CENT)
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


# --------------------------------------------------------------------------- #
# Invariants — the properties that must hold on EVERY simulated day, or the
# agent has a bug that live money would pay for. Checked against the real
# reconstructed book (`broker.structures()` → `group_positions`), never a
# re-derivation: a failure here is the agent's own view of the world breaking.
# --------------------------------------------------------------------------- #

# Hard rule #2 caps risk per trade at 2% of equity. A structure's max_loss is
# one trade's risk; the tolerance absorbs intraday equity drift between the
# entry (when Gate 2 checked) and this later snapshot — it is not slack in the
# gate itself, which the Hypothesis test in tests/test_gates.py owns exactly.
RISK_PER_TRADE = Decimal("0.02")
RISK_TOLERANCE = Decimal("1.10")


@dataclass(frozen=True, slots=True)
class Violation:
    """One broken invariant, tagged with where it happened so it is reproducible."""
    cycle: str
    kind: str
    detail: str


def _naked_or_undefined(s) -> str | None:
    """Hard rule #3: no naked short leg, and every structure has finite max loss.

    Reads the reconstructed legs, so a short with no covering long on the same
    right — the exact shape "temporarily while legging in" would produce — is
    caught regardless of how the book got into that state.
    """
    if not s.has_short_legs:
        return None                      # long-only: premium paid is the whole risk
    shorts_p = longs_p = shorts_c = longs_c = 0
    for leg in s.legs:
        occ = _parse_raw(leg.symbol)
        if occ.is_put:
            shorts_p, longs_p = (shorts_p + 1, longs_p) if leg.is_short else (shorts_p, longs_p + 1)
        else:
            shorts_c, longs_c = (shorts_c + 1, longs_c) if leg.is_short else (shorts_c, longs_c + 1)
    if shorts_p and not longs_p:
        return f"naked short put ({shorts_p} short, 0 long) on {s.underlying} {s.expiry}"
    if shorts_c and not longs_c:
        return f"naked short call ({shorts_c} short, 0 long) on {s.underlying} {s.expiry}"
    if not s.max_loss.is_finite() or s.max_loss <= 0:
        return f"non-positive/undefined max_loss ${s.max_loss} on {s.underlying} {s.expiry}"
    return None


def check_invariants(
    structures, equity: Decimal, prev_defects: frozenset[object], cycle: str,
) -> tuple[list[Violation], frozenset[object]]:
    """Every invariant that must hold this cycle. Returns violations + the current
    §2.6 defect set, so the caller can tell a transient defect from a persistent one.
    """
    out: list[Violation] = []

    if not equity.is_finite():
        out.append(Violation(cycle, "equity", f"equity is not a finite number: {equity}"))

    risk_cap = equity * RISK_PER_TRADE * RISK_TOLERANCE
    for s in structures:
        naked = _naked_or_undefined(s)
        if naked is not None:
            out.append(Violation(cycle, "defined-risk", naked))
        if s.max_loss.is_finite() and s.max_loss > risk_cap:
            out.append(Violation(
                cycle, "sizing",
                f"{s.underlying} {s.expiry} risks ${s.max_loss:,.0f} > 2%×equity "
                f"(${risk_cap:,.0f})"))

    # §2.6: an open structure with no resting target is a defect. A single cycle
    # is tolerated (flatten strips targets; the next sweep re-rests them); a defect
    # that survives into the cycle where it was already a defect is the real fault.
    defects = frozenset(s.structure_key for s in structures_missing_targets(structures))
    for s in structures:
        if s.structure_key in defects and s.structure_key in prev_defects:
            out.append(Violation(
                cycle, "§2.6-persistent",
                f"{s.underlying} {s.expiry} had no resting profit target for two "
                f"consecutive cycles"))
    return out, defects


# --------------------------------------------------------------------------- #
# One simulated day — the scripted plan, returning what happened instead of only
# printing it, so `make sim` (verbose) and `make soak` (batch) share one path.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class DayOutcome:
    regime: str
    seed: int
    day: date
    entries: int
    final_equity: Decimal
    day_pnl: Decimal
    realized: Decimal
    unrealized: Decimal
    open_legs: int
    violations: list[Violation]

    @property
    def traded(self) -> bool:
        return self.entries > 0

    @property
    def clean(self) -> bool:
        return not self.violations


def _drift_for(regime: str) -> Decimal:
    return {"trend_up": Decimal("0.0006"), "trend_down": Decimal("-0.0006"),
            "chop": Decimal("0.0001")}.get(regime, Decimal("0.0006"))


async def run_day(
    regime: str, seed: int, *, day: date, verbose: bool, friction: bool = True,
) -> DayOutcome:
    """Drive one full synthetic day through the real pipeline and score it.

    `verbose` prints the cycle-by-cycle narrative (`make sim`); the batch path runs
    it silently and reads the returned `DayOutcome`. A fresh `SimClient` per call
    makes each day an independent 100k draw — the batch wants i.i.d. days, not a
    compounding equity curve. `friction=False` runs the clean baseline (no wide
    spreads, slippage, partials or gaps).
    """
    rng = Random(seed)
    frictions = Frictions(rng=rng, enabled=friction)
    start = datetime.combine(day, time(9, 45), tzinfo=ET)
    client = SimClient(frictions)
    world = World(now=start, day=day, spots=dict(BASE_SPOT), trend=Decimal("0.001"),
                  client=client, frictions=frictions)
    world.rebuild()
    _install_bar_stubs(world, regime)
    broker = SimBroker(client, world)

    drift = _drift_for(regime)

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

    entries = 0
    violations: list[Violation] = []
    prev_defects: frozenset[object] = frozenset()

    for kind, mins, d, ivs, note in plan:
        if mins:
            # jitter the drift a little so no two runs are identical
            world.advance(mins, drift=d + Decimal(str(rng.uniform(-0.0002, 0.0002))),
                          iv_scale=ivs)
        # Between cycles the market moved — let any reached target fill before the
        # agent looks, so reconcile sees it vanish (the §2.6 "closed unobserved" path).
        for coid in client.fill_reached_targets():
            if verbose:
                print(f"\n    ~ [{_hms(world.now)}] resting profit target FILLED "
                      f"at the broker: {coid}")

        # A cycle that *raises* is the most severe finding there is: the agent threw
        # mid-session, which live would mean a stuck loop or a position it cannot act
        # on. We record it as a violation and stop the day rather than letting one
        # bad day abort a 150-day batch — surfacing the crash, never swallowing it.
        try:
            result = await _run(kind, broker)
        except Exception as exc:  # noqa: BLE001 — the crash *is* the finding
            detail = f"{type(exc).__name__}: {exc}"
            violations.append(Violation(kind.value, "pipeline-crash", detail[:200]))
            if verbose:
                print(f"\n[{_hms(world.now)}] {kind.value.upper()} — CRASHED")
                print(f"    !! INVARIANT [pipeline-crash] {detail[:200]}")
            break
        entries += result.submitted

        # Score the book the agent itself just reconstructed — same reconciliation
        # the manage sweep uses, so a failure is the agent's own view breaking.
        structures = await broker.structures()
        equity = Decimal(client.get_account().equity)
        cycle_violations, prev_defects = check_invariants(
            structures, equity, prev_defects, kind.value)
        violations.extend(cycle_violations)

        if verbose:
            _print_cycle(world, kind, result, note)
            _print_book(world)
            for v in cycle_violations:
                print(f"    !! INVARIANT [{v.kind}] {v.detail}")

    acct = client.get_account()
    cents = Decimal("0.01")
    pnl = (Decimal(acct.equity) - Decimal(100_000)).quantize(cents)
    realized = client.realized.quantize(cents)
    return DayOutcome(
        regime=regime, seed=seed, day=day, entries=entries,
        final_equity=Decimal(acct.equity), day_pnl=pnl, realized=realized,
        unrealized=(pnl - realized), open_legs=len([p for p in client.positions.values() if p.qty]),
        violations=violations,
    )


# --------------------------------------------------------------------------- #
# `make sim` — one narrated day.
# --------------------------------------------------------------------------- #

async def simulate(regime: str, seed: int, friction: bool) -> int:
    import logging

    from vigil.logging import configure

    # Quiet the per-cycle JSON logs (they go to stderr) so the narrative reads
    # clean; warnings — §2.6 defects, failed cycles — still surface.
    configure(level=logging.WARNING)

    await _reset_journal()               # clean slate — no rows bleed between runs
    day = today_et()
    frictions = "ON (spreads, slippage, partials, gaps)" if friction else "OFF (mid fills)"
    print("=" * 72)
    print(f" VIGIL offline day simulator — regime={regime} seed={seed} date={day}")
    print(" synthetic chain (BS-priced), real kernel + ladder + reconcile, no network")
    print(f" frictions: {frictions}")
    print("=" * 72)

    outcome = await run_day(regime, seed, day=day, verbose=True, friction=friction)

    print("\n" + "=" * 72)
    print(f" DONE  final equity {outcome.final_equity}  day P&L {outcome.day_pnl:+}  "
          f"realized {outcome.realized:+}  unrealized {outcome.unrealized:+}  "
          f"open legs {outcome.open_legs}")
    if outcome.violations:
        print(f" INVARIANTS  ✗ {len(outcome.violations)} violation(s) — see !! lines above")
    else:
        print(" INVARIANTS  ✓ all held")
    print("=" * 72)
    return 1 if outcome.violations else 0


# --------------------------------------------------------------------------- #
# `make soak` — many days × regimes, a performance distribution + invariant audit.
# --------------------------------------------------------------------------- #

def _fmt(d: Decimal | float) -> str:
    return f"{float(d):+,.0f}"


def _pctl(values: list[float], q: float) -> float:
    """The `q`-quantile by nearest-rank — no numpy for one number (PLAN §12)."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


async def _binding_gate() -> tuple[str, int, int] | None:
    """The gate that rejected the most candidates across the whole batch.

    Read from `reporting.gate_stats` — the same journal query the session report
    and the desk use — rather than counted here, so the soak and the report agree.
    """
    from vigil.db.repositories import reporting as R
    from vigil.db.session import get_session

    async with get_session() as db:
        stats = await R.gate_stats(db)
    blockers = [(name, ok, bad) for _, name, ok, bad in stats if bad > 0]
    if not blockers:
        return None
    name, ok, bad = max(blockers, key=lambda t: t[2])
    return name, bad, ok + bad


async def soak(days: int, regimes: list[str], friction: bool) -> int:
    """Run `days` independent days per regime, aggregate, audit invariants.

    Resets the journal once (not per day) so `gate_stats` accumulates the whole
    batch; each day still gets a fresh account book via its own `SimClient`, and a
    distinct trading date so the sessions never collide.
    """
    import logging

    from vigil.logging import configure

    configure(level=logging.ERROR)       # the batch is loud; keep only real errors
    await _reset_journal()

    base = today_et()
    outcomes: list[DayOutcome] = []
    total = days * len(regimes)
    done = 0
    for regime in regimes:
        for seed in range(1, days + 1):
            # Distinct past dates → one session per simulated day, no collisions.
            day = base - timedelta(days=done + 1)
            outcomes.append(
                await run_day(regime, seed, day=day, verbose=False, friction=friction))
            done += 1
            print(f"\r  running… {done}/{total} days", end="", flush=True)
    print("\r" + " " * 40 + "\r", end="")

    pnls = [float(o.day_pnl) for o in outcomes]
    traded = [o for o in outcomes if o.traded]
    wins = [o for o in traded if o.day_pnl > 0]
    all_violations = [(o, v) for o in outcomes for v in o.violations]
    clean_days = sum(1 for o in outcomes if o.clean)

    print("=" * 72)
    print(f" VIGIL soak · {total} days ({days} × {len(regimes)} regime(s)) · "
          f"frictions {'ON' if friction else 'OFF'}")
    print("=" * 72)

    # Per-regime one-liners, so a strategy that only works in a trend is obvious.
    print(" per regime:")
    for regime in regimes:
        rs = [o for o in outcomes if o.regime == regime]
        rt = [o for o in rs if o.traded]
        rw = [o for o in rt if o.day_pnl > 0]
        mean_pnl = sum(o.day_pnl for o in rs) / len(rs) if rs else Decimal(0)
        trade_rate = len(rt) / len(rs) if rs else 0.0
        win_rate = len(rw) / len(rt) if rt else None
        wr = "—" if win_rate is None else f"{win_rate:.0%}"
        print(f"   {regime:11} mean P&L {_fmt(mean_pnl):>9}  "
              f"traded {trade_rate:.0%}  win {wr}")

    print()
    if pnls:
        print(f" P&L per day   mean {_fmt(sum(pnls) / len(pnls)):>9}   "
              f"median {_fmt(_pctl(pnls, 0.5)):>9}")
        print(f"               p10 {_fmt(_pctl(pnls, 0.10)):>9}   "
              f"worst {_fmt(min(pnls)):>9}   best {_fmt(max(pnls)):>9}")
    overall_wr = f"{len(wins) / len(traded):.0%}" if traded else "—"
    stand_down = f"{1 - len(traded) / len(outcomes):.0%}" if outcomes else "—"
    print(f" win rate      {overall_wr} of traded days   stand-down {stand_down} of days")

    gate = await _binding_gate()
    if gate is not None:
        name, bad, seen = gate
        print(f" binding gate  '{name}' rejected {bad} of {seen} candidate(s) "
              f"({bad / seen:.0%})")

    crashes = [(o, v) for o, v in all_violations if v.kind == "pipeline-crash"]

    print()
    if not all_violations:
        print(f" INVARIANTS    ✓ {clean_days}/{total} days clean "
              f"(0 naked legs, 0 oversized trades, 0 persistent §2.6 defects)")
    else:
        if crashes:
            print(f" PIPELINE      ✗ {len(crashes)} day(s) CRASHED mid-session — "
                  f"the agent threw and could not continue:")
            for o, v in crashes[:6]:
                print(f"   · {o.regime}/seed{o.seed} [{v.cycle}] {v.detail}")
        print(f" INVARIANTS    ✗ {len(all_violations)} violation(s) across "
              f"{total - clean_days} day(s):")
        for o, v in [x for x in all_violations if x[1].kind != "pipeline-crash"][:12]:
            print(f"   · {o.regime}/seed{o.seed} [{v.cycle}:{v.kind}] {v.detail}")
    print("=" * 72)
    return 1 if all_violations else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline full-day pipeline simulator.")
    ap.add_argument("--regime", default="trend_up",
                    choices=["trend_up", "trend_down", "chop"],
                    help="steer the synthetic tape (default: trend_up)")
    ap.add_argument("--seed", type=int, default=1, help="price-path seed")
    ap.add_argument("--batch", action="store_true",
                    help="soak mode: many days × regimes, distribution + invariant audit")
    ap.add_argument("--days", type=int, default=50,
                    help="soak mode: days per regime (default: 50 → 150 total)")
    ap.add_argument("--no-friction", action="store_true",
                    help="disable market frictions (mid fills) — the old clean sim")
    args = ap.parse_args()
    friction = not args.no_friction
    if args.batch:
        # Always sweep all three regimes — a single-regime soak hides exactly the
        # failure (works in a trend, breaks in chop) the batch exists to find, so
        # --regime is ignored here on purpose.
        return asyncio.run(
            soak(args.days, ["trend_up", "trend_down", "chop"], friction))
    return asyncio.run(simulate(args.regime, args.seed, friction))


if __name__ == "__main__":
    raise SystemExit(main())
