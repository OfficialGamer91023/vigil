"""Fetching option chain snapshots within a bounded strike/expiry window.

§1.2: never pull a full chain. A SPY 0-2 DTE chain is 600-1000 contracts across
three expiries, and the free tier allows 200 requests/min. Every fetch here is
bounded by an ATM strike window and an explicit expiry range.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from decimal import Decimal

from alpaca.data.enums import DataFeed
from alpaca.data.models.snapshots import OptionsSnapshot
from alpaca.data.requests import (
    OptionChainRequest,
    OptionLatestQuoteRequest,
    StockLatestTradeRequest,
)
from alpaca.trading.enums import ContractType
from alpaca.trading.requests import GetOptionContractsRequest

from vigil.clock import is_expiry_live, today_et
from vigil.config import greeks_config
from vigil.data.alpaca_client import (
    OPTIONS_FEED,
    option_data_client,
    stock_data_client,
    trading_client,
)
from vigil.data.greeks import ModelGreeks, solve
from vigil.data.occ import OccSymbol, parse_occ

# Free tier serves IEX for stocks. Named once, for the same reason as OPTIONS_FEED.
STOCK_FEED = DataFeed.IEX


@dataclass(frozen=True, slots=True)
class Contract:
    """A chain snapshot with its symbol already parsed, and quote fields lifted out.

    The kernel and strategy code should never re-parse symbols or reach into
    nested SDK models; they take plain objects (CLAUDE.md conventions).
    """

    occ: OccSymbol
    snapshot: OptionsSnapshot
    # Locally modelled greeks, attached by `fetch_chain` only when the feed did
    # not supply them. See `data/greeks.py` and PLAN §1.3 (assumption A1).
    computed: ModelGreeks | None = None

    @property
    def bid(self) -> Decimal | None:
        q = self.snapshot.latest_quote
        return None if q is None else Decimal(str(q.bid_price))

    @property
    def ask(self) -> Decimal | None:
        q = self.snapshot.latest_quote
        return None if q is None else Decimal(str(q.ask_price))

    @property
    def mid(self) -> Decimal | None:
        """Midpoint of the NBBO. On the indicative feed this is a *derived* quote."""
        b, a = self.bid, self.ask
        if b is None or a is None or b <= 0 or a <= 0:
            return None
        return (b + a) / 2

    @property
    def spread_pct(self) -> Decimal | None:
        """Bid-ask spread as a fraction of mid — the liquidity metric Gate 8 uses."""
        b, a, m = self.bid, self.ask, self.mid
        if b is None or a is None or m is None or m == 0:
            return None
        return (a - b) / m

    @property
    def delta(self) -> float | None:
        """Feed delta if the feed has one, else the locally modelled delta.

        A1 measured 28 Aug 2026: the free indicative feed has none, so in practice
        this is always the modelled value today. The feed branch stays first so
        that an OPRA entitlement would restore real greeks with no code change and
        no flag to flip.
        """
        g = self.snapshot.greeks
        if g is not None:
            return g.delta
        return None if self.computed is None else self.computed.delta

    @property
    def iv(self) -> float | None:
        """Same precedence as `delta`. Feeds the IV percentile and VRP (§4.3.1)."""
        quoted = self.snapshot.implied_volatility
        if quoted is not None:
            return quoted
        return None if self.computed is None else self.computed.iv

    @property
    def greeks_are_modelled(self) -> bool:
        """Did `delta`/`iv` come from our model rather than the feed?

        Worth carrying explicitly: when a Gate 7 verdict is reviewed after the
        fact, "was that a quoted delta or one we inferred?" is the first question,
        and a journal that cannot answer it is guessing.
        """
        return self.snapshot.greeks is None and self.computed is not None


def spot_price(underlying: str) -> Decimal:
    """Last traded price of the underlying. Signals come from here, not the chain."""
    req = StockLatestTradeRequest(symbol_or_symbols=underlying, feed=STOCK_FEED)
    trade = stock_data_client().get_stock_latest_trade(req)[underlying]
    return Decimal(str(trade.price))


def _with_greeks(
    contract: Contract, *, spot: Decimal, now: datetime | None, rate: float
) -> Contract:
    """Attach modelled greeks to a contract whose feed snapshot has none (§1.3 A1).

    Returns the contract unchanged when the feed already supplied greeks, so this
    is a pure fallback rather than an override.

    A contract with no two-sided quote is also returned unchanged — deliberately.
    The obvious patch is to invert the last *trade* instead, but on the indicative
    feed trades are delayed 15 minutes, and a delta computed from a 15-minute-old
    price during a fast move is worse than no delta at all: `pick_by_delta` skips a
    missing one, while a stale one gets selected on and sized against.
    """
    if contract.snapshot.greeks is not None and contract.snapshot.implied_volatility is not None:
        return contract

    mid = contract.mid
    if mid is None:
        return contract

    model = solve(
        price=float(mid),
        spot=float(spot),
        strike=float(contract.occ.strike),
        expiry=contract.occ.expiry,
        rate=rate,
        is_put=contract.occ.is_put,
        now=now,
    )
    return contract if model is None else replace(contract, computed=model)


def fetch_chain(
    underlying: str,
    *,
    spot: Decimal,
    max_dte: int = 2,
    strike_window: Decimal = Decimal(12),
    asof: date | None = None,
    now: datetime | None = None,
    contract_type: ContractType | None = None,
) -> list[Contract]:
    """Snapshots for `underlying` within `max_dte` days and +/- `strike_window` of spot.

    `strike_window` is in dollars, not strikes: SPY has $1 strikes near the money,
    so +/-$12 is roughly the +/-10-strike window §1.2 prescribes, and it stays
    correct on underlyings with different strike spacing.
    """
    today = asof or today_et()

    req = OptionChainRequest(
        underlying_symbol=underlying,
        feed=OPTIONS_FEED,
        strike_price_gte=float(spot - strike_window),
        strike_price_lte=float(spot + strike_window),
        expiration_date_gte=today,
        expiration_date_lte=today + timedelta(days=max_dte),
        # None means "both rights"; the SDK omits the filter rather than sending null.
        type=contract_type,
    )

    snapshots = option_data_client().get_option_chain(req)
    rate = greeks_config().risk_free_rate

    out: list[Contract] = []
    for symbol, snap in snapshots.items():
        try:
            occ = parse_occ(symbol)
        except ValueError:
            # A symbol we cannot parse is a data problem, not a reason to abort a
            # whole cycle — skip it and let the caller's counts reveal the gap.
            continue
        # The API's expiration_date_gte filter is date-based, so after the close it
        # still returns today's now-settled expiry. Drop it: it is not tradeable and
        # its greeks come back null, which would look like a data-quality failure.
        if not is_expiry_live(occ.expiry):
            continue
        out.append(_with_greeks(Contract(occ=occ, snapshot=snap), spot=spot, now=now, rate=rate))

    # Sorted so downstream printing and strike-walking are deterministic.
    out.sort(key=lambda c: (c.occ.expiry, c.occ.is_put, c.occ.strike))
    return out


def fetch_open_interest(
    underlying: str,
    *,
    spot: Decimal,
    max_dte: int = 2,
    strike_window: Decimal = Decimal(25),
    asof: date | None = None,
) -> dict[str, int]:
    """Open interest per OCC symbol, for Gate 8.

    A separate call because open interest lives on the *contracts* endpoint, not
    the snapshot — the snapshot carries quotes and greeks only. Gate 8 treats a
    missing OI as 0 and rejects, so this must run before candidates are built or
    every proposal fails liquidity for the wrong reason.
    """
    today = asof or today_et()
    client = trading_client()
    out: dict[str, int] = {}
    page_token: str | None = None

    # **Paginate, always.** A single page silently truncates, and truncation is
    # indistinguishable from a real answer downstream: a symbol missing from this
    # map defaults to 0 open interest, Gate 8 rejects the leg, and the rejection
    # looks like a legitimate liquidity verdict. That is the same silent-and-total
    # failure mode §4.4.2 warns about for Gate 9 — a gate that rejects everything
    # for a reason that is not true.
    while True:
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            expiration_date_gte=today,
            expiration_date_lte=today + timedelta(days=max_dte),
            strike_price_gte=str(spot - strike_window),
            strike_price_lte=str(spot + strike_window),
            limit=1000,
            page_token=page_token,
        )
        page = client.get_option_contracts(req)
        for c in getattr(page, "option_contracts", None) or []:
            if getattr(c, "open_interest", None) is not None:
                out[c.symbol] = int(c.open_interest)

        page_token = getattr(page, "next_page_token", None)
        if not page_token:
            return out


def fetch_quotes(symbols: Sequence[str]) -> dict[str, tuple[Decimal, Decimal]]:
    """Latest bid/ask for specific OCC symbols. `{symbol: (bid, ask)}`.

    **Why this exists alongside `fetch_chain`.** Chain fetches are bounded by an
    ATM strike window, which is right for *finding* a trade. It is wrong for
    *exiting* one: a structure worth closing is usually one the underlying has
    moved toward, and a fast move can carry a short strike straight out of the
    window. Pricing a close from a window fetch would then silently fail to find
    the very legs that most need closing — at 15:39, on a position that must not
    be held into auto-exercise.

    So closes quote their legs by symbol. It is also cheaper: one request for the
    two-to-four contracts we actually hold, rather than a hundred we do not.

    Symbols with no two-sided quote are **omitted, not zero-filled**. A caller
    that cannot price a leg must know that, because a zero bid would price the
    package as worthless and submit a limit to match.
    """
    if not symbols:
        return {}
    req = OptionLatestQuoteRequest(symbol_or_symbols=list(symbols), feed=OPTIONS_FEED)
    quotes = option_data_client().get_option_latest_quote(req)
    out: dict[str, tuple[Decimal, Decimal]] = {}
    for symbol, q in quotes.items():
        bid, ask = getattr(q, "bid_price", None), getattr(q, "ask_price", None)
        if bid is None or ask is None or ask <= 0:
            continue
        out[symbol] = (Decimal(str(bid)), Decimal(str(ask)))
    return out
