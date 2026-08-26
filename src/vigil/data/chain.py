"""Fetching option chain snapshots within a bounded strike/expiry window.

§1.2: never pull a full chain. A SPY 0-2 DTE chain is 600-1000 contracts across
three expiries, and the free tier allows 200 requests/min. Every fetch here is
bounded by an ATM strike window and an explicit expiry range.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from alpaca.data.enums import DataFeed
from alpaca.data.models.snapshots import OptionsSnapshot
from alpaca.data.requests import OptionChainRequest, StockLatestTradeRequest
from alpaca.trading.enums import ContractType
from alpaca.trading.requests import GetOptionContractsRequest

from vigil.clock import is_expiry_live, today_et
from vigil.data.alpaca_client import (
    OPTIONS_FEED,
    option_data_client,
    stock_data_client,
    trading_client,
)
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
        g = self.snapshot.greeks
        return None if g is None else g.delta

    @property
    def iv(self) -> float | None:
        return self.snapshot.implied_volatility


def spot_price(underlying: str) -> Decimal:
    """Last traded price of the underlying. Signals come from here, not the chain."""
    req = StockLatestTradeRequest(symbol_or_symbols=underlying, feed=STOCK_FEED)
    trade = stock_data_client().get_stock_latest_trade(req)[underlying]
    return Decimal(str(trade.price))


def fetch_chain(
    underlying: str,
    *,
    spot: Decimal,
    max_dte: int = 2,
    strike_window: Decimal = Decimal(12),
    asof: date | None = None,
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
        out.append(Contract(occ=occ, snapshot=snap))

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
