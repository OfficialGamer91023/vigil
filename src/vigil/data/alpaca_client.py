"""Construction of Alpaca clients — the only place clients are built.

Every factory here routes through `load_settings()`, so the paper assertion in
`vigil.settings` runs before a client can exist. Nothing else in the codebase
should instantiate an Alpaca client directly.
"""

from __future__ import annotations

from functools import lru_cache

from alpaca.data.enums import OptionsFeed
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

from vigil.settings import Settings, load_settings

# The free tier serves *indicative* options data (derived quotes, trades delayed
# 15 min). Naming it once here means no call site can silently request OPRA and
# get a subscription error at 09:31 on a session day.
OPTIONS_FEED = OptionsFeed.INDICATIVE


@lru_cache(maxsize=1)
def settings() -> Settings:
    """Process-wide settings, loaded once.

    `lru_cache` on a zero-argument function is the idiomatic Python singleton:
    the first call computes, every later call returns the same object.
    """
    return load_settings()


@lru_cache(maxsize=1)
def trading_client() -> TradingClient:
    """Orders, positions, account. `paper=True` is hard-coded, not passed through."""
    s = settings()
    return TradingClient(api_key=s.api_key, secret_key=s.api_secret, paper=True)


@lru_cache(maxsize=1)
def option_data_client() -> OptionHistoricalDataClient:
    """Option chains, snapshots, quotes. Data clients are account-agnostic."""
    s = settings()
    return OptionHistoricalDataClient(api_key=s.api_key, secret_key=s.api_secret)


@lru_cache(maxsize=1)
def stock_data_client() -> StockHistoricalDataClient:
    """Underlying bars and quotes — the source of every *signal* (§1.2)."""
    s = settings()
    return StockHistoricalDataClient(api_key=s.api_key, secret_key=s.api_secret)
