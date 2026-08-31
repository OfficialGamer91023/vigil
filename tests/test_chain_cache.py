"""The optional Redis chain-snapshot cache (`vigil.data.cache`).

Two things matter here. First, the **golden round-trip**: alpaca-py's snapshot models
can't be rebuilt by pydantic's normal validation, so we reconstruct through the SDK's
own wire mappings — and if the SDK ever changes those, `test_round_trip_*` breaks
loudly instead of the cache silently corrupting quotes. Second, **graceful
degradation**: no Redis, a down Redis, or a garbage payload must each read as a plain
miss, never an exception into a trading cycle (hard rule #6). No real Redis is
touched — a dict-backed fake stands in.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from alpaca.data.models.snapshots import OptionsSnapshot
from alpaca.trading.enums import ContractType

from vigil.data import cache


def _snapshot(symbol: str = "SPY260904P00640000") -> OptionsSnapshot:
    """A snapshot shaped like the live feed: a two-sided quote plus greeks and IV."""
    raw = {
        "latestQuote": {
            "t": "2026-08-31T14:00:00Z",
            "bp": 1.20, "ap": 1.30, "bs": 10, "as": 12,
            "bx": "A", "ax": "B", "c": [], "z": "",
        },
        "greeks": {"delta": -0.16, "gamma": 0.02, "theta": -0.05, "vega": 0.10, "rho": 0.01},
        "impliedVolatility": 0.19,
    }
    return OptionsSnapshot(symbol, raw_data=raw)


class _FakeRedis:
    """A dict-backed stand-in for redis-py: only `get`/`set` with a TTL arg."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.last_ttl: int | None = None

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value
        self.last_ttl = ex


class _BrokenRedis:
    """A Redis that raises on every command — a down or hung server."""

    def get(self, key: str) -> str:
        raise RuntimeError("connection refused")

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        raise RuntimeError("connection refused")


# --------------------------------------------------------------------------- #
# Golden round-trip — the SDK-drift tripwire
# --------------------------------------------------------------------------- #

def test_round_trip_preserves_quote_greeks_and_iv() -> None:
    snap = _snapshot()
    back = cache._deserialize(cache._serialize(snap))
    assert back.symbol == snap.symbol
    assert back.latest_quote.bid_price == 1.20
    assert back.latest_quote.ask_price == 1.30
    assert back.greeks.delta == -0.16
    assert back.greeks.vega == 0.10
    assert back.implied_volatility == 0.19


def test_round_trip_survives_json() -> None:
    """The real path goes through `json.dumps`/`loads` — datetimes and all."""
    snap = _snapshot()
    blob = json.dumps(cache._serialize(snap))
    back = cache._deserialize(json.loads(blob))
    assert back.latest_quote.bid_price == 1.20
    assert back.greeks.delta == -0.16


def test_round_trip_handles_a_quote_only_snapshot() -> None:
    """The free indicative feed carries quotes but no greeks/IV — the common case."""
    raw = {"latestQuote": {"t": "2026-08-31T14:00:00Z", "bp": 0.9, "ap": 1.0,
                           "bs": 5, "as": 5, "bx": "A", "ax": "B", "c": [], "z": ""}}
    snap = OptionsSnapshot("SPY260904P00640000", raw_data=raw)
    back = cache._deserialize(cache._serialize(snap))
    assert back.latest_quote.ask_price == 1.0
    assert back.greeks is None
    assert back.implied_volatility is None


# --------------------------------------------------------------------------- #
# get/set through a fake client
# --------------------------------------------------------------------------- #

def test_set_then_get_returns_equivalent_snapshots(monkeypatch) -> None:
    fake = _FakeRedis()
    monkeypatch.setattr(cache, "_get_client", lambda: fake)

    snaps = {"SPY260904P00640000": _snapshot()}
    cache.set_snapshots("k", snaps, ttl=45)

    assert fake.last_ttl == 45  # the TTL is actually applied
    out = cache.get_snapshots("k")
    assert out is not None
    assert out["SPY260904P00640000"].latest_quote.bid_price == 1.20


def test_get_is_a_miss_on_a_cold_key(monkeypatch) -> None:
    monkeypatch.setattr(cache, "_get_client", lambda: _FakeRedis())
    assert cache.get_snapshots("never-written") is None


# --------------------------------------------------------------------------- #
# Graceful degradation — the rule-#6 guarantee
# --------------------------------------------------------------------------- #

def test_no_client_is_a_miss_and_a_noop(monkeypatch) -> None:
    monkeypatch.setattr(cache, "_get_client", lambda: None)
    assert cache.get_snapshots("k") is None
    cache.set_snapshots("k", {"x": _snapshot()})  # must not raise


def test_a_down_redis_reads_as_a_miss_never_raises(monkeypatch) -> None:
    monkeypatch.setattr(cache, "_get_client", lambda: _BrokenRedis())
    assert cache.get_snapshots("k") is None
    cache.set_snapshots("k", {"x": _snapshot()})  # swallowed, no raise


def test_a_garbage_payload_reads_as_a_miss(monkeypatch) -> None:
    """A stale/incompatible blob (e.g. after an SDK upgrade) degrades to a refetch."""
    fake = _FakeRedis()
    fake.store["k"] = "not json at all {"
    monkeypatch.setattr(cache, "_get_client", lambda: fake)
    assert cache.get_snapshots("k") is None


def test_no_redis_url_disables_the_cache(monkeypatch) -> None:
    """With REDIS_URL unset, the client is never built — caching is simply off."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    cache._reset_for_tests()
    assert cache._get_client() is None


# --------------------------------------------------------------------------- #
# Key construction
# --------------------------------------------------------------------------- #

def test_same_window_same_key_moved_spot_new_key() -> None:
    base = dict(strike_lo=628.0, strike_hi=652.0,
                expiry_lo=date(2026, 8, 31), expiry_hi=date(2026, 9, 2),
                contract_type=ContractType.PUT)
    k1 = cache.chain_cache_key("SPY", **base)
    k2 = cache.chain_cache_key("SPY", **base)
    moved = cache.chain_cache_key("SPY", **{**base, "strike_lo": 629.0, "strike_hi": 653.0})
    assert k1 == k2
    assert k1 != moved
    assert k1.startswith("vigil:chain:v1:SPY:")


def test_both_rights_and_single_right_differ() -> None:
    base: dict[str, Any] = dict(strike_lo=628.0, strike_hi=652.0,
                                expiry_lo=date(2026, 8, 31), expiry_hi=date(2026, 9, 2))
    puts = cache.chain_cache_key("SPY", contract_type=ContractType.PUT, **base)
    both = cache.chain_cache_key("SPY", contract_type=None, **base)
    assert puts != both
    assert both.endswith(":both")
