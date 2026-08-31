"""An optional Redis cache for option-chain snapshots (PLAN §1.2, §9).

**Pure optimization. Redis is the cache, never the record.** Everything held here
is re-derivable with one live fetch, so every path falls through to the caller's own
fetch on a miss, an unset `REDIS_URL`, an unreachable server, or a serialization
mismatch. Hard rule #6 — the worker trades with Redis stopped — is therefore a
*property of this module*: nothing it does can raise into a trading cycle.

Why it exists: the free tier allows 200 requests/min, and a 15-minute entry sweep
re-fetches the same ATM chain for the same underlying every cycle. A short-TTL cache
collapses those repeated fetches into one API call without changing what any caller
sees.

Sync, not async — the one caller (`fetch_chain`) already runs in a worker thread via
`asyncio.to_thread`, so a synchronous client fits with no event loop, and a tight
socket timeout means a *stopped* Redis fails instantly (connection refused) while a
merely *hung* one waits at most a fraction of a second before we treat it as a miss.

The fiddly part is serialization. alpaca-py's snapshot models take `(symbol,
raw_data)` in the API's wire shape and cannot be rebuilt by pydantic's normal
validation. So we dump each snapshot to a JSON-safe field dict, and on read invert
the SDK's *own* wire mappings to rebuild `raw_data` and call the real constructor —
the reconstructed object is exactly what a live fetch would have produced. A golden
round-trip test guards against the SDK ever changing those mappings underneath us.
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any

from alpaca.data.mappings import QUOTE_MAPPING, TRADE_MAPPING
from alpaca.data.models.snapshots import OptionsSnapshot
from alpaca.trading.enums import ContractType

from vigil.logging import get_logger

log = get_logger(__name__)

# A version tag in the key namespace so a serialization change invalidates every old
# entry by simply not matching, rather than deserializing stale bytes into the wrong
# shape. Bump it whenever the payload format below changes.
_KEY_PREFIX = "vigil:chain:v1"

#: How long a cached chain stays fresh. Sized to the entry sweep's cadence: long
#: enough that repeated cycles within a minute reuse one fetch, short enough that the
#: next real cycle re-reads a moved market. This is a data-layer knob, not a strategy
#: threshold, so it lives here as a named constant rather than in config/*.yaml.
CHAIN_CACHE_TTL_SECONDS = 45

# Keep a hung Redis from ever stalling a cycle. A stopped server refuses instantly;
# only a wedged one reaches this bound, and a quarter-second is far below any time
# gate's resolution.
_SOCKET_TIMEOUT = 0.25

# The SDK maps API wire keys -> model field names. We invert them to go back: a
# field-name dump (what we cache) -> the wire-shaped raw_data the constructor wants.
_QUOTE_TO_WIRE = {field: wire for wire, field in QUOTE_MAPPING.items()}
_TRADE_TO_WIRE = {field: wire for wire, field in TRADE_MAPPING.items()}

# Built once. redis-py connects lazily and its pool auto-reconnects, so constructing
# the client never touches the network; every command is what can fail, and every
# command is guarded. `None` means "caching disabled" (no URL / redis absent).
_client: Any | None = None
_client_built = False


def _get_client() -> Any | None:
    global _client, _client_built
    if _client_built:
        return _client
    _client_built = True
    url = os.getenv("REDIS_URL")
    if not url:
        return None  # no URL configured → caching is simply off, not an error
    try:
        import redis

        _client = redis.Redis.from_url(
            url,
            socket_connect_timeout=_SOCKET_TIMEOUT,
            socket_timeout=_SOCKET_TIMEOUT,
            decode_responses=True,
        )
    except Exception as exc:  # noqa: BLE001 — a cache that cannot init is just absent
        log.debug("chain_cache.init_failed", error=str(exc)[:200])
        _client = None
    return _client


def chain_cache_key(
    underlying: str,
    *,
    strike_lo: float,
    strike_hi: float,
    expiry_lo: date,
    expiry_hi: date,
    contract_type: ContractType | None,
) -> str:
    """A stable key for one bounded chain request. Two fetches with the same window
    on the same day share a key; a moved spot (new strike bounds) is a new key, which
    is exactly right — it should miss and re-read the market."""
    right = contract_type.value if contract_type is not None else "both"
    # Strikes to cents so float formatting can't split one logical window into two keys.
    return (
        f"{_KEY_PREFIX}:{underlying}:{expiry_lo.isoformat()}:{expiry_hi.isoformat()}"
        f":{strike_lo:.2f}:{strike_hi:.2f}:{right}"
    )


def _to_wire(field_dump: dict[str, Any], to_wire: dict[str, str]) -> dict[str, Any]:
    """Re-key a field-name dump back to the API's wire codes. Keys with no wire
    mapping (e.g. the nested `symbol`, which the parent supplies) are dropped."""
    return {to_wire[k]: v for k, v in field_dump.items() if k in to_wire}


def _serialize(snapshot: OptionsSnapshot) -> dict[str, Any]:
    """One snapshot → a JSON-safe field dict. `mode="json"` turns datetimes into ISO
    strings so the whole thing survives `json.dumps`."""
    q = snapshot.latest_quote
    t = snapshot.latest_trade
    g = snapshot.greeks
    return {
        "symbol": snapshot.symbol,
        "latest_quote": None if q is None else q.model_dump(mode="json"),
        "latest_trade": None if t is None else t.model_dump(mode="json"),
        "implied_volatility": snapshot.implied_volatility,
        "greeks": None if g is None else g.model_dump(mode="json"),
    }


def _deserialize(d: dict[str, Any]) -> OptionsSnapshot:
    """A field dict → a real `OptionsSnapshot`, rebuilt through the SDK constructor so
    the object is indistinguishable from a live fetch (validators run, types correct)."""
    raw: dict[str, Any] = {}
    if d.get("latest_quote"):
        raw["latestQuote"] = _to_wire(d["latest_quote"], _QUOTE_TO_WIRE)
    if d.get("latest_trade"):
        raw["latestTrade"] = _to_wire(d["latest_trade"], _TRADE_TO_WIRE)
    if d.get("implied_volatility") is not None:
        raw["impliedVolatility"] = d["implied_volatility"]
    if d.get("greeks"):
        # Greeks raw keys are the field names themselves (no mapping), so the dump is
        # already wire-shaped.
        raw["greeks"] = d["greeks"]
    return OptionsSnapshot(d["symbol"], raw_data=raw)


def get_snapshots(key: str) -> dict[str, OptionsSnapshot] | None:
    """Cached snapshots for `key`, or `None` on any miss — no cache, no Redis, a cold
    key, or a payload we can no longer parse. `None` always means "go fetch live"."""
    client = _get_client()
    if client is None:
        return None
    try:
        blob = client.get(key)
    except Exception as exc:  # noqa: BLE001 — a down/hung Redis is a miss, never a raise
        log.debug("chain_cache.get_failed", error=str(exc)[:200])
        return None
    if blob is None:
        return None
    try:
        raw = json.loads(blob)
        return {sym: _deserialize(payload) for sym, payload in raw.items()}
    except Exception as exc:  # noqa: BLE001 — a stale/incompatible payload → refetch
        log.debug("chain_cache.decode_failed", error=str(exc)[:200])
        return None


def set_snapshots(
    key: str,
    snapshots: dict[str, OptionsSnapshot],
    *,
    ttl: int = CHAIN_CACHE_TTL_SECONDS,
) -> None:
    """Store snapshots under `key` with a TTL. A write that fails costs nothing — the
    caller already holds the data — so every failure is swallowed."""
    client = _get_client()
    if client is None:
        return
    try:
        payload = {sym: _serialize(snap) for sym, snap in snapshots.items()}
        client.set(key, json.dumps(payload), ex=ttl)
    except Exception as exc:  # noqa: BLE001 — a cache write is best-effort by definition
        log.debug("chain_cache.set_failed", error=str(exc)[:200])


def _reset_for_tests() -> None:
    """Drop the memoized client so a test can flip `REDIS_URL` between cases."""
    global _client, _client_built
    _client, _client_built = None, False
