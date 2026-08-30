"""The clock-drift guard (a sibling of the account lock). **A skewed local clock
refuses to trade, it does not warn.**

Nearly every time-based decision the agent makes reads the *local* wall clock
through `vigil.clock.now_et()`: Gate 11's entry windows, the 0DTE label that drives
the 15:40 auto-exercise flatten, `is_expiry_live`, the escalation ladder's
`sessions_left`. All of that is only as trustworthy as the machine's clock. A box
whose time is wrong by minutes will happily mislabel a 1DTE contract as 0DTE, or
skip the flatten because it believes the session ends later than it does — and
auto-exercise turns that into a real, unbounded assignment overnight.

So at startup we compare `now_et()` against Alpaca's own clock endpoint (the same
clock the fills are stamped against) and refuse to trade if they disagree by more
than a tight tolerance. This is the local-time analogue of the account lock: the
account lock asserts *which* account we are trading; this asserts that *when* we
think it is matches the broker we are trading against.

Mirrors `vigil.account` on purpose — a pure comparison over inert data
(`assert_clock_synced`) that is testable with no network, a thin reader that hits
the client (`broker_now`), and a `verify_clock` wrapper for the startup path.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from vigil.clock import now_et

#: The most drift we tolerate between the local clock and Alpaca's. 60 seconds is
#: comfortably inside the smallest interval any time gate cares about (the entry
#: windows are 15–20 minutes, the flatten a hard minute boundary), while being far
#: wider than the sub-second skew a healthy NTP-synced host shows — so it fires on a
#: genuinely misconfigured clock, not on network jitter or a slow API round-trip.
MAX_CLOCK_DRIFT: timedelta = timedelta(seconds=60)


class ClockDriftError(RuntimeError):
    """The local clock and the broker's clock disagree beyond tolerance."""


def assert_clock_synced(
    local: datetime, broker: datetime, *, tolerance: timedelta = MAX_CLOCK_DRIFT
) -> float:
    """Compare two clocks. Returns the drift in seconds on success; raises on skew.

    Pure and network-free — the comparison is the part worth testing, and it should
    be testable without an SDK object graph, exactly as `account.assert_locked` is.

    Both arguments must be timezone-aware: subtracting aware datetimes compares the
    two *instants* regardless of the zones they are expressed in (local is US/
    Eastern, the broker's is UTC), whereas a naive value would silently assume the
    machine's zone — the very failure this module exists to catch. A naive input is
    therefore an error, not something to coerce.
    """
    for label, value in (("local", local), ("broker", broker)):
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ClockDriftError(
                f"the {label} timestamp {value!r} is timezone-naive; refusing to "
                f"compare clocks on an assumed zone."
            )
    drift = abs((local - broker).total_seconds())
    if drift > tolerance.total_seconds():
        raise ClockDriftError(
            f"CLOCK DRIFT. The local clock reads {local.isoformat()} but Alpaca's "
            f"clock reads {broker.isoformat()} — a {drift:.1f}s gap, over the "
            f"{tolerance.total_seconds():.0f}s tolerance. Refusing to trade: every "
            f"time gate (entry windows, the 0DTE flatten, expiry labelling) keys off "
            f"the local clock, so a skewed host mislabels expiries. Fix the host's "
            f"time sync (NTP) before starting the worker."
        )
    return drift


def broker_now(client: object | None = None) -> datetime:
    """Alpaca's current time, as a timezone-aware datetime.

    The import is local so `vigil.clock_guard` stays importable — and
    `assert_clock_synced` stays testable — on a machine with no credentials at all,
    the same discipline `vigil.account.live_identity` follows.
    """
    if client is None:
        from vigil.data.alpaca_client import trading_client

        client = trading_client()
    clock = client.get_clock()  # type: ignore[attr-defined]
    ts = clock.timestamp
    # alpaca-py's Clock.timestamp is a tz-aware UTC datetime. Assert the type rather
    # than trust it: the client is untyped here (kept as `object` so this module
    # imports without the SDK), and a guard that silently accepted a non-datetime
    # would defeat its own purpose. assert_clock_synced then rejects a naive one.
    if not isinstance(ts, datetime):
        raise ClockDriftError(
            f"Alpaca's clock endpoint returned {ts!r}, not a datetime; cannot "
            f"verify the local clock against it."
        )
    return ts


def verify_clock(
    *,
    client: object | None = None,
    local: datetime | None = None,
    tolerance: timedelta = MAX_CLOCK_DRIFT,
) -> float:
    """The startup call: assert the local clock matches the broker's. Returns drift.

    `local` defaults to `now_et()` and is injectable so a test can drive a known
    skew without touching the real clock.
    """
    return assert_clock_synced(
        local or now_et(), broker_now(client), tolerance=tolerance
    )
