"""First-trade preflight — a GO/HOLD readiness gate for the empirical probes.

The three live checks that gate the first real trade (A2 credit floor, A3/B-1 gate
stack, O-1 mleg submit) all need market hours. This script runs *before* the open and
verifies everything that can be verified without live quotes, so that at 09:30 the
only remaining variable is the market itself — not a missing key, a stale lock, or a
skewed clock discovered the hard way mid-probe.

It reuses the exact startup guards the worker uses (`load_settings`, `verify_account`,
`verify_clock`) rather than re-implementing them, so a GO here means the same asserts
the worker makes at boot already pass. Nothing is submitted and nothing is mutated —
this only reads.

Run:  uv run python scripts/preflight.py      (or: make preflight)
"""

from __future__ import annotations

from dataclasses import dataclass

from vigil.config import universe_config


@dataclass(frozen=True, slots=True)
class Check:
    """One readiness check. `critical` checks gate GO; advisory ones only inform."""

    name: str
    ok: bool
    detail: str
    critical: bool = True


def _settings_check() -> Check:
    """Credentials load and the environment unambiguously says paper (hard rule #1)."""
    from vigil.settings import LiveTradingRefused, load_settings

    try:
        s = load_settings()
    except LiveTradingRefused as exc:
        return Check("paper mode", False, f"REFUSED: {exc}")
    except Exception as exc:  # noqa: BLE001 — any load failure is a hold, reported plainly
        return Check("credentials", False, f"could not load .env: {type(exc).__name__}: {exc}")
    if not s.paper:
        return Check("paper mode", False, "settings.paper is False — refusing to proceed")
    return Check("paper mode", True, f"paper endpoint {s.trading_base_url}")


def _lock_present_check() -> Check:
    """The account lock file exists (hard rule #7). Its *match* is a separate,
    network check below — this one works offline."""
    from vigil.account import LOCK_PATH, read_lock

    locked = read_lock()
    if locked is None:
        return Check("account lock file", False, f"no lock at {LOCK_PATH} — run `make lock`")
    return Check("account lock file", True, f"locked to {locked}")


def _account_match_check(client: object) -> Check:
    """The live credentials resolve to the locked account (hard rule #7)."""
    from vigil.account import AccountLockError, verify_account

    try:
        acct_id = verify_account(client=client)
    except AccountLockError as exc:
        return Check("account matches lock", False, str(exc).split(".")[0])
    return Check("account matches lock", True, f"{acct_id} confirmed live")


def _clock_check(client: object) -> Check:
    """The host clock agrees with Alpaca's within tolerance (the 0DTE-flatten guard)."""
    from vigil.clock_guard import ClockDriftError, verify_clock

    try:
        drift = verify_clock(client=client)
    except ClockDriftError as exc:
        return Check("clock sync", False, str(exc).split(".")[0])
    return Check("clock sync", True, f"{drift:.1f}s drift, within tolerance")


def _market_schedule_check(client: object) -> Check:
    """Alpaca's authoritative clock: open now, or when next. Advisory — a closed
    market pre-open is expected, not a failure; this just tells you how long to wait."""
    clock = client.get_clock()  # type: ignore[attr-defined]
    if getattr(clock, "is_open", False):
        return Check(
            "market schedule", True, f"OPEN — next close {clock.next_close}", critical=False
        )
    return Check(
        "market schedule", True,
        f"closed — next open {clock.next_open} (run the probes then)", critical=False,
    )


def _vrp_seed_check() -> Check:
    """The regime router's realized-vol history is seeded (§4.3.1). Advisory: without
    it the router falls to the cold-start path and flags every verdict cold_start,
    which is correct but blunts A3/B-1 — worth knowing before, not during, the run."""
    from vigil.signals.history import rv_history

    universe = list(universe_config().primary)
    seeded = {u: len(rv_history(u)) for u in universe}
    missing = [u for u, n in seeded.items() if n == 0]
    detail = ", ".join(f"{u}:{n}" for u, n in seeded.items())
    if missing:
        return Check(
            "VRP seed", False, f"no history for {missing} ({detail}) — run `make vrp`",
            critical=False,
        )
    return Check("VRP seed", True, f"seeded ({detail})", critical=False)


def _imports_check() -> Check:
    """The three probe scripts import cleanly — catch a broken import here, not at the
    open. They live beside this file, so we import them by bare name with `scripts/`
    on the path, which is how `python scripts/<x>.py` resolves them at run time."""
    import sys
    from importlib import import_module
    from pathlib import Path

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    for m in ("a2_credit", "a4_mleg_smoke", "dry_run"):
        try:
            import_module(m)
        except Exception as exc:  # noqa: BLE001 — an import error is a hard hold
            return Check("probe scripts import", False, f"{m}: {type(exc).__name__}: {exc}")
    return Check("probe scripts import", True, "a2_credit, a4_mleg_smoke, dry_run")


def run_checks() -> list[Check]:
    """All checks, network ones sharing a single client. A client that cannot be built
    (no keys, offline) turns every network check into a reported hold, not a traceback."""
    checks = [_settings_check(), _lock_present_check(), _vrp_seed_check(), _imports_check()]

    client: object | None = None
    try:
        from vigil.data.alpaca_client import trading_client

        client = trading_client()
    except Exception as exc:  # noqa: BLE001 — offline or unkeyed: skip network checks loudly
        checks.append(
            Check("broker client", False, f"cannot reach Alpaca: {type(exc).__name__}: {exc}")
        )
        return checks

    for fn in (_account_match_check, _clock_check, _market_schedule_check):
        try:
            checks.append(fn(client))
        except Exception as exc:  # noqa: BLE001 — a network hiccup is a hold, not a crash
            checks.append(Check(fn.__name__.strip("_").replace("_check", ""), False,
                                f"check failed: {type(exc).__name__}: {exc}"))
    return checks


def main() -> int:
    checks = run_checks()

    print("═══ Vigil first-trade preflight ═══\n")
    for c in checks:
        mark = "✓" if c.ok else "✗"
        tag = "" if c.critical else "  (advisory)"
        print(f"  [{mark}] {c.name:<22} {c.detail}{tag}")

    critical_failed = [c for c in checks if c.critical and not c.ok]
    advisory_failed = [c for c in checks if not c.critical and not c.ok]

    print()
    if critical_failed:
        print(f"HOLD — {len(critical_failed)} critical check(s) failed. Fix before trading:")
        for c in critical_failed:
            print(f"    ✗ {c.name}: {c.detail}")
        return 1

    if advisory_failed:
        print("GO (with caveats) — all critical checks pass; advisories to note:")
        for c in advisory_failed:
            print(f"    · {c.name}: {c.detail}")
        return 0

    print("GO — all checks pass. When the market opens, run the probes in order:")
    print("    make a2        # A2: Gate 9 credit floor reachable?")
    print("    make dry-run   # A3/B-1: gate stack fires; names the binding gate")
    print("    make smoke     # O-1: dry-run the mleg; re-run with --submit to place it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
