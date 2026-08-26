"""A1 — does the indicative options feed return populated greeks and IV?

PLAN.md §1.3: roughly 60% of the design rests on this. Delta-based strike
selection, Gate 7 (dollar delta), the IV percentile and the VRP input all read
`greeks.delta` / `implied_volatility` off the snapshot endpoint. If those come
back null on the free indicative feed, we compute them locally with
Black-Scholes instead — and we want to know that today, not mid-session.

Run:  uv run python scripts/a1_greeks.py [SYMBOL ...]
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date

from vigil.clock import today_et
from vigil.data.chain import Contract, fetch_chain, spot_price

DEFAULT_UNIVERSE = ["SPY", "QQQ"]
# Below this fraction of contracts carrying greeks, delta-based selection is not
# viable and the A1 fallback (local Black-Scholes) is required.
POPULATED_THRESHOLD = 0.90


def _pct(n: int, total: int) -> str:
    return "  n/a" if total == 0 else f"{100 * n / total:5.1f}%"


def report(underlying: str, today: date) -> float:
    spot = spot_price(underlying)
    contracts = fetch_chain(underlying, spot=spot, max_dte=2, asof=today)

    print(f"\n=== {underlying} — spot {spot} — {len(contracts)} contracts in window ===")
    if not contracts:
        print("  no contracts returned; is the strike/expiry window right?")
        return 0.0

    by_expiry: dict[date, list[Contract]] = defaultdict(list)
    for c in contracts:
        by_expiry[c.occ.expiry].append(c)

    print(f"  {'expiry':<12}{'DTE':>4}{'n':>6}{'delta':>8}{'iv':>8}{'quote':>8}")
    total = have_delta = 0
    for expiry in sorted(by_expiry):
        group = by_expiry[expiry]
        d = sum(1 for c in group if c.delta is not None)
        v = sum(1 for c in group if c.iv is not None)
        q = sum(1 for c in group if c.mid is not None)
        total += len(group)
        have_delta += d
        print(
            f"  {expiry.isoformat():<12}{(expiry - today).days:>4}{len(group):>6}"
            f"{_pct(d, len(group)):>8}{_pct(v, len(group)):>8}{_pct(q, len(group)):>8}"
        )

    # A concrete sample matters as much as the counts: a feed can return a greeks
    # object full of zeros, which passes a null check but is useless for selection.
    sample = next((c for c in contracts if c.delta is not None), None)
    if sample is not None:
        g = sample.snapshot.greeks
        print(
            f"\n  sample {sample.occ.raw}: delta={g.delta} gamma={g.gamma} "
            f"theta={g.theta} vega={g.vega} iv={sample.iv} mid={sample.mid}"
        )
        nonzero = sum(1 for c in contracts if c.delta not in (None, 0.0))
        print(f"  non-zero deltas: {nonzero}/{total} ({_pct(nonzero, total)})")

    return have_delta / total if total else 0.0


def main() -> int:
    universe = sys.argv[1:] or DEFAULT_UNIVERSE
    today = today_et()
    rates = [report(u, today) for u in universe]

    worst = min(rates) if rates else 0.0
    print("\n" + "=" * 60)
    if worst >= POPULATED_THRESHOLD:
        print(f"A1 HOLDS — greeks populated on >={POPULATED_THRESHOLD:.0%} of contracts.")
        print("  Delta-based strike selection, Gate 7 and the VRP input proceed as planned.")
    else:
        print(f"A1 FAILS — worst symbol only {worst:.1%} populated.")
        print("  Fallback (PLAN §1.3): compute delta and IV locally via Black-Scholes.")
    print("=" * 60)
    # Exit code carries the verdict so this can gate a Day 0 checklist.
    return 0 if worst >= POPULATED_THRESHOLD else 1


if __name__ == "__main__":
    raise SystemExit(main())
