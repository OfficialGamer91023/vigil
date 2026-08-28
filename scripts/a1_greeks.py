"""A1 — is delta available, and from where?

PLAN.md §1.3: roughly 60% of the design rests on delta. Strike selection, Gate 7
(dollar delta), the IV percentile and the VRP input all need it.

**A1 was measured on 28 Aug 2026 and FAILED.** The free indicative feed omits
`greeks` and `implied_volatility` from the payload entirely — they are OPRA-only,
and OPRA requires a paid Algo Trader Plus subscription. Verified at the raw HTTP
level on every available feed value, pre-market and mid-session.

So this script now reports **two** numbers, and the distinction is the whole point:

  feed   — contracts carrying greeks from Alpaca. This is the real A1 measurement.
           It should read 0.0% until somebody pays for OPRA.
  model  — contracts where `data/greeks.py` recovered a delta from the quote.
           This is what the agent actually trades on.

Counting them together would let the fallback silently paper over a feed
regression, which is exactly the kind of "the number looks fine" failure the
measurement exists to catch.

Run:  uv run python scripts/a1_greeks.py [SYMBOL ...]
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date
from decimal import Decimal

from vigil.clock import today_et
from vigil.data.chain import Contract, fetch_chain, spot_price

DEFAULT_UNIVERSE = ["SPY", "QQQ"]

# Coverage is measured over **out-of-the-money contracts only**, and the reason is
# a measurement, not a convenience. Run against the full window, every unsolved
# contract on 28 Aug 2026 was deep ITM and quoted *below its own intrinsic value*
# (SPY 758C: mid 11.195, intrinsic 11.29). That is a real no-arbitrage violation
# in the derived indicative quote, there is no implied volatility that produces
# it, and `implied_vol` is right to refuse. Counting those refusals against the
# fallback would penalise it for being correct — on contracts this agent never
# trades, since every structure in §4.5 is built from the OTM wing.
USABLE_THRESHOLD = 0.95

# Coverage alone is not enough: a chain could be 100% solved and still be useless
# if nothing sits near the delta we sell. §4.5 sells 0.15-0.20; the band is widened
# slightly so a near miss still shows up as a number rather than an empty set.
BAND_LOW, BAND_HIGH = Decimal("0.10"), Decimal("0.30")
MIN_BAND_CONTRACTS = 3


def _pct(n: int, total: int) -> str:
    return "  n/a" if total == 0 else f"{100 * n / total:5.1f}%"


def _in_band(c: Contract) -> bool:
    return c.delta is not None and BAND_LOW <= abs(Decimal(str(c.delta))) <= BAND_HIGH


def _is_otm(c: Contract, spot: Decimal) -> bool:
    return c.occ.strike < spot if c.occ.is_put else c.occ.strike > spot


def report(underlying: str, today: date) -> tuple[float, float, int]:
    """Returns (feed coverage, OTM coverage, contracts in the selection band)."""
    spot = spot_price(underlying)
    contracts = fetch_chain(underlying, spot=spot, max_dte=2, asof=today)

    print(f"\n=== {underlying} — spot {spot} — {len(contracts)} contracts in window ===")
    if not contracts:
        print("  no contracts returned; is the strike/expiry window right?")
        return 0.0, 0.0, 0

    by_expiry: dict[date, list[Contract]] = defaultdict(list)
    for c in contracts:
        by_expiry[c.occ.expiry].append(c)

    header = f"  {'expiry':<12}{'DTE':>4}{'n':>5}{'otm':>5}{'feed':>7}{'otm cov':>9}"
    print(header + f"{'band':>6}{'quote':>8}")
    total = feed_total = otm_total = otm_solved = band_total = 0
    for expiry in sorted(by_expiry):
        group = by_expiry[expiry]
        otm = [c for c in group if _is_otm(c, spot)]
        fed = sum(1 for c in group if c.snapshot.greeks is not None)
        solved = sum(1 for c in otm if c.delta is not None)
        band = sum(1 for c in group if _in_band(c))
        quoted = sum(1 for c in group if c.mid is not None)

        total += len(group)
        feed_total += fed
        otm_total += len(otm)
        otm_solved += solved
        band_total += band

        print(
            f"  {expiry.isoformat():<12}{(expiry - today).days:>4}{len(group):>5}{len(otm):>5}"
            f"{_pct(fed, len(group)):>7}{_pct(solved, len(otm)):>9}"
            f"{band:>6}{_pct(quoted, len(group)):>8}"
        )

    # A concrete sample matters as much as the counts: a feed can return a greeks
    # object full of zeros, which passes a null check but is useless for selection.
    # The same trap applies to the model — a solver pinned at a bracket endpoint
    # would report full coverage while every delta was garbage.
    sample = next((c for c in contracts if _in_band(c)), None)
    if sample is not None:
        source = "modelled" if sample.greeks_are_modelled else "feed"
        print(
            f"\n  band sample {sample.occ.raw} [{source}]: delta={sample.delta:+.4f} "
            f"iv={sample.iv:.4f} mid={sample.mid}"
        )

    # Unsolved OTM contracts are the ones that would actually hurt. Print them:
    # a silent count cannot be diagnosed, and this is the file someone reads at
    # 09:31 when selection has gone quiet.
    unsolved = [c for c in contracts if _is_otm(c, spot) and c.delta is None]
    if unsolved:
        print(f"  UNSOLVED OTM ({len(unsolved)}): "
              + ", ".join(f"{c.occ.strike}{c.occ.right}@{c.mid}" for c in unsolved[:8]))

    itm_unsolved = sum(1 for c in contracts if not _is_otm(c, spot) and c.delta is None)
    if itm_unsolved:
        print(f"  ({itm_unsolved} ITM contracts unsolved — quoted at or below parity, expected)")

    return (
        feed_total / total if total else 0.0,
        otm_solved / otm_total if otm_total else 0.0,
        band_total,
    )


def main() -> int:
    universe = sys.argv[1:] or DEFAULT_UNIVERSE
    today = today_et()
    results = [report(u, today) for u in universe]

    worst_feed = min((f for f, _, _ in results), default=0.0)
    worst_otm = min((o for _, o, _ in results), default=0.0)
    worst_band = min((b for _, _, b in results), default=0)

    print("\n" + "=" * 60)
    if worst_feed >= USABLE_THRESHOLD:
        print(f"A1 HOLDS — the feed itself populates greeks on >={USABLE_THRESHOLD:.0%}.")
        print("  The local Black-Scholes fallback is inert. Nothing to do.")
    else:
        print(f"A1 FAILS as measured — feed greeks on only {worst_feed:.1%} of contracts.")
        print("  Expected: they are OPRA-only and this account is on the indicative feed.")

    ok = worst_otm >= USABLE_THRESHOLD and worst_band >= MIN_BAND_CONTRACTS
    print(f"\nFALLBACK: {worst_otm:.1%} of OTM contracts solved "
          f"(need {USABLE_THRESHOLD:.0%}); {worst_band} in the "
          f"{BAND_LOW}-{BAND_HIGH} delta band (need {MIN_BAND_CONTRACTS}).")
    if ok:
        print("\nDELTA IS AVAILABLE — modelled, not quoted.")
        print("  Strike selection, Gate 7 and the VRP input can proceed. Run A2 next.")
        return 0

    print("\nDELTA IS NOT AVAILABLE at the strikes this agent trades.")
    if worst_band < MIN_BAND_CONTRACTS:
        print("  The chain solved but nothing sits near 0.16 delta — check the strike window.")
    else:
        print("  OTM contracts are failing to solve. Check the quote column: the model")
        print("  needs a two-sided quote, and refuses any price outside no-arbitrage bounds.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
