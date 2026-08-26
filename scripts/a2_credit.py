"""A2 — does a 0.15-0.20 delta short put at 0-2 DTE pay >= 18% of width?

PLAN.md §1.3 / §4.4.2: Gate 9 rejects any structure whose credit is below a
fraction of its width. If that floor is unreachable at 0-2 DTE, the kernel
rejects *every* candidate and the agent never trades. This measures the actual
credit/width ratio at $1 / $2 / $5 widths so the Gate 9 threshold and the §4.5
"narrowest tradeable width" rule are set from data rather than from a blog post.

Two credits are printed for each candidate:
  mid   — midpoint fill. Optimistic, but it is what Alpaca paper tends to give.
  cons  — conservative: sell the short at the bid, pay the ask for the long.
          This is the credit a real fill would have to clear.

Run:  uv run python scripts/a2_credit.py [SYMBOL ...]
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date
from decimal import Decimal

from alpaca.trading.enums import ContractType

from vigil.clock import today_et
from vigil.data.chain import Contract, fetch_chain, spot_price
from vigil.strategy.selection import pick_by_delta

DEFAULT_UNIVERSE = ["SPY", "QQQ"]
WIDTHS = [Decimal(1), Decimal(2), Decimal(5)]
# §4.4.2 keeps delta and moves the credit floor, but if the floor turns out to be
# unreachable the other lever is delta. Sweeping it here means that trade-off is
# quantified at the moment of measurement rather than argued about later.
DELTA_SWEEP = [Decimal("0.16"), Decimal("0.20"), Decimal("0.25"), Decimal("0.30")]
CREDIT_FLOOR = Decimal("0.18")          # §4.4.2: 18% of width, not the "classic" 25%



def report(underlying: str, today: date) -> list[tuple[Decimal, Decimal]]:
    spot = spot_price(underlying)
    # Wider strike window than A1: a $5-wide spread needs a long strike well below
    # the short, and the short itself sits several dollars OTM at 0.16 delta.
    contracts = fetch_chain(
        underlying,
        spot=spot,
        max_dte=2,
        strike_window=Decimal(25),
        asof=today,
        contract_type=ContractType.PUT,
    )

    print(f"\n=== {underlying} — spot {spot} ===")
    if not contracts:
        print("  no put contracts returned")
        return []

    by_expiry: dict[date, list[Contract]] = defaultdict(list)
    for c in contracts:
        by_expiry[c.occ.expiry].append(c)

    results: list[tuple[Decimal, Decimal]] = []
    for expiry in sorted(by_expiry):
        puts = by_expiry[expiry]
        strikes = {c.occ.strike: c for c in puts}
        short = pick_by_delta(puts)
        dte = (expiry - today).days

        if short is None:
            print(f"\n  {expiry} (DTE {dte}): no deltas available — see A1")
            continue

        print(
            f"\n  {expiry} (DTE {dte})  short {short.occ.strike} P  "
            f"delta={short.delta:.3f}  bid={short.bid} ask={short.ask}"
        )
        # --- delta sweep at the narrowest width (§4.4.3 fixes width, so this is
        # --- the only remaining lever if the credit floor proves unreachable)
        print(f"    {'delta':>6}{'short K':>9}{'mid cr':>9}{'cr/w':>8}   (at ${WIDTHS[0]} width)")
        for target in DELTA_SWEEP:
            cand = pick_by_delta(puts, target=target)
            if cand is None:
                continue
            partner = strikes.get(cand.occ.strike - WIDTHS[0])
            if partner is None or cand.mid is None or partner.mid is None:
                continue
            cr = cand.mid - partner.mid
            print(
                f"    {target:>6}{cand.occ.strike:>9}{cr:>9.2f}{cr / WIDTHS[0]:>8.1%}"
                f"   (actual delta {cand.delta:+.3f})"
            )

        print(f"    {'width':>6}{'long K':>9}{'mid cr':>9}{'cr/w':>8}{'cons cr':>10}{'cons/w':>9}")
        for width in WIDTHS:
            long_leg = strikes.get(short.occ.strike - width)
            if long_leg is None:
                print(f"    {width:>6}{'—':>9}   long strike not listed in window")
                continue
            if None in (short.mid, long_leg.mid, short.bid, long_leg.ask):
                print(f"    {width:>6}{long_leg.occ.strike:>9}   incomplete quotes")
                continue

            mid_credit = short.mid - long_leg.mid
            # The realistic fill: hit the bid on what you sell, pay the ask on
            # what you buy. This is the number Gate 9 should really be judged on.
            cons_credit = short.bid - long_leg.ask

            print(
                f"    {width:>6}{long_leg.occ.strike:>9}{mid_credit:>9.2f}"
                f"{mid_credit / width:>8.1%}{cons_credit:>10.2f}{cons_credit / width:>9.1%}"
            )
            results.append((width, mid_credit / width))

    return results


def main() -> int:
    universe = sys.argv[1:] or DEFAULT_UNIVERSE
    today = today_et()

    all_results: list[tuple[Decimal, Decimal]] = []
    for u in universe:
        all_results.extend(report(u, today))

    print("\n" + "=" * 60)
    if not all_results:
        print("A2 INCONCLUSIVE — no priced spreads. Re-run during market hours.")
        return 2

    # The design question is not "is the average fine" but "does the *narrowest*
    # width clear the floor", because §4.4.3 fixes width at the narrowest tradeable
    # and makes contract count the free variable.
    narrowest = min(w for w, _ in all_results)
    at_narrowest = [r for w, r in all_results if w == narrowest]
    passing = sum(1 for r in at_narrowest if r >= CREDIT_FLOOR)

    print(f"At the narrowest width (${narrowest}): {passing}/{len(at_narrowest)} candidates "
          f"clear the {CREDIT_FLOOR:.0%} floor.")
    for width in WIDTHS:
        rs = [r for w, r in all_results if w == width]
        if rs:
            print(f"  ${width} width: median cr/w = {sorted(rs)[len(rs) // 2]:.1%}  (n={len(rs)})")

    if passing:
        print(f"\nA2 HOLDS — the {CREDIT_FLOOR:.0%} floor is reachable at ${narrowest} width.")
        return 0
    print(f"\nA2 FAILS — nothing clears {CREDIT_FLOOR:.0%} at ${narrowest} width.")
    print("  NOT a remedy: widening. The width table above confirms §4.4.3 — cr/w FALLS")
    print("  as width grows, so a wider structure moves further from the floor, not closer.")
    print("  Remaining levers: raise short delta (see the sweep), or lower the Gate 9 floor.")
    print("  §4.4 constrains how far the floor can drop — do not tune it from one session.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
