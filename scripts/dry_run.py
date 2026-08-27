"""End-to-end dry run: sense -> regime -> build -> gate. **Submits nothing.**

This is the A3 probe (PLAN §1.3): *is the gate stack permissive enough to fire on
a normal day?* It runs the real router, the real builders and the real kernel
against a live chain, and prints every gate verdict — so a rejection names the
binding gate instead of leaving the agent silently idle.

Run:  make dry-run          (or: uv run python scripts/dry_run.py [SYMBOL ...])
"""

from __future__ import annotations

import sys
from dataclasses import replace
from decimal import Decimal

from vigil.clock import is_market_open, now_et, today_et
from vigil.config import risk_config, strategy_config, universe_config
from vigil.data.alpaca_client import trading_client
from vigil.data.bars import daily_closes, prev_close_and_open, session_closes
from vigil.data.chain import fetch_chain, fetch_open_interest, spot_price
from vigil.domain import PortfolioState
from vigil.risk.context import KernelContext
from vigil.risk.kernel import evaluate
from vigil.signals.history import rv_history
from vigil.signals.regime import MarketSnapshot, classify
from vigil.signals.vol import realized_vol
from vigil.strategy.candidates import build_for_regime

# The primary universe comes from config/universe.yaml, not from a literal here:
# three scripts repeating ["SPY", "QQQ"] is three places to forget when the
# universe changes. Pass symbols on the command line to override.
DEFAULT_UNIVERSE = list(universe_config().primary)


def _live_portfolio() -> PortfolioState:
    """Read the real account so the portfolio gates reason about real numbers."""
    acct = trading_client().get_account()
    equity = Decimal(str(acct.equity))
    last = Decimal(str(acct.last_equity)) if acct.last_equity else equity
    return PortfolioState(
        equity=equity,
        # No peak history yet (that is the journal's job); today's equity is the
        # honest stand-in and makes Gate 4 a no-op rather than a fabricated pass.
        peak_equity=max(equity, last),
        day_pnl=equity - last,
    )


def run(underlying: str) -> bool:
    rcfg, scfg = risk_config(), strategy_config()
    print(f"\n{'=' * 72}\n{underlying}\n{'=' * 72}")

    spot = spot_price(underlying)
    closes = daily_closes(underlying)
    session = session_closes(underlying)
    rv = realized_vol(session.closes)
    if rv is None:
        print(f"  NOTE: only {len(session.closes)} regular-hours 5-min bars "
              f"available; realized vol is unmeasurable and the router will "
              f"stand down.")
    elif session.date != today_et():
        print(f"  NOTE: realized vol is from {session.date}, not today — the "
              f"current session has fewer than the minimum bars so far.")

    prev_close, session_open = prev_close_and_open(closes, spot)
    snap = MarketSnapshot(
        underlying=underlying,
        spot=spot,
        prev_close=prev_close,
        session_open=session_open,
        daily_closes=closes,
        iv_atm=0.0,
        rv_annual=rv,
        vrp_history=[],
        iv_history=[],
        # Backfilled by `make vrp`. Empty until then, which drops the router onto
        # the degenerate absolute test rather than the realized-vol proxy.
        rv_history=list(rv_history(underlying)),
    )

    chain = fetch_chain(underlying, spot=spot, max_dte=scfg.max_dte, strike_window=Decimal(25))
    if not chain:
        print("  no live chain in window (market closed, or window too narrow)")
        return False

    # ATM IV drives the VRP input; take it from the nearest-the-money contract.
    atm = min(chain, key=lambda c: abs(c.occ.strike - spot))
    # `replace` rather than __dict__: MarketSnapshot uses slots=True.
    snap = replace(snap, iv_atm=atm.iv or 0.0)

    verdict = classify(snap)
    print(f"  spot {spot}   regime {verdict.regime.value.upper()}"
          f"{'  [COLD START]' if verdict.cold_start else ''}")
    print(f"  {verdict.reason}")
    if verdict.structure is None:
        print("  -> router stands down; no candidates built.")
        return False

    state = _live_portfolio()
    print(f"  equity {state.equity}  day P&L {state.day_pnl}  open {len(state.open_structures)}")

    risk_budget = rcfg.max_risk_per_trade_pct * state.equity
    delta_budget = rcfg.max_dollar_delta_pct * state.equity - abs(state.net_dollar_delta)
    oi = fetch_open_interest(underlying, spot=spot, max_dte=scfg.max_dte)
    print(f"  risk budget ${risk_budget}   delta headroom ${delta_budget:.0f}   "
          f"OI for {len(oi)} contracts")

    expiries = sorted({c.occ.expiry for c in chain})
    built = 0
    for expiry in expiries:
        common = dict(
            underlying=underlying, spot=spot, expiry=expiry, risk_budget=risk_budget,
            remaining_delta_budget=delta_budget, open_interest=oi,
        )
        # **Through the dispatch, never around it.** This loop used to compute
        # `is_put = structure is PUT_CREDIT_SPREAD` itself, which mapped a
        # DEBIT_SPREAD verdict onto a call credit spread — the opposite trade.
        candidates = [build_for_regime(verdict, chain, **common)]  # type: ignore[arg-type]

        for cand in candidates:
            if cand is None:
                print(f"\n  {expiry} (DTE {(expiry - today_et()).days}): no buildable candidate")
                continue
            built += 1
            print(f"\n  {expiry} (DTE {(expiry - today_et()).days}): {cand.rationale}")
            print(f"    max loss ${cand.max_loss}  max profit ${cand.max_profit}  "
                  f"dollar delta ${cand.dollar_delta:.0f}")

            decision = evaluate(cand, state, KernelContext(now=now_et()), rcfg)
            for v in decision.verdicts:
                mark = "PASS" if v.passed else "FAIL"
                print(f"      [{mark}] {v.number:>2} {v.name:<24} {v.reason}")
            print(f"    => {decision.summary}")
            if decision.approved:
                return True
    if built == 0:
        print("\n  nothing buildable — check delta availability and strike coverage")
    return False


def main() -> int:
    universe = sys.argv[1:] or DEFAULT_UNIVERSE
    if not is_market_open():
        print(f"NOTE: market is closed ({now_et():%a %H:%M ET}). Quotes are stale; "
              f"treat every number below as indicative only.")

    approved = [run(u) for u in universe]
    print(f"\n{'=' * 72}")
    if any(approved):
        print("A3: the gate stack APPROVED at least one candidate — the agent can fire.")
        return 0
    print("A3: NO candidate approved. The per-gate output above names the binding gate.")
    print("    A gate that never passes is as broken as one that never fires (§5.2).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
