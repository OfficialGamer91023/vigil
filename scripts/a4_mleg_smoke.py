"""Smoke test: place ONE mleg put credit spread by hand, confirm it, cancel.

PLAN.md §11 Day -1: "Place one mleg limit order by hand via the SDK; confirm the
fill; cancel-all." This answers the question no amount of reading answers -- does
the whole premise work end to end on this account?

It also settles one thing the docs are ambiguous about and that §2.5's price
ladder depends on: **the sign convention of `limit_price` on a net-credit mleg
package.** The script submits with the convention it believes is right and prints
what came back, so the answer goes into docs/CLI_NOTES.md as a measured fact.

This is NOT part of the trading system. It bypasses the risk kernel by design,
because the kernel does not exist yet -- which is exactly why it is a one-shot
script behind an explicit flag and not an importable function.

Run:
  uv run python scripts/a4_mleg_smoke.py                    # dry run, submits nothing
  uv run python scripts/a4_mleg_smoke.py --submit           # actually places it
  uv run python scripts/a4_mleg_smoke.py --cancel-all       # clean up
"""

from __future__ import annotations

import sys
import time
import uuid
from decimal import Decimal

from alpaca.trading.enums import ContractType, OrderClass, OrderSide, PositionIntent, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest

from vigil.clock import today_et
from vigil.data.alpaca_client import trading_client
from vigil.data.chain import fetch_chain, spot_price
from vigil.strategy.selection import pick_by_delta

UNDERLYING = "SPY"
WIDTH = Decimal(1)          # §4.5: narrowest tradeable on SPY
QTY = 1                     # one contract; this is a probe, not a position


def build_spread() -> tuple[OptionLegRequest, OptionLegRequest, Decimal] | None:
    """Find a 0.16-delta put and the strike $1 below it, and price the package."""
    spot = spot_price(UNDERLYING)
    puts = fetch_chain(
        UNDERLYING, spot=spot, max_dte=2, strike_window=Decimal(25), contract_type=ContractType.PUT
    )
    if not puts:
        print("no put contracts in window")
        return None

    # Nearest expiry only: this is a probe of the order path, so the shortest-dated
    # contract minimises how long a stray fill could sit open.
    nearest = min(c.occ.expiry for c in puts)
    same_expiry = [c for c in puts if c.occ.expiry == nearest]
    strikes = {c.occ.strike: c for c in same_expiry}

    short = pick_by_delta(same_expiry)
    if short is None:
        print("no deltas available -- run scripts/a1_greeks.py first")
        return None
    long_leg = strikes.get(short.occ.strike - WIDTH)
    if long_leg is None or short.bid is None or long_leg.ask is None:
        print(f"cannot price a ${WIDTH} spread below {short.occ.strike}")
        return None

    # Conservative credit: sell at the bid, buy at the ask. Deliberately pessimistic
    # -- if this fills, anything the ladder later asks for will fill too.
    credit = short.bid - long_leg.ask
    print(f"spot {spot}  expiry {nearest} (DTE {(nearest - today_et()).days})")
    print(f"  SELL {short.occ.raw}  delta={short.delta:.3f}  bid={short.bid} ask={short.ask}")
    print(f"  BUY  {long_leg.occ.raw}  bid={long_leg.bid} ask={long_leg.ask}")
    print(f"  net credit {credit}  = {credit / WIDTH:.1%} of ${WIDTH} width")

    if credit <= 0:
        print("  credit is non-positive -- refusing to submit a debit as a credit spread")
        return None

    return (
        OptionLegRequest(
            symbol=short.occ.raw,
            ratio_qty=1,
            side=OrderSide.SELL,
            position_intent=PositionIntent.SELL_TO_OPEN,
        ),
        OptionLegRequest(
            symbol=long_leg.occ.raw,
            ratio_qty=1,
            side=OrderSide.BUY,
            position_intent=PositionIntent.BUY_TO_OPEN,
        ),
        credit,
    )


def submit(short_leg: OptionLegRequest, long_leg: OptionLegRequest, credit: Decimal) -> None:
    # client_order_id is the idempotency key (hard rule #9). Later it becomes a
    # UNIQUE NOT NULL column; here it just proves the field round-trips.
    coid = f"vigil-smoke-{uuid.uuid4().hex[:12]}"

    req = LimitOrderRequest(
        qty=QTY,
        order_class=OrderClass.MLEG,
        time_in_force=TimeInForce.DAY,   # never gtc for a probe
        limit_price=float(round(credit, 2)),
        legs=[short_leg, long_leg],
        client_order_id=coid,
        extended_hours=False,
    )
    order = trading_client().submit_order(req)
    print(f"\nsubmitted id={order.id} coid={coid} status={order.status}")

    # Poll briefly rather than assuming. Paper fills are fast but not instant.
    for _ in range(10):
        time.sleep(1)
        order = trading_client().get_order_by_id(order.id)
        print(f"  status={order.status} filled_qty={order.filled_qty} avg={order.filled_avg_price}")
        if str(order.status) in ("OrderStatus.FILLED", "filled"):
            break

    print("\nSIGN CONVENTION: submitted limit_price as a POSITIVE net credit.")
    print(f"  broker reported filled_avg_price = {order.filled_avg_price}")
    print("  -> record this in docs/CLI_NOTES.md before writing the §2.5 price ladder.")
    print("\nRemember to close: uv run python scripts/a4_mleg_smoke.py --cancel-all")


def cancel_all() -> None:
    client = trading_client()
    client.cancel_orders()
    print("cancelled all open orders")
    positions = client.get_all_positions()
    if positions:
        print(f"{len(positions)} open position(s) remain:")
        for p in positions:
            print(f"  {p.symbol} qty={p.qty} unrealized={p.unrealized_pl}")
        print("close them: client.close_all_positions(cancel_orders=True)")
    else:
        print("no open positions")


def main() -> int:
    if "--cancel-all" in sys.argv:
        cancel_all()
        return 0

    built = build_spread()
    if built is None:
        return 1

    if "--submit" not in sys.argv:
        print("\nDRY RUN -- nothing submitted. Re-run with --submit to place it.")
        return 0

    submit(*built)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
