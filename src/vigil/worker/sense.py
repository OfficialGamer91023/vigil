"""The `sense` step: build the router's view of one underlying.

Split out of `sessions.py` because it is the one part of a cycle that is a pure
data-gathering function — no decisions, no orders, no journal writes — and
because both the entry cycle and the dashboard want the same view without one of
them having to run a trading cycle to get it.

Everything here is derived from the **underlying** except `iv_atm`, which is read
from the nearest-the-money contract and used as pricing context rather than as a
signal in its own right (§1.2).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal

from vigil.data.bars import daily_closes, prev_close_and_open, session_closes
from vigil.data.chain import Contract
from vigil.signals.history import rv_history
from vigil.signals.regime import MarketSnapshot, RegimeVerdict, classify
from vigil.signals.vol import realized_vol
from vigil.worker.broker import Broker


@dataclass(frozen=True, slots=True)
class MarketView:
    """One underlying, sensed. Carries the chain so the caller need not refetch."""

    underlying: str
    spot: Decimal
    snapshot: MarketSnapshot
    verdict: RegimeVerdict
    chain: tuple[Contract, ...]
    open_interest: dict[str, int]
    # Non-fatal data gaps, surfaced rather than swallowed. A cycle that reasoned
    # from a degraded input must be identifiable in the journal afterwards.
    warnings: tuple[str, ...] = ()

    @property
    def expiries(self) -> tuple[object, ...]:
        return tuple(sorted({c.occ.expiry for c in self.chain}))


async def sense(
    broker: Broker, underlying: str, *, max_dte: int, strike_window: Decimal = Decimal(25)
) -> MarketView | None:
    """Gather everything the router needs. `None` when there is no tradeable chain.

    The four market-data reads are independent, so they run **concurrently** via
    `asyncio.gather`. Sequentially this is four round trips inside a cycle that
    has a deadline; concurrently it is one. It also matters for the 200 req/min
    ceiling only in the sense that it does not make it worse — `gather` changes
    the latency, not the request count.
    """
    spot = await broker.spot(underlying)

    closes, session, chain, oi = await asyncio.gather(
        asyncio.to_thread(daily_closes, underlying),
        asyncio.to_thread(session_closes, underlying),
        broker.chain(underlying, spot=spot, max_dte=max_dte, strike_window=strike_window),
        broker.open_interest(underlying, spot=spot, max_dte=max_dte),
    )

    warnings: list[str] = []
    rv = realized_vol(session)
    if rv is None:
        # Not fatal, but it must not pass silently: `rv_annual = 0` collapses
        # `vrp_raw` to `iv_atm`, which makes every VRP figure downstream a
        # confident wrong answer rather than a missing one.
        warnings.append(
            f"only {len(session)} 5-min bars for {underlying}; realized vol "
            f"unavailable and VRP is unreliable this cycle"
        )
    if not chain:
        warnings.append(f"no live {underlying} chain in window")
        return None

    prev_close, session_open = prev_close_and_open(closes, spot)
    atm = min(chain, key=lambda c: abs(c.occ.strike - spot))

    snapshot = MarketSnapshot(
        underlying=underlying,
        spot=spot,
        prev_close=prev_close,
        session_open=session_open,
        daily_closes=closes,
        iv_atm=atm.iv or 0.0,
        rv_annual=rv or 0.0,
        # No historical implied vol on the free tier, so these accumulate forward
        # one session at a time; empty drops the router onto the documented
        # cold-start path (§4.3.1) rather than onto a guess.
        vrp_history=[],
        iv_history=[],
        rv_history=list(rv_history(underlying)),
    )

    return MarketView(
        underlying=underlying,
        spot=spot,
        snapshot=snapshot,
        verdict=classify(snapshot),
        chain=tuple(chain),
        open_interest=oi,
        warnings=tuple(warnings),
    )
