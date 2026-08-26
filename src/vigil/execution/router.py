"""**The single submit path.** Hard rule #4.

Nothing else in this codebase may call Alpaca's order API. Every entry goes
through `submit_entry`, and `submit_entry` calls the risk kernel first — so there
is no code path, however deep, that reaches the broker ungated. If you find
yourself importing `TradingClient` to place an order somewhere else, that is a
bug, not a shortcut.

The submit sequence (§2.5, §2.6):

    kernel.evaluate  →  ladder rung  →  mleg limit  →  wait  →  re-price
                                                          ↓ filled
                                                    resting GTC profit target
"""

from __future__ import annotations

import time as _time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

from alpaca.trading.client import TradingClient
from alpaca.trading.models import Order

from vigil.config import RiskConfig, StrategyConfig, risk_config, strategy_config
from vigil.domain import KernelDecision, OpenStructure, PortfolioState, TradeProposal
from vigil.execution.mleg import (
    MlegConstructionError,
    build_close_from_legs,
    build_closing_order,
    build_entry_order,
)
from vigil.execution.pricing import (
    RUNG_WAIT_SECONDS,
    Ladder,
    credit_ladder,
    debit_ladder,
    debit_profit_target_price,
    natural_debit_ceiling,
    premium_multiple_target_price,
    profit_target_price,
)
from vigil.risk.context import KernelContext
from vigil.risk.kernel import evaluate

# Terminal states. Anything else means the order is still working.
_FILLED = {"filled"}
_DEAD = {"canceled", "cancelled", "expired", "rejected", "done_for_day"}


class RiskKernelRejection(RuntimeError):
    """The kernel refused the proposal. Never retried, never overridden."""

    def __init__(self, decision: KernelDecision) -> None:
        super().__init__(decision.summary)
        self.decision = decision


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    proposal: TradeProposal
    decision: KernelDecision
    entry_order: Order | None
    target_order: Order | None
    filled: bool
    rungs_used: int
    note: str = ""
    # How many contracts actually filled. Differs from `proposal.contracts` on a
    # partial, and the journal needs the real number, not the intended one.
    filled_contracts: int = 0

    @property
    def partial(self) -> bool:
        return self.filled and 0 < self.filled_contracts < self.proposal.contracts


def _filled_qty(order: Order) -> int:
    """Contracts (mleg *packages*) actually filled. Absent or unparseable means 0.

    Read defensively: this number decides whether we believe a position exists at
    the broker, and guessing high would rest an exit for contracts we do not own.
    """
    raw = getattr(order, "filled_qty", None)
    if raw is None:
        return 0
    try:
        return int(Decimal(str(raw)))
    except (ArithmeticError, ValueError):
        return 0


def _as_order(result: Order | dict[str, object]) -> Order:
    """Narrow alpaca-py's `Order | dict` return type.

    The SDK returns a raw dict when the client is in raw-data mode. We never
    enable that, so a dict here means a misconfigured client — worth failing
    loudly rather than duck-typing our way into an AttributeError later.
    """
    if isinstance(result, dict):
        raise TypeError("TradingClient returned raw data; Vigil requires model objects")
    return result


def _status(order: Order) -> str:
    """Normalise status across SDK enum / string representations."""
    raw = getattr(order.status, "value", order.status)
    return str(raw).lower().rsplit(".", 1)[-1]


def submit_entry(
    proposal: TradeProposal,
    state: PortfolioState,
    context: KernelContext,
    *,
    client: TradingClient,
    risk: RiskConfig | None = None,
    strategy: StrategyConfig | None = None,
    sleep: Callable[[float], None] = _time.sleep,
    poll_seconds: int = RUNG_WAIT_SECONDS,
) -> ExecutionResult:
    """Gate, then walk the ladder, then rest a profit target on the fill.

    `sleep` is injected so tests can drive the ladder without waiting a real 60
    seconds. It is the only concession this module makes to testability, and it
    changes no behaviour in production.
    """
    rcfg = risk or risk_config()
    scfg = strategy or strategy_config()

    # ---- the gate. Before anything reaches the broker. --------------------- #
    decision = evaluate(proposal, state, context, rcfg)
    if not decision.approved:
        raise RiskKernelRejection(decision)

    # ---- the ladder (§2.5) ------------------------------------------------- #
    # §2.5: start at the mid (`limit_price`), concede toward the natural. Credit
    # and debit structures concede in opposite directions and stop at different
    # kinds of boundary, which is why the ladder is chosen rather than assumed.
    ladder: Ladder = _ladder_for(proposal, rcfg)

    entry: Order | None = None
    for i, price in enumerate(ladder.rungs, start=1):
        submitted = _as_order(
            client.submit_order(build_entry_order(proposal, price, rung=i))
        )
        sleep(poll_seconds)
        entry = _as_order(client.get_order_by_id(submitted.id))
        status = _status(entry)

        if status in _FILLED:
            target = _rest_profit_target(proposal, entry, client=client, strategy=scfg)
            return ExecutionResult(proposal, decision, entry, target, True, i,
                                   filled_contracts=proposal.contracts)

        if status not in _DEAD:
            # Still working at this rung: cancel before re-pricing, so we never
            # have two live tickets for the same structure.
            client.cancel_order_by_id(entry.id)
            # Re-read after the cancel. A working order can be *partially* filled,
            # and cancelling only kills the remainder — the fill that already
            # happened is real and is not reported until the order settles.
            entry = _as_order(client.get_order_by_id(entry.id))

        # **Partial fills are positions.** ~10% of paper fills arrive partial
        # (§1.2), and the failure this guards against is specific: advancing to
        # the next rung would leave live contracts at the broker with no resting
        # exit — a reconciliation defect by §2.6's own definition — and then
        # submit a *second* entry ticket on top of them. Once any quantity has
        # filled, the ladder is over.
        got = _filled_qty(entry)
        if got >= proposal.contracts:
            # Filled during the cancel race. Not a partial after all.
            target = _rest_profit_target(proposal, entry, client=client, strategy=scfg)
            return ExecutionResult(proposal, decision, entry, target, True, i,
                                   note="filled while cancelling; ladder stopped",
                                   filled_contracts=proposal.contracts)
        if got > 0:
            target = _rest_profit_target(
                proposal, entry, client=client, strategy=scfg, contracts=got
            )
            return ExecutionResult(
                proposal, decision, entry, target, True, i,
                note=(f"PARTIAL_FILL {got}/{proposal.contracts} contracts; "
                      f"remainder cancelled, resting target sized to the fill"),
                filled_contracts=got,
            )

    return ExecutionResult(
        proposal, decision, entry, None, False, len(ladder),
        note="NO_FILL — ladder exhausted at the kernel's price boundary; "
             "a missed trade is free",
    )


def _ladder_for(proposal: TradeProposal, rcfg: RiskConfig) -> Ladder:
    """Pick the ladder that concedes in the direction this structure trades.

    A credit structure concedes by asking for *less*, bounded below by Gate 9's
    credit floor. A debit structure concedes by offering *more*, bounded above by
    the package's natural price. Running a credit ladder against a debit spread
    would bid progressively less for something we are trying to buy, which simply
    never fills.
    """
    if proposal.is_credit:
        floor = rcfg.min_credit_pct_of_width * proposal.width
        return credit_ladder(target_credit=proposal.limit_price, min_credit=floor)
    ceiling = natural_debit_ceiling(proposal.net_credit)
    return debit_ladder(target_debit=abs(proposal.limit_price), max_debit=ceiling)


def _rest_profit_target(
    proposal: TradeProposal,
    entry: Order,
    *,
    client: TradingClient,
    strategy: StrategyConfig,
    contracts: int | None = None,
) -> Order:
    """Submit the resting GTC closing order immediately on fill (§2.6).

    Not a polled target — an actual order in the book. It captures intraday spikes
    a 15-minute sweep sleeps through, and it survives the worker dying, which is
    the same principle as the LLM fallback applied to execution.

    An open structure without one of these is a **reconciliation defect**, so the
    exception here is deliberately not swallowed: a silent failure would leave a
    position with no exit and no signal that anything is wrong.

    `contracts` carries the *actual* filled quantity, which on a partial is less
    than the proposal asked for.
    """
    filled_price = (
        Decimal(str(entry.filled_avg_price))
        if entry.filled_avg_price
        else abs(proposal.net_credit)
    )
    if proposal.is_credit:
        target = profit_target_price(filled_price, strategy.profit_target_pct)
    elif proposal.is_long_only:
        # Max profit is unbounded, so "50% of max profit" has no meaning. Exit at
        # a multiple of the premium instead. Routing this through the debit-spread
        # formula with width 0 would rest a sell order at *half* the premium.
        target = premium_multiple_target_price(
            filled_price, strategy.convexity_profit_target_multiple
        )
    else:
        # A debit structure exits *above* what it paid — see
        # `debit_profit_target_price`. Using the credit formula here would rest an
        # exit below cost, i.e. an order to take a guaranteed loss.
        target = debit_profit_target_price(
            filled_price, proposal.width, strategy.profit_target_pct
        )
    order = build_closing_order(
        proposal, target,
        client_order_id=f"{proposal.client_order_id}-tgt",
        contracts=contracts,
    )
    return _as_order(client.submit_order(order))


# --------------------------------------------------------------------------- #
# Closing. Same module (hard rule #4), deliberately NOT kernel-gated.
# --------------------------------------------------------------------------- #

def submit_close(
    structure: OpenStructure,
    limit_price: Decimal,
    *,
    client: TradingClient,
    reason: str,
    good_till_cancelled: bool = False,
) -> Order:
    """Close an open structure. **The kernel does not vote on exits.**

    Hard rule #4 puts every order through this module, and this function honours
    that. What it deliberately does *not* do is call `evaluate()` first, and the
    reason is worth stating because the asymmetry looks like an oversight:

    **A gate that can block an exit is not a safety feature, it is a trap.** The
    twelve gates exist to bound the risk we *take on*. A closing order only ever
    reduces exposure — and every gate that could plausibly fire here would fire
    for a reason that is an argument *for* closing, not against it. Gate 3 halts
    new entries on a bad day; Gate 5 refuses a seventh structure; Gate 11 refuses
    entries in the last twenty minutes. Route the 15:40 flatten through Gate 11
    and it would refuse to flatten at 15:40.

    So closes are unconditional, and the safety property is the narrower one:
    they can only ever be constructed from legs that are already open, in the
    closing direction, for a quantity we already hold.
    """
    if not structure.legs:
        raise MlegConstructionError(
            f"cannot close {structure.underlying} {structure.expiry}: no leg symbols "
            "on the structure. Reconciliation must populate them from the broker."
        )
    order = build_close_from_legs(
        [(leg.symbol, leg.ratio_qty, leg.is_short) for leg in structure.legs],
        limit_price,
        contracts=structure.contracts,
        client_order_id=_close_client_order_id(structure, reason),
        good_till_cancelled=good_till_cancelled,
    )
    return _as_order(client.submit_order(order))


def _close_client_order_id(structure: OpenStructure, reason: str) -> str:
    """A unique, *legible* idempotency key for a close.

    Legible because these are the rows a human reads when asking what the agent
    did to a position; unique because hard rule #9 makes a duplicate an integrity
    error, and a sweep that retries after a timeout must not double-close.
    """
    import uuid

    tag = reason.split()[0][:12].lower().strip(":,;")
    return f"vigil-cls-{tag}-{uuid.uuid4().hex[:12]}"
