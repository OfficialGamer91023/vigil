"""Translating a `TradeProposal` into an Alpaca multi-leg order request.

§1.1: `order_class="mleg"`, up to 4 legs, each carrying `symbol`, `ratio_qty`,
`side` and `position_intent`. `limit_price` is the **net debit/credit for the
whole package**, filled as one unit — so there is no legging risk and no moment
where a short leg sits uncovered.

Verified against alpaca-py 0.44.0: the request carries **no top-level `side`**;
direction lives entirely on the legs (docs/CLI_NOTES.md §1).
"""

from __future__ import annotations

from decimal import Decimal

from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest

from vigil.domain import TradeProposal

# §1.1: an mleg ticket carries at most four legs. An iron condor is exactly four,
# so this is a real boundary rather than a theoretical one.
MAX_LEGS = 4


class MlegConstructionError(ValueError):
    """Raised when a proposal cannot be expressed as a valid mleg ticket."""


def rung_client_order_id(base: str, rung: int) -> str:
    """The idempotency key for one ladder rung.

    **Every rung is a distinct order and needs a distinct id.** Alpaca treats
    `client_order_id` as unique, so re-submitting the base id at a conceded price
    is rejected outright — and even if the broker allowed it, the `UNIQUE NOT NULL`
    constraint on `orders.client_order_id` (hard rule #9) would reject it at
    journal-write time. Suffixing keeps the base id as the *structure's* identity,
    which is what Gate 12's duplicate check reasons about, while giving each
    submission its own key.
    """
    return f"{base}-r{rung}"


def build_entry_order(
    proposal: TradeProposal, limit_price: Decimal, *, rung: int = 1
) -> LimitOrderRequest:
    """The opening ticket. `limit_price` comes from the ladder, not the proposal.

    Taking the price as an argument rather than reading `proposal.limit_price` is
    what lets the ladder re-price on a retry without mutating (or rebuilding) the
    proposal the kernel already approved. `rung` does the same job for the
    idempotency key.
    """
    if not 2 <= len(proposal.legs) <= MAX_LEGS:
        raise MlegConstructionError(
            f"mleg supports 2-{MAX_LEGS} legs, proposal has {len(proposal.legs)}"
        )
    if proposal.contracts < 1:
        raise MlegConstructionError(f"contract count must be >= 1, got {proposal.contracts}")

    legs = [
        OptionLegRequest(
            symbol=leg.symbol,
            ratio_qty=leg.ratio_qty,
            side=OrderSide.SELL if leg.is_short else OrderSide.BUY,
            position_intent=(
                PositionIntent.SELL_TO_OPEN if leg.is_short else PositionIntent.BUY_TO_OPEN
            ),
        )
        for leg in proposal.legs
    ]

    return LimitOrderRequest(
        qty=proposal.contracts,
        order_class=OrderClass.MLEG,
        # `day` on entries: an unfilled entry must not survive into a session
        # whose regime read no longer applies.
        time_in_force=TimeInForce.DAY,
        limit_price=float(limit_price),
        legs=legs,
        client_order_id=rung_client_order_id(proposal.client_order_id, rung),
        extended_hours=False,
    )


def build_closing_order(
    proposal: TradeProposal,
    limit_price: Decimal,
    *,
    client_order_id: str,
    contracts: int | None = None,
    good_till_cancelled: bool = True,
) -> LimitOrderRequest:
    """The mirror ticket that closes the structure — every leg's intent inverted.

    Used for the resting profit-target order (§2.6), which is submitted the moment
    an entry fill is confirmed and left live at the broker. GTC by default: the
    whole point is that it survives the worker dying.

    `contracts` overrides the proposal's size, because a **partial fill** must be
    covered for the quantity that actually filled. Resting a target for the size
    we *asked* for would leave an order the account cannot honour.
    """
    qty = proposal.contracts if contracts is None else contracts
    if qty < 1:
        raise MlegConstructionError(f"closing order needs >= 1 contract, got {qty}")
    legs = [
        OptionLegRequest(
            symbol=leg.symbol,
            ratio_qty=leg.ratio_qty,
            # Closing reverses the side: what we sold to open, we buy to close.
            side=OrderSide.BUY if leg.is_short else OrderSide.SELL,
            position_intent=(
                PositionIntent.BUY_TO_CLOSE if leg.is_short else PositionIntent.SELL_TO_CLOSE
            ),
        )
        for leg in proposal.legs
    ]

    return LimitOrderRequest(
        qty=qty,
        order_class=OrderClass.MLEG,
        time_in_force=TimeInForce.GTC if good_till_cancelled else TimeInForce.DAY,
        limit_price=float(limit_price),
        legs=legs,
        client_order_id=client_order_id,
        extended_hours=False,
    )
