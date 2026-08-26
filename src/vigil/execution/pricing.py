"""The price ladder (§2.5) and the profit-target price (§2.6).

Indicative option quotes are approximations, so **never a market order**. The
ladder starts at the credit we want and concedes one tick at a time toward the
natural, with a hard floor at the minimum acceptable credit. *A missed trade is
free; a bad fill is not* — so the floor never moves, and running out of rungs
means cancel-and-log, not "take whatever is there."
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_UP, Decimal

# Option prices trade in $0.01 increments on these underlyings. Rounding to the
# tick is not cosmetic — the broker rejects a limit that is not on one.
TICK = Decimal("0.01")

MAX_RUNGS = 3
RUNG_WAIT_SECONDS = 20


def round_to_tick(price: Decimal, *, favour_us: bool = True) -> Decimal:
    """Round to the contract tick.

    `favour_us` rounds a credit *up* and a debit *down* — i.e. always to the side
    that is worse for us to fill. Rounding the other way would quietly shave the
    credit floor by a cent per rung, which is exactly the amount Gate 9 is arguing
    about at 0-2 DTE.
    """
    rounding = ROUND_UP if favour_us else ROUND_DOWN
    return price.quantize(TICK, rounding=rounding)


@dataclass(frozen=True, slots=True)
class Ladder:
    """The full sequence of limit prices, computed once before the first submit.

    Precomputing means the retry path has no arithmetic in it: a re-price is an
    index increment, not a fresh calculation against a quote that has since moved.
    """

    rungs: tuple[Decimal, ...]
    floor: Decimal

    def __len__(self) -> int:
        return len(self.rungs)

    def __getitem__(self, i: int) -> Decimal:
        return self.rungs[i]


def credit_ladder(
    *, target_credit: Decimal, min_credit: Decimal, rungs: int = MAX_RUNGS
) -> Ladder:
    """Descending credits from `target_credit` down to `min_credit`.

    We are *receiving* the credit, so conceding means asking for less. The floor
    is included as the final rung and never crossed.
    """
    start = round_to_tick(target_credit)
    floor = round_to_tick(min_credit)
    if start <= floor:
        return Ladder(rungs=(floor,), floor=floor)

    span = start - floor
    step = round_to_tick(span / rungs, favour_us=False)
    if step <= 0:
        step = TICK

    out: list[Decimal] = []
    price = start
    while price > floor and len(out) < rungs:
        out.append(price)
        price -= step
    out.append(floor)
    # dict.fromkeys de-duplicates while preserving order — a narrow span can make
    # two rungs land on the same tick, and submitting the same price twice just
    # burns 20 seconds.
    return Ladder(rungs=tuple(dict.fromkeys(out)), floor=floor)


def debit_ladder(
    *, target_debit: Decimal, max_debit: Decimal, rungs: int = MAX_RUNGS
) -> Ladder:
    """Ascending debits from `target_debit` up to `max_debit`.

    The mirror of `credit_ladder`, and the mirror is not cosmetic: on a debit
    structure we are *paying*, so conceding means offering **more**, and the hard
    boundary is a ceiling rather than a floor. Reusing the credit ladder here
    would concede in the wrong direction — bidding progressively less for
    something we want to own, which never fills.
    """
    start = round_to_tick(target_debit, favour_us=False)
    ceiling = round_to_tick(max_debit, favour_us=False)
    if start >= ceiling:
        return Ladder(rungs=(ceiling,), floor=ceiling)

    span = ceiling - start
    step = round_to_tick(span / rungs, favour_us=False)
    if step <= 0:
        step = TICK

    out: list[Decimal] = []
    price = start
    while price < ceiling and len(out) < rungs:
        out.append(price)
        price += step
    out.append(ceiling)
    return Ladder(rungs=tuple(dict.fromkeys(out)), floor=ceiling)


def max_debit_for_ratio(width: Decimal, max_loss_to_profit_ratio: Decimal) -> Decimal:
    """The largest debit that still satisfies Gate 9's max-loss:max-profit ratio.

    For a debit spread, max loss is the debit `D` and max profit is `W − D`, so
    the gate reads `D / (W − D) ≤ r`, which rearranges to `D ≤ r·W / (1 + r)`.

    Deriving the ceiling from the gate — rather than inventing a separate
    `max_debit_pct_of_width` knob — is the same discipline the credit ladder
    already follows: the ladder never concedes to a price the kernel would then
    reject, and there is exactly one place that decides what "too expensive" means.
    """
    if width <= 0 or max_loss_to_profit_ratio <= 0:
        return Decimal(0)
    return max_loss_to_profit_ratio * width / (Decimal(1) + max_loss_to_profit_ratio)


def profit_target_price(net_credit: Decimal, target_pct: Decimal) -> Decimal:
    """The debit to pay to close a credit structure at `target_pct` of max profit.

    Closing at 50% of max profit on a $0.20 credit means buying it back for $0.10.
    Rounded *down* so the resting order is at least as aggressive as the target —
    an order that never fills is not a profit target, it is a decoration.
    """
    remaining = net_credit * (Decimal(1) - target_pct)
    price = round_to_tick(remaining, favour_us=False)
    # Never below one tick: a $0.00 limit is not a valid order.
    return max(price, TICK)


def debit_profit_target_price(
    net_debit: Decimal, width: Decimal, target_pct: Decimal
) -> Decimal:
    """The credit to receive to close a *debit* structure at `target_pct` of max profit.

    Max profit on a debit spread is `W − D`, so taking half of it means selling
    the package back for `D + 0.5·(W − D)` — a number **above** what we paid.
    The credit-structure version subtracts from the entry price; this one adds to
    it, and confusing the two would rest an exit at an instant loss.
    """
    if net_debit < 0:
        net_debit = abs(net_debit)
    profit = max(width - net_debit, Decimal(0))
    price = round_to_tick(net_debit + profit * target_pct, favour_us=False)
    return max(price, TICK)
