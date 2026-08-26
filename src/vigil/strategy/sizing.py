"""Contract sizing. §4.4.3: width is fixed, **contract count is the free variable**.

The arithmetic that settles it: under a fixed max-loss budget B, with n contracts
of width W collecting credit C,

    max loss = n · (W − C) · 100 ≤ B        premium = n · C · 100

so premium per dollar of risk is C/(W − C), which *rises* as W narrows. Measured
26 Aug 2026 and confirmed: credit/width was 14.5% at $1, 11.5% at $2, 8.9% at $5.
Therefore fix W at the narrowest the chain supports and solve for n.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal

from vigil.domain import CONTRACT_MULTIPLIER


def max_contracts_for_budget(
    width: Decimal, net_credit: Decimal, budget: Decimal
) -> int:
    """Largest whole contract count whose max loss fits inside `budget`.

    Rounds **down**, always. Rounding to nearest would let a position land a cent
    over the risk budget, and Gate 2 would then reject a proposal this function
    just declared acceptable — two components disagreeing about the same rule.

    Handles both signs of `net_credit`, and must: on a **debit** structure the max
    loss is the premium paid, not `width − credit`. Feeding a negative credit into
    the credit formula gives `width + debit`, which overstates the loss and
    silently undersizes the convexity sleeve. This mirrors
    `TradeProposal.max_loss_per_contract` exactly — the same rule, computed once
    per side of the codebase, which is the only way the two stay in agreement.
    """
    if net_credit > 0:
        loss_per_contract = (width - net_credit) * CONTRACT_MULTIPLIER
    else:
        loss_per_contract = abs(net_credit) * CONTRACT_MULTIPLIER
    if loss_per_contract <= 0 or budget <= 0:
        return 0
    n = (budget / loss_per_contract).to_integral_value(rounding=ROUND_DOWN)
    return max(int(n), 0)


def max_contracts_for_dollar_delta(
    delta_per_contract: Decimal, remaining_delta_budget: Decimal
) -> int:
    """Largest count whose dollar delta fits the *remaining* Gate 7 headroom.

    This exists because Gate 7 binds far harder than Gate 2 on directional
    structures: a $1-wide SPY put spread at 0.16/0.11 delta carries ~$3,800 of
    dollar delta, against a $5,000 portfolio limit at $100k equity. Sizing to the
    risk budget alone produces proposals the kernel then rejects every time.
    """
    per = abs(delta_per_contract)
    if per <= 0:
        # A delta-neutral structure is unconstrained by this gate, not blocked by
        # it — returning 0 here would silently ban iron condors.
        return 1_000_000
    if remaining_delta_budget <= 0:
        return 0
    n = (remaining_delta_budget / per).to_integral_value(rounding=ROUND_DOWN)
    return max(int(n), 0)


def size_position(
    *,
    width: Decimal,
    net_credit: Decimal,
    risk_budget: Decimal,
    delta_per_contract: Decimal,
    remaining_delta_budget: Decimal,
    size_multiplier: Decimal = Decimal(1),
) -> int:
    """The binding constraint wins.

    Returns 0 when any constraint forbids the trade — the caller must treat that
    as "no proposal", never as "one contract anyway".
    """
    by_risk = max_contracts_for_budget(width, net_credit, risk_budget)
    by_delta = max_contracts_for_dollar_delta(delta_per_contract, remaining_delta_budget)
    n = min(by_risk, by_delta)
    # The regime's size multiplier can only ever *reduce* size (§6.1) — a STRESS
    # session at 0.5x must not become 2x because a config value was mistyped.
    scaled = Decimal(n) * min(size_multiplier, Decimal(1))
    return max(int(scaled.to_integral_value(rounding=ROUND_DOWN)), 0)
