"""Candidate structure builders — verticals, iron condors, debit spreads.

Every candidate returned here is **fully specified**: max loss, max profit, and
per-leg liquidity metrics are all computed, never inferred. The kernel rejects
proposals with missing fields rather than filling them in, so a builder that
cannot populate a field must decline to emit the candidate at all.

**Two prices, and the distinction matters.**

- `net_credit` is the **conservative** credit: sell at the bid, buy at the ask.
  It is what a real fill has to clear, so it drives the economics the kernel
  judges — Gate 9's credit floor and the max-loss that Gate 2 sizes against.
  Mid-based pricing flatters every candidate and paper fills flatter it again
  (§1.2), so the pessimistic number is the honest one to be *judged* on.
- `limit_price` is the **mid**, because §2.5 says the ladder starts at the mid and
  concedes toward the natural. Starting at the conservative price would give away
  the whole spread on the first rung, every time.

Judging on the pessimistic number while *opening* at the optimistic one is the
correct asymmetry: we accept only trades that work even at a bad fill, then try
for a good one.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import date
from decimal import Decimal

from vigil.config import StrategyConfig, strategy_config
from vigil.data.chain import Contract
from vigil.domain import CONTRACT_MULTIPLIER, Leg, Regime, Structure, TradeProposal
from vigil.signals.regime import RegimeVerdict
from vigil.strategy.selection import pick_by_delta
from vigil.strategy.sizing import size_position


def _leg(contract: Contract, *, short: bool) -> Leg | None:
    """Build a Leg, or None if the contract lacks anything the kernel requires."""
    bid, ask, delta = contract.bid, contract.ask, contract.delta
    if bid is None or ask is None or delta is None:
        return None
    oi = getattr(contract.snapshot, "open_interest", None)
    return Leg(
        occ=contract.occ,
        ratio_qty=1,
        is_short=short,
        bid=bid,
        ask=ask,
        delta=delta,
        # The snapshot endpoint does not carry open interest; the caller supplies
        # it from the contracts endpoint. Default to 0 so Gate 8 rejects rather
        # than a missing field quietly passing as "fine".
        open_interest=int(oi) if oi is not None else 0,
    )


def _client_order_id(prefix: str) -> str:
    """Idempotency key (hard rule #9). Random, not derived from the structure —
    two legitimately identical structures on different days must not collide."""
    return f"vigil-{prefix}-{uuid.uuid4().hex[:16]}"


def build_vertical(
    contracts: Sequence[Contract],
    *,
    underlying: str,
    spot: Decimal,
    expiry: date,
    is_put: bool,
    risk_budget: Decimal,
    remaining_delta_budget: Decimal,
    open_interest: dict[str, int] | None = None,
    regime: Regime | None = None,
    size_multiplier: Decimal = Decimal(1),
    config: StrategyConfig | None = None,
) -> TradeProposal | None:
    """A credit spread: sell the ~0.16-delta strike, buy the next one out.

    Put spread for a bullish/neutral read, call spread for bearish — the geometry
    is a mirror image, so one function serves both via `is_put`.
    """
    cfg = config or strategy_config()
    width = cfg.width_for(underlying)

    side = [c for c in contracts if c.occ.is_put == is_put and c.occ.expiry == expiry]
    if not side:
        return None

    short_c = pick_by_delta(side, target=cfg.short_delta_target)
    if short_c is None or short_c.delta is None:
        return None

    # The long leg is *further out of the money*: below the short for puts, above
    # it for calls. Getting this backwards builds a debit spread that looks like a
    # credit spread, which is why the direction is derived rather than assumed.
    long_strike = short_c.occ.strike - width if is_put else short_c.occ.strike + width
    long_c = next((c for c in side if c.occ.strike == long_strike), None)
    if long_c is None:
        return None

    short_leg, long_leg = _leg(short_c, short=True), _leg(long_c, short=False)
    if short_leg is None or long_leg is None:
        return None

    if open_interest:
        short_leg = _with_oi(short_leg, open_interest)
        long_leg = _with_oi(long_leg, open_interest)

    # Sell the bid, pay the ask — the credit a real fill must clear. This is what
    # Gate 9 and the max-loss arithmetic are judged on.
    net_credit = short_leg.bid - long_leg.ask
    if net_credit <= 0:
        return None
    # The ladder's opening rung (§2.5). Always >= net_credit by construction.
    mid_credit = short_leg.mid - long_leg.mid

    delta_per_contract = (
        (Decimal(str(short_leg.delta)) * -1 + Decimal(str(long_leg.delta)))
        * CONTRACT_MULTIPLIER * spot
    )
    n = size_position(
        width=width,
        net_credit=net_credit,
        risk_budget=risk_budget,
        delta_per_contract=delta_per_contract,
        remaining_delta_budget=remaining_delta_budget,
        size_multiplier=size_multiplier,
    )
    if n < 1:
        return None

    structure = Structure.PUT_CREDIT_SPREAD if is_put else Structure.CALL_CREDIT_SPREAD
    return TradeProposal(
        structure=structure,
        underlying=underlying,
        spot=spot,
        expiry=expiry,
        legs=(short_leg, long_leg),
        contracts=n,
        net_credit=net_credit,
        width=width,
        client_order_id=_client_order_id("vert"),
        limit_price=mid_credit,
        regime=regime,
        rationale=(
            f"{structure.value} {short_leg.occ.strike}/{long_leg.occ.strike} "
            f"x{n}, credit {net_credit} ({net_credit / width:.1%} of width), "
            f"short delta {short_leg.delta:+.3f}"
        ),
    )


def _with_oi(leg: Leg, oi: dict[str, int]) -> Leg:
    from dataclasses import replace

    return replace(leg, open_interest=oi.get(leg.symbol, 0))


def build_debit_spread(
    contracts: Sequence[Contract],
    *,
    underlying: str,
    spot: Decimal,
    expiry: date,
    is_put: bool,
    risk_budget: Decimal,
    remaining_delta_budget: Decimal,
    open_interest: dict[str, int] | None = None,
    regime: Regime | None = None,
    size_multiplier: Decimal = Decimal(1),
    config: StrategyConfig | None = None,
) -> TradeProposal | None:
    """The convexity sleeve (§4.5, §4.7): buy movement when it is on sale.

    **This is the mirror image of `build_vertical` in every sign that matters.**
    We *buy* the nearer-the-money strike and *sell* one width further out, so the
    package is a net debit; the debit is the max loss, and `width − debit` is the
    max profit. Getting the leg roles backwards here builds a credit spread while
    believing it is a hedge — which is precisely the failure this builder exists
    to make impossible.

    Why it earns its place: every other structure in the book is short volatility
    with a capped payoff. §4.7 argues a six-session, rank-scored competition has a
    *convex* payoff, so a book made only of variance sales is optimising the wrong
    objective. This is the one component that pays when the market moves.
    """
    cfg = config or strategy_config()
    width = cfg.width_for(underlying)

    side = [c for c in contracts if c.occ.is_put == is_put and c.occ.expiry == expiry]
    if not side:
        return None

    long_c = pick_by_delta(side, target=cfg.convexity_long_delta_target)
    if long_c is None or long_c.delta is None:
        return None

    # The short leg is *further out of the money* — below for puts, above for
    # calls. Same geometry as the credit spread, opposite roles.
    short_strike = long_c.occ.strike - width if is_put else long_c.occ.strike + width
    short_c = next((c for c in side if c.occ.strike == short_strike), None)
    if short_c is None:
        return None

    long_leg, short_leg = _leg(long_c, short=False), _leg(short_c, short=True)
    if long_leg is None or short_leg is None:
        return None

    if open_interest:
        long_leg = _with_oi(long_leg, open_interest)
        short_leg = _with_oi(short_leg, open_interest)

    # Pay the ask, sell the bid — the debit a real fill must clear. Same
    # pessimism as the credit path: judge on the bad fill, open at the good one.
    net_debit = long_leg.ask - short_leg.bid
    if net_debit <= 0:
        return None
    # A debit that meets or exceeds the width cannot profit; the long leg is
    # priced above the whole payoff of the spread.
    if net_debit >= width:
        return None
    mid_debit = long_leg.mid - short_leg.mid
    if mid_debit <= 0:
        return None

    # `net_credit` is negative for a debit package — that sign is what drives
    # `is_credit`, `max_loss_per_contract` and the router's ladder choice.
    net_credit = -net_debit

    delta_per_contract = sum(
        (Decimal(str(leg.delta)) * leg.signed_ratio * CONTRACT_MULTIPLIER * spot
         for leg in (long_leg, short_leg)),
        Decimal(0),
    )
    n = size_position(
        width=width,
        net_credit=net_credit,
        risk_budget=risk_budget,
        delta_per_contract=delta_per_contract,
        remaining_delta_budget=remaining_delta_budget,
        size_multiplier=size_multiplier,
    )
    if n < 1:
        return None

    return TradeProposal(
        structure=Structure.DEBIT_SPREAD,
        underlying=underlying,
        spot=spot,
        expiry=expiry,
        legs=(long_leg, short_leg),
        contracts=n,
        net_credit=net_credit,
        width=width,
        client_order_id=_client_order_id("debt"),
        # `limit_price` is always a POSITIVE package price — see `TradeProposal`.
        # Direction lives in the sign of `net_credit` and in the legs' intents.
        limit_price=mid_debit,
        regime=regime,
        rationale=(
            f"debit_spread {long_leg.occ.strike}/{short_leg.occ.strike} "
            f"{'P' if is_put else 'C'} x{n}, debit {net_debit} "
            f"({net_debit / width:.1%} of width), long delta {long_leg.delta:+.3f}"
        ),
    )


def build_iron_condor(
    contracts: Sequence[Contract],
    *,
    underlying: str,
    spot: Decimal,
    expiry: date,
    risk_budget: Decimal,
    remaining_delta_budget: Decimal,
    open_interest: dict[str, int] | None = None,
    regime: Regime | None = None,
    size_multiplier: Decimal = Decimal(1),
    config: StrategyConfig | None = None,
) -> TradeProposal | None:
    """Both verticals at once — the CHOP workhorse, direction-agnostic.

    Max loss is **one** width minus the total credit, not two: the underlying
    cannot finish below the put spread and above the call spread simultaneously.
    Treating a condor's risk as two independent spreads roughly halves the size
    the account can actually carry, for no gain.
    """
    cfg = config or strategy_config()
    width = cfg.width_for(underlying)

    put_side = build_vertical(
        contracts, underlying=underlying, spot=spot, expiry=expiry, is_put=True,
        risk_budget=risk_budget, remaining_delta_budget=remaining_delta_budget,
        open_interest=open_interest, config=cfg)
    call_side = build_vertical(
        contracts, underlying=underlying, spot=spot, expiry=expiry, is_put=False,
        risk_budget=risk_budget, remaining_delta_budget=remaining_delta_budget,
        open_interest=open_interest, config=cfg)
    if put_side is None or call_side is None:
        return None

    legs = put_side.legs + call_side.legs
    net_credit = put_side.net_credit + call_side.net_credit
    mid_credit = put_side.limit_price + call_side.limit_price

    delta_per_contract = sum(
        (Decimal(str(leg.delta)) * leg.signed_ratio * CONTRACT_MULTIPLIER * spot for leg in legs),
        Decimal(0),
    )
    n = size_position(
        width=width,
        net_credit=net_credit,
        risk_budget=risk_budget,
        delta_per_contract=delta_per_contract,
        remaining_delta_budget=remaining_delta_budget,
        size_multiplier=size_multiplier,
    )
    if n < 1:
        return None

    return TradeProposal(
        structure=Structure.IRON_CONDOR,
        underlying=underlying,
        spot=spot,
        expiry=expiry,
        legs=legs,
        contracts=n,
        net_credit=net_credit,
        width=width,
        client_order_id=_client_order_id("cndr"),
        limit_price=mid_credit,
        regime=regime,
        rationale=(
            f"iron condor x{n}, total credit {net_credit} "
            f"({net_credit / width:.1%} of one width), "
            f"short strikes {put_side.legs[0].occ.strike}P / {call_side.legs[0].occ.strike}C"
        ),
    )


def build_for_regime(
    verdict: RegimeVerdict,
    contracts: Sequence[Contract],
    *,
    underlying: str,
    spot: Decimal,
    expiry: date,
    risk_budget: Decimal,
    remaining_delta_budget: Decimal,
    open_interest: dict[str, int] | None = None,
    config: StrategyConfig | None = None,
) -> TradeProposal | None:
    """Turn a regime verdict into the structure that regime actually asked for.

    **This dispatch is exhaustive on purpose.** The bug it replaces lived in a
    caller that wrote `is_put = structure is PUT_CREDIT_SPREAD` and passed the
    result straight to `build_vertical`: every structure that was not a put credit
    spread — including `DEBIT_SPREAD`, which asks to *buy* volatility — fell
    through to `is_put=False` and quietly built a **call credit spread**. The
    router sold premium on the one session the regime had identified as cheap.

    A `match` with a `case _:` that raises makes that failure mode unreachable: a
    new `Structure` member either gets a branch here or blows up loudly the first
    time it is routed, rather than being silently mapped onto the nearest
    structure that happens to typecheck.

    The convexity sleeve takes a *share* of the risk budget (§4.5), not the whole
    of it — a debit spread is the one structure that can lose its entire premium
    without the underlying doing anything unusual.
    """
    cfg = config or strategy_config()
    common: dict[str, object] = dict(
        underlying=underlying,
        spot=spot,
        expiry=expiry,
        remaining_delta_budget=remaining_delta_budget,
        open_interest=open_interest,
        regime=verdict.regime,
        size_multiplier=verdict.size_multiplier,
        config=cfg,
    )

    match verdict.structure:
        case None:
            # The router stood down. That is a decision, not a gap.
            return None
        case Structure.PUT_CREDIT_SPREAD:
            return build_vertical(contracts, is_put=True, risk_budget=risk_budget, **common)  # type: ignore[arg-type]
        case Structure.CALL_CREDIT_SPREAD:
            return build_vertical(contracts, is_put=False, risk_budget=risk_budget, **common)  # type: ignore[arg-type]
        case Structure.IRON_CONDOR:
            return build_iron_condor(contracts, risk_budget=risk_budget, **common)  # type: ignore[arg-type]
        case Structure.DEBIT_SPREAD:
            # Direction follows the trend read: buy calls into strength, puts into
            # weakness. With no trend available, a debit spread is a coin flip with
            # a premium attached, so decline rather than guess.
            if verdict.trend is None:
                return None
            sleeve = risk_budget * cfg.convexity_risk_share
            return build_debit_spread(
                contracts, is_put=verdict.trend < 0, risk_budget=sleeve, **common  # type: ignore[arg-type]
            )
        case _:  # pragma: no cover - unreachable while Structure is exhaustive
            raise ValueError(
                f"no builder wired for {verdict.structure!r}. Add a branch here "
                f"rather than letting it fall through to another structure."
            )
