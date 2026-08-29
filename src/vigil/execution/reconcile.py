"""Reconciliation (§2.3 step 2): broker truth vs. what we think we hold.

**The broker is the source of truth, always.** The journal records what we
*intended*; this module records what actually exists. When they disagree the
broker wins, because it is the thing that can lose money.

Why it has to exist at all: §2.6 deliberately leaves a resting GTC exit at the
broker so a position survives the worker dying. That guarantee is only half a
guarantee unless the worker, on restart, can *find its way back* to a position it
has no memory of. So structures are rebuilt from open positions, not read from
process state.

**Grouping is a heuristic, and it is labelled as one.** Alpaca reports individual
option positions; it has no concept of "the iron condor I opened at 10:32". Legs
are regrouped by `(underlying, expiry)`, which is right for everything this agent
builds — one structure per underlying per expiry at a time, enforced by Gate 12's
duplicate check and bounded by Gate 6's two-per-underlying cap. It would be wrong
for a book holding two different spreads on the same expiry, so if that ever
becomes possible the journal's structure ids have to become the grouping key
rather than a fallback.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal

from vigil.data.occ import parse_occ
from vigil.domain import CONTRACT_MULTIPLIER, OpenStructure, PositionLeg, Structure


@dataclass(frozen=True, slots=True)
class BrokerPosition:
    """One option position as the broker reports it.

    A plain dataclass rather than the SDK model so reconciliation is testable with
    no network and no `alpaca-py` object graph — the same reason the kernel takes
    inert data.
    """

    symbol: str
    qty: Decimal          # signed: negative is short
    avg_entry_price: Decimal

    @property
    def is_short(self) -> bool:
        return self.qty < 0

    @property
    def contracts(self) -> int:
        return int(abs(self.qty))


@dataclass(frozen=True, slots=True)
class RestingOrder:
    """An open order at the broker, reduced to what reconciliation asks of it."""

    order_id: str
    symbols: frozenset[str]
    is_closing: bool


def _infer_structure(
    short_puts: list[Decimal], short_calls: list[Decimal],
    long_puts: list[Decimal], long_calls: list[Decimal],
) -> Structure | None:
    """Name the shape from its legs. `None` when it matches nothing we build.

    An unrecognised shape is **not** an error and must not be dropped: it is
    still a live position that needs managing, and a structure the agent cannot
    name is exactly the orphan §2.3 says to adopt. The name is for the journal;
    the management rules key off the short strikes, which are known regardless.
    """
    if short_puts and short_calls:
        return Structure.IRON_CONDOR
    if short_puts and not short_calls:
        return Structure.PUT_CREDIT_SPREAD
    if short_calls and not short_puts:
        return Structure.CALL_CREDIT_SPREAD
    if long_puts and long_calls:
        return Structure.LONG_STRANGLE
    if long_puts or long_calls:
        return Structure.DEBIT_SPREAD
    return None


def group_positions(
    positions: list[BrokerPosition],
    *,
    resting: list[RestingOrder] | None = None,
) -> tuple[OpenStructure, ...]:
    """Rebuild `OpenStructure`s from raw broker positions. Pure.

    `max_loss` is computed from the widest strike span within a right, mirroring
    Gate 1's `_derived_width` — the same rule, so a reconciled structure and a
    proposed one report the same exposure and the concentration gate cannot
    disagree with itself depending on which path produced the number.
    """
    resting = resting or []
    groups: dict[tuple[str, object], list[BrokerPosition]] = {}
    for p in positions:
        try:
            occ = parse_occ(p.symbol)
        except ValueError:
            # Not an option (an equity leg, say). Out of scope: this agent trades
            # only defined-risk option structures, and silently folding a share
            # position into one would misstate its risk entirely.
            continue
        if p.contracts == 0:
            continue
        groups.setdefault((occ.underlying, occ.expiry), []).append(p)

    out: list[OpenStructure] = []
    for (underlying, expiry), legs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        parsed = [(parse_occ(p.symbol), p) for p in legs]
        short_puts = [o.strike for o, p in parsed if o.is_put and p.is_short]
        short_calls = [o.strike for o, p in parsed if not o.is_put and p.is_short]
        long_puts = [o.strike for o, p in parsed if o.is_put and not p.is_short]
        long_calls = [o.strike for o, p in parsed if not o.is_put and not p.is_short]

        widths: list[Decimal] = []
        for side in (short_puts + long_puts, short_calls + long_calls):
            if len(set(side)) >= 2:
                widths.append(max(side) - min(side))
        width = max(widths) if widths else Decimal(0)

        # Contracts: the largest leg size in the group. A partially closed
        # structure has uneven legs, and sizing the close to the *smallest* would
        # strand the remainder — the opposite of what reconciliation is for.
        contracts = max(p.contracts for p in legs)

        # Signed package price per contract: credit received is positive.
        net_credit = sum(
            (p.avg_entry_price * (Decimal(1) if p.is_short else Decimal(-1)) for p in legs),
            Decimal(0),
        )
        max_loss = (
            (width - net_credit) * CONTRACT_MULTIPLIER * contracts
            if net_credit > 0
            else abs(net_credit) * CONTRACT_MULTIPLIER * contracts
        )

        symbols = {p.symbol for p in legs}
        has_target = any(
            r.is_closing and symbols and symbols <= r.symbols for r in resting
        )

        out.append(OpenStructure(
            underlying=underlying,
            expiry=expiry,  # type: ignore[arg-type]
            strikes=tuple(sorted(o.strike for o, _ in parsed)),
            max_loss=max_loss,
            # Delta is not reported on a position and needs a live chain to
            # compute. Left at 0 and refreshed by the sense step; Gate 7 reads
            # the refreshed value, never this placeholder.
            dollar_delta=Decimal(0),
            has_resting_target=has_target,
            structure=_infer_structure(short_puts, short_calls, long_puts, long_calls),
            short_put_strikes=tuple(sorted(short_puts)),
            short_call_strikes=tuple(sorted(short_calls)),
            net_credit=net_credit,
            contracts=contracts,
            legs=tuple(
                PositionLeg(symbol=p.symbol, ratio_qty=1, is_short=p.is_short)
                for p in sorted(legs, key=lambda x: x.symbol)
            ),
        ))
    return tuple(out)


def refresh_deltas(
    structures: tuple[OpenStructure, ...],
    delta_by_symbol: Mapping[str, float | None],
    spot_by_underlying: Mapping[str, Decimal],
) -> tuple[tuple[OpenStructure, ...], tuple[OpenStructure, ...]]:
    """Write a live dollar delta onto each reconciled structure. Pure.

    `group_positions` cannot compute delta — a broker position carries no Greek,
    and this module stays network-free, so the live chain is not in its scope. It
    leaves `dollar_delta` at 0 (D-1); this fills it in once the sense step has a
    chain, using the **same** formula as `TradeProposal.dollar_delta`
    (Σ delta × 100 × spot × signed_qty, then × contracts) so a reconciled book and
    a proposed trade are measured on one ruler — the ruler Gate 7 reads (§5.1).

    Without this, `PortfolioState.net_dollar_delta` summed zeros and the
    portfolio-wide delta gate silently saw a single trade. It only ever mattered on
    the second concurrent position — exactly when Gate 7 is supposed to begin
    refusing.

    A structure is refreshed only when **every** leg has both a delta and a spot. A
    partial refresh is worse than a flagged gap: Gate 7 would sum a half-priced
    structure as if it were whole and *understate* the book in the safe-looking
    direction. Anything that cannot be fully priced keeps its placeholder and is
    returned in the second tuple for the caller to surface — never dropped, never
    silently zeroed as if measured.
    """
    refreshed: list[OpenStructure] = []
    unpriced: list[OpenStructure] = []
    for s in structures:
        spot = spot_by_underlying.get(s.underlying)
        deltas = [delta_by_symbol.get(leg.symbol) for leg in s.legs]
        if spot is None or not s.legs or any(d is None for d in deltas):
            unpriced.append(s)
            refreshed.append(s)  # keep the placeholder; it is flagged, not hidden
            continue
        total = Decimal(0)
        for leg, d in zip(s.legs, deltas, strict=True):
            # signed_ratio: +ratio_qty long, −ratio_qty short — the sign a short leg
            # contributes to portfolio delta, mirroring `Leg.signed_ratio`.
            signed_ratio = -leg.ratio_qty if leg.is_short else leg.ratio_qty
            total += Decimal(str(d)) * CONTRACT_MULTIPLIER * spot * signed_ratio
        refreshed.append(replace(s, dollar_delta=total * s.contracts))
    return tuple(refreshed), tuple(unpriced)


def structures_missing_targets(
    structures: tuple[OpenStructure, ...],
) -> tuple[OpenStructure, ...]:
    """The §2.6 defect list: open structures with no live resting exit.

    Named as a query rather than folded into the sweep because it is the number
    the dashboard shows and the deck quotes. "How many open positions currently
    have no exit order at the broker?" should be answerable in one call, and the
    honest answer on a healthy book is zero.
    """
    return tuple(s for s in structures if not s.has_resting_target)
