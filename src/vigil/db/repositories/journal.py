"""Journal writes. **All database access lives here, never in strategy or risk.**

CLAUDE.md: strategy and risk code takes plain objects, not sessions — the kernel
has to stay pure and testable without a database. So the translation between the
frozen dataclasses in `vigil.domain` and the ORM rows in `vigil.db.models`
happens in exactly one place, and it is this one.
"""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from vigil.data.occ import parse_occ
from vigil.db.models import GateVerdictRow, Proposal, ProposalLeg
from vigil.domain import KernelDecision, TradeProposal


def _finite_or_none(value: Decimal) -> Decimal | None:
    """Map an unbounded max profit to SQL NULL.

    A long strangle's max profit is `Decimal("Infinity")` (see
    `TradeProposal.max_profit_per_contract`), and Postgres `NUMERIC` cannot store
    it. **NULL is the honest encoding**: it says "unbounded", where a large
    stand-in would say "we measured this" and be wrong. Every reader of
    `proposals.max_profit` therefore has to handle NULL, which is the point.
    """
    return value if value.is_finite() else None


async def record_proposal(
    session: AsyncSession,
    proposal: TradeProposal,
    decision: KernelDecision,
    *,
    cycle_id: int,
) -> Proposal:
    """Persist a proposal, its legs, and **every** gate verdict.

    Passes are stored alongside rejections because §5 requires the full record:
    "did any of this ever actually fire?" has to be answerable from the table
    rather than from memory, and a table holding only failures cannot answer it.

    Written in one transaction so a proposal can never exist with a partial set
    of verdicts — the unique constraint on `(proposal_id, gate_no)` guards the
    same invariant from the other side.
    """
    row = Proposal(
        cycle_id=cycle_id,
        structure_type=proposal.structure.value,
        underlying=proposal.underlying,
        expiry=proposal.expiry,
        spot=proposal.spot,
        contracts=proposal.contracts,
        net_credit=proposal.net_credit,
        width=proposal.width,
        max_loss=proposal.max_loss,
        max_profit=_finite_or_none(proposal.max_profit),
        dollar_delta=proposal.dollar_delta,
        client_order_id=proposal.client_order_id,
        limit_price=proposal.limit_price,
        regime=proposal.regime.value if proposal.regime else None,
        rationale=proposal.rationale,
        approved=decision.approved,
    )
    session.add(row)
    await session.flush()          # assigns row.id without ending the transaction

    for leg in proposal.legs:
        occ = parse_occ(leg.symbol)
        session.add(ProposalLeg(
            proposal_id=row.id,
            symbol=leg.symbol,
            ratio_qty=leg.ratio_qty,
            is_short=leg.is_short,
            strike=occ.strike,
            is_put=occ.is_put,
            bid=leg.bid,
            ask=leg.ask,
            delta=Decimal(str(leg.delta)),
            open_interest=leg.open_interest,
        ))

    for v in decision.verdicts:
        session.add(GateVerdictRow(
            proposal_id=row.id,
            gate_no=v.number,
            name=v.name,
            passed=v.passed,
            reason=v.reason or None,
            detail=dict(v.detail) or None,
        ))

    await session.flush()
    return row
