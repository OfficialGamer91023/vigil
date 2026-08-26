"""The kernel: run every gate, collect every verdict, approve only on a clean sweep.

Deliberately does **not** short-circuit on the first failure. §5 requires every
verdict to be persisted, passes included, because the first question anyone asks
of a risk system is "did any of this ever actually fire?" — and that has to be
answerable from the record, not from memory. Running all twelve also means one
rejected proposal reports every reason it was rejected, not just the first.
"""

from __future__ import annotations

from vigil.config import RiskConfig, risk_config
from vigil.domain import KernelDecision, PortfolioState, TradeProposal
from vigil.risk.context import KernelContext
from vigil.risk.gates import ALL_GATES


def evaluate(
    proposal: TradeProposal,
    state: PortfolioState,
    context: KernelContext,
    config: RiskConfig | None = None,
) -> KernelDecision:
    """Run all twelve gates. Pure: no network, no LLM, no I/O of any kind."""
    cfg = config or risk_config()

    # An out-of-band halt is not a gate — it is a switch, and it outranks them.
    # Represented as a synthetic gate-0 verdict so the journal shows why nothing
    # was approved rather than showing an unexplained empty cycle.
    if state.halted:
        from vigil.domain import GateVerdict

        halted = GateVerdict(0, "halt_flag", passed=False, reason="HALT flag is set")
        return KernelDecision(approved=False, verdicts=(halted,))

    verdicts = tuple(gate(proposal, state, cfg, context) for gate in ALL_GATES)
    return KernelDecision(approved=all(v.passed for v in verdicts), verdicts=verdicts)


def first_approved(
    proposals: list[TradeProposal],
    state: PortfolioState,
    context: KernelContext,
    config: RiskConfig | None = None,
) -> tuple[TradeProposal | None, list[KernelDecision]]:
    """Evaluate candidates in order and return the first that passes.

    Every decision is returned, not just the winning one — a cycle that approves
    nothing must still be able to explain what it considered and why each failed.
    """
    decisions: list[KernelDecision] = []
    for p in proposals:
        d = evaluate(p, state, context, config)
        decisions.append(d)
        if d.approved:
            return p, decisions
    return None, decisions
