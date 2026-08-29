"""The six session runners — the agent itself (PLAN §2.3).

Every cycle is `sense -> reconcile -> manage -> think -> gate -> act -> log`, and
each runner below is that loop with a different amount of it switched on. The
parts have existed for days; this module is what makes them a trading agent
rather than a library.

**Two rules shape almost every decision in this file.**

1. *Management runs before entry, always.* Protecting capital outranks deploying
   it. Two mechanisms carry this and neither is a convention: `worker.schedule`
   emits MANAGE alongside every ENTRY, and `worker.runner.CYCLE_ORDER` runs the
   sweep first when both fall due in the same minute. `run_entry` then reconciles
   again before sizing anything, so it reasons about the book as the sweep left
   it rather than as it was fifteen minutes ago.

2. *Closes must never be blocked; entries may be.* An exit reduces exposure, so
   nothing — a halt flag, a failed journal write, a gate — may stand in its way.
   An entry adds exposure, so everything may. This is why journal failures are
   fatal to `run_entry` and merely logged in `run_manage`: hard rule #9 makes
   idempotency a *database* constraint, so an entry we cannot journal is an entry
   we cannot deduplicate, while a close we cannot journal is simply a close with
   worse paperwork.

Run one cycle by name (CLAUDE.md):

    python -m vigil.worker.sessions premarket | open | manage | entry | flatten | postclose
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import date
from decimal import Decimal
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession

from vigil.account import verify_account
from vigil.agent import PortfolioManager, Selection, build_manager
from vigil.clock import now_et, today_et
from vigil.config import (
    RiskConfig,
    StrategyConfig,
    risk_config,
    strategy_config,
    universe_config,
)
from vigil.control import FLATTEN_FLAG as _FLATTEN_FLAG
from vigil.control import HALT_FLAG as _HALT_FLAG
from vigil.db.repositories import journal as J
from vigil.db.session import get_session
from vigil.domain import KernelDecision, OpenStructure, PortfolioState, TradeProposal
from vigil.execution.manage import Action, ManagementDecision, sweep
from vigil.execution.pricing import closing_limit, package_mid
from vigil.execution.reconcile import structures_missing_targets
from vigil.execution.router import RiskKernelRejection
from vigil.logging import get_logger
from vigil.risk.context import KernelContext
from vigil.risk.kernel import evaluate
from vigil.strategy.candidates import build_for_regime
from vigil.worker.broker import Broker, portfolio_state
from vigil.worker.schedule import CycleKind
from vigil.worker.sense import MarketView, sense

log = get_logger(__name__)

# Control flags the API writes and the worker reads (§2.2).
# Re-exported from `vigil.control` so the API and the worker cannot drift apart on
# a string. The API must not import this module (it reaches the broker), so the
# names live somewhere neither side owns.
HALT_FLAG = _HALT_FLAG
FLATTEN_FLAG = _FLATTEN_FLAG


@dataclass(slots=True)
class CycleResult:
    """What one cycle did. Goes to the journal, the logs and the dashboard."""

    kind: CycleKind
    cycle_id: int | None = None
    regime: str | None = None
    cold_start: bool = False
    # True when this cycle is running because an operator set the FLATTEN flag,
    # rather than because 15:40 came round. The two want different things: the
    # scheduled stop closes what auto-exercise would otherwise take, while
    # `/api/control/flatten` documents itself as "cancel-all + close-all".
    flatten_requested: bool = False
    notes: list[str] = field(default_factory=list)
    proposals: int = 0
    approved: int = 0
    submitted: int = 0
    closed: int = 0
    warnings: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        self.notes.append(message)

    @property
    def summary(self) -> str:
        return "; ".join(self.notes) if self.notes else "no action"


@dataclass(slots=True)
class RunnerContext:
    """Everything a cycle needs that is not a decision.

    Assembled once per cycle rather than held on a long-lived object: the broker
    is the only source of truth about positions (§2.3), and a context that cached
    them across cycles would quietly become a second, staler one.
    """

    broker: Broker
    db: AsyncSession
    account_row_id: int
    session_row_id: int
    trading_date: date
    risk: RiskConfig
    strategy: StrategyConfig
    universe: tuple[str, ...]
    # The LLM portfolio manager, or None when it is switched off or unkeyed —
    # in which case the entry cycle ranks deterministically and never notices.
    # A snapshot like everything else here, so a mid-session config change is
    # picked up on the next cycle rather than held stale on a long-lived object.
    pm: PortfolioManager | None = None


# --------------------------------------------------------------------------- #
# Startup: the account lock, the account row, today's session row
# --------------------------------------------------------------------------- #

async def open_context(broker: Broker, db: AsyncSession) -> RunnerContext:
    """Verify the lock, then get-or-create the account and session rows.

    **The lock is checked before anything else touches the broker**, and it
    raises rather than warns (hard rule #7). Every cycle calls this, not just the
    premarket one: a worker restarted mid-session against a re-keyed `.env` is
    exactly the case a startup-only check would miss.
    """
    account = await broker.account()
    verify_account(client=broker.client)

    account_row = await J.ensure_account(
        db, alpaca_account_id=account.account_id, starting_equity=account.equity
    )
    session_row = await J.open_session(
        db,
        account_id=account_row.id,
        trading_date=today_et(),
        opening_equity=account.equity,
    )
    scfg = strategy_config()
    return RunnerContext(
        broker=broker,
        db=db,
        account_row_id=account_row.id,
        session_row_id=session_row.id,
        trading_date=today_et(),
        risk=risk_config(),
        strategy=scfg,
        universe=tuple(universe_config().primary),
        # Deliberately not built here. The manager holds an HTTP client and only
        # the entry cycle uses it, so constructing one for every premarket, manage
        # and flatten cycle would churn connection pools for nothing. `run_entry`
        # reaches for the process-level singleton instead; this field stays a seam
        # for a test to inject a fake.
    )


async def _state(ctx: RunnerContext, structures: tuple[OpenStructure, ...]) -> PortfolioState:
    """Assemble the kernel's view, with the high-water mark read from the journal.

    `peak_equity` deliberately does not come from process memory — see
    `broker.portfolio_state` for why that would disarm Gate 4 on every restart.
    """
    account = await ctx.broker.account()
    peak = await J.peak_equity(ctx.db, account_id=ctx.account_row_id)
    halted = await J.is_flag_active(ctx.db, HALT_FLAG)
    return portfolio_state(account, structures, peak_equity=peak, halted=halted)


# --------------------------------------------------------------------------- #
# reconcile — broker truth, written back to the registry
# --------------------------------------------------------------------------- #

async def reconcile(ctx: RunnerContext, result: CycleResult) -> tuple[OpenStructure, ...]:
    """Rebuild the book from the broker and sync the registry to it.

    The `mark_absent_structures_closed` call is what journals a resting GTC
    profit target that filled while the worker was asleep (§2.6). Nothing ever
    *observes* that close — the only evidence is the position's absence here — so
    without this step the registry accumulates phantom rows and Gate 5 starts
    refusing entries against positions that no longer exist.
    """
    structures = await ctx.broker.structures()
    for s in structures:
        await J.record_structure(ctx.db, s)
    vanished = await J.mark_absent_structures_closed(
        ctx.db, structures, reason="absent at reconcile (resting target filled, or closed)"
    )
    if vanished:
        result.note(f"{vanished} structure(s) closed while unobserved")

    defects = structures_missing_targets(structures)
    if defects:
        # §2.6 calls this a defect in those words. Surfaced on every cycle, not
        # only when the sweep gets round to repairing it, because the honest
        # answer on a healthy book is zero and a non-zero count is news.
        result.warnings.append(
            f"§2.6 DEFECT: {len(defects)} open structure(s) with no resting exit: "
            + ", ".join(f"{d.underlying} {d.expiry}" for d in defects)
        )
    return structures


# --------------------------------------------------------------------------- #
# manage — the sweep, and the orders it produces
# --------------------------------------------------------------------------- #

async def _close_structure(
    ctx: RunnerContext, decision: ManagementDecision, result: CycleResult
) -> bool:
    """Price and submit one close. Returns whether an order reached the broker.

    Legs are quoted **by symbol** rather than from a chain window: a structure
    worth closing is usually one the underlying has moved toward, and a fast move
    can carry a short strike out of any window we would have fetched.
    """
    s = decision.structure
    symbols = [leg.symbol for leg in s.legs]
    quotes = await ctx.broker.quotes(symbols)
    missing = [sym for sym in symbols if sym not in quotes]
    if missing:
        # Cannot price it, so cannot submit a limit for it — and hard rule #5
        # forbids the market order that would "solve" this. Loud, because an
        # unclosable position is the worst thing in the book.
        result.warnings.append(
            f"CANNOT PRICE {s.underlying} {s.expiry}: no two-sided quote for "
            f"{', '.join(missing)}. Close not submitted."
        )
        return False

    value = package_mid([(*quotes[leg.symbol], leg.is_short) for leg in s.legs])
    limit = closing_limit(value)
    order = await ctx.broker.submit_close(s, limit, reason=decision.reason)

    await J.record_order(
        ctx.db,
        client_order_id=str(order.client_order_id),
        broker_order_id=str(order.id),
        intent="close",
        limit_price=limit,
        qty=s.contracts,
        status=str(getattr(order.status, "value", order.status)),
    )
    await J.close_structure(
        ctx.db, underlying=s.underlying, expiry=s.expiry, reason=decision.reason
    )
    result.closed += 1
    result.note(f"closed {s.underlying} {s.expiry} @ {limit} — {decision.reason}")
    log.info(
        "structure.close", underlying=s.underlying, expiry=str(s.expiry),
        limit=str(limit), action=decision.action.value, reason=decision.reason,
    )
    return True


async def _replace_target(
    ctx: RunnerContext, decision: ManagementDecision, result: CycleResult
) -> None:
    """Repair a §2.6 defect: an open structure with no resting exit.

    Deliberately reported rather than silently re-submitted. Resting a *new* exit
    requires knowing the credit the structure was opened for, and reconciliation
    derives that from `avg_entry_price`, which the broker reports per leg and
    which drifts from the package price we actually paid. Resting an exit at a
    wrong price is worse than resting none: it looks repaired.

    The honest repair is the entry path's own guarantee — the router rests a
    target on every fill — plus this alarm when one is missing. Wiring an
    automatic re-rest is a Day-4 hardening task, and it belongs there rather than
    here, half-done.
    """
    s = decision.structure
    result.warnings.append(
        f"§2.6 DEFECT unrepaired: {s.underlying} {s.expiry} has no resting exit. "
        f"The 15:40 time stop and the breach rule still cover it."
    )
    log.warning("structure.no_resting_target", underlying=s.underlying, expiry=str(s.expiry))


async def run_manage(ctx: RunnerContext, result: CycleResult) -> tuple[OpenStructure, ...]:
    """The management sweep. **Never blocked by a halt flag.**

    A halt stops the agent taking risk on; it must not stop it taking risk off.
    Gate 3 makes the same distinction — it halts new entries while management
    keeps running — and a halt flag that closed that door too would turn a bad
    day into an unmanaged one.
    """
    structures = await reconcile(ctx, result)
    if not structures:
        result.note("book is flat")
        return structures

    # Sorted list, not a set: `zip`ping two separately-constructed sets would
    # depend on set iteration order matching between them, which is an
    # implementation detail. Getting it wrong would pair SPY's spot with QQQ's
    # structure and silently mis-answer every breach test in the sweep.
    underlyings = sorted({s.underlying for s in structures})
    spots = dict(
        zip(
            underlyings,
            await asyncio.gather(*(ctx.broker.spot(u) for u in underlyings)),
            strict=True,
        )
    )
    decisions = sweep(structures, spots=spots, now=now_et(), config=ctx.strategy)

    for decision in decisions:
        if decision.closes:
            await _close_structure(ctx, decision, result)
        elif decision.action is Action.REPLACE_TARGET:
            await _replace_target(ctx, decision, result)

    held = sum(1 for d in decisions if d.action is Action.HOLD)
    result.note(f"swept {len(decisions)} structure(s): {result.closed} closed, {held} held")
    return structures


# --------------------------------------------------------------------------- #
# think + gate + act — the entry path
# --------------------------------------------------------------------------- #

async def _candidates(
    ctx: RunnerContext, view: MarketView, state: PortfolioState
) -> list[TradeProposal]:
    """Every buildable candidate for one underlying, across live expiries.

    Budgets are computed once from the *current* state and passed to every
    builder, so a proposal is sized against the book as it actually is. They are
    an input to sizing, not a substitute for the gates: Gate 2 and Gate 7 re-check
    both, and the kernel's answer is the binding one.
    """
    risk_budget = ctx.risk.max_risk_per_trade_pct * state.equity
    delta_budget = (
        ctx.risk.max_dollar_delta_pct * state.equity - abs(state.net_dollar_delta)
    )
    out: list[TradeProposal] = []
    for expiry in sorted({c.occ.expiry for c in view.chain}):
        candidate = build_for_regime(
            view.verdict,
            list(view.chain),
            underlying=view.underlying,
            spot=view.spot,
            expiry=expiry,
            risk_budget=risk_budget,
            remaining_delta_budget=delta_budget,
            open_interest=view.open_interest,
            config=ctx.strategy,
        )
        if candidate is not None:
            out.append(candidate)
    return out


def _rank_key(proposal: TradeProposal) -> tuple[Decimal, Decimal]:
    """The ordering both the ranker and the fallback share, in one place.

    Credit as a share of width first — the one number Gate 9 already treats as the
    quality of a premium sale, so *most compensation per dollar of risk* — then
    lower max loss to break ties: same edge, smaller bet. Kept as a named function
    so `_rank` and `_best` cannot drift into two different strategies.
    """
    return (-proposal.credit_pct_of_width, proposal.max_loss)


def _rank(scored: list[tuple[TradeProposal, KernelDecision]]) -> list[TradeProposal]:
    """Order approved candidates best-first. **The deterministic fallback (§6.3).**

    This is the selection the LLM portfolio manager makes from the same
    pre-validated set when it is switched off — and the path taken whenever the
    model is slow, rate limited, or returns something the schema rejects. It
    exists first and stays forever: §6.3 requires every LLM call to have a
    deterministic fallback, and a fallback written after the model is a fallback
    nobody has run.
    """
    return [p for p, _ in sorted(scored, key=lambda pair: _rank_key(pair[0]))]


def _best(candidates: list[TradeProposal]) -> TradeProposal:
    """The single deterministic winner — the callback the model falls back to.

    Same key as `_rank`, so "the model chose" and "the model fell back" resolve to
    byte-identical trades whenever the model would have agreed with the ranking.
    """
    return min(candidates, key=_rank_key)


async def run_entry(ctx: RunnerContext, result: CycleResult) -> None:
    """Sense, build, gate, submit **at most one** structure.

    One per cycle is deliberate. Gate 5 caps the book at six and Gate 6 at two per
    underlying, but those are ceilings on what may exist, not permission to reach
    them in a single cycle: entering three structures against one regime read is
    one opinion sized three times, and the portfolio gates cannot see that because
    each proposal is legal on its own.
    """
    if await J.is_flag_active(ctx.db, HALT_FLAG):
        result.note("HALT flag active — no new entries (management still runs)")
        return

    structures = await reconcile(ctx, result)
    state = await _state(ctx, structures)
    await J.record_equity(ctx.db, account_id=ctx.account_row_id, state=state)

    context = KernelContext(now=now_et())
    scored: list[tuple[TradeProposal, KernelDecision]] = []
    # client_order_id -> journalled proposal row id, so the LLM memo can be
    # attached to the exact proposal the model chose. The id is unique and already
    # the row's natural key, so it is the honest join back to the record.
    id_by_coid: dict[str, int] = {}
    cold_start = False

    for underlying in ctx.universe:
        view = await sense(ctx.broker, underlying, max_dte=ctx.strategy.max_dte)
        if view is None:
            result.warnings.append(f"no tradeable chain for {underlying}")
            continue
        result.warnings.extend(view.warnings)
        result.regime = view.verdict.regime.value
        result.cold_start = result.cold_start or view.verdict.cold_start
        cold_start = cold_start or view.verdict.cold_start

        await J.record_market_snapshot(
            ctx.db,
            cycle_id=result.cycle_id or 0,
            underlying=underlying,
            spot=view.spot,
            iv_atm=Decimal(str(view.snapshot.iv_atm)),
            rv_annual=_opt_dec(view.snapshot.rv_annual),
            vrp_pct=_opt_dec(view.verdict.vrp_pct),
            iv_pct=_opt_dec(view.verdict.iv_pct),
            trend=_opt_dec(view.verdict.trend),
        )

        if view.verdict.structure is None:
            result.note(f"{underlying}: {view.verdict.regime.value} — stand down")
            continue

        # A context that knows which symbols the chain actually listed, so Gate 12
        # can check existence instead of skipping the check.
        symbols = frozenset(c.occ.raw for c in view.chain)
        ctx_for = KernelContext(now=context.now, available_symbols=symbols)

        for candidate in await _candidates(ctx, view, state):
            decision = evaluate(candidate, state, ctx_for, ctx.risk)
            # **Every verdict is persisted, passes included** (§5). A table of
            # only rejections cannot answer "did any of this ever fire?".
            prow = await J.record_proposal(
                ctx.db, candidate, decision, cycle_id=result.cycle_id or 0
            )
            result.proposals += 1
            if decision.approved:
                result.approved += 1
                scored.append((candidate, decision))
                id_by_coid[candidate.client_order_id] = prow.id
            else:
                log.info(
                    "proposal.rejected", underlying=underlying,
                    structure=candidate.structure.value, reason=decision.summary,
                )

    if not scored:
        result.note(f"{result.proposals} candidate(s), none approved")
        return

    selection = await _choose(ctx, scored, state, result.regime or "mixed", cold_start)
    # The memo is journalled whatever the model decided — a stand-down and a
    # fallback are both facts about the day worth keeping (§6.3 makes the fallback
    # *rate* a headline number, which a table of only successes could not compute).
    await _record_memo(ctx, selection, id_by_coid, cycle_id=result.cycle_id or 0)

    if selection.stood_down:
        result.note(f"PM stood down: {selection.memo or 'no memo'}")
        return

    await _submit(ctx, selection.proposal, state, result)


# --------------------------------------------------------------------------- #
# The portfolio manager: the model selects, or the ranker does (§6)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _portfolio_manager() -> PortfolioManager | None:
    """The process-wide manager, or None on the deterministic path.

    Cached because the client should live for the life of the worker, not be
    rebuilt each entry cycle — and because the enable/key decision does not change
    under a running process. A test never reaches this: it either injects a fake
    on the context or runs with the key stripped, in which case the first call
    here caches None and every entry cycle takes the deterministic path.
    """
    return build_manager()


async def _choose(
    ctx: RunnerContext,
    scored: list[tuple[TradeProposal, KernelDecision]],
    state: PortfolioState,
    regime: str,
    cold_start: bool,
) -> Selection:
    """Pick one proposal from the approved menu — via the model, or the ranker.

    The menu handed to the model is the approved set only, and `_best` is the
    fallback it defers to. Whatever comes back is still passed to `_submit`, which
    re-runs the whole kernel: the model narrows an already-safe set and cannot
    widen it, which is the structural form of "the LLM proposes, the kernel
    disposes" (§6).
    """
    candidates = [p for p, _ in scored]
    manager = ctx.pm or _portfolio_manager()
    if manager is None:
        # No model: the deterministic winner, wrapped so the memo path is uniform.
        return Selection(
            proposal=_best(candidates), fell_back=True,
            model="deterministic", effort="none",
            memo="deterministic ranker (model disabled)",
        )

    return await manager.select(
        candidates, state,
        regime=regime, cold_start=cold_start,
        risk=ctx.risk, strategy=ctx.strategy,
        fallback=_best,
    )


async def _record_memo(
    ctx: RunnerContext,
    selection: Selection,
    id_by_coid: dict[str, int],
    *,
    cycle_id: int,
) -> None:
    """Persist the model's cost and reasoning to `llm_memos` (§6.2).

    Linked to the chosen proposal so `GET /api/cycles/{id}` can show the memo
    beside the verdicts it acted on. A memo write must never sink an entry, so a
    failure here is logged and swallowed — the trade is the product, the memo is
    the paperwork, and hard rule #6's "runs with the journal degraded" applies to
    the observability tables first.
    """
    try:
        await J.record_llm_memo(
            ctx.db,
            cycle_id=cycle_id,
            proposal_id=id_by_coid.get(selection.proposal.client_order_id),
            model=selection.model,
            effort=selection.effort,
            latency_ms=selection.latency_ms,
            input_tokens=selection.input_tokens,
            cached_tokens=selection.cached_tokens,
            cache_write_tokens=selection.cache_write_tokens,
            output_tokens=selection.output_tokens,
            reasoning_tokens=selection.reasoning_tokens,
            memo=selection.memo or None,
            fell_back=selection.fell_back,
        )
    except Exception as exc:  # noqa: BLE001 — observability must not sink a trade
        log.warning("memo.write_failed", error=str(exc)[:200])


async def _submit(
    ctx: RunnerContext, proposal: TradeProposal, state: PortfolioState, result: CycleResult
) -> None:
    """Hand the winner to the single submit path and journal what came back.

    `submit_entry` re-runs the kernel — the evaluation in `run_entry` is for
    *selection*, this one is the one that binds. A rejection here is therefore not
    a redundancy failure but a real one (the book moved between ranking and
    submitting), and it is journalled rather than retried.
    """
    context = KernelContext(now=now_et())
    try:
        outcome = await ctx.broker.submit_entry(
            proposal, state, context, risk=ctx.risk, strategy=ctx.strategy
        )
    except RiskKernelRejection as exc:
        result.note(f"kernel refused at submit: {exc.decision.summary}")
        log.warning("entry.rejected_at_submit", reason=exc.decision.summary)
        return

    if outcome.entry_order is not None:
        await J.record_order(
            ctx.db,
            client_order_id=str(outcome.entry_order.client_order_id),
            broker_order_id=str(outcome.entry_order.id),
            intent="open",
            limit_price=proposal.limit_price,
            qty=proposal.contracts,
            status=str(getattr(outcome.entry_order.status, "value", outcome.entry_order.status)),
            rung=outcome.rungs_used,
        )
    if outcome.target_order is not None:
        await J.record_order(
            ctx.db,
            client_order_id=str(outcome.target_order.client_order_id),
            broker_order_id=str(outcome.target_order.id),
            intent="target",
            limit_price=Decimal(str(outcome.target_order.limit_price or 0)),
            qty=outcome.filled_contracts or proposal.contracts,
            status=str(getattr(outcome.target_order.status, "value", outcome.target_order.status)),
        )

    if not outcome.filled:
        result.note(f"no fill after {outcome.rungs_used} rung(s) — {outcome.note}")
        return

    result.submitted += 1
    result.note(
        f"FILLED {proposal.structure.value} {proposal.underlying} {proposal.expiry} "
        f"x{outcome.filled_contracts} @ rung {outcome.rungs_used}"
        + (f" — {outcome.note}" if outcome.note else "")
    )
    if outcome.partial:
        result.warnings.append(outcome.note)
    log.info(
        "entry.filled", underlying=proposal.underlying, structure=proposal.structure.value,
        contracts=outcome.filled_contracts, rung=outcome.rungs_used,
        max_loss=str(proposal.max_loss), partial=outcome.partial,
    )


# --------------------------------------------------------------------------- #
# The six cycles
# --------------------------------------------------------------------------- #

async def premarket(ctx: RunnerContext, result: CycleResult) -> None:
    """08:45 — verify the lock, adopt whatever the broker says we hold.

    The account assertion has already run in `open_context`. What this cycle adds
    is the overnight reconciliation: anything held into today is found, registered
    and checked for a resting exit before the market can move against it.
    """
    structures = await reconcile(ctx, result)
    state = await _state(ctx, structures)
    await J.record_equity(ctx.db, account_id=ctx.account_row_id, state=state)
    result.note(
        f"equity {state.equity}, {len(structures)} structure(s) carried, "
        f"open risk {state.open_risk}"
    )
    if state.halted:
        result.warnings.append("HALT flag is active — the session will not enter")


async def market_open(ctx: RunnerContext, result: CycleResult) -> None:
    """09:45 — the first read of the day. No entries: Gate 11 forbids them anyway.

    Deliberately runs a full sense pass without building anything. The regime the
    router sees at the open is the day's context, and journalling it *before* the
    first entry cycle means a later decision can be compared against what was
    known at the start rather than only against what was known when it fired.
    """
    structures = await reconcile(ctx, result)
    state = await _state(ctx, structures)
    await J.record_equity(ctx.db, account_id=ctx.account_row_id, state=state)

    for underlying in ctx.universe:
        view = await sense(ctx.broker, underlying, max_dte=ctx.strategy.max_dte)
        if view is None:
            result.warnings.append(f"no tradeable chain for {underlying} at the open")
            continue
        result.warnings.extend(view.warnings)
        result.regime = view.verdict.regime.value
        result.cold_start = result.cold_start or view.verdict.cold_start
        await J.record_market_snapshot(
            ctx.db,
            cycle_id=result.cycle_id or 0,
            underlying=underlying,
            spot=view.spot,
            iv_atm=Decimal(str(view.snapshot.iv_atm)),
            rv_annual=_opt_dec(view.snapshot.rv_annual),
            vrp_pct=_opt_dec(view.verdict.vrp_pct),
            iv_pct=_opt_dec(view.verdict.iv_pct),
            trend=_opt_dec(view.verdict.trend),
        )
        result.note(
            f"{underlying} {view.spot}: {view.verdict.regime.value} — {view.verdict.reason}"
        )


async def flatten(ctx: RunnerContext, result: CycleResult) -> None:
    """15:40 — the hard time stop. **Unconditional, and gated by nothing.**

    Auto-exercise makes anything ITM by $0.01 at expiry a position we did not
    choose (§1.1), so this is the one cycle with no opinion to form. Working
    orders are cancelled first: an entry ticket that fills at 15:41 would create
    a position after the flatten had already decided the book should be empty.

    `submit_close` is deliberately not kernel-gated — routing this through Gate 11
    would produce a flatten that refuses to flatten in the last twenty minutes,
    which is the only time it ever runs.

    **Two callers, two scopes.** The scheduled 15:40 run closes only what expires
    today, because auto-exercise is the thing it exists to prevent and a later
    expiry is a position we still want. An operator hitting
    `/api/control/flatten` is asking for something else — that route documents
    itself as "cancel-all + close-all" — so a requested flatten takes the whole
    book. Conflating the two would either leave an operator's flatten half done
    or make the daily stop liquidate positions it had no reason to touch.
    """
    await ctx.broker.cancel_all_orders()
    result.note("cancelled all working orders")

    structures = await reconcile(ctx, result)

    if result.flatten_requested:
        # The flag is only ever cleared here, and only on a cycle that *arrived*
        # to an empty book. `_close_structure` submits a limit order, not a fill
        # (hard rule #5 forbids the market order that would make it one), so a
        # cycle that has just sent closes cannot honestly claim the book is flat.
        # Leaving the flag set means the next cycle re-runs this one, sees the
        # fills, and clears it then — which is also the behaviour that keeps a
        # flatten whose closes never filled from silently resuming trading.
        if not structures:
            await J.set_flag(
                ctx.db, FLATTEN_FLAG, active=False, set_by="worker",
                reason="book confirmed flat; requested flatten complete",
            )
            result.note("book confirmed flat — FLATTEN cleared, entries resume next cycle")
            return
        targets = list(structures)
        why = "operator flatten requested via /api/control/flatten"
    else:
        targets = [s for s in structures if s.expiry <= ctx.trading_date]
        if not targets:
            result.note("nothing expiring today; book left as is")
            return
        why = ""

    for s in targets:
        reason = why or (
            f"15:40 hard flatten — expires {s.expiry}, auto-exercise is not a strategy"
        )
        await _close_structure(ctx, ManagementDecision(s, Action.CLOSE_TIME_STOP, reason), result)

    if result.closed < len(targets):
        result.warnings.append(
            f"FLATTEN INCOMPLETE: {len(targets) - result.closed} of {len(targets)} "
            f"structure(s) still open. Run `make flatten`."
        )
    elif result.flatten_requested:
        result.note(
            f"closes submitted for {len(targets)} structure(s); FLATTEN stays set until "
            f"a later cycle confirms the book is empty"
        )


async def postclose(ctx: RunnerContext, result: CycleResult) -> None:
    """16:15 — close the books. The day's last equity reading is the scored one."""
    structures = await ctx.broker.structures()
    state = await _state(ctx, structures)
    await J.record_equity(ctx.db, account_id=ctx.account_row_id, state=state)
    await J.close_session(
        ctx.db, session_id=ctx.session_row_id, closing_equity=state.equity
    )
    result.note(
        f"closing equity {state.equity}, day P&L {state.day_pnl}, "
        f"{len(structures)} structure(s) held overnight"
    )
    if structures:
        result.warnings.append(
            f"{len(structures)} position(s) held overnight — expected only for "
            f"expiries beyond today"
        )


def _opt_dec(value: float | None) -> Decimal | None:
    return None if value is None else Decimal(str(value))


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

async def manage(ctx: RunnerContext, result: CycleResult) -> None:
    """The MANAGE cycle. Thin, because `run_manage` also serves the ENTRY cycle.

    Entry calls `run_manage` directly and uses its return value (the reconciled
    book) to size against; the scheduled cycle only needs it to have happened.
    """
    await run_manage(ctx, result)


# Every cycle has the same signature, so dispatch is a plain lookup rather than a
# `match` with six branches that each call one function.
Cycle = Callable[[RunnerContext, CycleResult], Awaitable[None]]

_CYCLES: dict[CycleKind, Cycle] = {
    CycleKind.PREMARKET: premarket,
    CycleKind.OPEN: market_open,
    CycleKind.MANAGE: manage,
    CycleKind.ENTRY: run_entry,
    CycleKind.FLATTEN: flatten,
    CycleKind.POSTCLOSE: postclose,
}


async def run_cycle(kind: CycleKind, *, broker: Broker | None = None) -> CycleResult:
    """Run one cycle end to end, in **two** transactions.

    The split is the whole design of this function, and it was originally wrong.

    *Transaction one* writes identity: the account row, today's session row, and
    the cycle row — then commits. *Transaction two* does the work and commits
    only if the work succeeded.

    Two, not one, because `finished_at IS NULL` is meant to be a live query for
    "what died mid-cycle?" — and with a single transaction a crashing cycle rolls
    its own row away, leaving no evidence it ever ran. The failure that most needs
    a record was the one guaranteed not to have one. Committing the row before the
    work is what makes the query answerable.

    For the same reason `finish_cycle` runs **only on success**. Stamping
    `finished_at` in a `finally` would mark a crashed cycle as completed, which is
    the same erasure by a different route.

    The **FLATTEN control flag is checked here rather than inside a cycle**, so it
    can pre-empt any of them. Someone hitting `/api/control/flatten` at 11:02 is
    not asking for the 11:02 manage sweep to finish first.
    """
    b = broker or Broker()
    result = CycleResult(kind=kind)

    # ---- transaction one: identity, committed before any work happens ------ #
    async with get_session() as db:
        ctx = await open_context(b, db)
        # Read once and carry it: `flatten()` needs to know whether it is serving
        # an operator or the clock, and it is also the only cycle allowed to clear
        # the flag again.
        result.flatten_requested = await J.is_flag_active(db, FLATTEN_FLAG)
        if result.flatten_requested and kind is not CycleKind.FLATTEN:
            result.note("FLATTEN flag active — running the flatten cycle instead")
            kind, result.kind = CycleKind.FLATTEN, CycleKind.FLATTEN
        cycle_row = await J.start_cycle(db, session_id=ctx.session_row_id, kind=kind.value)
        result.cycle_id = cycle_row.id

    bound = log.bind(session_id=ctx.session_row_id, cycle_id=result.cycle_id, cycle=kind.value)
    bound.info("cycle.start")

    # ---- transaction two: the work ---------------------------------------- #
    async with get_session() as db:
        # `replace` rather than mutation: RunnerContext is a per-cycle snapshot,
        # and rebinding it to the new session keeps that reading true.
        ctx = replace(ctx, db=db)
        try:
            await _CYCLES[kind](ctx, result)
        except Exception as exc:
            # The cycle row survives in transaction one with `finished_at` NULL,
            # which is the signal. The exception is re-raised so the runner's
            # supervision sees it — a swallowed exception here is how an agent
            # looks alive while doing nothing, which §14 names as the worst
            # possible demo.
            result.warnings.append(f"CYCLE FAILED: {type(exc).__name__}: {exc}")
            bound.error("cycle.failed", error=str(exc), error_type=type(exc).__name__)
            raise
        await J.finish_cycle(
            db,
            cycle_id=result.cycle_id,
            regime=result.regime,
            cold_start=result.cold_start,
            notes=result.summary,
        )

    for warning in result.warnings:
        bound.warning("cycle.warning", detail=warning)
    bound.info(
        "cycle.done", proposals=result.proposals, approved=result.approved,
        submitted=result.submitted, closed=result.closed, summary=result.summary,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """`python -m vigil.worker.sessions <cycle>` — one cycle, by name."""
    import sys

    args = argv if argv is not None else sys.argv[1:]
    names = [k.value for k in CycleKind]
    if len(args) != 1 or args[0] not in names:
        print(f"usage: python -m vigil.worker.sessions [{' | '.join(names)}]")
        return 2

    result = asyncio.run(run_cycle(CycleKind(args[0])))
    print(f"\n{result.kind.value}: {result.summary}")
    for warning in result.warnings:
        print(f"  ! {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
