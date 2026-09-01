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
from vigil.clock_guard import verify_clock
from vigil.config import (
    LadderConfig,
    RiskConfig,
    StrategyConfig,
    ladder_config,
    risk_config,
    strategy_config,
    universe_config,
)
from vigil.control import FLATTEN_FLAG as _FLATTEN_FLAG
from vigil.control import HALT_FLAG as _HALT_FLAG
from vigil.db.repositories import journal as J
from vigil.db.session import get_session
from vigil.domain import KernelDecision, OpenStructure, PortfolioState, TradeProposal
from vigil.execution.manage import (
    Action,
    ManagementDecision,
    resting_target_price,
    sweep,
)
from vigil.execution.pricing import closing_limit, package_mid
from vigil.execution.reconcile import refresh_deltas, structures_missing_targets
from vigil.execution.router import RiskKernelRejection
from vigil.journal.emit import emit_session_journal
from vigil.logging import get_logger
from vigil.risk.context import KernelContext
from vigil.risk.kernel import evaluate
from vigil.strategy.candidates import build_for_regime
from vigil.strategy.ladder import effective_core_risk_pct, resolve_rung
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

# Cancelling a structure's resting target only *requests* the cancel; the broker
# moves the order through `pending_cancel` before `canceled`, and the leg quantity
# stays reserved (`held_for_orders`) until it lands there. So `_close_structure`
# polls the order to a terminal state before submitting its close. The poll is
# bounded on purpose: it runs inside the manage sweep and inside the 15:40 flatten,
# the two cycles that must always finish, so a target that never settles has to
# defer the close rather than hang the loop. Worst case here is
# `_CANCEL_SETTLE_ATTEMPTS - 1` pauses ≈ 1.25s per structure — negligible against a
# 15-minute sweep, tolerable inside the flatten's 20-minute window.
_CANCEL_SETTLE_ATTEMPTS = 6
_CANCEL_SETTLE_PAUSE = 0.25  # seconds between polls

# A resting target releases its reserved legs the moment it leaves the working set,
# whichever way that happens — an operator cancel, an end-of-day expiry, a broker
# rejection. Any of these frees the quantity the close needs; `filled` is handled
# separately because it means the structure is already gone.
_TARGET_FREED = frozenset({"canceled", "expired", "rejected"})


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
    ladder: LadderConfig
    universe: tuple[str, ...]
    # The LLM portfolio manager, or None when it is switched off or unkeyed —
    # in which case the entry cycle ranks deterministically and never notices.
    # A snapshot like everything else here, so a mid-session config change is
    # picked up on the next cycle rather than held stale on a long-lived object.
    pm: PortfolioManager | None = None


# --------------------------------------------------------------------------- #
# Startup: the account lock, the clock guard, the account row, today's session row
# --------------------------------------------------------------------------- #

async def open_context(broker: Broker, db: AsyncSession) -> RunnerContext:
    """Verify the lock and the clock, then get-or-create the account and session rows.

    **The lock is checked before anything else touches the broker**, and it
    raises rather than warns (hard rule #7). Every cycle calls this, not just the
    premarket one: a worker restarted mid-session against a re-keyed `.env` is
    exactly the case a startup-only check would miss.

    The clock guard rides alongside it for the same reason and with the same
    fail-closed contract: every time gate keys off the local clock, so a host whose
    time has drifted mislabels expiries and can skip the 0DTE flatten. Checking it
    each cycle, not once at boot, catches an NTP step or a suspend/resume that moves
    the clock mid-session. Both blocking reads are called synchronously here, as
    `verify_account` already is — this is startup, run once per cycle, not the hot path.
    """
    account = await broker.account()
    verify_account(client=broker.client)
    verify_clock(client=broker.client)

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
        ladder=ladder_config(),
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


async def _record_equity_carried(
    ctx: RunnerContext, structures: tuple[OpenStructure, ...]
) -> PortfolioState:
    """Record a fresh equity mark for a cycle that read no chain, and return the state.

    `manage` and `postclose` reconcile the book but never sense a chain, so every
    structure carries the reconcile placeholder `dollar_delta=0`. Recording that
    would zero the desk's portfolio delta between entries; instead the last
    refreshed value is carried forward (see `record_equity`). Recording *equity*
    here at all is what keeps the dashboard's equity and day-P&L current through the
    afternoon, rather than frozen at the last entry cycle.
    """
    state = await _state(ctx, structures)
    carried = (
        await J.last_net_dollar_delta(ctx.db, account_id=ctx.account_row_id)
        if structures else None
    )
    await J.record_equity(
        ctx.db, account_id=ctx.account_row_id, state=state, net_dollar_delta=carried
    )
    return state


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

async def _await_target_settled(ctx: RunnerContext, order_id: str) -> str:
    """Poll a just-cancelled resting target to a terminal state. Bounded.

    Returns one of:
      * ``"freed"``    — the order left the working set (``canceled``/``expired``/
        ``rejected``); its legs are released, so the close may be submitted.
      * ``"filled"``   — the target filled first, which already closed the
        structure; the caller must **not** submit a competing close.
      * ``"deferred"`` — it never reached a terminal state within the budget; the
        close is skipped this cycle and retried next sweep.

    The first read happens with no pause, so a broker that cancels synchronously
    (and the test stub that models it) settles on attempt zero. A pause is taken
    only *between* reads, never after the last — an unsettled cancel costs at most
    ``_CANCEL_SETTLE_ATTEMPTS - 1`` pauses, then defers.
    """
    for attempt in range(_CANCEL_SETTLE_ATTEMPTS):
        status = await ctx.broker.order_status(order_id)
        if status == "filled":
            return "filled"
        if status in _TARGET_FREED:
            return "freed"
        # pending_cancel / pending_new / new / accepted / partially_filled: the
        # cancel is still in flight (or the target is mid-fill), so the legs may
        # still be reserved. Wait a beat and re-read rather than racing a submit
        # against a held leg — the failure this whole poll exists to prevent.
        if attempt + 1 < _CANCEL_SETTLE_ATTEMPTS:
            await asyncio.sleep(_CANCEL_SETTLE_PAUSE)
    return "deferred"


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

    # The resting GTC profit target this structure carries (§2.6) reserves its leg
    # quantity at the broker (`held_for_orders`), so a close submitted against the
    # same legs is rejected for insufficient available quantity — the failure that
    # was tearing down the whole manage cycle every sweep. Cancel the structure's
    # own resting exit first to free the contracts, scoped by leg symbols so a
    # sibling structure's target is untouched. The 15:40 flatten does the same thing
    # wholesale (`cancel_all_orders` → close); this is the per-structure version.
    leg_symbols = set(symbols)
    try:
        for r in await ctx.broker.resting_orders():
            if not (r.is_closing and leg_symbols <= r.symbols):
                continue
            await ctx.broker.cancel_order(r.order_id)
            # A cancel is only a *request*: the target sits in `pending_cancel`
            # with its legs still reserved until the broker settles it, so submitting
            # the close now would race that and be rejected for insufficient quantity
            # — the exact failure the cancel-first ordering was meant to fix, still
            # live because the ordering alone does not wait. Poll to a terminal state.
            settled = await _await_target_settled(ctx, r.order_id)
            if settled == "filled":
                # The resting profit target filled while we were cancelling it, which
                # already closed this structure (§2.6). A close submitted now would
                # trade against a position that no longer exists — at best a rejected
                # order, at worst a brand-new, possibly naked, structure (hard rule
                # #3). So submit nothing: record the close the target performed and
                # stop. The next reconcile reads the flat book back (§2.3).
                await J.close_structure(
                    ctx.db, underlying=s.underlying, expiry=s.expiry,
                    reason=f"resting target filled ahead of {decision.reason}",
                )
                result.closed += 1
                result.note(
                    f"{s.underlying} {s.expiry} closed by its resting target "
                    f"{r.order_id[:8]} before the manage close could submit"
                )
                log.info(
                    "structure.target_filled_during_close",
                    underlying=s.underlying, expiry=str(s.expiry), order_id=r.order_id,
                )
                return True
            if settled == "deferred":
                # The cancel never reached a terminal state within the poll budget,
                # so the legs may still be reserved and a close would race it. Defer
                # rather than hang the sweep (or the flatten): loud warning, no submit,
                # safe retry next cycle — the cancel is idempotent. The position stays
                # covered by the 15:40 flatten and the breach rule meanwhile.
                result.warnings.append(
                    f"CLOSE DEFERRED {s.underlying} {s.expiry}: resting target "
                    f"{r.order_id[:8]} did not settle after cancel; not closed this "
                    f"cycle. The 15:40 flatten and breach rule still cover it."
                )
                log.warning(
                    "structure.close_deferred",
                    underlying=s.underlying, expiry=str(s.expiry), order_id=r.order_id,
                )
                return False
            # settled == "freed": the reservation is released; safe to close.
            result.note(
                f"cancelled resting target {r.order_id[:8]} on "
                f"{s.underlying} {s.expiry} before close"
            )
        order = await ctx.broker.submit_close(s, limit, reason=decision.reason)
    except Exception as exc:  # noqa: BLE001 — an unclosable position is the worst thing in the book
        # Degrade to a loud warning rather than let the error abort the sweep (the
        # old behaviour: the raw APIError escaped the cycle, `finished_at` never got
        # written, and every structure queued after this one went unmanaged). The
        # position stays covered by the 15:40 flatten and the breach rule, and a
        # retry next sweep is safe — the resting-target cancel is idempotent and the
        # close is a fresh order.
        result.warnings.append(
            f"CLOSE FAILED {s.underlying} {s.expiry}: {str(exc)[:120]}. Not closed "
            f"this cycle; the 15:40 flatten and breach rule still cover it."
        )
        log.warning(
            "structure.close_failed",
            underlying=s.underlying, expiry=str(s.expiry), error=str(exc)[:200],
        )
        return False

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
    """Repair a §2.6 defect: re-rest a resting GTC profit target that went missing.

    The router rests a target on every fill, so a healthy book never reaches here.
    A structure without one has usually had its target cancelled out from under it
    — a `cancel_all_orders` during a flatten that was then interrupted before the
    close filled, or a manual cancel — and the position is now covered only by the
    15:40 time stop and the breach rule.

    The earlier version only *reported* this, out of a caution that was right in
    spirit: re-resting needs the credit the structure was opened for, and a wrong
    price "looks repaired". `resting_target_price` answers that carefully — it
    reproduces the entry path's own arithmetic and returns `None` when the opening
    credit is genuinely unknown, so an adopted position we never priced still only
    raises the alarm. When it *can* price the target we place a real GTC exit and
    let the next reconcile read it back from the broker (§2.3: broker is truth).
    """
    s = decision.structure
    target = resting_target_price(s, ctx.strategy)
    if target is None:
        # No trustworthy opening credit — keep the alarm rather than guess a price.
        result.warnings.append(
            f"§2.6 DEFECT unrepaired: {s.underlying} {s.expiry} has no resting exit "
            f"and no known opening credit to price one from. The 15:40 time stop and "
            f"the breach rule still cover it."
        )
        log.warning(
            "structure.no_resting_target",
            underlying=s.underlying, expiry=str(s.expiry), reason="unknown opening credit",
        )
        return

    try:
        order = await ctx.broker.submit_close(
            s, target, reason="rerest resting profit target", good_till_cancelled=True
        )
    except Exception as exc:  # noqa: BLE001 — a missing exit is serious; surface any failure
        # An unclosable/unprotectable position is the worst thing in the book, so a
        # failed re-rest is loud, not swallowed — the same stance as an unpriceable
        # close. The time stop and breach rule remain in force meanwhile.
        result.warnings.append(
            f"§2.6 DEFECT unrepaired: could not re-rest a target for {s.underlying} "
            f"{s.expiry}: {str(exc)[:120]}. The 15:40 time stop still covers it."
        )
        log.warning(
            "structure.rerest_failed",
            underlying=s.underlying, expiry=str(s.expiry), error=str(exc)[:200],
        )
        return

    await J.record_order(
        ctx.db,
        client_order_id=str(order.client_order_id),
        broker_order_id=str(order.id),
        intent="target",
        limit_price=target,
        qty=s.contracts,
        status=str(getattr(order.status, "value", order.status)),
    )
    # Clear the §2.6 defect in the registry now, in the same cycle. `reconcile`
    # set `has_resting_target=False` at the top of the sweep (the target really was
    # gone), and no later cycle may run before the desk is read — so without this
    # the dashboard reports a defect the sweep has already repaired. The next
    # reconcile reads the broker back and confirms it (§2.3: broker is truth); this
    # only keeps the journal honest in the interim.
    await J.record_structure(ctx.db, replace(s, has_resting_target=True))
    result.note(
        f"re-rested resting target for {s.underlying} {s.expiry} @ {target} (§2.6 repair)"
    )
    log.info(
        "structure.rerest",
        underlying=s.underlying, expiry=str(s.expiry), target=str(target),
    )


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
        await _record_equity_carried(ctx, structures)
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
        # Belt-and-suspenders: `_close_structure` and `_replace_target` already
        # degrade their own submit failures to warnings, but the sweep protects
        # capital and must finish — so any *unforeseen* raise (a spot read, a future
        # handler) is contained here rather than skipping every structure queued
        # after it and leaving the cycle unfinished. `closes`-first ordering (see
        # `sweep`) means risk-off actions still run before this can matter.
        try:
            if decision.closes:
                await _close_structure(ctx, decision, result)
            elif decision.action is Action.REPLACE_TARGET:
                await _replace_target(ctx, decision, result)
        except Exception as exc:  # noqa: BLE001 — the sweep must complete for the rest of the book
            s = decision.structure
            result.warnings.append(
                f"management error on {s.underlying} {s.expiry}: {str(exc)[:120]} "
                f"— remaining structures still swept; 15:40 flatten still covers it"
            )
            log.warning(
                "manage.structure_failed",
                underlying=s.underlying, expiry=str(s.expiry), error=str(exc)[:200],
            )

    held = sum(1 for d in decisions if d.action is Action.HOLD)
    result.note(f"swept {len(decisions)} structure(s): {result.closed} closed, {held} held")
    # Record equity after the sweep, so the desk's equity/day-P&L stay current
    # through the afternoon instead of frozen at the last entry cycle. The re-rested
    # targets are already reflected (see `_replace_target`); the delta is carried.
    await _record_equity_carried(ctx, structures)
    return structures


# --------------------------------------------------------------------------- #
# think + gate + act — the entry path
# --------------------------------------------------------------------------- #

async def _candidates(
    ctx: RunnerContext,
    view: MarketView,
    state: PortfolioState,
    *,
    core_risk_pct: Decimal,
    convexity_share: Decimal,
) -> list[TradeProposal]:
    """Every buildable candidate for one underlying, across live expiries.

    Budgets are computed once from the *current* state and passed to every
    builder, so a proposal is sized against the book as it actually is. They are
    an input to sizing, not a substitute for the gates: Gate 2 and Gate 7 re-check
    both, and the kernel's answer is the binding one.

    `core_risk_pct` and `convexity_share` come from the escalation ladder (§4.7),
    already resolved and Gate-2-clamped by the caller, so this function stays a
    pure "size against these budgets" step with no knowledge of the tournament
    calendar — the two concerns are kept in separate places on purpose.
    """
    risk_budget = core_risk_pct * state.equity
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
            convexity_share=convexity_share,
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


def _stacks_on_open_structure(
    candidate: TradeProposal, occupied: set[tuple[str, date]]
) -> bool:
    """True if entering `candidate` would put a second structure on an
    (underlying, expiry) that already holds one.

    **Do not remove this guard without reading — it is a broker constraint, not a
    heuristic.** At the clearinghouse every option leg on the same underlying and
    expiry nets into ONE position, and `reconcile.group_positions` models the book
    on exactly that (underlying, expiry) key. A second structure there does not sit
    beside the first — it *merges* with it into a single `OpenStructure`. Two 4-leg
    credit condors so merged become an 8-leg geometry, and Alpaca's mleg close
    ticket is hard-capped at 4 legs, so the position can no longer be closed and
    strands into auto-exercise; the merged max_loss (both structures' wings summed)
    is fabricated too.

    The frozen kernel cannot catch this, and correctly so: Gate 6 permits two
    structures per underlying, Gate 12 blocks only *identical* strikes, and each
    proposal is legal in isolation — the defect exists only in the combined book.
    So the entry path enforces the one-per-(underlying, expiry) rule the kernel
    structurally cannot. A missed trade is free; an un-closable one is not. The
    frictional soak (`make soak`) is the regression proof; this predicate is the
    millisecond tripwire that fails the instant the guard is edited away.
    """
    return (candidate.underlying, candidate.expiry) in occupied


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

    context = KernelContext(now=now_et())
    scored: list[tuple[TradeProposal, KernelDecision]] = []
    # client_order_id -> journalled proposal row id, so the LLM memo can be
    # attached to the exact proposal the model chose. The id is unique and already
    # the row's natural key, so it is the honest join back to the record.
    id_by_coid: dict[str, int] = {}
    cold_start = False

    # --- Phase 1: sense every underlying and journal its snapshot. -----------
    # The chains are collected before *any* gating so the portfolio delta can be
    # refreshed from a complete book: Gate 7 sums across all open structures, so a
    # single un-refreshed one would understate the whole portfolio (D-1). Sensing
    # and gating were one loop before; splitting them is what lets the state the
    # kernel sees reflect the real book rather than a placeholder of zeros.
    views: list[MarketView] = []
    for underlying in ctx.universe:
        # The accumulated daily IV (Option 3) so CHEAP_VOL can be ranked. Read
        # here, where the session runner owns the database, and passed in — `sense`
        # stays journal-free like the kernel it feeds.
        real_iv = await J.daily_iv_history(ctx.db, underlying)
        view = await sense(
            ctx.broker, underlying, max_dte=ctx.strategy.max_dte, real_iv_history=real_iv
        )
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
        views.append(view)

    # --- Refresh Gate 7's input, then freeze the state the gates see. --------
    # `reconcile` leaves every structure's dollar delta at 0 (it has no chain); the
    # sensed chains carry the deltas, so fold them in now (D-1). Any structure whose
    # legs are not all in a sensed chain cannot be honestly priced and is surfaced,
    # not silently counted as zero.
    delta_by_symbol = {
        c.occ.raw: c.delta for v in views for c in v.chain if c.delta is not None
    }
    spot_by_underlying = {v.underlying: v.spot for v in views}
    structures, unpriced = refresh_deltas(structures, delta_by_symbol, spot_by_underlying)
    if unpriced:
        result.warnings.append(
            f"Gate 7 delta unresolved for {len(unpriced)} open structure(s) "
            f"(legs outside the sensed chain): "
            + ", ".join(f"{s.underlying} {s.expiry}" for s in unpriced)
        )
    state = await _state(ctx, structures)
    await J.record_equity(ctx.db, account_id=ctx.account_row_id, state=state)

    # --- The escalation ladder picks this cycle's size regime (§4.7). --------
    # Resolved once: it depends only on the tournament calendar and the day's P&L,
    # not on any single underlying, and applies to every candidate below. Core risk
    # is capped at Gate 2's ceiling *here* (never above it), so the ladder can only
    # tilt the convexity mix and size *down* — the kernel then re-checks the result
    # anyway, making a mis-tuned rung a rejected trade, not an oversized one.
    rung = resolve_rung(ctx.trading_date, state.day_pnl_pct, ctx.ladder)
    core_risk_pct = effective_core_risk_pct(rung, ctx.risk.max_risk_per_trade_pct)
    result.note(f"ladder: {rung.label}")
    if core_risk_pct < rung.core_risk_pct:
        # Fires when a 2.5% rung meets the 2.0% ceiling: the escalation then lives
        # entirely in the convexity share, and the journal records why the two
        # differ rather than leaving a reader to wonder which number won.
        result.note(
            f"core risk clamped {rung.core_risk_pct:.1%} → {core_risk_pct:.1%} "
            f"at the Gate 2 ceiling"
        )

    # The symbols each underlying's chain actually listed, so Gate 12 can check
    # strike existence instead of skipping the check. Built once and reused both
    # here (the scoring pass) and at `_submit` (the binding pass) — the binding
    # run rebuilding a *bare* context was the surviving half of the Gate-12 hole:
    # the gate that guards the order reaching the broker was the one running blind.
    symbols_by_underlying = {
        v.underlying: frozenset(c.occ.raw for c in v.chain) for v in views
    }

    # --- Portfolio constraint: one structure per (underlying, expiry). -------
    # A broker-imposed rule the frozen kernel structurally cannot enforce — see
    # `_stacks_on_open_structure` for why stacking produces an un-closable 8-leg book.
    # Read from the *live* reconciled book, so a pair frees the moment its structure's
    # resting target fills.
    occupied = {(s.underlying, s.expiry) for s in structures}
    occupied_noted: set[tuple[str, date]] = set()

    # --- Phase 2: build and gate candidates against the refreshed book. ------
    for view in views:
        if view.verdict.structure is None:
            result.note(f"{view.underlying}: {view.verdict.regime.value} — stand down")
            continue

        ctx_for = KernelContext(
            now=context.now, available_symbols=symbols_by_underlying[view.underlying]
        )

        for candidate in await _candidates(
            ctx, view, state,
            core_risk_pct=core_risk_pct,
            convexity_share=rung.convexity_share,
        ):
            if _stacks_on_open_structure(candidate, occupied):
                # Already holding a structure here; a second would merge into an
                # un-closable book at the broker. Noted once per pair so the journal
                # records the constraint firing without a line per rejected candidate.
                pair = (candidate.underlying, candidate.expiry)
                if pair not in occupied_noted:
                    occupied_noted.add(pair)
                    result.note(
                        f"{candidate.underlying} {candidate.expiry}: skipped — "
                        "already holds an open structure (one per underlying+expiry)"
                    )
                continue
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
                    "proposal.rejected", underlying=view.underlying,
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

    # The cycle row records *one* regime (journal §5). Phase 1 left `result.regime`
    # at whichever underlying was sensed last (e.g. QQQ); the honest value is the
    # regime of the symbol we actually traded, which each proposal already carries
    # per-leg. Overwrite it now, before the cycle is finalised (O-2).
    if selection.proposal.regime is not None:
        result.regime = selection.proposal.regime.value

    await _submit(
        ctx,
        selection.proposal,
        state,
        result,
        available_symbols=symbols_by_underlying.get(
            selection.proposal.underlying, frozenset()
        ),
    )


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
    ctx: RunnerContext,
    proposal: TradeProposal,
    state: PortfolioState,
    result: CycleResult,
    *,
    available_symbols: frozenset[str] = frozenset(),
) -> None:
    """Hand the winner to the single submit path and journal what came back.

    `submit_entry` re-runs the kernel — the evaluation in `run_entry` is for
    *selection*, this one is the one that binds. A rejection here is therefore not
    a redundancy failure but a real one (the book moved between ranking and
    submitting), and it is journalled rather than retried.

    `available_symbols` is threaded through to the binding context so Gate 12
    checks strike existence on the run that actually reaches the broker, not only
    on the scoring run (O-2). An empty set is the "not supplied" sentinel that
    makes Gate 12 skip (see `KernelContext`) — the pre-fix behaviour — so callers
    must supply the traded underlying's chain symbols.
    """
    context = KernelContext(now=now_et(), available_symbols=available_symbols)
    try:
        outcome = await ctx.broker.submit_entry(
            proposal, state, context, risk=ctx.risk, strategy=ctx.strategy
        )
    except RiskKernelRejection as exc:
        result.note(f"kernel refused at submit: {exc.decision.summary}")
        log.warning("entry.rejected_at_submit", reason=exc.decision.summary)
        return

    entry_row = None
    if outcome.entry_order is not None:
        entry_row = await J.record_order(
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

    # Record the fill against the entry order (D-5). The `fills` table is the only
    # place a *partial* is a first-class row rather than a note buried in a string:
    # §6.3 makes the partial rate a headline number, and "how often were we
    # partialled?" has to be a query, not a grep. `record_fill` was written but
    # never called — the row was silently dropped on every fill until now.
    if entry_row is not None and outcome.entry_order is not None:
        filled_avg = getattr(outcome.entry_order, "filled_avg_price", None)
        await J.record_fill(
            ctx.db,
            order_id=entry_row.id,
            filled_qty=outcome.filled_contracts,
            # Paper can leave the avg price unset on an odd fill; a zero here is an
            # honest "unknown", never a real $0 fill, and keeps the NUMERIC column
            # non-null. The order row still carries the limit we actually asked for.
            filled_avg_price=Decimal(str(filled_avg)) if filled_avg else Decimal(0),
            partial=outcome.partial,
        )

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
        # The accumulated daily IV (Option 3) so CHEAP_VOL can be ranked. Read
        # here, where the session runner owns the database, and passed in — `sense`
        # stays journal-free like the kernel it feeds.
        real_iv = await J.daily_iv_history(ctx.db, underlying)
        view = await sense(
            ctx.broker, underlying, max_dte=ctx.strategy.max_dte, real_iv_history=real_iv
        )
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
    # Carry the last refreshed delta forward: postclose senses no chain, so the raw
    # state's delta is a placeholder 0, and the scored closing snapshot should not
    # report a live book as delta-flat.
    state = await _record_equity_carried(ctx, structures)
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

    # Auto-journal: produce the day's report + build-in-public draft so the
    # competition deliverable exists without a human running the CLI. Best-effort by
    # design — the books are already closed above; narrating them must not be able to
    # undo that, so a failure here logs and is noted, never raised. This is the same
    # "a close we cannot journal is still a close" rule that governs the whole cycle.
    try:
        journalled = await emit_session_journal(ctx.db)
    except Exception as exc:  # noqa: BLE001 — journaling is never worth failing close for
        log.warning("postclose.journal_failed", error=str(exc)[:200])
        result.warnings.append(f"auto-journal failed: {type(exc).__name__}")
    else:
        if journalled is not None:
            source = (
                "template" if journalled.draft.fell_back else journalled.draft.model
            )
            result.note(
                f"journalled report + social draft ({source}) → "
                f"{journalled.report_path.name}, {journalled.social_path.name}"
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
