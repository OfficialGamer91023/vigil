"""The objects that cross module boundaries.

Deliberately plain frozen dataclasses, not SQLAlchemy models and not Pydantic:
the risk kernel must be testable with no database and no network, so what it
receives can only be inert data (CLAUDE.md conventions).

Money is `Decimal` everywhere. Greeks stay `float` — they are estimates from the
feed, not currency, and pretending otherwise would imply precision we do not have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import StrEnum

from vigil.data.occ import OccSymbol

# Every option contract controls 100 shares. The single most common source of
# off-by-100 errors in options code, so it is named once and never inlined.
CONTRACT_MULTIPLIER = Decimal(100)


class Structure(StrEnum):
    PUT_CREDIT_SPREAD = "put_credit_spread"
    CALL_CREDIT_SPREAD = "call_credit_spread"
    IRON_CONDOR = "iron_condor"
    # A condor with equal, narrow wings but short strikes at *different* deltas,
    # so the package carries a directional lean toward the trend while still
    # collecting two credits. It exists to resolve B-1: a single vertical's
    # credit/width is pinned below its short delta (credit/width ≈ N(−d₂) < |Δ|),
    # so at 0.16 delta it cannot reach Gate 9's 18% floor — but two credits can.
    # See strategy/candidates.build_broken_wing_condor and PLAN §4.4.2.
    BROKEN_WING_CONDOR = "broken_wing_condor"
    DEBIT_SPREAD = "debit_spread"
    LONG_STRANGLE = "long_strangle"


class Regime(StrEnum):
    CHOP = "chop"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    STRESS = "stress"
    CHEAP_VOL = "cheap_vol"


@dataclass(frozen=True, slots=True)
class Leg:
    """One leg of a proposed structure, with the pricing it was built from.

    Liquidity and greek fields are **required, not optional**: the kernel rejects
    proposals with missing fields rather than inferring them, so a builder that
    cannot populate these must not emit a proposal at all.
    """

    occ: OccSymbol
    ratio_qty: int
    is_short: bool
    bid: Decimal
    ask: Decimal
    delta: float
    open_interest: int

    @property
    def symbol(self) -> str:
        return self.occ.raw

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid

    @property
    def signed_ratio(self) -> int:
        """+1 long, −1 short. Used for delta and cash-flow arithmetic."""
        return -self.ratio_qty if self.is_short else self.ratio_qty


@dataclass(frozen=True, slots=True)
class TradeProposal:
    """A candidate the kernel will accept or reject. Never partially specified.

    `net_credit`, `max_loss` and `max_profit` are **per one contract of the
    package**; `contracts` scales them. Keeping the per-unit economics separate
    from the size is what lets the sizing gate reason about the two independently.

    **Two sign conventions, and they are deliberately different.**

    - `net_credit` is **signed**: positive when the package collects premium,
      negative when it pays. That sign is the single source of truth for whether
      this is a credit or a debit structure — `is_credit`, the max-loss formula
      and the router's choice of ladder all read it.
    - `limit_price` is always a **positive magnitude**, because that is what
      Alpaca's mleg ticket takes: direction is carried by each leg's
      `position_intent`, never by the sign of the package price. Gate 12 compares
      it against a same-signed package mid, so a negative debit here reads as a
      200% deviation from mid and is rejected.
    """

    structure: Structure
    underlying: str
    spot: Decimal
    expiry: date
    legs: tuple[Leg, ...]
    contracts: int
    net_credit: Decimal
    width: Decimal
    client_order_id: str
    limit_price: Decimal
    regime: Regime | None = None
    rationale: str = ""

    @property
    def is_credit(self) -> bool:
        return self.net_credit > 0

    @property
    def is_long_only(self) -> bool:
        """No short legs anywhere — a purely long-premium structure.

        Derived from the legs rather than from `structure`, so it stays true for
        any long-only shape someone adds later without needing a new branch. It
        is the property that decides whether `width` means anything: a long
        strangle's call and put strikes are not the edges of a spread, so the
        distance between them describes nothing about the payoff.
        """
        return bool(self.legs) and all(not leg.is_short for leg in self.legs)

    @property
    def max_loss_per_contract(self) -> Decimal:
        """Width minus credit for a vertical; the premium paid for anything long.

        Finite in every branch — which is Gate 1's whole subject. Note that the
        long-only case needs no special handling: with no short leg there is
        nothing to be assigned on, so the premium paid *is* the whole exposure.
        """
        if self.is_credit:
            return (self.width - self.net_credit) * CONTRACT_MULTIPLIER
        return abs(self.net_credit) * CONTRACT_MULTIPLIER

    @property
    def max_profit_per_contract(self) -> Decimal:
        """Capped for spreads; **unbounded** for a long-only structure.

        A long strangle has no short leg to cap the upside, so the honest answer
        is infinity rather than a large number that would look measured. Two
        consequences worth knowing:

        - Gate 9's `max_loss / max_profit` ratio evaluates to 0 and passes, which
          is correct: that gate exists to refuse risking a lot to make a little,
          and this structure is the opposite shape.
        - **The journal must store NULL here, not a number.** Postgres `NUMERIC`
          cannot hold an infinity, so the repository layer has to map it — a
          capped stand-in would be a fabricated measurement.
        """
        if self.is_credit:
            return self.net_credit * CONTRACT_MULTIPLIER
        if self.is_long_only:
            return Decimal("Infinity")
        return (self.width - abs(self.net_credit)) * CONTRACT_MULTIPLIER

    @property
    def max_loss(self) -> Decimal:
        return self.max_loss_per_contract * self.contracts

    @property
    def max_profit(self) -> Decimal:
        return self.max_profit_per_contract * self.contracts

    @property
    def credit_pct_of_width(self) -> Decimal:
        if self.width == 0:
            return Decimal(0)
        return abs(self.net_credit) / self.width

    @property
    def dollar_delta(self) -> Decimal:
        """Σ (delta × 100 × spot × signed qty). The Gate 7 unit (§5.1)."""
        total = Decimal(0)
        for leg in self.legs:
            total += Decimal(str(leg.delta)) * CONTRACT_MULTIPLIER * self.spot * leg.signed_ratio
        return total * self.contracts

    @property
    def structure_key(self) -> tuple[str, date, tuple[Decimal, ...]]:
        """Identity for the Gate 12 duplicate check — same legs, same expiry."""
        return (self.underlying, self.expiry, tuple(sorted(leg.occ.strike for leg in self.legs)))


@dataclass(frozen=True, slots=True)
class PositionLeg:
    """One leg of a position already at the broker.

    Deliberately thinner than `Leg`: closing a structure needs its symbol, its
    ratio and which way it was opened, and nothing else. Requiring live bids,
    asks, deltas and open interest to *exit* would mean a position could become
    unclosable exactly when the feed is worst — which is when exiting matters.
    """

    symbol: str
    ratio_qty: int
    is_short: bool


@dataclass(frozen=True, slots=True)
class OpenStructure:
    """A position already at the broker, as the kernel and the manage sweep see it.

    The short strikes are split by right rather than kept in one bag, because the
    only question the manage sweep asks of them is directional: a put spread is
    breached when spot falls *to or below* its short put, a call spread when spot
    rises *to or above* its short call. Netting the two into one collection would
    make that question unanswerable without re-deriving the right from the strike.
    """

    underlying: str
    expiry: date
    strikes: tuple[Decimal, ...]
    max_loss: Decimal
    dollar_delta: Decimal
    has_resting_target: bool = False
    structure: Structure | None = None
    short_put_strikes: tuple[Decimal, ...] = ()
    short_call_strikes: tuple[Decimal, ...] = ()
    # Package credit received (positive) or debit paid (negative), per contract.
    net_credit: Decimal = Decimal(0)
    contracts: int = 1
    legs: tuple[PositionLeg, ...] = ()

    @property
    def structure_key(self) -> tuple[str, date, tuple[Decimal, ...]]:
        return (self.underlying, self.expiry, tuple(sorted(self.strikes)))

    @property
    def has_short_legs(self) -> bool:
        """A long-only structure cannot be breached — there is nothing short to breach."""
        return bool(self.short_put_strikes or self.short_call_strikes)

    def is_breached(self, spot: Decimal) -> bool:
        """Has the underlying traded through a short strike?

        Uses the *nearest* short strike on each side — the first one the tape
        reaches is the one that matters, and on an iron condor the put and call
        sides are breached independently.
        """
        if self.short_put_strikes and spot <= max(self.short_put_strikes):
            return True
        return bool(self.short_call_strikes and spot >= min(self.short_call_strikes))


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Everything the portfolio-level gates need, and nothing else.

    A snapshot, not a live handle — the kernel must never be able to query the
    broker mid-decision.
    """

    equity: Decimal
    peak_equity: Decimal
    day_pnl: Decimal
    open_structures: tuple[OpenStructure, ...] = ()
    halted: bool = False
    known_client_order_ids: frozenset[str] = frozenset()

    @property
    def day_pnl_pct(self) -> Decimal:
        return Decimal(0) if self.equity == 0 else self.day_pnl / self.equity

    @property
    def drawdown_pct(self) -> Decimal:
        """Negative when below peak. Zero when at or above it."""
        if self.peak_equity <= 0:
            return Decimal(0)
        return (self.equity - self.peak_equity) / self.peak_equity

    @property
    def open_risk(self) -> Decimal:
        return sum((s.max_loss for s in self.open_structures), Decimal(0))

    @property
    def net_dollar_delta(self) -> Decimal:
        return sum((s.dollar_delta for s in self.open_structures), Decimal(0))


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """One gate's answer. Persisted even on a pass — §5 requires the full record."""

    number: int
    name: str
    passed: bool
    reason: str = ""
    # Free-form measured values, so a rejection can be debugged without a re-run.
    detail: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class KernelDecision:
    approved: bool
    verdicts: tuple[GateVerdict, ...]

    @property
    def failures(self) -> tuple[GateVerdict, ...]:
        return tuple(v for v in self.verdicts if not v.passed)

    @property
    def summary(self) -> str:
        if self.approved:
            return f"APPROVED ({len(self.verdicts)} gates passed)"
        parts = (f"gate {v.number} ({v.name}): {v.reason}" for v in self.failures)
        return "REJECTED: " + "; ".join(parts)
