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
    DEBIT_SPREAD = "debit_spread"


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
    def max_loss_per_contract(self) -> Decimal:
        """Width minus credit for a vertical; the debit paid for a debit spread.

        Both are finite by construction — which is Gate 1's whole subject.
        """
        if self.is_credit:
            return (self.width - self.net_credit) * CONTRACT_MULTIPLIER
        return abs(self.net_credit) * CONTRACT_MULTIPLIER

    @property
    def max_profit_per_contract(self) -> Decimal:
        if self.is_credit:
            return self.net_credit * CONTRACT_MULTIPLIER
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
class OpenStructure:
    """A position already at the broker, as the kernel needs to see it."""

    underlying: str
    expiry: date
    strikes: tuple[Decimal, ...]
    max_loss: Decimal
    dollar_delta: Decimal
    has_resting_target: bool = False

    @property
    def structure_key(self) -> tuple[str, date, tuple[Decimal, ...]]:
        return (self.underlying, self.expiry, tuple(sorted(self.strikes)))


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
