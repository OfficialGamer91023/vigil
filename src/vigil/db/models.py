"""The journal (PLAN §3). SQLAlchemy 2.0, async, Postgres.

**Why a relational database for a six-day project.** The journal genuinely is
relational — a cycle has proposals, a proposal has legs and twelve verdicts, a
structure has orders and orders have fills — and §13 asks for the schema and its
indexes as a deliverable. `gate_verdicts` is called out in §3 as *the most
valuable table in the repo*: rejections are the evidence the risk system works,
and "did any of this ever actually fire?" is the first question anyone asks.

**Money is `NUMERIC`, never float.** A `float` column would silently round a
credit and then the journal would disagree with the kernel about what a trade
risked. `Numeric(18, 4)` throughout, mapping to `Decimal` in Python.

**Timestamps are `timestamptz`, stored UTC.** Trading logic reasons in Eastern
(`vigil.clock`), but storage is unambiguous; a naive column would make the whole
B5 class of bug reachable through the database.

pgvector and `journal_embeddings` are **not** here. §6.4 and §11.1 cut journal
retrieval by default: six sessions produce ~20 trades, which is not a corpus, and
retrieving "the 5 most similar setups" from 20 rows returns noise that would be
demoed as intelligence.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Money. 18 digits with 4 after the point: enough for an option price to the
# hundredth of a cent and an account balance in the billions, with no float.
MONEY = Numeric(18, 4)
# Booleans carry a `server_default` as well as a Python default throughout. An
# ORM-only default leaves the column NOT NULL with no database-level fallback, so
# any writer that is not SQLAlchemy — psql, a repair script, a raw-SQL migration —
# fails on insert. The schema should be usable without the application.
# Greeks are estimates from the feed, not currency — a wider, coarser type is
# honest about that, and they are never summed into a balance.
GREEK = Numeric(12, 6)


class Base(DeclarativeBase):
    """Declarative base. `Mapped[...]` annotations drive the column types."""


def _ts() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Account(Base):
    """The paper account. One row, asserted against `config/account.lock`."""

    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alpaca_account_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    starting_equity: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    created_at: Mapped[datetime] = _ts()

    sessions: Mapped[list[Session]] = relationship(back_populates="account")


class Session(Base):
    """One trading day."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False)
    trading_date: Mapped[date] = mapped_column(Date, nullable=False)
    opening_equity: Mapped[Decimal | None] = mapped_column(MONEY)
    closing_equity: Mapped[Decimal | None] = mapped_column(MONEY)
    halted: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=false(), nullable=False)
    created_at: Mapped[datetime] = _ts()

    account: Mapped[Account] = relationship(back_populates="sessions")
    cycles: Mapped[list[Cycle]] = relationship(back_populates="session")

    __table_args__ = (UniqueConstraint("account_id", "trading_date", name="uq_session_day"),)


class Cycle(Base):
    """One turn of `sense -> reconcile -> manage -> think -> gate -> act -> log`."""

    __tablename__ = "cycles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    kind: Mapped[str] = mapped_column(String(24), nullable=False)  # premarket/entry/manage/...
    started_at: Mapped[datetime] = _ts()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    regime: Mapped[str | None] = mapped_column(String(24))
    # True when the router reasoned from the cold-start path (§4.3.1). Persisted
    # rather than logged because a cycle that reasoned from a proxy must be
    # distinguishable from one that did not, months later, in the write-up.
    cold_start: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=false(), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)

    session: Mapped[Session] = relationship(back_populates="cycles")
    proposals: Mapped[list[Proposal]] = relationship(back_populates="cycle")

    __table_args__ = (Index("ix_cycles_session", "session_id", "started_at"),)


class Proposal(Base):
    """A candidate the kernel judged. Stored whether or not it was approved."""

    __tablename__ = "proposals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    cycle_id: Mapped[int] = mapped_column(
        ForeignKey("cycles.id", ondelete="CASCADE"), nullable=False)
    structure_type: Mapped[str] = mapped_column(String(32), nullable=False)
    underlying: Mapped[str] = mapped_column(String(16), nullable=False)
    expiry: Mapped[date] = mapped_column(Date, nullable=False)
    spot: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    contracts: Mapped[int] = mapped_column(Integer, nullable=False)
    net_credit: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    width: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    max_loss: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    # **NULL means unbounded**, and that is not a missing value.
    # A long strangle has no short leg to cap the upside, so `max_profit` is
    # genuinely infinite — `Decimal("Infinity")` in Python, which NUMERIC cannot
    # store. Writing a large stand-in would fabricate a measurement; NULL states
    # the truth and forces every reader to handle it.
    max_profit: Mapped[Decimal | None] = mapped_column(MONEY)
    dollar_delta: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    client_order_id: Mapped[str] = mapped_column(String(64), nullable=False)
    limit_price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    regime: Mapped[str | None] = mapped_column(String(24))
    rationale: Mapped[str | None] = mapped_column(Text)
    approved: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = _ts()

    cycle: Mapped[Cycle] = relationship(back_populates="proposals")
    legs: Mapped[list[ProposalLeg]] = relationship(back_populates="proposal")
    verdicts: Mapped[list[GateVerdictRow]] = relationship(back_populates="proposal")

    __table_args__ = (Index("ix_proposals_cycle", "cycle_id"),)


class ProposalLeg(Base):
    __tablename__ = "proposal_legs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    proposal_id: Mapped[int] = mapped_column(
        ForeignKey("proposals.id", ondelete="CASCADE"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    ratio_qty: Mapped[int] = mapped_column(Integer, nullable=False)
    is_short: Mapped[bool] = mapped_column(Boolean, nullable=False)
    strike: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    is_put: Mapped[bool] = mapped_column(Boolean, nullable=False)
    bid: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    ask: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    delta: Mapped[Decimal] = mapped_column(GREEK, nullable=False)
    open_interest: Mapped[int] = mapped_column(Integer, nullable=False)

    proposal: Mapped[Proposal] = relationship(back_populates="legs")

    __table_args__ = (Index("ix_proposal_legs_proposal", "proposal_id"),)


class GateVerdictRow(Base):
    """**The most valuable table in the repo** (§3). Passes stored too.

    Twelve rows per proposal, always. The unique constraint on
    `(proposal_id, gate_no)` is what makes that literal rather than aspirational:
    a partial write cannot masquerade as a complete evaluation, and a gate cannot
    be recorded twice with different answers.
    """

    __tablename__ = "gate_verdicts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    proposal_id: Mapped[int] = mapped_column(
        ForeignKey("proposals.id", ondelete="CASCADE"), nullable=False)
    gate_no: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(48), nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    detail: Mapped[dict[str, str] | None] = mapped_column(JSONB)

    proposal: Mapped[Proposal] = relationship(back_populates="verdicts")

    __table_args__ = (
        UniqueConstraint("proposal_id", "gate_no", name="uq_verdict_proposal_gate"),
        # Powers the rejection-stats panel and the best slide in the deck (§3).
        Index("ix_gate_verdicts_gate_passed", "gate_no", "passed"),
        CheckConstraint("gate_no >= 0 AND gate_no <= 12", name="ck_gate_no_range"),
    )


class OpenStructureRow(Base):
    """The open-position registry. Broker is truth; this is reconciled each cycle."""

    __tablename__ = "structures"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    proposal_id: Mapped[int | None] = mapped_column(
        ForeignKey("proposals.id", ondelete="SET NULL"))
    underlying: Mapped[str] = mapped_column(String(16), nullable=False)
    expiry: Mapped[date] = mapped_column(Date, nullable=False)
    structure_type: Mapped[str | None] = mapped_column(String(32))
    contracts: Mapped[int] = mapped_column(Integer, nullable=False)
    net_credit: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    max_loss: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="open",
                                        server_default="open")
    has_resting_target: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=false(), nullable=False)
    opened_at: Mapped[datetime] = _ts()
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    close_reason: Mapped[str | None] = mapped_column(Text)

    orders: Mapped[list[Order]] = relationship(back_populates="structure")

    __table_args__ = (
        # **Partial index** (§3): the concentration gate runs every cycle and only
        # ever asks about *open* structures, so indexing the closed ones would be
        # paying to maintain rows no query reads.
        Index("ix_structures_open_underlying", "underlying",
              postgresql_where="status = 'open'"),
    )


class Order(Base):
    """Every ticket sent to the broker.

    `client_order_id` is `UNIQUE NOT NULL` — hard rule #9. **Idempotency is a
    database constraint, not application luck**: a retry after a timeout that
    actually succeeded raises an integrity error instead of double-filling. That
    is also why each price-ladder rung carries its own suffixed id; three rungs
    sharing one id would collide here even if the broker allowed it.
    """

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    structure_id: Mapped[int | None] = mapped_column(
        ForeignKey("structures.id", ondelete="SET NULL"))
    client_order_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(64))
    intent: Mapped[str] = mapped_column(String(16), nullable=False)  # open/close/target
    limit_price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    rung: Mapped[int | None] = mapped_column(Integer)
    submitted_at: Mapped[datetime] = _ts()

    structure: Mapped[OpenStructureRow | None] = relationship(back_populates="orders")
    fills: Mapped[list[Fill]] = relationship(back_populates="order")


class Fill(Base):
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    order_id: Mapped[int] = mapped_column(
        ForeignKey("orders.id", ondelete="CASCADE"), nullable=False)
    filled_qty: Mapped[int] = mapped_column(Integer, nullable=False)
    filled_avg_price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    # True when the order settled for less than it asked for (§1.2: ~10% of paper
    # fills). Stored as a flag so "how often did we get partialled?" is a query,
    # not an archaeology exercise across qty columns.
    partial: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=false(), nullable=False)
    filled_at: Mapped[datetime] = _ts()

    order: Mapped[Order] = relationship(back_populates="fills")

    __table_args__ = (Index("ix_fills_order", "order_id"),)


class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False)
    ts: Mapped[datetime] = _ts()
    equity: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    day_pnl: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    open_risk: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    net_dollar_delta: Mapped[Decimal] = mapped_column(MONEY, nullable=False)

    __table_args__ = (
        # DESC because the curve query is the dashboard's hot path and always
        # reads the recent end (§3).
        Index("ix_equity_account_ts", "account_id", ts.desc()),
    )


class MarketSnapshotRow(Base):
    """Per symbol per cycle — what the router actually saw when it decided."""

    __tablename__ = "market_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    cycle_id: Mapped[int] = mapped_column(
        ForeignKey("cycles.id", ondelete="CASCADE"), nullable=False)
    underlying: Mapped[str] = mapped_column(String(16), nullable=False)
    spot: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    iv_atm: Mapped[Decimal | None] = mapped_column(GREEK)
    rv_annual: Mapped[Decimal | None] = mapped_column(GREEK)
    vrp_pct: Mapped[Decimal | None] = mapped_column(GREEK)
    iv_pct: Mapped[Decimal | None] = mapped_column(GREEK)
    trend: Mapped[Decimal | None] = mapped_column(GREEK)

    __table_args__ = (Index("ix_market_snapshots_cycle", "cycle_id"),)


class LlmMemo(Base):
    """Model, effort, latency and tokens. Makes the cache-hit rate observable (§3).

    `cached_tokens` and `cache_write_tokens` are both persisted because the
    OpenAI Responses API reports both (CLI_NOTES §4) — a single read counter
    would leave "did the prefix actually cache?" unanswerable, and §6.2 makes
    prompt ordering load-bearing precisely because caching is automatic and
    prefix-based.
    """

    __tablename__ = "llm_memos"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    cycle_id: Mapped[int] = mapped_column(
        ForeignKey("cycles.id", ondelete="CASCADE"), nullable=False)
    proposal_id: Mapped[int | None] = mapped_column(
        ForeignKey("proposals.id", ondelete="SET NULL"))
    model: Mapped[str] = mapped_column(String(48), nullable=False)
    effort: Mapped[str | None] = mapped_column(String(16))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    cached_tokens: Mapped[int | None] = mapped_column(Integer)
    cache_write_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer)
    memo: Mapped[str | None] = mapped_column(Text)
    # True when the model failed or returned invalid JSON and the deterministic
    # path took over (§6.3). The fallback rate is a headline reliability number.
    fell_back: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=false(), nullable=False)
    created_at: Mapped[datetime] = _ts()

    __table_args__ = (Index("ix_llm_memos_cycle", "cycle_id"),)


class ControlFlag(Base):
    """Out-of-band halt / flatten (§5.2).

    A table rather than a file so the API service can write it without sharing a
    filesystem with the worker — §2.2 requires the API to never hold trading
    state, and a row it writes and the worker reads is the narrowest possible
    channel between them.
    """

    __tablename__ = "control_flags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(24), unique=True, nullable=False)
    active: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=false(), nullable=False)
    set_by: Mapped[str | None] = mapped_column(String(48))
    reason: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False)
