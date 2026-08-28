"""Response models. Pydantic, so the OpenAPI schema is generated rather than written.

**Money stays `Decimal` all the way to the JSON encoder.** CLAUDE.md says money is
`Decimal` in Python and `NUMERIC` in Postgres, never float, and that rule does not
stop at the serializer — `float(Decimal("0.05"))` is where a credit becomes
`0.05000000000000000277`. Pydantic emits `Decimal` as a JSON number without going
through binary floating point, so the wire value matches the row.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class Health(BaseModel):
    """`/health`. Answers "is the *agent* alive?", not "is this web server up?".

    A 200 from a FastAPI process proves almost nothing — the API is designed to
    run correctly while the worker is stopped, so an "ok" that only meant "the
    route resolved" would be actively misleading. `last_cycle_age_seconds` is the
    number that actually distinguishes a live agent from a dead one.
    """

    status: str
    database: bool
    last_cycle_at: datetime | None = None
    last_cycle_kind: str | None = None
    last_cycle_age_seconds: float | None = None
    halted: bool = False
    flatten_requested: bool = False


class StructureOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    underlying: str
    expiry: date
    structure_type: str | None
    contracts: int
    net_credit: Decimal
    max_loss: Decimal
    # §2.6: an open structure without a live resting GTC profit target is a
    # reconciliation defect. Surfaced on every structure so the defect is visible
    # on the dashboard rather than buried in a log line.
    has_resting_target: bool
    opened_at: datetime


class StateOut(BaseModel):
    """`/api/state` — the whole desk in one object."""

    account_id: str | None
    equity: Decimal | None
    day_pnl: Decimal | None
    open_risk: Decimal
    net_dollar_delta: Decimal | None
    open_structures: list[StructureOut]
    halted: bool
    flatten_requested: bool
    trading_date: date | None
    as_of: datetime | None


class EquityPoint(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ts: datetime
    equity: Decimal
    day_pnl: Decimal
    open_risk: Decimal
    net_dollar_delta: Decimal


class CycleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    kind: str
    started_at: datetime
    # NULL means the cycle never finished — `run_cycle` commits this row in its
    # own transaction precisely so a crash leaves evidence (see sessions.py).
    finished_at: datetime | None
    regime: str | None
    cold_start: bool
    notes: str | None


class VerdictOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    gate_no: int
    name: str
    passed: bool
    reason: str | None


class LegOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    ratio_qty: int
    is_short: bool
    strike: Decimal
    is_put: bool
    delta: Decimal
    open_interest: int


class ProposalOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    structure_type: str
    underlying: str
    expiry: date
    contracts: int
    net_credit: Decimal
    width: Decimal
    max_loss: Decimal
    # NULL is **unbounded**, not missing — a long strangle has no capped upside
    # and `Decimal("Infinity")` is not storable in NUMERIC (see models.py).
    max_profit: Decimal | None
    dollar_delta: Decimal
    regime: str | None
    rationale: str | None
    approved: bool
    legs: list[LegOut] = []
    verdicts: list[VerdictOut] = []


class CycleDetailOut(CycleOut):
    proposals: list[ProposalOut] = []


class MarketSnapshotOut(BaseModel):
    """One underlying as the router last saw it (`/api/market`).

    Every field after `spot` is nullable and that is meaningful, not sloppy:
    `rv_annual` is `None` when realized vol was unmeasurable, `vrp_pct` is `None`
    when there was no distribution to rank against, and `iv_atm` is `None` when
    the indicative feed served no greeks. The page renders a dash for each, so a
    missing measurement reads as missing rather than as zero — which is the exact
    confusion `MarketSnapshot.rv_annual` exists to prevent upstream.
    """

    model_config = ConfigDict(from_attributes=True)

    underlying: str
    spot: Decimal
    iv_atm: Decimal | None
    rv_annual: Decimal | None
    vrp_pct: Decimal | None
    iv_pct: Decimal | None
    trend: Decimal | None
    cycle_id: int


class OrderOut(BaseModel):
    """One ticket sent to the broker (`/api/orders`).

    `intent` is `open` / `close` / `target`, and `rung` is the price-ladder step
    an entry filled on — both are what make the execution story legible on the
    page rather than a flat list of limit prices.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    client_order_id: str
    intent: str
    limit_price: Decimal
    qty: int
    status: str
    rung: int | None
    submitted_at: datetime


class GateStat(BaseModel):
    gate_no: int
    name: str
    passed: int
    failed: int

    @property
    def total(self) -> int:
        return self.passed + self.failed


class ControlFlagOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    active: bool
    set_by: str | None
    reason: str | None
    updated_at: datetime


class ControlResult(BaseModel):
    """What a control route did.

    `effective_from` is prose rather than a timestamp on purpose: the API writes a
    flag and the **worker** acts on it at the top of its next cycle. Returning a
    time would imply this service did something to the account, and it did not —
    it wrote a row.
    """

    flag: str
    active: bool
    effective_from: str
    detail: str | None = None
