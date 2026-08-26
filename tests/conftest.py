"""Shared fixtures.

Chain contracts are built from **raw API JSON**, not by hand-constructing SDK
models, so these tests exercise the same parsing path a live snapshot takes. A
fixture that bypassed parsing would pass while the real feed failed.

`make_contract` is a *factory fixture*: the fixture returns a function rather than
a value, so one test can build many differently-shaped contracts. conftest.py is
auto-discovered by pytest and must never be imported for its fixtures — the
module-level helpers below are plain functions and may be imported directly.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Protocol
from zoneinfo import ZoneInfo

import pytest
from alpaca.data.models.snapshots import OptionsSnapshot

from vigil.data.chain import Contract
from vigil.data.occ import parse_occ
from vigil.domain import Leg, PortfolioState, Structure, TradeProposal
from vigil.risk.context import KernelContext

_TS = datetime(2026, 8, 26, 14, 30, tzinfo=UTC).isoformat()

ET = ZoneInfo("America/New_York")
DEFAULT_EXPIRY = date(2026, 8, 27)
# 11:00 ET on a Wednesday: inside regular hours, past the opening window, well
# before the closing window. Chosen so Gate 11 is never the accidental failure.
DEFAULT_NOW = datetime(2026, 8, 26, 11, 0, tzinfo=ET)


# --------------------------------------------------------------------------- #
# Chain contracts
# --------------------------------------------------------------------------- #

class MakeContract(Protocol):
    """The callable shape `make_contract` returns — keeps mypy meaningful in tests."""

    def __call__(
        self,
        symbol: str,
        *,
        delta: float | None = None,
        iv: float | None = None,
        bid: float | None = None,
        ask: float | None = None,
    ) -> Contract: ...


@pytest.fixture
def make_contract() -> MakeContract:
    def _make(
        symbol: str,
        *,
        delta: float | None = None,
        iv: float | None = None,
        bid: float | None = None,
        ask: float | None = None,
    ) -> Contract:
        raw: dict[str, object] = {}
        if delta is not None:
            # The feed sends all five greeks together; a partial object would not
            # be representative of anything the live endpoint returns.
            raw["greeks"] = {"delta": delta, "gamma": 0.01, "rho": 0.0, "theta": -0.5, "vega": 0.1}
        if iv is not None:
            raw["impliedVolatility"] = iv
        if bid is not None and ask is not None:
            raw["latestQuote"] = {"t": _TS, "bp": bid, "bs": 10, "ap": ask, "as": 10}
        return Contract(occ=parse_occ(symbol), snapshot=OptionsSnapshot(symbol, raw))

    return _make


# --------------------------------------------------------------------------- #
# Risk-kernel inputs
# --------------------------------------------------------------------------- #

def make_leg(
    symbol: str,
    *,
    short: bool = False,
    bid: str = "1.00",
    ask: str = "1.04",
    delta: float = -0.16,
    oi: int = 5000,
    ratio: int = 1,
) -> Leg:
    return Leg(
        occ=parse_occ(symbol),
        ratio_qty=ratio,
        is_short=short,
        bid=Decimal(bid),
        ask=Decimal(ask),
        delta=delta,
        open_interest=oi,
    )


@pytest.fixture
def put_credit_spread() -> TradeProposal:
    """SPY $1-wide put credit spread, 1 DTE, ~0.16 delta short, 20% credit.

    A baseline that passes all twelve gates, so each test can break exactly one
    thing and be certain the failure it sees is the one it caused.

    Sized at **1 contract**, and the reason is worth reading: Gate 2 would permit
    25 (max loss $80/contract against a $2,000 budget), but Gate 7 permits barely
    more than one. Net delta per spread is 0.16 - 0.11 = 0.05, and on a $765
    underlying that is 0.05 x 100 x 765.85 = ~$3,829 of dollar delta against a
    $5,000 portfolio limit. **Gate 7 binds ~15x harder than Gate 2 on directional
    structures** — see tests/test_gates.py::test_gate7_binds_far_harder_than_gate2.
    """
    return TradeProposal(
        structure=Structure.PUT_CREDIT_SPREAD,
        underlying="SPY",
        spot=Decimal("765.85"),
        expiry=DEFAULT_EXPIRY,
        legs=(
            make_leg("SPY260827P00761000", short=True, bid="0.50", ask="0.52", delta=-0.16),
            make_leg("SPY260827P00760000", short=False, bid="0.30", ask="0.32", delta=-0.11),
        ),
        contracts=1,
        net_credit=Decimal("0.20"),
        width=Decimal(1),
        client_order_id="vigil-test-0001",
        limit_price=Decimal("0.20"),
    )


@pytest.fixture
def flat_book() -> PortfolioState:
    """$100k, at its peak, flat on the day, nothing open."""
    return PortfolioState(
        equity=Decimal(100_000),
        peak_equity=Decimal(100_000),
        day_pnl=Decimal(0),
    )


@pytest.fixture
def ctx() -> KernelContext:
    return KernelContext(now=DEFAULT_NOW)


@pytest.fixture
def iron_condor() -> TradeProposal:
    """SPY $1-wide iron condor, 8 contracts, exactly delta-neutral.

    The multi-contract counterpart to `put_credit_spread`, and it has to be a
    condor to *be* multi-contract: Gate 7 binds at barely one contract on a
    directional structure, so any test that needs a partial fill (or any size at
    all) needs a book whose deltas cancel.

    Legs cancel by construction — short put −0.16 and short call +0.16 carry
    opposite signed ratios, as do the two longs — so `dollar_delta` is 0 and Gate
    7 is unconstrained. Max loss is **one** width minus the total credit, because
    the underlying cannot finish below the put spread and above the call spread
    at once: (1.00 − 0.20) × 100 × 8 = $640 against a $2,000 budget.
    """
    return TradeProposal(
        structure=Structure.IRON_CONDOR,
        underlying="SPY",
        spot=Decimal("765.85"),
        expiry=DEFAULT_EXPIRY,
        legs=(
            make_leg("SPY260827P00761000", short=True, bid="0.40", ask="0.42", delta=-0.16),
            make_leg("SPY260827P00760000", short=False, bid="0.28", ask="0.30", delta=-0.11),
            make_leg("SPY260827C00770000", short=True, bid="0.40", ask="0.42", delta=0.16),
            make_leg("SPY260827C00771000", short=False, bid="0.28", ask="0.30", delta=0.11),
        ),
        contracts=8,
        # Conservative: sell the bid, buy the ask -> (0.40 - 0.30) x 2 sides.
        net_credit=Decimal("0.20"),
        width=Decimal(1),
        client_order_id="vigil-test-cndr-0001",
        # The mid the ladder opens at: (0.41 - 0.29) x 2 sides.
        limit_price=Decimal("0.24"),
    )


@pytest.fixture
def debit_spread() -> TradeProposal:
    """SPY $1-wide call debit spread — the convexity sleeve's structure (§4.5).

    `net_credit` is **negative** because the package pays premium; `limit_price`
    stays a positive magnitude because that is what the broker takes. Max loss is
    the $0.41 debit, max profit the remaining $0.59 of width, so Gate 9's
    loss:profit ratio reads 0.69:1 and the credit floor does not apply.
    """
    return TradeProposal(
        structure=Structure.DEBIT_SPREAD,
        underlying="SPY",
        spot=Decimal("765.85"),
        expiry=DEFAULT_EXPIRY,
        legs=(
            make_leg("SPY260827C00766000", short=False, bid="0.60", ask="0.62", delta=0.35),
            make_leg("SPY260827C00767000", short=True, bid="0.21", ask="0.23", delta=0.29),
        ),
        contracts=1,
        # Conservative: pay the ask, sell the bid -> 0.62 - 0.21. Negative because
        # the package pays out.
        net_credit=Decimal("-0.41"),
        width=Decimal(1),
        client_order_id="vigil-test-debt-0001",
        # The mid: 0.61 - 0.22. A positive magnitude, per TradeProposal's contract.
        limit_price=Decimal("0.39"),
    )
