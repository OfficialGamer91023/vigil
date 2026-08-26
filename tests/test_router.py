"""The single submit path (hard rule #4).

No network: a fake TradingClient records what it was asked to do. The point of
these tests is not that Alpaca works — it is that **nothing reaches a broker
without passing the kernel**, and that a fill always leaves a resting exit behind.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

import pytest

from tests.conftest import DEFAULT_EXPIRY
from vigil.execution.router import RiskKernelRejection, submit_entry
from vigil.risk.context import KernelContext


@dataclass
class FakeOrder:
    id: str
    status: str
    filled_avg_price: str | None = None
    filled_qty: str = "0"


class DuplicateClientOrderId(RuntimeError):
    """What Alpaca does when a `client_order_id` is reused.

    Modelled explicitly because the fake is the only thing standing between a
    broker-side uniqueness rule and a test suite that would otherwise certify a
    ladder that cannot actually concede.
    """


@dataclass
class FakeTradingClient:
    """Records every call. `fill_on_rung` decides which ladder rung fills.

    `partial_qty` makes a rung come back **partially** filled, which is the ~10%
    case §1.2 describes and the one the router used to drop on the floor.
    """

    fill_on_rung: int | None = 1
    partial_qty: int | None = None
    submitted: list[Any] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    seen_ids: set[str] = field(default_factory=set)
    _n: int = 0

    def _register(self, req: Any) -> None:
        """Enforce `client_order_id` uniqueness the way the broker does."""
        coid = req.client_order_id
        if coid in self.seen_ids:
            raise DuplicateClientOrderId(coid)
        self.seen_ids.add(coid)

    def _order_for(self, n: int, order_id: str) -> FakeOrder:
        if self.partial_qty is not None and n == (self.fill_on_rung or 1):
            return FakeOrder(id=order_id, status="partially_filled",
                             filled_avg_price="0.20", filled_qty=str(self.partial_qty))
        filled = self.fill_on_rung is not None and n == self.fill_on_rung
        return FakeOrder(
            id=order_id,
            status="filled" if filled else "new",
            filled_avg_price="0.20" if filled else None,
            filled_qty="8" if filled else "0",
        )

    def submit_order(self, req: Any) -> FakeOrder:
        self._register(req)
        self.submitted.append(req)
        # A closing order (buy_to_close present) is the resting target, never a rung.
        # `.value` is the lowercase wire format ("buy_to_close"); str() on the
        # enum gives "PositionIntent.BUY_TO_CLOSE", which a substring check misses.
        intents = {leg.position_intent.value for leg in req.legs}
        if any("close" in i for i in intents):
            return FakeOrder(id=f"tgt-{len(self.submitted)}", status="new")
        self._n += 1
        return self._order_for(self._n, f"ord-{self._n}")

    def get_order_by_id(self, order_id: str) -> FakeOrder:
        n = int(str(order_id).split("-")[-1])
        return self._order_for(n, str(order_id))

    def cancel_order_by_id(self, order_id: str) -> None:
        self.cancelled.append(str(order_id))


def _noop_sleep(_seconds: float) -> None:
    """The ladder waits 20s per rung in production; tests must not."""


def test_a_rejected_proposal_never_reaches_the_broker(put_credit_spread, flat_book, ctx) -> None:
    """The rule the whole architecture exists to enforce."""
    client = FakeTradingClient()
    halted = replace(flat_book, halted=True)

    with pytest.raises(RiskKernelRejection):
        submit_entry(put_credit_spread, halted, ctx, client=client, sleep=_noop_sleep)

    assert client.submitted == [], "an order was submitted despite kernel rejection"


def test_oversized_proposal_is_rejected_before_submission(
    put_credit_spread, flat_book, ctx
) -> None:
    client = FakeTradingClient()
    huge = replace(put_credit_spread, contracts=5_000)

    with pytest.raises(RiskKernelRejection) as exc:
        submit_entry(huge, flat_book, ctx, client=client, sleep=_noop_sleep)

    assert client.submitted == []
    assert {2, 7} & {v.number for v in exc.value.decision.failures}


def test_a_fill_always_leaves_a_resting_profit_target(put_credit_spread, flat_book, ctx) -> None:
    """§2.6: an open structure with no resting exit is a reconciliation defect."""
    client = FakeTradingClient(fill_on_rung=1)
    result = submit_entry(put_credit_spread, flat_book, ctx, client=client, sleep=_noop_sleep)

    assert result.filled
    assert result.target_order is not None
    # Two tickets: the entry, then the resting close.
    assert len(client.submitted) == 2
    closing = client.submitted[1]
    assert all("close" in leg.position_intent.value for leg in closing.legs)
    assert closing.time_in_force.value == "gtc"


def test_the_target_price_is_derived_from_the_actual_fill(
    put_credit_spread, flat_book, ctx
) -> None:
    """Not from the proposal — a partial or improved fill changes the target."""
    client = FakeTradingClient(fill_on_rung=1)
    submit_entry(put_credit_spread, flat_book, ctx, client=client, sleep=_noop_sleep)
    # Filled at $0.20; 50% target means buying it back at $0.10.
    assert client.submitted[1].limit_price == 0.10


def test_the_ladder_concedes_and_cancels_between_rungs(put_credit_spread, flat_book, ctx) -> None:
    """Never two live tickets for one structure."""
    client = FakeTradingClient(fill_on_rung=2)
    result = submit_entry(put_credit_spread, flat_book, ctx, client=client, sleep=_noop_sleep)

    assert result.filled and result.rungs_used == 2
    assert client.cancelled == ["ord-1"], "the unfilled rung was not cancelled"
    # The second rung asked for less credit than the first.
    assert client.submitted[1].limit_price < client.submitted[0].limit_price


def test_exhausting_the_ladder_logs_no_fill_rather_than_chasing(
    put_credit_spread, flat_book, ctx
) -> None:
    """A missed trade is free; a bad fill is not."""
    client = FakeTradingClient(fill_on_rung=None)
    result = submit_entry(put_credit_spread, flat_book, ctx, client=client, sleep=_noop_sleep)

    assert not result.filled
    assert result.target_order is None
    assert "NO_FILL" in result.note
    # Every rung was cancelled; nothing was left working at the broker.
    assert len(client.cancelled) == len(client.submitted)


def test_no_rung_is_ever_priced_below_the_credit_floor(
    put_credit_spread, flat_book, ctx
) -> None:
    """The floor is Gate 9's threshold — conceding past it submits what the
    kernel would now reject."""
    from vigil.config import risk_config

    client = FakeTradingClient(fill_on_rung=None)
    submit_entry(put_credit_spread, flat_book, ctx, client=client, sleep=_noop_sleep)

    floor = risk_config().min_credit_pct_of_width * put_credit_spread.width
    assert all(Decimal(str(r.limit_price)) >= floor for r in client.submitted)


def test_kernel_decision_is_returned_even_on_success(
    put_credit_spread, flat_book, ctx
) -> None:
    """Every verdict is persisted, passes included — §5 wants the full record."""
    client = FakeTradingClient(fill_on_rung=1)
    result = submit_entry(put_credit_spread, flat_book, ctx, client=client, sleep=_noop_sleep)
    assert len(result.decision.verdicts) == 12
    assert all(v.passed for v in result.decision.verdicts)


def test_context_without_a_chain_still_submits(put_credit_spread, flat_book) -> None:
    """A missing optional input must not masquerade as a risk violation."""
    from tests.conftest import DEFAULT_NOW

    client = FakeTradingClient(fill_on_rung=1)
    result = submit_entry(
        put_credit_spread, flat_book, KernelContext(now=DEFAULT_NOW),
        client=client, sleep=_noop_sleep,
    )
    assert result.filled


# --------------------------------------------------------------------------- #
# B1 — every ladder rung is a distinct order and needs a distinct id
# --------------------------------------------------------------------------- #

def test_each_ladder_rung_carries_a_unique_client_order_id(
    put_credit_spread, flat_book, ctx
) -> None:
    """Alpaca rejects a reused `client_order_id`, so a ladder that reuses one
    cannot concede past its first rung. The fake enforces the same rule."""
    client = FakeTradingClient(fill_on_rung=None)
    submit_entry(put_credit_spread, flat_book, ctx, client=client, sleep=_noop_sleep)

    ids = [r.client_order_id for r in client.submitted]
    assert len(ids) == len(set(ids)), f"duplicate client_order_id across rungs: {ids}"
    assert len(ids) > 1, "the ladder never conceded, so this proves nothing"


def test_rung_ids_share_the_proposal_id_as_their_stem(
    put_credit_spread, flat_book, ctx
) -> None:
    """The base id stays the structure's identity — that is what Gate 12's
    duplicate check and the journal's idempotency constraint reason about."""
    client = FakeTradingClient(fill_on_rung=None)
    submit_entry(put_credit_spread, flat_book, ctx, client=client, sleep=_noop_sleep)

    base = put_credit_spread.client_order_id
    assert all(r.client_order_id.startswith(f"{base}-r") for r in client.submitted)


# --------------------------------------------------------------------------- #
# B2 — a partial fill is a position, not a non-event
# --------------------------------------------------------------------------- #

def test_a_partial_fill_still_gets_a_resting_target(
    iron_condor, flat_book, ctx
) -> None:
    """§2.6: an open structure with no resting exit is a reconciliation defect —
    and a partial fill is an open structure."""
    client = FakeTradingClient(fill_on_rung=1, partial_qty=3)
    result = submit_entry(iron_condor, flat_book, ctx, client=client, sleep=_noop_sleep)

    assert result.filled and result.partial
    assert result.filled_contracts == 3
    assert result.target_order is not None
    closing = client.submitted[-1]
    assert all("close" in leg.position_intent.value for leg in closing.legs)


def test_the_resting_target_is_sized_to_what_actually_filled(
    iron_condor, flat_book, ctx
) -> None:
    """Resting a target for the size we *asked* for leaves an order the account
    cannot honour."""
    client = FakeTradingClient(fill_on_rung=1, partial_qty=3)
    submit_entry(iron_condor, flat_book, ctx, client=client, sleep=_noop_sleep)

    assert iron_condor.contracts != 3, "fixture must differ from the partial qty"
    assert client.submitted[-1].qty == 3


def test_a_partial_fill_stops_the_ladder(iron_condor, flat_book, ctx) -> None:
    """The old behaviour cancelled the partial and submitted a SECOND entry
    ticket on top of live contracts. Once anything fills, the ladder is over."""
    client = FakeTradingClient(fill_on_rung=1, partial_qty=3)
    submit_entry(iron_condor, flat_book, ctx, client=client, sleep=_noop_sleep)

    entries = [r for r in client.submitted
               if all("open" in leg.position_intent.value for leg in r.legs)]
    assert len(entries) == 1, f"submitted {len(entries)} entry tickets after a partial"


def test_a_full_fill_is_not_reported_as_partial(put_credit_spread, flat_book, ctx) -> None:
    client = FakeTradingClient(fill_on_rung=1)
    result = submit_entry(put_credit_spread, flat_book, ctx, client=client, sleep=_noop_sleep)
    assert result.filled and not result.partial
    assert result.filled_contracts == put_credit_spread.contracts


# --------------------------------------------------------------------------- #
# Closing — the kernel does not vote on exits
# --------------------------------------------------------------------------- #

def _open_spread(target: bool = True):
    from vigil.domain import OpenStructure, PositionLeg, Structure

    return OpenStructure(
        underlying="SPY", expiry=DEFAULT_EXPIRY,
        strikes=(Decimal("761"), Decimal("760")),
        max_loss=Decimal(80), dollar_delta=Decimal(-3800),
        has_resting_target=target, structure=Structure.PUT_CREDIT_SPREAD,
        short_put_strikes=(Decimal("761"),), net_credit=Decimal("0.20"), contracts=4,
        legs=(PositionLeg("SPY260827P00761000", 1, True),
              PositionLeg("SPY260827P00760000", 1, False)),
    )


def test_a_close_reverses_every_leg() -> None:
    from vigil.execution.router import submit_close

    client = FakeTradingClient()
    submit_close(_open_spread(), Decimal("0.10"), client=client, reason="breach")
    req = client.submitted[0]
    assert all("close" in leg.position_intent.value for leg in req.legs)
    assert req.qty == 4


def test_a_close_is_not_blocked_by_a_halted_book() -> None:
    """**The asymmetry that matters.** A gate that can block an exit is a trap.

    Route the 15:40 flatten through Gate 11 and it would refuse to flatten at
    15:40; route it through Gate 3 and a bad day would trap the book that made
    the day bad. Closes are unconditional.
    """
    from vigil.execution.router import submit_close

    client = FakeTradingClient()
    order = submit_close(_open_spread(), Decimal("0.10"), client=client, reason="time_stop")
    assert order is not None
    assert len(client.submitted) == 1


def test_a_close_defaults_to_day_not_gtc() -> None:
    """The resting *profit target* is GTC because it must survive worker death.
    A management close is immediate — leaving it working overnight would reopen
    tomorrow's session with a stale order against a position already gone."""
    from vigil.execution.router import submit_close

    client = FakeTradingClient()
    submit_close(_open_spread(), Decimal("0.10"), client=client, reason="breach")
    assert client.submitted[0].time_in_force.value == "day"


def test_closes_carry_unique_client_order_ids() -> None:
    """Hard rule #9: a sweep that retries after a timeout must not double-close."""
    from vigil.execution.router import submit_close

    client = FakeTradingClient()
    for _ in range(3):
        submit_close(_open_spread(), Decimal("0.10"), client=client, reason="breach")
    ids = [r.client_order_id for r in client.submitted]
    assert len(set(ids)) == 3
    assert all(i.startswith("vigil-cls-breach-") for i in ids)


def test_a_structure_with_no_leg_symbols_refuses_to_close_loudly() -> None:
    """Reconciliation must populate legs from the broker. Silently skipping a
    structure we cannot close would leave it unmanaged and unreported."""
    from dataclasses import replace as dc_replace

    from vigil.execution.mleg import MlegConstructionError
    from vigil.execution.router import submit_close

    client = FakeTradingClient()
    with pytest.raises(MlegConstructionError, match="no leg symbols"):
        submit_close(dc_replace(_open_spread(), legs=()), Decimal("0.10"),
                     client=client, reason="breach")
    assert client.submitted == []
