"""mleg ticket construction (§1.1)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from vigil.execution.mleg import MlegConstructionError, build_closing_order, build_entry_order


def test_entry_order_has_no_top_level_side(put_credit_spread) -> None:
    """Verified against alpaca-py: direction lives on the legs, not the ticket."""
    req = build_entry_order(put_credit_spread, Decimal("0.20")).to_request_fields()
    assert req["order_class"] == "mleg"
    assert "side" not in req or req.get("side") is None
    assert len(req["legs"]) == 2


def test_entry_order_opens_both_legs(put_credit_spread) -> None:
    req = build_entry_order(put_credit_spread, Decimal("0.20")).to_request_fields()
    intents = {leg["side"]: leg["position_intent"] for leg in req["legs"]}
    assert intents == {"sell": "sell_to_open", "buy": "buy_to_open"}


def test_closing_order_inverts_every_leg(put_credit_spread) -> None:
    """What we sold to open, we buy to close — and vice versa."""
    req = build_closing_order(
        put_credit_spread, Decimal("0.10"), client_order_id="x-tgt"
    ).to_request_fields()
    intents = {leg["side"]: leg["position_intent"] for leg in req["legs"]}
    assert intents == {"buy": "buy_to_close", "sell": "sell_to_close"}


def test_resting_target_is_gtc_so_it_survives_the_worker_dying(put_credit_spread) -> None:
    """§2.6's whole point: the exit must not depend on our uptime."""
    req = build_closing_order(
        put_credit_spread, Decimal("0.10"), client_order_id="x-tgt"
    ).to_request_fields()
    assert req["time_in_force"] == "gtc"


def test_entry_is_day_so_a_stale_read_does_not_survive_the_session(put_credit_spread) -> None:
    req = build_entry_order(put_credit_spread, Decimal("0.20")).to_request_fields()
    assert req["time_in_force"] == "day"


def test_ladder_price_overrides_the_proposal_price(put_credit_spread) -> None:
    """Re-pricing must not require rebuilding the proposal the kernel approved."""
    req = build_entry_order(put_credit_spread, Decimal("0.18")).to_request_fields()
    assert req["limit_price"] == 0.18


def test_rejects_more_legs_than_mleg_supports(put_credit_spread) -> None:
    from dataclasses import replace

    too_many = replace(put_credit_spread, legs=put_credit_spread.legs * 3)
    with pytest.raises(MlegConstructionError):
        build_entry_order(too_many, Decimal("0.20"))
