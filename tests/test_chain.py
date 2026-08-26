"""Open-interest fetching. No network: a fake client serves canned pages.

The bug pinned here is a **silent** one, which is the reason it is worth a test
at all. `get_option_contracts` is paged; reading only the first page drops
symbols, a dropped symbol defaults to 0 open interest in the map, and Gate 8 then
rejects the leg for insufficient liquidity. The rejection looks entirely
legitimate in the journal — the agent simply stops trading and every verdict
explains itself convincingly. That is the same silent-and-total failure mode
PLAN §4.4.2 warns about for Gate 9.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest

from vigil.data import chain as chain_mod


@dataclass
class FakeContract:
    symbol: str
    open_interest: int | None


@dataclass
class FakePage:
    option_contracts: list[FakeContract]
    next_page_token: str | None


class FakePagedClient:
    """Serves `pages` in order, honouring the token the caller sends back."""

    def __init__(self, pages: list[FakePage]) -> None:
        self.pages = pages
        self.requests: list[str | None] = []

    def get_option_contracts(self, req: object) -> FakePage:
        token = getattr(req, "page_token", None)
        self.requests.append(token)
        index = 0 if token is None else int(token)
        return self.pages[index]


@pytest.fixture
def patched(monkeypatch):
    def _install(pages: list[FakePage]) -> FakePagedClient:
        client = FakePagedClient(pages)
        monkeypatch.setattr(chain_mod, "trading_client", lambda: client)
        return client

    return _install


def _fetch() -> dict[str, int]:
    return chain_mod.fetch_open_interest("SPY", spot=Decimal(765), max_dte=2)


def test_open_interest_follows_every_page(patched) -> None:
    """Three pages in, three pages out."""
    client = patched([
        FakePage([FakeContract("SPY260828P00760000", 4_000)], next_page_token="1"),
        FakePage([FakeContract("SPY260828P00761000", 5_000)], next_page_token="2"),
        FakePage([FakeContract("SPY260828P00762000", 6_000)], next_page_token=None),
    ])
    oi = _fetch()

    assert len(oi) == 3, "later pages were dropped"
    assert oi["SPY260828P00762000"] == 6_000
    assert client.requests == [None, "1", "2"]


def test_a_single_page_response_still_terminates(patched) -> None:
    """The loop must not depend on a token being present."""
    patched([FakePage([FakeContract("SPY260828P00760000", 4_000)], next_page_token=None)])
    assert _fetch() == {"SPY260828P00760000": 4_000}


def test_an_empty_token_ends_the_walk_rather_than_looping(patched) -> None:
    """Alpaca returns `""` rather than null at the end of some listings, and a
    truthiness check is what keeps that from becoming an infinite loop."""
    client = patched([
        FakePage([FakeContract("SPY260828P00760000", 4_000)], next_page_token=""),
    ])
    assert _fetch() == {"SPY260828P00760000": 4_000}
    assert len(client.requests) == 1


def test_contracts_without_open_interest_are_omitted_not_zeroed(patched) -> None:
    """Omitting lets Gate 8 fail closed on a real absence. Writing an explicit 0
    would look like a measured value."""
    patched([
        FakePage(
            [
                FakeContract("SPY260828P00760000", None),
                FakeContract("SPY260828P00761000", 5_000),
            ],
            next_page_token=None,
        )
    ])
    oi = _fetch()
    assert "SPY260828P00760000" not in oi
    assert oi["SPY260828P00761000"] == 5_000
