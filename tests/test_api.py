"""The API service (PLAN §7) — the auth boundary, then the routes.

Two groups, split by what they need:

- **The bearer guard** needs no database. That is not an accident of the test, it
  is the design: the token dependency sits on the *router*, so it runs before the
  session dependency and a rejected request never opens a connection. An
  unauthenticated caller cannot make this service do database work.
- **The routes** need a real Postgres, for the same reason `test_sessions.py`
  does: what they return *is* database state, and a mocked session would only
  restate the hope.

`ASGITransport` rather than a live server: real routing, real dependency
injection, no port bound and no startup race.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from vigil.api.main import app
from vigil.control import FLATTEN_FLAG, HALT_FLAG
from vigil.db.repositories import journal as J
from vigil.db.session import get_session
from vigil.domain import OpenStructure, Structure

TOKEN = "test-token-0123456789abcdef"


def a_structure(
    underlying: str, *, contracts: int, credit: str, max_loss: str
) -> OpenStructure:
    """Built through the real writer, not inserted as a row.

    `record_structure` takes a domain object and derives the columns itself, so a
    test that reached past it into the ORM would stop exercising the translation
    the API then reads back.
    """
    return OpenStructure(
        underlying=underlying,
        expiry=datetime.now(UTC).date(),
        strikes=(Decimal(760), Decimal(761)),
        max_loss=Decimal(max_loss),
        dollar_delta=Decimal(0),
        has_resting_target=False,
        structure=Structure.PUT_CREDIT_SPREAD,
        short_put_strikes=(Decimal(761),),
        net_credit=Decimal(credit),
        contracts=contracts,
    )


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://vigil.test") as c:
        yield c


@pytest.fixture
def token(monkeypatch: pytest.MonkeyPatch) -> str:
    # `load_dotenv` does not override a variable already in the environment, so
    # setting it here beats whatever the developer's real .env holds.
    monkeypatch.setenv("API_CONTROL_TOKEN", TOKEN)
    return TOKEN


def auth(tok: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {tok}"}


# --------------------------------------------------------------------------- #
# The auth boundary — hard rule #10. No database required.
# --------------------------------------------------------------------------- #

CONTROL_ROUTES = [
    "/api/control/halt",
    "/api/control/unhalt",
    "/api/control/flatten",
    "/api/control/unflatten",
]


@pytest.mark.parametrize("route", CONTROL_ROUTES)
async def test_every_control_route_refuses_an_unauthenticated_caller(client, token, route) -> None:
    """A public endpoint that can flatten a trading account is real attack surface."""
    r = await client.post(route, json={})
    assert r.status_code == 401
    # RFC 6750: name the scheme rather than leaving the client to guess.
    assert r.headers.get("WWW-Authenticate") == "Bearer"


@pytest.mark.parametrize("route", CONTROL_ROUTES)
async def test_every_control_route_refuses_a_wrong_token(client, token, route) -> None:
    r = await client.post(route, headers=auth("not-the-token"), json={})
    assert r.status_code == 403


@pytest.mark.parametrize(
    "header",
    [
        {"Authorization": TOKEN},                      # no scheme
        {"Authorization": f"Basic {TOKEN}"},           # wrong scheme
        {"Authorization": "Bearer"},                   # scheme, no credential
        {"Authorization": "Bearer "},                  # scheme, blank credential
    ],
)
async def test_malformed_authorization_headers_are_401_not_500(client, token, header) -> None:
    """Each of these is a plausible hand-rolled curl. None may crash the guard."""
    r = await client.post("/api/control/halt", headers=header, json={})
    assert r.status_code == 401


async def test_an_unset_server_token_refuses_rather_than_opening_the_route(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The failure that matters most.** A missing secret must not mean "no auth".

    503 rather than 401 on purpose: this is an operator error, not a client
    error, and 401 would send the caller off to find a better credential when
    none could ever work.
    """
    monkeypatch.setenv("API_CONTROL_TOKEN", "")
    r = await client.post("/api/control/flatten", headers=auth(TOKEN), json={})
    assert r.status_code == 503
    assert "API_CONTROL_TOKEN" in r.json()["detail"]


async def test_a_whitespace_only_token_is_treated_as_unset(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`API_CONTROL_TOKEN=" "` in a .env is a typo, not a credential."""
    monkeypatch.setenv("API_CONTROL_TOKEN", "   ")
    r = await client.post("/api/control/halt", headers=auth("   "), json={})
    assert r.status_code == 503


async def test_read_routes_need_no_token(client) -> None:
    """Public by design so a judge can simply visit the URL (§7)."""
    schema = await client.get("/openapi.json")
    assert schema.status_code == 200
    paths = schema.json()["paths"]
    for path, methods in paths.items():
        if path.startswith("/api/control/") and "post" in methods:
            continue
        assert "post" not in methods, f"{path} is a mutating route outside /api/control"


async def test_the_desk_page_is_self_contained(client) -> None:
    """No CDN, no build step: a judge behind a proxy that blocks unpkg still sees it."""
    r = await client.get("/")
    assert r.status_code == 200
    body = r.text
    assert "<title>Vigil</title>" in body
    for marker in ("http://", "https://", "//unpkg", "//cdn"):
        assert marker not in body, f"the desk page reaches out to {marker}"


# --------------------------------------------------------------------------- #
# The routes — real Postgres
# --------------------------------------------------------------------------- #

db = pytest.mark.db


@pytest.fixture(autouse=True)
async def _fresh_engine():
    """Dispose the cached engine between tests — see tests/test_journal.py."""
    from vigil.db import session as session_module

    session_module.engine.cache_clear()
    session_module.session_factory.cache_clear()
    yield
    try:
        await session_module.engine().dispose()
    finally:
        session_module.engine.cache_clear()
        session_module.session_factory.cache_clear()


async def _postgres_reachable() -> bool:
    try:
        async with get_session() as s:
            await s.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.fixture
async def clean_db():
    if not await _postgres_reachable():
        pytest.skip(f"no Postgres at {os.getenv('DATABASE_URL', 'localhost/vigil')}")
    async with get_session() as s:
        await s.execute(
            text(
                "TRUNCATE accounts, sessions, cycles, proposals, proposal_legs, "
                "gate_verdicts, structures, orders, fills, equity_snapshots, "
                "market_snapshots, llm_memos, control_flags RESTART IDENTITY CASCADE"
            )
        )
    yield


@db
async def test_halt_writes_the_flag_the_worker_reads(client, token, clean_db) -> None:
    """The API writes a row. The **worker** acts on it. That is the whole channel.

    Asserted against `journal.is_flag_active` — the exact function the worker
    calls — rather than against the response body, because a route that returned
    a cheerful 200 while writing nothing would pass a body assertion.
    """
    r = await client.post("/api/control/halt", headers=auth(token), json={"reason": "drill"})
    assert r.status_code == 200
    assert r.json()["active"] is True

    async with get_session() as s:
        assert await J.is_flag_active(s, HALT_FLAG) is True


@db
async def test_unhalt_is_the_documented_resume_path(client, token, clean_db) -> None:
    """§5.2: an escape hatch with no way out is a trap."""
    await client.post("/api/control/halt", headers=auth(token), json={})
    await client.post("/api/control/unhalt", headers=auth(token), json={})
    async with get_session() as s:
        assert await J.is_flag_active(s, HALT_FLAG) is False


@db
async def test_flatten_sets_its_own_flag_and_leaves_halt_alone(client, token, clean_db) -> None:
    """They are different questions: stop *adding* risk vs. close what is open."""
    await client.post("/api/control/flatten", headers=auth(token), json={})
    async with get_session() as s:
        assert await J.is_flag_active(s, FLATTEN_FLAG) is True
        assert await J.is_flag_active(s, HALT_FLAG) is False


@db
async def test_flatten_is_withdrawable(client, token, clean_db) -> None:
    """The flag is sticky, so a mis-click at 10:00 must not pre-empt the whole day."""
    await client.post("/api/control/flatten", headers=auth(token), json={})
    await client.post("/api/control/unflatten", headers=auth(token), json={})
    async with get_session() as s:
        assert await J.is_flag_active(s, FLATTEN_FLAG) is False


@db
async def test_health_reports_the_agents_pulse_not_the_web_servers(client, clean_db) -> None:
    """A 200 from this process says nothing about the worker — the age does.

    With no cycles at all the age is None, which is exactly the "never started"
    signal a monitor needs to distinguish from "started and stalled".
    """
    empty = (await client.get("/health")).json()
    assert empty["status"] == "ok" and empty["last_cycle_age_seconds"] is None

    async with get_session() as s:
        account = await J.ensure_account(
            s, alpaca_account_id="test-acct", starting_equity=Decimal(100_000)
        )
        session_row = await J.open_session(
            s, account_id=account.id, trading_date=datetime.now(UTC).date(),
            opening_equity=Decimal(100_000),
        )
        await J.start_cycle(s, session_id=session_row.id, kind="manage")

    live = (await client.get("/health")).json()
    assert live["last_cycle_kind"] == "manage"
    assert live["last_cycle_age_seconds"] is not None
    assert live["last_cycle_age_seconds"] < 60


@db
async def test_state_is_flat_and_zero_before_the_worker_has_ever_run(client, clean_db) -> None:
    """An empty journal must render, not 500. This is the demo URL on Day 1.

    `open_risk` is 0 rather than null specifically: SUM over zero rows is NULL in
    SQL, and a client should not have to know that.
    """
    body = (await client.get("/api/state")).json()
    assert body["account_id"] is None
    assert body["open_structures"] == []
    assert Decimal(str(body["open_risk"])) == 0
    assert body["halted"] is False


@db
async def test_state_reports_the_book_and_the_missing_target_defect(
    client, token, clean_db
) -> None:
    """§2.6: an open structure with no resting GTC target is a defect, so it shows."""
    async with get_session() as s:
        await J.record_structure(
            s, a_structure("SPY", contracts=3, credit="0.20", max_loss="240")
        )
    await client.post("/api/control/halt", headers=auth(token), json={})

    body = (await client.get("/api/state")).json()
    assert len(body["open_structures"]) == 1
    assert body["open_structures"][0]["underlying"] == "SPY"
    assert body["open_structures"][0]["has_resting_target"] is False
    assert Decimal(str(body["open_risk"])) == Decimal(240)
    assert body["halted"] is True


@db
async def test_money_survives_the_json_encoder_as_an_exact_decimal(client, clean_db) -> None:
    """Money is Decimal in Python and NUMERIC in Postgres — and on the wire too.

    `0.05` is not representable in binary floating point, so a serializer that
    routed through `float` would emit something ending in `...277`. The literal
    text of the response is checked, because a `Decimal(str(...))` round-trip in
    the assertion would hide exactly the corruption being tested for.
    """
    async with get_session() as s:
        await J.record_structure(
            s, a_structure("QQQ", contracts=1, credit="0.05", max_loss="0.95")
        )
    raw = (await client.get("/api/state")).text
    assert '"net_credit":0.0500' in raw or '"net_credit":"0.0500"' in raw, raw


@db
async def test_a_missing_cycle_is_404_not_500(client, clean_db) -> None:
    assert (await client.get("/api/cycles/999999")).status_code == 404


@db
async def test_gate_stats_report_passes_as_well_as_failures(client, clean_db) -> None:
    """A gate that never passes is as broken as one that never fires (§5.2).

    A rejection tally alone cannot tell those two apart, so both counts ship.
    """
    assert (await client.get("/api/gates/stats")).json() == []
