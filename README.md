# Vigil

An autonomous options trading agent built for the **lablab.ai × Alpaca AI Trading Agents Hackathon** (28 Aug – 4 Sept 2026). It trades defined-risk, short-dated option structures on liquid ETFs in an Alpaca paper account, journals every decision, and serves a live desk terminal.

**The core thesis:** A deterministic risk kernel decides what is *allowed*; an LLM portfolio manager decides what is *chosen* from that allowed set. No model output ever reaches the broker without passing through strict, hardcoded risk gates.

## Detailed Overview

Vigil is designed to harvest premium by selling short-dated (0–2 DTE) options on highly liquid ETFs (primarily SPY and QQQ). It uses a sophisticated architecture splitting responsibilities between deterministic code and an AI model:

### 1. The Strategy Engine & Regime Router
The agent does not rely on the LLM to perform mathematical or financial computations. A deterministic "regime router" computes a state vector from underlying market data (trend, realized volatility, implied volatility, intraday structure). Based on the regime (CHOP, TREND-UP, TREND-DOWN, STRESS, CHEAP-VOL), it defaults to a specific structure: an iron condor in chop, a credit spread with the trend, a long strangle when volatility is cheap, and a stand-down under stress.

Variance risk premium is measured as a **rolling percentile**, not as `IV − RV > 0`. The sign test is true on nearly every session, which would make STRESS unreachable and collapse the router to "always sell".

### 2. The AI Portfolio Manager (LLM)
Once the engine generates valid candidate structures, the LLM acts as the Portfolio Manager. Equipped with the Alpaca MCP server (read-only), it can investigate the options chain, check news, and review current account health. It then selects 0–1 structures from the pre-validated candidates, outputting its decision and a structured rationale memo.

Every LLM call has a deterministic fallback, and the model is never on the path for *closing* a position.

### 3. The Risk Kernel
Before any order is sent to Alpaca, it must pass **12 deterministic risk gates** — pure functions with no network access and no LLM involvement. The kernel does not short-circuit: every gate runs on every proposal and every verdict is persisted, passes included, so "did any of this ever actually fire?" is answerable from the table rather than from memory. It enforces:

| # | Gate | Rule |
|---|---|---|
| 1 | Defined risk | Finite, computable max loss; width derived from leg strikes, never taken on trust; no naked shorts |
| 2 | Per-trade risk | Max loss ≤ **2.0%** of equity |
| 3 | Daily loss stop | Day P&L ≤ **−3%** → halt new entries; management keeps running |
| 4 | Total drawdown | Equity ≤ **−8%** from peak → full halt + flatten, human required |
| 5 | Concurrency | ≤ 6 open structures |
| 6 | Concentration | ≤ 2 per underlying; ≤ 40% of open risk in one name |
| 7 | Portfolio delta | `\|Σ delta × 100 × spot\| ≤ 5%` of equity — **dollar** delta, never a bare delta count |
| 8 | Liquidity | Per leg: OI ≥ 100, bid > 0, spread ≤ 10% of mid (or ≤ $0.10 on cheap legs) |
| 9 | Credit quality | Credit ≥ **18%** of width; max-loss/max-profit ≤ 5.5 : 1 |
| 10 | Event blackout | No earnings inside contract life; no entry 5 min before or 15 min after a macro print |
| 11 | Time windows | No entry in the first 15 or last 20 minutes; no 0DTE entry after 14:30 ET |
| 12 | Idempotency & sanity | `client_order_id` unique (a DB constraint); limit within ±20% of mid; whole-number qty ≥ 1 |

Gate 2's invariant is proved by a **Hypothesis property test** across generated inputs, not three hand-picked examples: no approved proposal ever exceeds 2% of equity.

**On 2%, not 1%:** 1% per trade is a career parameter defending against ruin over twenty years. This is six sessions of paper money with truncated downside and rank-based scoring — a tournament, not a career. At 1% the best realistic case for the whole competition is roughly +0.4%.

**On the absence of a stop-loss:** there is deliberately no mark-based stop on credit spreads. A `2× credit` stop at 0–2 DTE needs an ~80% win rate to break even while touch probability at 0.16 delta is ~32%, and max loss is *already* bounded on entry by Gate 2 — so the stop bounded nothing and converted recoverable drawdowns into realized losses. Exits are a resting 50% profit target, a short-strike breach, and the 15:40 time stop.

### 4. Execution & Position Management
Vigil uses Alpaca's multi-leg (`mleg`) order support to enter complex positions in a single ticket, guaranteeing defined risk at the broker level. Orders are **limit only** — the free-tier options feed is indicative, and a market order against an approximate quote is an invitation.

Cycles run on a fixed in-process schedule: `manage` every 15 minutes (tightened to **5 minutes** while a 0DTE structure is open), `entry` every 30 minutes between 10:30 and 14:30 ET, and a hard `flatten` at 15:40. Management always runs before new-entry logic — protecting capital outranks deploying it — and the schedule enforces that structurally rather than leaving it to convention.

## Tech Stack

**Built:**
- **Core & Data:** `Python 3.12` · `uv` · `alpaca-py` (trading + options chain snapshots) · Alpaca CLI (ops only)
- **Storage:** `Postgres 16` · `SQLAlchemy 2.0 async` · `asyncpg` · `Alembic` — money is `Decimal`/`NUMERIC`, never float
- **Validation & Quality:** `Pydantic`-style frozen dataclasses at the kernel boundary · `ruff` · `mypy --strict` · `pytest` · `hypothesis`
- **AI Agent:** `openai` SDK against the **Responses API**, model `gpt-5.5`, strict `json_schema` structured outputs, `reasoning.effort` as the cost dial · deterministic ranker as the tested fallback on every path
- **API / Cache:** `FastAPI` (read + control only, never trades) · a server-rendered desk page · `Redis` (optional chain-snapshot cache — never the record; the worker trades with it stopped)

**Deferred / cut:**
- **arq LLM queue** — the LLM runs inline in the cycle; arq was reserved for slow jobs only and is currently unused (hard rule #6 is why the *trading loop* was never put on it).
- **Alpaca MCP read-only toolsets** — selection reads a menu built into the prompt, not live tools.
- **Next.js desk terminal** — post-hackathon polish; the server-rendered page already satisfies the demo-URL requirement.

Deliberately **rejected**: LangChain/LangGraph, the OpenAI Agents SDK, index options (SPX/XSP — Alpaca serves no market data for them), Celery/RabbitMQ, a backtesting framework, TA-Lib, an options pricing library, market orders, a mark-based stop-loss, a ≤5% token convexity hedge, polling for profit targets, running the loop inside FastAPI, and an unauthenticated control plane. Each rejection is recorded with its reasoning — the list is part of the deliverable.

## Non-negotiables

- **Paper trading only:** a dedicated Alpaca paper account funded with exactly $100,000. `ALPACA_PAPER_TRADE` must be exactly the string `"true"`; a live flag is a misconfiguration to refuse, not a mode to honour. Startup asserts the account ID against `config/account.lock`.
- **Defined risk:** every position has a strictly defined maximum loss. No naked short legs — not even temporarily while legging in.
- **Deterministic gating:** every order passes the risk kernel. There is exactly one submit path (`execution/router.py`), using limit orders with a price ladder.
- **Resting limits:** every open structure carries a live resting GTC profit-target order at the broker, rather than polling for targets. An open structure without one is a reconciliation defect.
- **API separation:** the API service never trades. The worker must run correctly with the API, the frontend and Redis all stopped.
- **Emergency paths:** `POST /api/control/halt` stops new entries; `/flatten` immediately closes all positions. Mutating routes carry a bearer token from the first commit.
- **Idempotency is a database constraint:** `orders.client_order_id` is `UNIQUE NOT NULL`, so a retry raises an integrity error rather than double-filling.

## Status

| Layer | State |
|---|---|
| `settings` · `clock` · `config` (typed YAML) | ✅ |
| `data/` — Alpaca clients, windowed chain fetch, OCC parsing | ✅ |
| `signals/` — indicators, realized vol, regime router | ✅ |
| `strategy/` — vertical, iron condor, debit spread, long strangle, sizing | ✅ |
| `risk/` — all 12 gates + kernel | ✅ (frozen after Wed 2 Sep) |
| `execution/` — mleg, price ladder, router, reconcile, manage | ✅ |
| `db/` — 12 tables, repositories, initial Alembic migration | ✅ |
| `worker/` — pure cron table (`datetime → due cycles`) | ✅ |
| `worker/` — broker adapter, sense step, six session runners, scheduler loop | ✅ |
| `config/account.lock` + startup assertion | ✅ |
| `agent/` — LLM portfolio manager (`gpt-5.5`, strict structured output, deterministic fallback) | ✅ |
| `api/` — FastAPI read + control routes, SSE, desk page | ✅ |
| `data/cache.py` — optional Redis chain-snapshot cache (golden round-trip test) | ✅ |
| `strategy/ladder.py` — escalation ladder, wired into entry sizing | ✅ |
| `clock_guard.py` — refuses to trade on host↔broker clock skew > 60s | ✅ |
| `journal/` — session report (`python -m vigil.journal.report`), social draft | ✅ |
| Docker Compose (`postgres` · `redis` · `migrate` · `worker` · `api`) | ✅ |
| `execution/` — mleg, ladder, router, reconcile, manage, **auto re-rest of a lost §2.6 target** | ✅ |
| First-trade probes (A1/A2/A3, O-1 sign check) — **run live 31 Aug**; agent placed its first kernel-gated trade | ✅ |
| arq LLM queue | ⬜ (deferred — LLM runs inline) |
| CI — GitHub Actions: `ruff` + `mypy` + `alembic check` + `pytest` on a real Postgres | ✅ |

Tests: **511 passing, 1 deliberately skipped**, `ruff` clean, `mypy --strict` clean across 64 source
files, schema in sync with models — enforced on every push by [CI](.github/workflows/ci.yml). A dated
audit of what is verified, still open, and deferred lives in [`docs/AUDIT.md`](docs/AUDIT.md).

## Quick start

```bash
cp .env.example .env      # fill in paper keys
uv sync
make setup                # verify the package imports
make migrate              # create the database and run Alembic
make test                 # pytest
make lint                 # ruff + mypy

make measure              # re-verify the three load-bearing assumptions against a live chain
make dry-run              # full pipeline — sense → regime → build → gate. Submits nothing.

make lock                 # ONCE, on the fresh paper account: write config/account.lock
make worker               # the agent — in-process scheduler, no Redis, no API needed
make cycle C=manage       # one cycle by hand: premarket|open|manage|entry|flatten|postclose

make api                  # the read/control API + desk page on :8000
make flatten              # emergency: guarded cancel-all + close-all

make build                # build the image
make up                   # postgres + redis + migrate + worker, detached
make logs                 # tail the stack
make down                 # stop it
```

The API serves the desk page at `/`, OpenAPI docs at `/docs`, and:

| Route | Auth |
|---|---|
| `GET /health` · `/api/state` · `/api/equity` · `/api/cycles[/{id}]` · `/api/gates/stats` · `/api/market` · `/api/orders` · `/api/stream` | public |
| `POST /api/control/{halt,unhalt,flatten,unflatten}` | **bearer** |

The desk page renders equity and its curve, open risk against the 12% book
ceiling, the per-underlying market read, the book, gate pass rates, the order
log, and a decision feed where **clicking a cycle expands every proposal it
considered with all twelve verdicts and the reason each rejection gave**. It
updates on the SSE stream, with a 30-second poll behind it.

```bash
curl -XPOST -H "Authorization: Bearer $API_CONTROL_TOKEN" localhost:8000/api/control/halt
```

An **unset** `API_CONTROL_TOKEN` makes those routes answer `503`, not `200` — a
missing secret must never read as "no auth required". `tests/test_api_isolation.py`
walks the API's import graph and fails if any module in `vigil/api/` can reach the
broker, the submit path or the kernel, so hard rule #6 is checked rather than
merely documented.

The compose Postgres publishes on **5433**, not 5432, so it coexists with a local
Postgres instead of fighting it for the port. The local one backs `pytest -m db`;
the compose one backs the containerised worker.

`worker` deliberately does **not** `depends_on` redis — hard rule #6 says the
agent runs with Redis stopped, and `docker compose stop redis` is that claim
being demonstrable rather than asserted.

Until `config/account.lock` exists every cycle refuses to start, naming the account
the current `.env` actually resolves to. That is hard rule #7 working, not a setup bug.

## Licence

MIT — see [LICENSE](LICENSE).
