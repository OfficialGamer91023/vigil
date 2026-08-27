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

**Planned (see roadmap below):**
- **AI Agent:** `openai` SDK against the **Responses API**, model `gpt-5.5`, strict `json_schema` structured outputs, `reasoning.effort` as the cost dial · Alpaca MCP attached read-only
- **API / Cache:** `FastAPI` (read + control only, never trades) · `Redis` (chain-snapshot cache, SSE fanout) · `arq` (queued slow LLM jobs only — *not* the trading loop)
- **Frontend:** a server-rendered page for the demo URL; a `Next.js` desk terminal is post-hackathon polish

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
| `agent/` — LLM portfolio manager | ⬜ (deterministic fallback is live) |
| `api/` — FastAPI read + control routes, SSE | ⬜ |
| Redis chain cache · arq LLM queue | ⬜ |
| `journal/` — daily report, social draft | ⬜ |
| Docker Compose · CI | ⬜ |

Tests: 285 passing, `ruff` clean, `mypy --strict` clean, schema in sync with models.

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

make flatten              # emergency: guarded cancel-all + close-all
```

Until `config/account.lock` exists every cycle refuses to start, naming the account
the current `.env` actually resolves to. That is hard rule #7 working, not a setup bug.

## Licence

MIT — see [LICENSE](LICENSE).
