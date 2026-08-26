# Vigil

An autonomous options trading agent built for the **lablab.ai × Alpaca AI Trading Agents Hackathon** (28 Aug – 4 Sept 2026). It trades defined-risk, short-dated option structures on liquid ETFs in an Alpaca paper account, journals every decision, and serves a live desk terminal.

**The core thesis:** A deterministic risk kernel decides what is *allowed*; an LLM portfolio manager decides what is *chosen* from that allowed set. No model output ever reaches the broker without passing through strict, hardcoded risk gates.

## Detailed Overview

Vigil is designed to harvest premium by selling short-dated (0-2 DTE) options on highly liquid ETFs (primarily SPY and QQQ). It uses a sophisticated architecture splitting responsibilities between deterministic code and an AI model:

### 1. The Strategy Engine & Regime Router
The agent does not rely on the LLM to perform mathematical or financial computations. A deterministic "regime router" computes a state vector from underlying market data (trend, realized volatility, implied volatility, intraday structure). Based on the regime (CHOP, TREND-UP, TREND-DOWN, STRESS, CHEAP-VOL), it defaults to a specific structure like an Iron Condor or a Credit Spread. 

### 2. The AI Portfolio Manager (LLM)
Once the engine generates valid candidate structures, the LLM acts as the Portfolio Manager. Equipped with the Alpaca MCP Server, it can investigate the options chain, check news, and review current account health. It then selects 0-1 structures from the pre-validated candidates, outputting its decision and a structured rationale memo.

### 3. The Risk Kernel
Before any order is sent to Alpaca, it must pass 12 deterministic risk gates. This kernel is strictly separated from the LLM. It enforces:
- Defined risk (no naked shorts)
- Max loss per trade (≤1.0% of equity)
- Daily loss limits (≤ -3%)
- Maximum open structures and concentration limits
- Liquidity checks and minimum credit quality
- No trading during specific event blackouts

### 4. Execution & Position Management
Vigil utilizes Alpaca's multi-leg (`mleg`) order support to enter complex positions in a single ticket, guaranteeing defined risk at the broker level. The agent runs on a 15-minute scheduled cycle. On every run, before looking for new entries, it actively manages existing positions: taking profits at predetermined targets, enforcing time stops, and executing a mandatory end-of-day flatten to eliminate overnight risk.

## Tech Stack

- **Core & Data:** `Python 3.12` · `alpaca-py` (Trading API) · Alpaca CLI · Alpaca MCP
- **AI Agent:** Anthropic `openai` SDK (Claude 3.5 Sonnet/Opus) · `Pydantic` for structured schemas
- **Storage & Messaging:** `Postgres 16 + pgvector` · `SQLAlchemy 2.0 async` · `Alembic` · `Redis` · `arq` (worker queue)
- **Frontend / Terminal:** `Next.js 15` · `FastAPI` (backend API)
- **Infrastructure:** `Docker Compose` · `GitHub Actions`

## Non-negotiables

- **Paper trading only:** Uses a dedicated Alpaca paper account funded with exactly $100,000.
- **Defined risk:** Every position has a strictly defined maximum loss. No naked short legs are permitted.
- **Deterministic gating:** Every order must pass the risk kernel. There is only one submit path, utilizing limit orders with price ladders.
- **Resting limits:** Every open structure carries a live resting profit-target order at the broker, rather than polling for targets.
- **API separation:** The API service never trades; the worker agent runs autonomously even if the frontend is down.
- **Emergency paths:** `POST /api/control/halt` stops new entries; `/flatten` immediately closes all positions.
