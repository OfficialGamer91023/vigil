# Vigil

An autonomous options trading agent for the **lablab.ai × Alpaca AI Trading Agents Hackathon**
(28 Aug – 4 Sept 2026). It trades defined-risk, short-dated option structures on liquid ETFs in an Alpaca
paper account, journals every decision, and serves a live desk terminal.

**The idea in one line:** a deterministic risk kernel decides what is *allowed*; an LLM portfolio manager
decides what is *chosen* from that allowed set. No model output reaches the broker ungated.

## Start here

| Document | What it is |
|---|---|
| **[docs/PRIMER.md](docs/PRIMER.md)** | Options and trading-bot background from zero, plus the reasoning behind every parameter and architecture decision. **Read this first.** |
| **[docs/PLAN.md](docs/PLAN.md)** | Architecture, data model, the 12 risk gates, the stack, 7-day timeline with cut lines |
| **[CLAUDE.md](CLAUDE.md)** | Working rules for Claude Code sessions in this repo |

## Stack

`Python 3.12` · `FastAPI` · `Postgres 16 + pgvector` · `SQLAlchemy 2.0 async` · `Alembic` · `Redis` ·
`arq` · `openai` (GPT-5.5) · `alpaca-py` + Alpaca CLI + MCP · `Next.js 15` · `Docker Compose` ·
`GitHub Actions`

Five processes: an **arq worker** that is the agent, a **FastAPI** service that reads the journal and
streams it, **Postgres** as the record, **Redis** as cache and bus, and a **Next.js** desk terminal.
See [PLAN.md §10](docs/PLAN.md) for why each piece is present — and **§12 for what was deliberately
rejected**, which is the more useful list.

## Status

Planning complete and **critically reviewed** — the review found three issues serious enough to change the
strategy, and they are documented in place rather than quietly patched:

1. **The exit policy had negative expected value.** A 50% profit target paired with a 2× credit stop needs
   an 80% win rate; touch probability at a 0.16-delta short strike is ~32%. The mark-based stop is deleted.
   → [PLAN §4.4.1](docs/PLAN.md), [PRIMER §2.2](docs/PRIMER.md)
2. **Two headline parameters were jointly infeasible.** Credit ≥25% of width is a 30–45 DTE number; paired
   with a 0–2 DTE mandate it rejects every candidate and the agent never trades. Floor moved to 18%.
   → [PLAN §4.4.2](docs/PLAN.md)
3. **The book was sized for the wrong objective function.** A six-session, rank-scored competition with
   truncated downside has a *convex* payoff; the plan was a variance seller whose best case was ~+0.4%.
   → [PLAN §4.7](docs/PLAN.md), [PRIMER §2.4](docs/PRIMER.md)

### Day −1 (Wed 26 Aug) — measurement and hardening

**A1 holds:** the indicative feed returns populated greeks and IV (99% coverage once expired contracts
are filtered out), so no local Black–Scholes is needed. **A2 is more interesting:** single verticals
price at 8–11% of width conservatively and *fail* the 18% floor, while iron condors clear it at 19–24%
— a condor collects two credits against one width of risk. That points at condors as the default
structure, pending a live re-measurement. Details in [docs/CLI_NOTES.md §5](docs/CLI_NOTES.md).

A pre-implementation audit then found six defects in the paths a live session depends on. All six are
fixed, each with a regression test:

| | Defect | Consequence |
|---|---|---|
| B1 | Every price-ladder rung reused one `client_order_id` | Alpaca rejects the duplicate, so the ladder could never concede past rung 1 |
| B2 | `partially_filled` was treated as "still working" | Cancelled the partial and submitted a second entry ticket, leaving live contracts with no resting exit |
| B3 | CHEAP-VOL routed to a structure with no builder | Fell through to a **call credit spread** — sold volatility on the session identified as cheap |
| B4 | Gate 1 took `width` on trust | A $5-wide spread declaring $1 reported $82 max loss instead of $482 and **passed all twelve gates** |
| B5 | Gate 11 read wall-clock fields off the caller's timezone | The same instant passed or failed depending on the zone attached; a UTC container would permit pre-market entry |
| B6 | Open-interest fetch read one page | Truncation looked identical to genuinely illiquid contracts, and Gate 8 rejected for a reason that was not true |

The convexity sleeve ([PLAN §4.7](docs/PLAN.md)) now exists as a real debit-spread builder with its own
ladder — it is the only convex payoff in the book, and the plan's whole argument against finishing at
+0.4% rests on it.

**Still open:** the sign convention of `limit_price` on a net-credit mleg package (A4) is unverified and
five components now share the assumption; and A3 has been probed at a single point rather than replayed
across 60 sessions.

## Non-negotiables

- Paper trading only, on a **fresh** dedicated account funded to $100,000
- Defined risk on every position — no naked short legs
- Every order passes the risk kernel; one submit path; always limit orders
- Every open structure carries a **live resting profit-target order at the broker**, not a polled target
- The API service never trades; the worker runs correctly with everything else stopped
- `POST /api/control/halt` stops new entries; `/flatten` closes everything
