# Vigil — Submission

**lablab.ai × Alpaca AI Trading Agents Hackathon · 28 Aug – 4 Sept 2026**

An autonomous options desk. A deterministic risk kernel decides what is *allowed*;
the strategy layer decides what is *chosen* from that allowed set. It trades defined-risk,
short-dated option structures on liquid ETFs in an Alpaca paper account, journals every
decision to Postgres, and serves a live desk terminal.

---

## Alpaca paper account

| | |
|---|---|
| **Account ID** | `8603b2b8-c0a9-4fe8-8e5e-910d489f6132` |
| **Starting balance** | **$100,000.00** |
| Account opened | 2026-08-25 |
| Status | ACTIVE · options level 3 · paper only |

This id is not a secret — the submission publishes it. It is also written to
`config/account.lock`, and **startup refuses to trade any account whose id does not
match** (hard rule #7). A mismatch raises; it does not warn.

> Verify with `make lock` output, or `alpaca-cli status`.

---

## Requirement checklist

| # | Requirement | Status |
|---|---|---|
| 1 | Autonomous AI trading agent on Alpaca's Trading API | ✅ `src/vigil/worker/` — in-process scheduler, six session cycles |
| 2 | Uses Alpaca's **MCP server or CLI** | ✅ **CLI** — `alpaca-cli` for ops, `scripts/flatten.sh`, surface recorded in `docs/CLI_NOTES.md` |
| 3 | All strategies incorporate options | ✅ multi-leg (`mleg`) verticals, iron condors, long strangles — no equity legs anywhere |
| 4 | Brand-new paper account, $100,000 | ✅ see above |
| 5 | One-page write-up: AI logic · risk gates · Alpaca infrastructure | ⬜ `WRITEUP.md` |
| 6 | Paper account ID submitted | ✅ this file |
| 7 | Public repo · video · slides · cover image · demo URL | repo ✅ · rest ⬜ |

**Repo:** https://github.com/OfficialGamer91023/vigil (MIT)
**Demo URL:** _pending_ — the desk terminal is served at `/` by the API service
**Video · slides · cover image:** _pending_
**Social posts:** _pending_ — tag `@lablabai` and `@AlpacaHQ`

---

## The six scored sessions

Confirmed against Alpaca's own calendar endpoint — Labor Day (Mon 7 Sep) falls after the
window, so nothing interrupts it.

| # | Date | |
|---|---|---|
| 1 | Fri 28 Aug 2026 | |
| 2 | Mon 31 Aug 2026 | |
| 3 | Tue 1 Sep 2026 | |
| 4 | Wed 2 Sep 2026 | risk kernel frozen after today |
| 5 | Thu 3 Sep 2026 | |
| 6 | Fri 4 Sep 2026 | no new risk after 12:00 ET · **flat by 15:40 ET** |

The book is flat at the end of session six on purpose: the judged number should be clean
realized cash, not a mark-to-model on open positions that could mutate via auto-exercise
after submission.

---

## What is defensible about it

- **The LLM proposes, the risk kernel disposes.** No model output reaches the broker
  without passing all twelve gates in `src/vigil/risk/`. The gates are pure functions of
  `(proposal, portfolio_state, config, context)` — no network, no I/O, impossible to
  talk out of a decision mid-cycle.
- **Every verdict is journalled, passes included.** The kernel deliberately does not
  short-circuit on the first failure, because *"did any of this ever actually fire?"* has
  to be answerable from the record rather than from memory. `GET /api/cycles/{id}` returns
  every proposal with all twelve verdicts and the reason each rejection gave, and the desk
  page renders it.
- **One submit path.** `execution/router.py` is the only code that reaches Alpaca's order
  API, and it calls the kernel first. Anything else touching it is a bug.
- **Defined risk only.** Every structure has a computable, finite max loss. No naked short
  legs — and Gate 1 re-derives the width from the leg strikes rather than believing the
  number a proposal declares.
- **Limit orders only.** The free-tier options feed is indicative, so a market order on an
  approximate quote is an open invitation.
- **The API never trades.** `tests/test_api_isolation.py` walks the import graph of every
  module under `vigil.api` and fails if any of them can reach the broker, the submit path
  or the kernel. The worker runs correctly with the API, the frontend and Redis all stopped.
- **Idempotency is a database constraint.** `orders.client_order_id` is `UNIQUE NOT NULL`,
  so a retry after a timeout that actually succeeded raises rather than double-filling.
- **A Hypothesis property test** proves no approved proposal can exceed 2% of equity across
  generated inputs, not three hand-picked examples.

---

## Running it

```bash
cp .env.example .env      # paper keys + API_CONTROL_TOKEN
uv sync
make lock                 # writes config/account.lock — required before any cycle
make migrate
make worker               # the agent
make api                  # desk terminal on http://localhost:8000
```

Safety controls, bearer-token guarded from the first commit:

```bash
curl -XPOST -H "Authorization: Bearer $API_CONTROL_TOKEN" $API/api/control/halt
curl -XPOST -H "Authorization: Bearer $API_CONTROL_TOKEN" $API/api/control/flatten
make flatten              # immediate, does not depend on the worker or the API
```

---

## Honest caveats

Kept here because a submission that hides them is worth less than one that does not.

- **Paper fills are optimistic.** They match NBBO, fill limit orders generously and ignore
  available size, and roughly 10% arrive as partials. Partial fills are handled explicitly
  and the journal carries a slippage haircut, but paper P&L still flatters the strategy.
- **Signals come from the underlying, never the option chain.** The free tier serves an
  indicative options feed with 15-minute-delayed trades; option data is used only for
  pricing, greeks and liquidity checks.
- **Index options (SPX/XSP/VIX) are not used.** They are tradeable in paper, but Alpaca
  serves no market data for them, and an agent cannot trade what it cannot quote.
- **The LLM portfolio manager is wired, but the deterministic ranker is the tested
  primary.** `gpt-5.5` (via the openai Responses API, strict structured output) selects at
  most one structure from the menu the risk kernel has *already approved* — so the model can
  only ever narrow a safe set, never widen it, and whatever it picks is re-gated at the
  submit path. Every failure — timeout, error, an index off the end of the menu — falls back
  to the same credit-per-width ranker, and the fallback rate and prompt-cache hit rate are
  journalled to `llm_memos`. The model is never on the path for *closing* a position. It can
  be switched off entirely with one flag (`config/agent.yaml: enabled`), in which case the
  ranker runs and nothing else changes — because the fallback is the whole system with the
  model removed, and it was written and tested first.
