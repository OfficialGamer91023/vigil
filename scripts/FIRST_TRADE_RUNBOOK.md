# First-trade runbook

The order to settle the three empirical blockers on the first live session, with the
decision criteria and the cleanup. All commands run from the repo root. Paper only.

Preflight can (and should) run **before** the open; the three probes need market hours.

---

## 0. Preflight — any time before the open

```bash
make preflight
```

Verifies, without live quotes: paper mode, the account lock matches the live account,
host↔broker clock sync, the market schedule (prints next open), the VRP seed, and that
the probe scripts import. **A `GO` (exit 0) means the only remaining variable is the
market.** A `HOLD` names the fix. Advisories (VRP seed, market closed) don't block.

---

## 1. A2 — is the Gate 9 credit floor reachable? *(§4.4.2)*

```bash
make a2                       # uv run python scripts/a2_credit.py [SYMBOL ...]
```

- **GO:** `A2 HOLDS — the 18% floor is reachable at $1 width.` (exit 0)
- **NO-GO:** `A2 FAILS …` (exit 1) → do **not** widen (cr/w falls as width grows). The
  levers are: raise short delta (see the printed sweep) or lower the Gate 9 floor — and
  the floor is not to be tuned from one session (§4.4).
- **Inconclusive:** exit 2, no priced spreads → re-run during market hours.

## 2. A3 / B-1 — does the gate stack fire, and name the binding gate? *(§1.3, §5.2)*

```bash
make dry-run                  # full pipeline: sense → regime → build → gate, no submit
```

- **GO:** `A3: the gate stack APPROVED at least one candidate` (exit 0).
- **NO-GO:** exit 1 → the per-gate output names the binding gate. A gate that *never*
  passes is as broken as one that never fires.

**B-1 specifically** (trend → broken-wing condor must clear the Gate 9 floor) only
builds when the live regime is trending. To confirm it against live quotes without
waiting for a trend day, force it:

```bash
make dry-run ARGS="SPY --regime trend_up"
```

This forces the *build* only (the kernel still judges it; nothing is submitted). Look
for the broken-wing condor clearing gate 9 in the per-gate list. Choices: `chop`,
`trend_up`, `trend_down`, `stress`, `cheap_vol`.

## 3. O-1 — the mleg submit and the net-credit sign convention *(§2.5)*

```bash
make smoke                    # DRY RUN — prices the spread, submits nothing
```

Read the printed spread and net credit. If it looks right and the credit is positive:

```bash
uv run python scripts/a4_mleg_smoke.py --submit
```

This places **one** 1-contract put credit spread and polls the fill. Record the printed
`filled_avg_price` and the sign — `limit_price` is submitted as a **positive net
credit**; confirm the broker agrees. That measured fact is what §2.5's price ladder
depends on.

### Cleanup — mandatory, do not skip

```bash
uv run python scripts/a4_mleg_smoke.py --cancel-all
```

Cancels open orders and reports any remaining position. If one is still open, close it
(the script prints the one-liner). **Never leave a probe position to sit** — a 0DTE leg
left open runs into the 15:40 flatten / auto-exercise rule.

---

### One-line summary

```
make preflight   →   (open)   →   make a2   →   make dry-run   →   make smoke
                                                                   → --submit → --cancel-all
```
