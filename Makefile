# Vigil — Day -1 targets only. Compose/migrate/CI arrive Day 0 (PLAN §11).
.PHONY: setup lint test measure a1 a2 a3 vrp smoke dry-run worker cycle flatten cli db migrate lock \
	up down logs ps build stack-migrate clean

# --- The macOS .pth trap ----------------------------------------------------
# uv marks .venv hidden on macOS, and CPython's site.addpackage SILENTLY SKIPS
# hidden .pth files. That makes the editable install of `vigil` a no-op, and
# `import vigil` fails with no explanation. Clearing the flag works until the next
# uv invocation re-applies it, so nothing here depends on the .pth: PYTHONPATH puts
# src on the path directly. pyproject's [tool.pytest.ini_options] does the same for
# pytest. --no-sync keeps uv from rebuilding on every source edit.
# Re-run `make setup` after changing dependencies.
RUN = PYTHONPATH=src uv run --no-sync

setup:
	uv sync
	@$(RUN) python -c "import vigil; print('vigil importable:', vigil.__file__)"

lint:
	$(RUN) ruff check . && $(RUN) mypy src

test:
	$(RUN) pytest -q

# --- Day -1: measure the three load-bearing assumptions (PLAN §1.3) ----------
measure: a1 a2 vrp a3

# A1 — are greeks/IV populated on the indicative feed?
a1:
	$(RUN) python scripts/a1_greeks.py

# A2 — does a 0.16-delta short put pay >= 18% of width at 0-2 DTE?
a2:
	$(RUN) python scripts/a2_credit.py

# Backfill the 60-session realized-vol series for the regime router
vrp:
	$(RUN) python scripts/backfill_vrp.py

# A3 - does the gate stack ever fire? Replays 60 sessions of the regime router.
a3:
	$(RUN) python scripts/a3_replay.py

# Full pipeline against a live chain: sense -> regime -> build -> gate. Submits nothing.
dry-run:
	$(RUN) python scripts/dry_run.py

# Dry run of the mleg probe. Pass --submit yourself to actually place an order.
smoke:
	$(RUN) python scripts/a4_mleg_smoke.py

# --- The agent (PLAN §2.3) --------------------------------------------------
# The in-process scheduler. No Redis, no API, no web — hard rule #6 says the
# worker must run correctly with all three stopped, and this is that claim being
# true rather than asserted.
worker:
	$(RUN) python -m vigil.worker.runner

# One cycle by name, for driving the agent by hand:
#   make cycle C=premarket | open | manage | entry | flatten | postclose
cycle:
	$(RUN) python -m vigil.worker.sessions $(C)

# --- Docker Compose (PLAN §9) ------------------------------------------------
# postgres + redis + migrate + worker. `api` joins on Day 3.
#
# The compose Postgres publishes on **5433**, not 5432, so it coexists with a
# local Postgres rather than fighting it for the port. Host tooling that should
# talk to the *containerised* database needs:
#   DATABASE_URL=postgresql+asyncpg://vigil:vigil@localhost:5433/vigil
COMPOSE = docker compose

build:
	$(COMPOSE) build

# Brings up the stack. `migrate` runs to completion first (compose waits on
# service_completed_successfully), so the worker never races a half-applied schema.
up:
	$(COMPOSE) up -d
	@$(COMPOSE) ps

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

# Migrations against the containerised database, run inside the image so the
# alembic version matches the one the worker imports.
stack-migrate:
	$(COMPOSE) run --rm migrate

# --- Journal (PLAN §3) ------------------------------------------------------
# Creates the database if it is missing. Migrations never run create_all()
# against a real database (CLAUDE.md) — the schema only ever moves via Alembic.
db:
	@psql -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='vigil'" | grep -q 1 \
		|| createdb vigil
	@echo "database ready: vigil"

migrate: db
	$(RUN) alembic upgrade head

# --- The account lock (hard rule #7) ----------------------------------------
# Run ONCE on the fresh paper account, before Day 1. Startup refuses to trade any
# account whose id does not match config/account.lock.
lock:
	$(RUN) python scripts/lock_account.py

# --- Ops (PLAN §2.1) --------------------------------------------------------
# The CLI is a separate tool, not a project dependency: alpaca-cli requires
# Python >= 3.14 while this project is pinned to 3.12 (docs/CLI_NOTES.md §2).
cli:
	uv tool install alpaca-cli
	alpaca-cli --version

# Emergency flatten. Goes through the script, never the CLI directly — the CLI
# keeps its own paper/live mode that vigil/settings.py cannot see, and the script
# is where that gap is closed. See docs/CLI_NOTES.md §2.
flatten:
	./scripts/flatten.sh

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache
