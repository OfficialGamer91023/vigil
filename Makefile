# Vigil — Day -1 targets only. Compose/migrate/CI arrive Day 0 (PLAN §11).
.PHONY: setup lint test measure a1 a2 vrp smoke dry-run clean

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
measure: a1 a2 vrp

# A1 — are greeks/IV populated on the indicative feed?
a1:
	$(RUN) python scripts/a1_greeks.py

# A2 — does a 0.16-delta short put pay >= 18% of width at 0-2 DTE?
a2:
	$(RUN) python scripts/a2_credit.py

# Backfill the 60-session realized-vol series for the regime router
vrp:
	$(RUN) python scripts/backfill_vrp.py

# Full pipeline against a live chain: sense -> regime -> build -> gate. Submits nothing.
dry-run:
	$(RUN) python scripts/dry_run.py

# Dry run of the mleg probe. Pass --submit yourself to actually place an order.
smoke:
	$(RUN) python scripts/a4_mleg_smoke.py

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache
