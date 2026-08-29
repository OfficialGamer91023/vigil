"""Reporting and social drafts (PLAN §9: `report · social_draft`).

The read-side counterpart to the worker. Nothing here trades, holds trading
state, or reaches the broker — it turns the journal the worker already wrote into
things a human reads: a session report on the terminal, and a build-in-public
post drafted from the day's own record. Both are pure consumers of Postgres, so
this package is safe to run against a stopped worker.

**Deliberately no eager re-exports.** `report.py` is a `python -m` entry point,
and a package `__init__` that imported it would trip runpy's "already in
sys.modules" warning every time the command ran. Import the pieces from their
modules directly — `from vigil.journal.report import build_report`.
"""

from __future__ import annotations
