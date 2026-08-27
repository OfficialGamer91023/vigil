"""Structured logging (JSON lines), per CLAUDE.md conventions.

**Why structlog rather than the standard library.** Every log line the agent
emits needs `session_id` and `cycle_id` attached, and the interesting lines are
events with fields — `entry.filled` with a contract count and a rung — not
sentences. With `logging` that means either formatting those fields into a string
(and losing them) or threading a dict through every call site by hand.
`log.bind(cycle_id=...)` returns a logger that carries the context, so a cycle
binds once and everything below it inherits.

**Why JSON lines.** The journal is Postgres; these are the operational record
that sits beside it. A crash at 14:32 is diagnosed by `jq` over a log file, and a
line that has to be regex-parsed to find the cycle it belonged to is a line that
does not get read.

The console renderer is used when stderr is a TTY, because a human watching a
session run should not have to read JSON, and JSON is emitted otherwise — which
is what a container's log driver actually collects.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_configured = False


def configure(*, level: int = logging.INFO, force_json: bool = False) -> None:
    """Idempotent. Safe to call from every entry point, and it is.

    The guard matters: `structlog.configure` replaces global state, and a second
    call from a script that also imported the worker would silently reset the
    processor chain half way through a run.
    """
    global _configured
    if _configured:
        return

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if force_json or not sys.stderr.isatty()
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            # UTC and ISO-8601, matching the journal's `timestamptz` columns.
            # Trading logic reasons in Eastern; storage never does.
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> Any:
    """A bound logger tagged with its module. Configures on first use."""
    configure()
    return structlog.get_logger(name)
