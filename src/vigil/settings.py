"""Environment loading and the paper-trading guard.

This module is deliberately the *only* place that reads Alpaca credentials out of
the environment, so the paper-account assertion cannot be bypassed by importing
something else. Hard rule #1: no code path may reach a live endpoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Walk up from src/vigil/settings.py to the repo root so scripts work regardless
# of the directory they are invoked from.
REPO_ROOT = Path(__file__).resolve().parents[2]


class LiveTradingRefused(RuntimeError):
    """Raised when the environment does not unambiguously say 'paper'."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable snapshot of the environment.

    `frozen=True` means nothing downstream can reassign `paper` after the guard has
    run — the check below would otherwise be advisory rather than binding.
    `slots=True` blocks adding new attributes at runtime for the same reason.
    """

    api_key: str
    api_secret: str
    paper: bool

    @property
    def trading_base_url(self) -> str:
        # Stated explicitly rather than relying on an SDK default, so that the
        # live host never appears anywhere in this codebase.
        return "https://paper-api.alpaca.markets"


def load_settings(*, require_credentials: bool = True) -> Settings:
    """Read .env, assert paper mode, and return the settings.

    `require_credentials` exists so offline unit tests can import this module
    without a populated .env; it never relaxes the paper assertion.
    """
    load_dotenv(REPO_ROOT / ".env")

    # Exact string match, not a truthiness test. "false", "0", "" and an unset
    # variable must all refuse — only the literal "true" proceeds. A permissive
    # parse here is precisely how a guard like this fails open.
    raw = os.getenv("ALPACA_PAPER_TRADE", "").strip().lower()
    if raw != "true":
        raise LiveTradingRefused(
            f"ALPACA_PAPER_TRADE must be exactly 'true' (got {raw!r}). "
            "Refusing to construct a client."
        )

    # There is no supported live path, so the presence of a live flag is treated
    # as a misconfiguration to refuse, not a mode to honour.
    if os.getenv("ALPACA_LIVE_TRADE"):
        raise LiveTradingRefused("ALPACA_LIVE_TRADE is set. This codebase is paper-only.")

    key = os.getenv("ALPACA_API_KEY_ID", "")
    secret = os.getenv("ALPACA_API_SECRET_KEY", "")
    if require_credentials and not (key and secret):
        raise RuntimeError(
            "ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY are missing. "
            "Copy .env.example to .env and fill in your paper keys."
        )

    return Settings(api_key=key, api_secret=secret, paper=True)
