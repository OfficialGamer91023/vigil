#!/usr/bin/env bash
# Emergency flatten (PLAN §2.1): cancel every open order, close every position.
#
# WHY THIS IS A SCRIPT AND NOT TWO LINES IN THE MAKEFILE
#
# The Alpaca CLI keeps its own credentials and its own paper/live mode in
# ~/.config/alpaca-cli/config.json — it reads neither .env nor ALPACA_PAPER_TRADE.
# So the hard-rule-#1 guard in vigil/settings.py, which refuses to build a client
# unless ALPACA_PAPER_TRADE is exactly "true", protects the SDK path ONLY. The CLI
# ships `config set-mode live`, and a live-configured CLI would happily flatten a
# live account with nothing in this repository objecting.
#
# This script is where that gap is closed: it re-asserts PAPER against the CLI's
# own reported state before running anything mutating. Never invoke the CLI's
# mutating commands directly — go through here.
set -euo pipefail

BIN="${ALPACA_CLI:-alpaca-cli}"
command -v "$BIN" >/dev/null || {
  echo "FATAL: $BIN not found. Install with: uv tool install alpaca-cli" >&2
  exit 127
}

# Credentials come from the environment so the secret is not copied into a second
# file on disk (~/.alpaca.json). .env is the single source (hard rule #2).
#
# **The environment wins over the file, and that precedence is load-bearing.**
# `vigil.settings` reads .env with python-dotenv's `load_dotenv()`, which defaults
# to `override=False` — an already-set variable is left alone. A naive
# `set -a; . ./.env` does the opposite: it overwrites the environment from the
# file, so `ALPACA_PAPER_TRADE=false ./scripts/flatten.sh` would be silently
# rewritten to `true` and the guard below would pass. Two guards enforcing the
# same rule with opposite precedence is worse than one guard.
if [ -f .env ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue ;; esac
    key=${line%%=*}
    case "$key" in *[!A-Za-z0-9_]*|'') continue ;; esac
    [ -n "${!key-}" ] && continue
    export "$key=${line#*=}"
  done < .env
fi
export APCA_API_KEY_ID="${ALPACA_API_KEY_ID:-}"
export APCA_API_SECRET_KEY="${ALPACA_API_SECRET_KEY:-}"

if [ "${ALPACA_PAPER_TRADE:-}" != "true" ]; then
  echo "FATAL: ALPACA_PAPER_TRADE is not exactly 'true'. Refusing." >&2
  exit 1
fi
if [ -n "${ALPACA_LIVE_TRADE:-}" ]; then
  echo "FATAL: ALPACA_LIVE_TRADE is set. This codebase is paper-only." >&2
  exit 1
fi

# The load-bearing check: what does the CLI itself think it is pointed at?
# Strip ANSI escapes and OSC-8 hyperlinks first — the CLI renders Rich output even
# when piped, so a naive grep matches nothing and would fail open.
mode="$("$BIN" config show 2>&1 | perl -pe 's/\e\[[0-9;]*m//g; s/\e\]8;[^\a\e]*(\a|\e\\)//g')"
if ! grep -q "Current Mode: PAPER" <<<"$mode"; then
  echo "FATAL: alpaca-cli is NOT in paper mode. Refusing to flatten." >&2
  echo "  Fix with: $BIN config set-mode paper" >&2
  grep -i "Current Mode\|Endpoint" <<<"$mode" >&2 || true
  exit 1
fi
if ! grep -q "paper-api.alpaca.markets" <<<"$mode"; then
  echo "FATAL: alpaca-cli endpoint is not the paper API. Refusing." >&2
  exit 1
fi

echo "alpaca-cli confirmed PAPER. Flattening."
# Orders first. Closing a position leaves its resting GTC profit target (§2.6)
# orphaned at the broker, and an orphaned close order on a position that no longer
# exists is exactly the reconciliation defect the manage sweep hunts for.
"$BIN" trading orders cancel --all || true
"$BIN" trading positions close --all --cancel-orders
echo "Flatten complete. Verify with: $BIN pos"
