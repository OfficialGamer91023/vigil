"""Write `config/account.lock` from the account the current `.env` reaches.

Run once, on the fresh $100k paper account, before Day 1 trading:

    make lock

Prints the identity first and refuses to overwrite an existing, different lock —
re-pointing the agent at another account is a deliberate act, not a side effect
of re-running a setup script.
"""

from __future__ import annotations

from vigil.account import LOCK_PATH, live_identity, read_lock, write_lock


def main() -> int:
    identity = live_identity()
    existing = read_lock()
    print(f"live account : {identity.account_id}")
    print(f"equity       : {identity.equity}")
    print(f"status       : {identity.status}")

    if existing == identity.account_id:
        print(f"\nalready locked to this account ({LOCK_PATH}). Nothing to do.")
        return 0

    write_lock(identity.account_id, equity=identity.equity)
    print(f"\nlocked -> {LOCK_PATH}")
    print("Startup now refuses to trade any other account (hard rule #7).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
