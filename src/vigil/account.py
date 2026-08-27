"""The account lock (hard rule #7). **A mismatch refuses to trade, it does not warn.**

The failure this exists to prevent is mundane and expensive: two paper accounts in
one browser, the wrong key pair pasted into `.env`, and an agent that spends the
competition trading an account nobody is scoring — or worse, one already carrying
positions a human put there.

`config/account.lock` holds the Alpaca account id the submission declares. Startup
compares it against the id the live credentials actually resolve to, and refuses
on any disagreement.

**Fail closed on a *missing* lock, too.** The obvious alternative — "no lock file
means no constraint" — makes the guard vanish in exactly the situation it is for:
a fresh checkout on a new machine, where `.env` is most likely to point somewhere
unintended. A guard that is silently absent when unconfigured is not a guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vigil.settings import REPO_ROOT

LOCK_PATH: Path = REPO_ROOT / "config" / "account.lock"


class AccountLockError(RuntimeError):
    """The live account is not the account this repository is locked to."""


@dataclass(frozen=True, slots=True)
class AccountIdentity:
    """Who we are actually connected to, as the guard sees it."""

    account_id: str
    equity: str
    status: str


def read_lock(path: Path | None = None) -> str | None:
    """The locked account id, or `None` if no lock has been written.

    Comment lines (`#`) and blank lines are ignored so the file can explain
    itself — this is a file a human reads once, in a hurry, when something is
    wrong, and an unadorned UUID tells them nothing.
    """
    p = path or LOCK_PATH
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            return stripped
    return None


def write_lock(account_id: str, *, path: Path | None = None, equity: str = "") -> Path:
    """Write the lock. **Bootstrap only** — never called from a trading path.

    Deliberately refuses to overwrite an existing lock. Re-locking is a decision
    with consequences (it is how the agent would be pointed at a different
    account mid-competition), so it requires deleting the file by hand — an
    action nobody performs by accident.
    """
    p = path or LOCK_PATH
    existing = read_lock(p)
    if existing is not None and existing != account_id:
        raise AccountLockError(
            f"{p} is already locked to {existing}. Refusing to re-lock to "
            f"{account_id}. Delete the file by hand if this is deliberate."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "# Vigil is locked to this Alpaca paper account (hard rule #7).\n"
        "# Startup compares this id against the live credentials in .env and\n"
        "# refuses to trade on any mismatch. Not a secret: the submission\n"
        "# publishes this id.\n"
        f"{account_id}\n"
        + (f"# equity at lock time: {equity}\n" if equity else "")
    )
    return p


def assert_locked(identity: AccountIdentity, *, path: Path | None = None) -> str:
    """Compare a live identity against the lock. Returns the id on success.

    Takes an `AccountIdentity` rather than a client for the same reason the risk
    kernel takes inert data: the comparison is the part worth testing, and it
    should be testable without a network call or a mocked SDK object graph.
    """
    locked = read_lock(path)
    p = path or LOCK_PATH
    if locked is None:
        raise AccountLockError(
            f"No account lock at {p}. Vigil refuses to trade an unlocked account "
            f"(hard rule #7). The live credentials resolve to {identity.account_id} "
            f"(equity {identity.equity}) — if that is the intended account, run "
            f"`make lock` to write it."
        )
    if locked != identity.account_id:
        raise AccountLockError(
            f"ACCOUNT MISMATCH. Locked to {locked}, but the credentials in .env "
            f"resolve to {identity.account_id}. Refusing to trade. Check "
            f"ALPACA_API_KEY_ID, or delete {p} and re-lock if the account "
            f"genuinely changed."
        )
    return locked


def live_identity(client: object | None = None) -> AccountIdentity:
    """Read the account the current credentials actually reach.

    The import is local so that `vigil.account` stays importable — and the pure
    functions above stay testable — on a machine with no credentials at all.
    """
    if client is None:
        from vigil.data.alpaca_client import trading_client

        client = trading_client()
    acct = client.get_account()  # type: ignore[attr-defined]
    return AccountIdentity(
        account_id=str(acct.id),
        equity=str(acct.equity),
        status=str(getattr(acct.status, "value", acct.status)),
    )


def verify_account(*, client: object | None = None, path: Path | None = None) -> str:
    """The startup call: read the live account, assert it matches the lock."""
    return assert_locked(live_identity(client), path=path)
