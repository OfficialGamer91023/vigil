"""The account lock (hard rule #7).

Every test here uses `tmp_path` rather than the real `config/account.lock`: a
test that wrote to the repo's lock file could silently unlock the live account,
which is precisely the failure the module exists to prevent.
"""

from __future__ import annotations

import pytest

from vigil.account import (
    AccountIdentity,
    AccountLockError,
    assert_locked,
    read_lock,
    write_lock,
)

IDENTITY = AccountIdentity(account_id="acct-abc-123", equity="100000", status="ACTIVE")


def test_write_then_read_round_trips(tmp_path):
    path = tmp_path / "account.lock"
    write_lock("acct-abc-123", path=path, equity="100000")
    assert read_lock(path) == "acct-abc-123"


def test_comments_and_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "account.lock"
    path.write_text("# a comment\n\n   \nacct-abc-123\n# trailing note\n")
    assert read_lock(path) == "acct-abc-123"


def test_matching_account_passes(tmp_path):
    path = tmp_path / "account.lock"
    write_lock("acct-abc-123", path=path)
    assert assert_locked(IDENTITY, path=path) == "acct-abc-123"


def test_mismatch_refuses_rather_than_warns(tmp_path):
    """Hard rule #7 in one test: *refuse* to trade, not warn."""
    path = tmp_path / "account.lock"
    write_lock("acct-someone-elses", path=path)
    with pytest.raises(AccountLockError, match="ACCOUNT MISMATCH"):
        assert_locked(IDENTITY, path=path)


def test_missing_lock_refuses_too(tmp_path):
    """Fail closed. A guard that vanishes when unconfigured is not a guard.

    This is the case that actually happens — a fresh checkout on a new machine,
    where `.env` is most likely to point somewhere unintended.
    """
    with pytest.raises(AccountLockError, match="No account lock"):
        assert_locked(IDENTITY, path=tmp_path / "nope.lock")


def test_relock_to_a_different_account_is_refused(tmp_path):
    path = tmp_path / "account.lock"
    write_lock("acct-abc-123", path=path)
    with pytest.raises(AccountLockError, match="already locked"):
        write_lock("acct-different", path=path)
    # And the original survives the attempt.
    assert read_lock(path) == "acct-abc-123"


def test_relock_to_the_same_account_is_idempotent(tmp_path):
    path = tmp_path / "account.lock"
    write_lock("acct-abc-123", path=path)
    write_lock("acct-abc-123", path=path)
    assert read_lock(path) == "acct-abc-123"
