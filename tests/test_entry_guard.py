"""The entry-path stacking guard (`run_entry`'s one-per-(underlying, expiry) rule).

A millisecond tripwire for a two-line guard whose *absence* is catastrophic and
silent: two structures on one underlying+expiry merge — at the clearinghouse and in
`reconcile.group_positions` — into a single un-closable 8-leg book that strands into
auto-exercise. The frictional soak (`make soak`) proves this across hundreds of
randomised days, but a 5-minute macro test is the wrong thing to rely on to catch a
contributor "optimising away" a guard they didn't understand. These four assertions
run without a broker, a chain or a database, and they document *why* the guard exists
so it survives future edits.

Pure by design: `_stacks_on_open_structure` is the exact predicate `run_entry` calls,
so testing it here tests the production decision, not a paraphrase of it.
"""

from __future__ import annotations

from datetime import timedelta

from tests.conftest import DEFAULT_EXPIRY
from vigil.worker.sessions import _stacks_on_open_structure


def test_a_second_structure_on_the_same_underlying_and_expiry_is_refused(iron_condor):
    """The whole point: SPY is already occupied at this expiry, so a second SPY
    structure at the same expiry would merge into an 8-leg book Alpaca's 4-leg
    close ticket cannot unwind. It must be refused."""
    occupied = {("SPY", DEFAULT_EXPIRY)}
    assert _stacks_on_open_structure(iron_condor, occupied) is True


def test_a_free_underlying_at_the_same_expiry_is_allowed(iron_condor):
    """The guard is surgical, not a blanket freeze: holding QQQ at this expiry does
    not block a SPY entry, because the two never net into one broker position. This
    is why the live sim still fills a QQQ condor in the same cycle SPY is skipped."""
    occupied = {("QQQ", DEFAULT_EXPIRY)}
    assert _stacks_on_open_structure(iron_condor, occupied) is False


def test_the_same_underlying_at_a_different_expiry_is_allowed(iron_condor):
    """Different expiries are different positions at the broker — a SPY structure
    expiring tomorrow does not pool with one expiring today, so it is allowed. The
    key is the *pair*, never the underlying alone."""
    occupied = {("SPY", DEFAULT_EXPIRY + timedelta(days=1))}
    assert _stacks_on_open_structure(iron_condor, occupied) is False


def test_an_empty_book_stacks_on_nothing(put_credit_spread):
    """The common case: a flat book occupies no pair, so the first entry of the day
    is never blocked."""
    assert _stacks_on_open_structure(put_credit_spread, set()) is False
