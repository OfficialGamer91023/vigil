"""The read/control API (PLAN §7). **This package never trades.**

Hard rule #6, in the plainest possible terms: `src/vigil/api/` may read the
journal and write control flags, and nothing else. It must never place an order,
hold trading state, or sit between the risk kernel and the broker — and the
worker must run correctly with this service, the frontend and Redis all stopped.

That is a structural claim, not a stylistic one, so `tests/test_api_isolation.py`
asserts it by walking this package's import graph: no module here may reach
`alpaca`, `vigil.execution`, `vigil.risk` or `vigil.worker.broker`, transitively.
A rule enforced by a test is a rule; a rule stated in a docstring is a hope.

The clearest expression of it: **this service does not need Alpaca credentials at
all.** It never constructs a client, so it never calls `load_settings()`. An API
container with an empty `.env` still serves the journal.
"""
