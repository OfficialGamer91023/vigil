"""The names of the out-of-band control flags (§5.2). Constants, nothing else.

**Why a module of two strings.** The API writes these rows and the worker reads
them, so both need the same vocabulary — but `src/vigil/api/` must not import
`vigil.worker.sessions`, which pulls in the broker and the whole execution path.
Hard rule #6 says the API never trades, and the cheapest way to keep that true is
to give it nothing to trade *with*.

So the shared vocabulary lives here, in a module that imports nothing. The two
processes agree on a string and a table row, and on nothing else. That narrowness
is the design: a stopped API simply never sets a flag, and the worker keeps
trading correctly, which is what the rule asks for.
"""

from __future__ import annotations

from typing import Final

#: Stops **new entries** at the top of the next cycle. Management keeps running —
#: a halt that also stopped position management would leave open structures
#: unattended, which is worse than the condition that triggered it.
HALT_FLAG: Final = "halt"

#: Pre-empts whatever cycle is due and runs the flatten instead: cancel-all,
#: close-all. Checked in `run_cycle` rather than inside a cycle so it can override
#: any of them — someone hitting `/api/control/flatten` at 11:02 is not asking for
#: the 11:02 manage sweep to finish first.
FLATTEN_FLAG: Final = "flatten"
