"""Loading the backfilled realized-volatility series (PLAN §4.3.1 cold start).

`scripts/backfill_vrp.py` writes one JSON file per underlying under `data/raw/`,
mapping session date to annualized realized vol. This module reads them back.

Deliberately a plain file read rather than a database table: the series is a
*measurement input*, regenerated on demand from Alpaca, not a record of anything
the agent did. Putting it in Postgres would imply it is authoritative when it is
actually derived, and would make the regime router depend on a database it has no
other reason to need.
"""

from __future__ import annotations

import json
from functools import lru_cache

from vigil.settings import REPO_ROOT

RV_DIR = REPO_ROOT / "data" / "raw"


@lru_cache(maxsize=8)
def rv_history(underlying: str) -> tuple[float, ...]:
    """Trailing annualized realized vol, oldest first. Empty when not backfilled.

    Returns a tuple rather than a list so `lru_cache` can hand the same object to
    every caller without any of them being able to mutate the shared series.
    """
    path = RV_DIR / f"rv_{underlying.lower()}.json"
    if not path.exists():
        return ()
    data = json.loads(path.read_text())
    # Sorted by session date, so the series is chronological regardless of the
    # order json.dumps happened to write.
    return tuple(float(v) for _, v in sorted(data.items()))
