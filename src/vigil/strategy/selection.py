"""Strike selection — choosing which contract to sell.

Lives in the package rather than in a measurement script because it is real
strategy logic that the candidate builders (§4) will reuse. Pure function of a
contract list: no network, no session, testable with a fixture chain.
"""

from __future__ import annotations

from decimal import Decimal

from vigil.data.chain import Contract

# §4.5: short strike delta 0.15-0.20. 0.16 is the midpoint the plan targets.
TARGET_SHORT_DELTA = Decimal("0.16")


def pick_by_delta(
    contracts: list[Contract], *, target: Decimal = TARGET_SHORT_DELTA
) -> Contract | None:
    """The contract whose |delta| is closest to `target`.

    Absolute value because puts carry negative delta and calls positive, so one
    comparison serves both sides of the book. Contracts with no delta are skipped
    rather than estimated -- the kernel rejects proposals with missing fields, and
    inferring a delta here would smuggle a guess past that rule.
    """
    scored = [(abs(Decimal(str(c.delta))) - target, c) for c in contracts if c.delta is not None]
    if not scored:
        return None
    return min(scored, key=lambda pair: abs(pair[0]))[1]
