"""External facts the gates need but must never fetch themselves.

The kernel is a pure function of `(proposal, portfolio_state, config, context)`.
Anything that would otherwise require a network call — the current time, which
symbols the chain actually listed, when earnings are — is resolved by the caller
and handed in as inert data. That is what makes the kernel testable without a
broker and impossible to "convince" mid-decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass(frozen=True, slots=True)
class KernelContext:
    now: datetime
    # Symbols the chain actually returned this cycle. Empty means "not supplied",
    # and Gate 12 skips the existence check rather than failing every proposal —
    # a missing input must not masquerade as a risk violation.
    available_symbols: frozenset[str] = frozenset()
    earnings_dates: Mapping[str, tuple[date, ...]] = field(default_factory=dict)
    macro_events: tuple[datetime, ...] = ()

    def __post_init__(self) -> None:
        """Refuse a naive `now`, and refuse naive macro events.

        Gate 11 compares `now` against 09:30/16:00 US/Eastern. A naive datetime
        has no offset to convert *from*, so `astimezone` would silently assume the
        machine's local zone — which is exactly the class of bug `clock.py` was
        written to kill, arriving through a different door. There is no safe
        default here, so the only correct behaviour is to reject the input.

        `frozen=True` blocks assignment but not validation: `__post_init__` still
        runs during construction, before the instance is sealed.
        """
        if self.now.tzinfo is None or self.now.tzinfo.utcoffset(self.now) is None:
            raise ValueError(
                f"KernelContext.now must be timezone-aware, got naive {self.now!r}. "
                "Use vigil.clock.now_et()."
            )
        for event in self.macro_events:
            if event.tzinfo is None or event.tzinfo.utcoffset(event) is None:
                raise ValueError(f"macro_events must be timezone-aware, got naive {event!r}")
