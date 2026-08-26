"""OCC option symbol parsing.

`SPY260904P00640000` = underlying + YYMMDD + C/P + strike x1000 padded to 8 digits.

The root is variable length (SPY, QQQ, but also BRK.B-style roots and 1-char roots),
so everything is parsed from the *right*, where the field widths are fixed. Parsing
left-to-right by assuming a 3-character root is the classic bug here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

# Fixed-width tail: 6 (date) + 1 (call/put) + 8 (strike) = 15 characters.
_TAIL_LEN = 15
# Strikes are encoded as thousandths of a dollar. Decimal, not float — money never
# touches binary floating point in this codebase (CLAUDE.md conventions).
_STRIKE_SCALE = Decimal(1000)


@dataclass(frozen=True, slots=True)
class OccSymbol:
    raw: str
    underlying: str
    expiry: date
    is_put: bool
    strike: Decimal

    @property
    def right(self) -> str:
        return "P" if self.is_put else "C"

    def dte(self, asof: date) -> int:
        """Calendar days to expiry. 0 means it expires today (0DTE)."""
        return (self.expiry - asof).days


def parse_occ(symbol: str) -> OccSymbol:
    if len(symbol) <= _TAIL_LEN:
        raise ValueError(f"not an OCC symbol: {symbol!r}")

    root, tail = symbol[:-_TAIL_LEN], symbol[-_TAIL_LEN:]
    yy, mm, dd = tail[0:2], tail[2:4], tail[4:6]
    right = tail[6]
    strike_raw = tail[7:]

    if right not in ("C", "P"):
        raise ValueError(f"bad option right {right!r} in {symbol!r}")

    return OccSymbol(
        raw=symbol,
        underlying=root,
        # OCC years are 2-digit; the listed-options universe has no pre-2000
        # expiries, so the 2000+ offset is unambiguous.
        expiry=date(2000 + int(yy), int(mm), int(dd)),
        is_put=right == "P",
        strike=Decimal(strike_raw) / _STRIKE_SCALE,
    )
