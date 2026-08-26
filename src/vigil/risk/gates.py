"""The twelve gates. PLAN §5.

Each gate is a pure function returning a `GateVerdict` — never a bool, never an
exception. Callers persist every verdict, including passes, because "did any of
this ever fire?" is the first question asked of a risk system and the answer has
to come from the record rather than from memory.

**Frozen after Day 4 (Wed 2 Sep).** Thresholds in `config/risk.yaml` stay tunable;
the logic in this file does not change.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from decimal import Decimal
from math import ceil, gcd

from vigil.clock import ET, MARKET_CLOSE, MARKET_OPEN
from vigil.config import RiskConfig
from vigil.domain import GateVerdict, PortfolioState, TradeProposal
from vigil.risk.context import KernelContext


def _ok(n: int, name: str, **detail: str) -> GateVerdict:
    return GateVerdict(number=n, name=name, passed=True, detail=detail)


def _fail(n: int, name: str, reason: str, **detail: str) -> GateVerdict:
    return GateVerdict(number=n, name=name, passed=False, reason=reason, detail=detail)


# --------------------------------------------------------------------------- 1
def gate_defined_risk(p: TradeProposal, _s: PortfolioState, _c: RiskConfig,
                      _x: KernelContext) -> GateVerdict:
    """Max loss finite and computable; every short leg covered in the same ticket.

    The only truly fatal failure mode. Alpaca's mleg construction already makes a
    naked short impossible, and we check it again anyway — a broker-side guarantee
    we depend on but do not verify is a guarantee we do not actually have.
    """
    name = "defined_risk"
    if not p.legs:
        return _fail(1, name, "proposal has no legs")

    expiries = {leg.occ.expiry for leg in p.legs}
    underlyings = {leg.occ.underlying for leg in p.legs}
    if len(underlyings) != 1:
        return _fail(1, name, f"legs span multiple underlyings: {sorted(underlyings)}")

    # A structure whose legs sit in different expiries (a calendar) does not have
    # a max loss derivable from strike distance, and `TradeProposal` carries a
    # single `expiry` field that could only describe one of them. Nothing builds
    # one today, so the honest answer is to refuse rather than to price it wrong.
    # Adding calendars means changing this gate — deliberately, before the freeze.
    if len(expiries) != 1:
        return _fail(1, name, f"legs span multiple expiries: {sorted(expiries)}")
    if next(iter(expiries)) != p.expiry:
        return _fail(1, name, f"leg expiry {next(iter(expiries))} != proposal expiry {p.expiry}")

    # Cover check per right: a short call is covered by a long call, never by a
    # long put. Netting across rights is exactly how a "hedged" book ends up naked.
    for is_put in (True, False):
        side = [leg for leg in p.legs if leg.occ.is_put == is_put]
        short_qty = sum(leg.ratio_qty for leg in side if leg.is_short)
        long_qty = sum(leg.ratio_qty for leg in side if not leg.is_short)
        if short_qty > long_qty:
            right = "put" if is_put else "call"
            return _fail(1, name, f"uncovered short {right} leg "
                                  f"({short_qty} short vs {long_qty} long)")

    if p.width <= 0:
        return _fail(1, name, f"width must be positive, got {p.width}")

    # **Width is derived, never believed.** `max_loss` is `(width − credit) × 100 × n`,
    # so a proposal that understates its width understates the very number Gate 2
    # sizes against — and every other gate then reasons from the understatement.
    # Checking it here is what stops the kernel from being *convincible*, which
    # matters from the moment the LLM starts emitting proposals rather than the
    # builders. Max loss on a multi-right structure is one width, not their sum:
    # the underlying cannot finish below the put spread and above the call spread
    # at once, so the widest side is the exposure.
    derived = _derived_width(p)
    if derived is None:
        return _fail(1, name, "cannot derive width: no right has two distinct strikes")
    if derived != p.width:
        return _fail(1, name, f"declared width {p.width} != strike distance {derived}",
                     declared=str(p.width), derived=str(derived))

    max_loss = p.max_loss
    if not max_loss.is_finite() or max_loss <= 0:
        return _fail(1, name, f"max loss not finite/positive: {max_loss}")

    return _ok(1, name, max_loss=str(max_loss), width=str(derived),
               expiry=str(p.expiry))


def _derived_width(p: TradeProposal) -> Decimal | None:
    """The widest strike span within a single right. `None` when undecidable.

    Per right rather than across the whole proposal: an iron condor's put strikes
    and call strikes are far apart, and the span between them is not a width — it
    is the profit zone.
    """
    spans: list[Decimal] = []
    for is_put in (True, False):
        strikes = {leg.occ.strike for leg in p.legs if leg.occ.is_put == is_put}
        if len(strikes) >= 2:
            spans.append(max(strikes) - min(strikes))
    return max(spans) if spans else None


# --------------------------------------------------------------------------- 2
def gate_per_trade_risk(p: TradeProposal, s: PortfolioState, c: RiskConfig,
                        _x: KernelContext) -> GateVerdict:
    """Max loss ≤ 2% of equity. The ladder may size below this, never above (§4.7)."""
    name = "per_trade_risk"
    if s.equity <= 0:
        return _fail(2, name, f"non-positive equity: {s.equity}")
    budget = c.max_risk_per_trade_pct * s.equity
    if p.max_loss > budget:
        return _fail(2, name, f"max loss {p.max_loss} exceeds {c.max_risk_per_trade_pct:.1%} "
                              f"of equity ({budget})",
                     max_loss=str(p.max_loss), budget=str(budget))
    return _ok(2, name, max_loss=str(p.max_loss), budget=str(budget))


# --------------------------------------------------------------------------- 3
def gate_daily_loss(_p: TradeProposal, s: PortfolioState, c: RiskConfig,
                    _x: KernelContext) -> GateVerdict:
    """Day P&L ≤ −3% halts *new entries*. Management of open positions continues."""
    name = "daily_loss_stop"
    if s.day_pnl_pct <= c.daily_loss_halt_pct:
        return _fail(3, name, f"day P&L {s.day_pnl_pct:.2%} at or below "
                              f"{c.daily_loss_halt_pct:.2%} — no new entries",
                     day_pnl_pct=str(s.day_pnl_pct))
    return _ok(3, name, day_pnl_pct=str(s.day_pnl_pct))


# --------------------------------------------------------------------------- 4
def gate_drawdown(_p: TradeProposal, s: PortfolioState, c: RiskConfig,
                  _x: KernelContext) -> GateVerdict:
    """Equity ≤ −8% from peak → full halt. Human-in-the-loop by design (§5.2)."""
    name = "max_drawdown"
    if s.drawdown_pct <= c.max_drawdown_pct:
        return _fail(4, name, f"drawdown {s.drawdown_pct:.2%} at or below "
                              f"{c.max_drawdown_pct:.2%} — full halt, human required",
                     drawdown_pct=str(s.drawdown_pct))
    return _ok(4, name, drawdown_pct=str(s.drawdown_pct))


# --------------------------------------------------------------------------- 5
def gate_concurrency(_p: TradeProposal, s: PortfolioState, c: RiskConfig,
                     _x: KernelContext) -> GateVerdict:
    """≤ 6 open structures. Bounded blast radius."""
    name = "concurrency"
    n = len(s.open_structures)
    if n >= c.max_open_structures:
        return _fail(5, name, f"{n} open structures, limit {c.max_open_structures}", open=str(n))
    return _ok(5, name, open=str(n))


# --------------------------------------------------------------------------- 6
def gate_concentration(p: TradeProposal, s: PortfolioState, c: RiskConfig,
                       _x: KernelContext) -> GateVerdict:
    """≤2 per underlying, and ≤40% of open risk in one name.

    Correlated condors on the same underlying are one trade wearing two hats.
    """
    name = "concentration"
    same = [o for o in s.open_structures if o.underlying == p.underlying]
    if len(same) >= c.max_structures_per_underlying:
        return _fail(6, name, f"{len(same)} open on {p.underlying}, "
                              f"limit {c.max_structures_per_underlying}")

    # Share is computed *including* the proposal — the question is what the book
    # looks like after this fills, not before.
    name_risk = sum((o.max_loss for o in same), Decimal(0)) + p.max_loss
    total_risk = s.open_risk + p.max_loss
    share = Decimal(0) if total_risk == 0 else name_risk / total_risk

    # The cap is only *satisfiable* once the book is diversified enough: with N
    # structures the smallest achievable single-name share is 1/N, so a 40% cap is
    # arithmetically impossible below 3. Enforcing it earlier would reject every
    # opening trade — the gate would block the book it is meant to shape.
    min_structures = ceil(1 / c.max_risk_share_per_underlying)
    projected_count = len(s.open_structures) + 1
    if projected_count >= min_structures and share > c.max_risk_share_per_underlying:
        return _fail(6, name, f"{p.underlying} would hold {share:.1%} of open risk, "
                              f"limit {c.max_risk_share_per_underlying:.0%}",
                     share=str(share))
    return _ok(6, name, share=str(share), same_underlying=str(len(same)))


# --------------------------------------------------------------------------- 7
def gate_dollar_delta(p: TradeProposal, s: PortfolioState, c: RiskConfig,
                      _x: KernelContext) -> GateVerdict:
    """|Σ delta × 100 × spot × qty| ≤ 5% of equity, portfolio-wide, after this fill.

    Dollars, never a bare delta count — the original units were ambiguous by a
    factor of ~10 and this gate freezes on Day 4 (§5.1).
    """
    name = "portfolio_dollar_delta"
    if s.equity <= 0:
        return _fail(7, name, f"non-positive equity: {s.equity}")
    projected = s.net_dollar_delta + p.dollar_delta
    limit = c.max_dollar_delta_pct * s.equity
    if abs(projected) > limit:
        return _fail(7, name, f"projected dollar delta {projected:.0f} exceeds ±{limit:.0f}",
                     projected=str(projected), limit=str(limit))
    return _ok(7, name, projected=str(projected), limit=str(limit))


# --------------------------------------------------------------------------- 8
def gate_liquidity(p: TradeProposal, _s: PortfolioState, c: RiskConfig,
                   _x: KernelContext) -> GateVerdict:
    """Per leg: OI ≥ 100, bid > 0, spread ≤ 10% of mid (or ≤ $0.10 when cheap).

    An untradeable fill is fake P&L — and paper fills are generous enough to hide
    exactly this, which is why the check lives in the kernel rather than in hope.
    """
    name = "liquidity"
    for leg in p.legs:
        if leg.open_interest < c.min_open_interest:
            return _fail(8, name, f"{leg.symbol} OI {leg.open_interest} < {c.min_open_interest}")
        if leg.bid <= 0:
            return _fail(8, name, f"{leg.symbol} has no bid ({leg.bid})")
        mid = leg.mid
        if mid <= 0:
            return _fail(8, name, f"{leg.symbol} non-positive mid ({mid})")
        # A $0.05/$0.15 market is 100% of mid but is also the tightest a cheap
        # long wing ever gets. The absolute allowance stops the percentage rule
        # from banning the protective leg that makes the risk defined.
        if mid <= c.cheap_leg_mid_threshold:
            if leg.spread > c.max_spread_abs_cheap_leg:
                return _fail(8, name, f"{leg.symbol} spread {leg.spread} > "
                                      f"${c.max_spread_abs_cheap_leg} on a cheap leg")
        elif leg.spread / mid > c.max_spread_pct_of_mid:
            return _fail(8, name, f"{leg.symbol} spread {leg.spread / mid:.1%} of mid > "
                                  f"{c.max_spread_pct_of_mid:.0%}")
    return _ok(8, name, legs=str(len(p.legs)))


# --------------------------------------------------------------------------- 9
def gate_credit_quality(p: TradeProposal, _s: PortfolioState, c: RiskConfig,
                        _x: KernelContext) -> GateVerdict:
    """Credit ≥ 18% of width, and max-loss:max-profit ≤ 5.5:1.

    Refuses picking up pennies in front of a steamroller. The 18% floor is a 0-2
    DTE number; 25% is a 30-45 DTE number that rejects every candidate (§4.4.2).
    """
    name = "credit_quality"
    if p.is_credit:
        pct = p.credit_pct_of_width
        if pct < c.min_credit_pct_of_width:
            return _fail(9, name, f"credit {pct:.1%} of width < "
                                  f"{c.min_credit_pct_of_width:.0%}", credit_pct=str(pct))
    profit = p.max_profit
    if profit <= 0:
        return _fail(9, name, f"non-positive max profit: {profit}")
    ratio = p.max_loss / profit
    if ratio > c.max_loss_to_profit_ratio:
        return _fail(9, name, f"loss:profit {ratio:.2f}:1 exceeds "
                              f"{c.max_loss_to_profit_ratio}:1", ratio=str(ratio))
    return _ok(9, name, credit_pct=str(p.credit_pct_of_width), ratio=str(ratio))


# -------------------------------------------------------------------------- 10
def gate_event_blackout(p: TradeProposal, _s: PortfolioState, c: RiskConfig,
                        x: KernelContext) -> GateVerdict:
    """No earnings inside the contract's life; no entry around a macro print.

    Scheduled gaps break the assumption a short-vol structure is priced on.

    The window is **asymmetric on purpose**. Before a print, the risk is that the
    quote we are pricing against is about to be repriced — a few minutes is
    enough. After it, the tape is genuinely disorderly: spreads widen, the
    indicative feed lags hardest, and the move keeps developing. A symmetric
    window treats the calm side and the violent side as equivalent, which they
    plainly are not.
    """
    name = "event_blackout"
    now = x.now.astimezone(ET)
    for ed in x.earnings_dates.get(p.underlying, ()):
        if now.date() <= ed <= p.expiry:
            return _fail(10, name, f"{p.underlying} earnings {ed} falls inside contract life")

    for event in x.macro_events:
        # Positive when the print is still ahead of us, negative once it has landed.
        minutes_until = (event - now).total_seconds() / 60
        if 0 <= minutes_until <= c.macro_blackout_before_minutes:
            return _fail(10, name, f"macro print at {event.astimezone(ET):%H:%M} in "
                                   f"{minutes_until:.0f} min "
                                   f"(blackout {c.macro_blackout_before_minutes} min before)")
        if -c.macro_blackout_after_minutes <= minutes_until < 0:
            return _fail(10, name, f"macro print at {event.astimezone(ET):%H:%M} was "
                                   f"{-minutes_until:.0f} min ago "
                                   f"(blackout {c.macro_blackout_after_minutes} min after)")
    return _ok(10, name)


# -------------------------------------------------------------------------- 11
def gate_time_windows(p: TradeProposal, _s: PortfolioState, c: RiskConfig,
                      x: KernelContext) -> GateVerdict:
    """No entries in the first 15 or last 20 minutes; no 0DTE entry after 14:30.

    The opening and closing auctions are not our edge — they are where an
    indicative quote is least like a fill.

    **Every wall-clock field is read in US/Eastern**, never in whatever zone the
    caller happened to attach. `09:30` and `16:00` are Eastern facts, so reading
    `.time()` off a UTC-aware timestamp compares a New York boundary against a
    London reading of the same instant and shifts every window by four hours.
    The conversion is the whole reason this gate can be trusted inside a
    container whose clock reasons in UTC.
    """
    name = "time_windows"
    now = x.now.astimezone(ET)
    now_t: time = now.time()
    today = now.date()

    open_dt = datetime.combine(today, MARKET_OPEN, tzinfo=ET)
    close_dt = datetime.combine(today, MARKET_CLOSE, tzinfo=ET)

    if now_t < MARKET_OPEN or now_t >= MARKET_CLOSE:
        return _fail(11, name, f"{now_t:%H:%M} is outside regular hours")
    if now < open_dt + timedelta(minutes=c.no_entry_first_minutes):
        return _fail(11, name, f"within the first {c.no_entry_first_minutes} minutes")
    if now > close_dt - timedelta(minutes=c.no_entry_last_minutes):
        return _fail(11, name, f"within the last {c.no_entry_last_minutes} minutes")
    if p.expiry == today and now_t >= c.zero_dte_entry_cutoff:
        return _fail(11, name, f"0DTE entry after {c.zero_dte_entry_cutoff:%H:%M}")
    return _ok(11, name, now=f"{now_t:%H:%M}")


# -------------------------------------------------------------------------- 12
def gate_sanity(p: TradeProposal, s: PortfolioState, c: RiskConfig,
                x: KernelContext) -> GateVerdict:
    """Idempotency and the boring checks that prevent the commonest agent failure.

    Duplicate submission is the classic autonomous-agent bug: a retry after a
    timeout that actually succeeded. The database's UNIQUE constraint is the real
    defence (hard rule #9); this is the cheap check that runs first.
    """
    name = "idempotency_sanity"
    if not p.client_order_id:
        return _fail(12, name, "missing client_order_id")
    if p.client_order_id in s.known_client_order_ids:
        return _fail(12, name, f"duplicate client_order_id {p.client_order_id}")
    if any(o.structure_key == p.structure_key for o in s.open_structures):
        return _fail(12, name, f"identical structure already open on {p.underlying}")

    if p.contracts < 1:
        return _fail(12, name, f"contract count must be ≥ 1, got {p.contracts}")

    # Alpaca requires leg ratios with GCD 1 — 2:2 must be expressed as 1:1, with
    # the size carried by qty instead.
    ratios = [leg.ratio_qty for leg in p.legs]
    if any(r < 1 for r in ratios):
        return _fail(12, name, f"leg ratios must be ≥ 1, got {ratios}")
    divisor = 0
    for r in ratios:
        divisor = gcd(divisor, r)
    if divisor != 1:
        return _fail(12, name, f"leg ratios must have GCD 1, got {ratios} (gcd {divisor})")

    if x.available_symbols:
        missing = [leg.symbol for leg in p.legs if leg.symbol not in x.available_symbols]
        if missing:
            return _fail(12, name, f"symbols not present in the chain: {missing}")

    # Limit within ±20% of the package mid. Catches a fat-finger or a mispriced
    # ladder rung before it reaches the broker.
    package_mid = sum((Decimal(leg.signed_ratio) * leg.mid for leg in p.legs), Decimal(0))
    package_mid = -package_mid if p.is_credit else package_mid
    if package_mid > 0:
        deviation = abs(p.limit_price - package_mid) / package_mid
        if deviation > c.max_limit_deviation_from_mid:
            return _fail(12, name, f"limit {p.limit_price} is {deviation:.0%} from mid "
                                   f"{package_mid}, max {c.max_limit_deviation_from_mid:.0%}",
                         package_mid=str(package_mid))
    return _ok(12, name, package_mid=str(package_mid))


# The canonical order. Kept explicit rather than derived from module contents so
# the sequence cannot change by accident when a function is added or renamed.
ALL_GATES = (
    gate_defined_risk,
    gate_per_trade_risk,
    gate_daily_loss,
    gate_drawdown,
    gate_concurrency,
    gate_concentration,
    gate_dollar_delta,
    gate_liquidity,
    gate_credit_quality,
    gate_event_blackout,
    gate_time_windows,
    gate_sanity,
)
