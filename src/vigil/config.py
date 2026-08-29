"""Typed access to config/*.yaml.

Config is loaded into frozen dataclasses rather than passed around as raw dicts,
so a typo in a YAML key fails at load time with a clear error instead of silently
becoming a `None` threshold inside a risk gate at 09:31.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from vigil.settings import REPO_ROOT

CONFIG_DIR = REPO_ROOT / "config"


def _load(name: str) -> dict[str, Any]:
    path: Path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"missing config file: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data


def _dec(raw: dict[str, Any], key: str) -> Decimal:
    """Read a number as Decimal via str().

    `Decimal(0.02)` captures the binary float's error; `Decimal("0.02")` does not.
    YAML gives us a float, so the str() round-trip is what keeps thresholds exact.
    """
    if key not in raw:
        raise KeyError(f"missing config key: {key}")
    return Decimal(str(raw[key]))


def _time(raw: dict[str, Any], key: str) -> time:
    return time.fromisoformat(str(raw[key]))


@dataclass(frozen=True, slots=True)
class RiskConfig:
    max_risk_per_trade_pct: Decimal
    daily_loss_halt_pct: Decimal
    max_drawdown_pct: Decimal
    max_open_structures: int
    max_structures_per_underlying: int
    max_risk_share_per_underlying: Decimal
    max_dollar_delta_pct: Decimal
    min_open_interest: int
    max_spread_pct_of_mid: Decimal
    max_spread_abs_cheap_leg: Decimal
    cheap_leg_mid_threshold: Decimal
    min_credit_pct_of_width: Decimal
    max_loss_to_profit_ratio: Decimal
    macro_blackout_before_minutes: int
    macro_blackout_after_minutes: int
    no_entry_first_minutes: int
    no_entry_last_minutes: int
    zero_dte_entry_cutoff: time
    hard_flatten: time
    max_limit_deviation_from_mid: Decimal

    @classmethod
    def load(cls) -> RiskConfig:
        r = _load("risk.yaml")
        return cls(
            max_risk_per_trade_pct=_dec(r, "max_risk_per_trade_pct"),
            daily_loss_halt_pct=_dec(r, "daily_loss_halt_pct"),
            max_drawdown_pct=_dec(r, "max_drawdown_pct"),
            max_open_structures=int(r["max_open_structures"]),
            max_structures_per_underlying=int(r["max_structures_per_underlying"]),
            max_risk_share_per_underlying=_dec(r, "max_risk_share_per_underlying"),
            max_dollar_delta_pct=_dec(r, "max_dollar_delta_pct"),
            min_open_interest=int(r["min_open_interest"]),
            max_spread_pct_of_mid=_dec(r, "max_spread_pct_of_mid"),
            max_spread_abs_cheap_leg=_dec(r, "max_spread_abs_cheap_leg"),
            cheap_leg_mid_threshold=_dec(r, "cheap_leg_mid_threshold"),
            min_credit_pct_of_width=_dec(r, "min_credit_pct_of_width"),
            max_loss_to_profit_ratio=_dec(r, "max_loss_to_profit_ratio"),
            macro_blackout_before_minutes=int(r["macro_blackout_before_minutes"]),
            macro_blackout_after_minutes=int(r["macro_blackout_after_minutes"]),
            no_entry_first_minutes=int(r["no_entry_first_minutes"]),
            no_entry_last_minutes=int(r["no_entry_last_minutes"]),
            zero_dte_entry_cutoff=_time(r, "zero_dte_entry_cutoff"),
            hard_flatten=_time(r, "hard_flatten"),
            max_limit_deviation_from_mid=_dec(r, "max_limit_deviation_from_mid"),
        )


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    short_delta_target: Decimal
    short_delta_min: Decimal
    short_delta_max: Decimal
    min_dte: int
    max_dte: int
    width_by_underlying: dict[str, Decimal]
    default_width: Decimal
    profit_target_pct: Decimal
    breach_exit_min_minutes_left: int
    time_stop: time
    convexity_risk_share: Decimal
    convexity_min_dte: int
    convexity_max_dte: int
    convexity_long_delta_target: Decimal
    convexity_strangle_delta: Decimal
    convexity_profit_target_multiple: Decimal

    def width_for(self, underlying: str) -> Decimal:
        """§4.4.3: the narrowest tradeable width, per underlying."""
        return self.width_by_underlying.get(underlying, self.default_width)

    @classmethod
    def load(cls) -> StrategyConfig:
        s = _load("strategy.yaml")
        return cls(
            short_delta_target=_dec(s, "short_delta_target"),
            short_delta_min=_dec(s, "short_delta_min"),
            short_delta_max=_dec(s, "short_delta_max"),
            min_dte=int(s["min_dte"]),
            max_dte=int(s["max_dte"]),
            width_by_underlying={k: Decimal(str(v)) for k, v in s["width_by_underlying"].items()},
            default_width=_dec(s, "default_width"),
            profit_target_pct=_dec(s, "profit_target_pct"),
            breach_exit_min_minutes_left=int(s["breach_exit_min_minutes_left"]),
            time_stop=_time(s, "time_stop"),
            convexity_risk_share=_dec(s, "convexity_risk_share"),
            convexity_min_dte=int(s["convexity_min_dte"]),
            convexity_max_dte=int(s["convexity_max_dte"]),
            convexity_long_delta_target=_dec(s, "convexity_long_delta_target"),
            convexity_strangle_delta=_dec(s, "convexity_strangle_delta"),
            convexity_profit_target_multiple=_dec(s, "convexity_profit_target_multiple"),
        )


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    ema_fast: int
    ema_slow: int
    trend_threshold: Decimal
    vrp_sell_floor_pct: Decimal
    vrp_stress_pct: Decimal
    stress_gap_pct: Decimal
    cheap_vol_iv_pct: Decimal
    vrp_raw_rich_abs: Decimal
    vrp_override_size: Decimal
    iv_seed_min_sessions: int

    @classmethod
    def load(cls) -> RegimeConfig:
        r = _load("strategy.yaml")["regime"]
        return cls(
            ema_fast=int(r["ema_fast"]),
            ema_slow=int(r["ema_slow"]),
            trend_threshold=_dec(r, "trend_threshold"),
            vrp_sell_floor_pct=_dec(r, "vrp_sell_floor_pct"),
            vrp_stress_pct=_dec(r, "vrp_stress_pct"),
            stress_gap_pct=_dec(r, "stress_gap_pct"),
            cheap_vol_iv_pct=_dec(r, "cheap_vol_iv_pct"),
            vrp_raw_rich_abs=_dec(r, "vrp_raw_rich_abs"),
            vrp_override_size=_dec(r, "vrp_override_size"),
            # Option 3 — how many IV points (real + synthetic seed) before the
            # CHEAP_VOL percentile is allowed to mean something. See signals/iv_seed.py.
            iv_seed_min_sessions=int(r["iv_seed_min_sessions"]),
        )


@dataclass(frozen=True, slots=True)
class GreeksConfig:
    """Inputs to the local Black-Scholes fallback (§1.3 A1). See data/greeks.py.

    `risk_free_rate` is a plain `float`, not a `Decimal`, and that is deliberate.
    Decimal exists in this codebase to keep binary floating point away from money.
    This number is not money: it goes straight into `exp()` and `log()`, which are
    float operations, so a Decimal here would only be converted back and would
    imply a precision guarantee the model does not have.
    """

    risk_free_rate: float

    @classmethod
    def load(cls) -> GreeksConfig:
        g = _load("strategy.yaml")["greeks"]
        return cls(risk_free_rate=float(g["risk_free_rate"]))


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """The LLM portfolio manager's knobs (§6.2). See config/agent.yaml.

    None of these are safety controls — the model selects from an already-gated
    menu, so its worst case is a worse *legal* trade, never an illegal one. That
    is why this is a separate config from `risk.yaml`: the two must not be tuned
    with the same hand, and only `risk.yaml` is frozen after Day 4.

    `enabled` is read here but the *effective* switch is `enabled AND a key is
    present` — the manager treats a missing OPENAI_API_KEY as disabled rather than
    as an error, because "run the deterministic path" is always a valid answer and
    a crash on a missing key would violate §6.3's promise that the model is never
    load-bearing.
    """

    model: str
    entry_effort: str
    review_effort: str
    timeout_seconds: float
    max_output_tokens: int
    enabled: bool

    @classmethod
    def load(cls) -> AgentConfig:
        a = _load("agent.yaml")
        return cls(
            model=str(a["model"]),
            entry_effort=str(a["entry_effort"]),
            review_effort=str(a["review_effort"]),
            # float() not _dec(): a timeout is passed straight to the SDK/asyncio,
            # which want a float, and it is not money — Decimal would buy nothing
            # and only be converted back.
            timeout_seconds=float(a["timeout_seconds"]),
            max_output_tokens=int(a["max_output_tokens"]),
            enabled=bool(a["enabled"]),
        )


@dataclass(frozen=True, slots=True)
class UniverseConfig:
    primary: tuple[str, ...]
    secondary: tuple[str, ...]
    megacaps: tuple[str, ...]
    excluded: tuple[str, ...]

    @property
    def tradeable(self) -> tuple[str, ...]:
        return self.primary + self.secondary

    def is_allowed(self, underlying: str) -> bool:
        return underlying not in self.excluded

    @classmethod
    def load(cls) -> UniverseConfig:
        u = _load("universe.yaml")
        return cls(
            primary=tuple(u["primary"]),
            secondary=tuple(u["secondary"]),
            megacaps=tuple(u.get("megacaps", [])),
            excluded=tuple(u.get("excluded", [])),
        )


# Config is read once per process; the files do not change under a running worker.
risk_config = lru_cache(maxsize=1)(RiskConfig.load)
strategy_config = lru_cache(maxsize=1)(StrategyConfig.load)
universe_config = lru_cache(maxsize=1)(UniverseConfig.load)
regime_config = lru_cache(maxsize=1)(RegimeConfig.load)
greeks_config = lru_cache(maxsize=1)(GreeksConfig.load)
agent_config = lru_cache(maxsize=1)(AgentConfig.load)
