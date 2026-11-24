"""Confidence-based MNQ lot sizing with day-green + win-streak gate.

Env (all optional):
  BASE_CONTRACTS=1
  MAX_CONTRACTS=4
  SIZE_WIN_STREAK_MIN=2          # consecutive wins required before size>1
  SIZE_REQUIRE_DAY_GREEN=1       # 1/true: daily_pnl>0 required for size>1
  SIZE_CONF_2=0.72               # confidence >= this -> 2 (if gate open)
  SIZE_CONF_3=0.78               # -> 3
  SIZE_CONF_4=0.85               # -> 4

Default (2026-10-07): sizing is PURELY by confidence (no gate):
  SIZE_WIN_STREAK_MIN=0, SIZE_REQUIRE_DAY_GREEN=0.
Optional gate (env): size > BASE only when day green and/or win_streak >= min.
  Always hard-capped at MAX_CONTRACTS. Existing daily loss limit still blocks new entries.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple, Any


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def load_size_scale_config() -> Dict[str, Any]:
    base = max(1, _env_int("BASE_CONTRACTS", 1))
    max_c = max(base, _env_int("MAX_CONTRACTS", 4))
    return {
        "base": base,
        "max_contracts": max_c,
        "win_streak_min": max(0, _env_int("SIZE_WIN_STREAK_MIN", 0)),
        "require_day_green": _env_bool("SIZE_REQUIRE_DAY_GREEN", False),
        "conf_2": _env_float("SIZE_CONF_2", 0.72),
        "conf_3": _env_float("SIZE_CONF_3", 0.78),
        "conf_4": _env_float("SIZE_CONF_4", 0.85),
        # 2nd/3rd concurrent trade only when entry confidence >= this.
        "extra_trade_min_conf": _env_float("EXTRA_TRADE_MIN_CONF", _env_float("SIZE_CONF_2", 0.72)),
    }


def confidence_to_desired(
    confidence: float,
    conf_2: float,
    conf_3: float,
    conf_4: float,
    base: int = 1,
) -> int:
    """Map entry confidence into a desired contract band (before gates)."""
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        c = 0.0
    if c >= conf_4:
        return 4
    if c >= conf_3:
        return 3
    if c >= conf_2:
        return 2
    return max(1, int(base))


def compute_sized_contracts(
    confidence: float,
    *,
    daily_pnl: float,
    win_streak: int,
    base: int = 1,
    max_contracts: int = 4,
    win_streak_min: int = 2,
    require_day_green: bool = True,
    conf_2: float = 0.72,
    conf_3: float = 0.78,
    conf_4: float = 0.85,
) -> Tuple[int, str]:
    """Return (contracts, reason) for a live entry.

    Bands (defaults): conf <0.72 ->1; >=0.72 ->2; >=0.78 ->3; >=0.85 ->4.
    Gate: size > base only when day green (daily_pnl > 0) AND win_streak >= min.
    """
    base = max(1, int(base))
    max_contracts = max(base, int(max_contracts))
    desired = confidence_to_desired(confidence, conf_2, conf_3, conf_4, base=base)
    desired = min(max(desired, 1), max_contracts)

    try:
        conf_f = float(confidence)
    except (TypeError, ValueError):
        conf_f = 0.0

    if desired <= base:
        return base, f"base conf={conf_f:.3f}<{conf_2:.2f}"

    blocks = []
    if require_day_green and float(daily_pnl) <= 0:
        blocks.append(f"day_not_green pnl={float(daily_pnl):.2f}")
    if int(win_streak) < int(win_streak_min):
        blocks.append(f"streak={int(win_streak)}<{int(win_streak_min)}")

    if blocks:
        return base, "gated " + ",".join(blocks) + f" want={desired}"

    return desired, f"scale conf={conf_f:.3f}"


def format_size_log(contracts: int, confidence: float, reason: str) -> str:
    try:
        conf_f = float(confidence)
    except (TypeError, ValueError):
        conf_f = 0.0
    return f"contracts={int(contracts)} confidence={conf_f:.3f} reason={reason}"
