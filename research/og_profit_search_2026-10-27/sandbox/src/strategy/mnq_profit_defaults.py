"""
Locked MNQ live recipe (docs/PROFITABLE_LIVE.md).

L6 winner (level_strategy_search): 15m+5m agree, quiet 1m, 1m ATR×1.5 clip 6–25,
1R TP, no level entry gate. Applied when data/mnq_profit_config.json has
ema15_eod.profit_mode and the operator did not pick a different non-scalp strategy.
Leftover SCALP_MODE=hybrid is upgraded to ema15_eod (hybrid lost on real 30s).
Env keys outside FORCE_KEYS still win if already set.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

EMA15_ALIASES = frozenset({
    "ema15_eod",
    "ema15",
    "15m_ema_eod",
})

HYBRID_MODE_ALIASES = frozenset({
    "hybrid",
    "scalp_hybrid",
    "pullback",
    "continuation",
})

# Overwritten when locking so a leftover hybrid .env cannot revive 30s scalps.
# Skipped when EMA15_CLEAN_FORWARD=1 (verified A_d15_60 / w2_open / fix25 / eod500).
FORCE_KEYS = frozenset({
    "SCALP_MODE",
    "STRATEGY_MODE",
    "USE_30S_BARS",
    "MAX_HOLD_SECONDS",
    "MNQ_MAX_POSITIONS",
    "SCALP_SL_PTS",
    "SCALP_TP_PTS",
    "SCALP_SESSIONS",
    "MAX_TRADES_PER_DAY",
    "EMA15_MAX_TRADES_DAY",
    "EMA15_MAX_OPEN_LOTS",
    "EMA15_REQUIRE_DAILY",
    "EMA15_REQUIRE_60M",
    "EMA15_REQUIRE_5M",
    "EMA15_SL_TF",
    "EMA15_SL_ATR_MULT",
    "EMA15_SL_MIN",
    "EMA15_SL_MAX",
    "EMA15_TP_R",
    "EMA15_LEVEL_MODE",
    "SCALP_BREAKEVEN_ENABLED",
    "SCALP_TRAIL_AFTER_BE",
})

# Keys that define the L6 vs clean-forward recipe — never force-overwrite when
# EMA15_CLEAN_FORWARD is on (caller already set the PASS recipe).
CLEAN_FORWARD_PROTECT = frozenset({
    "EMA15_REQUIRE_DAILY",
    "EMA15_REQUIRE_60M",
    "EMA15_REQUIRE_5M",
    "EMA15_SL_TF",
    "EMA15_SL_ATR_MULT",
    "EMA15_SL_MIN",
    "EMA15_SL_MAX",
    "EMA15_TP_R",
    "EMA15_LEVEL_MODE",
    "EMA15_WINDOWS",
    "EMA15_SIDE",
    "EMA15_SL_FIXED",
    "EMA15_REQUIRE_QUIET",
    "EMA15_MIN_CONFIDENCE",
    "EMA15_CONTINUOUS",
    "EMA15_ENTRY_COOLDOWN_MIN",
    "SCALP_TP_PTS",
    "MAX_TRADES_PER_DAY",
    "EMA15_MAX_TRADES_DAY",
})

PROFIT_ENV_DEFAULTS = {
    "SCALP_MODE": "ema15_eod",
    "STRATEGY_MODE": "ema15_eod",
    "SESSION_MODE": "rth",
    "OVERNIGHT_TRADING": "false",
    "SCALP_SESSIONS": "rth=09:30-16:00",
    "SCAN_SLEEP_OPEN_SEC": "5",
    "SCAN_SLEEP_IDLE_SEC": "10",
    "USE_ORDER_FLOW": "false",
    "MNQ_MAX_POSITIONS": "2",
    "MAX_HOLD_SECONDS": "23400",
    "USE_30S_BARS": "false",
    "SCALP_SL_PTS": "25",
    "SCALP_TP_PTS": "500",
    "MAX_TRADES_PER_DAY": "20",
    "EMA15_MAX_TRADES_DAY": "20",
    "EMA15_MAX_OPEN_LOTS": "2",
    "LOSS_COOLDOWN_MINUTES": "0",
    "MTF_MAX_CONSEC_LOSSES": "99",
    # L6: 15m+5m stack, 1m ATR stop, no level entry gate
    "EMA15_REQUIRE_DAILY": "false",
    "EMA15_REQUIRE_60M": "false",
    "EMA15_REQUIRE_5M": "true",
    "EMA15_SL_TF": "1m",
    "EMA15_SL_ATR_MULT": "1.5",
    "EMA15_SL_MIN": "6",
    "EMA15_SL_MAX": "25",
    "EMA15_TP_R": "1.0",
    "EMA15_LEVEL_MODE": "0",
    "SCALP_BREAKEVEN_ENABLED": "false",
    "SCALP_TRAIL_AFTER_BE": "false",
}


def _explicit_strategy(environ: Mapping[str, str]) -> str:
    return (
        (environ.get("SCALP_MODE") or environ.get("STRATEGY_MODE") or "")
        .strip()
        .lower()
    )


def should_lock_profit_mode(
    cfg: Optional[Dict],
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    block = (cfg or {}).get("ema15_eod") or {}
    if not (block.get("profit_mode") or block.get("enabled")):
        return False
    env = environ if environ is not None else {}
    explicit = _explicit_strategy(env)
    if not explicit:
        return True
    if explicit in EMA15_ALIASES or explicit in HYBRID_MODE_ALIASES:
        return True
    return False


def apply_profit_env_defaults(
    cfg: Optional[Dict],
    environ: MutableMapping[str, str],
) -> List[str]:
    if not should_lock_profit_mode(cfg, environ):
        return []
    clean = str(environ.get("EMA15_CLEAN_FORWARD", "") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    applied: List[str] = []
    for key, value in PROFIT_ENV_DEFAULTS.items():
        if clean and key in CLEAN_FORWARD_PROTECT:
            continue
        empty = not str(environ.get(key, "")).strip()
        force = key in FORCE_KEYS and not (clean and key in CLEAN_FORWARD_PROTECT)
        if empty or force:
            if str(environ.get(key, "")) != value:
                environ[key] = value
                applied.append(key)
    return applied


def profit_mode_applied_summary(applied: Iterable[str]) -> str:
    keys = list(applied)
    if not keys:
        return "Profit mode defaults: none (env already set or disabled)"
    return "Profit mode defaults applied: " + ", ".join(keys)
