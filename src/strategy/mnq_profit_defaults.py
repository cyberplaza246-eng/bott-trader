"""
Locked MNQ live recipe (docs/PROFITABLE_LIVE.md).

Applied only when data/mnq_profit_config.json has scalp_hybrid.profit_mode
(or enabled) and the operator did not pick a different STRATEGY_MODE / SCALP_MODE.
Env vars already set always win.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

HYBRID_MODE_ALIASES = frozenset({
    "hybrid",
    "scalp_hybrid",
    "pullback",
    "continuation",
})

# Keys filled only when unset — never overwrite an operator .env choice.
PROFIT_ENV_DEFAULTS = {
    "SCALP_MODE": "hybrid",
    "STRATEGY_MODE": "scalp_hybrid",
    "SESSION_MODE": "rth",
    "OVERNIGHT_TRADING": "false",
    "SCALP_SESSIONS": "morning=09:30-12:00;afternoon=13:30-16:00",
    "SCAN_SLEEP_OPEN_SEC": "5",
    "SCAN_SLEEP_IDLE_SEC": "10",
    "USE_ORDER_FLOW": "true",
    "ORDER_FLOW_MODE": "block",
    "MTF_MAX_CONSEC_LOSSES": "3",
    "MTF_CONSEC_LOSS_PAUSE_MINUTES": "30",
    "LOSS_COOLDOWN_MINUTES": "15",
    "SCALP_AGGRESSIVE": "true",
    "SCALP_FAST_MODE": "true",
    "MNQ_MAX_POSITIONS": "2",
    "MAX_HOLD_SECONDS": "30",
    "USE_30S_BARS": "true",
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
    """True when JSON asks for profit mode and user did not pick another strategy."""
    sh = (cfg or {}).get("scalp_hybrid") or {}
    if not (sh.get("profit_mode") or sh.get("enabled")):
        return False
    env = environ if environ is not None else {}
    explicit = _explicit_strategy(env)
    if not explicit:
        return True
    return explicit in HYBRID_MODE_ALIASES


def apply_profit_env_defaults(
    cfg: Optional[Dict],
    environ: MutableMapping[str, str],
) -> List[str]:
    """setdefault locked live knobs. Returns keys that were filled."""
    if not should_lock_profit_mode(cfg, environ):
        return []
    applied: List[str] = []
    for key, value in PROFIT_ENV_DEFAULTS.items():
        if not str(environ.get(key, "")).strip():
            environ[key] = value
            applied.append(key)
    return applied


def profit_mode_applied_summary(applied: Iterable[str]) -> str:
    keys = list(applied)
    if not keys:
        return "Profit mode defaults: none (env already set or disabled)"
    return "Profit mode defaults applied: " + ", ".join(keys)
