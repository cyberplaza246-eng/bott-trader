"""
REVAMP exit flags (overnight research winner 2026-10-06,
`combo_tri__trail_mult4.0__tgt_breakout_r2.0__tgt_mr_bb_far`).

OFF by default -> original behaviour (trail 2.0 ATR, breakout trailed, MR target = mid BB).
Turn on with env REVAMP_EXITS=1 (or pass revamp_exits=True / trail_atr_mult=...).

  REVAMP_EXITS=1                 master switch (default 0)
  REVAMP_TRAIL_ATR_MULT=4.0      engine ATR trail multiple when revamp on (orig 2.0)
  REVAMP_BREAKOUT_TARGET_R=2.0   breakout sleeve fixed target in R (0/none -> keep trailing)
  REVAMP_MR_TARGET=bb_far        MR target: bb_far (opposite band) | bb_mid (original)
  REVAMP_TREND_TP_R=1.5          trend sleeve fixed TP in R (entry +/- R x stop distance);
                                 0/none/off -> old trail-only trend (no TP). Added 2026-10-08.

Session / overnight entry gate (ensemble rth_only):
  REVAMP_RTH_ONLY / RTH_ONLY     explicit True/False (wins if set)
  SESSION_MODE=extended          OR OVERNIGHT_TRADING=true -> allow Globex overnight entries
  default                        RTH-only 09:30-16:00 ET
"""
from __future__ import annotations

import os

ORIGINAL_TRAIL_ATR_MULT = 2.0

_EXTENDED_SESSION_ALIASES = frozenset(
    ("extended", "overnight", "globex", "extended_hours", "24h")
)


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def revamp_enabled() -> bool:
    return _truthy(os.getenv("REVAMP_EXITS", "0"))


def revamp_trail_mult() -> float:
    return float(os.getenv("REVAMP_TRAIL_ATR_MULT", "4.0"))


def revamp_breakout_target_r() -> float | None:
    v = os.getenv("REVAMP_BREAKOUT_TARGET_R", "2.0").strip().lower()
    if v in ("", "0", "none", "off", "trail"):
        return None
    return float(v)


def revamp_trend_target_r() -> float | None:
    """Trend sleeve fixed target in R (default 1.5). None -> trail-only (old behaviour)."""
    v = os.getenv("REVAMP_TREND_TP_R", "1.5").strip().lower()
    if v in ("", "0", "0.0", "none", "off", "trail"):
        return None
    try:
        r = float(v)
    except ValueError:
        return 1.5
    return r if r > 0 else None


def revamp_mr_target() -> str:
    v = os.getenv("REVAMP_MR_TARGET", "bb_far").strip().lower()
    if v not in ("bb_far", "bb_mid"):
        raise ValueError(f"REVAMP_MR_TARGET must be bb_far or bb_mid, got {v!r}")
    return v


def rth_only_from_env(default: bool = True) -> bool:
    """Whether the research ensemble should zero entries outside 09:30-16:00 ET.

    Priority:
      1. REVAMP_RTH_ONLY or RTH_ONLY if set
      2. SESSION_MODE=extended|overnight|globex... -> False (allow overnight)
      3. OVERNIGHT_TRADING=true -> False
      4. default (True = RTH-only)
    """
    for key in ("REVAMP_RTH_ONLY", "RTH_ONLY"):
        raw = os.getenv(key)
        if raw is not None and str(raw).strip() != "":
            return _truthy(raw)
    session = os.getenv("SESSION_MODE", "").strip().lower()
    if session in _EXTENDED_SESSION_ALIASES:
        return False
    if _truthy(os.getenv("OVERNIGHT_TRADING", "0")):
        return False
    return default


def overnight_entries_enabled() -> bool:
    """True when Globex overnight entries are allowed (rth_only gate off)."""
    return not rth_only_from_env(default=True)


def session_hours_label() -> str:
    if overnight_entries_enabled():
        return (
            "EXTENDED/Globex — Sun 6:00 PM – Fri 5:00 PM ET "
            "(daily halt ~5:00–6:00 PM ET Mon–Thu; no RTH-only entry filter)"
        )
    return "RTH-only — Mon–Fri 09:30–16:00 ET (overnight Globex entries blocked)"


def describe() -> str:
    if not revamp_enabled():
        return f"REVAMP_EXITS=0 (original): trail {ORIGINAL_TRAIL_ATR_MULT} ATR, breakout trailed, MR target bb_mid"
    r = revamp_breakout_target_r()
    base = (
        f"REVAMP_EXITS=1: trail {revamp_trail_mult()} ATR, breakout "
        f"{'fixed ' + str(r) + 'R target (no trail)' if r else 'trailed'}, MR target {revamp_mr_target()}"
        f", trend {('fixed ' + str(revamp_trend_target_r()) + 'R target') if revamp_trend_target_r() else 'trailed'}"
    )
    return f"{base}; session={session_hours_label()}"
