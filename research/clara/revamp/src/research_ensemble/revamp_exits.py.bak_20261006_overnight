"""
REVAMP exit flags (overnight research winner 2026-10-06,
`combo_tri__trail_mult4.0__tgt_breakout_r2.0__tgt_mr_bb_far`).

OFF by default -> original behaviour (trail 2.0 ATR, breakout trailed, MR target = mid BB).
Turn on with env REVAMP_EXITS=1 (or pass revamp_exits=True / trail_atr_mult=...).

  REVAMP_EXITS=1                 master switch (default 0)
  REVAMP_TRAIL_ATR_MULT=4.0      engine ATR trail multiple when revamp on (orig 2.0)
  REVAMP_BREAKOUT_TARGET_R=2.0   breakout sleeve fixed target in R (0/none -> keep trailing)
  REVAMP_MR_TARGET=bb_far        MR target: bb_far (opposite band) | bb_mid (original)
"""
from __future__ import annotations

import os

ORIGINAL_TRAIL_ATR_MULT = 2.0


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


def revamp_mr_target() -> str:
    v = os.getenv("REVAMP_MR_TARGET", "bb_far").strip().lower()
    if v not in ("bb_far", "bb_mid"):
        raise ValueError(f"REVAMP_MR_TARGET must be bb_far or bb_mid, got {v!r}")
    return v


def describe() -> str:
    if not revamp_enabled():
        return f"REVAMP_EXITS=0 (original): trail {ORIGINAL_TRAIL_ATR_MULT} ATR, breakout trailed, MR target bb_mid"
    r = revamp_breakout_target_r()
    return (f"REVAMP_EXITS=1: trail {revamp_trail_mult()} ATR, breakout "
            f"{'fixed ' + str(r) + 'R target (no trail)' if r else 'trailed'}, MR target {revamp_mr_target()}")
