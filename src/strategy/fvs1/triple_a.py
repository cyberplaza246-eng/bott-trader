"""FVS-1 Triple-A state machine: Layer 0 → 1 → 2 → 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.strategy.entry_diagnostics import gate_line
from src.strategy.fvs1.absorption import AbsorptionZone, detect_absorption_sequence
from src.strategy.fvs1.config import FVS1Config, is_fvs1_session_et
from src.strategy.fvs1.volume_profile import (
    ImpulseLeg,
    VolumeProfile,
    build_volume_profile,
    detect_out_of_balance,
    find_recent_impulse_leg,
    price_at_value_edge,
)


LAYER_IDLE = 0
LAYER0_OK = 1
LAYER1_OK = 2
LAYER2_OK = 3


@dataclass
class FVS1SideState:
    layer: int = LAYER_IDLE
    direction: int = 0
    impulse: Optional[ImpulseLeg] = None
    profile: Optional[VolumeProfile] = None
    absorption: Optional[AbsorptionZone] = None
    entry_level: float = 0.0


@dataclass
class FVS1State:
    long: FVS1SideState = field(default_factory=FVS1SideState)
    short: FVS1SideState = field(default_factory=FVS1SideState)


@dataclass
class FVS1RiskState:
    consecutive_losses: int = 0
    recent_pnls: List[float] = field(default_factory=list)
    session_halted: bool = False

    def record_trade(self, pnl: float, cfg: FVS1Config) -> None:
        self.recent_pnls.append(pnl)
        if len(self.recent_pnls) > cfg.expectancy_window:
            self.recent_pnls = self.recent_pnls[-cfg.expectancy_window:]
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        if self.consecutive_losses >= cfg.max_consecutive_losses:
            self.session_halted = True

    def expectancy(self) -> float:
        if not self.recent_pnls:
            return 0.0
        return sum(self.recent_pnls) / len(self.recent_pnls)

    def risk_blocked(self, cfg: FVS1Config) -> Tuple[bool, str]:
        if self.session_halted:
            return True, f"{cfg.max_consecutive_losses} consecutive losses — session halt"
        if cfg.pause_on_negative_e and len(self.recent_pnls) >= 10:
            e = self.expectancy()
            if e < 0:
                return True, f"negative expectancy E={e:.2f} over {len(self.recent_pnls)} trades"
        return False, ""


def _copy_state(state: FVS1State) -> FVS1State:
    import copy
    return copy.deepcopy(state)


def _side_state(state: FVS1State, direction: int) -> FVS1SideState:
    return state.long if direction == 1 else state.short


def _infer_direction(row_5m: pd.Series, adx_min: int) -> int:
    adx = float(row_5m.get("adx", 0) or 0)
    if pd.isna(adx) or adx < adx_min:
        return 0
    di_p = float(row_5m.get("di_plus", 0) or 0)
    di_m = float(row_5m.get("di_minus", 0) or 0)
    if di_p > di_m + 2:
        return 1
    if di_m > di_p + 2:
        return -1
    close = float(row_5m["close"])
    vwap = float(row_5m.get("vwap", float("nan")))
    if not pd.isna(vwap):
        if close > vwap:
            return 1
        if close < vwap:
            return -1
    return 0


def _aggression_trigger(
    row_30s: pd.Series,
    prev_30s: pd.Series,
    direction: int,
    absorption: AbsorptionZone,
    body_pct_min: float,
) -> Tuple[bool, str]:
    o, c = float(row_30s["open"]), float(row_30s["close"])
    h, l = float(row_30s["high"]), float(row_30s["low"])
    rng = h - l if h > l else 1e-9
    body = abs(c - o) / rng
    if direction == 1:
        if c <= o:
            return False, "30s not bullish"
        if c <= float(prev_30s["high"]):
            return False, "no break of prev 30s high"
        if c <= absorption.zone_high:
            return False, "close still inside absorption zone"
        if body < body_pct_min:
            return False, f"body {body:.0%} < {body_pct_min:.0%}"
        return True, f"aggressive break above {absorption.zone_high:.2f}"
    if c >= o:
        return False, "30s not bearish"
    if c >= float(prev_30s["low"]):
        return False, "no break of prev 30s low"
    if c >= absorption.zone_low:
        return False, "close still inside absorption zone"
    if body < body_pct_min:
        return False, f"body {body:.0%} < {body_pct_min:.0%}"
    return True, f"aggressive break below {absorption.zone_low:.2f}"


def _structural_brackets(
    direction: int,
    entry: float,
    profile: VolumeProfile,
    absorption: AbsorptionZone,
    cfg: FVS1Config,
) -> Tuple[float, float]:
    if direction == 1:
        sl = absorption.zone_low - cfg.sl_buffer_pts
        tp = profile.vah
        if tp <= entry:
            tp = entry + (entry - sl) * cfg.min_rr
    else:
        sl = absorption.zone_high + cfg.sl_buffer_pts
        tp = profile.val
        if tp >= entry:
            tp = entry - (sl - entry) * cfg.min_rr
    return sl, tp


def _evaluate_side_gates(
    side: str,
    direction: int,
    df_trigger: pd.DataFrame,
    idx_trigger: int,
    row_5m: pd.Series,
    row_30s: Optional[pd.Series],
    prev_30s: Optional[pd.Series],
    side_st: FVS1SideState,
    cfg: FVS1Config,
    flow_snap: Optional[Dict] = None,
) -> List[str]:
    lines: List[str] = []

    # Layer 0
    adx = float(row_5m.get("adx", 0) or 0)
    if pd.isna(adx) or adx < cfg.adx_min:
        lines.append(gate_line(False, side, f"[L0] ADX {adx:.0f} < {cfg.adx_min}"))
        side_st.layer = LAYER_IDLE
        return lines
    oob, oob_reason = detect_out_of_balance(
        df_trigger, idx_trigger, lookback=min(cfg.balance_lookback, idx_trigger),
        atr_mult=cfg.balance_atr_mult,
    )
    if not oob:
        lines.append(gate_line(False, side, f"[L0] balanced market — {oob_reason}"))
        side_st.layer = LAYER_IDLE
        return lines
    leg = find_recent_impulse_leg(
        df_trigger, idx_trigger, direction,
        min_bars=cfg.impulse_min_bars, min_body_pct=cfg.impulse_min_body_pct,
        max_scan=cfg.impulse_max_bars,
    )
    if leg is None:
        lines.append(gate_line(False, side, f"[L0] no recent {side} impulse leg (≥{cfg.impulse_min_bars} bars)"))
        side_st.layer = LAYER_IDLE
        return lines
    close = float(df_trigger.iloc[idx_trigger]["close"])
    lines.append(gate_line(
        True, side,
        f"[L0] out of balance + impulse ({oob_reason}; leg ends bar {leg.end_idx})",
    ))
    side_st.layer = LAYER0_OK
    side_st.direction = direction
    side_st.impulse = leg

    # Layer 1
    profile = build_volume_profile(
        df_trigger, leg,
        n_bins=cfg.vp_bins,
        value_area_pct=cfg.value_area_pct,
        lvn_percentile=cfg.lvn_percentile,
    )
    if profile is None:
        lines.append(gate_line(False, side, "[L1] volume profile failed"))
        return lines
    at_edge, level, edge_reason = price_at_value_edge(
        close, profile, direction, cfg.absorption_level_tol_pts,
    )
    if not at_edge:
        lines.append(gate_line(False, side, f"[L1] not at value edge — {edge_reason}"))
        return lines
    side_st.profile = profile
    lvn_note = f", {len(profile.lvn_prices)} LVN" if profile.lvn_prices else ""
    lines.append(gate_line(
        True, side,
        f"[L1] VP POC={profile.poc:.2f} VAL={profile.val:.2f} VAH={profile.vah:.2f}{lvn_note}",
    ))
    side_st.layer = LAYER1_OK

    # Layer 2 — absorption completes before trigger bar
    abs_end_idx = max(0, idx_trigger - 1)
    absorption = detect_absorption_sequence(
        df_trigger, abs_end_idx, level, direction,
        min_bars=cfg.absorption_min_bars,
        vol_mult=cfg.absorption_vol_mult,
        range_mult=cfg.absorption_range_mult,
        tol_pts=cfg.absorption_level_tol_pts,
    )
    if absorption is None:
        lines.append(gate_line(
            False, side,
            f"[L2] no absorption at {'VAL' if direction == 1 else 'VAH'} {level:.2f} "
            f"(need {cfg.absorption_min_bars}+ bars)",
        ))
        return lines
    side_st.absorption = absorption
    lines.append(gate_line(
        True, side,
        f"[L2] absorption {absorption.bar_count} bars at {level:.2f} "
        f"[{absorption.zone_low:.2f}-{absorption.zone_high:.2f}]",
    ))
    side_st.layer = LAYER2_OK

    # Layer 3
    if row_30s is None or prev_30s is None:
        lines.append(gate_line(False, side, "[L3] 30s bars unavailable"))
        return lines
    fired, trig_reason = _aggression_trigger(
        row_30s, prev_30s, direction, absorption, cfg.aggression_body_pct,
    )
    if not fired:
        lines.append(gate_line(False, side, f"[L3] waiting aggression — {trig_reason}"))
        return lines
    if flow_snap:
        delta = float(flow_snap.get("delta", 0) or 0)
        buy_pct = float(flow_snap.get("buy_pct", 0.5) or 0.5)
        if direction == 1 and delta < 0 and buy_pct < 0.48:
            lines.append(gate_line(False, side, f"[L3] flow contradicts long (delta={delta:.0f})"))
            return lines
        if direction == -1 and delta > 0 and buy_pct > 0.52:
            lines.append(gate_line(False, side, f"[L3] flow contradicts short (delta={delta:.0f})"))
            return lines

    entry = float(row_30s["close"])
    sl, tp = _structural_brackets(direction, entry, profile, absorption, cfg)
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    rr = reward / risk if risk > 0 else 0
    if rr < cfg.min_rr:
        lines.append(gate_line(False, side, f"[L3] R:R {rr:.2f} < {cfg.min_rr} (SL={sl:.2f} TP={tp:.2f})"))
        return lines
    side_st.entry_level = entry
    side_st.layer = LAYER2_OK  # ready
    lines.append(gate_line(True, side, f"[L3] aggression — {trig_reason}"))
    lines.append(gate_line(
        True, side,
        f"[L3] structural SL={sl:.2f} TP={tp:.2f} R:R={rr:.2f} max_hold={cfg.max_hold_seconds}s",
    ))
    lines.append(gate_line(True, side, "[L3] ALL GATES PASSED — FVS-1 entry ready"))
    return lines


def evaluate_fvs1_gates(
    df_1m: pd.DataFrame,
    row_5m: pd.Series,
    row_30s: Optional[pd.Series],
    prev_30s: Optional[pd.Series],
    state: FVS1State,
    cfg: FVS1Config,
    *,
    df_30s: Optional[pd.DataFrame] = None,
    now_et=None,
    flow_snap: Optional[Dict] = None,
    risk: Optional[FVS1RiskState] = None,
) -> Tuple[List[str], FVS1State]:
    st = _copy_state(state)
    lines: List[str] = []
    lines.append("   --- FVS-1 Triple-A (Fabio Valentini Scalper) ---")

    in_sess, sess_label = is_fvs1_session_et(now_et, cfg.sessions)
    if not in_sess:
        lines.append(gate_line(False, "GLOBAL", f"outside FVS1 session ({sess_label})"))
        return lines, st
    lines.append(gate_line(True, "GLOBAL", f"FVS1 session active ({sess_label})"))

    if risk is not None:
        blocked, reason = risk.risk_blocked(cfg)
        if blocked:
            lines.append(gate_line(False, "GLOBAL", reason))
            return lines, st
        if risk.recent_pnls:
            lines.append(gate_line(
                True, "GLOBAL",
                f"expectancy E={risk.expectancy():.2f} over {len(risk.recent_pnls)} trades",
            ))

    idx_1m = len(df_1m) - 1
    if df_30s is not None and len(df_30s) >= cfg.impulse_min_bars + 5:
        df_trigger = df_30s
        idx_trigger = len(df_30s) - 1
    else:
        df_trigger = df_1m
        idx_trigger = idx_1m
    for direction in (1, -1):
        side = "long" if direction == 1 else "short"
        side_st = _side_state(st, direction)
        lines.extend(_evaluate_side_gates(
            side, direction, df_trigger, idx_trigger, row_5m, row_30s, prev_30s,
            side_st, cfg, flow_snap=flow_snap,
        ))
    return lines, st


def check_fvs1_entry(
    symbol: str,
    df_1m: pd.DataFrame,
    row_5m: pd.Series,
    row_30s: pd.Series,
    prev_30s: pd.Series,
    state: FVS1State,
    cfg: FVS1Config,
    *,
    df_30s: Optional[pd.DataFrame] = None,
    now_et=None,
    flow_snap: Optional[Dict] = None,
    risk: Optional[FVS1RiskState] = None,
) -> Tuple[Optional[Dict[str, Any]], FVS1State]:
    """Return signal dict when all layers pass, else None."""
    if cfg.log_only:
        return None, state

    in_sess, _ = is_fvs1_session_et(now_et, cfg.sessions)
    if not in_sess:
        return None, state
    if risk is not None:
        blocked, _ = risk.risk_blocked(cfg)
        if blocked:
            return None, state

    st = _copy_state(state)
    idx_1m = len(df_1m) - 1
    if df_30s is not None and len(df_30s) >= cfg.impulse_min_bars + 5:
        df_trigger = df_30s
        idx_trigger = len(df_30s) - 1
    else:
        df_trigger = df_1m
        idx_trigger = idx_1m

    for direction in (1, -1):
        side_st = _side_state(st, direction)
        lines = _evaluate_side_gates(
            "long" if direction == 1 else "short",
            direction, df_trigger, idx_trigger, row_5m, row_30s, prev_30s,
            side_st, cfg, flow_snap=flow_snap,
        )
        if not any("ALL GATES PASSED" in ln for ln in lines):
            continue
        profile = side_st.profile
        absorption = side_st.absorption
        if profile is None or absorption is None:
            continue
        entry = float(row_30s["close"])
        sl, tp = _structural_brackets(direction, entry, profile, absorption, cfg)
        dir_str = "long" if direction == 1 else "short"
        atr = float(df_1m.iloc[-1].get("atr", 0) or 0)
        return {
            "symbol": symbol,
            "direction": dir_str,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "atr": atr if not pd.isna(atr) else abs(entry - sl),
            "scalp_mode": "fvs1",
            "fvs1": True,
            "structure_capped": True,
            "max_hold_seconds": cfg.max_hold_seconds,
            "poc": profile.poc,
            "val": profile.val,
            "vah": profile.vah,
        }, st

    return None, st
