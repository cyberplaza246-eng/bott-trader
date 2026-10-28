"""
Flow-confirmed counter-trend entries — relax DI/ADX when order flow contradicts 5M EMA trend.

Used by live bot (tick flow) and backtest (1M bar volume proxy).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw in ("true", "1", "yes", "on"):
        return True
    if raw in ("false", "0", "no", "off"):
        return False
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def resolve_flow_counter_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge mnq_profit_config flow_counter_trend block with env overrides."""
    cfg = cfg or {}
    block = dict(cfg.get("flow_counter_trend") or {})

    enabled = block.get("enabled", True)
    if os.getenv("FLOW_COUNTER_TREND", "").strip():
        enabled = _env_bool("FLOW_COUNTER_TREND", enabled)

    return {
        "enabled": enabled,
        "di_margin": _env_float(
            "FLOW_COUNTER_DI_MARGIN",
            float(block.get("di_margin", block.get("flow_counter_di_margin", 5.0))),
        ),
        "di_margin_strong_delta": _env_float(
            "FLOW_COUNTER_DI_MARGIN_STRONG",
            float(block.get("di_margin_strong_delta", 3.0)),
        ),
        "delta_strong_abs": _env_float(
            "FLOW_COUNTER_DELTA_STRONG_ABS",
            float(block.get("delta_strong_abs", 150.0)),
        ),
        "regime_di_margin": _env_float(
            "FLOW_COUNTER_REGIME_DI_MARGIN",
            float(block.get("regime_di_margin", 3.0)),
        ),
        "adx_min": _env_int(
            "FLOW_COUNTER_ADX_MIN",
            int(block.get("adx_min", block.get("flow_counter_adx_min", 17))),
        ),
        "delta_min": _env_float(
            "FLOW_COUNTER_DELTA_MIN",
            float(block.get("delta_min", block.get("flow_counter_delta_min", 100.0))),
        ),
        "delta_min_extreme": _env_float(
            "FLOW_COUNTER_DELTA_MIN_EXTREME",
            float(block.get("delta_min_extreme", 50.0)),
        ),
        "buy_pct_extreme_low": float(block.get("buy_pct_extreme_low", 0.40)),
        "buy_pct_extreme_high": float(block.get("buy_pct_extreme_high", 0.60)),
        "buy_pct_short_max": float(
            block.get("buy_pct_short_max", block.get("flow_counter_buy_pct_short", 0.45))
        ),
        "buy_pct_long_min": float(
            block.get("buy_pct_long_min", block.get("flow_counter_buy_pct_long", 0.55))
        ),
        "regime_scans": _env_int(
            "FLOW_COUNTER_REGIME_SCANS",
            int(block.get("regime_scans", block.get("flow_regime_scans", 3))),
        ),
        "di_counter": float(cfg.get("di_counter", 20.0)),
        "counter_adx": int(cfg.get("counter_adx", 25)),
        "counter_trend_shorts": bool(cfg.get("counter_trend_shorts", True)),
        "counter_trend_longs": bool(cfg.get("counter_trend_longs", True)),
    }


def _effective_delta_min(fc: Dict[str, Any], snapshot: Optional[Dict[str, Any]]) -> float:
    """Lower delta bar when buy% is extreme (<40% or >60%)."""
    base = float(fc.get("delta_min", 100.0))
    if not snapshot:
        return base
    buy_pct = float(snapshot.get("buy_pct", 0.5))
    low = float(fc.get("buy_pct_extreme_low", 0.40))
    high = float(fc.get("buy_pct_extreme_high", 0.60))
    if buy_pct <= low or buy_pct >= high:
        return float(fc.get("delta_min_extreme", 50.0))
    return base


def _effective_di_margin(
    fc: Dict[str, Any],
    snapshot: Optional[Dict[str, Any]],
    *,
    regime: bool = False,
) -> float:
    """Flow-confirmed DI gap: 5 default, 3 when |delta| is large or regime streak override."""
    if regime:
        return float(fc.get("regime_di_margin", 3.0))
    margin = float(fc.get("di_margin", 5.0))
    if snapshot:
        delta_abs = abs(float(snapshot.get("delta", 0)))
        if delta_abs >= float(fc.get("delta_strong_abs", 150.0)):
            margin = min(margin, float(fc.get("di_margin_strong_delta", 3.0)))
    return margin


def leans_direction(
    direction: str,
    snapshot: Optional[Dict[str, Any]],
    fc: Dict[str, Any],
    *,
    min_ticks: int = 3,
) -> bool:
    """Softer flow lean for regime streak counting (delta sign + buy%)."""
    if not snapshot or int(snapshot.get("tick_count", 0)) < min_ticks:
        return False
    delta = float(snapshot.get("delta", 0))
    buy_pct = float(snapshot.get("buy_pct", 0.5))
    if direction in ("short", "sell"):
        return delta < 0 and buy_pct <= float(fc.get("buy_pct_lean_short_max", 0.50))
    return delta > 0 and buy_pct >= float(fc.get("buy_pct_lean_long_min", 0.50))


def strongly_confirms_direction(
    direction: str,
    snapshot: Optional[Dict[str, Any]],
    fc: Dict[str, Any],
    *,
    min_ticks: int = 3,
) -> bool:
    """Strong flow confirmation for counter-trend / regime override."""
    if not snapshot or int(snapshot.get("tick_count", 0)) < min_ticks:
        return False
    delta = float(snapshot.get("delta", 0))
    buy_pct = float(snapshot.get("buy_pct", 0.5))
    delta_min = _effective_delta_min(fc, snapshot)

    if direction in ("short", "sell"):
        return delta <= -delta_min and buy_pct <= float(fc.get("buy_pct_short_max", 0.45))
    return delta >= delta_min and buy_pct >= float(fc.get("buy_pct_long_min", 0.55))


def flow_contradicts_5m_trend(trend_5m: str, snapshot: Optional[Dict[str, Any]], fc: Dict[str, Any]) -> bool:
    """True when rolling flow leans against 5M EMA trend (regime streak input)."""
    if trend_5m == "bullish":
        return leans_direction("short", snapshot, fc)
    if trend_5m == "bearish":
        return leans_direction("long", snapshot, fc)
    return False


def evaluate_counter_trend(
    direction: str,
    ctx_5m: Dict[str, Any],
    flow_snap: Optional[Dict[str, Any]],
    fc: Dict[str, Any],
    *,
    flow_streak: int = 0,
) -> Tuple[bool, str, str]:
    """
    Decide if counter-trend entry is allowed.

    Returns (is_counter, allow_reason, near_miss_detail).
    """
    trend = ctx_5m.get("trend")
    adx = float(ctx_5m.get("adx", 0))
    di_plus = float(ctx_5m.get("di_plus", 0))
    di_minus = float(ctx_5m.get("di_minus", 0))

    if direction == "short":
        if not fc.get("counter_trend_shorts", True):
            return False, "", "counter-trend shorts disabled"
        if trend != "bullish":
            return False, "", ""
        di_diff = di_minus - di_plus
        want_di = fc["di_counter"]
        want_adx = fc["counter_adx"]
        standard = di_diff >= want_di and adx >= want_adx
        if standard:
            return True, "DI+ADX counter-trend", ""

        if not fc.get("enabled", True):
            return False, "", _near_miss_short(di_diff, adx, want_di, want_adx, flow_snap, fc)

        flow_strong = strongly_confirms_direction("short", flow_snap, fc)
        relaxed_di = _effective_di_margin(fc, flow_snap)
        regime_di = _effective_di_margin(fc, flow_snap, regime=True)
        relaxed_adx = int(fc["adx_min"])
        flow_ok = flow_strong and di_diff >= relaxed_di and adx >= relaxed_adx
        regime_ok = (
            flow_streak >= int(fc.get("regime_scans", 3))
            and leans_direction("short", flow_snap, fc)
            and di_diff >= regime_di
            and adx >= relaxed_adx
        )

        if flow_ok:
            snap = flow_snap or {}
            reason = (
                f"flow + DI confirm despite bullish 5M EMA "
                f"(DI−+{di_diff:.0f}, ADX {adx:.0f}, "
                f"delta={snap.get('delta', 0):+.0f}, buy={float(snap.get('buy_pct', 0.5)):.0%})"
            )
            return True, reason, ""
        if regime_ok:
            reason = (
                f"flow regime override ({flow_streak} scans) despite bullish 5M "
                f"(DI−+{di_diff:.0f}, ADX {adx:.0f})"
            )
            return True, reason, ""

        return False, "", _near_miss_short(
            di_diff, adx, want_di, want_adx, flow_snap, fc, flow_strong=flow_strong,
        )

    # long counter-trend in bearish 5M
    if not fc.get("counter_trend_longs", True):
        return False, "", "counter-trend longs disabled"
    if trend != "bearish":
        return False, "", ""
    di_diff = di_plus - di_minus
    want_di = fc["di_counter"]
    want_adx = fc["counter_adx"]
    standard = di_diff >= want_di and adx >= want_adx
    if standard:
        return True, "DI+ADX counter-trend", ""

    if not fc.get("enabled", True):
        return False, "", _near_miss_long(di_diff, adx, want_di, want_adx, flow_snap, fc)

    flow_strong = strongly_confirms_direction("long", flow_snap, fc)
    relaxed_di = _effective_di_margin(fc, flow_snap)
    regime_di = _effective_di_margin(fc, flow_snap, regime=True)
    relaxed_adx = int(fc["adx_min"])
    flow_ok = flow_strong and di_diff >= relaxed_di and adx >= relaxed_adx
    regime_ok = (
        flow_streak >= int(fc.get("regime_scans", 3))
        and leans_direction("long", flow_snap, fc)
        and di_diff >= regime_di
        and adx >= relaxed_adx
    )

    if flow_ok:
        snap = flow_snap or {}
        reason = (
            f"flow + DI confirm despite bearish 5M EMA "
            f"(DI++{di_diff:.0f}, ADX {adx:.0f}, "
            f"delta={snap.get('delta', 0):+.0f}, buy={float(snap.get('buy_pct', 0.5)):.0%})"
        )
        return True, reason, ""
    if regime_ok:
        reason = (
            f"flow regime override ({flow_streak} scans) despite bearish 5M "
            f"(DI++{di_diff:.0f}, ADX {adx:.0f})"
        )
        return True, reason, ""

    return False, "", _near_miss_long(
        di_diff, adx, want_di, want_adx, flow_snap, fc, flow_strong=flow_strong,
    )


def _near_miss_short(
    di_diff: float,
    adx: float,
    want_di: float,
    want_adx: int,
    flow_snap: Optional[Dict[str, Any]],
    fc: Dict[str, Any],
    *,
    flow_strong: bool = False,
) -> str:
    parts = []
    if di_diff < want_di:
        gap = want_di - di_diff
        parts.append(f"DI margin {di_diff:.0f}/{want_di:.0f} (need +{gap:.0f})")
    if adx < want_adx:
        parts.append(f"ADX {adx:.0f}/{want_adx}")
    if fc.get("enabled") and flow_snap:
        snap = flow_snap
        delta = float(snap.get("delta", 0))
        buy_pct = float(snap.get("buy_pct", 0.5))
        relaxed_di = _effective_di_margin(fc, snap)
        regime_di = _effective_di_margin(fc, snap, regime=True)
        relaxed_adx = int(fc["adx_min"])
        eff_delta = _effective_delta_min(fc, snap)
        if strongly_confirms_direction("short", snap, fc):
            flow_strong = True
        if flow_strong:
            if di_diff >= relaxed_di and adx >= relaxed_adx:
                parts.append("flow confirms sell — thresholds met (check FLOW_COUNTER_TREND)")
            else:
                miss = []
                if di_diff < relaxed_di:
                    miss.append(f"flow DI {di_diff:.0f}/{relaxed_di:.0f}")
                if adx < relaxed_adx:
                    miss.append(f"ADX {adx:.0f}/{relaxed_adx}")
                parts.append(
                    f"flow confirms sell (delta={delta:+.0f}, {100 * (1 - buy_pct):.0f}% sell) "
                    f"but {', '.join(miss)}"
                )
        else:
            extreme = buy_pct <= float(fc.get("buy_pct_extreme_low", 0.40))
            delta_note = f"delta≤-{eff_delta:.0f}" + (" (extreme buy%)" if extreme else "")
            parts.append(
                f"flow weak for counter SHORT (delta={delta:+.0f}, buy={buy_pct:.0%}; "
                f"need {delta_note} & buy≤{fc['buy_pct_short_max']:.0%})"
            )
        lean = leans_direction("short", snap, fc)
        if lean and di_diff >= regime_di and adx >= relaxed_adx:
            parts.append(
                f"regime path: DI≥{regime_di:.0f} after {fc.get('regime_scans', 3)} lean-sell scans"
            )
        elif di_diff < regime_di:
            parts.append(f"regime DI {di_diff:.0f}/{regime_di:.0f} (need {fc.get('regime_scans', 3)}+ lean-sell scans)")
        elif not lean:
            parts.append(f"flow not leaning sell (delta={delta:+.0f}, buy={buy_pct:.0%})")
    if not parts:
        return "5M bullish — no counter-trend SHORT path"
    return "Counter-trend SHORT blocked: " + "; ".join(parts)


def _near_miss_long(
    di_diff: float,
    adx: float,
    want_di: float,
    want_adx: int,
    flow_snap: Optional[Dict[str, Any]],
    fc: Dict[str, Any],
    *,
    flow_strong: bool = False,
) -> str:
    parts = []
    if di_diff < want_di:
        gap = want_di - di_diff
        parts.append(f"DI margin {di_diff:.0f}/{want_di:.0f} (need +{gap:.0f})")
    if adx < want_adx:
        parts.append(f"ADX {adx:.0f}/{want_adx}")
    if fc.get("enabled") and flow_snap:
        snap = flow_snap
        delta = float(snap.get("delta", 0))
        buy_pct = float(snap.get("buy_pct", 0.5))
        relaxed_di = _effective_di_margin(fc, snap)
        regime_di = _effective_di_margin(fc, snap, regime=True)
        relaxed_adx = int(fc["adx_min"])
        eff_delta = _effective_delta_min(fc, snap)
        if strongly_confirms_direction("long", snap, fc):
            flow_strong = True
        if flow_strong:
            if di_diff >= relaxed_di and adx >= relaxed_adx:
                parts.append("flow confirms buy — thresholds met (check FLOW_COUNTER_TREND)")
            else:
                miss = []
                if di_diff < relaxed_di:
                    miss.append(f"flow DI {di_diff:.0f}/{relaxed_di:.0f}")
                if adx < relaxed_adx:
                    miss.append(f"ADX {adx:.0f}/{relaxed_adx}")
                parts.append(
                    f"flow confirms buy (delta={delta:+.0f}, buy={buy_pct:.0%}) "
                    f"but {', '.join(miss)}"
                )
        else:
            extreme = buy_pct >= float(fc.get("buy_pct_extreme_high", 0.60))
            delta_note = f"delta≥+{eff_delta:.0f}" + (" (extreme buy%)" if extreme else "")
            parts.append(
                f"flow weak for counter LONG (delta={delta:+.0f}, buy={buy_pct:.0%}; "
                f"need {delta_note} & buy≥{fc['buy_pct_long_min']:.0%})"
            )
        lean = leans_direction("long", snap, fc)
        if lean and di_diff >= regime_di and adx >= relaxed_adx:
            parts.append(
                f"regime path: DI≥{regime_di:.0f} after {fc.get('regime_scans', 3)} lean-buy scans"
            )
        elif di_diff < regime_di:
            parts.append(f"regime DI {di_diff:.0f}/{regime_di:.0f} (need {fc.get('regime_scans', 3)}+ lean-buy scans)")
        elif not lean:
            parts.append(f"flow not leaning buy (delta={delta:+.0f}, buy={buy_pct:.0%})")
    if not parts:
        return "5M bearish — no counter-trend LONG path"
    return "Counter-trend LONG blocked: " + "; ".join(parts)


def resolve_flow_entry_guard(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge flow_entry_guard block from profit config with env overrides."""
    cfg = cfg or {}
    block = dict(cfg.get("flow_entry_guard") or {})
    enabled = block.get("enabled", True)
    if os.getenv("FLOW_ENTRY_GUARD", "").strip():
        enabled = _env_bool("FLOW_ENTRY_GUARD", enabled)
    return {
        "enabled": enabled,
        "long_buy_pct_min": float(
            block.get("long_buy_pct_min", _env_float("FLOW_LONG_BUY_PCT_MIN", 0.48))
        ),
        "short_buy_pct_max": float(
            block.get("short_buy_pct_max", _env_float("FLOW_SHORT_BUY_PCT_MAX", 0.52))
        ),
        "min_ticks": _env_int("FLOW_ENTRY_GUARD_MIN_TICKS", int(block.get("min_ticks", 3))),
    }


def flow_blocks_long(
    snapshot: Optional[Dict[str, Any]],
    guard: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Block trend long when sell flow dominates (delta<0 and buy% below threshold)."""
    guard = guard or resolve_flow_entry_guard()
    if not guard.get("enabled", True):
        return False, ""
    min_ticks = int(guard.get("min_ticks", 3))
    if not snapshot or int(snapshot.get("tick_count", 0)) < min_ticks:
        return False, ""
    delta = float(snapshot.get("delta", 0))
    buy_pct = float(snapshot.get("buy_pct", 0.5))
    buy_min = float(guard.get("long_buy_pct_min", 0.48))
    if delta < 0 and buy_pct < buy_min:
        sell_pct = 1.0 - buy_pct
        return True, (
            f"Sell flow blocks LONG (delta={delta:+.0f}, "
            f"{sell_pct:.0%} sell, need buy≥{buy_min:.0%})"
        )
    return False, ""


def flow_blocks_short(
    snapshot: Optional[Dict[str, Any]],
    guard: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Block trend short when buy flow dominates (delta>0 and buy% above threshold)."""
    guard = guard or resolve_flow_entry_guard()
    if not guard.get("enabled", True):
        return False, ""
    min_ticks = int(guard.get("min_ticks", 3))
    if not snapshot or int(snapshot.get("tick_count", 0)) < min_ticks:
        return False, ""
    delta = float(snapshot.get("delta", 0))
    buy_pct = float(snapshot.get("buy_pct", 0.5))
    buy_max = float(guard.get("short_buy_pct_max", 0.52))
    if delta > 0 and buy_pct > buy_max:
        return True, (
            f"Buy flow blocks SHORT (delta={delta:+.0f}, "
            f"buy={buy_pct:.0%}, need buy≤{buy_max:.0%})"
        )
    return False, ""


def flow_confirms_long_direction(
    snapshot: Optional[Dict[str, Any]],
    guard: Optional[Dict[str, Any]] = None,
) -> bool:
    """True when flow is not blocking a trend long."""
    blocked, _ = flow_blocks_long(snapshot, guard)
    return not blocked


def proxy_flow_from_bars(df_1m, window: int = 5) -> Dict[str, Any]:
    """Backtest proxy: classify recent 1M bars by direction × volume."""
    if df_1m is None or len(df_1m) < window:
        return {"tick_count": 0, "delta": 0.0, "buy_pct": 0.5, "sell_pct": 0.5}
    recent = df_1m.tail(window)
    buy_vol = 0.0
    sell_vol = 0.0
    for _, bar in recent.iterrows():
        vol = float(bar.get("volume", 0) or 0)
        if vol <= 0:
            vol = 1.0
        if float(bar["close"]) >= float(bar["open"]):
            buy_vol += vol
        else:
            sell_vol += vol
    total = buy_vol + sell_vol
    delta = buy_vol - sell_vol
    buy_pct = (buy_vol / total) if total > 0 else 0.5
    return {
        "tick_count": len(recent),
        "delta": delta,
        "buy_vol": buy_vol,
        "sell_vol": sell_vol,
        "buy_pct": buy_pct,
        "sell_pct": 1.0 - buy_pct,
        "window_sec": window * 60,
    }
