"""
Structured PASS/BLOCK gate diagnostics for live MTF entry decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.ai.entry_quality import check_long_entry_quality, check_short_entry_quality, entry_quality_enabled
from src.utils.flow_counter_trend import (
    evaluate_counter_trend,
    flow_blocks_long,
    flow_blocks_short,
    flow_confirms_long_direction,
)
from src.ai.order_flow import TickFlowTracker


def gate_line(passed: bool, direction: str, msg: str) -> str:
    prefix = "PASS" if passed else "BLOCK"
    return f"   [{prefix}] {direction.upper()}: {msg}"


@dataclass
class GateEvalContext:
    """Thresholds and flags passed from start_live_mtf_scalping."""

    adx_threshold: int
    di_tolerance: float
    di_flow_tolerance: float
    di_relax_strength: float
    flow_relax_strength: float
    flow_relax_long_buy_pct: float
    use_order_flow: bool
    use_flow_di_override: bool
    use_flow_adx_relax: bool
    flow_adx_relax: int
    flow_entry_guard: Dict
    flow_counter_cfg: Dict
    use_15m_bias: bool
    use_15m_entry_gate: bool
    candle_confirmation: bool
    volume_ratio_threshold: float
    bb_extreme_low: float
    bb_extreme_high: float
    vwap_required: bool
    entry_quality: Dict
    session_mode: str
    max_1m_bar_pts: float
    strategy_mode: str = "baseline"
    effective_adx_fn: Optional[Callable[..., int]] = None
    adaptive_rsi_long_fn: Optional[Callable[..., Tuple[float, float]]] = None
    adaptive_rsi_short_fn: Optional[Callable[..., Tuple[float, float]]] = None
    adaptive_pullback_atr_fn: Optional[Callable[..., float]] = None
    adaptive_vwap_ok_fn: Optional[Callable[..., bool]] = None
    bias_15m_allows_fn: Optional[Callable[[str, Dict], bool]] = None
    strong_trend_relaxed_fn: Optional[Callable[..., bool]] = None


def _flow_blocks_long_relaxed(
    snapshot: Optional[Dict],
    guard: Dict,
    adx: float,
    relax_strength: float,
    relax_buy_pct: float,
) -> Tuple[bool, str]:
    blocked, reason = flow_blocks_long(snapshot, guard)
    if blocked and adx >= relax_strength and snapshot:
        buy_pct = float(snapshot.get("buy_pct", 0.5))
        if buy_pct >= relax_buy_pct:
            return False, ""
    return blocked, reason


def evaluate_long_gates(
    row: pd.Series,
    ctx_5m: Dict,
    ctx_15m: Dict,
    flow_snap: Optional[Dict],
    flow_streak: int,
    cfg: GateEvalContext,
    df_1m: Optional[pd.DataFrame] = None,
) -> List[str]:
    lines: List[str] = []
    fc = dict(cfg.flow_counter_cfg)
    is_ct, ct_reason, ct_near = evaluate_counter_trend(
        "long", ctx_5m, flow_snap, fc, flow_streak=flow_streak,
    )

    if cfg.use_15m_entry_gate and cfg.use_15m_bias:
        ok = cfg.bias_15m_allows_fn("long", ctx_15m) if cfg.bias_15m_allows_fn else True
        trend_15 = ctx_15m.get("trend")
        lines.append(gate_line(
            ok or is_ct,
            "long",
            f"15M bias {trend_15} aligned for buys" if ok else f"15M bias {trend_15} blocks longs",
        ))
        if not ok and not is_ct:
            return lines
    elif cfg.use_15m_bias:
        lines.append(gate_line(True, "long", f"15M {ctx_15m.get('trend')} (direction only — not gating)"))

    trend = ctx_5m.get("trend")
    if trend != "bullish" and not is_ct:
        detail = ct_near or f"5M trend is {trend}, need bullish"
        lines.append(gate_line(False, "long", detail))
        return lines
    if is_ct:
        lines.append(gate_line(True, "long", f"counter-trend allowed — {ct_reason}"))
    else:
        lines.append(gate_line(trend == "bullish", "long", f"5M trend {trend}"))

    adx = float(ctx_5m.get("adx", 0))
    adx_min = cfg.adx_threshold
    if cfg.effective_adx_fn:
        adx_min = cfg.effective_adx_fn(ctx_5m, flow_snap, "long")
    adx_ok = adx >= adx_min
    relax_note = f" ({adx_min} w/ flow relax)" if adx_min < cfg.adx_threshold else ""
    lines.append(gate_line(
        adx_ok,
        "long",
        f"ADX strength {adx:.0f} >= {adx_min}{relax_note}" if adx_ok
        else f"ADX strength {adx:.0f} < {adx_min} (need {cfg.adx_threshold}+)",
    ))
    if not adx_ok:
        return lines

    if not is_ct and cfg.use_order_flow and flow_snap:
        blocked, flow_reason = _flow_blocks_long_relaxed(
            flow_snap, cfg.flow_entry_guard, adx,
            cfg.flow_relax_strength, cfg.flow_relax_long_buy_pct,
        )
        delta = float(flow_snap.get("delta", 0))
        buy_pct = float(flow_snap.get("buy_pct", 0.5))
        buy_min = float(cfg.flow_entry_guard.get("long_buy_pct_min", 0.48))
        if blocked:
            lines.append(gate_line(False, "long", flow_reason))
            return lines
        relaxed = adx >= cfg.flow_relax_strength and buy_pct >= cfg.flow_relax_long_buy_pct
        if relaxed and delta < 0:
            lines.append(gate_line(
                True, "long",
                f"flow delta={delta:+.0f} buy%={buy_pct:.0%} (relaxed: strength>={cfg.flow_relax_strength:.0f}, buy%>={cfg.flow_relax_long_buy_pct:.0%})",
            ))
        else:
            lines.append(gate_line(
                True, "long",
                f"flow delta={delta:+.0f} buy%={buy_pct:.0%} (need delta>=0 or buy%>={buy_min:.0%})",
            ))
    elif cfg.use_order_flow:
        lines.append(gate_line(True, "long", "order flow unavailable — guard skipped"))

    di_plus = float(ctx_5m.get("di_plus", 0))
    di_minus = float(ctx_5m.get("di_minus", 0))
    if not is_ct and adx >= cfg.di_relax_strength:
        lines.append(gate_line(
            True, "long",
            f"DI+ {di_plus:.0f} vs DI− {di_minus:.0f} (DI check skipped — strength>={cfg.di_relax_strength:.0f})",
        ))
    elif not is_ct:
        di_tol = cfg.di_tolerance
        flow_confirms = (
            cfg.use_order_flow
            and cfg.use_flow_di_override
            and flow_snap
            and trend == "bullish"
            and (cfg.bias_15m_allows_fn("long", ctx_15m) if cfg.bias_15m_allows_fn else True)
            and flow_confirms_long_direction(flow_snap, cfg.flow_entry_guard)
            and TickFlowTracker.confirms_direction("long", flow_snap)
        )
        if flow_confirms:
            di_tol = cfg.di_flow_tolerance
        di_ok = di_plus >= (di_minus - di_tol)
        gap = di_plus - di_minus
        lines.append(gate_line(
            di_ok, "long",
            f"DI+ {di_plus:.0f} vs DI− {di_minus:.0f} (gap {gap:+.0f}; need DI+ ≥ DI−−{di_tol:.0f})",
        ))
        if not di_ok:
            return lines
    else:
        lines.append(gate_line(True, "long", "DI skipped (counter-trend)"))

    price = float(row["close"])
    ema_9 = float(row["ema_9"])
    ema_21 = float(row["ema_21"])
    rsi = float(row["rsi"]) if not pd.isna(row["rsi"]) else float("nan")
    macd_hist = float(row["macd_hist"]) if not pd.isna(row.get("macd_hist")) else float("nan")
    macd_hist_prev = float(row["macd_hist_prev"]) if not pd.isna(row.get("macd_hist_prev")) else float("nan")
    volume_ratio = float(row["volume_ratio"]) if not pd.isna(row.get("volume_ratio")) else float("nan")
    bb_pctb = float(row["bb_pctb"]) if not pd.isna(row.get("bb_pctb")) else float("nan")
    atr = float(row["atr"]) if not pd.isna(row.get("atr")) else 0.0
    vwap = float(row.get("vwap", float("nan")))
    candle_open = float(row["open"])

    if cfg.adaptive_vwap_ok_fn:
        vwap_ok = cfg.adaptive_vwap_ok_fn(price, vwap, "long", ctx_5m)
        lines.append(gate_line(vwap_ok, "long", "price above VWAP (strong uptrend rule)" if vwap_ok else "price below VWAP"))
        if not vwap_ok:
            return lines

    if cfg.candle_confirmation:
        ok = price > candle_open
        lines.append(gate_line(ok, "long", "green 1M candle" if ok else "waiting for green 1M candle"))
        if not ok:
            return lines

    ema9_tol = atr * 0.1
    ema9_ok = price >= (ema_9 - ema9_tol)
    lines.append(gate_line(ema9_ok, "long", f"price near EMA9 ({price:.2f} vs {ema_9:.2f})"))
    if not ema9_ok:
        return lines

    pullback_mult = cfg.adaptive_pullback_atr_fn(ctx_5m, "long", is_ct) if cfg.adaptive_pullback_atr_fn else 1.5
    pullback = atr * pullback_mult
    if not is_ct:
        pb_ok = abs(price - ema_21) <= pullback
        lines.append(gate_line(
            pb_ok, "long",
            f"pullback to EMA21 (dist {abs(price - ema_21):.2f} ≤ {pullback:.2f} ATR×{pullback_mult:.1f})",
        ))
        if not pb_ok:
            return lines

    rsi_lo, rsi_hi = (35, 60)
    if cfg.adaptive_rsi_long_fn:
        rsi_lo, rsi_hi = cfg.adaptive_rsi_long_fn(ctx_5m, is_ct)
    rsi_ok = not pd.isna(rsi) and rsi_lo <= rsi <= rsi_hi
    lines.append(gate_line(
        rsi_ok, "long",
        f"RSI {rsi:.0f} in [{rsi_lo:.0f}-{rsi_hi:.0f}]" if rsi_ok else f"RSI {rsi:.0f} outside [{rsi_lo:.0f}-{rsi_hi:.0f}]",
    ))
    if not rsi_ok:
        return lines

    macd_ready = not pd.isna(macd_hist) and not pd.isna(macd_hist_prev)
    relax = False
    if cfg.strong_trend_relaxed_fn:
        relax = cfg.strong_trend_relaxed_fn("long", ctx_5m, ctx_15m) or is_ct
    macd_ok = macd_ready and (relax or macd_hist > macd_hist_prev)
    lines.append(gate_line(
        macd_ok, "long",
        "MACD turning up" if macd_ok else ("MACD warming up" if not macd_ready else "MACD flat/falling"),
    ))
    if not macd_ok:
        return lines

    vol_ok = not pd.isna(volume_ratio) and volume_ratio >= cfg.volume_ratio_threshold
    lines.append(gate_line(
        vol_ok, "long",
        f"volume ratio {volume_ratio:.2f} >= {cfg.volume_ratio_threshold}" if vol_ok
        else f"volume ratio {volume_ratio:.2f} < {cfg.volume_ratio_threshold}",
    ))
    if not vol_ok:
        return lines

    bb_ok = not pd.isna(bb_pctb) and cfg.bb_extreme_low < bb_pctb < cfg.bb_extreme_high
    lines.append(gate_line(
        bb_ok, "long",
        f"BB %B {bb_pctb:.2f} not at extreme" if bb_ok else f"BB %B {bb_pctb:.2f} at extreme",
    ))
    if not bb_ok:
        return lines

    if entry_quality_enabled(cfg.entry_quality):
        ts = row.get("datetime")
        ok, reason = check_long_entry_quality(
            row, ctx_5m, cfg.entry_quality,
            df_1m=df_1m, timestamp=ts, session_mode=cfg.session_mode,
            is_counter_trend=is_ct,
        )
        lines.append(gate_line(ok, "long", reason if not ok else "entry quality filters passed"))
        if not ok:
            return lines
    else:
        lines.append(gate_line(True, "long", "entry quality filters off"))

    lines.append(gate_line(True, "long", "ALL GATES PASSED — ready for long entry"))
    return lines


def evaluate_short_gates(
    row: pd.Series,
    ctx_5m: Dict,
    ctx_15m: Dict,
    flow_snap: Optional[Dict],
    flow_streak: int,
    cfg: GateEvalContext,
    df_1m: Optional[pd.DataFrame] = None,
) -> List[str]:
    lines: List[str] = []
    fc = dict(cfg.flow_counter_cfg)
    is_ct, ct_reason, ct_near = evaluate_counter_trend(
        "short", ctx_5m, flow_snap, fc, flow_streak=flow_streak,
    )

    trend = ctx_5m.get("trend")
    if trend != "bearish" and not is_ct:
        detail = ct_near or f"5M trend is {trend}"
        lines.append(gate_line(False, "short", detail))
        return lines
    if is_ct:
        lines.append(gate_line(True, "short", f"counter-trend allowed — {ct_reason}"))
    else:
        lines.append(gate_line(trend == "bearish", "short", f"5M trend {trend}"))

    if cfg.use_15m_entry_gate and cfg.use_15m_bias:
        ok = cfg.bias_15m_allows_fn("short", ctx_15m) if cfg.bias_15m_allows_fn else True
        trend_15 = ctx_15m.get("trend")
        lines.append(gate_line(
            ok or is_ct,
            "short",
            f"15M bias {trend_15} aligned for sells" if ok else f"15M bias {trend_15} blocks shorts",
        ))
        if not ok and not is_ct:
            return lines
    elif cfg.use_15m_bias:
        lines.append(gate_line(True, "short", f"15M {ctx_15m.get('trend')} (direction only — not gating)"))

    adx = float(ctx_5m.get("adx", 0))
    adx_min = cfg.adx_threshold
    if cfg.effective_adx_fn:
        adx_min = cfg.effective_adx_fn(ctx_5m, flow_snap, "short")
    adx_ok = adx >= adx_min
    relax_note = f" ({adx_min} w/ flow relax)" if adx_min < cfg.adx_threshold else ""
    lines.append(gate_line(
        adx_ok,
        "short",
        f"ADX strength {adx:.0f} >= {adx_min}{relax_note}" if adx_ok
        else f"ADX strength {adx:.0f} < {adx_min} (need {cfg.adx_threshold}+)",
    ))
    if not adx_ok:
        return lines

    if not is_ct and cfg.use_order_flow and flow_snap:
        blocked, flow_reason = flow_blocks_short(flow_snap, cfg.flow_entry_guard)
        delta = float(flow_snap.get("delta", 0))
        buy_pct = float(flow_snap.get("buy_pct", 0.5))
        buy_max = float(cfg.flow_entry_guard.get("short_buy_pct_max", 0.52))
        if blocked:
            lines.append(gate_line(False, "short", flow_reason))
            return lines
        lines.append(gate_line(
            True, "short",
            f"flow delta={delta:+.0f} buy%={buy_pct:.0%} (need delta<=0 or buy%<={buy_max:.0%})",
        ))
    elif cfg.use_order_flow:
        lines.append(gate_line(True, "short", "order flow unavailable — guard skipped"))

    di_plus = float(ctx_5m.get("di_plus", 0))
    di_minus = float(ctx_5m.get("di_minus", 0))
    if not is_ct and adx >= cfg.di_relax_strength:
        lines.append(gate_line(
            True, "short",
            f"DI− {di_minus:.0f} vs DI+ {di_plus:.0f} (DI check skipped — strength>={cfg.di_relax_strength:.0f})",
        ))
    elif not is_ct:
        di_tol = cfg.di_tolerance
        flow_confirms = (
            cfg.use_order_flow
            and cfg.use_flow_di_override
            and flow_snap
            and trend == "bearish"
            and (cfg.bias_15m_allows_fn("short", ctx_15m) if cfg.bias_15m_allows_fn else True)
            and TickFlowTracker.confirms_direction("short", flow_snap)
        )
        if flow_confirms:
            di_tol = cfg.di_flow_tolerance
        di_ok = di_minus >= (di_plus - di_tol)
        gap = di_minus - di_plus
        lines.append(gate_line(
            di_ok, "short",
            f"DI− {di_minus:.0f} vs DI+ {di_plus:.0f} (gap {gap:+.0f}; need DI− ≥ DI+−{di_tol:.0f})",
        ))
        if not di_ok:
            return lines
    else:
        lines.append(gate_line(True, "short", "DI skipped (counter-trend)"))

    price = float(row["close"])
    ema_9 = float(row["ema_9"])
    ema_21 = float(row["ema_21"])
    rsi = float(row["rsi"]) if not pd.isna(row["rsi"]) else float("nan")
    macd_hist = float(row["macd_hist"]) if not pd.isna(row.get("macd_hist")) else float("nan")
    macd_hist_prev = float(row["macd_hist_prev"]) if not pd.isna(row.get("macd_hist_prev")) else float("nan")
    volume_ratio = float(row["volume_ratio"]) if not pd.isna(row.get("volume_ratio")) else float("nan")
    bb_pctb = float(row["bb_pctb"]) if not pd.isna(row.get("bb_pctb")) else float("nan")
    atr = float(row["atr"]) if not pd.isna(row.get("atr")) else 0.0
    vwap = float(row.get("vwap", float("nan")))
    candle_open = float(row["open"])

    if cfg.adaptive_vwap_ok_fn:
        vwap_ok = cfg.adaptive_vwap_ok_fn(price, vwap, "short", ctx_5m)
        lines.append(gate_line(vwap_ok, "short", "price below VWAP (strong downtrend rule)" if vwap_ok else "price above VWAP"))
        if not vwap_ok:
            return lines

    if cfg.candle_confirmation:
        ok = price < candle_open
        lines.append(gate_line(ok, "short", "red 1M candle" if ok else "waiting for red 1M candle"))
        if not ok:
            return lines

    ema9_tol = atr * 0.1
    ema9_ok = price <= (ema_9 + ema9_tol)
    lines.append(gate_line(ema9_ok, "short", f"price near EMA9 ({price:.2f} vs {ema_9:.2f})"))
    if not ema9_ok:
        return lines

    pullback_mult = cfg.adaptive_pullback_atr_fn(ctx_5m, "short", is_ct) if cfg.adaptive_pullback_atr_fn else 1.5
    pullback = atr * pullback_mult
    if not is_ct:
        pb_ok = abs(price - ema_21) <= pullback
        lines.append(gate_line(
            pb_ok, "short",
            f"pullback to EMA21 (dist {abs(price - ema_21):.2f} ≤ {pullback:.2f})",
        ))
        if not pb_ok:
            return lines

    rsi_lo, rsi_hi = (40, 65)
    if cfg.adaptive_rsi_short_fn:
        rsi_lo, rsi_hi = cfg.adaptive_rsi_short_fn(ctx_5m, is_ct)
    rsi_ok = not pd.isna(rsi) and rsi_lo <= rsi <= rsi_hi
    lines.append(gate_line(
        rsi_ok, "short",
        f"RSI {rsi:.0f} in [{rsi_lo:.0f}-{rsi_hi:.0f}]" if rsi_ok else f"RSI {rsi:.0f} outside [{rsi_lo:.0f}-{rsi_hi:.0f}]",
    ))
    if not rsi_ok:
        return lines

    macd_ready = not pd.isna(macd_hist) and not pd.isna(macd_hist_prev)
    relax = False
    if cfg.strong_trend_relaxed_fn:
        relax = cfg.strong_trend_relaxed_fn("short", ctx_5m, ctx_15m) or is_ct
    macd_ok = macd_ready and (relax or macd_hist < macd_hist_prev)
    lines.append(gate_line(
        macd_ok, "short",
        "MACD turning down" if macd_ok else ("MACD warming up" if not macd_ready else "MACD flat/rising"),
    ))
    if not macd_ok:
        return lines

    vol_ok = not pd.isna(volume_ratio) and volume_ratio >= cfg.volume_ratio_threshold
    lines.append(gate_line(
        vol_ok, "short",
        f"volume ratio {volume_ratio:.2f} >= {cfg.volume_ratio_threshold}" if vol_ok
        else f"volume ratio {volume_ratio:.2f} < {cfg.volume_ratio_threshold}",
    ))
    if not vol_ok:
        return lines

    bb_ok = not pd.isna(bb_pctb) and cfg.bb_extreme_low < bb_pctb < cfg.bb_extreme_high
    lines.append(gate_line(
        bb_ok, "short",
        f"BB %B {bb_pctb:.2f} not at extreme" if bb_ok else f"BB %B {bb_pctb:.2f} at extreme",
    ))
    if not bb_ok:
        return lines

    if entry_quality_enabled(cfg.entry_quality):
        ts = row.get("datetime")
        ok, reason = check_short_entry_quality(
            row, ctx_5m, cfg.entry_quality,
            is_counter_trend=is_ct,
            df_1m=df_1m, timestamp=ts, session_mode=cfg.session_mode,
        )
        lines.append(gate_line(ok, "short", reason if not ok else "entry quality filters passed"))
        if not ok:
            return lines
    else:
        lines.append(gate_line(True, "short", "entry quality filters off"))

    lines.append(gate_line(True, "short", "ALL GATES PASSED — ready for short entry"))
    return lines


def evaluate_global_gates(
    symbol: str,
    row: pd.Series,
    *,
    daily_limit_ok: bool,
    daily_pnl: float,
    daily_limit: float,
    trade_limit_reached: bool,
    trade_count: int,
    trade_limit: int,
    loss_cooldown_min: int,
    open_count: int,
    max_positions: int,
    untracked_broker: int,
    broker_block: bool,
    atr_ok: bool,
    volatility_ok: bool,
    candle_range: float,
    max_bar_pts: float,
    new_1m_bar: bool,
    daily_half_stop: bool = False,
    bars_5m: Optional[int] = None,
    bars_5m_ideal: Optional[int] = None,
    bars_5m_floor: Optional[int] = None,
    skip_5m_warming: bool = False,
    use_1m_trend: bool = False,
    bars_1m: Optional[int] = None,
    new_trigger_bar: Optional[bool] = None,
) -> List[str]:
    lines: List[str] = []
    trigger_bar = new_trigger_bar if new_trigger_bar is not None else new_1m_bar
    if skip_5m_warming or use_1m_trend:
        n1 = bars_1m if bars_1m is not None else 0
        lines.append(gate_line(
            True, "GLOBAL",
            f"fast scalp — 1M trend ({n1} bars, 5M optional)",
        ))
    elif bars_5m is not None and bars_5m_ideal is not None:
        floor = bars_5m_floor if bars_5m_floor is not None else bars_5m_ideal
        if bars_5m < floor:
            lines.append(gate_line(
                False, "GLOBAL",
                f"5M indicators warming ({bars_5m}/{bars_5m_ideal} bars, floor {floor})",
            ))
        elif bars_5m < bars_5m_ideal:
            lines.append(gate_line(
                True, "GLOBAL",
                f"5M partial OK ({bars_5m}/{bars_5m_ideal} bars) — entries allowed",
            ))
        else:
            lines.append(gate_line(
                True, "GLOBAL",
                f"5M history ready ({bars_5m}/{bars_5m_ideal} bars)",
            ))

    if not daily_limit_ok:
        lines.append(gate_line(False, "GLOBAL", f"daily loss limit hit (${daily_pnl:.2f} / −${daily_limit:.0f})"))
    elif daily_half_stop:
        lines.append(gate_line(False, "GLOBAL", f"daily half-stop active (${daily_pnl:.2f})"))
    else:
        lines.append(gate_line(True, "GLOBAL", f"daily P&L ${daily_pnl:+.2f} within limit"))

    if trade_limit_reached:
        lines.append(gate_line(False, "GLOBAL", f"max trades/day {trade_count}/{trade_limit}"))
    else:
        lines.append(gate_line(True, "GLOBAL", f"trades today {trade_count}/{trade_limit}"))

    if loss_cooldown_min > 0:
        lines.append(gate_line(False, "GLOBAL", f"loss cooldown — {loss_cooldown_min} min left"))
    else:
        lines.append(gate_line(True, "GLOBAL", "no loss cooldown"))

    if open_count >= max_positions:
        lines.append(gate_line(False, "GLOBAL", f"max positions {open_count}/{max_positions}"))
    else:
        lines.append(gate_line(True, "GLOBAL", f"positions {open_count}/{max_positions}"))

    if broker_block and untracked_broker > 0:
        lines.append(gate_line(False, "GLOBAL", f"broker untracked exposure ({untracked_broker}) — entries blocked"))
    else:
        lines.append(gate_line(True, "GLOBAL", "broker position check OK"))

    if not atr_ok:
        lines.append(gate_line(False, "GLOBAL", "warming up — insufficient ATR/history"))
    else:
        lines.append(gate_line(True, "GLOBAL", "ATR/history ready"))

    if not volatility_ok:
        lines.append(gate_line(
            False, "GLOBAL",
            f"1M bar range {candle_range:.0f} pts > max {max_bar_pts:.0f} pts",
        ))
    else:
        lines.append(gate_line(
            True, "GLOBAL",
            f"1M bar range {candle_range:.0f} pts ≤ max {max_bar_pts:.0f} pts",
        ))

    if trigger_bar:
        bar_label = "30s" if new_trigger_bar is not None and new_trigger_bar != new_1m_bar else "1M"
        lines.append(gate_line(True, "GLOBAL", f"new {bar_label} bar — entry evaluation active"))
    else:
        wait_label = "30s" if new_trigger_bar is not None else "1M"
        lines.append(gate_line(False, "GLOBAL", f"same {wait_label} bar — entry deferred"))

    return lines
