#!/usr/bin/env python3
"""
Compare MTF strategy variants (A/C/B/D) on shared MNQ OHLC for MNQ + NQ profitability.

Variants:
  A — baseline (current tp13_adx17 rules)
  C — bear adaptive (strong bear: wider RSI/pullback, below VWAP)
  B — 15M bias filter on top of 5M
  D — 15M bias + full adaptive (bear + bull)
  BC — B + C combined (recommended path)

Usage:
    python scripts/backtest_adaptive_variants.py
    python scripts/backtest_adaptive_variants.py --apply   # write winner to mnq_profit_config.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_mtf_scalping as mtf
from src.ai.entry_quality import (
    check_long_entry_quality,
    check_short_entry_quality,
    parse_entry_quality,
)
from src.ai.mnq_context import compute_vwap
from src.utils.bias_15m import compute_15m_trend, resolve_15m_bias_buffer_pts, resolve_15m_bias_mode
from src.utils.flow_counter_trend import (
    evaluate_counter_trend,
    flow_blocks_long,
    flow_blocks_short,
    flow_confirms_long_direction,
    flow_contradicts_5m_trend,
    proxy_flow_from_bars,
    resolve_flow_counter_config,
    resolve_flow_entry_guard,
)
from src.utils.trading_session import coerce_session_mode, is_session_open_et


# Base params from dual-symbol winner
BASE_CFG = {
    "atr": 1.2,
    "tp": 1.3,
    "adx": 17,
    "vol": 0.4,
    "di_tol": 3.0,
    "tp_buffer": 0.5,
    "max_tr": 20,
    "min_rr": 1.0,
    "di_counter": 20.0,
    "counter_adx": 25,
}

STRONG_TREND_ADX = 30  # default; overridden by cfg["strong_trend_adx"]


def _strong_adx(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("strong_trend_adx", STRONG_TREND_ADX))


def _vwap_adx(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("vwap_adx_min", _strong_adx(cfg)))


def _strong_relax_adx(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("strong_trend_relax_adx", cfg.get("strong_trend_adx", 40)))


def strong_trend_relaxed(
    ctx_5m: Dict,
    ctx_15m: Dict,
    direction: str,
    cfg: Dict[str, Any],
) -> bool:
    """5M ADX >= threshold and 5M+15M hard-aligned — unlock MACD skip / lower min_rr."""
    if not cfg.get("strong_trend_skip_macd"):
        return False
    if ctx_5m.get("adx", 0) < _strong_relax_adx(cfg):
        return False
    want = "bullish" if direction == "long" else "bearish"
    if ctx_5m.get("trend") != want:
        return False
    return ctx_15m.get("trend") == want


def entry_min_rr(
    direction: str,
    ctx_5m: Dict,
    ctx_15m: Dict,
    variant: StrategyVariant,
    cfg: Dict[str, Any],
) -> float:
    if strong_trend_relaxed(ctx_5m, ctx_15m, direction, cfg):
        return float(cfg.get("strong_trend_min_rr", variant.min_rr))
    return variant.min_rr


# Live bot limits (start_live_mtf_scalping.py)
LIVE_SYMBOL_LIMITS = {
    "NQ": {"max_loss_per_trade": 250.0, "daily_loss_limit": 300.0, "point_value": 20.0},
    "MNQ": {"max_loss_per_trade": 250.0, "daily_loss_limit": 300.0, "point_value": 2.0},
}
VOLATILITY_FILTER_POINTS = 45


def cap_sl_distance(symbol: str, sl_distance: float) -> float:
    lim = LIVE_SYMBOL_LIMITS.get(symbol, {})
    max_loss = lim.get("max_loss_per_trade")
    pv = lim.get("point_value", mtf.SYMBOL_SPECS.get(symbol, {}).get("point_value", 2.0))
    if not max_loss:
        return sl_distance
    return min(sl_distance, max_loss / pv)


def is_live_et_session(dt, mode: str = "rth") -> bool:
    """Match live bot session gate (rth or extended from cfg/env)."""
    try:
        return is_session_open_et(dt, coerce_session_mode(mode))
    except Exception:
        return True


@dataclass
class StrategyVariant:
    name: str
    label: str
    use_15m_bias: bool = False
    bear_adaptive: bool = False
    bull_adaptive: bool = False
    counter_trend_shorts: bool = True
    min_rr: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)


VARIANTS: List[StrategyVariant] = [
    StrategyVariant("A", "baseline"),
    StrategyVariant("C", "bear_adaptive", bear_adaptive=True),
    StrategyVariant("B", "15m_bias", use_15m_bias=True),
    StrategyVariant("BC", "15m_bear_adaptive", use_15m_bias=True, bear_adaptive=True),
    StrategyVariant("D", "full_adaptive", use_15m_bias=True, bear_adaptive=True, bull_adaptive=True),
]


def load_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    df_1m = pd.read_csv(os.path.join(data_dir, "MNQ_1m.csv"), parse_dates=["datetime"])
    df_5m = pd.read_csv(os.path.join(data_dir, "MNQ_5m.csv"), parse_dates=["datetime"])
    df_1m = df_1m.sort_values("datetime").reset_index(drop=True)
    df_5m = df_5m.sort_values("datetime").reset_index(drop=True)
    return df_1m, df_5m


def resample_15m(df_1m: pd.DataFrame) -> pd.DataFrame:
    df = df_1m.set_index("datetime")
    out = (
        df.resample("15min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    return out


def get_15m_context(
    df_15m: pd.DataFrame,
    timestamp: pd.Timestamp,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    mask = df_15m["datetime"] <= timestamp
    if not mask.any():
        return {"trend": None, "adx": 0}
    row = df_15m[mask].iloc[-1]
    close = float(row["close"])
    ema50 = float(row["ema_50"])
    ema200 = float(row["ema_200"])
    mode = resolve_15m_bias_mode(cfg)
    buffer_pts = resolve_15m_bias_buffer_pts(cfg)
    trend = compute_15m_trend(close, ema50, ema200, mode, buffer_pts)
    return {
        "trend": trend,
        "adx": row["adx"],
        "close": close,
        "ema_50": ema50,
        "ema_200": ema200,
        "bias_mode": mode,
    }


def apply_base_cfg(cfg: Dict[str, Any]) -> None:
    mtf.TP_MULT = cfg["tp"]
    mtf.ADX_THRESHOLD = cfg["adx"]
    mtf.VOLUME_RATIO_THRESHOLD = cfg["vol"]
    mtf.DI_TOLERANCE = cfg["di_tol"]
    mtf.TP_BUFFER_ATR_MULT = cfg["tp_buffer"]
    mtf.MAX_TRADES_PER_DAY = cfg.get("max_tr", 20)


def rsi_bounds(
    direction: str,
    variant: StrategyVariant,
    ctx_5m: Dict,
    is_counter_trend: bool,
    cfg: Dict[str, Any],
) -> Tuple[float, float]:
    strong = _strong_adx(cfg)
    if direction == "long":
        if is_counter_trend:
            return 30, 70
        lo, hi = mtf.RSI_LONG_MIN, mtf.RSI_LONG_MAX
        if variant.bull_adaptive and ctx_5m["trend"] == "bullish" and ctx_5m["adx"] >= strong:
            lo, hi = int(cfg.get("bull_rsi_lo", 35)), int(cfg.get("bull_rsi_hi", 70))
        return lo, hi
    lo, hi = mtf.RSI_SHORT_MIN, mtf.RSI_SHORT_MAX
    if is_counter_trend:
        return 30, 70
    if variant.bear_adaptive and ctx_5m["trend"] == "bearish" and ctx_5m["adx"] >= strong:
        return int(cfg.get("bear_rsi_lo", 25)), int(cfg.get("bear_rsi_hi", 65))
    return lo, hi


def pullback_atr(
    variant: StrategyVariant, ctx_5m: Dict, direction: str, is_counter_trend: bool, cfg: Dict[str, Any]
) -> float:
    base = 1.5
    strong = _strong_adx(cfg)
    if is_counter_trend:
        base = 2.5
    elif direction == "short" and variant.bear_adaptive:
        if ctx_5m["trend"] == "bearish" and ctx_5m["adx"] >= strong:
            base = float(cfg.get("bear_pullback_atr", 2.0))
    elif direction == "long" and variant.bull_adaptive:
        if ctx_5m["trend"] == "bullish" and ctx_5m["adx"] >= strong:
            base = float(cfg.get("bull_pullback_atr", 2.0))
    strong_pb_adx = int(cfg.get("strong_pullback_adx", 40))
    pb_mult = float(cfg.get("pullback_atr_mult", os.getenv("PULLBACK_ATR_MULT", "1.5")))
    if not is_counter_trend and ctx_5m.get("adx", 0) >= strong_pb_adx and pb_mult > 1.0:
        base *= pb_mult
    return base


def vwap_ok(
    variant: StrategyVariant, price: float, vwap: float, direction: str, ctx_5m: Dict, cfg: Dict[str, Any]
) -> bool:
    if not cfg.get("vwap_required", True):
        return True
    if pd.isna(vwap):
        return True
    vwap_threshold = _vwap_adx(cfg)
    if variant.bear_adaptive and direction == "short":
        if ctx_5m["trend"] == "bearish" and ctx_5m["adx"] >= vwap_threshold:
            return price < vwap
    if variant.bull_adaptive and direction == "long":
        if ctx_5m["trend"] == "bullish" and ctx_5m["adx"] >= vwap_threshold:
            return price > vwap
    return True


def bias_15m_ok(variant: StrategyVariant, ctx_15m: Dict, direction: str, cfg: Dict[str, Any]) -> bool:
    if not cfg.get("15m_entry_gate", False):
        return True
    if not variant.use_15m_bias:
        return True
    trend = ctx_15m.get("trend")
    want = "bullish" if direction == "long" else "bearish"
    if cfg.get("soft_15m_bias"):
        if trend in (None, "neutral"):
            return True
        return trend == want
    return trend == want


def _flow_counter_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    fc = resolve_flow_counter_config(cfg)
    fc["di_counter"] = float(cfg.get("di_counter", 20.0))
    fc["counter_adx"] = int(cfg.get("counter_adx", 25))
    fc["counter_trend_shorts"] = bool(cfg.get("counter_trend_shorts", True))
    fc["counter_trend_longs"] = bool(cfg.get("counter_trend_longs", True))
    return fc


def _backtest_flow_snap(cfg: Dict[str, Any], df_1m: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
    if not cfg.get("use_flow_proxy", True):
        return None
    return proxy_flow_from_bars(df_1m, window=int(cfg.get("flow_proxy_bars", 5)))


def check_long_entry(
    row: pd.Series,
    ctx_5m: Dict,
    ctx_15m: Dict,
    variant: StrategyVariant,
    cfg: Dict[str, Any],
    df_1m: Optional[pd.DataFrame] = None,
    flow_streak: int = 0,
) -> bool:
    fc = _flow_counter_cfg(cfg)
    flow_snap = _backtest_flow_snap(cfg, df_1m)
    is_counter, _, _ = evaluate_counter_trend(
        "long", ctx_5m, flow_snap, fc, flow_streak=flow_streak,
    )
    if not is_counter and not bias_15m_ok(variant, ctx_15m, "long", cfg):
        return False
    if ctx_5m["trend"] != "bullish" and not is_counter:
        return False
    if ctx_5m["adx"] < mtf.ADX_THRESHOLD:
        return False

    guard = resolve_flow_entry_guard(cfg)
    if not is_counter and cfg.get("use_flow_proxy", True):
        blocked, _ = flow_blocks_long(flow_snap, guard)
        if blocked:
            return False

    di_tol = mtf.DI_TOLERANCE
    if (
        not is_counter
        and cfg.get("use_flow_proxy", True)
        and flow_snap
        and ctx_5m["trend"] == "bullish"
        and bias_15m_ok(variant, ctx_15m, "long", cfg)
        and flow_confirms_long_direction(flow_snap, guard)
        and float(flow_snap.get("delta", 0)) > 0
    ):
        di_tol = float(cfg.get("di_flow_tol", mtf.DI_TOLERANCE + 5))

    if not is_counter and ctx_5m["di_plus"] < (ctx_5m["di_minus"] - di_tol):
        return False

    price = row["close"]
    ema_9, ema_21 = row["ema_9"], row["ema_21"]
    rsi = row["rsi"]
    macd_hist, macd_hist_prev = row["macd_hist"], row["macd_hist_prev"]
    volume_ratio = row["volume_ratio"]
    bb_pctb = row["bb_pctb"]
    atr = row["atr"]
    vwap = row.get("vwap", float("nan"))

    if pd.isna(atr) or atr <= 0:
        return False
    if not vwap_ok(variant, price, vwap, "long", ctx_5m, cfg):
        return False

    zone = atr * pullback_atr(variant, ctx_5m, "long", is_counter, cfg)
    rsi_lo, rsi_hi = rsi_bounds("long", variant, ctx_5m, is_counter, cfg)
    ema9_tol = atr * float(cfg.get("ema9_tol_atr", 0.1))

    relax = strong_trend_relaxed(ctx_5m, ctx_15m, "long", cfg) or is_counter
    macd_ok = relax or (
        not pd.isna(macd_hist)
        and not pd.isna(macd_hist_prev)
        and macd_hist > macd_hist_prev
    )
    pullback_ok = is_counter or abs(price - ema_21) <= zone
    base_ok = (
        pullback_ok
        and rsi_lo <= rsi <= rsi_hi
        and macd_ok
        and volume_ratio >= mtf.VOLUME_RATIO_THRESHOLD
        and mtf.BB_EXTREME_LOW < bb_pctb < mtf.BB_EXTREME_HIGH
        and price > (ema_9 - ema9_tol)
    )
    if not base_ok:
        return False
    eq = parse_entry_quality(cfg)
    session_mode = coerce_session_mode(cfg.get("session_mode", "rth"))
    ok, _ = check_long_entry_quality(
        row, ctx_5m, eq, df_1m=df_1m, timestamp=row.get("datetime"),
        session_mode=session_mode, is_counter_trend=is_counter,
    )
    return ok


def check_short_entry(
    row: pd.Series,
    ctx_5m: Dict,
    ctx_15m: Dict,
    variant: StrategyVariant,
    cfg: Dict,
    df_1m: Optional[pd.DataFrame] = None,
    flow_streak: int = 0,
) -> bool:
    fc = _flow_counter_cfg(cfg)
    if not variant.counter_trend_shorts:
        fc["counter_trend_shorts"] = False
    flow_snap = _backtest_flow_snap(cfg, df_1m)
    is_counter, _, _ = evaluate_counter_trend(
        "short", ctx_5m, flow_snap, fc, flow_streak=flow_streak,
    )

    if ctx_5m["trend"] != "bearish" and not is_counter:
        return False
    if not bias_15m_ok(variant, ctx_15m, "short", cfg) and not is_counter:
        return False
    if ctx_5m["adx"] < mtf.ADX_THRESHOLD:
        return False

    guard = resolve_flow_entry_guard(cfg)
    if not is_counter and cfg.get("use_flow_proxy", True):
        blocked, _ = flow_blocks_short(flow_snap, guard)
        if blocked:
            return False

    if not is_counter and ctx_5m["di_minus"] < (ctx_5m["di_plus"] - mtf.DI_TOLERANCE):
        return False

    price = row["close"]
    ema_9, ema_21 = row["ema_9"], row["ema_21"]
    rsi = row["rsi"]
    macd_hist, macd_hist_prev = row["macd_hist"], row["macd_hist_prev"]
    volume_ratio = row["volume_ratio"]
    bb_pctb = row["bb_pctb"]
    atr = row["atr"]
    vwap = row.get("vwap", float("nan"))

    if pd.isna(atr) or atr <= 0:
        return False
    if not vwap_ok(variant, price, vwap, "short", ctx_5m, cfg):
        return False

    zone = atr * pullback_atr(variant, ctx_5m, "short", is_counter, cfg)
    rsi_lo, rsi_hi = rsi_bounds("short", variant, ctx_5m, is_counter, cfg)
    ema9_tol = atr * float(cfg.get("ema9_tol_atr", 0.1))

    if not is_counter and abs(price - ema_21) > zone:
        return False
    if pd.isna(rsi) or not (rsi_lo <= rsi <= rsi_hi):
        return False
    relax = strong_trend_relaxed(ctx_5m, ctx_15m, "short", cfg) or is_counter
    if not relax and macd_hist >= macd_hist_prev:
        return False
    if volume_ratio < mtf.VOLUME_RATIO_THRESHOLD:
        return False
    if bb_pctb <= mtf.BB_EXTREME_LOW or bb_pctb >= mtf.BB_EXTREME_HIGH:
        return False
    if price > (ema_9 + ema9_tol):
        return False
    eq = parse_entry_quality(cfg)
    session_mode = coerce_session_mode(cfg.get("session_mode", "rth"))
    ok, _ = check_short_entry_quality(
        row, ctx_5m, eq,
        is_counter_trend=is_counter,
        df_1m=df_1m,
        timestamp=row.get("datetime"),
        session_mode=session_mode,
    )
    return ok


def tp_after_structure(
    direction: str,
    entry: float,
    sl_dist: float,
    tp_dist: float,
    atr: float,
    ctx_5m: Dict,
    min_rr: float,
) -> Optional[float]:
    if direction == "long":
        tp_rr = entry + tp_dist
        tp_buf = atr * mtf.TP_BUFFER_ATR_MULT
        tp_struct = ctx_5m["resistance"] - tp_buf
        tp_final = min(tp_rr, tp_struct) if tp_struct > entry else tp_rr
        rr = (tp_final - entry) / sl_dist if sl_dist > 0 else 0
    else:
        tp_rr = entry - tp_dist
        tp_buf = atr * mtf.TP_BUFFER_ATR_MULT
        tp_struct = ctx_5m["support"] + tp_buf
        tp_final = max(tp_rr, tp_struct) if tp_struct < entry else tp_rr
        rr = (entry - tp_final) / sl_dist if sl_dist > 0 else 0
    if rr < min_rr:
        return None
    return tp_final


class AdaptiveBacktester(mtf.MultiTimeframeBacktester):
    """MTF backtester with variant-specific entry rules."""

    def __init__(
        self,
        symbol: str,
        variant: StrategyVariant,
        cfg: Dict[str, Any],
        entry_filter: Optional[Callable[..., bool]] = None,
    ):
        super().__init__(symbol)
        self.variant = variant
        self.cfg = cfg
        self.atr_mult = cfg["atr"]
        self.entry_filter = entry_filter
        self.skipped_filters: Dict[str, int] = {"news": 0, "llm": 0}
        if cfg.get("live_like") and symbol in LIVE_SYMBOL_LIMITS:
            self.daily_loss_limit = LIVE_SYMBOL_LIMITS[symbol]["daily_loss_limit"]
            self.point_value = LIVE_SYMBOL_LIMITS[symbol]["point_value"]

    def run_variant(
        self,
        df_1m: pd.DataFrame,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
    ) -> Dict[str, Any]:
        df_1m = df_1m.copy()
        df_5m = df_5m.copy()
        df_15m = df_15m.copy()

        df_1m["vwap"] = compute_vwap(df_1m)
        df_1m = mtf.add_indicators_1m(df_1m)
        df_5m = mtf.add_indicators_5m(df_5m)
        df_15m = mtf.add_indicators_5m(df_15m)

        warmup = max(200, mtf.TREND_EMA_SLOW)
        position: Optional[mtf.Trade] = None
        flow_streak = 0
        fc = _flow_counter_cfg(self.cfg)

        for i in range(warmup, len(df_1m)):
            row = df_1m.iloc[i]
            dt = row["datetime"]

            trade_date = dt.date() if hasattr(dt, "date") else None
            if trade_date != self.current_date:
                self.current_date = trade_date
                self.daily_pnl = 0.0
                self.daily_trades = 0
                self.stopped_for_day = False

            if self.stopped_for_day:
                if position:
                    position = self._check_exit(position, row)
                continue

            ctx_5m = mtf.get_5m_context(df_5m, dt)
            ctx_15m = get_15m_context(df_15m, dt, self.cfg)

            if position:
                position = self._check_exit(position, row)
            else:
                if self.daily_trades >= mtf.MAX_TRADES_PER_DAY:
                    continue
                session_mode = coerce_session_mode(self.cfg.get("session_mode", "rth"))
                session_ok = (
                    is_live_et_session(dt, session_mode)
                    if self.cfg.get("live_like")
                    else mtf.is_trading_session(dt)
                )
                if not session_ok:
                    continue

                atr = row["atr"]
                if pd.isna(atr) or atr <= 0:
                    continue

                candle_range = row["high"] - row["low"]
                vol_cap = self.cfg.get("volatility_filter_points", VOLATILITY_FILTER_POINTS)
                if self.cfg.get("live_like") and candle_range > vol_cap:
                    continue

                sl_distance = atr * self.atr_mult
                if self.cfg.get("live_like"):
                    sl_distance = cap_sl_distance(self.symbol, sl_distance)
                tp_distance = sl_distance * mtf.TP_MULT
                entry_price = row["close"]

                df_slice = df_1m.iloc[: i + 1]
                flow_snap = _backtest_flow_snap(self.cfg, df_slice)
                if flow_contradicts_5m_trend(ctx_5m.get("trend", ""), flow_snap, fc):
                    flow_streak += 1
                else:
                    flow_streak = 0
                if check_long_entry(
                    row, ctx_5m, ctx_15m, self.variant, self.cfg, df_slice, flow_streak=flow_streak,
                ):
                    min_rr = entry_min_rr("long", ctx_5m, ctx_15m, self.variant, self.cfg)
                    tp_final = tp_after_structure(
                        "long", entry_price, sl_distance, tp_distance, atr, ctx_5m, min_rr
                    )
                    if tp_final is not None:
                        sl = entry_price - sl_distance
                        if self._entry_allowed(
                            "long", dt, row, ctx_5m, df_1m, df_5m, df_15m,
                            entry_price, sl, tp_final,
                        ):
                            position = mtf.Trade(
                                entry_time=dt,
                                direction="LONG",
                                entry_price=entry_price,
                                sl=sl,
                                tp=tp_final,
                                initial_sl=sl,
                                highest_price=entry_price,
                            )
                            self.daily_trades += 1

                elif check_short_entry(
                    row, ctx_5m, ctx_15m, self.variant, self.cfg, df_slice, flow_streak=flow_streak,
                ):
                    min_rr = entry_min_rr("short", ctx_5m, ctx_15m, self.variant, self.cfg)
                    tp_final = tp_after_structure(
                        "short", entry_price, sl_distance, tp_distance, atr, ctx_5m, min_rr
                    )
                    if tp_final is not None:
                        sl = entry_price + sl_distance
                        if self._entry_allowed(
                            "short", dt, row, ctx_5m, df_1m, df_5m, df_15m,
                            entry_price, sl, tp_final,
                        ):
                            position = mtf.Trade(
                                entry_time=dt,
                                direction="SHORT",
                                entry_price=entry_price,
                                sl=sl,
                                tp=tp_final,
                                initial_sl=sl,
                                lowest_price=entry_price,
                            )
                            self.daily_trades += 1

        if position:
            position.exit_time = df_1m.iloc[-1]["datetime"]
            position.exit_price = df_1m.iloc[-1]["close"]
            position.exit_reason = "END"
            position.pnl = self._calc_pnl(position)
            self.trades.append(position)
            self.balance += position.pnl

        return self._compute_stats()

    def _entry_allowed(
        self,
        direction: str,
        dt,
        row: pd.Series,
        ctx_5m: Dict,
        df_1m: pd.DataFrame,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        entry: float,
        sl: float,
        tp: float,
    ) -> bool:
        if not self.entry_filter:
            return True
        return self.entry_filter(
            direction=direction,
            dt=dt,
            row=row,
            ctx_5m=ctx_5m,
            df_1m=df_1m,
            df_5m=df_5m,
            df_15m=df_15m,
            entry=entry,
            sl=sl,
            tp=tp,
            backtester=self,
        )


def run_pair(
    variant: StrategyVariant,
    cfg: Dict[str, Any],
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
) -> Dict[str, Any]:
    apply_base_cfg(cfg)
    variant.min_rr = cfg.get("min_rr", 1.0)

    mnq_bt = AdaptiveBacktester("MNQ", variant, cfg)
    nq_bt = AdaptiveBacktester("NQ", variant, cfg)

    with contextlib.redirect_stdout(io.StringIO()):
        mnq_stats = mnq_bt.run_variant(df_1m, df_5m, df_15m)
        nq_stats = nq_bt.run_variant(df_1m, df_5m, df_15m)

    def pack(stats: Dict) -> Dict[str, Any]:
        if stats.get("error"):
            return {"trades": 0, "wr": 0, "pf": 0, "pnl": 0, "dd": 0}
        return {
            "trades": stats["total_trades"],
            "wr": stats["win_rate"],
            "pf": stats["profit_factor"],
            "pnl": stats["total_pnl"],
            "dd": stats["max_drawdown_pct"],
        }

    mnq = pack(mnq_stats)
    nq = pack(nq_stats)
    return {"mnq": mnq, "nq": nq, "mnq_stats": mnq_stats, "nq_stats": nq_stats}


def score_pair(mnq: Dict, nq: Dict) -> float:
    if mnq["pf"] < 1.0 or nq["pf"] < 1.0:
        return -999.0
    if mnq["pnl"] <= 0 or nq["pnl"] <= 0:
        return -999.0
    return min(mnq["pf"], nq["pf"]) * 100 + (mnq["wr"] + nq["wr"]) / 2 + min(mnq["pnl"], nq["pnl"]) / 1000


def load_live_profit_cfg() -> Dict[str, Any]:
    """Merge keys from data/mnq_profit_config.json for live/backtest parity."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mnq_profit_config.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def merge_run_cfg(base: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay live profit config onto base backtest params."""
    run_cfg = dict(base)
    live = load_live_profit_cfg()
    merge_keys = (
        "atr", "tp", "adx", "vol", "di_tol", "tp_buffer", "max_tr", "min_rr",
        "di_counter", "counter_adx", "15m_entry_gate", "use_15m_bias", "soft_15m_bias",
        "15m_bias_mode", "strong_trend_adx", "bear_adaptive", "bull_adaptive",
        "pullback_atr_mult", "strong_pullback_adx", "vwap_required", "entry_quality",
        "counter_trend_shorts", "counter_trend_longs", "flow_counter_trend",
        "strong_trend_skip_macd", "strong_trend_min_rr",
        "strong_trend_relax_adx", "vwap_adx_min",
    )
    for key in merge_keys:
        if key in live:
            run_cfg[key] = live[key]
    return run_cfg


def run_flow_counter_comparison(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    base_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare D variant before (std counter only) vs after (flow-confirmed counter-trend)."""
    variant = next(v for v in VARIANTS if v.name == "D")
    before_cfg = dict(base_cfg)
    before_cfg["flow_counter_trend"] = {"enabled": False}
    before_cfg["counter_trend_longs"] = False
    after_cfg = dict(base_cfg)
    if "flow_counter_trend" not in after_cfg:
        after_cfg["flow_counter_trend"] = {"enabled": True}

    before = run_pair(variant, before_cfg, df_1m, df_5m, df_15m)
    after = run_pair(variant, after_cfg, df_1m, df_5m, df_15m)

    print(f"\n{'='*78}")
    print("  FLOW COUNTER-TREND COMPARISON (variant D — MNQ rithmic data)")
    print(f"{'='*78}")
    for label, row in (("BEFORE (std DI>=20 ADX>=25, no flow longs)", before), ("AFTER  (flow counter-trend ON)", after)):
        mnq = row["mnq"]
        print(
            f"  {label}: MNQ {mnq['trades']:>3}tr WR={mnq['wr']:>4.0f}% "
            f"PF={mnq['pf']:.2f} PnL=${mnq['pnl']:>7,.0f} | "
            f"NQ PF={row['nq']['pf']:.2f} PnL=${row['nq']['pnl']:>8,.0f}"
        )
    mnq_delta_pf = after["mnq"]["pf"] - before["mnq"]["pf"]
    mnq_delta_tr = after["mnq"]["trades"] - before["mnq"]["trades"]
    verdict = "OK — PF held" if after["mnq"]["pf"] >= before["mnq"]["pf"] * 0.95 else "WARN — PF dropped"
    print(f"  Delta MNQ: {mnq_delta_tr:+d} trades, PF {mnq_delta_pf:+.2f} — {verdict}")
    print(f"{'='*78}\n")
    return {"before": before, "after": after, "verdict": verdict}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write winning variant to mnq_profit_config.json")
    parser.add_argument(
        "--compare-flow-counter",
        action="store_true",
        help="Run before/after flow counter-trend comparison on variant D and exit",
    )
    args = parser.parse_args()

    df_1m, df_5m = load_frames()
    df_15m = resample_15m(df_1m)
    run_cfg = merge_run_cfg(BASE_CFG)

    if args.compare_flow_counter:
        run_flow_counter_comparison(df_1m, df_5m, df_15m, run_cfg)
        return
    live_name = load_live_profit_cfg().get("name")
    if live_name:
        print(f"Live config merged: {live_name} (15m_entry_gate={run_cfg.get('15m_entry_gate', False)})")

    print(f"\n{'='*78}")
    print(f"  ADAPTIVE VARIANT BACKTEST — MNQ + NQ ({len(df_1m):,} × 1m bars)")
    print(f"  Range: {df_1m['datetime'].iloc[0]} -> {df_1m['datetime'].iloc[-1]}")
    print(f"  Base: ATRx{run_cfg['atr']} TPx{run_cfg['tp']} ADX>={run_cfg['adx']} max_tr={run_cfg['max_tr']}")
    print(f"{'='*78}\n")

    results: List[Dict[str, Any]] = []
    for variant in VARIANTS:
        row = run_pair(variant, run_cfg, df_1m, df_5m, df_15m)
        mnq, nq = row["mnq"], row["nq"]
        score = score_pair(mnq, nq)
        entry = {
            "variant": variant.name,
            "label": variant.label,
            "use_15m": variant.use_15m_bias,
            "bear_adaptive": variant.bear_adaptive,
            "bull_adaptive": variant.bull_adaptive,
            "mnq": mnq,
            "nq": nq,
            "score": score,
        }
        results.append(entry)
        ok = "OK" if score > 0 else "--"
        print(
            f"{ok} {variant.name:<3} {variant.label:<20} | "
            f"MNQ {mnq['trades']:>3}tr WR={mnq['wr']:>4.0f}% PF={mnq['pf']:.2f} ${mnq['pnl']:>7,.0f} | "
            f"NQ PF={nq['pf']:.2f} ${nq['pnl']:>8,.0f}",
            flush=True,
        )

    viable = [r for r in results if r["score"] > 0]
    viable.sort(key=lambda x: x["score"], reverse=True)
    if viable:
        best = viable[0]
    else:
        best = max(results, key=lambda x: x["mnq"]["pf"] + x["nq"]["pf"])
        print("\nWARN: No variant profitable on BOTH - picking best combined PF")

    print(f"\nWINNER: {best['variant']} ({best['label']})")
    print(
        f"   MNQ: {best['mnq']['trades']} trades | WR {best['mnq']['wr']:.1f}% | "
        f"PF {best['mnq']['pf']:.2f} | PnL ${best['mnq']['pnl']:,.0f}"
    )
    print(
        f"   NQ:  {best['nq']['trades']} trades | WR {best['nq']['wr']:.1f}% | "
        f"PF {best['nq']['pf']:.2f} | PnL ${best['nq']['pnl']:,.0f}"
    )

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    out_path = os.path.join(data_dir, "adaptive_backtest_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_cfg": run_cfg,
                "results": results,
                "winner": best,
                "data_bars_1m": len(df_1m),
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {out_path}")

    if args.apply:
        winner_variant = next(v for v in VARIANTS if v.name == best["variant"])
        cfg_out = {
            "name": f"{best['variant']}_{best['label']}",
            **run_cfg,
            "strategy_mode": best["label"],
            "use_15m_bias": winner_variant.use_15m_bias,
            "15m_entry_gate": run_cfg.get("15m_entry_gate", False),
            "bear_adaptive": winner_variant.bear_adaptive,
            "bull_adaptive": winner_variant.bull_adaptive,
            "counter_trend_shorts": winner_variant.counter_trend_shorts,
            "strong_trend_adx": STRONG_TREND_ADX,
            "trades_mnq": best["mnq"]["trades"],
            "wr_mnq": best["mnq"]["wr"],
            "pf_mnq": best["mnq"]["pf"],
            "pnl_mnq": best["mnq"]["pnl"],
            "trades_nq": best["nq"]["trades"],
            "wr_nq": best["nq"]["wr"],
            "pf_nq": best["nq"]["pf"],
            "pnl_nq": best["nq"]["pnl"],
            "note": f"Adaptive backtest winner — variant {best['variant']}",
        }
        profit_path = os.path.join(data_dir, "mnq_profit_config.json")
        with open(profit_path, "w", encoding="utf-8") as f:
            json.dump(cfg_out, f, indent=2)
        print(f"Applied: {profit_path}")


if __name__ == "__main__":
    main()
