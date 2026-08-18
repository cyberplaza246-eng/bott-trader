#!/usr/bin/env python3
"""
Quick scalp momentum backtest — 5M trend + 1M pullback + 30s trigger.

Versions:
  A — 5M + 1M only (1M bar as trigger proxy)
  B — 5M + 1M + 30s trigger (synthetic 30s from 1m unless --csv-30s)
  C — B + no-chase filter (skip extended candles / far from EMA)
  D — C + adaptive trend (ADX-scaled pullback zone)

Usage:
    python scripts/backtest_scalp_momentum.py
    python scripts/backtest_scalp_momentum.py --sl 8 --tp 15 --max-hold 90
    python scripts/backtest_scalp_momentum.py --symbol MNQ --csv-1m data/MNQ_1m_rithmic.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_mtf_scalping as mtf
from src.ai.mnq_context import compute_vwap


ADX_MIN = 20
EMA_TREND = 20
PULLBACK_ATR = 0.5
PULLBACK_ATR_STRONG = 0.75
PULLBACK_ATR_WEAK = 0.35
STRONG_ADX = 30
WEAK_ADX = 25
NO_CHASE_SIZE_MULT = 1.5
NO_CHASE_EMA_ATR = 1.0
CANDLE_AVG_PERIOD = 20
MAX_SETUP_BARS = 3
MAX_TRADES_PER_DAY = 12
POINT_VALUE = {"MNQ": 2.0, "NQ": 20.0, "MES": 5.0}
COMMISSION = 0.62
WARMUP = 250


@dataclass
class ScalpTrade:
    entry_time: pd.Timestamp
    direction: str
    entry_price: float
    sl: float
    tp: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    exit_reason: str = ""
    version: str = ""
    hold_seconds: float = 0.0


@dataclass
class VariantConfig:
    name: str
    use_30s_trigger: bool = False
    no_chase: bool = False
    adaptive_trend: bool = False


def normalize_dt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True).dt.as_unit("ns")
    return out.sort_values("datetime").reset_index(drop=True)


def resample_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    d = df_1m.set_index("datetime")
    return (
        d.resample("5min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )


def resample_30s_synthetic(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Split each 1m bar into two synthetic 30s bars (imperfect intra-minute path)."""
    n = len(df_1m)
    dt = pd.to_datetime(df_1m["datetime"], utc=True).dt.as_unit("ns")
    o = df_1m["open"].to_numpy(dtype=float)
    h = df_1m["high"].to_numpy(dtype=float)
    l = df_1m["low"].to_numpy(dtype=float)
    c = df_1m["close"].to_numpy(dtype=float)
    v = df_1m["volume"].fillna(0).to_numpy(dtype=int)
    mid = (o + c) / 2.0

    dt1 = dt.to_numpy()
    dt2 = (dt + pd.Timedelta(seconds=30)).to_numpy()

    out_o = np.repeat(mid, 2)
    out_o[0::2] = o
    out_c = np.repeat(mid, 2)
    out_c[1::2] = c

    hi_a = np.maximum(o, mid)
    lo_a = np.minimum(o, mid)
    hi_b = np.maximum(mid, c)
    lo_b = np.minimum(mid, c)
    out_h = np.empty(n * 2)
    out_l = np.empty(n * 2)
    out_h[0::2] = hi_a
    out_h[1::2] = hi_b
    out_l[0::2] = lo_a
    out_l[1::2] = lo_b

    vol_half = np.maximum(v // 2, 1)
    out_v = np.empty(n * 2, dtype=int)
    out_v[0::2] = vol_half
    out_v[1::2] = np.maximum(v - vol_half, 1)

    out_dt = np.empty(n * 2, dtype=dt1.dtype)
    out_dt[0::2] = dt1
    out_dt[1::2] = dt2

    return normalize_dt(
        pd.DataFrame(
            {
                "datetime": out_dt,
                "open": out_o,
                "high": out_h,
                "low": out_l,
                "close": out_c,
                "volume": out_v,
            }
        )
    )


def add_5m_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_20"] = mtf.calculate_ema(df["close"], EMA_TREND)
    df["adx"], df["di_plus"], df["di_minus"] = mtf.calculate_adx(df, 14)
    df["atr"] = mtf.calculate_atr(df, 14)
    df["vwap"] = compute_vwap(df)
    return df


def add_1m_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_20"] = mtf.calculate_ema(df["close"], EMA_TREND)
    df["atr"] = mtf.calculate_atr(df, 14)
    df["range"] = (df["high"] - df["low"]).abs()
    df["avg_range"] = df["range"].rolling(CANDLE_AVG_PERIOD).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_prev"] = df["rsi"].shift(1)
    return df


def attach_context(
    trigger: pd.DataFrame,
    ctx: pd.DataFrame,
    cols: List[str],
    suffix: str,
) -> pd.DataFrame:
    right = ctx[["datetime"] + cols].rename(columns={c: f"{c}{suffix}" for c in cols})
    return pd.merge_asof(
        trigger.sort_values("datetime"),
        right.sort_values("datetime"),
        on="datetime",
        direction="backward",
    ).reset_index(drop=True)


def trend_array(close: np.ndarray, vwap: np.ndarray, ema20: np.ndarray, adx: np.ndarray) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)  # -1 short, 0 none, 1 long
    ok = (~np.isnan(adx)) & (adx >= ADX_MIN) & (~np.isnan(vwap)) & (~np.isnan(ema20))
    long_m = ok & (close > vwap) & (close > ema20)
    short_m = ok & (close < vwap) & (close < ema20)
    out[long_m] = 1
    out[short_m] = -1
    return out


def pullback_mask(
    close: np.ndarray,
    ema20: np.ndarray,
    atr: np.ndarray,
    direction: np.ndarray,
    pb_limit: float,
) -> np.ndarray:
    zone = atr * pb_limit
    dist = np.abs(close - ema20)
    ok = (~np.isnan(atr)) & (atr > 0) & (dist <= zone)
    long_ok = ok & (direction == 1) & (close >= ema20 - zone * 0.2)
    short_ok = ok & (direction == -1) & (close <= ema20 + zone * 0.2)
    return long_ok | short_ok


def trigger_mask(o: np.ndarray, c: np.ndarray, ph: np.ndarray, pl: np.ndarray, direction: int) -> np.ndarray:
    if direction == 1:
        return (c > o) & (c > ph)
    return (c < o) & (c < pl)


def no_chase_mask(
    close: np.ndarray,
    ema20: np.ndarray,
    atr: np.ndarray,
    avg_range: np.ndarray,
    bar_range: np.ndarray,
) -> np.ndarray:
    big = (~np.isnan(avg_range)) & (avg_range > 0) & (bar_range > NO_CHASE_SIZE_MULT * avg_range)
    far = (~np.isnan(ema20)) & (~np.isnan(atr)) & (atr > 0) & (np.abs(close - ema20) > NO_CHASE_EMA_ATR * atr)
    return big | far


def simulate_exit(
    dt: np.ndarray,
    hi: np.ndarray,
    lo: np.ndarray,
    cl: np.ndarray,
    start: int,
    direction: int,
    entry_price: float,
    sl: float,
    tp: float,
    entry_ts: pd.Timestamp,
    max_hold_sec: int,
) -> Tuple[pd.Timestamp, float, str, float]:
    for j in range(start, len(dt)):
        ts = pd.Timestamp(dt[j])
        if direction == 1:
            if lo[j] <= sl:
                return ts, sl, "SL", (ts - entry_ts).total_seconds()
            if hi[j] >= tp:
                return ts, tp, "TP", (ts - entry_ts).total_seconds()
        else:
            if hi[j] >= sl:
                return ts, sl, "SL", (ts - entry_ts).total_seconds()
            if lo[j] <= tp:
                return ts, tp, "TP", (ts - entry_ts).total_seconds()
        hold = (ts - entry_ts).total_seconds()
        if max_hold_sec > 0 and hold >= max_hold_sec:
            return ts, float(cl[j]), "MAX_HOLD", hold
    ts = pd.Timestamp(dt[-1])
    return ts, float(cl[-1]), "END", (ts - entry_ts).total_seconds()


def run_variant(
    variant: VariantConfig,
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_30s: pd.DataFrame,
    symbol: str,
    sl_pts: float,
    tp_pts: float,
    max_hold_sec: int,
) -> Dict[str, Any]:
    trigger = df_1m if not variant.use_30s_trigger else df_30s
    trigger = attach_context(trigger, df_1m, ["ema_20", "atr", "avg_range", "range"], "_1m")
    trigger = attach_context(trigger, df_5m, ["close", "vwap", "ema_20", "adx"], "_5m")

    dt = trigger["datetime"].to_numpy()
    o = trigger["open"].to_numpy(dtype=float)
    hi = trigger["high"].to_numpy(dtype=float)
    lo = trigger["low"].to_numpy(dtype=float)
    cl = trigger["close"].to_numpy(dtype=float)
    ema20_1m = trigger["ema_20_1m"].to_numpy(dtype=float)
    atr_1m = trigger["atr_1m"].to_numpy(dtype=float)
    avg_rng = trigger["avg_range_1m"].to_numpy(dtype=float)
    bar_rng = trigger["range_1m"].to_numpy(dtype=float)

    close_5m = trigger["close_5m"].to_numpy(dtype=float)
    vwap_5m = trigger["vwap_5m"].to_numpy(dtype=float)
    ema20_5m = trigger["ema_20_5m"].to_numpy(dtype=float)
    adx_5m = trigger["adx_5m"].to_numpy(dtype=float)

    trend = trend_array(close_5m, vwap_5m, ema20_5m, adx_5m)
    ph = np.roll(hi, 1)
    pl = np.roll(lo, 1)
    ph[0] = np.nan
    pl[0] = np.nan

    setup_window = MAX_SETUP_BARS * (2 if variant.use_30s_trigger else 1)
    trades: List[ScalpTrade] = []
    daily: Dict[str, int] = {}
    pv = POINT_VALUE.get(symbol, 2.0)

    phase = 0  # 0 wait/trend, 1 setup
    trend_dir = 0
    setup_until = -1
    i = max(WARMUP, 1)

    while i < len(trigger):
        ts = pd.Timestamp(dt[i])
        day_key = str(ts.date())
        if daily.get(day_key, 0) >= MAX_TRADES_PER_DAY:
            i += 1
            continue

        direction = int(trend[i])
        adx = adx_5m[i]
        if variant.adaptive_trend:
            if adx >= STRONG_ADX:
                pb_limit = PULLBACK_ATR_STRONG
            elif adx < WEAK_ADX:
                pb_limit = PULLBACK_ATR_WEAK
            else:
                pb_limit = PULLBACK_ATR
        else:
            pb_limit = PULLBACK_ATR

        if direction == 0:
            phase = 0
            trend_dir = 0
            i += 1
            continue

        if phase == 0:
            if not pullback_mask(
                np.array([cl[i]]),
                np.array([ema20_1m[i]]),
                np.array([atr_1m[i]]),
                np.array([direction]),
                pb_limit,
            )[0]:
                i += 1
                continue
            phase = 1
            trend_dir = direction
            setup_until = i + setup_window
            i += 1
            continue

        if phase == 1:
            if i > setup_until or direction != trend_dir:
                phase = 0
                trend_dir = 0
                continue
            if not trigger_mask(
                np.array([o[i]]),
                np.array([cl[i]]),
                np.array([ph[i]]),
                np.array([pl[i]]),
                trend_dir,
            )[0]:
                i += 1
                continue
            if variant.no_chase and no_chase_mask(
                np.array([cl[i]]),
                np.array([ema20_1m[i]]),
                np.array([atr_1m[i]]),
                np.array([avg_rng[i]]),
                np.array([bar_rng[i]]),
            )[0]:
                i += 1
                continue

            entry = float(cl[i])
            if trend_dir == 1:
                sl = entry - sl_pts
                tp = entry + tp_pts
            else:
                sl = entry + sl_pts
                tp = entry - tp_pts

            exit_ts, exit_px, reason, hold_sec = simulate_exit(
                dt, hi, lo, cl, i + 1, trend_dir, entry, sl, tp, ts, max_hold_sec
            )
            pts = (exit_px - entry) if trend_dir == 1 else (entry - exit_px)
            trades.append(
                ScalpTrade(
                    entry_time=ts,
                    direction="long" if trend_dir == 1 else "short",
                    entry_price=entry,
                    sl=sl,
                    tp=tp,
                    exit_time=exit_ts,
                    exit_price=exit_px,
                    pnl=pts * pv - COMMISSION * 2,
                    exit_reason=reason,
                    version=variant.name,
                    hold_seconds=hold_sec,
                )
            )
            daily[day_key] = daily.get(day_key, 0) + 1
            phase = 0
            trend_dir = 0
            j = i + 1
            while j < len(dt) and pd.Timestamp(dt[j]) <= exit_ts:
                j += 1
            i = max(j, i + 1)
            continue

        i += 1

    return summarize(trades, variant.name, symbol)


def summarize(trades: List[ScalpTrade], version: str, symbol: str) -> Dict[str, Any]:
    if not trades:
        return {
            "version": version,
            "symbol": symbol,
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "avg_hold_sec": 0.0,
            "tp_exits": 0,
            "sl_exits": 0,
            "max_hold_exits": 0,
        }

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

    equity = peak = max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    return {
        "version": version,
        "symbol": symbol,
        "trades": len(trades),
        "win_rate": round(100.0 * len(wins) / len(trades), 1),
        "profit_factor": round(min(pf, 999.0), 2),
        "expectancy": round(float(np.mean(pnls)), 2),
        "total_pnl": round(sum(pnls), 2),
        "max_drawdown": round(max_dd, 2),
        "avg_hold_sec": round(float(np.mean([t.hold_seconds for t in trades])), 1),
        "tp_exits": sum(1 for t in trades if t.exit_reason == "TP"),
        "sl_exits": sum(1 for t in trades if t.exit_reason == "SL"),
        "max_hold_exits": sum(1 for t in trades if t.exit_reason == "MAX_HOLD"),
    }


def print_table(rows: List[Dict[str, Any]], data_note: str) -> None:
    print(f"\n{'=' * 88}")
    print("  SCALP MOMENTUM BACKTEST — A/B/C/D")
    print(f"  {data_note}")
    print(f"{'=' * 88}")
    hdr = f"{'Ver':<4} {'Trades':>6} {'WR%':>6} {'PF':>6} {'Exp$':>8} {'PnL$':>10} {'MaxDD$':>8} {'Hold s':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['version']:<4} {r['trades']:>6} {r['win_rate']:>6.1f} "
            f"{r['profit_factor']:>6.2f} {r['expectancy']:>8.2f} {r['total_pnl']:>10.2f} "
            f"{r['max_drawdown']:>8.2f} {r['avg_hold_sec']:>7.1f}"
        )
    print()


def load_data(
    csv_1m: Optional[str],
    csv_5m: Optional[str],
    csv_30s: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    p1 = csv_1m or os.path.join(data_dir, "MNQ_1m.csv")
    df_1m = normalize_dt(pd.read_csv(p1, parse_dates=["datetime"]))

    if csv_5m and os.path.isfile(csv_5m):
        df_5m = normalize_dt(pd.read_csv(csv_5m, parse_dates=["datetime"]))
    elif os.path.isfile(p1.replace("_1m", "_5m")):
        df_5m = normalize_dt(pd.read_csv(p1.replace("_1m", "_5m"), parse_dates=["datetime"]))
    else:
        df_5m = normalize_dt(resample_5m(df_1m))

    if csv_30s and os.path.isfile(csv_30s):
        df_30s = normalize_dt(pd.read_csv(csv_30s, parse_dates=["datetime"]))
        data_note = f"30s: real CSV ({len(df_30s):,} bars) | 1m: {os.path.basename(p1)} ({len(df_1m):,})"
    else:
        df_30s = resample_30s_synthetic(df_1m)
        data_note = (
            f"30s: SYNTHETIC (2x per 1m bar — imperfect) | "
            f"1m: {os.path.basename(p1)} ({len(df_1m):,}) | 5m: {len(df_5m):,}"
        )
    return df_1m, df_5m, df_30s, data_note


def main() -> None:
    parser = argparse.ArgumentParser(description="Scalp momentum backtest A/B/C/D")
    parser.add_argument("--symbol", default="MNQ")
    parser.add_argument("--csv-1m", default=None)
    parser.add_argument("--csv-5m", default=None)
    parser.add_argument("--csv-30s", default=None, help="Real 30s CSV if available")
    parser.add_argument("--sl", type=float, default=8.0, help="Stop loss points")
    parser.add_argument("--tp", type=float, default=15.0, help="Take profit points")
    parser.add_argument("--max-hold", type=int, default=90, help="Max hold seconds (0=off)")
    args = parser.parse_args()

    df_1m_raw, df_5m_raw, df_30s, data_note = load_data(args.csv_1m, args.csv_5m, args.csv_30s)
    df_1m = add_1m_indicators(df_1m_raw)
    df_5m = add_5m_indicators(df_5m_raw)
    start, end = df_1m["datetime"].iloc[0], df_1m["datetime"].iloc[-1]
    print(f"\nLoaded {args.symbol}: {len(df_1m):,} x 1m  ({start} -> {end})")
    print(f"SL={args.sl}pt  TP={args.tp}pt  MAX_HOLD={args.max_hold}s")

    variants = [
        VariantConfig("A", use_30s_trigger=False),
        VariantConfig("B", use_30s_trigger=True),
        VariantConfig("C", use_30s_trigger=True, no_chase=True),
        VariantConfig("D", use_30s_trigger=True, no_chase=True, adaptive_trend=True),
    ]

    results: List[Dict[str, Any]] = []
    for v in variants:
        print(f"  Running variant {v.name}...", flush=True)
        stats = run_variant(v, df_1m, df_5m, df_30s, args.symbol, args.sl, args.tp, args.max_hold)
        stats["params"] = {"sl_pts": args.sl, "tp_pts": args.tp, "max_hold_sec": args.max_hold}
        stats["data"] = data_note
        stats["period"] = {"start": str(start), "end": str(end)}
        results.append(stats)

    print_table(results, data_note)

    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "scalp_backtest_results.json")
    payload = {
        "symbol": args.symbol,
        "data_note": data_note,
        "period": {"start": str(start), "end": str(end), "bars_1m": len(df_1m)},
        "params": {"sl_pts": args.sl, "tp_pts": args.tp, "max_hold_sec": args.max_hold},
        "variants": results,
        "live_30s_test": "scripts/test_rithmic_30s_bars.py — real SECOND_BAR confirmed when credentials set",
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
