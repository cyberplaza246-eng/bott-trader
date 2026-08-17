#!/usr/bin/env python
"""
Standalone reconstruction + test of the LiquiditySweepAnalyzer sweep +
Market-Structure-Shift (MSS) entry logic, isolated from the ML confidence
layer, sentiment, dynamic SL/TP, and every other filter it was bundled with
live (VWAP, ADX, EMA, RSI-as-a-filter, PTJ 200-SMA -- none of those are
added here; RSI is used only where the ORIGINAL sweep logic itself used it,
as documented in the audit below).

Source reconstructed from: legacy/src/ai/liquidity_sweep.py
  detect_swing_points()  -> swing pivot detection (both layers)
  detect_regime()        -> Layer 1: 5M swing structure & directional bias
  detect_sweep()          -> Layer 2: liquidity sweep of a swing level (1M)
  detect_mss()             -> Layer 3: market structure shift / displacement

This script does NOT modify start_live_rithmic.py or any live-trading file.
It is a read-only reconstruction run against locally-held 1M/5M OHLCV data
already in data/*.csv (no new data purchased or downloaded).

Usage:
    python scripts/research_sweep_mss.py --symbol MES MNQ NQ
"""
from __future__ import annotations

import argparse
import bisect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.instruments.instrument_registry import REGISTRY  # noqa: E402
from src.strategies.indicators import atr as atr_ind, rsi as rsi_ind  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────
# Constants copied VERBATIM from legacy/src/ai/liquidity_sweep.py.
# Not one of these is tuned in this script -- see the research-log entry
# for the single, explicitly-flagged exception (SWEEP_WINDOW / MAX_SWEEP_AGE
# / CONFIRMATION_WINDOW are bar-counts; with real 1M data available they are
# used exactly as originally specified, in 1M bars, no rescaling needed).
# ─────────────────────────────────────────────────────────────────────────
SWING_LOOKBACK = 3
SWING_MIN_POINTS = 2
SWEEP_WINDOW = 30          # 1M candles searched for a sweep event
MAX_SWEEP_AGE = 5          # only accept sweeps in the last N candles (no stale re-fire)
RSI_SLOPE_WINDOW = 3
RSI_SWEEP_LONG_MAX = 55
RSI_SWEEP_SHORT_MIN = 45
CONFIRMATION_WINDOW = 20   # candles after sweep to find MSS displacement
BODY_RATIO_MIN = 0.30
SL_ATR_BUFFER = 0.50       # SL = sweep wick +/- 0.5 x ATR (documented constant)
TP_R_MULT = 1.5            # "1.5R default" per module docstring -- NOT the
                            # later hand-tuned per-symbol RR values found in
                            # src/risk/sl_tp.py's edit history, which are
                            # excluded deliberately (see research-log entry).
MAX_HOLD_BARS_1M = 500     # ~8h safety valve, identical across all 3 ablations,
                            # not part of the original logic (added because a
                            # bracket-only backtest needs SOME terminal exit).
ATR_PERIOD = 14
RSI_PERIOD = 14
COMMISSION_SLIPPAGE_TICKS = 1  # matches project convention: commission_rt + 1 tick


# ─────────────────────────────────────────────────────────────────────────
# Swing point detection -- causal-safe: a pivot at index i is only "known"
# once bar i+lookback exists. Precomputing over the full series is safe
# PROVIDED consumers only use a swing point once the bar that confirms it
# (idx + lookback) is <= the current evaluation bar. That rule is enforced
# everywhere below via `confirmed_idx = idx + lookback`.
# ─────────────────────────────────────────────────────────────────────────
def detect_swing_points(high: np.ndarray, low: np.ndarray, lookback: int) -> list[dict]:
    n = len(high)
    points = []
    for i in range(lookback, n - lookback):
        window_h = high[i - lookback: i + lookback + 1]
        if high[i] == window_h.max() and np.argmax(window_h == high[i]) == lookback:
            points.append({"idx": i, "confirmed_idx": i + lookback, "price": high[i], "type": "high"})
        window_l = low[i - lookback: i + lookback + 1]
        if low[i] == window_l.min() and np.argmax(window_l == low[i]) == lookback:
            points.append({"idx": i, "confirmed_idx": i + lookback, "price": low[i], "type": "low"})
    points.sort(key=lambda p: p["idx"])

    last_h, last_l = None, None
    for p in points:
        if p["type"] == "high":
            p["label"] = ("HH" if p["price"] > last_h else "LH") if last_h is not None else None
            last_h = p["price"]
        else:
            p["label"] = ("HL" if p["price"] > last_l else "LL") if last_l is not None else None
            last_l = p["price"]
    return points


def bias_series_5m(df5: pd.DataFrame) -> pd.Series:
    """Layer 1: directional bias from 5M swing structure, as of each 5M bar
    close. No ADX/ATR regime labels included (those never gate entries in
    the original code -- they only relabel 'regime' string, which is unused
    downstream for filtering); rule #11 keeps ADX out entirely regardless."""
    high, low, close = df5["high"].values, df5["low"].values, df5["close"].values
    n = len(df5)
    swings = detect_swing_points(high, low, SWING_LOOKBACK)

    bias = np.full(n, None, dtype=object)
    highs_so_far, lows_so_far = [], []
    swing_ptr = 0
    for i in range(n):
        while swing_ptr < len(swings) and swings[swing_ptr]["confirmed_idx"] <= i:
            sp = swings[swing_ptr]
            (highs_so_far if sp["type"] == "high" else lows_so_far).append(sp)
            swing_ptr += 1

        recent_highs = highs_so_far[-4:]
        recent_lows = lows_so_far[-4:]
        if len(recent_highs) + len(recent_lows) < SWING_MIN_POINTS:
            continue
        hh = sum(1 for s in recent_highs if s["label"] == "HH")
        lh = sum(1 for s in recent_highs if s["label"] == "LH")
        hl = sum(1 for s in recent_lows if s["label"] == "HL")
        ll = sum(1 for s in recent_lows if s["label"] == "LL")
        bull, bear = hh + hl, lh + ll
        if bull >= 1 and bull > bear:
            bias[i] = "BUY"
        elif bear >= 1 and bear > bull:
            bias[i] = "SELL"
        # tie -> no bias asserted here (original used EMA 20/50 tiebreak;
        # EMA is excluded from this experiment per rule #11, so ties = no bias)
    return pd.Series(bias, index=df5.index)


def get_pip_size(price: float) -> float:
    return 0.25  # all three instruments here are 0.25-tick futures


def build_signals(df1: pd.DataFrame, bias_1m: pd.Series, rsi1: np.ndarray) -> pd.DataFrame:
    """Single forward pass over 1M bars producing three independent signal
    series (sweep-only, mss-only/BOS, sweep+MSS), each firing at most once
    per underlying structural event (deduplicated by sweep/break index)."""
    high, low, close, opn = (df1["high"].values, df1["low"].values,
                              df1["close"].values, df1["open"].values)
    n = len(df1)
    bias = bias_1m.values

    swings1m = detect_swing_points(high, low, SWING_LOOKBACK)
    # Group swings by confirmation index for O(1) "swings confirmed by bar i" access.
    swings_by_conf = {}
    for sp in swings1m:
        swings_by_conf.setdefault(sp["confirmed_idx"], []).append(sp)

    sweep_signal = np.full(n, None, dtype=object)   # 'BUY'/'SELL' -- (a) sweep only
    mss_signal = np.full(n, None, dtype=object)      # 'BUY'/'SELL' -- (b) MSS/BOS only, no sweep required
    full_signal = np.full(n, None, dtype=object)     # 'BUY'/'SELL' -- (c) sweep THEN MSS (as originally coded)
    sweep_wick_out = np.full(n, np.nan)
    mss_level_out = np.full(n, np.nan)

    # confirmed_highs/confirmed_lows are append-only in increasing `idx` order
    # (confirmed_idx = idx + lookback is monotonic), so a parallel idx array
    # lets every "most recent swing with idx <= X" lookup use bisect (O(log n))
    # instead of rescanning the whole growing list every bar (was O(n) per
    # bar -> O(n^2) overall; this was the actual cause of the multi-minute
    # runtimes, not the size of the dataset).
    confirmed_highs: list[dict] = []
    confirmed_lows: list[dict] = []
    confirmed_highs_idx: list[int] = []
    confirmed_lows_idx: list[int] = []
    already_fired_sweep_idx: set[int] = set()
    already_fired_break_idx: set[int] = set()

    for i in range(SWING_LOOKBACK * 2 + SWEEP_WINDOW + 5, n):
        for sp in swings_by_conf.get(i, []):
            if sp["type"] == "high":
                confirmed_highs.append(sp)
                confirmed_highs_idx.append(sp["idx"])
            else:
                confirmed_lows.append(sp)
                confirmed_lows_idx.append(sp["idx"])

        b = bias[i]
        if b is None:
            continue

        # ---- (b) MSS / break-of-structure, standalone, no sweep required ----
        # "Break of structure": a displacement candle at bar i closes beyond
        # the most recent internal swing level in the bias direction.
        if b == "BUY" and confirmed_highs:
            level = confirmed_highs[-1]["price"]
            level_idx = confirmed_highs[-1]["idx"]
            body = close[i] - opn[i]
            rng = high[i] - low[i]
            body_ratio = body / rng if rng > 0 else 0
            is_disp = (close[i] > opn[i]) and (body_ratio >= BODY_RATIO_MIN)
            if is_disp and close[i] > level and level_idx not in already_fired_break_idx:
                mss_signal[i] = "BUY"
                mss_level_out[i] = level
                already_fired_break_idx.add(level_idx)
        elif b == "SELL" and confirmed_lows:
            level = confirmed_lows[-1]["price"]
            level_idx = confirmed_lows[-1]["idx"]
            body = opn[i] - close[i]
            rng = high[i] - low[i]
            body_ratio = body / rng if rng > 0 else 0
            is_disp = (close[i] < opn[i]) and (body_ratio >= BODY_RATIO_MIN)
            if is_disp and close[i] < level and level_idx not in already_fired_break_idx:
                mss_signal[i] = "SELL"
                mss_level_out[i] = level
                already_fired_break_idx.add(level_idx)

        # ---- (a) sweep detection: scan last MAX_SWEEP_AGE candles for a
        #      wick-through-level-then-close-back-inside event, at swing
        #      levels confirmed as of >= SWEEP_WINDOW bars ago (matches
        #      original's df_1m.iloc[:len-SWEEP_WINDOW] exclusion) ----
        stale_cutoff = i - SWEEP_WINDOW
        if b == "BUY":
            pos = bisect.bisect_right(confirmed_lows_idx, stale_cutoff)
            candidates = confirmed_lows[max(0, pos - 3):pos]
        else:
            pos = bisect.bisect_right(confirmed_highs_idx, stale_cutoff)
            candidates = confirmed_highs[max(0, pos - 3):pos]

        tol = get_pip_size(close[i])
        found_sweep = None
        for j in range(max(0, i - MAX_SWEEP_AGE + 1), i + 1):
            for lvl in candidates:
                level = lvl["price"]
                if b == "BUY":
                    swept = low[j] <= level + tol and close[j] > level
                    rsi_ok = rsi1[j] <= RSI_SWEEP_LONG_MAX
                else:
                    swept = high[j] >= level - tol and close[j] < level
                    rsi_ok = rsi1[j] >= RSI_SWEEP_SHORT_MIN
                if not (swept and rsi_ok):
                    continue
                # RSI slope confirmation using bars between sweep candle j and now (i) --
                # all already-elapsed data relative to "now" = i, not a lookahead.
                slope_ok = j >= i  # sweep is the current bar -> can't confirm yet, auto-accept
                for look in range(1, RSI_SLOPE_WINDOW + 1):
                    k = j + look
                    if k > i:
                        break
                    if (b == "BUY" and rsi1[k] > rsi1[j]) or (b == "SELL" and rsi1[k] < rsi1[j]):
                        slope_ok = True
                        break
                if not slope_ok:
                    continue
                found_sweep = (j, level, low[j] if b == "BUY" else high[j])
                break
            if found_sweep:
                break

        if found_sweep and found_sweep[0] not in already_fired_sweep_idx:
            sweep_idx, swept_level, wick = found_sweep
            sweep_signal[i] = b
            sweep_wick_out[i] = wick
            already_fired_sweep_idx.add(sweep_idx)

            # ---- (c) MSS confirmation after this sweep: pre-sweep internal
            #      structure level, then a displacement candle (at or after
            #      the sweep bar, within CONFIRMATION_WINDOW) breaking it,
            #      on the correct side of the sweep wick ----
            pre_highs_pos = bisect.bisect_left(confirmed_highs_idx, sweep_idx)
            pre_lows_pos = bisect.bisect_left(confirmed_lows_idx, sweep_idx)
            if b == "BUY":
                mss_level = confirmed_highs[pre_highs_pos - 1]["price"] if pre_highs_pos > 0 else (
                    high[max(0, sweep_idx - 10):sweep_idx].max() if sweep_idx > 0 else None)
            else:
                mss_level = confirmed_lows[pre_lows_pos - 1]["price"] if pre_lows_pos > 0 else (
                    low[max(0, sweep_idx - 10):sweep_idx].min() if sweep_idx > 0 else None)

            if mss_level is not None:
                for k in range(sweep_idx, min(sweep_idx + CONFIRMATION_WINDOW, i + 1)):
                    if k == 0:
                        continue
                    body = (close[k] - opn[k]) if b == "BUY" else (opn[k] - close[k])
                    rng = high[k] - low[k]
                    body_ratio = body / rng if rng > 0 else 0
                    is_disp = (body > 0) and (body_ratio >= BODY_RATIO_MIN)
                    if not is_disp:
                        continue
                    if b == "BUY" and close[k] > mss_level and close[k] > wick:
                        full_signal[k] = "BUY"
                        break
                    if b == "SELL" and close[k] < mss_level and close[k] < wick:
                        full_signal[k] = "SELL"
                        break

    return pd.DataFrame({
        "sweep_signal": sweep_signal,
        "mss_signal": mss_signal,
        "full_signal": full_signal,
        "sweep_wick": sweep_wick_out,
        "mss_level": mss_level_out,
    }, index=df1.index)


def simulate_trades(df1: pd.DataFrame, signal_col: pd.Series, symbol: str,
                     reference_col: pd.Series, atr1: np.ndarray) -> list[dict]:
    """Execute at NEXT bar's close after a signal bar. SL = reference (sweep
    wick, or broken structure level for mss-only) +/- 0.5xATR. TP = 1.5R.
    No intrabar-fill assumption for entry; SL/TP touch checked from the bar
    AFTER entry onward using that bar's high/low (standard bracket
    simulation, conservative same-bar-both-touched -> assume SL first)."""
    spec = REGISTRY[symbol]
    high, low, close = df1["high"].values, df1["low"].values, df1["close"].values
    n = len(df1)
    sig = signal_col.values
    ref = reference_col.values
    trades = []

    i = 0
    while i < n - 1:
        s = sig[i]
        if s is None or (isinstance(s, float) and pd.isna(s)):
            i += 1
            continue
        entry_idx = i + 1
        if entry_idx >= n or pd.isna(ref[i]) or pd.isna(atr1[i]) or atr1[i] <= 0:
            i += 1
            continue
        entry_price = close[entry_idx]
        sl_dist = SL_ATR_BUFFER * atr1[i]
        if s == "BUY":
            sl = ref[i] - sl_dist
            risk = entry_price - sl
            if risk <= 0:
                i += 1
                continue
            tp = entry_price + TP_R_MULT * risk
        else:
            sl = ref[i] + sl_dist
            risk = sl - entry_price
            if risk <= 0:
                i += 1
                continue
            tp = entry_price - TP_R_MULT * risk

        exit_price, exit_type, exit_idx = None, None, None
        for j in range(entry_idx + 1, min(entry_idx + 1 + MAX_HOLD_BARS_1M, n)):
            hit_sl = (low[j] <= sl) if s == "BUY" else (high[j] >= sl)
            hit_tp = (high[j] >= tp) if s == "BUY" else (low[j] <= tp)
            if hit_sl:
                exit_price, exit_type, exit_idx = sl, "STOP", j
                break
            if hit_tp:
                exit_price, exit_type, exit_idx = tp, "TARGET", j
                break
        if exit_price is None:
            exit_idx = min(entry_idx + MAX_HOLD_BARS_1M, n - 1)
            exit_price, exit_type = close[exit_idx], "TIME"

        direction = 1 if s == "BUY" else -1
        ticks = (exit_price - entry_price) / spec.tick_size * direction
        gross = ticks * spec.tick_value_usd
        commission = spec.commission_rt
        slippage = COMMISSION_SLIPPAGE_TICKS * spec.tick_size * spec.tick_value_usd
        net = gross - commission - slippage

        trades.append({
            "entry_time": df1.index[entry_idx], "exit_time": df1.index[exit_idx],
            "direction": "long" if s == "BUY" else "short",
            "entry": entry_price, "exit": exit_price, "exit_type": exit_type,
            "r_risked": risk, "pnl": net,
        })
        i = exit_idx + 1  # no overlapping trades

    return trades


def stats_block(trades: list[dict], label: str) -> dict:
    n = len(trades)
    if n == 0:
        return {"label": label, "n": 0, "insufficient": True}
    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / n * 100
    gross_w = wins.sum() if len(wins) else 0.0
    gross_l = abs(losses.sum()) if len(losses) else 0.0
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    expectancy = pnls.mean()
    cum = np.cumsum(pnls)
    max_dd = float(np.max(np.maximum.accumulate(cum) - cum)) if n else 0.0
    if n >= 2:
        t_stat, p_val = stats.ttest_1samp(pnls, 0.0)
    else:
        t_stat, p_val = float("nan"), float("nan")
    return {
        "label": label, "n": n, "insufficient": n < 30,
        "win_rate": win_rate, "expectancy": expectancy, "net_pnl": pnls.sum(),
        "profit_factor": pf, "max_dd": max_dd, "t_stat": t_stat, "p_value": p_val,
    }


def print_stats(s: dict):
    if s["n"] == 0:
        print(f"  {s['label']:28s} n=0  (no signals generated)")
        return
    flag = "  [INSUFFICIENT n<30]" if s["insufficient"] else ""
    print(f"  {s['label']:28s} n={s['n']:4d}  WR={s['win_rate']:5.1f}%  "
          f"exp=${s['expectancy']:7.2f}  net=${s['net_pnl']:9.2f}  "
          f"PF={s['profit_factor']:5.2f}  maxDD=${s['max_dd']:8.2f}  "
          f"t={s['t_stat']:6.2f}  p={s['p_value']:.4f}{flag}")


def run_symbol(symbol: str) -> dict:
    df5 = pd.read_csv(f"data/{symbol}_5m.csv")
    df1 = pd.read_csv(f"data/{symbol}_1m.csv")
    for df in (df5, df1):
        col = "datetime" if "datetime" in df.columns else "date"
        df["datetime"] = pd.to_datetime(df[col], utc=True)
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

    print(f"\n{'=' * 78}\n{symbol}: {len(df1)} 1M bars, {len(df5)} 5M bars "
          f"({df1.index[0]} -> {df1.index[-1]})\n{'=' * 78}")

    bias5 = bias_series_5m(df5)
    bias5_df = bias5.to_frame("bias")
    # As-of merge: each 1M bar gets the most recently CLOSED 5M bar's bias
    # (backward direction -- never a same-or-future 5M bar).
    bias_1m = pd.merge_asof(
        df1[[]].reset_index(), bias5_df.reset_index(),
        on="datetime", direction="backward",
    ).set_index("datetime")["bias"]

    rsi1 = rsi_ind(df1["close"], RSI_PERIOD).values
    atr1 = atr_ind(df1["high"], df1["low"], df1["close"], ATR_PERIOD).values

    sig_df = build_signals(df1, bias_1m, rsi1)

    trades_sweep = simulate_trades(df1, sig_df["sweep_signal"], symbol, sig_df["sweep_wick"], atr1)
    trades_mss = simulate_trades(df1, sig_df["mss_signal"], symbol, sig_df["mss_level"], atr1)
    trades_full = simulate_trades(df1, sig_df["full_signal"], symbol, sig_df["sweep_wick"], atr1)

    results = {
        "A_sweep_only": stats_block(trades_sweep, f"{symbol} A: sweep only"),
        "B_mss_only": stats_block(trades_mss, f"{symbol} B: MSS/BOS only"),
        "C_sweep_and_mss": stats_block(trades_full, f"{symbol} C: sweep + MSS"),
    }
    n = len(df1)
    split = int(n * 0.8)
    split_date = df1.index[split]
    print(f"  Holdout split at {split_date} (last ~20% of history)")
    for key, trades in [("A_sweep_only", trades_sweep), ("B_mss_only", trades_mss), ("C_sweep_and_mss", trades_full)]:
        full_trades = [t for t in trades]
        holdout_trades = [t for t in trades if t["entry_time"] >= split_date]
        print_stats(results[key])
        print_stats(stats_block(holdout_trades, f"{symbol} {key} [holdout tail]"))
    results["_trades"] = {"A": trades_sweep, "B": trades_mss, "C": trades_full}
    results["_split_date"] = split_date
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", nargs="+", default=["MES", "MNQ", "NQ"])
    args = parser.parse_args()

    all_results = {}
    for symbol in args.symbol:
        all_results[symbol] = run_symbol(symbol)

    print(f"\n{'=' * 78}\nAGGREGATE ACROSS SYMBOLS (full history)\n{'=' * 78}")
    for key, name in [("A_sweep_only", "A: sweep only"), ("B_mss_only", "B: MSS/BOS only"),
                       ("C_sweep_and_mss", "C: sweep + MSS")]:
        pooled = []
        for symbol in args.symbol:
            pooled.extend(all_results[symbol]["_trades"][key[0]])
        print_stats(stats_block(pooled, f"ALL: {name}"))


if __name__ == "__main__":
    main()
