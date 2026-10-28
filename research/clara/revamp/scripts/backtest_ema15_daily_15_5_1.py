#!/usr/bin/env python3
"""
Compare ema15_eod trend filters: daily+15m+60m (baseline) vs daily+15m+5m+1m agreement.

Variant A — daily + 15m + 60m + quiet 1m (current live recipe, 6 gapped windows).
Variant D — daily + 15m + 5m + 1m agree + quiet 1m close (replace 60m with 5m+1m).

Shared exits: ATR×2 stop 20–60, 1R TP, flatten 15:50, max 20/day, window re-entry,
6 gapped windows, noon+ sep/ATR >= 0.45, 2-lot cap with sep gate.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_gapped_vs_full as gapped
import scripts.backtest_ema15_official as official
import scripts.backtest_scalp_momentum as scalp_bt
import scripts.find_real_1m_edge as edge
from src.strategy.mnq_15m_ema_eod import (
    DEFAULT_MAX_OPEN_LOTS,
    EMA_FAST,
    EMA_SLOW,
    ENTRY_MINUTE,
    FLAT_MINUTE,
    MAX_TRADES_DAY,
    SEP_MIN,
    TP_R,
    _as_utc_frame,
    _ohlc_resample,
    atr15_on_1m,
    daily_trend_on_1m,
    ema_series,
    high_confidence,
    quiet_1m_ok,
    sep15_on_1m,
    sl_pts_from_atr,
    trend_15m_on_1m,
    trend_60m_on_1m,
    window_index,
    window_requires_quality,
)

ET = pytz.timezone("US/Eastern")
WARMUP = 80


def trend_tf_on_1m(df_1m: pd.DataFrame, rule: str, *, completed_only: bool = True) -> pd.Series:
    """EMA8/21 trend on resampled bars, ffilled to 1m index."""
    tmp = _as_utc_frame(df_1m)
    bars = _ohlc_resample(tmp, rule)
    fast = ema_series(bars["close"], EMA_FAST)
    slow = ema_series(bars["close"], EMA_SLOW)
    trend = pd.Series(0, index=bars.index, dtype=int)
    trend.loc[fast > slow] = 1
    trend.loc[fast < slow] = -1
    if completed_only:
        trend = trend.shift(1)
    return trend.reindex(tmp.index, method="ffill").fillna(0).astype(int)


def trend_5m_on_1m(df_1m: pd.DataFrame, *, completed_only: bool = True) -> pd.Series:
    return trend_tf_on_1m(df_1m, "5min", completed_only=completed_only)


def trend_1m_completed(df_1m: pd.DataFrame) -> pd.Series:
    """Prior completed 1m bar EMA8 vs EMA21 (shift(1) avoids same-bar leak)."""
    tmp = _as_utc_frame(df_1m)
    fast = ema_series(tmp["close"], EMA_FAST)
    slow = ema_series(tmp["close"], EMA_SLOW)
    trend = pd.Series(0, index=tmp.index, dtype=int)
    trend.loc[fast > slow] = 1
    trend.loc[fast < slow] = -1
    return trend.shift(1).fillna(0).astype(int)


def high_confidence_5m(side: int, trend5: int, sep: float, sep_min: float = SEP_MIN) -> bool:
    """Afternoon / 2nd-lot gate using 5m (replaces 60m in daily+15+5+1 stack)."""
    if side == 0 or trend5 != side:
        return False
    if sep is None or (isinstance(sep, float) and (np.isnan(sep) or sep < sep_min)):
        return False
    return True


def daily_stack_ok(side: int, td: int, t5: int, t1: int) -> bool:
    """Daily + 15m + 5m + 1m all agree on direction."""
    return side != 0 and side == td and side == t5 and side == t1


def simulate_variant(
    df: pd.DataFrame,
    *,
    variant: str,
) -> List[scalp_bt.ScalpTrade]:
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    work = df.copy()
    work["datetime"] = dt
    t15 = trend_15m_on_1m(work).to_numpy(dtype=int)
    td = daily_trend_on_1m(work).to_numpy(dtype=int)
    t60 = trend_60m_on_1m(work).to_numpy(dtype=int)
    t5 = trend_5m_on_1m(work).to_numpy(dtype=int)
    t1 = trend_1m_completed(work).to_numpy(dtype=int)
    sep = sep15_on_1m(work).to_numpy(dtype=float)
    atr = atr15_on_1m(work).to_numpy(dtype=float)
    op = work["open"].to_numpy(dtype=float)
    hi = work["high"].to_numpy(dtype=float)
    lo = work["low"].to_numpy(dtype=float)
    cl = work["close"].to_numpy(dtype=float)
    et = dt.tz_convert(ET)
    mins = (et.hour * 60 + et.minute).to_numpy()
    dow = et.weekday.to_numpy()
    dates = et.date

    trades: List[scalp_bt.ScalpTrade] = []
    opens: List[Dict] = []
    fired_windows: Set[int] = set()
    counts: Dict = {}

    for i in range(WARMUP, len(work)):
        ts = pd.Timestamp(dt[i])
        m = int(mins[i])
        day = dates[i]
        still = []
        for p in opens:
            reason = exit_px = None
            if m >= FLAT_MINUTE:
                reason, exit_px = "EOD", float(cl[i])
            elif p["dir"] == 1:
                if lo[i] <= p["sl"]:
                    reason, exit_px = "SL", p["sl"]
                elif hi[i] >= p["tp"]:
                    reason, exit_px = "TP", p["tp"]
            else:
                if hi[i] >= p["sl"]:
                    reason, exit_px = "SL", p["sl"]
                elif lo[i] <= p["tp"]:
                    reason, exit_px = "TP", p["tp"]
            if reason:
                trades.append(gapped._close_position(p, ts, float(exit_px), reason))
                if p.get("win") is not None:
                    fired_windows.discard(int(p["win"]))
            else:
                still.append(p)
        opens = still

        if dow[i] >= 5:
            continue
        if m < ENTRY_MINUTE or m >= FLAT_MINUTE:
            continue
        if counts.get(day, 0) >= MAX_TRADES_DAY:
            continue

        win = window_index(m)
        if win is None or win in fired_windows:
            continue

        side = int(t15[i])
        if side == 0:
            continue

        if variant == "A":
            if side != int(td[i]) or side != int(t60[i]):
                continue
            if not quiet_1m_ok(side, op[i], cl[i]):
                continue
            conf = high_confidence(side, int(t60[i]), float(sep[i]), SEP_MIN)
        elif variant == "D":
            if not daily_stack_ok(side, int(td[i]), int(t5[i]), int(t1[i])):
                continue
            if not quiet_1m_ok(side, op[i], cl[i]):
                continue
            conf = high_confidence_5m(side, int(t5[i]), float(sep[i]), SEP_MIN)
        else:
            raise ValueError(f"unknown variant {variant}")

        if window_requires_quality(win) and not conf:
            continue
        if len(opens) >= DEFAULT_MAX_OPEN_LOTS:
            continue
        if len(opens) == 1 and (not conf or opens[0]["dir"] != side):
            continue

        use = sl_pts_from_atr(float(atr[i]))
        entry = float(cl[i])
        sl = entry - use if side == 1 else entry + use
        tp_dist = use * TP_R
        tp = entry + tp_dist if side == 1 else entry - tp_dist
        opens.append({
            "ts": ts,
            "dir": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "win": win,
            "variant": variant,
        })
        fired_windows.add(win)
        counts[day] = counts.get(day, 0) + 1

    return trades


def report(label: str, trades: List[scalp_bt.ScalpTrade]) -> Dict:
    ins, oos = edge.split_is_oos(trades)
    s_all = edge.summarize(trades)
    s_is = edge.summarize(ins)
    s_oos = edge.summarize(oos)
    avg_fpd, max_fpd = gapped.fills_per_day(trades)
    reasons: Dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print(f"\n=== {label} ===")
    print(
        f"  ALL  n={s_all['trades']:4d}  WR={s_all['win_rate']:5.1f}%  PF={s_all['profit_factor']:5.2f}  "
        f"${s_all['total_pnl']:9.1f}  DD=${s_all['max_drawdown']:7.1f}  avg/day={avg_fpd:.2f}  exits={reasons}"
    )
    print(
        f"  IS   n={s_is['trades']:4d}  WR={s_is['win_rate']:5.1f}%  PF={s_is['profit_factor']:5.2f}  "
        f"${s_is['total_pnl']:9.1f}  DD=${s_is['max_drawdown']:7.1f}"
    )
    print(
        f"  OOS  n={s_oos['trades']:4d}  WR={s_oos['win_rate']:5.1f}%  PF={s_oos['profit_factor']:5.2f}  "
        f"${s_oos['total_pnl']:9.1f}  DD=${s_oos['max_drawdown']:7.1f}"
    )
    return {
        "label": label,
        "trades": s_all["trades"],
        "win_rate": s_all["win_rate"],
        "pf": s_all["profit_factor"],
        "pnl": s_all["total_pnl"],
        "max_dd": s_all["max_drawdown"],
        "avg_fpd": avg_fpd,
        "is_trades": s_is["trades"],
        "is_pf": s_is["profit_factor"],
        "is_pnl": s_is["total_pnl"],
        "is_dd": s_is["max_drawdown"],
        "oos_trades": s_oos["trades"],
        "oos_pf": s_oos["profit_factor"],
        "oos_pnl": s_oos["total_pnl"],
        "oos_dd": s_oos["max_drawdown"],
    }


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    print("ema15_eod: daily+15m+60m (baseline) vs daily+15m+5m+1m agreement")
    print(f"file={csv}")
    print(
        "Rules: 6 gapped windows, sep/ATR>=0.45 noon+, ATRx2 SL 20-60, 1R TP, "
        f"flatten 15:50, cap={MAX_TRADES_DAY}/day, re-entry after close, max {DEFAULT_MAX_OPEN_LOTS} lots"
    )
    df = official.load_1m(csv)
    print(f"bars={len(df):,}  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    rows = [
        report("A - daily+15m+60m + quiet 1m", simulate_variant(df, variant="A")),
        report("D - daily+15+5+1 + quiet 1m", simulate_variant(df, variant="D")),
    ]

    print("\n" + "=" * 88)
    print("COMPARISON TABLE  (IS/OOS split at Jun 2026)")
    print("=" * 88)
    hdr = f"{'Variant':<32} {'Trades':>6} {'WR%':>6} {'PF':>6} {'Net $':>10} {'MaxDD $':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['label']:<32} {r['trades']:6d} {r['win_rate']:6.1f} {r['pf']:6.2f} "
            f"{r['pnl']:10.1f} {r['max_dd']:9.1f}"
        )

    print("\nOOS (Jun 2026+)")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['label']:<32} {r['oos_trades']:6d} "
            f"{'—':>6} {r['oos_pf']:6.2f} {r['oos_pnl']:10.1f} {r['oos_dd']:9.1f}"
        )

    baseline, variant_d = rows[0], rows[1]
    print("\nRECOMMENDATION")
    print(
        f"  A (baseline): {baseline['trades']} trades, PF {baseline['pf']:.2f}, "
        f"${baseline['pnl']:.0f} ALL / ${baseline['oos_pnl']:.0f} OOS, DD ${baseline['max_dd']:.0f}"
    )
    print(
        f"  D (daily+15+5+1): {variant_d['trades']} trades, PF {variant_d['pf']:.2f}, "
        f"${variant_d['pnl']:.0f} ALL / ${variant_d['oos_pnl']:.0f} OOS, DD ${variant_d['max_dd']:.0f}"
    )
    d_wins_oos = variant_d["oos_pnl"] > baseline["oos_pnl"]
    d_wins_pf = variant_d["pf"] >= baseline["pf"] and variant_d["oos_pf"] >= baseline["oos_pf"]
    d_wins_dd = variant_d["max_dd"] <= baseline["max_dd"]
    if d_wins_oos and d_wins_pf:
        winner = "D"
        print("  Winner: D — beats baseline on OOS PnL and PF. Worth a paper test.")
    elif d_wins_oos and d_wins_dd:
        winner = "D (marginal)"
        print("  Winner: D — better OOS $ and lower DD, but PF mixed. Paper test with caution.")
    elif baseline["oos_pnl"] >= variant_d["oos_pnl"] and baseline["pf"] >= variant_d["pf"]:
        winner = "A"
        print("  Winner: A — keep live daily+15m+60m stack.")
    else:
        winner = "A (marginal)"
        print("  Winner: A — D does not clearly beat baseline on OOS metrics.")
    print(f"  Verdict: {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
