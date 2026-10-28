#!/usr/bin/env python3
"""Compare: official daily+15m+60m vs L6 morning-only (and L6 full for reference)."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Set

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_gapped_vs_full as gapped
import scripts.backtest_ema15_official as official
import scripts.find_real_1m_edge as edge
from src.strategy.mnq_15m_ema_eod import (
    DEFAULT_MAX_OPEN_LOTS,
    ENTRY_MINUTE,
    FLAT_MINUTE,
    MAX_TRADES_DAY,
    SEP_MIN,
    TP_R,
    atr15_on_1m,
    daily_trend_on_1m,
    quiet_1m_ok,
    sep15_on_1m,
    trend_15m_on_1m,
    window_index,
    window_label,
    window_requires_quality,
)

WARMUP = resp.WARMUP


def simulate_l6(
    df: pd.DataFrame,
    *,
    morning_only: bool = False,
) -> List[resp.TradeRec]:
    """Variant B + 1m ATR×1.5; optionally skip noon+ windows (12:00/13:30/14:30)."""
    stop_cfg = next(c for c in resp.STOP_CONFIGS if c["id"] == "1m_atr1.5_6_25")
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    work = df.copy()
    work["datetime"] = dt
    t15 = trend_15m_on_1m(work).to_numpy(dtype=int)
    td = daily_trend_on_1m(work).to_numpy(dtype=int)
    t5 = resp.trend_5m_on_1m(work).to_numpy(dtype=int)
    sep = sep15_on_1m(work).to_numpy(dtype=float)
    atr15 = atr15_on_1m(work).to_numpy(dtype=float)
    atr5 = resp.atr5_on_1m(work).to_numpy(dtype=float)
    atr1 = resp.atr1_on_1m(work).to_numpy(dtype=float)
    op = work["open"].to_numpy(dtype=float)
    hi = work["high"].to_numpy(dtype=float)
    lo = work["low"].to_numpy(dtype=float)
    cl = work["close"].to_numpy(dtype=float)
    et = dt.tz_convert(resp.ET)
    mins = (et.hour * 60 + et.minute).to_numpy()
    dow = et.weekday.to_numpy()
    dates = et.date

    recs: List[resp.TradeRec] = []
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
                tr = gapped._close_position(p, ts, float(exit_px), reason)
                tag = "l6_morning" if morning_only else "l6_full"
                tr.version = f"{tag}_{stop_cfg['id']}"
                recs.append(
                    resp.TradeRec(tr, p["win_name"], p["sl_pts"], "B", stop_cfg["id"])
                )
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
        if morning_only and window_requires_quality(win):
            continue

        side = int(t15[i])
        if not resp.stack_ok("B", side, int(td[i]), int(t15[i]), int(t5[i])):
            continue
        if not quiet_1m_ok(side, op[i], cl[i]):
            continue

        conf = resp.high_confidence_5m(side, int(t5[i]), float(sep[i]), SEP_MIN)
        if window_requires_quality(win) and not conf:
            continue
        if len(opens) >= DEFAULT_MAX_OPEN_LOTS:
            continue
        if len(opens) == 1 and (not conf or opens[0]["dir"] != side):
            continue

        use = resp.stop_pts(stop_cfg, float(atr15[i]), float(atr5[i]), float(atr1[i]))
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
            "sl_pts": use,
            "win": win,
            "win_name": window_label(win),
        })
        fired_windows.add(win)
        counts[day] = counts.get(day, 0) + 1

    return recs


def dump_line(label: str, trades: List) -> Dict[str, Any]:
    m = resp.group_metrics(trades)
    print(
        f"  {label:<18} n={m['trades']:4d} WR={m['win_rate']:5.1f}% "
        f"PF={m['profit_factor']:5.2f} Net=${m['total_pnl']:8.1f} "
        f"Exp=${m['expectancy']:7.2f} DD=${m['max_drawdown']:7.1f}"
    )
    return m


def report_recipe(name: str, trades: List) -> Dict[str, Any]:
    ins, oos = edge.split_is_oos(trades)
    print(f"\n{'=' * 90}")
    print(name)
    print("=" * 90)
    overall = dump_line("ALL", trades)
    is_m = dump_line("IS", ins)
    oos_m = dump_line("OOS", oos)

    by_month: Dict[str, Any] = {}
    buckets: Dict[str, List] = {}
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        key = ts.tz_convert(resp.ET).strftime("%Y-%m")
        buckets.setdefault(key, []).append(t)
    print("  By month:")
    for k in sorted(buckets):
        by_month[k] = dump_line(k, buckets[k])

    ok = (
        oos_m["trades"] >= 30
        and oos_m["profit_factor"] >= 1.15
        and oos_m["total_pnl"] > 0
        and is_m["profit_factor"] >= 1.05
        and is_m["total_pnl"] > 0
    )
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
    return {
        "name": name,
        "overall": overall,
        "is": is_m,
        "oos": oos_m,
        "by_month": by_month,
        "pass": ok,
    }


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    out = os.path.join("data", "trendcatcher_followups_backtest.json")

    print("FOLLOW-UP BACKTESTS")
    print("1) Official: daily+15m+60m, 15m ATR×2 clip 20-60, TP cap 500, 6 windows")
    print("2) L6 morning-only: 15m+5m, quiet 1m, 1m ATR×1.5, windows 9:35/10:15/11:00 only")
    print("3) L6 full (reference): all 6 windows")
    print(f"Data: {csv}")
    print("Costs: $2/pt + $1.24 RT | IS/OOS split: Jun 2026")

    df = official.load_1m(csv)
    print(f"bars={len(df):,}  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    results: List[Dict[str, Any]] = []

    official_trades = official.simulate_official(df)
    results.append(
        report_recipe(
            "OFFICIAL (daily+15m+60m, 15m ATR×2, TP500, 6 windows)",
            official_trades,
        )
    )

    stop_cfg = next(c for c in resp.STOP_CONFIGS if c["id"] == "1m_atr1.5_6_25")
    morning_recs = simulate_l6(df, morning_only=True)
    results.append(
        report_recipe(
            "L6 MORNING-ONLY (9:35/10:15/11:00, 1m ATR×1.5, 1R TP)",
            [r.trade for r in morning_recs],
        )
    )

    full_recs = resp.simulate(df, variant="B", stop_cfg=stop_cfg)
    results.append(
        report_recipe(
            "L6 FULL (reference, all 6 windows)",
            [r.trade for r in full_recs],
        )
    )

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    for r in results:
        o, oo = r["overall"], r["oos"]
        flag = "PASS" if r["pass"] else "FAIL"
        print(
            f"  [{flag}] {r['name'][:50]:<50} "
            f"OOS PF={oo['profit_factor']:.2f} ${oo['total_pnl']:.0f} | "
            f"ALL PF={o['profit_factor']:.2f} ${o['total_pnl']:.0f}"
        )

    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "csv": csv,
                "range": [str(df["datetime"].iloc[0]), str(df["datetime"].iloc[-1])],
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
