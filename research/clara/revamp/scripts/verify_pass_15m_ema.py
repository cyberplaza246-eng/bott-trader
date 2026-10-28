#!/usr/bin/env python3
"""Verify the PASS from find_real_1m_simple with completed-bar honesty check."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_mtf_scalping as mtf
import scripts.backtest_scalp_momentum as scalp_bt
import scripts.find_real_1m_edge as edge
from scripts.find_real_1m_simple import signals_15m_trend

ET = pytz.timezone("US/Eastern")


def signals_15m_completed(frame: pd.DataFrame, long_only: bool = False):
    dt = pd.to_datetime(frame["datetime"], utc=True)
    tmp = frame.copy()
    tmp.index = dt
    b15 = tmp.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    e8 = mtf.calculate_ema(b15["close"], 8)
    e21 = mtf.calculate_ema(b15["close"], 21)
    trend = pd.Series(0, index=b15.index, dtype=int)
    trend.loc[e8 > e21] = 1
    trend.loc[e8 < e21] = -1
    trend = trend.shift(1)  # completed 15m only
    aligned = trend.reindex(dt, method="ffill").fillna(0).to_numpy(dtype=int)
    idx = pd.DatetimeIndex(dt).tz_convert(ET)
    mins = idx.hour * 60 + idx.minute
    dates = idx.date
    long_s = np.zeros(len(frame), dtype=bool)
    short_s = np.zeros(len(frame), dtype=bool)
    last = None
    taken = False
    for i, (d, m, t) in enumerate(zip(dates, mins, aligned)):
        if d != last:
            last, taken = d, False
        if taken or t == 0 or m < 9 * 60 + 35 or m >= 15 * 60:
            continue
        if t > 0:
            long_s[i] = True
            taken = True
        elif t < 0 and not long_only:
            short_s[i] = True
            taken = True
    return long_s, short_s


def score(name, frame, lo, sh, sl, tp):
    trades = edge.run_signals(frame, lo, sh, sl, tp)
    ins, oos = edge.split_is_oos(trades)
    is_s, oos_s = edge.summarize(ins), edge.summarize(oos)
    all_s = edge.summarize(trades)
    ok = (
        oos_s["trades"] >= 30
        and oos_s["profit_factor"] >= 1.15
        and oos_s["total_pnl"] > 0
        and is_s["profit_factor"] >= 1.05
        and is_s["total_pnl"] > 0
    )
    print(name)
    print(
        f"  IS  n={is_s['trades']} WR={is_s['win_rate']:.1f}% "
        f"PF={is_s['profit_factor']:.2f} ${is_s['total_pnl']:.1f} DD=${is_s['max_drawdown']:.1f}"
    )
    print(
        f"  OOS n={oos_s['trades']} WR={oos_s['win_rate']:.1f}% "
        f"PF={oos_s['profit_factor']:.2f} ${oos_s['total_pnl']:.1f} DD=${oos_s['max_drawdown']:.1f}"
    )
    print(f"  ALL n={all_s['trades']} PF={all_s['profit_factor']:.2f} ${all_s['total_pnl']:.1f}")
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
    return {"name": name, "is": is_s, "oos": oos_s, "all": all_s, "pass": ok}


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    raw_1m, raw_5m, _, _ = scalp_bt.load_data(csv, None, None, allow_synthetic_30s=True)
    df_1m = scalp_bt.add_1m_indicators(raw_1m)
    frame = scalp_bt.attach_context(
        df_1m, scalp_bt.add_5m_indicators(raw_5m), ["close", "vwap", "ema_20", "adx"], "_5m"
    )
    print(f"bars={len(frame):,}  {frame['datetime'].iloc[0]} -> {frame['datetime'].iloc[-1]}")
    print("PASS gates: OOS n>=30 PF>=1.15 $>0 | IS PF>=1.05 $>0\n")

    rows = []
    lo, sh = signals_15m_trend(frame, False)
    rows.append(score("ORIGINAL simple-search (ffill, may form bar): 15m EMA both EOD sl30/tp500", frame, lo, sh, 30, 500))
    print()
    lo, sh = signals_15m_completed(frame, False)
    rows.append(score("HONEST completed 15m EMA8/21 both 1/day EOD sl30/tp500", frame, lo, sh, 30, 500))
    print()
    for sl in (25, 35, 40, 50):
        lo, sh = signals_15m_completed(frame, False)
        rows.append(score(f"HONEST completed 15m EOD sl{sl}/tp500", frame, lo, sh, sl, 500))
        print()

    passed = [r for r in rows if r["pass"]]
    print(f"PASS count: {len(passed)}")
    for r in passed:
        print(f"  - {r['name']}")

    out = os.path.join("data", "pass_verified_15m_ema.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "recipe": "1 trade/day after 9:35 ET with 15m EMA8/21 direction, SL fixed, TP cap 500 / EOD",
                "results": rows,
                "passed": [r["name"] for r in passed],
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Saved {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
