#!/usr/bin/env python3
"""
Bidirectional locked engine: long AND short allowed any RTH time.

No time-direction lock. Direction comes from 15/5/1 EMA stack alignment only.
Uses proven lower-TF mechanics (not dynamic structure engine).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_official as official
import scripts.find_real_1m_edge as edge
import scripts.pass_hunt_shorts as hunt


def report(label: str, trades, cfg) -> dict:
    ins, oos = edge.split_is_oos(trades)
    im, om = resp.group_metrics(ins), resp.group_metrics(oos)
    am = resp.group_metrics(trades)
    longs = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]
    lm, sm = resp.group_metrics(longs), resp.group_metrics(shorts)
    ok = hunt.is_pass(im, om)
    flag = "PASS" if ok else "FAIL"
    print(f"\n{label}  [{flag}]")
    print(f"  ALL n={am['trades']} PF={am['profit_factor']:.2f} ${am['total_pnl']:.0f}")
    print(
        f"  Long n={lm['trades']} PF={lm['profit_factor']:.2f} ${lm['total_pnl']:.0f} | "
        f"Short n={sm['trades']} PF={sm['profit_factor']:.2f} ${sm['total_pnl']:.0f}"
    )
    print(
        f"  OOS n={om['trades']} PF={om['profit_factor']:.2f} ${om['total_pnl']:.0f} "
        f"WR={om['win_rate']:.1f}% Exp=${om['expectancy']:.2f}"
    )
    print(f"  IS  n={im['trades']} PF={im['profit_factor']:.2f} ${im['total_pnl']:.0f}")
    return {
        "label": label,
        "config": cfg,
        "overall": am,
        "is": im,
        "oos": om,
        "long": lm,
        "short": sm,
        "pass": ok,
    }


def main() -> int:
    csv = os.path.join("data", "MNQ_1m.csv")
    out = os.path.join("data", "bidirectional_locked_backtest.json")

    print("Bidirectional locked engine — long & short any RTH time (no direction clock)")
    df = official.load_1m(csv)
    a = hunt.prepare(df)

    configs = [
        (
            "All 6 windows | 15/5/1 | quiet 1m | 1R both | no VWAP",
            {
                "stack": "15_5_1",
                "loop": "window_loop",
                "entry": "quiet",
                "win": "w6_all",
                "stop": "atr1_1.5_6_25",
                "tp": "1R",
                "side": "both",
                "sep": False,
                "cooldown": 0,
                "max_day": 6,
                "max_lots": 1,
                "morning": False,
                "vwap": "none",
                "sep_short_min": None,
            },
        ),
        (
            "All 6 windows | 15/5/1 | quiet 1m | 1R | VWAP with side",
            {
                "stack": "15_5_1",
                "loop": "window_loop",
                "entry": "quiet",
                "win": "w6_all",
                "stop": "atr1_1.5_6_25",
                "tp": "1R",
                "side": "both",
                "sep": False,
                "cooldown": 0,
                "max_day": 6,
                "max_lots": 1,
                "morning": False,
                "vwap": "with_side",
                "sep_short_min": None,
            },
        ),
        (
            "Morning windows only (9:35+11:00) | both directions",
            {
                **hunt.LONG_WINNER,
                "side": "both",
            },
        ),
    ]

    results = [report(label, hunt.simulate_ext(a, cfg), cfg) for label, cfg in configs]

    with open(out, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, default=str)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
