#!/usr/bin/env python3
"""
Stage 2 — forward/paper validation tracker (FROZEN spec, no tuning).

Reads data/frozen_mnq_combined_spec.json for gates.
Runs combined backtest on latest MNQ_1m.csv and reports:
  - Development OOS (pre-forward) — historical reference only
  - Forward period (entries >= forward_start) — progress toward 30-50 trades
  - Predetermined PASS / BORDERLINE / FAIL / IN_PROGRESS

DO NOT modify strategy parameters from this script.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_combined_bidirectional as combo
import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_official as official
import scripts.backtest_scalp_momentum as scalp_bt

SPEC_PATH = os.path.join("data", "frozen_mnq_combined_spec.json")
DEFAULT_CSV = os.path.join("data", "MNQ_1m.csv")


def load_spec() -> Dict[str, Any]:
    with open(SPEC_PATH, encoding="utf-8") as f:
        return json.load(f)


def entry_ts(t: scalp_bt.ScalpTrade) -> pd.Timestamp:
    ts = pd.Timestamp(t.entry_time)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def filter_from(trades: List[scalp_bt.ScalpTrade], start: pd.Timestamp) -> List[scalp_bt.ScalpTrade]:
    return [t for t in trades if entry_ts(t) >= start]


def metrics(trades: List[scalp_bt.ScalpTrade]) -> Dict[str, Any]:
    if not trades:
        return {"trades": 0, "profit_factor": 0.0, "net_pnl": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}
    m = resp.group_metrics(trades)
    return {
        "trades": m["trades"],
        "win_rate": m["win_rate"],
        "profit_factor": m["profit_factor"],
        "net_pnl": round(m["total_pnl"], 2),
        "max_drawdown": round(m["max_drawdown"], 2),
        "expectancy": round(m["expectancy"], 2),
    }


def decide(fw: Dict[str, Any], m: Dict[str, Any], spec: Dict[str, Any]) -> str:
    n = m["trades"]
    pf = m["profit_factor"]
    net = m["net_pnl"]
    gates = spec["forward_validation"]
    min_g = gates["min_trades_good"]
    max_g = gates["max_trades_good"]
    ext = gates["extension_trades"]

    if n < min_g:
        return f"IN_PROGRESS — {n}/{min_g} forward trades (need {min_g}-{max_g} before decision)"

    good_pf = gates["good"]["pf_min"]
    if n >= min_g and pf >= good_pf and net > 0:
        if n <= max_g:
            return "GOOD — BUILD MQL5 (parity backtest then reduced-risk live)"
        return "GOOD — BUILD MQL5 (past 50 trades; metrics still pass)"

    if n >= ext:
        if pf < gates["fail"]["pf_below"]:
            return "FAIL — KILL strategy (75+ trades, PF < 1.0)"
        if gates["borderline"]["pf_min"] <= pf <= gates["borderline"]["pf_max"]:
            return "BORDERLINE at extension — human final decision"
        if pf >= good_pf and net > 0:
            return "GOOD at extension — BUILD MQL5"

    if gates["borderline"]["pf_min"] <= pf <= gates["borderline"]["pf_max"] and net > 0:
        if n < ext:
            return f"BORDERLINE — extend to {ext} trades (currently {n}, PF {pf:.2f})"
        return "BORDERLINE at extension — final decision"

    if pf < gates["fail"]["pf_below"]:
        return f"FAIL — KILL strategy (PF {pf:.2f} < 1.0 after {n} forward trades)"

    return f"IN_PROGRESS — {n} trades, PF {pf:.2f}, net ${net:.0f}"


def main() -> int:
    spec = load_spec()
    fw = spec["forward_validation"]
    forward_start = pd.Timestamp(fw["forward_start"], tz="UTC")
    csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV

    print("=" * 72)
    print("STAGE 2 — FORWARD VALIDATION (frozen spec, no changes)")
    print("=" * 72)
    print(f"Forward period: entries >= {forward_start.date()}")
    print(f"Gates: {fw['min_trades_good']}-{fw['max_trades_good']} trades | "
          f"GOOD PF>={fw['good']['pf_min']} | BORDERLINE {fw['borderline']['pf_min']}-"
          f"{fw['borderline']['pf_max']} | FAIL PF<{fw['fail']['pf_below']}")
    print()

    df = official.load_1m(csv)
    a = combo.prepare(df)
    trades = combo.simulate_combined_recommended(a)

    forward = filter_from(trades, forward_start)
    dev = [t for t in trades if entry_ts(t) < forward_start]

    fm = metrics(forward)
    dm = metrics(dev)

    print(f"Data: {csv} | bars={len(df):,} | end={df['datetime'].iloc[-1]}")
    print()
    print("Development (reference only, do not re-optimize):")
    print(f"  n={dm['trades']} PF={dm['profit_factor']:.2f} Net=${dm['net_pnl']:.0f} DD=${dm['max_drawdown']:.0f}")
    print()
    print("Forward period (decision sample):")
    print(f"  n={fm['trades']} PF={fm['profit_factor']:.2f} Net=${fm['net_pnl']:.0f} "
          f"WR={fm['win_rate']:.1f}% DD=${fm['max_drawdown']:.0f}")
    if fm["trades"]:
        longs = [t for t in forward if t.direction == "long"]
        shorts = [t for t in forward if t.direction == "short"]
        lm, sm = metrics(longs), metrics(shorts)
        print(f"  Long: n={lm['trades']} PF={lm['profit_factor']:.2f} ${lm['net_pnl']:.0f}")
        print(f"  Short: n={sm['trades']} PF={sm['profit_factor']:.2f} ${sm['net_pnl']:.0f}")

    verdict = decide(fw, fm, spec)
    print()
    print(f"VERDICT: {verdict}")

    out = {
        "as_of_data_end": str(df["datetime"].iloc[-1]),
        "development": dm,
        "forward": fm,
        "verdict": verdict,
        "forward_trade_count": fm["trades"],
        "target_min": fw["min_trades_good"],
        "target_max": fw["max_trades_good"],
    }
    out_path = os.path.join("data", "forward_validation_status.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
