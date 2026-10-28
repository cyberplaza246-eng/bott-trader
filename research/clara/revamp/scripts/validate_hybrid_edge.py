#!/usr/bin/env python3
"""
Prove (or kill) the locked MNQ hybrid recipe before live.

1) Ablation — TIME-exit lengths + synthetic 30s vs native 1m trigger
2) Walk-forward — same locked params on in-sample vs later dates
3) Paper replay — write a live-style journal (does not touch live trade_journal.jsonl)

Usage:
    python scripts/validate_hybrid_edge.py
    python scripts/validate_hybrid_edge.py --symbol MNQ
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_scalp_hybrid as hybrid
import scripts.backtest_scalp_momentum as scalp_bt
from src.strategy.scalp_hybrid import add_30s_body_stats, default_rsi_gate_cfg
from src.utils.trading_session import DEFAULT_SCALP_SESSIONS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LOCKED_SL = 6.0
LOCKED_TP = 14.0
LOCKED_HOLD = 30
PF_PASS = 1.2
PF_WEAK = 1.0
MIN_TRADES = 30


def _utc(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts, tz="UTC")


def prepare_frames(
    csv_1m: str,
    csv_5m: Optional[str],
    csv_30s: Optional[str] = None,
    allow_synthetic_30s: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    df_1m_raw, df_5m_raw, df_30s_raw, note = scalp_bt.load_data(
        csv_1m, csv_5m, csv_30s, allow_synthetic_30s=allow_synthetic_30s,
    )
    return (
        scalp_bt.add_1m_indicators(df_1m_raw),
        scalp_bt.add_5m_indicators(df_5m_raw),
        add_30s_body_stats(df_30s_raw),
        note,
    )


def run_locked(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_30s: pd.DataFrame,
    symbol: str,
    *,
    max_hold: int,
    trigger_sec: int,
) -> Dict[str, Any]:
    stats = hybrid.run_hybrid_variant(
        hybrid.HybridVariant("hybrid", True, True),
        df_1m, df_5m, df_30s, symbol,
        LOCKED_SL, LOCKED_TP, max_hold,
        hybrid.HybridParams.ultra_fast(rsi_gate_cfg=default_rsi_gate_cfg()),
        rth_windows=list(DEFAULT_SCALP_SESSIONS),
        trigger_bar_seconds=trigger_sec,
        return_trades=True,
    )
    return stats


def summarize_raw(trades: List[Any], version: str, symbol: str) -> Dict[str, Any]:
    return scalp_bt.summarize(trades, version, symbol)


def filter_trades(trades: List[Any], start: str, end: str) -> List[Any]:
    lo, hi = _utc(start), _utc(end)
    out = []
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        if lo <= ts < hi:
            out.append(t)
    return out


def grade(stats: Dict[str, Any]) -> str:
    if stats["trades"] < MIN_TRADES:
        return "INSUFFICIENT"
    if stats["profit_factor"] < PF_WEAK:
        return "FAIL"
    if stats["profit_factor"] < PF_PASS:
        return "WEAK"
    return "PASS"


def print_row(label: str, stats: Dict[str, Any]) -> None:
    print(
        f"{label:<28} {stats['trades']:>6} {stats['win_rate']:>6.1f} "
        f"{stats['profit_factor']:>6.2f} {stats['expectancy']:>8.2f} "
        f"{stats['total_pnl']:>10.2f} {stats['max_drawdown']:>8.2f} "
        f"{stats['avg_hold_sec']:>7.1f} {stats.get('max_hold_exits', 0):>5} "
        f"{stats.get('sl_exits', 0):>4} {stats.get('tp_exits', 0):>4}  {grade(stats)}"
    )


def write_paper_journal(trades: List[Any], path: str, symbol: str) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.isfile(path):
        os.remove(path)
    wins = 0
    pnl = 0.0
    with open(path, "w", encoding="utf-8") as f:
        for t in trades:
            rec = {
                "source": "paper_replay",
                "symbol": symbol,
                "direction": t.direction,
                "entry_mode": t.version,
                "entry_time": str(t.entry_time),
                "exit_time": str(t.exit_time),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "sl": t.sl,
                "tp": t.tp,
                "hold_seconds": t.hold_seconds,
                "pnl": round(t.pnl, 2),
                "exit_reason": t.exit_reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(rec, default=str) + "\n")
            pnl += t.pnl
            if t.pnl > 0:
                wins += 1
    n = len(trades)
    return {
        "path": path,
        "trades": n,
        "win_rate": round(100.0 * wins / n, 1) if n else 0.0,
        "total_pnl": round(pnl, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation + walk-forward + paper replay")
    parser.add_argument("--symbol", default="MNQ")
    parser.add_argument("--csv-1m", default=os.path.join(DATA, "MNQ_1m.csv"))
    parser.add_argument("--csv-5m", default=None)
    parser.add_argument("--csv-30s", default=None, help="Real 30s CSV (Databento 1s resample)")
    parser.add_argument(
        "--synthetic-30s",
        action="store_true",
        help="Allow inventing 30s from 1m (default: refuse)",
    )
    parser.add_argument("--skip-rithmic", action="store_true")
    parser.add_argument(
        "--rithmic-1m",
        default=os.path.join(DATA, "MNQ_1m_rithmic.csv"),
    )
    parser.add_argument(
        "--rithmic-5m",
        default=os.path.join(DATA, "MNQ_5m_rithmic.csv"),
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()
    print("\n=== 1) ABLATION (is the 30s TIME edge real?) ===")
    print(f"Symbol {symbol}  SL/TP {LOCKED_SL}/{LOCKED_TP}  RTH windows on")
    if not args.csv_30s and not args.synthetic_30s:
        for cand in (
            os.path.join(DATA, "MNQ_30s_databento.csv"),
            os.path.join(DATA, "MNQ_30s_august_databento.csv"),
        ):
            if os.path.isfile(cand):
                args.csv_30s = cand
                break
    df_1m, df_5m, df_30s, note = prepare_frames(
        args.csv_1m, args.csv_5m, args.csv_30s,
        allow_synthetic_30s=args.synthetic_30s,
    )
    print(f"Primary: {note}")
    print(
        f"{'Variant':<28} {'Trades':>6} {'WR%':>6} {'PF':>6} {'Exp$':>8} "
        f"{'PnL$':>10} {'MaxDD$':>8} {'Hold s':>7} {'TIME':>5} {'SL':>4} {'TP':>4}  Grade"
    )
    print("-" * 118)

    ablation_specs = (
        ("30s trigger + TIME 30s", 30, 30),
        ("30s trigger + TIME 60s", 30, 60),
        ("30s trigger + TIME 90s", 30, 90),
        ("30s trigger + SL/TP only", 30, 0),
        ("1m trigger + TIME 30s", 60, 30),
        ("1m trigger + SL/TP only", 60, 0),
    )
    ablation_rows: List[Dict[str, Any]] = []
    locked_trades: List[Any] = []
    for label, trig, hold in ablation_specs:
        print(f"  running {label}...", flush=True)
        stats = run_locked(df_1m, df_5m, df_30s, symbol, max_hold=hold, trigger_sec=trig)
        raw = stats.pop("trades_raw", [])
        if label.startswith("30s trigger + TIME 30s"):
            locked_trades = raw
        row = {"label": label, "trigger_sec": trig, "max_hold": hold, **stats, "grade": grade(stats)}
        ablation_rows.append(row)
        print_row(label, stats)

    locked = next(r for r in ablation_rows if r["label"].startswith("30s trigger + TIME 30s"))
    sltp = next(r for r in ablation_rows if r["label"] == "30s trigger + SL/TP only")
    one_min = next(r for r in ablation_rows if r["label"] == "1m trigger + SL/TP only")
    if locked["grade"] == "PASS" and one_min["grade"] in ("FAIL", "WEAK"):
        ablation_verdict = (
            "TRIGGER RISK — native 1-minute entries lose. The locked PF lives on 30s bars "
            "(synthetic in this file). SL/TP-only on those 30s still passed, so TIME exit is "
            "not the whole story — live is only trustworthy if Rithmic SECOND_BAR behaves like this."
        )
    elif locked["grade"] == "PASS" and sltp["grade"] in ("FAIL", "WEAK", "INSUFFICIENT"):
        ablation_verdict = (
            "TIME-EXIT RISK — locked 30s TIME looks good, SL/TP-only does not. "
            "Do not treat live 30s TIME as proven."
        )
    elif locked["grade"] == "PASS":
        ablation_verdict = "EDGE HOLDS on 30s and 1m. Still paper before live."
    else:
        ablation_verdict = "LOCKED RECIPE DID NOT PASS ablation. Do not go live on this setup."
    print(f"\nAblation verdict: {ablation_verdict}")

    print("\n=== 2) WALK-FORWARD (locked 30s + TIME 30s, RTH) ===")
    hdr = f"{'Fold':<28} {'Trades':>6} {'WR%':>6} {'PF':>6} {'Exp$':>8} {'PnL$':>10} {'MaxDD$':>8}  Grade"
    print(hdr)
    print("-" * len(hdr))
    wf_rows: List[Dict[str, Any]] = []

    folds = (
        ("IS Dec–Feb (primary)", "2025-12-29", "2026-03-01"),
        ("OOS Mar 1–11 (primary)", "2026-03-01", "2026-03-12"),
    )
    for label, start, end in folds:
        chunk = filter_trades(locked_trades, start, end)
        stats = summarize_raw(chunk, label, symbol)
        wf_rows.append({"label": label, "start": start, "end": end, **stats, "grade": grade(stats)})
        print_row(label, stats)

    rithmic_1m = args.rithmic_1m
    if (not args.skip_rithmic) and os.path.isfile(rithmic_1m):
        print("  running locked recipe on Rithmic CSV...", flush=True)
        r1, r5, r30, rnote = prepare_frames(rithmic_1m, args.rithmic_5m)
        print(f"Rithmic: {rnote}")
        rstats = run_locked(r1, r5, r30, symbol, max_hold=LOCKED_HOLD, trigger_sec=30)
        rtrades = rstats.pop("trades_raw", [])
        r_folds = (
            ("Rithmic May 1–20", "2026-05-01", "2026-05-20"),
            ("OOS Rithmic May 20–Jun 4", "2026-05-20", "2026-06-04"),
        )
        for label, start, end in r_folds:
            chunk = filter_trades(rtrades, start, end)
            stats = summarize_raw(chunk, label, symbol)
            wf_rows.append({"label": label, "start": start, "end": end, **stats, "grade": grade(stats)})
            print_row(label, stats)
    else:
        print("  skip Rithmic CSV (file missing)")

    oos = [r for r in wf_rows if r["label"].startswith("OOS")]
    if oos and all(r["grade"] == "PASS" for r in oos):
        wf_verdict = "OOS folds PASS. Recipe survives a later window."
    elif oos and any(r["grade"] == "FAIL" for r in oos):
        wf_verdict = "OOS fold FAILED. Do not size up — retune or stay paper."
    else:
        wf_verdict = "OOS mixed or thin. Need more history before calling it robust."
    print(f"\nWalk-forward verdict: {wf_verdict}")

    print("\n=== 3) PAPER REPLAY (locked recipe, primary CSV) ===")
    journal_path = os.path.join(DATA, "paper_trade_journal.jsonl")
    journal = write_paper_journal(locked_trades, journal_path, symbol)
    paper_stats = summarize_raw(locked_trades, "paper_replay", symbol)
    print_row("paper replay (full primary)", paper_stats)
    print(f"Journal: {journal_path}  ({journal['trades']} rows)")
    print(
        "This is the same gate/exit path as live hybrid, on historical bars — "
        "not a Rithmic fill."
    )

    payload = {
        "symbol": symbol,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked": {"sl": LOCKED_SL, "tp": LOCKED_TP, "max_hold": LOCKED_HOLD, "rth": True},
        "ablation": ablation_rows,
        "ablation_verdict": ablation_verdict,
        "walk_forward": wf_rows,
        "walk_forward_verdict": wf_verdict,
        "paper_journal": journal,
        "paper": {k: paper_stats[k] for k in paper_stats if k != "trades_raw"},
    }
    out = os.path.join(DATA, "hybrid_edge_validation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
