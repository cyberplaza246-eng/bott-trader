#!/usr/bin/env python3
"""Analyze a trade-loss window and optionally tune adaptive + TP/SL settings.

Primary workflow:
1. Load trade_history from data/adaptive_learning.json
2. Filter by UTC date window (inclusive)
3. Produce markdown + CSV reports under logs/
4. Optionally replay filtered trades into AdaptiveLearner
5. Optionally apply conservative SL/TP overrides

Examples:
  python scripts/analyze_loss_window_and_tune.py \
    --start-date 2026-03-05 --end-date 2026-03-06

  python scripts/analyze_loss_window_and_tune.py \
    --start-date 2026-03-05 --end-date 2026-03-06 \
    --replay-adaptive --apply
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone, date
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

LEARNING_PATH = "data/adaptive_learning.json"
RISK_OVERRIDES_PATH = "data/risk_overrides.json"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_iso_timestamp(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def parse_date(raw: str) -> date:
    return datetime.strptime(raw, "%Y-%m-%d").date()


def load_learning_data(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_trades_by_utc_dates(
    trades: List[Dict[str, Any]], start_d: date, end_d: date
) -> Tuple[List[Dict[str, Any]], int]:
    filtered: List[Dict[str, Any]] = []
    missing_ts = 0

    for trade in trades:
        dt = parse_iso_timestamp(trade.get("timestamp"))
        if dt is None:
            missing_ts += 1
            continue

        d = dt.date()
        if start_d <= d <= end_d:
            normalized = dict(trade)
            normalized["timestamp"] = dt.isoformat()
            filtered.append(normalized)

    filtered.sort(key=lambda t: t.get("timestamp", ""))
    return filtered, missing_ts


def max_consecutive_losses(trades: List[Dict[str, Any]]) -> int:
    best = 0
    cur = 0
    for t in trades:
        pnl = float(t.get("profit_loss", 0) or 0)
        if pnl < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def summarize_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    pnls = [float(t.get("profit_loss", 0) or 0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    by_pair = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0})
    by_session = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0})
    by_hour = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0})
    by_regime = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0})
    by_signal = Counter()

    for t in trades:
        pnl = float(t.get("profit_loss", 0) or 0)
        pair = str(t.get("pair", "UNKNOWN"))
        session = str(t.get("session", "unknown"))
        regime = str(t.get("regime", "unknown"))

        h = t.get("hour")
        try:
            hour_key = str(int(h))
        except Exception:
            dt = parse_iso_timestamp(t.get("timestamp"))
            hour_key = str(dt.hour if dt else "unknown")

        sig = str(t.get("signal", "UNKNOWN"))
        is_win = pnl > 0
        is_loss = pnl < 0

        by_signal[sig] += 1

        for bucket, key in (
            (by_pair, pair),
            (by_session, session),
            (by_hour, hour_key),
            (by_regime, regime),
        ):
            bucket[key]["trades"] += 1
            bucket[key]["wins"] += 1 if is_win else 0
            bucket[key]["losses"] += 1 if is_loss else 0
            bucket[key]["net_pnl"] += pnl

    largest_loss = min(losses) if losses else 0.0
    fixed_loss_profile = False
    if losses:
        rounded = [round(x, 5) for x in losses]
        fixed_loss_profile = len(set(rounded)) <= 2

    out = {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len([p for p in pnls if p == 0]),
        "win_rate": (len(wins) / len(trades)) if trades else 0.0,
        "net_pnl": sum(pnls),
        "avg_win": mean(wins) if wins else 0.0,
        "avg_loss": mean(losses) if losses else 0.0,
        "largest_loss": largest_loss,
        "max_consecutive_losses": max_consecutive_losses(trades),
        "by_pair": dict(by_pair),
        "by_session": dict(by_session),
        "by_hour": dict(by_hour),
        "by_regime": dict(by_regime),
        "signal_bias": dict(by_signal),
        "fixed_loss_profile": fixed_loss_profile,
    }
    return out


def safe_wr(bucket: Dict[str, Any]) -> float:
    total = int(bucket.get("wins", 0)) + int(bucket.get("losses", 0))
    if total <= 0:
        return 0.0
    return float(bucket.get("wins", 0)) / total


def suggest_tuning(summary: Dict[str, Any], existing_learning: Dict[str, Any]) -> Dict[str, Any]:
    pair_stats = summary["by_pair"]
    existing_sl = existing_learning.get("sl_multiplier_by_pair", {})

    sl_updates: Dict[str, float] = {}
    for pair, stats in pair_stats.items():
        wr = safe_wr(stats)
        losses = int(stats.get("losses", 0))
        current = float(existing_sl.get(pair, 0.8))

        if losses >= 10 and wr < 0.35:
            # Widen SL for severe stop-out clusters.
            proposed = min(1.2, current + 0.10)
        elif wr > 0.60 and int(stats.get("wins", 0)) >= 10:
            proposed = max(0.6, current - 0.05)
        else:
            proposed = current

        sl_updates[pair] = round(proposed, 3)

    # Conservative TP map: lower ambition when loss window is severe.
    global_wr = float(summary["win_rate"])
    fixed_loss = bool(summary["fixed_loss_profile"])

    if global_wr < 0.30 or fixed_loss:
        tp = {
            "tp_base_ratio": 1.15,
            "tp_expanding": 1.35,
            "tp_contracting": 1.05,
            "tp_asian_session": 1.05,
            "tp_london_open": 1.25,
            "tp_ny_overlap": 1.30,
            "tp_quiet_hours": 1.00,
        }
    elif global_wr < 0.45:
        tp = {
            "tp_base_ratio": 1.20,
            "tp_expanding": 1.45,
            "tp_contracting": 1.10,
            "tp_asian_session": 1.10,
            "tp_london_open": 1.35,
            "tp_ny_overlap": 1.40,
            "tp_quiet_hours": 1.05,
        }
    else:
        tp = {
            "tp_base_ratio": 1.30,
            "tp_expanding": 1.50,
            "tp_contracting": 1.20,
            "tp_asian_session": 1.20,
            "tp_london_open": 1.40,
            "tp_ny_overlap": 1.50,
            "tp_quiet_hours": 1.10,
        }

    return {
        "sl_multiplier_by_pair": sl_updates,
        "tp_overrides": tp,
        "reason": "Conservative profile selected from date-window loss concentration",
    }


def write_markdown_report(
    path: str,
    start_date: str,
    end_date: str,
    summary: Dict[str, Any],
    tuning: Dict[str, Any],
    missing_ts: int,
) -> None:
    lines: List[str] = []
    lines.append(f"# Loss Window Analysis ({start_date} to {end_date}, UTC)")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Trades: {summary['trades']}")
    lines.append(f"- Wins: {summary['wins']}")
    lines.append(f"- Losses: {summary['losses']}")
    lines.append(f"- Breakeven: {summary['breakeven']}")
    lines.append(f"- Win Rate: {summary['win_rate'] * 100:.1f}%")
    lines.append(f"- Net PnL: {summary['net_pnl']:+.2f}")
    lines.append(f"- Avg Win: {summary['avg_win']:+.2f}")
    lines.append(f"- Avg Loss: {summary['avg_loss']:+.2f}")
    lines.append(f"- Largest Loss: {summary['largest_loss']:+.2f}")
    lines.append(f"- Max Consecutive Losses: {summary['max_consecutive_losses']}")
    lines.append(f"- Missing/invalid timestamps skipped: {missing_ts}")
    lines.append("")

    lines.append("## By Pair")
    for pair, stats in sorted(summary["by_pair"].items()):
        wr = safe_wr(stats) * 100
        lines.append(
            f"- {pair}: trades={stats['trades']}, WR={wr:.1f}%, "
            f"net={stats['net_pnl']:+.2f}"
        )
    lines.append("")

    lines.append("## By Session")
    for session, stats in sorted(summary["by_session"].items()):
        wr = safe_wr(stats) * 100
        lines.append(
            f"- {session}: trades={stats['trades']}, WR={wr:.1f}%, "
            f"net={stats['net_pnl']:+.2f}"
        )
    lines.append("")

    lines.append("## Signal Bias")
    for signal, count in sorted(summary["signal_bias"].items()):
        lines.append(f"- {signal}: {count}")
    lines.append("")

    lines.append("## Tuning Recommendations")
    lines.append(f"- Reason: {tuning['reason']}")
    for pair, mult in sorted(tuning["sl_multiplier_by_pair"].items()):
        lines.append(f"- SL multiplier {pair}: {mult:.3f}x ATR")
    lines.append("- TP overrides:")
    for key, value in sorted(tuning["tp_overrides"].items()):
        lines.append(f"  - {key}: {value:.3f}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_csv_report(path: str, summary: Dict[str, Any]) -> None:
    rows: List[List[Any]] = []
    rows.append(["scope", "key", "trades", "wins", "losses", "win_rate", "net_pnl"])

    for scope in ("by_pair", "by_session", "by_hour", "by_regime"):
        for key, stats in sorted(summary[scope].items()):
            wr = safe_wr(stats)
            rows.append(
                [scope, key, stats["trades"], stats["wins"], stats["losses"], round(wr, 6), round(stats["net_pnl"], 6)]
            )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def replay_into_adaptive(
    trades: List[Dict[str, Any]],
    tuning: Dict[str, Any],
    apply_changes: bool,
) -> Dict[str, Any]:
    from src.ai.adaptive_learner import AdaptiveLearner
    from src.utils.logger import bot_logger

    learner = AdaptiveLearner()
    baseline = learner.get_performance_summary()

    original_save = learner._save
    original_logger_disabled = bot_logger.disabled
    learner._save = lambda: None
    bot_logger.disabled = True
    try:
        for t in trades:
            replay_trade = {
                "pair": t.get("pair", "UNKNOWN"),
                "profit_loss": float(t.get("profit_loss", 0) or 0),
                "model_signals": deepcopy(t.get("model_signals", {})),
                "signal": t.get("signal", "UNKNOWN"),
                "regime": t.get("regime", learner.current_regime),
                "timestamp": t.get("timestamp"),
                "session": t.get("session"),
                "hour": t.get("hour"),
            }
            learner.record_trade(replay_trade)
    finally:
        learner._save = original_save
        bot_logger.disabled = original_logger_disabled

    # Apply recommended SL multipliers only in apply mode.
    if apply_changes:
        learner.sl_multiplier_by_pair.update(tuning["sl_multiplier_by_pair"])
        learner._save()

    updated = learner.get_performance_summary()
    return {
        "baseline": baseline,
        "updated": updated,
    }


def apply_risk_overrides(path: str, tp_overrides: Dict[str, float], meta: Dict[str, Any]) -> None:
    payload = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            payload = {}

    payload.update(tp_overrides)
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["source_window_utc"] = {
        "start_date": meta["start_date"],
        "end_date": meta["end_date"],
        "trade_count": meta["trade_count"],
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze losses by date window and tune TP/SL")
    parser.add_argument("--start-date", required=True, help="UTC start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="UTC end date (YYYY-MM-DD)")
    parser.add_argument("--learning-path", default=LEARNING_PATH, help="Path to adaptive_learning.json")
    parser.add_argument("--output-dir", default="logs", help="Directory for analysis outputs")
    parser.add_argument("--min-trades", type=int, default=20, help="Minimum trades required")
    parser.add_argument("--replay-adaptive", action="store_true", help="Replay filtered trades into AdaptiveLearner")
    parser.add_argument("--apply", action="store_true", help="Apply tuned SL/TP settings")

    args = parser.parse_args()

    start_d = parse_date(args.start_date)
    end_d = parse_date(args.end_date)
    if end_d < start_d:
        raise ValueError("end-date must be on or after start-date")

    data = load_learning_data(args.learning_path)
    trades = data.get("trade_history", [])
    filtered, missing_ts = filter_trades_by_utc_dates(trades, start_d, end_d)

    if len(filtered) < args.min_trades:
        print(
            f"Insufficient trades in window: {len(filtered)} < {args.min_trades}. "
            "No tuning applied."
        )
        return 2

    summary = summarize_trades(filtered)
    tuning = suggest_tuning(summary, data)

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"loss_window_{args.start_date}_{args.end_date}_{stamp}"
    md_path = os.path.join(args.output_dir, f"{base}.md")
    csv_path = os.path.join(args.output_dir, f"{base}.csv")

    write_markdown_report(md_path, args.start_date, args.end_date, summary, tuning, missing_ts)
    write_csv_report(csv_path, summary)

    replay_stats = None
    if args.replay_adaptive:
        replay_stats = replay_into_adaptive(filtered, tuning, apply_changes=args.apply)

    if args.apply:
        apply_risk_overrides(
            RISK_OVERRIDES_PATH,
            tuning["tp_overrides"],
            {
                "start_date": args.start_date,
                "end_date": args.end_date,
                "trade_count": len(filtered),
            },
        )

    print("Analysis complete")
    print(f"- Window trades: {summary['trades']}")
    print(f"- Win rate: {summary['win_rate'] * 100:.1f}%")
    print(f"- Net PnL: {summary['net_pnl']:+.2f}")
    print(f"- Report (md): {md_path}")
    print(f"- Report (csv): {csv_path}")
    print(f"- Suggested SL updates: {tuning['sl_multiplier_by_pair']}")
    print(f"- Suggested TP overrides: {tuning['tp_overrides']}")

    if args.replay_adaptive:
        b = replay_stats["baseline"]
        u = replay_stats["updated"]
        print(
            "- Adaptive replay stats: "
            f"before_trades={b['total_trades']} after_trades={u['total_trades']} "
            f"before_wr={b['win_rate'] * 100:.1f}% after_wr={u['win_rate'] * 100:.1f}%"
        )

    if args.apply:
        print(f"- Applied TP overrides file: {RISK_OVERRIDES_PATH}")
        print(f"- Adaptive state updated: {args.learning_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
