#!/usr/bin/env python
"""
Multiple-testing / selection-bias audit for the walk-forward parameter
search: re-runs every grid combo on every fold (not just the winner) and
compares the OOS P&L of the actually-selected ("best in-sample") combo
against the AVERAGE OOS P&L across all combos in that fold's grid.

If the optimizer is finding real structure, the selected combo should
consistently beat the naive average by a meaningful margin. If it's mostly
picking the luckiest in-sample draw, the selected combo's OOS performance
should look similar to (or not reliably better than) just picking any combo
at random -- i.e. the walk-forward "edge" would be a selection-bias artifact,
not evidence the strategy has real predictive structure.

Usage:
    python scripts/audit_selection_bias.py --symbol MNQ --strategy vwap_pullback_trend
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import DEFAULT_ACCOUNT_SIZE, DEFAULT_RISK_PCT, DEFAULT_TIMEFRAME  # noqa: E402
from src.backtest.engine import run_backtest  # noqa: E402
from src.backtest.metrics import compute_metrics  # noqa: E402
from src.data.loader import load_ohlcv  # noqa: E402
from src.strategies.breakout import BreakoutStrategy  # noqa: E402
from src.strategies.mean_reversion import MeanReversionStrategy  # noqa: E402
from src.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy  # noqa: E402
from src.strategies.trend_following import TrendFollowingStrategy  # noqa: E402
from src.strategies.vwap_ema_cross import VwapEmaCrossStrategy  # noqa: E402
from src.strategies.vwap_pullback_trend import VwapPullbackTrendStrategy  # noqa: E402

STRATEGY_GRIDS = {
    "trend": (TrendFollowingStrategy, {"fast": [10, 20], "slow": [40, 50], "atr_stop_mult": [1.5, 2.5]}),
    "mean_reversion": (MeanReversionStrategy, {"bb_std": [1.5, 2.0, 2.5], "rsi_oversold": [25, 30], "adx_range_max": [15, 25]}),
    "breakout": (BreakoutStrategy, {"donchian_period": [10, 20, 30], "volume_mult": [1.2, 1.5, 2.0]}),
    "vwap_cross": (VwapEmaCrossStrategy, {"confirm_bars": [1, 3, 5], "atr_stop_buffer_mult": [0.25, 0.5, 1.0]}),
    "orb": (OpeningRangeBreakoutStrategy, {"or_minutes": [15, 30], "entry_window_minutes": [60, 120], "volume_mult": [1.0, 1.3]}),
    "vwap_pullback_trend": (VwapPullbackTrendStrategy, {"pullback_lookback_bars": [3, 5, 8], "min_body_atr_mult": [0.2, 0.3, 0.5]}),
}


def _combos(grid):
    keys = list(grid)
    for values in itertools.product(*grid.values()):
        yield dict(zip(keys, values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--strategy", required=True, choices=list(STRATEGY_GRIDS))
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--train-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--account-size", type=float, default=DEFAULT_ACCOUNT_SIZE)
    parser.add_argument("--risk-pct", type=float, default=DEFAULT_RISK_PCT)
    parser.add_argument("--min-train-trades", type=int, default=10)
    args = parser.parse_args()

    strategy_cls, grid = STRATEGY_GRIDS[args.strategy]
    df = load_ohlcv(args.symbol, args.timeframe)

    fold_reports = []
    train_start = df.index[0]
    end = df.index[-1]
    fold = 0

    while True:
        train_end = train_start + pd.Timedelta(days=args.train_days)
        test_end = train_end + pd.Timedelta(days=args.test_days)
        if test_end > end:
            break

        train_df = df[(df.index >= train_start) & (df.index < train_end)]
        test_df = df[(df.index >= train_end) & (df.index < test_end)]

        combo_results = []
        for params in _combos(grid):
            strat = strategy_cls(**params)
            train_res = run_backtest(train_df, strat, args.symbol, args.timeframe, args.account_size, args.risk_pct)
            train_m = compute_metrics(train_res, args.account_size)
            if train_m["trade_count"] < args.min_train_trades:
                continue
            test_res = run_backtest(test_df, strat, args.symbol, args.timeframe, args.account_size, args.risk_pct)
            test_m = compute_metrics(test_res, args.account_size)
            combo_results.append({
                "params": params, "train_pnl": train_m["total_pnl"], "train_trades": train_m["trade_count"],
                "test_pnl": test_m["total_pnl"], "test_trades": test_m["trade_count"],
            })

        if combo_results:
            selected = max(combo_results, key=lambda r: r["train_pnl"])
            avg_test_pnl = sum(r["test_pnl"] for r in combo_results) / len(combo_results)
            median_test_pnl = sorted(r["test_pnl"] for r in combo_results)[len(combo_results) // 2]
            best_possible_test_pnl = max(r["test_pnl"] for r in combo_results)
            worst_test_pnl = min(r["test_pnl"] for r in combo_results)

            fold_reports.append({
                "fold": fold, "n_eligible_combos": len(combo_results),
                "selected_params": selected["params"], "selected_train_pnl": round(selected["train_pnl"], 2),
                "selected_test_pnl": round(selected["test_pnl"], 2),
                "avg_test_pnl_all_combos": round(avg_test_pnl, 2),
                "median_test_pnl_all_combos": round(median_test_pnl, 2),
                "best_possible_test_pnl": round(best_possible_test_pnl, 2),
                "worst_test_pnl_all_combos": round(worst_test_pnl, 2),
            })

        fold += 1
        train_start = train_start + pd.Timedelta(days=args.test_days)

    total_selected = sum(f["selected_test_pnl"] for f in fold_reports)
    total_avg_naive = sum(f["avg_test_pnl_all_combos"] for f in fold_reports)
    total_median_naive = sum(f["median_test_pnl_all_combos"] for f in fold_reports)
    total_best_possible = sum(f["best_possible_test_pnl"] for f in fold_reports)

    summary = {
        "symbol": args.symbol, "strategy": args.strategy, "folds": len(fold_reports),
        "total_oos_pnl_selected_by_optimizer": round(total_selected, 2),
        "total_oos_pnl_naive_average_combo": round(total_avg_naive, 2),
        "total_oos_pnl_naive_median_combo": round(total_median_naive, 2),
        "total_oos_pnl_best_possible_in_hindsight": round(total_best_possible, 2),
        "optimizer_beat_naive_average_by": round(total_selected - total_avg_naive, 2),
        "interpretation": (
            "Optimizer meaningfully beats naive average -> some real signal in the parameter choice"
            if total_selected > total_avg_naive * 1.5 and total_selected > 0
            else "Optimizer's edge over naive average is small/absent -> consistent with selection-bias "
                 "noise harvesting, not real predictive structure"
        ),
    }

    print(json.dumps(summary, indent=2))
    print("\nPer-fold detail:")
    for f in fold_reports:
        print(json.dumps(f))


if __name__ == "__main__":
    main()
