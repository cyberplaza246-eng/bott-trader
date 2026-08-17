#!/usr/bin/env python
"""
Walk-forward parameter search across strategies and symbols. For each
strategy: re-optimize on a rolling train window, test the chosen params on
the following out-of-sample window, and aggregate only the OOS results —
that aggregate is the honest estimate of what the strategy would actually
have delivered, not the in-sample-fit numbers.

Usage:
    python scripts/walk_forward_search.py --symbol MNQ --strategy trend
    python scripts/walk_forward_search.py --symbol MNQ MES NQ --strategy all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DEFAULT_ACCOUNT_SIZE, DEFAULT_RISK_PCT, DEFAULT_TIMEFRAME, REPORTS_DIR, SUPPORTED_SYMBOLS  # noqa: E402
from src.backtest.walk_forward import aggregate_oos, walk_forward_search  # noqa: E402
from src.data.loader import load_ohlcv  # noqa: E402
from src.regime.allocator import RegimeAllocatorStrategy  # noqa: E402
from src.strategies.breakout import BreakoutStrategy  # noqa: E402
from src.strategies.ensemble import EnsembleStrategy  # noqa: E402
from src.strategies.extreme_displacement_reversion import ExtremeDisplacementReversionStrategy  # noqa: E402
from src.strategies.mean_reversion import MeanReversionStrategy  # noqa: E402
from src.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy  # noqa: E402
from src.strategies.orb_failure import OrbFailureStrategy  # noqa: E402
from src.strategies.trend_following import TrendFollowingStrategy  # noqa: E402
from src.strategies.vol_expansion_momentum import VolExpansionMomentumStrategy  # noqa: E402
from src.strategies.volume_shock_continuation import VolumeShockContinuationStrategy  # noqa: E402
from src.strategies.vwap_ema_cross import VwapEmaCrossStrategy  # noqa: E402
from src.strategies.vwap_pullback_trend import VwapPullbackTrendStrategy  # noqa: E402
from src.strategies.vwap_pullback_trend_v2 import VwapPullbackTrendV2Strategy  # noqa: E402

# Kept intentionally small — this is a grid search re-run per fold, so combo
# count multiplies directly into runtime.
STRATEGY_GRIDS = {
    "trend": (TrendFollowingStrategy, {
        "fast": [10, 20], "slow": [40, 50], "atr_stop_mult": [1.5, 2.5],
    }),
    "mean_reversion": (MeanReversionStrategy, {
        "bb_std": [1.5, 2.0, 2.5], "rsi_oversold": [25, 30], "adx_range_max": [15, 25],
    }),
    "breakout": (BreakoutStrategy, {
        "donchian_period": [10, 20, 30], "volume_mult": [1.2, 1.5, 2.0],
    }),
    "vwap_cross": (VwapEmaCrossStrategy, {
        "confirm_bars": [1, 3, 5], "atr_stop_buffer_mult": [0.25, 0.5, 1.0],
    }),
    "ensemble": (EnsembleStrategy, {}),
    "orb": (OpeningRangeBreakoutStrategy, {
        "or_minutes": [15, 30], "entry_window_minutes": [60, 120], "volume_mult": [1.0, 1.3],
    }),
    "orb_failure": (OrbFailureStrategy, {
        "or_minutes": [15, 30], "failure_bars": [3, 5],
    }),
    "vwap_pullback_trend": (VwapPullbackTrendStrategy, {
        "pullback_lookback_bars": [3, 5, 8], "min_body_atr_mult": [0.2, 0.3, 0.5],
    }),
    "vwap_pullback_trend_v2": (VwapPullbackTrendV2Strategy, {
        # Only min_adx is swept here -- pullback_lookback_bars/min_body_atr_mult
        # are held at the audit-validated values (not re-optimized), so this
        # doesn't reintroduce the overfitting risk of a fresh 3-parameter search.
        "min_adx": [15.0, 20.0, 25.0],
    }),
    "regime_bot": (RegimeAllocatorStrategy, {}),
    "vol_expansion_momentum": (VolExpansionMomentumStrategy, {
        "atr_expansion_mult": [1.3, 1.8], "min_body_range_pct": [0.5, 0.7],
    }),
    "extreme_displacement_reversion": (ExtremeDisplacementReversionStrategy, {
        "z_threshold": [2.0, 3.0], "displacement_bars": [3, 8],
    }),
    "volume_shock_continuation": (VolumeShockContinuationStrategy, {
        "volume_shock_mult": [2.0, 4.0], "volume_baseline_period": [10, 20],
    }),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", nargs="+", default=SUPPORTED_SYMBOLS, choices=SUPPORTED_SYMBOLS)
    parser.add_argument("--strategy", nargs="+", default=list(STRATEGY_GRIDS),
                         choices=list(STRATEGY_GRIDS) + ["all"])
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--train-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--account-size", type=float, default=DEFAULT_ACCOUNT_SIZE)
    parser.add_argument("--risk-pct", type=float, default=DEFAULT_RISK_PCT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategies = list(STRATEGY_GRIDS) if "all" in args.strategy else args.strategy

    reports_dir = Path(__file__).resolve().parent.parent / REPORTS_DIR / "walk_forward"
    reports_dir.mkdir(parents=True, exist_ok=True)

    leaderboard = []

    for symbol in args.symbol:
        df = load_ohlcv(symbol, args.timeframe)
        for strategy_name in strategies:
            strategy_cls, param_grid = STRATEGY_GRIDS[strategy_name]
            print(f"\n=== {symbol} / {strategy_name} ===", flush=True)

            folds = walk_forward_search(
                df=df, strategy_cls=strategy_cls, param_grid=param_grid,
                symbol=symbol, timeframe=args.timeframe,
                account_size=args.account_size, risk_pct=args.risk_pct,
                train_days=args.train_days, test_days=args.test_days,
            )
            oos = aggregate_oos(folds)
            print(f"  OOS: {json.dumps(oos)}", flush=True)

            fold_dump = [
                {
                    "fold": f.fold,
                    "train_start": str(f.train_start), "train_end": str(f.train_end),
                    "test_start": str(f.test_start), "test_end": str(f.test_end),
                    "best_params": f.best_params,
                    "train_metrics": f.train_metrics,
                    "test_metrics": f.test_metrics,
                }
                for f in folds
            ]
            out_path = reports_dir / f"{symbol}_{strategy_name}.json"
            out_path.write_text(json.dumps({"symbol": symbol, "strategy": strategy_name, "oos_summary": oos, "folds": fold_dump}, indent=2))

            leaderboard.append({"symbol": symbol, "strategy": strategy_name, **oos})

    leaderboard_path = reports_dir / "leaderboard.json"
    existing = []
    if leaderboard_path.exists():
        existing = json.loads(leaderboard_path.read_text())
    this_run_keys = {(r["symbol"], r["strategy"]) for r in leaderboard}
    merged = [r for r in existing if (r["symbol"], r["strategy"]) not in this_run_keys] + leaderboard
    merged.sort(key=lambda r: r.get("total_oos_pnl", float("-inf")), reverse=True)
    leaderboard_path.write_text(json.dumps(merged, indent=2))
    leaderboard = merged

    print("\n\n=== LEADERBOARD (sorted by total OOS P&L) ===")
    for row in leaderboard:
        print(json.dumps(row))
    print(f"\nWrote {leaderboard_path}")


if __name__ == "__main__":
    main()
