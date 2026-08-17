#!/usr/bin/env python
"""
Run a backtest for one symbol/timeframe/strategy and write results to reports/.

Example:
    python scripts/run_backtest.py --symbol MNQ --timeframe 5m --strategy ensemble \
        --start 2025-12-29 --end 2026-03-11
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (  # noqa: E402
    DEFAULT_ACCOUNT_SIZE,
    DEFAULT_RISK_PCT,
    DEFAULT_TIMEFRAME,
    REPORTS_DIR,
    SUPPORTED_STRATEGIES,
    SUPPORTED_SYMBOLS,
    SUPPORTED_TIMEFRAMES,
)
from src.backtest.engine import run_backtest  # noqa: E402
from src.backtest.metrics import compute_metrics  # noqa: E402
from src.data.loader import load_ohlcv, slice_range  # noqa: E402
from src.strategies import get_strategy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True, choices=SUPPORTED_SYMBOLS)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, choices=SUPPORTED_TIMEFRAMES)
    parser.add_argument("--strategy", required=True, choices=SUPPORTED_STRATEGIES)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD")
    parser.add_argument("--account-size", type=float, default=DEFAULT_ACCOUNT_SIZE)
    parser.add_argument("--risk-pct", type=float, default=DEFAULT_RISK_PCT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_ohlcv(args.symbol, args.timeframe)
    df = slice_range(df, args.start, args.end)
    if df.empty:
        raise SystemExit(f"No data for {args.symbol} {args.timeframe} in range {args.start}..{args.end}")

    strategy = get_strategy(args.strategy)
    result = run_backtest(
        df=df,
        strategy=strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        account_size=args.account_size,
        risk_pct=args.risk_pct,
    )
    metrics = compute_metrics(result, args.account_size)

    reports_dir = Path(__file__).resolve().parent.parent / REPORTS_DIR
    reports_dir.mkdir(exist_ok=True)
    run_tag = f"{args.symbol}_{args.timeframe}_{args.strategy}"

    trade_log_path = reports_dir / f"{run_tag}_trades.csv"
    equity_path = reports_dir / f"{run_tag}_equity.csv"
    summary_path = reports_dir / f"{run_tag}_summary.json"

    import pandas as pd

    trades_df = pd.DataFrame([vars(t) for t in result.trades])
    trades_df.to_csv(trade_log_path, index=False)
    result.equity_curve.to_csv(equity_path, header=True)
    summary_path.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"\nWrote: {trade_log_path}\n       {equity_path}\n       {summary_path}")


if __name__ == "__main__":
    main()
