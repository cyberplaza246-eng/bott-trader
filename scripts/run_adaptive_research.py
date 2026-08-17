#!/usr/bin/env python
"""
Stage 1 Research Engine: runs the expectancy-driven regime allocator
(src/regime/adaptive_engine.py) over full historical data for a symbol,
writes the per-decision research log and a summary that reports both:
  - full-history performance (every decision only conditions on trades
    already closed as of that decision -- walk-forward-safe by construction)
  - a true held-out final-3-months slice, isolated from anything used to
    shape the regime feature design

Both get run through the same significance bar used everywhere else in this
project (t-stat >= 2.0 on >= 30 trades, cost-adjusted, positive P&L).

Usage:
    python scripts/run_adaptive_research.py --symbol MNQ --timeframe 5m
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DEFAULT_ACCOUNT_SIZE, DEFAULT_RISK_PCT, DEFAULT_TIMEFRAME, REPORTS_DIR, SUPPORTED_SYMBOLS  # noqa: E402
from src.backtest.walk_forward import SIGNIFICANCE_MIN_T_STAT, SIGNIFICANCE_MIN_TRADES, _t_stat  # noqa: E402
from src.data.loader import load_ohlcv  # noqa: E402
from src.regime.adaptive_engine import run_adaptive_backtest  # noqa: E402
from src.strategies.breakout import BreakoutStrategy  # noqa: E402
from src.strategies.mean_reversion import MeanReversionStrategy  # noqa: E402
from src.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy  # noqa: E402
from src.strategies.orb_failure import OrbFailureStrategy  # noqa: E402
from src.strategies.trend_following import TrendFollowingStrategy  # noqa: E402
from src.strategies.vwap_pullback_trend import VwapPullbackTrendStrategy  # noqa: E402

HOLD_OUT_DAYS = 90


def default_pool():
    return {
        "trend": [TrendFollowingStrategy(), VwapPullbackTrendStrategy()],
        "range": [MeanReversionStrategy()],
        "breakout": [OpeningRangeBreakoutStrategy(), BreakoutStrategy(), OrbFailureStrategy()],
    }


def summarize(trades, label: str) -> dict:
    pnls = [t.pnl for t in trades]
    n = len(pnls)
    if n == 0:
        return {"segment": label, "trade_count": 0, "total_pnl": 0.0, "significant": False}
    total = sum(pnls)
    t_stat = _t_stat(pnls)
    wins = [p for p in pnls if p > 0]
    significant = bool(
        n >= SIGNIFICANCE_MIN_TRADES and t_stat is not None and t_stat >= SIGNIFICANCE_MIN_T_STAT and total > 0
    )
    return {
        "segment": label,
        "trade_count": n,
        "win_rate": round(len(wins) / n, 4),
        "total_pnl": round(total, 2),
        "t_stat": round(t_stat, 3) if t_stat is not None else None,
        "significant": significant,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True, choices=SUPPORTED_SYMBOLS)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--account-size", type=float, default=DEFAULT_ACCOUNT_SIZE)
    parser.add_argument("--risk-pct", type=float, default=DEFAULT_RISK_PCT)
    args = parser.parse_args()

    df = load_ohlcv(args.symbol, args.timeframe)
    result, log_df = run_adaptive_backtest(
        df=df, symbol=args.symbol, timeframe=args.timeframe, strategy_pool=default_pool(),
        account_size=args.account_size, risk_pct=args.risk_pct,
    )

    hold_out_start = df.index[-1] - __import__("pandas").Timedelta(days=HOLD_OUT_DAYS)
    full_history_trades = result.trades
    hold_out_trades = [t for t in result.trades if t.entry_time >= hold_out_start]

    summary = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "full_history": summarize(full_history_trades, "full_history"),
        "final_hold_out": summarize(hold_out_trades, f"final_{HOLD_OUT_DAYS}_days"),
        "selected_strategy_counts": log_df["selected_strategy"].value_counts().to_dict(),
    }

    reports_dir = Path(__file__).resolve().parent.parent / REPORTS_DIR / "adaptive"
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(reports_dir / f"{args.symbol}_research_log.csv", index=False)
    (reports_dir / f"{args.symbol}_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
