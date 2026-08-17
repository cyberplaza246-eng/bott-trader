"""
Walk-forward parameter search: re-optimize strategy parameters on a rolling
train window, then evaluate those exact parameters on the following
out-of-sample test window (never re-optimized on data it will be judged on).
Concatenating only the OOS test folds gives an honest read on whether a
strategy's edge holds up outside the window it was tuned on, rather than the
inflated numbers you'd get by fitting params to the whole dataset at once.

Significance criteria (min trade count + t-stat threshold on pooled OOS
trade P&L) follow the same bar used by published falsification studies of
OHLCV-based intraday signals: a strategy doesn't get called "working" just
because its total P&L is positive — that's easy to hit by chance with few
trades. It has to clear a t-stat of 2.0 on a large-enough sample.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Type

import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_metrics

SIGNIFICANCE_MIN_TRADES = 30
SIGNIFICANCE_MIN_T_STAT = 2.0


@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: Dict[str, Any]
    train_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    test_trade_pnls: List[float] = field(default_factory=list)


def _param_combos(param_grid: Dict[str, Sequence[Any]]):
    if not param_grid:
        yield {}
        return
    keys = list(param_grid)
    for values in itertools.product(*param_grid.values()):
        yield dict(zip(keys, values))


def walk_forward_search(
    df: pd.DataFrame,
    strategy_cls: Type,
    param_grid: Dict[str, Sequence[Any]],
    symbol: str,
    timeframe: str,
    account_size: float,
    risk_pct: float,
    train_days: int = 180,
    test_days: int = 60,
    min_train_trades: int = 10,
) -> List[FoldResult]:
    results: List[FoldResult] = []
    end = df.index[-1]
    train_start = df.index[0]
    fold = 0

    while True:
        train_end = train_start + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)
        if test_end > end:
            break

        train_df = df[(df.index >= train_start) & (df.index < train_end)]
        test_df = df[(df.index >= train_end) & (df.index < test_end)]

        best_params, best_train_metrics, best_score = None, None, float("-inf")
        for params in _param_combos(param_grid):
            strat = strategy_cls(**params)
            res = run_backtest(train_df, strat, symbol, timeframe, account_size, risk_pct)
            m = compute_metrics(res, account_size)
            if m["trade_count"] < min_train_trades:
                continue
            if m["total_pnl"] > best_score:
                best_score, best_params, best_train_metrics = m["total_pnl"], params, m

        if best_params is not None:
            test_strat = strategy_cls(**best_params)
            test_res = run_backtest(test_df, test_strat, symbol, timeframe, account_size, risk_pct)
            test_metrics = compute_metrics(test_res, account_size)
            results.append(
                FoldResult(
                    fold=fold,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=train_end,
                    test_end=test_end,
                    best_params=best_params,
                    train_metrics=best_train_metrics,
                    test_metrics=test_metrics,
                    test_trade_pnls=[t.pnl for t in test_res.trades],
                )
            )

        fold += 1
        train_start = train_start + pd.Timedelta(days=test_days)

    return results


def _t_stat(pnls: List[float]) -> float | None:
    n = len(pnls)
    if n < 2:
        return None
    mean = sum(pnls) / n
    variance = sum((p - mean) ** 2 for p in pnls) / (n - 1)
    std = math.sqrt(variance)
    if std == 0:
        return None
    return mean / (std / math.sqrt(n))


def aggregate_oos(fold_results: List[FoldResult]) -> Dict[str, Any]:
    """Aggregate metrics across only the out-of-sample test folds, and apply
    a statistical significance test on the pooled OOS trade P&L (not just
    the sign of the total) — same bar used by published OHLCV-signal
    falsification studies: t-stat >= 2.0 on >= 30 trades."""
    if not fold_results:
        return {"folds": 0, "total_oos_trades": 0, "total_oos_pnl": 0.0, "significant": False}

    all_pnls = [float(p) for f in fold_results for p in f.test_trade_pnls]
    total_trades = len(all_pnls)
    total_pnl = sum(all_pnls)
    profitable_folds = sum(1 for f in fold_results if f.test_metrics["total_pnl"] > 0)
    win_rates = [f.test_metrics["win_rate"] for f in fold_results if f.test_metrics["win_rate"] is not None]

    t_stat = _t_stat(all_pnls)
    significant = bool(
        total_trades >= SIGNIFICANCE_MIN_TRADES
        and t_stat is not None
        and t_stat >= SIGNIFICANCE_MIN_T_STAT
        and total_pnl > 0
    )

    return {
        "folds": len(fold_results),
        "profitable_folds": profitable_folds,
        "total_oos_trades": total_trades,
        "total_oos_pnl": round(float(total_pnl), 2),
        "avg_oos_win_rate": round(sum(win_rates) / len(win_rates), 4) if win_rates else None,
        "avg_pnl_per_fold": round(total_pnl / len(fold_results), 2),
        "t_stat": round(t_stat, 3) if t_stat is not None else None,
        "significant": significant,
    }
