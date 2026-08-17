"""Summary performance metrics computed from a BacktestResult."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult


def compute_metrics(result: BacktestResult, account_size: float) -> Dict[str, Any]:
    trades = result.trades
    n_trades = len(trades)

    if n_trades == 0:
        return {
            "symbol": result.symbol,
            "strategy": result.strategy,
            "timeframe": result.timeframe,
            "trade_count": 0,
            "win_rate": None,
            "profit_factor": None,
            "expectancy": None,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe": None,
            "final_equity": account_size,
        }

    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    win_rate = len(wins) / n_trades
    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    expectancy = pnls.mean()
    total_pnl = pnls.sum()

    equity = result.equity_curve
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_drawdown = drawdown.min()

    bar_returns = equity.pct_change().dropna()
    sharpe = None
    if len(bar_returns) > 1 and bar_returns.std() > 0:
        sharpe = float(bar_returns.mean() / bar_returns.std() * np.sqrt(252 * 78))  # ~78 5m bars/session

    return {
        "symbol": result.symbol,
        "strategy": result.strategy,
        "timeframe": result.timeframe,
        "trade_count": n_trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else None,
        "expectancy": round(float(expectancy), 2),
        "total_pnl": round(float(total_pnl), 2),
        "max_drawdown": round(float(max_drawdown), 2),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "final_equity": round(float(equity.iloc[-1]), 2),
    }
