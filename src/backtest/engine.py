"""
Generic bar-by-bar backtest engine.

Single open position at a time. A strategy's `generate_signals` supplies an
entry direction and initial stop per bar; when `target_price` is NaN the
engine runs an ATR-trailing stop instead of a fixed take-profit (used by the
trend-following and breakout strategies, which aim to let winners run).

Fills are modeled at the close of the signal bar (+ slippage); stop/target
hits are detected using the *next* bar's high/low onward to avoid look-ahead
bias — a trade can never be closed on the same bar it was opened.

A strategy may optionally set `self.breakeven_r_mult = <float>` to move the
stop to entry price once favorable excursion reaches that many multiples of
initial risk (a real risk-management practice — lock in scratch, not just a
fixed stop or pure trail). Approximates a partial-scale-out plan (e.g. "2R
target, move stop to breakeven at 1R") without the engine supporting true
partial position exits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from config.instruments import get_spec
from config.settings import DEFAULT_SLIPPAGE_TICKS
from src.risk.position_sizing import contracts_for_risk
from src.strategies.base import Strategy, StrategySignals
from src.strategies.indicators import atr

TRAIL_ATR_PERIOD = 14
TRAIL_ATR_MULT = 2.0


@dataclass
class Trade:
    symbol: str
    direction: int  # 1 long, -1 short
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    contracts: int
    gross_pnl: float
    commission: float
    pnl: float  # net = gross_pnl - commission
    exit_reason: str  # "stop" | "target" | "trail" | "eod"


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    timeframe: str
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    symbol: str,
    timeframe: str,
    account_size: float,
    risk_pct: float,
    slippage_ticks: int = DEFAULT_SLIPPAGE_TICKS,
) -> BacktestResult:
    spec = get_spec(symbol)
    signals: StrategySignals = strategy.generate_signals(df)
    trail_atr = atr(df["high"], df["low"], df["close"], TRAIL_ATR_PERIOD)
    slippage_price = slippage_ticks * spec.tick_size
    breakeven_r_mult = getattr(strategy, "breakeven_r_mult", None)

    trades: List[Trade] = []
    equity = account_size
    equity_curve = []

    position: Optional[dict] = None
    n = len(df)

    for i in range(n):
        ts = df.index[i]
        row = df.iloc[i]

        if position is not None:
            direction = position["direction"]
            if direction == 1:
                position["trail_extreme"] = max(position["trail_extreme"], row["high"])
                if breakeven_r_mult is not None and not position["breakeven_triggered"]:
                    favorable = row["high"] - position["entry_price"]
                    if favorable >= breakeven_r_mult * position["initial_risk"]:
                        position["stop"] = max(position["stop"], position["entry_price"])
                        position["breakeven_triggered"] = True
                if position["trailing"]:
                    candidate = position["trail_extreme"] - TRAIL_ATR_MULT * trail_atr.iloc[i]
                    position["stop"] = max(position["stop"], candidate)
            else:
                position["trail_extreme"] = min(position["trail_extreme"], row["low"])
                if breakeven_r_mult is not None and not position["breakeven_triggered"]:
                    favorable = position["entry_price"] - row["low"]
                    if favorable >= breakeven_r_mult * position["initial_risk"]:
                        position["stop"] = min(position["stop"], position["entry_price"])
                        position["breakeven_triggered"] = True
                if position["trailing"]:
                    candidate = position["trail_extreme"] + TRAIL_ATR_MULT * trail_atr.iloc[i]
                    position["stop"] = min(position["stop"], candidate)

            exit_price = None
            exit_reason = None
            if direction == 1:
                if row["low"] <= position["stop"]:
                    exit_price, exit_reason = position["stop"], "trail" if position["trailing"] else "stop"
                elif position["target"] is not None and row["high"] >= position["target"]:
                    exit_price, exit_reason = position["target"], "target"
            else:
                if row["high"] >= position["stop"]:
                    exit_price, exit_reason = position["stop"], "trail" if position["trailing"] else "stop"
                elif position["target"] is not None and row["low"] <= position["target"]:
                    exit_price, exit_reason = position["target"], "target"

            if i == n - 1 and exit_price is None:
                exit_price, exit_reason = row["close"], "eod"

            if exit_price is not None:
                ticks = (exit_price - position["entry_price"]) / spec.tick_size * direction
                gross_pnl = ticks * spec.tick_value_usd * position["contracts"]
                commission = spec.commission_rt * position["contracts"]
                pnl = gross_pnl - commission
                equity += pnl
                trades.append(
                    Trade(
                        symbol=symbol,
                        direction=direction,
                        entry_time=position["entry_time"],
                        entry_price=position["entry_price"],
                        exit_time=ts,
                        exit_price=exit_price,
                        contracts=position["contracts"],
                        gross_pnl=gross_pnl,
                        commission=commission,
                        pnl=pnl,
                        exit_reason=exit_reason,
                    )
                )
                position = None

        if position is None and i < n - 1:
            entry_signal = signals.entries.iloc[i]
            if entry_signal != 0:
                direction = int(entry_signal)
                raw_stop = signals.stop_price.iloc[i]
                raw_target = signals.target_price.iloc[i]
                if not np.isnan(raw_stop):
                    entry_price = row["close"] + direction * slippage_price
                    stop = raw_stop
                    target = None if np.isnan(raw_target) else raw_target
                    contracts = contracts_for_risk(equity, risk_pct, entry_price, stop, spec)
                    if contracts > 0:
                        position = {
                            "direction": direction,
                            "entry_time": ts,
                            "entry_price": entry_price,
                            "stop": stop,
                            "target": target,
                            "contracts": contracts,
                            "trailing": target is None,
                            "trail_extreme": row["high"] if direction == 1 else row["low"],
                            "initial_risk": abs(entry_price - stop),
                            "breakeven_triggered": False,
                        }

        equity_curve.append((ts, equity))

    equity_series = pd.Series(
        [v for _, v in equity_curve], index=[t for t, _ in equity_curve], name="equity"
    )
    return BacktestResult(symbol=symbol, strategy=strategy.name, timeframe=timeframe, trades=trades, equity_curve=equity_series)
