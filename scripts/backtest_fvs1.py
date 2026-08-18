#!/usr/bin/env python3
"""
FVS-1 backtest on MNQ 1M bars (synthetic 30s trigger from 1M).

Usage:
    python scripts/backtest_fvs1.py
    python scripts/backtest_fvs1.py --csv data/MNQ_1m.csv --fee 1.50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_mtf_scalping as mtf
import scripts.backtest_scalp_momentum as scalp_bt
from src.ai.mnq_context import compute_vwap
from src.strategy.fvs1.config import FVS1Config, is_fvs1_session_et
from src.strategy.fvs1.triple_a import FVS1State, check_fvs1_entry, evaluate_fvs1_gates


POINT_VALUE = {"MNQ": 2.0, "NQ": 20.0}
WARMUP = 250
MAX_TRADES_PER_DAY = 15


@dataclass
class FVS1Trade:
    entry_time: pd.Timestamp
    direction: str
    entry_price: float
    sl: float
    tp: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    exit_reason: str = ""
    hold_seconds: float = 0.0


@dataclass
class BacktestResult:
    trades: List[FVS1Trade] = field(default_factory=list)
    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0

    @property
    def count(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        return (self.wins / self.count * 100.0) if self.count else 0.0

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

    @property
    def avg_hold(self) -> float:
        holds = [t.hold_seconds for t in self.trades if t.hold_seconds > 0]
        return sum(holds) / len(holds) if holds else 0.0

    @property
    def expectancy(self) -> float:
        return self.total_pnl / self.count if self.count else 0.0


def prepare_data(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_1m = scalp_bt.normalize_dt(pd.read_csv(csv_path))
    df_5m = scalp_bt.resample_5m(df_1m)
    df_5m = scalp_bt.add_5m_indicators(df_5m)
    df_1m_raw = scalp_bt.add_1m_indicators(df_1m)
    df_30s = scalp_bt.resample_30s_synthetic(df_1m_raw)
    df_30s = scalp_bt.attach_context(df_30s, df_1m_raw, ["ema_20", "atr"], "_1m")
    df_30s = scalp_bt.attach_context(df_30s, df_5m, ["adx", "di_plus", "di_minus", "vwap"], "_5m")
    return df_1m_raw, df_5m, df_30s


def run_backtest(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_30s: pd.DataFrame,
    cfg: FVS1Config,
    symbol: str = "MNQ",
) -> BacktestResult:
    cfg_bt = FVS1Config.from_env()
    for k, v in cfg.__dict__.items():
        if k != "sessions":
            setattr(cfg_bt, k, v)
    cfg_bt.log_only = False

    result = BacktestResult()
    open_trade: Optional[FVS1Trade] = None
    daily_counts: Dict[str, int] = {}
    pv = POINT_VALUE.get(symbol, 2.0)
    fee = cfg_bt.round_trip_fee

    state = FVS1State()
    i30 = 0
    n30 = len(df_30s)

    while i30 < n30:
        row_30s = df_30s.iloc[i30]
        dt = pd.Timestamp(row_30s["datetime"])
        if hasattr(dt, "tz_convert"):
            try:
                dt_et = dt.tz_convert("US/Eastern")
            except Exception:
                dt_et = dt
        else:
            dt_et = dt

        # Map to 1m index
        idx_1m = df_1m["datetime"].searchsorted(row_30s["datetime"], side="right") - 1
        if idx_1m < WARMUP:
            i30 += 1
            continue

        row_1m_slice = df_1m.iloc[: idx_1m + 1]
        row_1m = df_1m.iloc[idx_1m]
        h, l, c = float(row_30s["high"]), float(row_30s["low"]), float(row_30s["close"])

        if open_trade:
            ot = open_trade
            hold = (dt - ot.entry_time).total_seconds()
            exit_px = None
            reason = ""
            if ot.direction == "long":
                if l <= ot.sl:
                    exit_px, reason = ot.sl, "SL"
                elif h >= ot.tp:
                    exit_px, reason = ot.tp, "TP"
            else:
                if h >= ot.sl:
                    exit_px, reason = ot.sl, "SL"
                elif l <= ot.tp:
                    exit_px, reason = ot.tp, "TP"
            if exit_px is None and hold >= cfg_bt.max_hold_seconds:
                exit_px, reason = c, "MAX_HOLD"
            if exit_px is not None:
                if ot.direction == "long":
                    pnl = (exit_px - ot.entry_price) * pv - fee
                else:
                    pnl = (ot.entry_price - exit_px) * pv - fee
                ot.exit_time = dt
                ot.exit_price = exit_px
                ot.pnl = pnl
                ot.exit_reason = reason
                ot.hold_seconds = hold
                result.trades.append(ot)
                result.total_pnl += pnl
                if pnl > 0:
                    result.wins += 1
                else:
                    result.losses += 1
                open_trade = None
            i30 += 1
            continue

        day_key = str(dt.date())
        if daily_counts.get(day_key, 0) >= MAX_TRADES_PER_DAY:
            i30 += 1
            continue

        in_sess, _ = is_fvs1_session_et(dt_et, cfg_bt.sessions)
        if not in_sess:
            i30 += 1
            continue

        prev_30s = df_30s.iloc[i30 - 1] if i30 > 0 else None
        if prev_30s is None:
            i30 += 1
            continue

        row_5m = pd.Series({
            "adx": row_30s.get("adx_5m", 0),
            "di_plus": row_30s.get("di_plus_5m", 0),
            "di_minus": row_30s.get("di_minus_5m", 0),
            "vwap": row_30s.get("vwap_5m", 0),
            "close": row_1m["close"],
        })

        signal, state = check_fvs1_entry(
            symbol, row_1m_slice, row_5m, row_30s, prev_30s, state, cfg_bt,
            df_30s=df_30s.iloc[: i30 + 1], now_et=dt_et,
        )
        if signal:
            open_trade = FVS1Trade(
                entry_time=dt,
                direction=signal["direction"],
                entry_price=signal["entry"],
                sl=signal["sl"],
                tp=signal["tp"],
            )
            daily_counts[day_key] = daily_counts.get(day_key, 0) + 1
        i30 += 1

    return result


def summarize(result: BacktestResult, cfg: FVS1Config) -> Dict[str, Any]:
    return {
        "strategy": "fvs1",
        "trades": result.count,
        "wins": result.wins,
        "losses": result.losses,
        "win_rate_pct": round(result.win_rate, 2),
        "profit_factor": round(result.profit_factor, 3) if result.profit_factor != float("inf") else 999.0,
        "total_pnl": round(result.total_pnl, 2),
        "expectancy": round(result.expectancy, 2),
        "avg_hold_seconds": round(result.avg_hold, 1),
        "round_trip_fee": cfg.round_trip_fee,
        "max_hold_seconds": cfg.max_hold_seconds,
        "sample_trades": [
            {
                "entry_time": str(t.entry_time),
                "direction": t.direction,
                "entry": t.entry_price,
                "sl": t.sl,
                "tp": t.tp,
                "exit": t.exit_price,
                "pnl": round(t.pnl, 2),
                "reason": t.exit_reason,
                "hold_s": round(t.hold_seconds, 1),
            }
            for t in result.trades[:20]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FVS-1 backtest")
    parser.add_argument("--csv", default="data/MNQ_1m.csv")
    parser.add_argument("--symbol", default="MNQ")
    parser.add_argument("--fee", type=float, default=None)
    parser.add_argument("--out", default="data/fvs1_backtest_results.json")
    args = parser.parse_args()

    cfg = FVS1Config.from_env()
    if args.fee is not None:
        cfg.round_trip_fee = args.fee
    cfg.log_only = False

    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), csv_path)

    print(f"Loading {csv_path}...")
    df_1m, df_5m, df_30s = prepare_data(csv_path)
    print(f"  1M={len(df_1m)} 5M={len(df_5m)} 30s={len(df_30s)}")

    result = run_backtest(df_1m, df_5m, df_30s, cfg, symbol=args.symbol)
    summary = summarize(result, cfg)

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFVS-1 Backtest ({args.symbol})")
    print(f"  Trades: {summary['trades']}  WR: {summary['win_rate_pct']:.1f}%  PF: {summary['profit_factor']:.2f}")
    print(f"  Total PnL: ${summary['total_pnl']:.2f}  E: ${summary['expectancy']:.2f}/trade")
    print(f"  Avg hold: {summary['avg_hold_seconds']:.0f}s  Fee: ${cfg.round_trip_fee:.2f} RT")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
