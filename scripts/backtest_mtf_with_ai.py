#!/usr/bin/env python3
"""
Backtest MTF scalping: baseline vs smart rules vs smart + DeepSeek.

Usage:
  python scripts/backtest_mtf_with_ai.py
  python scripts/backtest_mtf_with_ai.py --use-llm --max-llm-calls 80
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import timezone
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from scripts.backtest_mtf_scalping import (
    ADX_THRESHOLD,
    INITIAL_BALANCE,
    MAX_TRADES_PER_DAY,
    RESISTANCE_LOOKBACK,
    TP_MULT,
    TP_BUFFER_ATR_MULT,
    Trade,
    add_indicators_1m,
    add_indicators_5m,
    check_long_entry,
    check_short_entry,
    get_5m_context,
    load_data,
)
from src.ai.llm_advisor import LLMTradeAdvisor
from src.ai.mnq_context import compute_vwap
from src.ai.mnq_smart_filters import MNQSmartFilters


LLM_CACHE_PATH = "data/llm_backtest_cache.json"
RESULTS_PATH = "data/mtf_ai_backtest_results.json"


@dataclass
class ModeConfig:
    name: str
    use_smart: bool = False
    use_llm: bool = False


class AIEnhancedBacktester:
    def __init__(self, symbol: str, mode: ModeConfig, atr_mult: float = 1.2):
        self.symbol = symbol
        self.mode = mode
        self.atr_mult = atr_mult
        self.point_value = 2.0
        self.balance = INITIAL_BALANCE
        self.trades: List[Trade] = []
        self.skipped = {"mtf": 0, "smart": 0, "llm": 0}
        self.smart = MNQSmartFilters() if mode.use_smart or mode.use_llm else None
        self.llm = LLMTradeAdvisor() if mode.use_llm else None
        self.llm_cache = self._load_llm_cache()
        self.llm_calls = 0

    def _load_llm_cache(self) -> dict:
        if os.path.exists(LLM_CACHE_PATH):
            try:
                with open(LLM_CACHE_PATH, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_llm_cache(self):
        os.makedirs(os.path.dirname(LLM_CACHE_PATH), exist_ok=True)
        with open(LLM_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.llm_cache, f, indent=2)

    def _calc_pnl(self, trade: Trade) -> float:
        if trade.direction == "LONG":
            pts = trade.exit_price - trade.entry_price
        else:
            pts = trade.entry_price - trade.exit_price
        return pts * self.point_value - 2.50

    def _record_trade(self, trade: Trade):
        pnl = self._calc_pnl(trade)
        trade.pnl = pnl
        self.balance += pnl
        self.trades.append(deepcopy(trade))

    def _check_exit(self, trade: Trade, row: pd.Series) -> Optional[Trade]:
        hi, lo = row["high"], row["low"]
        if trade.direction == "LONG":
            if lo <= trade.sl:
                trade.exit_time = row["datetime"]
                trade.exit_price = trade.sl
                trade.exit_reason = "SL"
                self._record_trade(trade)
                return None
            if hi >= trade.tp:
                trade.exit_time = row["datetime"]
                trade.exit_price = trade.tp
                trade.exit_reason = "TP"
                self._record_trade(trade)
                return None
        else:
            if hi >= trade.sl:
                trade.exit_time = row["datetime"]
                trade.exit_price = trade.sl
                trade.exit_reason = "SL"
                self._record_trade(trade)
                return None
            if lo <= trade.tp:
                trade.exit_time = row["datetime"]
                trade.exit_price = trade.tp
                trade.exit_reason = "TP"
                self._record_trade(trade)
                return None
        return trade

    def _llm_allow(self, signal: dict, row, ctx_5m, df_1m, df_5m, df_15m, dt, max_calls: int) -> bool:
        key = f"{dt.isoformat()}:{signal['direction']}"
        if key in self.llm_cache:
            return self.llm_cache[key].get("allowed", True)

        if self.llm_calls >= max_calls:
            # Fall back to smart-only decision when budget exhausted
            return True

        self.llm_calls += 1
        review = self.llm.evaluate_trade(
            signal,
            ctx_5m,
            row_1m=row,
            df_1m=df_1m,
            df_5m=df_5m,
            df_15m=df_15m,
            dt=dt,
        )
        self.llm_cache[key] = review
        if self.llm_calls % 10 == 0:
            self._save_llm_cache()
        return review.get("allowed", True)

    def run(
        self,
        df_1m: pd.DataFrame,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        max_llm_calls: int = 9999,
    ) -> Dict:
        df_1m = add_indicators_1m(df_1m.copy())
        df_5m = add_indicators_5m(df_5m.copy())
        df_1m["vwap"] = compute_vwap(df_1m)

        warmup = max(200, 200)
        position: Optional[Trade] = None
        daily_trades = 0
        current_date = None

        for i in range(warmup, len(df_1m)):
            row = df_1m.iloc[i]
            dt = row["datetime"]
            if hasattr(dt, "tzinfo") and dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            trade_date = dt.date() if hasattr(dt, "date") else None
            if trade_date != current_date:
                current_date = trade_date
                daily_trades = 0

            if position:
                position = self._check_exit(position, row)
                continue

            if daily_trades >= MAX_TRADES_PER_DAY:
                continue

            ctx_5m = get_5m_context(df_5m, dt)
            atr = row["atr"]
            if pd.isna(atr) or atr <= 0:
                continue

            direction = None
            if check_long_entry(row, ctx_5m):
                direction = "LONG"
            elif check_short_entry(row, ctx_5m):
                direction = "SHORT"
            else:
                continue

            sl_distance = atr * self.atr_mult
            tp_distance = sl_distance * TP_MULT
            entry_price = row["close"]

            if direction == "LONG":
                tp_rr = entry_price + tp_distance
                tp_buffer = atr * TP_BUFFER_ATR_MULT
                tp_res = ctx_5m["resistance"] - tp_buffer
                tp_final = min(tp_rr, tp_res) if tp_res > entry_price else tp_rr
                sl = entry_price - sl_distance
            else:
                tp_rr = entry_price - tp_distance
                tp_buffer = atr * TP_BUFFER_ATR_MULT
                tp_sup = ctx_5m["support"] + tp_buffer
                tp_final = max(tp_rr, tp_sup) if tp_sup < entry_price else tp_rr
                sl = entry_price + sl_distance

            signal = {
                "symbol": self.symbol,
                "direction": direction.lower(),
                "entry": entry_price,
                "sl": sl,
                "tp": tp_final,
            }

            if self.mode.use_smart or self.mode.use_llm:
                verdict = self.smart.evaluate(
                    direction.lower(), dt, row, ctx_5m, df_1m, df_5m, df_15m
                )
                if not verdict["allowed"]:
                    self.skipped["smart"] += 1
                    continue

            if self.mode.use_llm:
                if not self._llm_allow(
                    signal, row, ctx_5m, df_1m, df_5m, df_15m, dt, max_llm_calls
                ):
                    self.skipped["llm"] += 1
                    continue

            position = Trade(
                entry_time=dt,
                direction=direction,
                entry_price=entry_price,
                sl=sl,
                tp=tp_final,
                initial_sl=sl,
            )
            daily_trades += 1

        if position:
            last = df_1m.iloc[-1]
            position.exit_time = last["datetime"]
            position.exit_price = last["close"]
            position.exit_reason = "END"
            self._record_trade(position)

        if self.mode.use_llm:
            self._save_llm_cache()

        return self._stats()

    def _stats(self) -> Dict:
        if not self.trades:
            return {
                "mode": self.mode.name,
                "trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_pnl": 0,
                "skipped": self.skipped,
                "llm_calls": self.llm_calls,
            }
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        gp = sum(t.pnl for t in wins)
        gl = abs(sum(t.pnl for t in losses)) or 1
        return {
            "mode": self.mode.name,
            "trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.trades) * 100,
            "profit_factor": gp / gl,
            "total_pnl": sum(t.pnl for t in self.trades),
            "final_balance": self.balance,
            "skipped": self.skipped,
            "llm_calls": self.llm_calls,
        }


def resample_15m(df_5m: pd.DataFrame) -> pd.DataFrame:
    d = df_5m.copy()
    d = d.set_index("datetime")
    o = d["open"].resample("15min").first()
    h = d["high"].resample("15min").max()
    l = d["low"].resample("15min").min()
    c = d["close"].resample("15min").last()
    v = d["volume"].resample("15min").sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna()
    out = out.reset_index()
    out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["ema_200"] = out["close"].ewm(span=200, adjust=False).mean()
    return out


def print_row(s: Dict):
    print(
        f"  {s['mode']:<22} | {s['trades']:>4} trades | "
        f"WR {s['win_rate']:>5.1f}% | PF {s['profit_factor']:>4.2f} | "
        f"PnL ${s['total_pnl']:>8,.0f} | skipped smart={s['skipped']['smart']} llm={s['skipped']['llm']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="MNQ")
    parser.add_argument("--use-llm", action="store_true", help="Run DeepSeek on smart-filtered signals")
    parser.add_argument("--max-llm-calls", type=int, default=120, help="Cap live LLM API calls")
    args = parser.parse_args()

    print("=" * 78)
    print("  MNQ MTF BACKTEST — Baseline vs Smart Rules vs DeepSeek")
    print("=" * 78)

    df_1m, df_5m = load_data(args.symbol)
    df_15m = resample_15m(df_5m)
    print(f"Data: {len(df_1m):,} 1m | {len(df_5m):,} 5m | {df_1m['datetime'].min()} → {df_1m['datetime'].max()}")

    modes = [
        ModeConfig("1_baseline_mtf"),
        ModeConfig("2_mtf_smart_rules", use_smart=True),
    ]
    if args.use_llm:
        modes.append(ModeConfig("3_mtf_smart_deepseek", use_smart=True, use_llm=True))

    results = []
    for mode in modes:
        print(f"\nRunning {mode.name}...")
        bt = AIEnhancedBacktester(args.symbol, mode)
        max_llm = args.max_llm_calls if mode.use_llm else 0
        stats = bt.run(df_1m, df_5m, df_15m, max_llm_calls=max_llm)
        results.append(stats)
        print_row(stats)

    payload = {"symbol": args.symbol, "results": results}
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 78)
    print("  SUMMARY (your advice = session + news + structure + MTF alignment)")
    print("=" * 78)
    base = results[0]
    for r in results[1:]:
        d_pnl = r["total_pnl"] - base["total_pnl"]
        d_pf = r["profit_factor"] - base["profit_factor"]
        print(f"  {r['mode']}: ΔPnL ${d_pnl:+,.0f} | ΔPF {d_pf:+.2f} vs baseline")
    print(f"\n  Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
