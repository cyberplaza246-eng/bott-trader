#!/usr/bin/env python3
"""
Rithmic extended-session backtest — variant D with DeepSeek news bias / LLM advisor.

Compares:
  A) Baseline (no LLM news, no LLM advisor)
  B) News bias only (USE_LLM_NEWS)
  C) LLM trade advisor only (LLM_ENABLED)
  D) Both enabled

Usage:
  python scripts/backtest_rithmic_llm.py --cached
  python scripts/backtest_rithmic_llm.py --cached --max-llm-calls 150 --max-news-calls 80
  python scripts/backtest_rithmic_llm.py --cached --cache-only   # no live API calls

Policy scorer (live only for now):
  USE_POLICY_SCORER=true POLICY_SCORER_MODE=advisory python start_live_mtf_scalping.py --paper
  # Backtest hook: policy scorer shares NewsAPI fetch with news_bias; extend BacktestFilterFactory
  # with PolicyScorer.evaluate_entry() when block-mode validation is needed.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import sys
from dataclasses import dataclass
from datetime import timezone
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from scripts.backtest_adaptive_variants import (
    AdaptiveBacktester,
    StrategyVariant,
    apply_base_cfg,
    resample_15m,
)
from scripts.backtest_nq_rithmic import resample_5m
from scripts.backtest_rithmic_adaptive import load_cached, load_profit_cfg, variant_from_cfg
from src.ai.llm_advisor import LLMTradeAdvisor
from src.ai.news_bias import NewsBiasAdvisor


LLM_CACHE_PATH = "data/llm_backtest_cache.json"
NEWS_CACHE_PATH = "data/news_backtest_cache.json"
RESULTS_PATH = "data/rithmic_llm_backtest.json"


@dataclass
class LLMMode:
    name: str
    use_news: bool = False
    use_llm: bool = False


MODES = [
    LLMMode("A_baseline"),
    LLMMode("B_news_only", use_news=True),
    LLMMode("C_llm_only", use_llm=True),
    LLMMode("D_both", use_news=True, use_llm=True),
]


class BacktestFilterFactory:
    """Build entry_filter callables with file-backed LLM/news caches."""

    def __init__(
        self,
        symbol: str,
        mode: LLMMode,
        max_llm_calls: int,
        max_news_calls: int,
        cache_only: bool,
    ):
        self.symbol = symbol
        self.mode = mode
        self.max_llm_calls = max_llm_calls
        self.max_news_calls = max_news_calls
        self.cache_only = cache_only
        self.llm_cache = self._load_json(LLM_CACHE_PATH)
        self.news_cache = self._load_json(NEWS_CACHE_PATH)
        self.llm_calls = 0
        self.news_calls = 0
        self.news_bias: Optional[NewsBiasAdvisor] = None
        self.llm: Optional[LLMTradeAdvisor] = None

        if mode.use_news:
            os.environ["USE_LLM_NEWS"] = "true"
            self.news_bias = NewsBiasAdvisor()
        if mode.use_llm:
            os.environ["LLM_ENABLED"] = "true"
            self.llm = LLMTradeAdvisor()

    @staticmethod
    def _load_json(path: str) -> dict:
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_caches(self):
        os.makedirs(os.path.dirname(LLM_CACHE_PATH), exist_ok=True)
        with open(LLM_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.llm_cache, f, indent=2)
        with open(NEWS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.news_cache, f, indent=2)

    def _news_bucket_key(self, dt) -> str:
        if hasattr(dt, "tzinfo") and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        hour = dt.astimezone(timezone.utc).strftime("%Y%m%d%H")
        return f"{self.symbol}:{hour}"

    def _llm_key(self, dt, direction: str) -> str:
        if hasattr(dt, "tzinfo") and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return f"{dt.isoformat()}:{direction.lower()}"

    def _get_news_bias(self, dt) -> Dict[str, Any]:
        key = self._news_bucket_key(dt)
        if key in self.news_cache:
            return self.news_cache[key]

        if self.cache_only or self.news_calls >= self.max_news_calls:
            return {
                "bias": "neutral",
                "confidence": 0.0,
                "reason": "cache miss / budget exhausted",
                "source": "fallback",
            }

        self.news_calls += 1
        as_of = dt if hasattr(dt, "tzinfo") and dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        bias = self.news_bias.get_bias(self.symbol, as_of=as_of)
        self.news_cache[key] = bias
        if self.news_calls % 5 == 0:
            self._save_caches()
        return bias

    def _llm_allow(
        self,
        direction: str,
        dt,
        row,
        ctx_5m,
        df_1m,
        df_5m,
        df_15m,
        entry: float,
        sl: float,
        tp: float,
    ) -> bool:
        key = self._llm_key(dt, direction)
        if key in self.llm_cache:
            return self.llm_cache[key].get("allowed", True)

        if self.cache_only or self.llm_calls >= self.max_llm_calls:
            return True

        self.llm_calls += 1
        signal = {
            "symbol": self.symbol,
            "direction": direction.lower(),
            "entry": entry,
            "sl": sl,
            "tp": tp,
        }
        review = self.llm.evaluate_trade(
            signal,
            ctx_5m,
            row_1m=row,
            df_1m=df_1m,
            df_5m=df_5m,
            df_15m=df_15m,
            dt=dt if hasattr(dt, "tzinfo") and dt.tzinfo else dt.replace(tzinfo=timezone.utc),
        )
        self.llm_cache[key] = review
        if self.llm_calls % 10 == 0:
            self._save_caches()
        return review.get("allowed", True)

    def make_filter(self):
        if not self.mode.use_news and not self.mode.use_llm:
            return None

        def entry_filter(**kwargs) -> bool:
            backtester: AdaptiveBacktester = kwargs["backtester"]
            direction = kwargs["direction"]
            dt = kwargs["dt"]
            row = kwargs["row"]
            ctx_5m = kwargs["ctx_5m"]
            df_1m = kwargs["df_1m"]
            df_5m = kwargs["df_5m"]
            df_15m = kwargs["df_15m"]
            entry = kwargs["entry"]
            sl = kwargs["sl"]
            tp = kwargs["tp"]

            if self.mode.use_news and self.news_bias:
                bias = self._get_news_bias(dt)
                verdict = self.news_bias.evaluate_direction(
                    direction, symbol=self.symbol, bias=bias
                )
                if not verdict.get("allowed", True):
                    backtester.skipped_filters["news"] += 1
                    return False

            if self.mode.use_llm and self.llm:
                if not self._llm_allow(
                    direction, dt, row, ctx_5m, df_1m, df_5m, df_15m, entry, sl, tp
                ):
                    backtester.skipped_filters["llm"] += 1
                    return False

            return True

        return entry_filter

    def finalize(self):
        if self.mode.use_news or self.mode.use_llm:
            self._save_caches()


def run_one(
    symbol: str,
    df_1m: pd.DataFrame,
    cfg: Dict[str, Any],
    variant: StrategyVariant,
    mode: LLMMode,
    max_llm_calls: int,
    max_news_calls: int,
    cache_only: bool,
) -> Dict[str, Any]:
    df_5m = resample_5m(df_1m)
    df_15m = resample_15m(df_1m)
    apply_base_cfg(cfg)
    variant.min_rr = cfg.get("min_rr", 1.0)

    factory = BacktestFilterFactory(
        symbol, mode, max_llm_calls, max_news_calls, cache_only
    )
    bt = AdaptiveBacktester(symbol, variant, cfg, entry_filter=factory.make_filter())
    with contextlib.redirect_stdout(io.StringIO()):
        stats = bt.run_variant(df_1m, df_5m, df_15m)
    factory.finalize()

    if stats.get("error"):
        return {"error": stats["error"]}
    return {
        "mode": mode.name,
        "trades": stats["total_trades"],
        "wr": stats["win_rate"],
        "pf": stats["profit_factor"],
        "pnl": stats["total_pnl"],
        "dd": stats["max_drawdown_pct"],
        "skipped_news": bt.skipped_filters.get("news", 0),
        "skipped_llm": bt.skipped_filters.get("llm", 0),
        "llm_api_calls": factory.llm_calls,
        "news_api_calls": factory.news_calls,
    }


def print_table(results: Dict[str, Any], symbols: List[str]):
    print(f"\n{'='*90}")
    print("  RITHMIC LLM BACKTEST — Variant D (extended session)")
    print(f"{'='*90}")
    hdr = f"{'Mode':<14} | {'Sym':<4} | {'Trades':>6} | {'WR%':>6} | {'PF':>5} | {'Net P&L':>10} | Skipped"
    print(hdr)
    print("-" * len(hdr))

    for mode_name in ["A_baseline", "B_news_only", "C_llm_only", "D_both"]:
        for sym in symbols:
            row = results["modes"][mode_name]["symbols"].get(sym, {})
            if "error" in row:
                print(f"{mode_name:<14} | {sym:<4} | ERROR: {row['error']}")
                continue
            skip = f"n={row.get('skipped_news', 0)} l={row.get('skipped_llm', 0)}"
            print(
                f"{mode_name:<14} | {sym:<4} | {row['trades']:>6} | "
                f"{row['wr']:>5.1f}% | {row['pf']:>5.2f} | ${row['pnl']:>9,.0f} | {skip}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", action="store_true", help="Use data/*_rithmic.csv")
    parser.add_argument("--symbol", default="both", choices=["NQ", "MNQ", "both"])
    parser.add_argument("--session", default="extended", choices=["rth", "extended"])
    parser.add_argument("--max-llm-calls", type=int, default=400)
    parser.add_argument("--max-news-calls", type=int, default=200)
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Use only file caches (no live API); misses fall back to allow/neutral",
    )
    args = parser.parse_args()

    if not args.cached:
        print("Note: --cached recommended; fetching Rithmic data not wired in this script.")
        print("Run: python scripts/backtest_nq_rithmic.py --symbol NQ --bars 20000 --save-only")
        sys.exit(1)

    cfg = load_profit_cfg()
    cfg["live_like"] = True
    cfg["session_mode"] = args.session
    variant = variant_from_cfg(cfg)
    symbols = ["NQ", "MNQ"] if args.symbol == "both" else [args.symbol]

    print(f"\nConfig: {cfg.get('name')} | session={args.session} | cache_only={args.cache_only}")
    print(f"LLM budget: {args.max_llm_calls} calls | News budget: {args.max_news_calls} calls\n")

    results: Dict[str, Any] = {
        "config": {k: cfg[k] for k in cfg if k != "entry_quality"},
        "session": args.session,
        "cache_only": args.cache_only,
        "modes": {},
    }

    for mode in MODES:
        print(f"Running {mode.name}...")
        mode_result: Dict[str, Any] = {"use_news": mode.use_news, "use_llm": mode.use_llm, "symbols": {}}
        for sym in symbols:
            df_1m = load_cached(sym)[0]
            run_cfg = copy.deepcopy(cfg)
            stats = run_one(
                sym, df_1m, run_cfg, variant, mode,
                args.max_llm_calls, args.max_news_calls, args.cache_only,
            )
            mode_result["symbols"][sym] = stats
            if "error" not in stats:
                print(
                    f"  {sym}: {stats['trades']} tr | WR {stats['wr']:.1f}% | "
                    f"PF {stats['pf']:.2f} | ${stats['pnl']:,.0f} | "
                    f"API llm={stats['llm_api_calls']} news={stats['news_api_calls']}"
                )
        results["modes"][mode.name] = mode_result

    print_table(results, symbols)

    base = results["modes"]["A_baseline"]["symbols"]
    print(f"\n{'='*90}")
    print("  DELTA vs A_baseline")
    print(f"{'='*90}")
    for mode_name in ["B_news_only", "C_llm_only", "D_both"]:
        for sym in symbols:
            b = base[sym]
            r = results["modes"][mode_name]["symbols"][sym]
            if "error" in b or "error" in r:
                continue
            print(
                f"  {mode_name} {sym}: dTrades {r['trades']-b['trades']:+d} | "
                f"dWR {r['wr']-b['wr']:+.1f}% | dPF {r['pf']-b['pf']:+.2f} | "
                f"dPnL ${r['pnl']-b['pnl']:+,.0f}"
            )

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
