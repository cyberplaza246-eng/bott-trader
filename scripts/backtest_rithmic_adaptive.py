#!/usr/bin/env python3
"""
Rithmic backtest — variant D (full adaptive) with live-like rules.

Mirrors live bot:
  - Rithmic 1m OHLC (LucidTrading feed)
  - Session: rth (9:30-4:30 ET) or extended (Globex) via cfg session_mode / env
  - NQ max $500/trade SL cap, $1000 daily loss
  - MNQ max $250/trade, $300 daily
  - Volatility filter 45 pts
  - 15M bias + bear/bull adaptive (from mnq_profit_config.json)

Usage:
    python scripts/backtest_rithmic_adaptive.py
    python scripts/backtest_rithmic_adaptive.py --symbol NQ --bars 20000
    python scripts/backtest_rithmic_adaptive.py --cached   # use saved CSV, no fetch
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import scripts.backtest_mtf_scalping as mtf
from scripts.backtest_adaptive_variants import (
    AdaptiveBacktester,
    StrategyVariant,
    apply_base_cfg,
    resample_15m,
)
from scripts.backtest_nq_rithmic import fetch_rithmic_bars, resample_5m, save_csv
from src.ai.entry_quality import parse_entry_quality
from src.utils.trading_session import coerce_session_mode, resolve_session_mode_from_env


def load_profit_cfg() -> Dict[str, Any]:
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mnq_profit_config.json")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    return {
        "atr": float(cfg.get("atr", 1.2)),
        "tp": float(cfg.get("tp", 1.3)),
        "adx": int(cfg.get("adx", 17)),
        "vol": float(cfg.get("vol", 0.4)),
        "di_tol": float(cfg.get("di_tol", 3.0)),
        "tp_buffer": float(cfg.get("tp_buffer", 0.5)),
        "max_tr": int(cfg.get("max_tr", 20)),
        "min_rr": float(cfg.get("min_rr", 1.0)),
        "di_counter": float(cfg.get("di_counter", 20.0)),
        "counter_adx": int(cfg.get("counter_adx", 25)),
        "live_like": True,
        "session_mode": coerce_session_mode(
            cfg.get("session_mode") or resolve_session_mode_from_env()
        ),
        "name": cfg.get("name", "custom"),
        "strategy_mode": cfg.get("strategy_mode", "full_adaptive"),
        "use_15m_bias": bool(cfg.get("use_15m_bias", True)),
        "bear_adaptive": bool(cfg.get("bear_adaptive", True)),
        "bull_adaptive": bool(cfg.get("bull_adaptive", True)),
        "strong_trend_adx": int(cfg.get("strong_trend_adx", 30)),
        "vwap_adx_min": int(cfg.get("vwap_adx_min", cfg.get("strong_trend_adx", 30))),
        "soft_15m_bias": bool(cfg.get("soft_15m_bias", False)),
        "15m_bias_mode": cfg.get("15m_bias_mode", "ema_cross"),
        "15m_bias_buffer_pts": float(cfg.get("15m_bias_buffer_pts", 0)),
        "vwap_required": bool(cfg.get("vwap_required", True)),
        "entry_quality": parse_entry_quality(cfg),
        "strong_trend_skip_macd": bool(cfg.get("strong_trend_skip_macd", False)),
        "strong_trend_min_rr": float(cfg.get("strong_trend_min_rr", 0.6)),
        "strong_trend_relax_adx": int(cfg.get("strong_trend_relax_adx", cfg.get("strong_trend_adx", 40))),
        "bull_rsi_lo": int(cfg.get("bull_rsi_lo", 35)),
        "bull_rsi_hi": int(cfg.get("bull_rsi_hi", 70)),
    }


def variant_from_cfg(cfg: Dict[str, Any]) -> StrategyVariant:
    return StrategyVariant(
        name="D",
        label=cfg.get("strategy_mode", "full_adaptive"),
        use_15m_bias=cfg.get("use_15m_bias", True),
        bear_adaptive=cfg.get("bear_adaptive", True),
        bull_adaptive=cfg.get("bull_adaptive", True),
        counter_trend_shorts=True,
        min_rr=cfg.get("min_rr", 1.0),
    )


def load_cached(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    p1 = os.path.join(data_dir, f"{symbol}_1m_rithmic.csv")
    p5 = os.path.join(data_dir, f"{symbol}_5m_rithmic.csv")
    if not os.path.isfile(p1):
        raise FileNotFoundError(f"No cached Rithmic CSV: {p1}")
    df_1m = pd.read_csv(p1, parse_dates=["datetime"])
    df_5m = pd.read_csv(p5, parse_dates=["datetime"]) if os.path.isfile(p5) else resample_5m(df_1m)
    df_1m = df_1m.sort_values("datetime").reset_index(drop=True)
    return df_1m, df_5m


def run_symbol(symbol: str, df_1m: pd.DataFrame, cfg: Dict[str, Any], variant: StrategyVariant) -> Dict[str, Any]:
    df_5m = resample_5m(df_1m)
    df_15m = resample_15m(df_1m)
    apply_base_cfg(cfg)
    variant.min_rr = cfg.get("min_rr", 1.0)
    bt = AdaptiveBacktester(symbol, variant, cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        stats = bt.run_variant(df_1m, df_5m, df_15m)
    if stats.get("error"):
        return {"symbol": symbol, "error": stats["error"]}
    return {
        "symbol": symbol,
        "source": "rithmic",
        "live_like": True,
        "bars_1m": len(df_1m),
        "start": str(df_1m["datetime"].iloc[0]),
        "end": str(df_1m["datetime"].iloc[-1]),
        "total_trades": stats["total_trades"],
        "win_rate": stats["win_rate"],
        "profit_factor": stats["profit_factor"],
        "total_pnl": stats["total_pnl"],
        "max_drawdown_pct": stats["max_drawdown_pct"],
        "avg_win": stats.get("avg_win", 0),
        "avg_loss": stats.get("avg_loss", 0),
    }


def run_baseline(symbol: str, df_1m: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Variant A with same live_like rules (fair apples-to-apples comparison)."""
    from scripts.backtest_adaptive_variants import VARIANTS

    baseline_variant = next(v for v in VARIANTS if v.name == "A")
    r = run_symbol(symbol, df_1m, cfg, baseline_variant)
    return {
        "label": "baseline_live_like",
        "variant": "A",
        "total_trades": r.get("total_trades", 0),
        "win_rate": r.get("win_rate", 0),
        "profit_factor": r.get("profit_factor", 0),
        "total_pnl": r.get("total_pnl", 0),
        "max_drawdown_pct": r.get("max_drawdown_pct", 0),
    }


COMPARE_VARIANTS = [
    ("A", "live_baseline_soft15m", {"use_15m_bias": True, "soft_15m_bias": True, "volatility_filter_points": 45}),
    ("B", "no_15m_bias", {"use_15m_bias": False, "soft_15m_bias": True, "volatility_filter_points": 45}),
    ("C", "hard_15m_bias", {"use_15m_bias": True, "soft_15m_bias": False, "volatility_filter_points": 45}),
    ("D", "relaxed_vol_60pt", {"use_15m_bias": True, "soft_15m_bias": True, "volatility_filter_points": 60}),
]


def cfg_for_compare(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(base)
    cfg.update(overrides)
    cfg["live_like"] = True
    return cfg


def variant_for_compare(cfg: Dict[str, Any], name: str) -> StrategyVariant:
    return StrategyVariant(
        name=name,
        label=cfg.get("strategy_mode", "full_adaptive"),
        use_15m_bias=cfg.get("use_15m_bias", True),
        bear_adaptive=cfg.get("bear_adaptive", True),
        bull_adaptive=cfg.get("bull_adaptive", True),
        counter_trend_shorts=True,
        min_rr=cfg.get("min_rr", 1.0),
    )


def pack_stats(run: Dict[str, Any]) -> Dict[str, Any]:
    if run.get("error"):
        return {"trades": 0, "wr": 0, "pf": 0, "pnl": 0, "dd": 0, "error": run["error"]}
    return {
        "trades": run["total_trades"],
        "wr": run["win_rate"],
        "pf": run["profit_factor"],
        "pnl": run["total_pnl"],
        "dd": run["max_drawdown_pct"],
    }


COMPARE_15M_BIAS_VARIANTS = [
    ("A", "ema_cross", {"use_15m_bias": True, "15m_bias_mode": "ema_cross", "volatility_filter_points": 45}),
    ("B", "price_ema50", {"use_15m_bias": True, "15m_bias_mode": "price_ema50", "volatility_filter_points": 45}),
    ("C", "hybrid", {"use_15m_bias": True, "15m_bias_mode": "hybrid", "volatility_filter_points": 45}),
    ("D", "no_15m", {"use_15m_bias": False, "15m_bias_mode": "ema_cross", "volatility_filter_points": 45}),
]


def run_compare_15m_bias(symbols: List[str], session: str) -> Dict[str, Any]:
    """Compare 15M bias modes on variant D full config (cached Rithmic, extended session)."""
    base = load_profit_cfg()
    base["session_mode"] = coerce_session_mode(session)
    base["live_like"] = True

    data: Dict[str, Any] = {
        "session": session,
        "compare": "15m_bias_mode",
        "base_config": {k: base[k] for k in base if k != "entry_quality"},
        "variants": {},
    }
    frames: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        frames[sym] = load_cached(sym)[0]

    print(f"\n{'='*100}")
    print("  RITHMIC 15M BIAS MODE COMPARE — variant D full config (extended, cached)")
    print(f"  Range NQ: {frames['NQ']['datetime'].iloc[0]} -> {frames['NQ']['datetime'].iloc[-1]}")
    print(
        f"  Bars: NQ={len(frames['NQ']):,}"
        + (f" MNQ={len(frames.get('MNQ', frames['NQ'])):,}" if "MNQ" in frames else "")
    )
    print(f"{'='*100}\n")

    hdr = (
        f"{'Var':<4} {'15m_mode':<14} | {'Sym':<4} | {'Trades':>6} | {'WR%':>6} | "
        f"{'PF':>5} | {'Net P&L':>10} | {'MaxDD%':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    for code, label, overrides in COMPARE_15M_BIAS_VARIANTS:
        cfg = cfg_for_compare(base, overrides)
        variant = variant_for_compare(cfg, "D")
        entry: Dict[str, Any] = {"label": label, "overrides": overrides, "symbols": {}}
        for sym in symbols:
            stats = run_symbol(sym, frames[sym], cfg, variant)
            packed = pack_stats(stats)
            entry["symbols"][sym] = packed
            print(
                f"{code:<4} {label:<14} | {sym:<4} | {packed['trades']:>6} | "
                f"{packed['wr']:>5.1f}% | {packed['pf']:>5.2f} | ${packed['pnl']:>9,.0f} | "
                f"{packed['dd']:>6.1f}%"
            )
        data["variants"][code] = entry

    out = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "rithmic_15m_bias_compare.json"
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved: {out}")
    return data


def run_compare(symbols: List[str], session: str) -> Dict[str, Any]:
    base = load_profit_cfg()
    base["session_mode"] = coerce_session_mode(session)
    base["live_like"] = True

    data: Dict[str, Any] = {
        "session": session,
        "base_config": {k: base[k] for k in base if k != "entry_quality"},
        "variants": {},
    }
    frames: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        frames[sym] = load_cached(sym)[0]

    print(f"\n{'='*96}")
    print("  RITHMIC VARIANT COMPARE — D config (extended session, cached)")
    print(f"  Range NQ: {frames['NQ']['datetime'].iloc[0]} -> {frames['NQ']['datetime'].iloc[-1]}")
    print(f"  Bars: NQ={len(frames['NQ']):,}" + (f" MNQ={len(frames.get('MNQ', frames['NQ'])):,}" if "MNQ" in frames else ""))
    print(f"{'='*96}\n")

    hdr = f"{'Var':<4} {'Label':<22} | {'Sym':<4} | {'Trades':>6} | {'WR%':>6} | {'PF':>5} | {'Net P&L':>10} | {'MaxDD%':>7}"
    print(hdr)
    print("-" * len(hdr))

    for code, label, overrides in COMPARE_VARIANTS:
        cfg = cfg_for_compare(base, overrides)
        variant = variant_for_compare(cfg, code)
        entry: Dict[str, Any] = {"label": label, "overrides": overrides, "symbols": {}}
        for sym in symbols:
            stats = run_symbol(sym, frames[sym], cfg, variant)
            packed = pack_stats(stats)
            entry["symbols"][sym] = packed
            print(
                f"{code:<4} {label:<22} | {sym:<4} | {packed['trades']:>6} | "
                f"{packed['wr']:>5.1f}% | {packed['pf']:>5.2f} | ${packed['pnl']:>9,.0f} | "
                f"{packed['dd']:>6.1f}%"
            )
        data["variants"][code] = entry

    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rithmic_compare_variants.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved: {out}")
    return data


COMPARE_STRONG_TREND_VARIANTS = [
    ("A", "baseline", {}),
    (
        "B",
        "strong_trend_relax",
        {
            "strong_trend_skip_macd": True,
            "strong_trend_min_rr": 0.6,
            "strong_trend_relax_adx": 40,
        },
    ),
    (
        "C",
        "strong_trend_relax_rsi80",
        {
            "strong_trend_skip_macd": True,
            "strong_trend_min_rr": 0.6,
            "strong_trend_relax_adx": 40,
            "bull_rsi_hi": 80,
        },
    ),
]


def run_compare_strong_trend(symbols: List[str], session: str) -> Dict[str, Any]:
    """Compare baseline vs strong-trend entry relaxations (variant D, cached Rithmic)."""
    base = load_profit_cfg()
    base["session_mode"] = coerce_session_mode(session)
    base["live_like"] = True
    base["15m_bias_mode"] = base.get("15m_bias_mode") or "price_ema50"

    data: Dict[str, Any] = {
        "session": session,
        "compare": "strong_trend_entry_relax",
        "base_config": {k: base[k] for k in base if k != "entry_quality"},
        "variants": {},
    }
    frames: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        frames[sym] = load_cached(sym)[0]

    print(f"\n{'='*100}")
    print("  RITHMIC STRONG-TREND ENTRY RELAX — variant D (extended, cached)")
    print(f"  15m_bias_mode={base.get('15m_bias_mode')} | min_rr={base.get('min_rr')}")
    print(f"  Range NQ: {frames['NQ']['datetime'].iloc[0]} -> {frames['NQ']['datetime'].iloc[-1]}")
    print(
        f"  Bars: NQ={len(frames['NQ']):,}"
        + (f" MNQ={len(frames.get('MNQ', frames['NQ'])):,}" if "MNQ" in frames else "")
    )
    print(f"{'='*100}\n")

    hdr = (
        f"{'Var':<4} {'Label':<26} | {'Sym':<4} | {'Trades':>6} | {'WR%':>6} | "
        f"{'PF':>5} | {'Net P&L':>10} | {'MaxDD%':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    for code, label, overrides in COMPARE_STRONG_TREND_VARIANTS:
        cfg = cfg_for_compare(base, overrides)
        variant = variant_for_compare(cfg, "D")
        entry: Dict[str, Any] = {"label": label, "overrides": overrides, "symbols": {}}
        for sym in symbols:
            stats = run_symbol(sym, frames[sym], cfg, variant)
            packed = pack_stats(stats)
            entry["symbols"][sym] = packed
            print(
                f"{code:<4} {label:<26} | {sym:<4} | {packed['trades']:>6} | "
                f"{packed['wr']:>5.1f}% | {packed['pf']:>5.2f} | ${packed['pnl']:>9,.0f} | "
                f"{packed['dd']:>6.1f}%"
            )
        data["variants"][code] = entry

    out = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "rithmic_strong_trend_compare.json"
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved: {out}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="NQ", choices=["NQ", "MNQ", "both"])
    parser.add_argument("--bars", type=int, default=20000, help="Max 1m bars from Rithmic")
    parser.add_argument("--cached", action="store_true", help="Use data/*_rithmic.csv (skip fetch)")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run A/B/C/D variant comparison (15m bias, vol filter) on cached Rithmic data",
    )
    parser.add_argument(
        "--compare-15m-bias",
        action="store_true",
        help="Compare 15m_bias_mode: ema_cross vs price_ema50 vs hybrid vs no_15m (variant D)",
    )
    parser.add_argument(
        "--compare-strong-trend",
        action="store_true",
        help="Compare baseline vs strong-trend MACD skip + min_rr relax (variant D)",
    )
    parser.add_argument("--session", default="extended", choices=["rth", "extended"])
    args = parser.parse_args()

    if args.compare_strong_trend:
        if not args.cached:
            print("ERROR: --compare-strong-trend requires --cached")
            sys.exit(1)
        symbols = ["NQ", "MNQ"] if args.symbol == "both" else [args.symbol]
        run_compare_strong_trend(symbols, args.session)
        return

    if args.compare_15m_bias:
        if not args.cached:
            print("ERROR: --compare-15m-bias requires --cached")
            sys.exit(1)
        symbols = ["NQ", "MNQ"] if args.symbol == "both" else [args.symbol]
        run_compare_15m_bias(symbols, args.session)
        return

    if args.compare:
        if not args.cached:
            print("ERROR: --compare requires --cached")
            sys.exit(1)
        symbols = ["NQ", "MNQ"] if args.symbol == "both" else [args.symbol]
        run_compare(symbols, args.session)
        return

    cfg = load_profit_cfg()
    cfg["session_mode"] = coerce_session_mode(args.session)
    variant = variant_from_cfg(cfg)
    symbols: List[str] = ["MNQ", "NQ"] if args.symbol == "both" else [args.symbol]

    print(f"\n{'='*72}")
    print("  RITHMIC LIVE-LIKE BACKTEST")
    print(f"  Strategy: {cfg['name']} ({cfg['strategy_mode']})")
    print(f"  Session: {args.session} | NQ SL cap $500 | MNQ SL cap $250")
    print(f"{'='*72}\n")

    results: Dict[str, Any] = {"config": cfg, "symbols": {}}

    for sym in symbols:
        print(f"\n--- {sym} ---")
        try:
            if args.cached:
                df_1m, _ = load_cached(sym)
                print(f"Loaded cached {len(df_1m):,} bars")
            else:
                df_1m = fetch_rithmic_bars(sym, args.bars, chunked=True)
                df_5m = resample_5m(df_1m)
                save_csv(sym, df_1m, df_5m)
        except Exception as e:
            print(f"FAIL {sym}: {e}")
            results["symbols"][sym] = {"error": str(e)}
            continue

        adaptive = run_symbol(sym, df_1m, cfg, variant)
        baseline = run_baseline(sym, df_1m, cfg)

        ok = adaptive["profit_factor"] >= 1.0 and adaptive["total_pnl"] > 0
        mark = "OK" if ok else "--"
        print(
            f"{mark} Adaptive D: {adaptive['total_trades']} tr | WR {adaptive['win_rate']:.1f}% | "
            f"PF {adaptive['profit_factor']:.2f} | PnL ${adaptive['total_pnl']:,.0f} | "
            f"DD {adaptive['max_drawdown_pct']:.1f}%"
        )
        print(
            f"   Baseline A (live-like): {baseline['total_trades']} tr | WR {baseline['win_rate']:.1f}% | "
            f"PF {baseline['profit_factor']:.2f} | PnL ${baseline['total_pnl']:,.0f}"
        )
        print(f"   Range: {adaptive['start']} -> {adaptive['end']}")

        results["symbols"][sym] = {"adaptive": adaptive, "baseline": baseline}

    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rithmic_adaptive_backtest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
