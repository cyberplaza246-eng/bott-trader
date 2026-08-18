#!/usr/bin/env python3
"""
Fetch NQ (or MNQ) history from Rithmic and run the MTF scalping backtest.

Usage:
    python scripts/backtest_nq_rithmic.py
    python scripts/backtest_nq_rithmic.py --symbol NQ --bars 10000
    python scripts/backtest_nq_rithmic.py --symbol MNQ --save-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.broker.rithmic_connector import RithmicConnector
import scripts.backtest_mtf_scalping as mtf


def apply_profit_config() -> None:
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mnq_profit_config.json")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    mtf.TP_MULT = float(cfg.get("tp", mtf.TP_MULT))
    mtf.ADX_THRESHOLD = int(cfg.get("adx", mtf.ADX_THRESHOLD))
    mtf.VOLUME_RATIO_THRESHOLD = float(cfg.get("vol", mtf.VOLUME_RATIO_THRESHOLD))
    mtf.DI_TOLERANCE = float(cfg.get("di_tol", mtf.DI_TOLERANCE))
    mtf.TP_BUFFER_ATR_MULT = float(cfg.get("tp_buffer", mtf.TP_BUFFER_ATR_MULT))
    mtf.MAX_TRADES_PER_DAY = int(cfg.get("max_tr", mtf.MAX_TRADES_PER_DAY))
    print(
        f"Config: TP×{mtf.TP_MULT} ADX≥{mtf.ADX_THRESHOLD} vol≥{mtf.VOLUME_RATIO_THRESHOLD} "
        f"({cfg.get('name', 'custom')})"
    )


def resample_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    d = df_1m.copy()
    if "datetime" not in d.columns:
        raise ValueError("1m data missing datetime column")
    d["datetime"] = pd.to_datetime(d["datetime"], utc=True)
    d = d.set_index("datetime").sort_index()
    df_5m = d.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna().reset_index()
    return df_5m


def fetch_rithmic_bars(symbol: str, bars_1m: int, chunked: bool = True) -> pd.DataFrame:
    os.environ["RITHMIC_DISABLE_YAHOO_FALLBACK"] = "true"
    broker = RithmicConnector()
    broker.initialize()
    if not broker.connected:
        raise RuntimeError(
            "Rithmic not connected — check RITHMIC_USER_ID, RITHMIC_PASSWORD, RITHMIC_SYSTEM in .env"
        )

    print(f"Fetching up to {bars_1m:,} × 1m bars for {symbol} from Rithmic ({broker._system})...")
    if chunked and bars_1m > 8000:
        df_1m = broker.fetch_history_chunked(symbol, timeframe_minutes=1, max_bars=bars_1m)
    else:
        df_1m = broker.get_candles(symbol, timeframe_minutes=1, num_candles=bars_1m)
    broker.shutdown()

    if df_1m is None or len(df_1m) < 500:
        raise RuntimeError(f"Insufficient Rithmic data for {symbol}: {0 if df_1m is None else len(df_1m)} bars")

    df_1m = df_1m.sort_values("datetime").reset_index(drop=True)
    df_1m["datetime"] = pd.to_datetime(df_1m["datetime"], utc=True)
    start, end = df_1m["datetime"].iloc[0], df_1m["datetime"].iloc[-1]
    print(f"✅ Got {len(df_1m):,} 1m bars  ({start} → {end})")
    return df_1m


def save_csv(symbol: str, df_1m: pd.DataFrame, df_5m: pd.DataFrame, suffix: str = "rithmic") -> None:
    """Save Rithmic fetch without overwriting long-history CSVs."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    p1 = os.path.join(data_dir, f"{symbol}_1m_{suffix}.csv")
    p5 = os.path.join(data_dir, f"{symbol}_5m_{suffix}.csv")
    df_1m.to_csv(p1, index=False)
    df_5m.to_csv(p5, index=False)
    print(f"💾 Saved {p1}")
    print(f"💾 Saved {p5}")


def main():
    parser = argparse.ArgumentParser(description="MTF backtest on Rithmic historical data")
    parser.add_argument("--symbol", default="NQ", choices=["NQ", "MNQ"])
    parser.add_argument("--bars", type=int, default=10000, help="Max 1m bars to request")
    parser.add_argument("--save-only", action="store_true", help="Download and save CSV only")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  Rithmic MTF Backtest — {args.symbol}")
    print(f"{'='*70}\n")

    df_1m = fetch_rithmic_bars(args.symbol, args.bars)
    df_5m = resample_5m(df_1m)
    print(f"   Resampled → {len(df_5m):,} 5m bars")
    save_csv(args.symbol, df_1m, df_5m)

    if args.save_only:
        return

    apply_profit_config()
    bt = mtf.MultiTimeframeBacktester(args.symbol)
    stats = bt.run(df_1m, df_5m)
    mtf.print_results(stats)

    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", f"{args.symbol.lower()}_rithmic_backtest.json")
    summary = {
        "symbol": args.symbol,
        "source": "rithmic",
        "bars_1m": len(df_1m),
        "bars_5m": len(df_5m),
        "start": str(df_1m["datetime"].iloc[0]),
        "end": str(df_1m["datetime"].iloc[-1]),
        **{k: stats[k] for k in ("total_trades", "win_rate", "profit_factor", "total_pnl", "max_drawdown_pct")},
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n📄 Results saved to {out}")


if __name__ == "__main__":
    main()
