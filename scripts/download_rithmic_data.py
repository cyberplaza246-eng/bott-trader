#!/usr/bin/env python
"""
Download historical OHLCV data from Rithmic and save to CSV.

Run this on your LIVE machine (Windows) where Rithmic credentials are configured.
It will download 1M and 5M bars, save them to data/, then optionally run the
MTF backtest on the fresh data.

Usage:
    python scripts/download_rithmic_data.py --symbols MES MNQ --days 90
    python scripts/download_rithmic_data.py --symbols MES MNQ --days 90 --backtest
"""

import os
import sys
import asyncio
import argparse
import time as _time
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


async def download_bars(symbol: str, timeframe_minutes: int, days: int) -> pd.DataFrame:
    """Download historical bars from Rithmic in chunks.

    Rithmic limits history requests, so we fetch in day-sized chunks
    and concatenate.
    """
    from src.broker.rithmic_connector import RithmicConnector

    broker = RithmicConnector()

    if not broker._connected:
        print(f"❌ Failed to connect to Rithmic")
        return None

    print(f"✅ Connected to Rithmic ({broker._system})")

    # Access the async client through the broker's event loop
    client = broker._client
    if client is None:
        print("❌ Rithmic client not initialised")
        return None

    from async_rithmic import TimeBarType

    # Resolve front-month contract
    rith_sym = broker._resolved_contracts.get(symbol)
    if not rith_sym:
        # Fallback to raw resolve
        rith_sym, _ = broker._resolve_symbol(symbol)
    exchange = "CME"

    bar_type = TimeBarType.MINUTE_BAR
    period = timeframe_minutes

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    # Fetch in chunks (7-day windows) to avoid Rithmic limits
    chunk_days = 7
    all_rows = []
    current_start = start_time

    print(f"📥 Downloading {symbol} {timeframe_minutes}m bars: "
          f"{start_time.strftime('%Y-%m-%d')} → {end_time.strftime('%Y-%m-%d')} "
          f"({days} days)")

    chunk_num = 0
    total_chunks = (days + chunk_days - 1) // chunk_days

    while current_start < end_time:
        chunk_end = min(current_start + timedelta(days=chunk_days), end_time)
        chunk_num += 1

        print(f"   [{chunk_num}/{total_chunks}] "
              f"{current_start.strftime('%m/%d')} - {chunk_end.strftime('%m/%d')}...",
              end=" ", flush=True)

        try:
            # Use broker's serialised history method to avoid lock issues
            bars = await asyncio.wait_for(
                client.get_historical_time_bars(
                    symbol=rith_sym,
                    exchange=exchange,
                    start_time=current_start,
                    end_time=chunk_end,
                    bar_type=bar_type,
                    bar_type_periods=period,
                ),
                timeout=30.0,
            )

            if bars:
                for b in bars:
                    all_rows.append({
                        "datetime": b.get("bar_end_datetime", b.get("datetime")),
                        "open": float(b.get("open_price", 0)),
                        "high": float(b.get("high_price", 0)),
                        "low": float(b.get("low_price", 0)),
                        "close": float(b.get("close_price", 0)),
                        "volume": int(b.get("volume", 0)),
                    })
                print(f"{len(bars)} bars")
            else:
                print("0 bars (no data)")

        except asyncio.TimeoutError:
            print("TIMEOUT - skipping chunk")
        except Exception as e:
            print(f"ERROR: {e}")

        # Small delay between requests to avoid rate limiting
        await asyncio.sleep(1.0)
        current_start = chunk_end

    if not all_rows:
        print(f"❌ No data downloaded for {symbol} {timeframe_minutes}m")
        return None

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    # Remove zero-price rows
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]

    print(f"✅ {symbol} {timeframe_minutes}m: {len(df)} total bars "
          f"({df['datetime'].iloc[0].strftime('%Y-%m-%d')} → "
          f"{df['datetime'].iloc[-1].strftime('%Y-%m-%d')})")

    return df.reset_index(drop=True)


def save_csv(df: pd.DataFrame, symbol: str, timeframe: int):
    """Save dataframe to data/ directory, backing up the old file."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)

    filename = f"{symbol}_{timeframe}m.csv"
    filepath = os.path.join(data_dir, filename)

    # Backup existing file
    if os.path.exists(filepath):
        backup = filepath + f".bak.pre_rithmic_{datetime.now().strftime('%Y%m%d')}"
        os.rename(filepath, backup)
        print(f"   📦 Backed up old {filename} → {os.path.basename(backup)}")

    df.to_csv(filepath, index=False)
    print(f"   💾 Saved {len(df)} rows → {filepath}")


async def main_async(symbols: list, days: int):
    """Download all requested data."""
    results = {}

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  {symbol}")
        print(f"{'='*60}")

        # Download 1-minute bars
        df_1m = await download_bars(symbol, 1, days)
        if df_1m is not None and len(df_1m) > 0:
            save_csv(df_1m, symbol, 1)
            results[f"{symbol}_1m"] = len(df_1m)

            # Generate 5-minute bars from 1-minute data (more reliable than
            # separate 5m download which can have gaps)
            print(f"\n   📊 Resampling {symbol} 1m → 5m...")
            df_5m = df_1m.set_index("datetime").resample("5min").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna().reset_index()
            save_csv(df_5m, symbol, 5)
            results[f"{symbol}_5m"] = len(df_5m)
        else:
            print(f"   ❌ Skipping {symbol} — no 1m data")

    return results


def main():
    parser = argparse.ArgumentParser(description="Download Rithmic historical data")
    parser.add_argument("--symbols", nargs="+", default=["MES", "MNQ"],
                        help="Symbols to download (default: MES MNQ)")
    parser.add_argument("--days", type=int, default=90,
                        help="Days of history to download (default: 90)")
    parser.add_argument("--backtest", action="store_true",
                        help="Run MTF backtest after download")
    args = parser.parse_args()

    print("=" * 60)
    print("  Rithmic Historical Data Downloader")
    print(f"  Symbols: {', '.join(args.symbols)}")
    print(f"  Days:    {args.days}")
    print("=" * 60)

    results = asyncio.run(main_async(args.symbols, args.days))

    if not results:
        print("\n❌ No data downloaded. Check Rithmic connection.")
        return

    print(f"\n{'='*60}")
    print("  DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for key, count in results.items():
        print(f"   {key}: {count:,} bars")

    if args.backtest:
        print(f"\n{'='*60}")
        print("  RUNNING BACKTESTS ON FRESH DATA")
        print(f"{'='*60}")
        import subprocess
        for symbol in args.symbols:
            print(f"\n--- {symbol} ---")
            subprocess.run([
                sys.executable, "scripts/backtest_mtf_scalping.py",
                "--symbol", symbol
            ])


if __name__ == "__main__":
    main()
