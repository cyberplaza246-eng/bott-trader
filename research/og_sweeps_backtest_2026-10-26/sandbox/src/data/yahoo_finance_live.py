"""
Yahoo Finance Live Data — Fallback market data provider.

Used when the primary broker (Rithmic) is unavailable.
Provides candles and latest price via yfinance (free, no API key).

Note: CME futures data through Yahoo Finance has a 15-20 minute delay.
This is acceptable as a temporary fallback but not for primary scalping signals.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd

from src.instruments import get_instrument
from src.utils.logger import bot_logger, error_logger

# Yahoo Finance futures symbol mapping (continuous front-month)
YF_SYMBOLS = {
    "MES": "MES=F",
    "MNQ": "MNQ=F",
    "ES": "ES=F",
    "NQ": "NQ=F",
    "CL": "CL=F",
    "GC": "GC=F",
    # Forex
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
}

# yfinance interval strings and their max lookback days
_YF_INTERVAL_MAP = {
    1: ("1m", 7),
    5: ("5m", 60),
    15: ("15m", 60),
    30: ("30m", 60),
    60: ("1h", 730),
    240: ("1h", 730),   # 4h not native — download 1h, resample
    1440: ("1d", 3650),
}


class YahooFinanceLive:
    """Thin wrapper around yfinance for live fallback data."""

    def get_candles(
        self,
        symbol: str,
        timeframe_minutes: int = 1,
        num_candles: int = 100,
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV candles via yfinance."""
        try:
            import yfinance as yf
        except ImportError:
            error_logger.error("yfinance not installed — pip install yfinance")
            return None

        yf_sym = YF_SYMBOLS.get(symbol)
        if not yf_sym:
            error_logger.error(f"No Yahoo Finance mapping for {symbol}")
            return None

        # Pick interval and max lookback
        interval_str, max_days = _YF_INTERVAL_MAP.get(
            timeframe_minutes, ("1h", 730)
        )

        # Calculate period needed
        bars_per_day = 1440 / max(timeframe_minutes, 1) * 0.4  # ~40% trading hours
        days_needed = int(num_candles / max(bars_per_day, 1)) + 2
        days_needed = min(days_needed, max_days)

        start = (datetime.now() - timedelta(days=days_needed)).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")

        try:
            ticker = yf.Ticker(yf_sym)
            df = ticker.history(start=start, end=end, interval=interval_str)

            if df is None or df.empty:
                return None

            df = df.reset_index()

            # Normalise column names
            rename_map = {}
            for col in df.columns:
                cl = col.lower()
                if cl in ("date", "datetime", "index"):
                    rename_map[col] = "datetime"
                elif cl == "open":
                    rename_map[col] = "open"
                elif cl == "high":
                    rename_map[col] = "high"
                elif cl == "low":
                    rename_map[col] = "low"
                elif cl == "close":
                    rename_map[col] = "close"
                elif cl == "volume":
                    rename_map[col] = "volume"

            df = df.rename(columns=rename_map)
            required = {"datetime", "open", "high", "low", "close"}
            if not required.issubset(set(df.columns)):
                return None

            if "volume" not in df.columns:
                df["volume"] = 0

            df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
            df = df.dropna()

            # Resample to 4h if requested
            if timeframe_minutes == 240 and interval_str == "1h":
                df = df.set_index("datetime")
                df = df.resample("4h").agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }).dropna().reset_index()

            bot_logger.info(
                f"Yahoo Finance fallback: {symbol} {len(df)} candles "
                f"({interval_str})"
            )
            return df.tail(num_candles).reset_index(drop=True)

        except Exception as e:
            error_logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return None

    def get_latest_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get latest price via yfinance, estimate bid/ask from instrument spread."""
        try:
            import yfinance as yf
        except ImportError:
            return None

        yf_sym = YF_SYMBOLS.get(symbol)
        if not yf_sym:
            return None

        try:
            ticker = yf.Ticker(yf_sym)
            info = ticker.fast_info
            price = getattr(info, "last_price", None)
            if price is None or price <= 0:
                # Fallback: get last close
                price = getattr(info, "previous_close", None)
            if price is None or price <= 0:
                return None

            spec = get_instrument(symbol)
            half_spread = spec.spread_default / 2
            return {
                "bid": price - half_spread,
                "ask": price + half_spread,
                "last": price,
            }
        except Exception as e:
            error_logger.error(f"Yahoo Finance price error for {symbol}: {e}")
            return None
