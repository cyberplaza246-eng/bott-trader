"""
Historical Data Downloader
Downloads forex data from free sources for LSTM training and backtesting.

Sources:
  1. Yahoo Finance (yfinance) - free, reliable
  2. Simulated high-quality data (fallback when no internet/API)
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.logger import bot_logger, error_logger

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')


# Yahoo Finance symbol mapping
YF_SYMBOLS = {
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
    'USD/JPY': 'USDJPY=X',
    'AUD/USD': 'AUDUSD=X',
    'USD/CAD': 'USDCAD=X',
    'NZD/USD': 'NZDUSD=X',
    'USD/CHF': 'USDCHF=X',
}


class HistoricalDownloader:
    """Download and store historical forex data for training"""

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def download(self, pair: str, days: int = 365 * 2, interval: str = '1h') -> pd.DataFrame:
        """
        Download historical data for a pair.

        Args:
            pair:     e.g. 'EUR/USD'
            days:     how many calendar days of history
            interval: candle interval – '1h', '1d', '5m', '15m', '4h'

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume
        """
        csv_path = self._csv_path(pair, interval)

        # Try loading cached data first
        if os.path.exists(csv_path):
            cached = pd.read_csv(csv_path, parse_dates=['datetime'])
            age_days = (datetime.now() - cached['datetime'].iloc[-1]).days
            if age_days < 1:
                bot_logger.info(f"Using cached data for {pair} ({len(cached)} candles)")
                return cached

        # Try Yahoo Finance
        df = self._download_yfinance(pair, days, interval)

        # Fallback: generate high-quality synthetic data
        if df is None or len(df) < 100:
            bot_logger.warning(f"Yahoo Finance unavailable for {pair}, generating synthetic data")
            df = self._generate_training_data(pair, days, interval)

        # Cache to disk
        df.to_csv(csv_path, index=False)
        bot_logger.info(f"Saved {len(df)} candles for {pair} → {csv_path}")

        return df

    def download_all(self, pairs: list, days: int = 365 * 2, interval: str = '1h') -> dict:
        """Download data for multiple pairs."""
        results = {}
        for pair in pairs:
            try:
                results[pair] = self.download(pair, days, interval)
                bot_logger.info(f"✅ {pair}: {len(results[pair])} candles downloaded")
            except Exception as e:
                error_logger.error(f"Failed to download {pair}: {e}")
        return results

    # ------------------------------------------------------------------
    # Yahoo Finance
    # ------------------------------------------------------------------
    def _download_yfinance(self, pair: str, days: int, interval: str) -> pd.DataFrame | None:
        try:
            import yfinance as yf
        except ImportError:
            bot_logger.warning("yfinance not installed – pip install yfinance")
            return None

        symbol = YF_SYMBOLS.get(pair)
        if not symbol:
            bot_logger.warning(f"No Yahoo Finance symbol mapping for {pair}")
            return None

        # yfinance interval mapping
        yf_interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '1h',  # 4h not natively supported – download 1h then resample
            '1d': '1d',
        }
        yf_interval = yf_interval_map.get(interval, '1h')

        # yfinance limits intraday history; adjust period accordingly
        if yf_interval in ('1m', '5m', '15m', '30m'):
            max_days = 60
        elif yf_interval == '1h':
            max_days = 730
        else:
            max_days = days

        actual_days = min(days, max_days)
        start = (datetime.now() - timedelta(days=actual_days)).strftime('%Y-%m-%d')
        end = datetime.now().strftime('%Y-%m-%d')

        try:
            bot_logger.info(f"Downloading {pair} from Yahoo Finance ({yf_interval}, {actual_days}d)...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=yf_interval)

            if df is None or df.empty:
                return None

            df = df.reset_index()

            # Normalise column names
            rename_map = {}
            for col in df.columns:
                cl = col.lower()
                if cl in ('date', 'datetime', 'index'):
                    rename_map[col] = 'datetime'
                elif cl == 'open':
                    rename_map[col] = 'open'
                elif cl == 'high':
                    rename_map[col] = 'high'
                elif cl == 'low':
                    rename_map[col] = 'low'
                elif cl == 'close':
                    rename_map[col] = 'close'
                elif cl == 'volume':
                    rename_map[col] = 'volume'

            df = df.rename(columns=rename_map)
            required = {'datetime', 'open', 'high', 'low', 'close'}
            if not required.issubset(set(df.columns)):
                return None

            if 'volume' not in df.columns:
                df['volume'] = 0

            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
            df = df.dropna()

            # Resample to 4h if requested
            if interval == '4h' and yf_interval == '1h':
                df = df.set_index('datetime')
                df = df.resample('4h').agg({
                    'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last', 'volume': 'sum'
                }).dropna().reset_index()

            return df

        except Exception as e:
            error_logger.error(f"yfinance error for {pair}: {e}")
            return None

    # ------------------------------------------------------------------
    # Synthetic Data Generator (realistic forex prices)
    # ------------------------------------------------------------------
    def _generate_training_data(self, pair: str, days: int, interval: str) -> pd.DataFrame:
        """
        Generate high-quality synthetic forex data with:
        - Trend, mean-reversion, volatility clustering (GARCH-like)
        - Session-based volume patterns (London / NY / Asian sessions)
        - Realistic spread and wicks
        """
        base_prices = {
            'EUR/USD': 1.0850, 'GBP/USD': 1.2650, 'USD/JPY': 150.50,
            'AUD/USD': 0.6550, 'USD/CAD': 1.3600, 'NZD/USD': 0.6100,
            'USD/CHF': 0.8750,
        }
        base = base_prices.get(pair, 1.1000)
        is_jpy = 'JPY' in pair

        interval_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440,
        }
        tf_min = interval_minutes.get(interval, 60)
        candles_per_day = 1440 / tf_min
        num_candles = int(days * candles_per_day)

        np.random.seed(42)  # Reproducible training data

        # Generate prices with volatility clustering
        volatility_base = 0.0008 if not is_jpy else 0.08
        vol_scale = np.sqrt(tf_min / 1440)
        candle_vol = volatility_base * vol_scale

        prices = np.zeros(num_candles)
        prices[0] = base
        vol = candle_vol

        for i in range(1, num_candles):
            # GARCH-like volatility clustering
            vol = 0.94 * vol + 0.06 * candle_vol * abs(np.random.normal(0, 1))
            vol = max(candle_vol * 0.3, min(vol, candle_vol * 3.0))

            # Mean reversion + slight trend
            mean_revert = (base - prices[i - 1]) * 0.002
            trend = np.random.normal(0, vol * 0.05)
            noise = np.random.normal(mean_revert + trend, base * vol)
            prices[i] = prices[i - 1] + noise

        # Build OHLCV
        now = datetime.now()
        timestamps = [now - timedelta(minutes=tf_min * (num_candles - i)) for i in range(num_candles)]

        data = []
        for i in range(num_candles):
            o = prices[i]
            intra_vol = base * candle_vol * 0.5
            c = o + np.random.normal(0, intra_vol * 0.7)
            h = max(o, c) + abs(np.random.normal(0, intra_vol))
            l = min(o, c) - abs(np.random.normal(0, intra_vol))

            # Session-based volume (hour of day)
            hour = timestamps[i].hour
            if 8 <= hour < 16:      # London
                vol_mult = 1.5
            elif 13 <= hour < 21:   # NY overlap
                vol_mult = 1.8
            elif 0 <= hour < 8:     # Asian
                vol_mult = 0.7
            else:
                vol_mult = 1.0

            volume = max(100, int(np.random.lognormal(8, 0.8) * vol_mult))
            decimals = 3 if is_jpy else 5

            data.append({
                'datetime': timestamps[i],
                'open': round(o, decimals),
                'high': round(h, decimals),
                'low': round(l, decimals),
                'close': round(c, decimals),
                'volume': volume,
            })

        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _csv_path(self, pair: str, interval: str) -> str:
        safe_pair = pair.replace('/', '_')
        return os.path.join(DATA_DIR, f'{safe_pair}_{interval}.csv')

    def get_cached_pairs(self) -> list:
        """List pairs that already have cached data."""
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        pairs = []
        for f in files:
            parts = f.replace('.csv', '').rsplit('_', 1)
            if len(parts) == 2:
                pairs.append(parts[0].replace('_', '/'))
        return list(set(pairs))
