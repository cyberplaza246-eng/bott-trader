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
        Generate realistic synthetic forex data with:
        - Close-to-close continuity (open = prev close, no gaps)
        - Slight trend persistence (AR(1) φ ≈ 0.03) matching real forex
        - GARCH-like volatility clustering
        - Session-based volume/volatility (London / NY / Asian)
        - Momentum bursts and mean-reversion regimes
        - Calibrated intraday volatility (3-5 pips/5m for EUR/USD)
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

        np.random.seed(42)

        # ── Calibrated volatility per bar ──────────────────────────
        pip_size = 0.01 if is_jpy else 0.0001
        sigma_5m = 3.5 * pip_size        # ~3.5 pip σ per 5m candle
        candle_sigma = sigma_5m * np.sqrt(tf_min / 5.0)

        if 'GBP' in pair:
            candle_sigma *= 1.3

        # ── Generate close-to-close time series ────────────────────
        # AR(1) process: r_t = φ * r_{t-1} + ε_t  (φ > 0 → trend persistence)
        phi = 0.03               # slight positive serial correlation (real ≈ 0.02-0.05)
        vol = candle_sigma
        prev_return = 0.0
        closes = np.zeros(num_candles)
        closes[0] = base
        vols = np.zeros(num_candles)
        vols[0] = vol

        # ── Regime switching: trending vs ranging ──────────────────
        regime = 'ranging'        # start ranging
        regime_duration = 0
        trend_dir = 0.0           # trend bias when trending

        for i in range(1, num_candles):
            # Regime switch (average 200 bars per regime)
            regime_duration += 1
            if np.random.random() < 1.0 / 200:
                if regime == 'ranging':
                    regime = 'trending'
                    trend_dir = np.random.choice([-1.0, 1.0]) * candle_sigma * 0.15
                else:
                    regime = 'ranging'
                    trend_dir = 0.0
                regime_duration = 0

            # GARCH(1,1) volatility: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
            alpha, beta = 0.08, 0.90   # α + β < 1 for stationarity
            omega = candle_sigma * candle_sigma * (1.0 - alpha - beta)
            vol_sq = omega + alpha * (prev_return ** 2) + beta * (vol ** 2)
            vol = np.sqrt(max(vol_sq, (candle_sigma * 0.2) ** 2))
            vol = min(vol, candle_sigma * 3.5)
            vols[i] = vol

            # AR(1) return with regime drift
            mean_revert = (base - closes[i - 1]) * 0.0003  # very gentle pull to base
            innovation = np.random.normal(0, vol)
            ret = phi * prev_return + mean_revert + trend_dir + innovation
            prev_return = ret
            closes[i] = closes[i - 1] + ret

        # ── Build OHLCV from close series ──────────────────────────
        now = datetime.now()
        timestamps = [now - timedelta(minutes=tf_min * (num_candles - i)) for i in range(num_candles)]

        data = []
        for i in range(num_candles):
            o = closes[i - 1] if i > 0 else closes[0]  # open = previous close
            c = closes[i]
            bar_vol = vols[i] if i > 0 else candle_sigma

            # Session-based volatility scaling
            hour = timestamps[i].hour
            if 8 <= hour < 12:        # London AM
                sess_mult = 1.3
            elif 13 <= hour < 17:     # NY overlap
                sess_mult = 1.4
            elif 0 <= hour < 8:       # Asian
                sess_mult = 0.7
            else:
                sess_mult = 1.0

            wick_vol = bar_vol * sess_mult * 0.5

            # Wicks extend beyond body
            h = max(o, c) + abs(np.random.normal(0, wick_vol))
            l = min(o, c) - abs(np.random.normal(0, wick_vol))

            # Volume
            vol_mult = sess_mult
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
