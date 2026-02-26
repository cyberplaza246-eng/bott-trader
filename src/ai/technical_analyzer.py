"""
Technical Analysis Indicators — God Tier v2

Enhanced with:
  - RSI divergence detection (bullish/bearish)
  - Multi-indicator weighted scoring
  - Trend strength composite score
  - Volatility-adjusted signals
  - VWAP proximity analysis
"""
import pandas as pd
import numpy as np
import pandas_ta as pta
from src.utils.logger import bot_logger


class TechnicalAnalyzer:
    """Calculate technical indicators and generate high-quality signals"""

    def __init__(self):
        self.rsi_period = 14
        self.rsi_overbought = 65
        self.rsi_oversold = 35
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.bb_std = 2
        self.atr_period = 14
        self.adx_period = 14
        self.adx_trend_threshold = 15

    def calculate_indicators(self, df):
        """Calculate all technical indicators with divergence detection."""
        df = df.copy()

        # RSI
        df['rsi'] = pta.rsi(df['close'], length=self.rsi_period)

        # MACD
        macd_df = pta.macd(df['close'],
                           fast=self.macd_fast,
                           slow=self.macd_slow,
                           signal=self.macd_signal)
        if macd_df is not None:
            df['macd'] = macd_df.iloc[:, 0]
            df['macd_histogram'] = macd_df.iloc[:, 1]
            df['macd_signal'] = macd_df.iloc[:, 2]
        else:
            df['macd'] = 0.0
            df['macd_histogram'] = 0.0
            df['macd_signal'] = 0.0

        # Bollinger Bands
        bb_df = pta.bbands(df['close'], length=self.bb_period, std=self.bb_std)
        if bb_df is not None:
            df['bb_lower'] = bb_df.iloc[:, 0]
            df['bb_middle'] = bb_df.iloc[:, 1]
            df['bb_upper'] = bb_df.iloc[:, 2]
        else:
            df['bb_lower'] = df['close']
            df['bb_middle'] = df['close']
            df['bb_upper'] = df['close']

        # EMA 200 (macro trend filter)
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

        # EMA 50 (medium trend)
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        # ATR (volatility)
        df['atr'] = pta.atr(df['high'], df['low'], df['close'], length=self.atr_period)

        # ADX (trend strength)
        adx_df = pta.adx(df['high'], df['low'], df['close'], length=self.adx_period)
        if adx_df is not None:
            adx_col = [c for c in adx_df.columns if 'ADX' in c]
            df['adx'] = adx_df[adx_col[0]].values if adx_col else 0.0
            # Also get DI+ and DI- for directional confirmation
            dmp_col = [c for c in adx_df.columns if 'DMP' in c]
            dmn_col = [c for c in adx_df.columns if 'DMN' in c]
            df['di_plus'] = adx_df[dmp_col[0]].values if dmp_col else 0.0
            df['di_minus'] = adx_df[dmn_col[0]].values if dmn_col else 0.0
        else:
            df['adx'] = 0.0
            df['di_plus'] = 0.0
            df['di_minus'] = 0.0

        # Stochastic
        stoch_df = pta.stoch(df['high'], df['low'], df['close'],
                             k=14, d=3, smooth_k=3)
        if stoch_df is not None:
            df['stoch_k'] = stoch_df.iloc[:, 0]
            df['stoch_d'] = stoch_df.iloc[:, 1]
        else:
            df['stoch_k'] = 50.0
            df['stoch_d'] = 50.0

        # RSI Divergence detection
        df['rsi_divergence'] = self._detect_rsi_divergence(df)

        # MACD Histogram momentum (acceleration)
        if 'macd_histogram' in df.columns:
            df['macd_accel'] = df['macd_histogram'].diff()

        # Bollinger Band %B (position within bands)
        bb_range = df['bb_upper'] - df['bb_lower']
        df['bb_pctb'] = (df['close'] - df['bb_lower']) / bb_range.replace(0, np.nan)
        df['bb_pctb'] = df['bb_pctb'].fillna(0.5)

        # Fill NaN values
        df = df.bfill().ffill()

        return df

    def _detect_rsi_divergence(self, df, lookback=14):
        """
        Detect bullish/bearish RSI divergence.

        Bullish divergence: Price makes lower low, RSI makes higher low
        Bearish divergence: Price makes higher high, RSI makes lower high

        Returns: Series with values 'bullish', 'bearish', or 'none'
        """
        divergence = pd.Series(['none'] * len(df), index=df.index)

        if len(df) < lookback * 2 or 'rsi' not in df.columns:
            return divergence

        for i in range(lookback * 2, len(df)):
            window_price = df['close'].iloc[i - lookback:i + 1]
            window_rsi = df['rsi'].iloc[i - lookback:i + 1]

            if window_rsi.isna().any():
                continue

            price_low_idx = window_price.idxmin()
            price_high_idx = window_price.idxmax()

            # Bullish divergence: price at new low but RSI higher
            price_recent = df['close'].iloc[i]
            rsi_recent = df['rsi'].iloc[i]

            # Compare current vs prior swing
            prior_price_low = window_price.min()
            prior_rsi_at_low = df['rsi'].loc[price_low_idx] if price_low_idx in df.index else rsi_recent

            if price_recent <= prior_price_low * 1.001 and rsi_recent > prior_rsi_at_low + 2:
                divergence.iloc[i] = 'bullish'

            # Bearish divergence: price at new high but RSI lower
            prior_price_high = window_price.max()
            prior_rsi_at_high = df['rsi'].loc[price_high_idx] if price_high_idx in df.index else rsi_recent

            if price_recent >= prior_price_high * 0.999 and rsi_recent < prior_rsi_at_high - 2:
                divergence.iloc[i] = 'bearish'

        return divergence

    def get_signal(self, df):
        """
        Generate BUY/SELL signal with weighted multi-indicator scoring.
        """
        if len(df) < self.bb_period:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        # Weighted scoring system
        buy_score = 0.0
        sell_score = 0.0
        reason_parts = []

        # === RSI Analysis (weight: 0.20) ===
        rsi = latest['rsi']
        if rsi < self.rsi_oversold:
            buy_score += 0.20
            reason_parts.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > self.rsi_overbought:
            sell_score += 0.20
            reason_parts.append(f"RSI overbought ({rsi:.1f})")
        elif rsi < 45:
            buy_score += 0.05
        elif rsi > 55:
            sell_score += 0.05

        # === RSI Divergence (weight: 0.15 — high-value signal) ===
        divergence = latest.get('rsi_divergence', 'none')
        if divergence == 'bullish':
            buy_score += 0.15
            reason_parts.append("Bullish RSI divergence 🔥")
        elif divergence == 'bearish':
            sell_score += 0.15
            reason_parts.append("Bearish RSI divergence 🔥")

        # === Bollinger Bands (weight: 0.15) ===
        bb_pctb = latest.get('bb_pctb', 0.5)
        if bb_pctb < 0.0:
            buy_score += 0.15
            reason_parts.append(f"Price below BB lower (%B={bb_pctb:.2f})")
        elif bb_pctb > 1.0:
            sell_score += 0.15
            reason_parts.append(f"Price above BB upper (%B={bb_pctb:.2f})")
        elif bb_pctb < 0.2:
            buy_score += 0.08
            reason_parts.append(f"Near BB lower (%B={bb_pctb:.2f})")
        elif bb_pctb > 0.8:
            sell_score += 0.08
            reason_parts.append(f"Near BB upper (%B={bb_pctb:.2f})")

        # === MACD Analysis (weight: 0.20) ===
        macd_hist = latest['macd_histogram']
        macd_accel = latest.get('macd_accel', 0) or 0

        if latest['macd'] > latest['macd_signal'] and macd_hist > 0:
            buy_score += 0.15
            reason_parts.append("MACD bullish")
            # MACD acceleration bonus
            if macd_accel > 0:
                buy_score += 0.05
                reason_parts.append("MACD accelerating ↑")
        elif latest['macd'] < latest['macd_signal'] and macd_hist < 0:
            sell_score += 0.15
            reason_parts.append("MACD bearish")
            if macd_accel < 0:
                sell_score += 0.05
                reason_parts.append("MACD accelerating ↓")

        # MACD histogram flip (momentum change — very powerful)
        prev_hist = prev.get('macd_histogram', 0) or 0
        if prev_hist < 0 and macd_hist > 0:
            buy_score += 0.10
            reason_parts.append("MACD histogram flipped bullish 🔥")
        elif prev_hist > 0 and macd_hist < 0:
            sell_score += 0.10
            reason_parts.append("MACD histogram flipped bearish 🔥")

        # === Stochastic (weight: 0.15) ===
        stoch_k = latest['stoch_k']
        stoch_d = latest['stoch_d']
        if stoch_k < 20 and stoch_d < 20:
            buy_score += 0.12
            reason_parts.append(f"Stochastic oversold ({stoch_k:.0f}/{stoch_d:.0f})")
            # Stochastic bullish crossover
            if stoch_k > stoch_d:
                buy_score += 0.05
                reason_parts.append("Stoch K crossed above D")
        elif stoch_k > 80 and stoch_d > 80:
            sell_score += 0.12
            reason_parts.append(f"Stochastic overbought ({stoch_k:.0f}/{stoch_d:.0f})")
            if stoch_k < stoch_d:
                sell_score += 0.05
                reason_parts.append("Stoch K crossed below D")

        # === DI+/DI- Directional Index (weight: 0.10) ===
        di_plus = latest.get('di_plus', 0) or 0
        di_minus = latest.get('di_minus', 0) or 0
        if di_plus > di_minus and di_plus > 20:
            buy_score += 0.10
            reason_parts.append(f"DI+ dominant ({di_plus:.0f} vs {di_minus:.0f})")
        elif di_minus > di_plus and di_minus > 20:
            sell_score += 0.10
            reason_parts.append(f"DI- dominant ({di_minus:.0f} vs {di_plus:.0f})")

        # === EMA 50/200 alignment (weight: 0.05) ===
        ema_50 = latest.get('ema_50', latest['close'])
        ema_200 = latest.get('ema_200', latest['close'])
        price = latest['close']
        if price > ema_50 and ema_50 > ema_200:
            buy_score += 0.05
            reason_parts.append("Price > EMA50 > EMA200 (bullish alignment)")
        elif price < ema_50 and ema_50 < ema_200:
            sell_score += 0.05
            reason_parts.append("Price < EMA50 < EMA200 (bearish alignment)")

        # === Determine final signal ===
        net_score = buy_score - sell_score

        if net_score > 0.15:
            signal = 'BUY'
            confidence = min(buy_score, 1.0)
        elif net_score < -0.15:
            signal = 'SELL'
            confidence = min(sell_score, 1.0)
        else:
            signal = 'HOLD'
            confidence = 0.0

        # ADX trend filter
        adx_value = latest.get('adx', 0)
        if signal != 'HOLD' and adx_value < self.adx_trend_threshold:
            reason_parts.append(f"ADX weak ({adx_value:.0f}<{self.adx_trend_threshold})")
            confidence *= 0.6
        elif adx_value >= 25:
            # Strong trend — boost confidence
            confidence *= 1.15
            reason_parts.append(f"ADX strong ({adx_value:.0f})")
        elif adx_value >= self.adx_trend_threshold:
            confidence *= 1.05
            reason_parts.append(f"ADX trending ({adx_value:.0f})")

        confidence = min(abs(confidence), 1.0)
        reason = " | ".join(reason_parts) if reason_parts else "No clear signals"

        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'rsi': latest['rsi'],
            'macd': latest['macd'],
            'bb_position': bb_pctb,
            'adx': adx_value,
            'divergence': divergence,
            'buy_score': buy_score,
            'sell_score': sell_score,
        }
