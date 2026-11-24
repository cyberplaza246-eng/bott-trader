"""
Clean Scalping Strategy - Simple, Proven Framework

Based on:
1. EMA Trend Filter (9/21/200)
2. RSI Momentum (40/60 zones)
3. MACD Confirmation
4. Volume Confirmation
5. ATR Volatility Filter
6. Location-based edge (S/R, sweeps, VWAP)

Multi-timeframe: 5m for trend, 1m for entry
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger('CleanScalper')


class CleanScalper:
    """Simple, effective scalping strategy with clear rules."""

    # EMA Settings
    EMA_FAST = 9
    EMA_MEDIUM = 21
    EMA_TREND = 200

    # RSI Settings (tighter for scalping)
    RSI_LENGTH = 14
    RSI_OVERSOLD = 40  # Not 30 - too slow for scalping
    RSI_OVERBOUGHT = 60  # Not 70 - too slow for scalping

    # MACD Settings
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9

    # Volume Settings
    VOLUME_MA_LENGTH = 20

    # ATR Settings
    ATR_LENGTH = 14
    ATR_MA_LENGTH = 20

    # Session windows (EST converted to UTC)
    # 9:30-11:30 EST = 14:30-16:30 UTC
    # 2:30-4:00 PM EST = 19:30-21:00 UTC
    SESSION_WINDOWS = [
        {'start': 14, 'end': 17},  # Morning session (14:30-16:30 UTC, extended slightly)
        {'start': 19, 'end': 21},  # Afternoon session (19:30-21:00 UTC)
    ]

    # Minimum confirmations required (out of 6)
    MIN_CONFIRMATIONS = 3.5  # Optimized
    
    # Sweep gate - when True, only trade on sweep rejections
    REQUIRE_SWEEP = False  # Used as bonus instead

    # SL/TP in ticks - OPTIMIZED SETTINGS
    INSTRUMENT_CONFIG = {
        'MES': {
            'tick_size': 0.25,
            'tick_value': 1.25,  # $1.25 per tick
            'sl_ticks': 12,      # Optimized
            'tp_ticks': 30,      # 2.5 R:R - best for MES
        },
        'MNQ': {
            'tick_size': 0.25,
            'tick_value': 0.50,  # $0.50 per tick
            'sl_ticks': 30,      # Optimized
            'tp_ticks': 60,      # 2.0 R:R - best for MNQ
        },
    }

    def __init__(self, min_confirmations: int = None):
        if min_confirmations is not None:
            self.MIN_CONFIRMATIONS = min_confirmations

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators."""
        df = df.copy()

        # EMAs
        df['ema_9'] = df['close'].ewm(span=self.EMA_FAST, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=self.EMA_MEDIUM, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=self.EMA_TREND, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.RSI_LENGTH).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.RSI_LENGTH).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_prev'] = df['rsi'].shift(1)

        # MACD
        ema_fast = df['close'].ewm(span=self.MACD_FAST, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.MACD_SLOW, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.MACD_SIGNAL, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_prev'] = df['macd'].shift(1)
        df['macd_signal_prev'] = df['macd_signal'].shift(1)
        df['macd_hist_prev'] = df['macd_hist'].shift(1)

        # Volume
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=self.VOLUME_MA_LENGTH).mean()
        else:
            df['volume_ma'] = 1  # Fallback

        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=self.ATR_LENGTH).mean()
        df['atr_ma'] = df['atr'].rolling(window=self.ATR_MA_LENGTH).mean()

        # Swing highs/lows for location edge (simple 5-bar lookback)
        df['swing_high'] = df['high'].rolling(window=5, center=True).max()
        df['swing_low'] = df['low'].rolling(window=5, center=True).min()
        
        # Recent swing levels (last 20 bars)
        df['recent_high'] = df['high'].rolling(window=20).max()
        df['recent_low'] = df['low'].rolling(window=20).min()

        return df

    def check_trend_bias(self, row: pd.Series) -> Tuple[str, bool]:
        """
        Check trend direction using EMAs.
        
        Returns:
            (bias: 'LONG'/'SHORT'/'NEUTRAL', ema_cross: bool)
        """
        close = row['close']
        ema_9 = row['ema_9']
        ema_21 = row['ema_21']
        ema_200 = row['ema_200']

        # Check EMA alignment
        price_above_200 = close > ema_200
        price_below_200 = close < ema_200
        fast_above_med = ema_9 > ema_21
        fast_below_med = ema_9 < ema_21

        if price_above_200 and fast_above_med:
            return 'LONG', True
        elif price_below_200 and fast_below_med:
            return 'SHORT', True
        else:
            return 'NEUTRAL', False

    def check_rsi_signal(self, row: pd.Series) -> str:
        """
        Check RSI momentum signal.
        
        Returns: 'LONG'/'SHORT'/'NEUTRAL'
        """
        rsi = row['rsi']
        rsi_prev = row.get('rsi_prev', rsi)

        if pd.isna(rsi_prev):
            rsi_prev = rsi

        # RSI crosses ABOVE 40 → Long
        if rsi_prev <= self.RSI_OVERSOLD < rsi:
            return 'LONG'
        # RSI crosses BELOW 60 → Short
        if rsi_prev >= self.RSI_OVERBOUGHT > rsi:
            return 'SHORT'
        # Also count if RSI is in favorable zone (relaxed)
        if rsi > self.RSI_OVERSOLD and rsi < 55:  # Not overbought but above oversold
            return 'LONG_ZONE'
        if rsi < self.RSI_OVERBOUGHT and rsi > 45:  # Not oversold but below overbought
            return 'SHORT_ZONE'

        return 'NEUTRAL'

    def check_macd_signal(self, row: pd.Series) -> str:
        """
        Check MACD momentum confirmation.
        
        Returns: 'LONG'/'SHORT'/'NEUTRAL'
        """
        macd = row['macd']
        signal = row['macd_signal']
        hist = row['macd_hist']
        macd_prev = row.get('macd_prev', macd)
        signal_prev = row.get('macd_signal_prev', signal)
        hist_prev = row.get('macd_hist_prev', hist)

        if pd.isna(macd_prev):
            macd_prev = macd
        if pd.isna(signal_prev):
            signal_prev = signal
        if pd.isna(hist_prev):
            hist_prev = hist

        # Bullish: MACD crosses above signal AND histogram turns positive
        macd_cross_up = macd_prev <= signal_prev and macd > signal
        hist_turns_pos = hist_prev <= 0 < hist

        # Bearish: MACD crosses below signal AND histogram turns negative
        macd_cross_down = macd_prev >= signal_prev and macd < signal
        hist_turns_neg = hist_prev >= 0 > hist

        if macd_cross_up or (macd > signal and hist_turns_pos):
            return 'LONG'
        if macd_cross_down or (macd < signal and hist_turns_neg):
            return 'SHORT'
        
        # Relaxed: just check direction
        if macd > signal and hist > 0:
            return 'LONG_ZONE'
        if macd < signal and hist < 0:
            return 'SHORT_ZONE'

        return 'NEUTRAL'

    def check_volume(self, row: pd.Series) -> bool:
        """Check if volume is above average."""
        if 'volume' not in row.index:
            return True  # Skip check if no volume data
        
        volume = row.get('volume', 0)
        volume_ma = row.get('volume_ma', 1)
        
        if pd.isna(volume) or pd.isna(volume_ma) or volume_ma == 0:
            return True
            
        return volume > volume_ma

    def check_atr_volatility(self, row: pd.Series) -> bool:
        """Check if ATR is above average (market is active)."""
        atr = row.get('atr', 0)
        atr_ma = row.get('atr_ma', 0)
        
        if pd.isna(atr) or pd.isna(atr_ma) or atr_ma == 0:
            return True
            
        return atr > atr_ma

    def check_session(self, candle_hour: int) -> bool:
        """Check if current hour is in active trading session."""
        if candle_hour is None:
            return True  # Allow if no time info
            
        for window in self.SESSION_WINDOWS:
            if window['start'] <= candle_hour < window['end']:
                return True
        return False

    def check_location_edge(self, row: pd.Series, direction: str) -> bool:
        """
        Check if price is at a significant level (location edge).
        
        Near recent highs/lows improves signal quality.
        """
        close = row['close']
        atr = row.get('atr', 0)
        
        if pd.isna(atr) or atr == 0:
            return False
            
        recent_high = row.get('recent_high', close)
        recent_low = row.get('recent_low', close)
        
        if pd.isna(recent_high) or pd.isna(recent_low):
            return False
        
        # Near recent low for longs (pullback entry)
        if direction == 'LONG':
            distance_to_low = close - recent_low
            if distance_to_low < atr * 1.5:  # Within 1.5 ATR of recent low
                return True
                
        # Near recent high for shorts (pullback entry)
        if direction == 'SHORT':
            distance_to_high = recent_high - close
            if distance_to_high < atr * 1.5:  # Within 1.5 ATR of recent high
                return True
                
        return False

    def detect_sweep(self, df: pd.DataFrame) -> dict:
        """
        Detect if price swept a recent swing high/low and rejected.
        This is the KEY edge - trading at liquidity levels.
        
        Returns:
            dict with 'detected', 'direction', 'swept_level', 'rejection'
        """
        result = {
            'detected': False,
            'direction': None,
            'swept_level': None,
            'rejection': False,
        }
        
        if len(df) < 25:
            return result
            
        # Use last bar
        current = df.iloc[-1]
        close = current['close']
        high = current['high']
        low = current['low']
        open_price = current['open']
        
        # Look for swing levels in recent history (excluding current bar)
        lookback = df.iloc[-25:-1]
        
        if len(lookback) < 20:
            return result
            
        # Find swing highs/lows (local extremes)
        highs = lookback['high'].values
        lows = lookback['low'].values
        
        # Simple swing detection - highest high and lowest low in past 20 bars
        swing_high = lookback['high'].max()
        swing_low = lookback['low'].min()
        
        atr = current.get('atr', 0)
        if pd.isna(atr) or atr <= 0:
            atr = (high - low) * 2  # Fallback
        
        # Check for bearish sweep (high takes out swing high then rejects)
        # Current candle high > swing high AND close < swing high
        if high > swing_high and close < swing_high:
            # Rejection: upper wick should be significant
            upper_wick = high - max(close, open_price)
            body = abs(close - open_price)
            if upper_wick > body * 0.5:  # Good rejection
                result['detected'] = True
                result['direction'] = 'SELL'
                result['swept_level'] = swing_high
                result['rejection'] = True
                return result
        
        # Check for bullish sweep (low takes out swing low then rejects)
        # Current candle low < swing low AND close > swing low
        if low < swing_low and close > swing_low:
            # Rejection: lower wick should be significant
            lower_wick = min(close, open_price) - low
            body = abs(close - open_price)
            if lower_wick > body * 0.5:  # Good rejection
                result['detected'] = True
                result['direction'] = 'BUY'
                result['swept_level'] = swing_low
                result['rejection'] = True
                return result
        
        return result

    def get_signal(self, df_1m: pd.DataFrame, pair: str, 
                   df_5m: Optional[pd.DataFrame] = None,
                   candle_hour: Optional[int] = None,
                   precalculated: bool = False) -> Dict:
        """
        Generate trading signal using multi-timeframe analysis.
        
        Args:
            df_1m: 1-minute DataFrame (entry timeframe)
            pair: Instrument symbol
            df_5m: 5-minute DataFrame (trend timeframe)
            candle_hour: Current hour (UTC)
            precalculated: If True, skip indicator calculation
            
        Returns:
            dict with signal, confidence, sl_tp, details
        """
        result = {
            'signal': 'SKIP',
            'confidence': 0.0,
            'confirmations': 0,
            'details': '',
            'sl_tp': None,
        }

        # Need sufficient data
        if len(df_1m) < 210:
            result['details'] = 'Insufficient 1m data'
            return result

        # Calculate indicators on 1m (skip if precalculated)
        if not precalculated or 'ema_9' not in df_1m.columns:
            df_1m = self.calculate_indicators(df_1m)
        row_1m = df_1m.iloc[-1]

        # ═══════════════════════════════════════════════════════════
        # SWEEP GATE - The real edge is location
        # ═══════════════════════════════════════════════════════════
        sweep = self.detect_sweep(df_1m)
        if self.REQUIRE_SWEEP and not sweep['detected']:
            result['details'] = 'No sweep detected'
            return result
        
        sweep_direction = sweep.get('direction')  # BUY or SELL

        # Calculate indicators on 5m if available (trend timeframe)
        if df_5m is not None and len(df_5m) >= 210:
            if not precalculated or 'ema_9' not in df_5m.columns:
                df_5m = self.calculate_indicators(df_5m)
            row_5m = df_5m.iloc[-1]
            use_5m_trend = True
        else:
            row_5m = row_1m  # Fallback
            use_5m_trend = False

        # ═══════════════════════════════════════════════════════════
        # Check confirmations
        # ═══════════════════════════════════════════════════════════
        confirmations = {
            'trend': None,
            'rsi': None,
            'macd': None,
            'volume': False,
            'atr': False,
            'location': False,
            'session': False,
        }

        # 1. Trend Filter (use 5m for trend)
        trend_bias, ema_cross = self.check_trend_bias(row_5m)
        confirmations['trend'] = trend_bias

        # 2. RSI Momentum (use 1m for entry timing)
        rsi_signal = self.check_rsi_signal(row_1m)
        confirmations['rsi'] = rsi_signal

        # 3. MACD Confirmation (use 1m)
        macd_signal = self.check_macd_signal(row_1m)
        confirmations['macd'] = macd_signal

        # 4. Volume (use 1m)
        confirmations['volume'] = self.check_volume(row_1m)

        # 5. ATR Volatility (use 1m)
        confirmations['atr'] = self.check_atr_volatility(row_1m)

        # 6. Session check
        confirmations['session'] = self.check_session(candle_hour)

        # ═══════════════════════════════════════════════════════════
        # Count confirmations for each direction
        # ═══════════════════════════════════════════════════════════
        long_confirms = 0
        short_confirms = 0

        # Trend is the PRIMARY filter - must be present
        if trend_bias == 'LONG':
            long_confirms += 2.0  # Trend is mandatory
        elif trend_bias == 'SHORT':
            short_confirms += 2.0  # Trend is mandatory

        # RSI (only count strong crosses, not zones)
        if rsi_signal == 'LONG':
            long_confirms += 1.0
        elif rsi_signal == 'LONG_ZONE':
            long_confirms += 0.3  # Weak bonus for zone
        if rsi_signal == 'SHORT':
            short_confirms += 1.0
        elif rsi_signal == 'SHORT_ZONE':
            short_confirms += 0.3  # Weak bonus for zone

        # MACD (only count strong crossovers)
        if macd_signal == 'LONG':
            long_confirms += 1.0
        elif macd_signal == 'LONG_ZONE':
            long_confirms += 0.3
        if macd_signal == 'SHORT':
            short_confirms += 1.0
        elif macd_signal == 'SHORT_ZONE':
            short_confirms += 0.3

        # Sweep detection - major bonus for location edge
        if sweep['detected']:
            if sweep['direction'] == 'BUY':
                long_confirms += 1.5
            elif sweep['direction'] == 'SELL':
                short_confirms += 1.5
            confirmations['sweep'] = sweep['direction']

        # Volume & ATR (add to both if active, neutral otherwise)
        if confirmations['volume']:
            long_confirms += 0.5
            short_confirms += 0.5
        if confirmations['atr']:
            long_confirms += 0.5
            short_confirms += 0.5

        # Location edge (bonus for good entries)
        if self.check_location_edge(row_1m, 'LONG'):
            long_confirms += 0.5
            confirmations['location'] = True
        elif self.check_location_edge(row_1m, 'SHORT'):
            short_confirms += 0.5
            confirmations['location'] = True

        # Pullback detection - key for good entries
        # Long: price should be pulling back toward EMA21 (within 1 ATR)
        # Short: price should be bouncing up toward EMA21 (within 1 ATR)
        atr = row_1m.get('atr', 0)
        if not pd.isna(atr) and atr > 0:
            price_to_ema21 = row_1m['close'] - row_1m['ema_21']
            # For LONG: want price near or just above EMA21 (pullback in uptrend)
            if abs(price_to_ema21) < atr * 1.5:  # Close to EMA21
                if price_to_ema21 >= 0:  # Price at or above EMA21
                    long_confirms += 0.5
                    confirmations['pullback'] = 'LONG'
                else:  # Price just below EMA21
                    short_confirms += 0.5
                    confirmations['pullback'] = 'SHORT'

        # Session (bonus)
        if confirmations['session']:
            long_confirms += 0.3
            short_confirms += 0.3

        # ═══════════════════════════════════════════════════════════
        # Determine signal
        # ═══════════════════════════════════════════════════════════
        direction = None
        confirm_count = 0

        if long_confirms >= self.MIN_CONFIRMATIONS and long_confirms > short_confirms:
            direction = 'BUY'
            confirm_count = long_confirms
        elif short_confirms >= self.MIN_CONFIRMATIONS and short_confirms > long_confirms:
            direction = 'SELL'
            confirm_count = short_confirms
        else:
            result['details'] = (
                f"Insufficient confirms: LONG={long_confirms:.1f} SHORT={short_confirms:.1f} "
                f"(need {self.MIN_CONFIRMATIONS})"
            )
            return result

        # When sweep is required, direction must match sweep
        if self.REQUIRE_SWEEP and sweep_direction and direction != sweep_direction:
            result['details'] = (
                f"Direction mismatch: sweep={sweep_direction}, indicators={direction}"
            )
            return result

        # ═══════════════════════════════════════════════════════════
        # Calculate confidence
        # ═══════════════════════════════════════════════════════════
        # Base confidence from confirmation count
        confidence = min(0.4 + (confirm_count - self.MIN_CONFIRMATIONS) * 0.15, 0.85)

        # Bonus for strong signals
        if ema_cross:
            confidence += 0.05
        if confirmations['location']:
            confidence += 0.05
        if confirmations['session']:
            confidence += 0.03

        # Penalty for counter-trend (RSI/MACD against trend)
        if direction == 'BUY' and trend_bias == 'SHORT':
            confidence -= 0.15
        elif direction == 'SELL' and trend_bias == 'LONG':
            confidence -= 0.15

        confidence = max(0.0, min(1.0, confidence))

        # ═══════════════════════════════════════════════════════════
        # Calculate SL/TP
        # ═══════════════════════════════════════════════════════════
        config = self.INSTRUMENT_CONFIG.get(pair, self.INSTRUMENT_CONFIG['MES'])
        tick_size = config['tick_size']
        sl_ticks = config['sl_ticks']
        tp_ticks = config['tp_ticks']

        entry_price = float(row_1m['close'])
        sl_distance = sl_ticks * tick_size
        tp_distance = tp_ticks * tick_size

        if direction == 'BUY':
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        # ═══════════════════════════════════════════════════════════
        # Build result
        # ═══════════════════════════════════════════════════════════
        result['signal'] = direction
        result['confidence'] = confidence
        result['confirmations'] = confirm_count
        result['sl_tp'] = {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'sl_ticks': sl_ticks,
            'tp_ticks': tp_ticks,
            'risk_reward': tp_ticks / sl_ticks,
        }

        trend_label = f"5m" if use_5m_trend else "1m"
        result['details'] = (
            f"✅ {direction} | Confirms={confirm_count:.1f} | "
            f"Trend({trend_label})={trend_bias} | RSI={rsi_signal} | MACD={macd_signal} | "
            f"Vol={'✓' if confirmations['volume'] else '✗'} | "
            f"ATR={'✓' if confirmations['atr'] else '✗'} | "
            f"Loc={'✓' if confirmations['location'] else '✗'} | "
            f"Sess={'✓' if confirmations['session'] else '✗'}"
        )

        return result


def run_backtest(df_5m: pd.DataFrame, df_1m: pd.DataFrame, pair: str,
                 initial_balance: float = 50000,
                 min_confirmations: int = 3,
                 max_contracts: int = 2) -> Dict:
    """
    Simple backtest of the clean scalper strategy.
    
    Args:
        df_5m: 5-minute OHLCV data
        df_1m: 1-minute OHLCV data
        pair: Instrument symbol
        initial_balance: Starting balance
        min_confirmations: Min confirmations for entry
        max_contracts: Max position size
        
    Returns:
        dict with results
    """
    scalper = CleanScalper(min_confirmations=min_confirmations)
    config = scalper.INSTRUMENT_CONFIG.get(pair, scalper.INSTRUMENT_CONFIG['MES'])
    tick_value = config['tick_value']
    tick_size = config['tick_size']
    commission_rt = 0.62  # Per contract round-trip

    balance = initial_balance
    trades = []
    equity_curve = [balance]
    
    # Ensure datetime columns and pre-calculate indicators ONCE
    df_5m = df_5m.copy()
    df_1m = df_1m.copy()
    if 'datetime' in df_5m.columns:
        df_5m['datetime'] = pd.to_datetime(df_5m['datetime'])
    if 'datetime' in df_1m.columns:
        df_1m['datetime'] = pd.to_datetime(df_1m['datetime'])
    
    # Pre-calculate all indicators (much faster)
    df_5m = scalper.calculate_indicators(df_5m)
    df_1m = scalper.calculate_indicators(df_1m)

    open_position = None
    cooldown_until = 0
    max_hold_bars = 60  # Max hold = 60 1m bars = 1 hour

    # Iterate through 1m bars
    lookback = 250
    
    for idx in range(lookback, len(df_1m)):
        row = df_1m.iloc[idx]
        candle_dt = row['datetime'] if 'datetime' in df_1m.columns else None
        candle_hour = candle_dt.hour if candle_dt else None

        # Handle open position
        if open_position:
            bars_held = idx - open_position['entry_idx']
            
            # Check SL/TP
            if open_position['type'] == 'BUY':
                hit_sl = row['low'] <= open_position['stop_loss']
                hit_tp = row['high'] >= open_position['take_profit']
            else:
                hit_sl = row['high'] >= open_position['stop_loss']
                hit_tp = row['low'] <= open_position['take_profit']

            timed_out = bars_held >= max_hold_bars

            exit_price = None
            exit_type = None

            if hit_sl and hit_tp:
                # Determine which hit first based on candle direction
                is_bullish = row['close'] >= row['open']
                if open_position['type'] == 'BUY':
                    exit_price = open_position['stop_loss'] if is_bullish else open_position['take_profit']
                    exit_type = 'SL' if is_bullish else 'TP'
                else:
                    exit_price = open_position['take_profit'] if is_bullish else open_position['stop_loss']
                    exit_type = 'TP' if is_bullish else 'SL'
            elif hit_sl:
                exit_price = open_position['stop_loss']
                exit_type = 'SL'
            elif hit_tp:
                exit_price = open_position['take_profit']
                exit_type = 'TP'
            elif timed_out:
                exit_price = row['close']
                exit_type = 'TIMEOUT'

            if exit_price:
                # Calculate P&L
                if open_position['type'] == 'BUY':
                    ticks_pnl = (exit_price - open_position['entry_price']) / tick_size
                else:
                    ticks_pnl = (open_position['entry_price'] - exit_price) / tick_size

                contracts = open_position['contracts']
                gross_pnl = ticks_pnl * tick_value * contracts
                commission = commission_rt * contracts
                net_pnl = gross_pnl - commission

                balance += net_pnl
                equity_curve.append(balance)

                trades.append({
                    'entry_idx': open_position['entry_idx'],
                    'exit_idx': idx,
                    'type': open_position['type'],
                    'entry_price': open_position['entry_price'],
                    'exit_price': exit_price,
                    'exit_type': exit_type,
                    'contracts': contracts,
                    'ticks': ticks_pnl,
                    'gross_pnl': gross_pnl,
                    'commission': commission,
                    'net_pnl': net_pnl,
                    'bars_held': bars_held,
                })

                open_position = None
                cooldown_until = idx + 5  # 5-bar cooldown
                continue

        # Skip if in cooldown or already in position
        if idx < cooldown_until or open_position:
            continue

        # Get 5m subset - use simple index mapping (5m bars = 1m idx // 5)
        # This is approximate but much faster than datetime matching
        df_5m_subset = None
        if len(df_5m) > 210:
            # Approximate 5m index from 1m index
            approx_5m_idx = min(idx // 5, len(df_5m) - 1)
            start_5m = max(0, approx_5m_idx - 250)
            df_5m_subset = df_5m.iloc[start_5m:approx_5m_idx + 1]
            if len(df_5m_subset) < 210:
                df_5m_subset = None

        # Get 1m subset
        df_1m_subset = df_1m.iloc[max(0, idx - 250):idx + 1]

        # Generate signal (indicators already pre-calculated)
        signal = scalper.get_signal(
            df_1m_subset, pair,
            df_5m=df_5m_subset,
            candle_hour=candle_hour,
            precalculated=True
        )

        if signal['signal'] not in ('BUY', 'SELL'):
            continue

        # Open position
        sl_tp = signal['sl_tp']
        contracts = min(max_contracts, max(1, int(balance * 0.01 / (config['sl_ticks'] * tick_value))))
        contracts = max(1, min(contracts, max_contracts))

        open_position = {
            'entry_idx': idx,
            'entry_price': sl_tp['entry_price'],
            'stop_loss': sl_tp['stop_loss'],
            'take_profit': sl_tp['take_profit'],
            'type': signal['signal'],
            'contracts': contracts,
            'confidence': signal['confidence'],
        }

    # Close any remaining position
    if open_position:
        last_row = df_1m.iloc[-1]
        exit_price = last_row['close']
        if open_position['type'] == 'BUY':
            ticks_pnl = (exit_price - open_position['entry_price']) / tick_size
        else:
            ticks_pnl = (open_position['entry_price'] - exit_price) / tick_size
        
        contracts = open_position['contracts']
        gross_pnl = ticks_pnl * tick_value * contracts
        commission = commission_rt * contracts
        net_pnl = gross_pnl - commission
        balance += net_pnl
        equity_curve.append(balance)
        
        trades.append({
            'entry_idx': open_position['entry_idx'],
            'exit_idx': len(df_1m) - 1,
            'type': open_position['type'],
            'entry_price': open_position['entry_price'],
            'exit_price': exit_price,
            'exit_type': 'EOD',
            'contracts': contracts,
            'ticks': ticks_pnl,
            'gross_pnl': gross_pnl,
            'commission': commission,
            'net_pnl': net_pnl,
            'bars_held': len(df_1m) - 1 - open_position['entry_idx'],
        })

    # Calculate stats
    total_trades = len(trades)
    winners = [t for t in trades if t['net_pnl'] > 0]
    losers = [t for t in trades if t['net_pnl'] <= 0]
    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

    gross_profit = sum(t['gross_pnl'] for t in winners)
    gross_loss = sum(t['gross_pnl'] for t in losers)
    total_commission = sum(t['commission'] for t in trades)
    net_pnl = balance - initial_balance

    # Max drawdown
    peak = initial_balance
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    pf = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')

    return {
        'trades': trades,
        'total_trades': total_trades,
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': win_rate,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'commission': total_commission,
        'net_pnl': net_pnl,
        'final_balance': balance,
        'max_drawdown_pct': max_dd,
        'profit_factor': pf,
        'equity_curve': equity_curve,
    }


if __name__ == '__main__':
    import sys
    
    pair = sys.argv[1] if len(sys.argv) > 1 else 'MES'
    
    df_5m = pd.read_csv(f'data/{pair}_5m.csv', parse_dates=['datetime'])
    df_1m = pd.read_csv(f'data/{pair}_1m.csv', parse_dates=['datetime'])
    
    print(f"Running clean scalper backtest on {pair}...")
    print(f"5m candles: {len(df_5m)}, 1m candles: {len(df_1m)}")
    
    results = run_backtest(df_5m, df_1m, pair, min_confirmations=3)
    
    print(f"\n{'='*50}")
    print(f"RESULTS: {pair}")
    print(f"{'='*50}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"Net P&L: ${results['net_pnl']:.2f}")
    print(f"Gross Profit: ${results['gross_profit']:.2f}")
    print(f"Gross Loss: ${results['gross_loss']:.2f}")
    print(f"Commission: ${results['commission']:.2f}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Final Balance: ${results['final_balance']:.2f}")
