"""
Clean Scalping Strategy - Simple, Proven Framework

Based on:
1. EMA Trend Filter (9/21/200)
2. RSI Momentum (40/60 zones)
3. MACD Confirmation
4. Volume Confirmation
5. ATR Volatility Filter
6. Location-based edge (S/R, sweeps, VWAP)
7. LiquiditySweepAnalyzer (4-layer ICT pipeline: Bias→Sweep→MSS→Entry)

Multi-timeframe: 5m for trend, 1m for entry
When USE_ADVANCED_SWEEP=True, the LiquiditySweepAnalyzer replaces the
basic detect_sweep() and acts as the primary entry gate.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger('CleanScalper')

# Try to import the advanced sweep analyzer
try:
    from src.ai.liquidity_sweep import LiquiditySweepAnalyzer
    _SWEEP_ANALYZER_AVAILABLE = True
except ImportError:
    _SWEEP_ANALYZER_AVAILABLE = False
    logger.warning("LiquiditySweepAnalyzer not available — falling back to basic sweep")

try:
    from src.risk.sl_tp import calculate_sl_tp as calculate_dynamic_sl_tp
    _DYNAMIC_SLTP_AVAILABLE = True
except ImportError:
    _DYNAMIC_SLTP_AVAILABLE = False
    logger.warning("Dynamic SL/TP module not available — falling back to fixed ticks")


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
    # Raised from 3.5 → 4.0: require trend + strong RSI or MACD + session
    MIN_CONFIRMATIONS = 4.0

    # Hard session gate: only trade during active sessions (not just bonus)
    REQUIRE_SESSION = True

    # Require at least one full RSI or MACD crossover (not just zone)
    REQUIRE_STRONG_SIGNAL = True

    # Sweep gate - when True, only trade on sweep rejections
    REQUIRE_SWEEP = True

    # Require true MSS confirmation for advanced sweep entries.
    # This avoids softer sweep-only entries that often reduce win rate.
    REQUIRE_TRUE_MSS = True

    # Quality filters for execution
    MIN_ENTRY_CONFIDENCE = 0.70
    MIN_DYNAMIC_RR = 1.6
    MIN_ENTRY_CONFIDENCE_BY_SYMBOL = {
        'MES': 0.70,
        'MNQ': 0.66,
        'NQ': 0.66,
    }
    MIN_DYNAMIC_RR_BY_SYMBOL = {
        'MES': 1.6,
        'MNQ': 1.5,
        'NQ': 1.5,
    }

    # Use the advanced 4-layer LiquiditySweepAnalyzer as the primary sweep detector.
    # When True: LiquiditySweepAnalyzer.get_signal() is called and its detected sweep
    # awards +3.0 points (vs +2.0 for basic sweep) + MSS confirmation gives a further
    # +1.5 bonus.  EMA/RSI/MACD still run as confirmation boosters.
    # When False: fall back to the original simple swing high/low sweep detector.
    USE_ADVANCED_SWEEP = True

    # SL/TP in ticks - OPTIMIZED SETTINGS
    # MES: 12 ticks SL (3 pts) / 36 ticks TP (9 pts) = 3.0 R:R (profitable at >25% WR)
    INSTRUMENT_CONFIG = {
        'MES': {
            'tick_size': 0.25,
            'tick_value': 1.25,  # $1.25 per tick
            'sl_ticks': 12,      # 3 points SL - matches MES structure
            'tp_ticks': 36,      # 9 points TP - 3.0 R:R
        },
        'MNQ': {
            'tick_size': 0.25,
            'tick_value': 0.50,  # $0.50 per tick
            'sl_ticks': 15,
            'tp_ticks': 45,      # 3.0 R:R
        },
        'NQ': {
            'tick_size': 0.25,
            'tick_value': 5.00,  # $5.00 per tick (full NQ)
            'sl_ticks': 15,      # Same geometry as MNQ (same underlying)
            'tp_ticks': 45,      # 3.0 R:R
        },
    }

    def __init__(self, min_confirmations: int = None, use_advanced_sweep: bool = None):
        if min_confirmations is not None:
            self.MIN_CONFIRMATIONS = min_confirmations
        if use_advanced_sweep is not None:
            self.USE_ADVANCED_SWEEP = use_advanced_sweep
        # Instantiate the advanced sweep analyzer once (shared across calls)
        if self.USE_ADVANCED_SWEEP and _SWEEP_ANALYZER_AVAILABLE:
            self._sweep_analyzer = LiquiditySweepAnalyzer()
        else:
            self._sweep_analyzer = None

    def _build_sr_levels(self, df_1m: pd.DataFrame, df_5m: Optional[pd.DataFrame] = None) -> Dict:
        """Build support/resistance from 5m structure (preferred) with 1m fallback."""
        source_df = df_5m if df_5m is not None and len(df_5m) >= 80 else df_1m
        if source_df is None or len(source_df) < 40:
            return {'support_levels': [], 'resistance_levels': []}

        lookback = source_df.tail(200)
        highs = lookback['high'].astype(float)
        lows = lookback['low'].astype(float)

        # Use rolling extrema as lightweight local structure levels.
        roll_high = highs.rolling(window=7, center=True).max()
        roll_low = lows.rolling(window=7, center=True).min()

        resistance_levels = []
        support_levels = []

        for i in range(3, len(lookback) - 3):
            h = float(highs.iloc[i])
            l = float(lows.iloc[i])
            if h == float(roll_high.iloc[i]):
                resistance_levels.append(h)
            if l == float(roll_low.iloc[i]):
                support_levels.append(l)

        # Keep latest unique-ish levels (dedupe by small tolerance)
        def _dedupe(levels, tol=0.25):
            out = []
            for lv in sorted(levels):
                if not out or abs(lv - out[-1]) > tol:
                    out.append(lv)
            return out

        return {
            'support_levels': _dedupe(support_levels)[-12:],
            'resistance_levels': _dedupe(resistance_levels)[:12],
        }

    def _compute_sl_tp(self, df_1m: pd.DataFrame, pair: str, direction: str,
                       df_5m: Optional[pd.DataFrame],
                       adv_sweep_signal: Optional[Dict], fixed_entry_price: float) -> Dict:
        """Compute SL/TP from advanced sweep+structure when possible, else fixed ticks."""
        config = self.INSTRUMENT_CONFIG.get(pair, self.INSTRUMENT_CONFIG['MES'])
        tick_size = config['tick_size']
        sl_ticks = config['sl_ticks']
        tp_ticks = config['tp_ticks']

        # Try dynamic SL/TP first (sweep wick + ATR + S/R)
        if _DYNAMIC_SLTP_AVAILABLE and len(df_1m) >= 60:
            try:
                sr_levels = self._build_sr_levels(df_1m, df_5m=df_5m)
                sweep_wick = None
                if adv_sweep_signal and isinstance(adv_sweep_signal, dict):
                    sweep_wick = (adv_sweep_signal.get('sweep') or {}).get('sweep_wick')

                dynamic = calculate_dynamic_sl_tp(
                    df=df_1m,
                    direction=direction,
                    pair=pair,
                    timeframe='1m',
                    sr_levels=sr_levels,
                    sweep_wick=sweep_wick,
                )

                if dynamic and dynamic.get('stop_loss') and dynamic.get('take_profit'):
                    entry_price = float(dynamic.get('entry_price', fixed_entry_price))
                    return {
                        'entry_price': entry_price,
                        'stop_loss': float(dynamic['stop_loss']),
                        'take_profit': float(dynamic['take_profit']),
                        'sl_ticks': abs(entry_price - float(dynamic['stop_loss'])) / tick_size,
                        'tp_ticks': abs(float(dynamic['take_profit']) - entry_price) / tick_size,
                        'risk_reward': float(dynamic.get('rr_ratio', 0.0)),
                        'method': 'dynamic_structure_sweep',
                    }
            except Exception:
                pass

        # Fallback: fixed tick SL/TP
        sl_distance = sl_ticks * tick_size
        tp_distance = tp_ticks * tick_size
        if direction == 'BUY':
            stop_loss = fixed_entry_price - sl_distance
            take_profit = fixed_entry_price + tp_distance
        else:
            stop_loss = fixed_entry_price + sl_distance
            take_profit = fixed_entry_price - tp_distance

        return {
            'entry_price': fixed_entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'sl_ticks': sl_ticks,
            'tp_ticks': tp_ticks,
            'risk_reward': tp_ticks / sl_ticks,
            'method': 'fixed_ticks',
        }

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

        # --- Advanced sweep detection (LiquiditySweepAnalyzer 4-layer pipeline) ---
        # Use pre-cached result from backtest loop when available (set by run_backtest()
        # to avoid recomputing the expensive pipeline 70K times).  In live mode the
        # cache is not set, so we compute it here.
        adv_sweep_signal = getattr(self, '_cached_adv_sweep', None)
        if adv_sweep_signal is None and self._sweep_analyzer is not None:
            try:
                adv_sweep_signal = self._sweep_analyzer.get_signal(
                    df_1m, pair, df_5m=df_5m
                )
            except Exception as e:
                logger.debug(f"Advanced sweep failed, falling back to basic: {e}")
                adv_sweep_signal = None

        # Even when using advanced sweep we still run the basic detector for the
        # coarse "location edge" check — it's cheap and used only as fallback bonus.
        sweep = self.detect_sweep(df_1m)
        if self.REQUIRE_SWEEP and not sweep['detected'] and adv_sweep_signal is None:
            result['details'] = 'No sweep detected'
            return result

        sweep_direction = sweep.get('direction')  # BUY or SELL (basic detector)

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

        # ── HARD GATE: Session filter ─────────────────────────────
        # Only trade during active sessions; otherwise skip entirely
        if self.REQUIRE_SESSION and not confirmations['session']:
            result['details'] = 'Outside active session'
            return result

        # ── HARD GATE: Trend must be directional ──────────────────
        if trend_bias == 'NEUTRAL':
            result['details'] = 'No trend bias (NEUTRAL EMAs)'
            return result


        # ═══════════════════════════════════════════════════════════
        # Count confirmations for each direction
        # ═══════════════════════════════════════════════════════════
        long_confirms = 0
        short_confirms = 0

        # Trend is the PRIMARY filter - worth 2 points
        if trend_bias == 'LONG':
            long_confirms += 2.0
        elif trend_bias == 'SHORT':
            short_confirms += 2.0

        # RSI - strong cross = 1.5, zone = 0.5
        has_strong_rsi = False
        if rsi_signal == 'LONG':
            long_confirms += 1.5
            has_strong_rsi = True
        elif rsi_signal == 'LONG_ZONE':
            long_confirms += 0.5
        if rsi_signal == 'SHORT':
            short_confirms += 1.5
            has_strong_rsi = True
        elif rsi_signal == 'SHORT_ZONE':
            short_confirms += 0.5

        # MACD - strong crossover = 1.5, zone = 0.5
        has_strong_macd = False
        if macd_signal == 'LONG':
            long_confirms += 1.5
            has_strong_macd = True
        elif macd_signal == 'LONG_ZONE':
            long_confirms += 0.5
        if macd_signal == 'SHORT':
            short_confirms += 1.5
            has_strong_macd = True
        elif macd_signal == 'SHORT_ZONE':
            short_confirms += 0.5

        # Sweep detection - major bonus (liquidity level rejection)
        # Advanced analyzer: ICT 4-layer sweep+MSS gives higher-quality signal
        adv_sweep_direction = None
        if adv_sweep_signal is not None:
            adv_sweep_direction = adv_sweep_signal.get('signal')  # 'BUY'/'SELL'/'SKIP'
            adv_mss_confirmed = bool((adv_sweep_signal.get('mss') or {}).get('confirmed'))
            adv_confidence = adv_sweep_signal.get('confidence', 0.0)

            if adv_sweep_direction == 'BUY':
                long_confirms += 3.0   # Higher quality than basic +2.0
                if adv_mss_confirmed:
                    long_confirms += 1.5  # MSS = extra structural confirmation
                confirmations['adv_sweep'] = 'BUY'
            elif adv_sweep_direction == 'SELL':
                short_confirms += 3.0
                if adv_mss_confirmed:
                    short_confirms += 1.5
                confirmations['adv_sweep'] = 'SELL'
            elif sweep['detected']:
                # Advanced didn't fire but basic did — use smaller basic bonus
                if sweep['direction'] == 'BUY':
                    long_confirms += 2.0
                elif sweep['direction'] == 'SELL':
                    short_confirms += 2.0
                confirmations['sweep'] = sweep['direction']
        elif sweep['detected']:
            # No advanced analyzer — use basic sweep bonus
            if sweep['direction'] == 'BUY':
                long_confirms += 2.0
            elif sweep['direction'] == 'SELL':
                short_confirms += 2.0
            confirmations['sweep'] = sweep['direction']

        # Volume: add to direction if above average (was: both directions)
        # Now directional: high volume validates BOTH but gives smaller weight
        if confirmations['volume']:
            long_confirms += 0.3
            short_confirms += 0.3

        # Location edge (directional bonus)
        if self.check_location_edge(row_1m, 'LONG'):
            long_confirms += 0.7
            confirmations['location'] = True
        elif self.check_location_edge(row_1m, 'SHORT'):
            short_confirms += 0.7
            confirmations['location'] = True

        # Pullback detection - key for good entries
        atr = row_1m.get('atr', 0)
        if not pd.isna(atr) and atr > 0:
            price_to_ema21 = row_1m['close'] - row_1m['ema_21']
            if abs(price_to_ema21) < atr * 1.0:  # Tight: within 1 ATR of EMA21
                if price_to_ema21 >= 0:
                    long_confirms += 0.7
                    confirmations['pullback'] = 'LONG'
                else:
                    short_confirms += 0.7
                    confirmations['pullback'] = 'SHORT'

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

        # ── HARD GATE: Require at least one strong RSI or MACD signal ─
        if self.REQUIRE_STRONG_SIGNAL and not (has_strong_rsi or has_strong_macd):
            result['details'] = 'No strong RSI/MACD crossover signal'
            return result

        # ── HARD GATE: Direction must align with 5m trend ─────────
        trend_dir = 'BUY' if trend_bias == 'LONG' else ('SELL' if trend_bias == 'SHORT' else None)
        if trend_dir is not None and direction != trend_dir:
            result['details'] = f'Counter-trend trade blocked: trend={trend_bias}, signal={direction}'
            return result

        # When sweep is required, direction must match sweep
        if self.REQUIRE_SWEEP and sweep_direction and direction != sweep_direction:
            result['details'] = (
                f"Direction mismatch: sweep={sweep_direction}, indicators={direction}"
            )
            return result

        # Hard gate: if advanced sweep fired in the OPPOSITE direction, skip
        if adv_sweep_direction in ('BUY', 'SELL') and adv_sweep_direction != direction:
            result['details'] = (
                f"Advanced sweep opposes signal: sweep={adv_sweep_direction}, "
                f"indicators={direction} — skipped"
            )
            return result

        # Hard gate: require true MSS confirmation when advanced sweep is active.
        if self.REQUIRE_TRUE_MSS and adv_sweep_signal is not None and adv_sweep_direction == direction:
            mss_details = str((adv_sweep_signal.get('mss') or {}).get('details', ''))
            if 'Sweep-only entry' in mss_details:
                result['details'] = 'Advanced sweep found but true MSS not confirmed'
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
        # Advanced sweep bonus: ICT sweep+MSS adds structural conviction
        if adv_sweep_signal is not None and adv_sweep_direction == direction:
            adv_conf = adv_sweep_signal.get('confidence', 0.0)
            adv_mss = bool((adv_sweep_signal.get('mss') or {}).get('confirmed'))
            confidence += 0.08  # Sweep fired in our direction
            if adv_mss:
                confidence += 0.05  # MSS confirmed = extra structural conviction
            if adv_conf > 0.7:
                confidence += 0.04  # High-confidence sweep

        # Penalty for counter-trend (RSI/MACD against trend)
        if direction == 'BUY' and trend_bias == 'SHORT':
            confidence -= 0.15
        elif direction == 'SELL' and trend_bias == 'LONG':
            confidence -= 0.15

        confidence = max(0.0, min(1.0, confidence))

        min_conf = self.MIN_ENTRY_CONFIDENCE_BY_SYMBOL.get(pair, self.MIN_ENTRY_CONFIDENCE)

        # Hard gate: only take high-confidence entries.
        if confidence < min_conf:
            result['details'] = f'Confidence too low: {confidence:.2f} < {min_conf:.2f}'
            return result

        # ═══════════════════════════════════════════════════════════
        # Calculate SL/TP (dynamic: sweep + S/R + ATR, with fixed fallback)
        # ═══════════════════════════════════════════════════════════
        entry_price = float(row_1m['close'])
        sl_tp = self._compute_sl_tp(
            df_1m=df_1m,
            pair=pair,
            direction=direction,
            df_5m=df_5m,
            adv_sweep_signal=adv_sweep_signal,
            fixed_entry_price=entry_price,
        )

        # Hard gate: skip weak reward-to-risk setups.
        rr = float(sl_tp.get('risk_reward', 0.0) or 0.0)
        min_rr = self.MIN_DYNAMIC_RR_BY_SYMBOL.get(pair, self.MIN_DYNAMIC_RR)
        if rr < min_rr:
            result['details'] = f'R:R too low: {rr:.2f} < {min_rr:.2f}'
            return result

        # ═══════════════════════════════════════════════════════════
        # Build result
        # ═══════════════════════════════════════════════════════════
        result['signal'] = direction
        result['confidence'] = confidence
        result['confirmations'] = confirm_count
        result['sl_tp'] = sl_tp

        trend_label = f"5m" if use_5m_trend else "1m"
        adv_sweep_label = f"AdvSweep={adv_sweep_direction or 'SKIP'}" if adv_sweep_signal else "AdvSweep=OFF"
        adv_mss_label = f"MSS={'✓' if adv_sweep_signal and (adv_sweep_signal.get('mss') or {}).get('confirmed') else '✗'}"
        result['details'] = (
            f"✅ {direction} | Confirms={confirm_count:.1f} | "
            f"Trend({trend_label})={trend_bias} | RSI={rsi_signal} | MACD={macd_signal} | "
            f"Vol={'✓' if confirmations['volume'] else '✗'} | "
            f"ATR={'✓' if confirmations['atr'] else '✗'} | "
            f"Loc={'✓' if confirmations['location'] else '✗'} | "
            f"Sess={'✓' if confirmations['session'] else '✗'} | "
            f"{adv_sweep_label} | {adv_mss_label} | SLTP={sl_tp.get('method', 'n/a')}"
        )

        return result


def run_backtest(df_5m: pd.DataFrame, df_1m: pd.DataFrame, pair: str,
                 initial_balance: float = 50000,
                 min_confirmations: int = 3,
                 max_contracts: int = 2,
                 use_advanced_sweep: bool = True,
                 exit_profile_override: dict = None) -> Dict:
    """
    Simple backtest of the clean scalper strategy.
    
    Args:
        df_5m: 5-minute OHLCV data
        df_1m: 1-minute OHLCV data
        pair: Instrument symbol
        initial_balance: Starting balance
        min_confirmations: Min confirmations for entry
        max_contracts: Max position size
        use_advanced_sweep: Whether to use LiquiditySweepAnalyzer (default: True)
        
    Returns:
        dict with results
    """
    scalper = CleanScalper(min_confirmations=min_confirmations,
                           use_advanced_sweep=use_advanced_sweep)
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

    # ATR regime filter: avoid dead chop and panic-volatility bars.
    # Rank current ATR over recent history to classify volatility regime.
    df_1m['atr_pct_rank'] = df_1m['atr'].rolling(window=400, min_periods=120).rank(pct=True)

    open_position = None
    cooldown_until = 0
    consecutive_losses = 0
    max_hold_bars = 60  # Max hold = 60 1m bars = 1 hour

    regime_bounds_by_symbol = {
        'MES': (0.15, 0.95),
        'MNQ': (0.20, 0.92),
        'NQ': (0.20, 0.92),
    }
    regime_low, regime_high = regime_bounds_by_symbol.get(pair, (0.15, 0.95))

    # Exit-management profile by symbol.
    # Later BE/trailing activation avoids clipping winners too early.
    exit_profile_by_symbol = {
        'MES': {
            'be_trigger_r': 1.30,
            'be_buffer_ticks': 2,
            'trail_trigger_r': 1.90,
            'trail_r_mult': 0.90,
            'trail_atr_mult': 1.00,
        },
        'MNQ': {
            'be_trigger_r': 1.35,
            'be_buffer_ticks': 8,
            'trail_trigger_r': 2.20,
            'trail_r_mult': 1.05,
            'trail_atr_mult': 1.35,
        },
        'NQ': {
            'be_trigger_r': 1.35,
            'be_buffer_ticks': 8,
            'trail_trigger_r': 2.20,
            'trail_r_mult': 1.05,
            'trail_atr_mult': 1.35,
        },
    }
    exit_profile = exit_profile_override if exit_profile_override is not None else exit_profile_by_symbol.get(pair, exit_profile_by_symbol['MES'])

    # ── Pre-compute advanced sweep signals for all 5m bars (efficient) ──
    # Instead of calling the expensive analyzer 21K times in the 1m loop,
    # we run one pre-computation pass over the 5m data and cache the results.
    adv_sweep_cache: dict = {}  # {5m_bar_idx: sweep_signal_dict}
    if scalper._sweep_analyzer is not None and df_5m is not None and len(df_5m) > 250:
        logger.info(f"Pre-computing advanced sweep signals for {len(df_5m)} 5m bars...")
        # Get 1m bar count for approximate mapping
        _sweep_lookback = 250
        _sweep_step = 5  # only recompute every 5 5m-bars (= every 25m) for speed
        for _i5 in range(_sweep_lookback, len(df_5m), _sweep_step):
            _df5_slice = df_5m.iloc[max(0, _i5 - _sweep_lookback):_i5 + 1]
            # Approximate 1m slice from 5m index
            _i1 = min(_i5 * 5, len(df_1m) - 1)
            _df1_slice = df_1m.iloc[max(0, _i1 - _sweep_lookback):_i1 + 1]
            if len(_df1_slice) < 60:
                continue
            try:
                _sig = scalper._sweep_analyzer.get_signal(_df1_slice, pair, df_5m=_df5_slice)
                # Fill in all 5 bars covered by this step
                for _k in range(_sweep_step):
                    adv_sweep_cache[_i5 + _k] = _sig
            except Exception:
                pass
        logger.info(f"Advanced sweep pre-computation done: {len(adv_sweep_cache)} entries")

    # Iterate through 1m bars
    lookback = 250

    for idx in range(lookback, len(df_1m)):
        row = df_1m.iloc[idx]
        candle_dt = row['datetime'] if 'datetime' in df_1m.columns else None
        candle_hour = candle_dt.hour if candle_dt else None

        # Handle open position
        if open_position:
            bars_held = idx - open_position['entry_idx']

            # Profit-protection logic:
            # 1) move SL to breakeven only after a clearer move in our favor
            # 2) activate ATR-informed trailing after deeper R expansion
            initial_r = open_position['initial_r']
            if initial_r > 0:
                if open_position['type'] == 'BUY':
                    favorable_move = row['high'] - open_position['entry_price']
                else:
                    favorable_move = open_position['entry_price'] - row['low']

                be_trigger_r = float(exit_profile.get('be_trigger_r', 1.2))
                be_buffer_ticks = int(exit_profile.get('be_buffer_ticks', 4))
                if not open_position['be_moved'] and favorable_move >= initial_r * be_trigger_r:
                    if open_position['type'] == 'BUY':
                        be_stop = open_position['entry_price'] + (tick_size * be_buffer_ticks)
                        open_position['stop_loss'] = max(open_position['stop_loss'], be_stop)
                    else:
                        be_stop = open_position['entry_price'] - (tick_size * be_buffer_ticks)
                        open_position['stop_loss'] = min(open_position['stop_loss'], be_stop)
                    open_position['be_moved'] = True

                trail_trigger_r = float(exit_profile.get('trail_trigger_r', 2.0))
                if favorable_move >= initial_r * trail_trigger_r:
                    open_position['trail_active'] = True

                # Start trailing only after breakeven has been secured.
                if open_position['trail_active'] and open_position['be_moved']:
                    atr_now = row.get('atr', np.nan)
                    if pd.isna(atr_now) or atr_now <= 0:
                        atr_now = initial_r
                    trail_r_mult = float(exit_profile.get('trail_r_mult', 0.95))
                    trail_atr_mult = float(exit_profile.get('trail_atr_mult', 1.2))
                    trail_distance = max(initial_r * trail_r_mult, atr_now * trail_atr_mult)
                    if open_position['type'] == 'BUY':
                        trail_stop = row['high'] - trail_distance
                        open_position['stop_loss'] = max(open_position['stop_loss'], trail_stop)
                    else:
                        trail_stop = row['low'] + trail_distance
                        open_position['stop_loss'] = min(open_position['stop_loss'], trail_stop)
            
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

                if net_pnl > 0:
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1

                open_position = None
                # Adaptive cooldown: cool off more after losing streaks.
                base_cd = 20 if exit_type == 'SL' else 5
                streak_cd = min(30, consecutive_losses * 6)
                if consecutive_losses >= 3:
                    streak_cd += 20
                cooldown_until = idx + base_cd + streak_cd
                continue

        # Skip if in cooldown or already in position
        if idx < cooldown_until or open_position:
            continue

        # Skip bars outside target volatility regime for this symbol.
        atr_rank = row.get('atr_pct_rank', np.nan)
        if pd.isna(atr_rank) or atr_rank < regime_low or atr_rank > regime_high:
            continue

        # Get 5m subset - use simple index mapping (5m bars = 1m idx // 5)
        # This is approximate but much faster than datetime matching
        df_5m_subset = None
        approx_5m_idx = -1
        if len(df_5m) > 210:
            # Approximate 5m index from 1m index
            approx_5m_idx = min(idx // 5, len(df_5m) - 1)
            start_5m = max(0, approx_5m_idx - 250)
            df_5m_subset = df_5m.iloc[start_5m:approx_5m_idx + 1]
            if len(df_5m_subset) < 210:
                df_5m_subset = None

        # Look up pre-computed advanced sweep signal from the batch cache
        if scalper._sweep_analyzer is not None:
            # Cache is indexed by 5m bar, filled every _sweep_step=5 bars
            _sweep_step = 5
            _lookup_key = (approx_5m_idx // _sweep_step) * _sweep_step if approx_5m_idx >= 0 else -1
            scalper._cached_adv_sweep = adv_sweep_cache.get(_lookup_key, None)

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
            'initial_r': abs(float(sl_tp['entry_price']) - float(sl_tp['stop_loss'])),
            'be_moved': False,
            'trail_active': False,
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
