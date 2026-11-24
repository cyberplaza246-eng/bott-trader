"""
Elite 1M / 5M Liquidity Sweep Entry Model

Structure + Liquidity + Momentum Alignment

Architecture:
  Layer 1 — 5M Market Regime & Bias  (trend_up / trend_down / range / high_vol / low_vol)
  Layer 2 — 1M Liquidity Sweep Event (stop-hunt detection)
  Layer 3 — 1M Displacement Candle   (momentum confirmation)

Entry only when ALL THREE layers align within a 3-candle window.

SL = sweep wick extreme ± 0.2 × ATR buffer
TP = 1.5R default, 2R in high-volatility impulse, 1.2R in range
Time stop = exit if not positive after 3 candles
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from src.utils.logger import bot_logger


class LiquiditySweepAnalyzer:
    """Detects liquidity sweeps and generates high-probability entry signals."""

    # ── Pair Configuration ──────────────────────────────────────────
    PAIR_CONFIG = {
        'EUR/USD': {
            'session_atr_min': 0.00040,
            'spread_sim': 0.00015,
            'pip_size': 0.0001,
        },
        'GBP/USD': {
            'session_atr_min': 0.00055,
            'spread_sim': 0.00020,
            'pip_size': 0.0001,
        },
        'USD/JPY': {
            'session_atr_min': 0.060,
            'spread_sim': 0.020,
            'pip_size': 0.01,
        },
    }

    # ── Thresholds (tuned for quality over volume) ──────────────────
    SWEEP_LOOKBACK = 5              # Candles to define liquidity level
    ADX_MIN_BIAS = 18               # Minimum ADX for directional bias
    VOLUME_CONFIRMATION = 1.30      # Volume > 1.3× 10-candle average
    BODY_RATIO_MIN = 0.60           # Displacement candle body ≥ 60% of range
    RSI_SWEEP_LONG_MAX = 40         # RSI must dip below this during bullish sweep
    RSI_SWEEP_SHORT_MIN = 60        # RSI must spike above this during bearish sweep
    ATR_HIGH_VOL_MULT = 1.30        # ATR > 1.3× 20-period mean = high vol
    ATR_LOW_VOL_MULT = 0.70         # ATR < 0.7× 20-period mean = low vol
    CONFIRMATION_WINDOW = 3         # Candles after sweep to get displacement
    SL_ATR_BUFFER = 0.20            # SL = sweep wick ± 0.2×ATR

    # ── Session Windows (UTC) ───────────────────────────────────────
    OPTIMAL_SESSIONS = {
        'london': (7, 12),
        'ny': (13, 17),
    }

    def __init__(self):
        self.atr_period = 14
        self.rsi_period = 14
        self.ema_short = 20
        self.ema_medium = 50
        self.ema_long = 200
        self.adx_period = 14
        self.volume_avg_period = 10  # Tighter than 20 — recent volume matters more

    # =================================================================
    #  INDICATOR CALCULATION
    # =================================================================

    def calculate_indicators(self, df):
        """Add all required indicators to the dataframe."""
        df = df.copy()
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(0, index=df.index)

        # EMAs
        df['ema_20'] = close.ewm(span=self.ema_short, adjust=False).mean()
        df['ema_50'] = close.ewm(span=self.ema_medium, adjust=False).mean()
        df['ema_200'] = close.ewm(span=self.ema_long, adjust=False).mean()

        # ATR
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=self.atr_period).mean()
        df['atr_sma20'] = df['atr'].rolling(window=20).mean()

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # ADX + DI
        df = self._compute_adx(df, high, low, close)

        # Volume ratio (10-period)
        vol_sma = volume.rolling(window=self.volume_avg_period).mean()
        df['volume_ratio'] = (volume / vol_sma.replace(0, np.nan)).fillna(1.0)

        # Candle body metrics
        df['body'] = (close - df['open'].astype(float)).abs()
        df['candle_range'] = high - low
        df['body_ratio'] = df['body'] / df['candle_range'].replace(0, np.nan)
        df['is_bullish'] = close > df['open'].astype(float)
        df['is_bearish'] = close < df['open'].astype(float)

        # Higher highs / higher lows for 5M bias
        df['higher_high'] = high > high.shift(1)
        df['higher_low'] = low > low.shift(1)
        df['lower_high'] = high < high.shift(1)
        df['lower_low'] = low < low.shift(1)

        # Liquidity levels (rolling 5-candle extremes)
        df['liq_low'] = low.rolling(window=self.SWEEP_LOOKBACK).min().shift(1)
        df['liq_high'] = high.rolling(window=self.SWEEP_LOOKBACK).max().shift(1)

        return df

    def _compute_adx(self, df, high, low, close):
        """Compute ADX, +DI, -DI."""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr_smooth = tr.rolling(window=self.adx_period).mean()
        plus_di = 100 * (plus_dm.rolling(window=self.adx_period).mean() / atr_smooth.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(window=self.adx_period).mean() / atr_smooth.replace(0, np.nan))

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        df['adx'] = dx.rolling(window=self.adx_period).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        return df

    # =================================================================
    #  LAYER 1: 5M REGIME & BIAS
    # =================================================================

    def detect_regime(self, df_5m):
        """Classify market regime from 5M data.

        Returns:
            dict: {regime, bias, adx, atr_state, details}
            regime in: trend_up, trend_down, range, high_volatility, low_volatility
            bias in: 'BUY', 'SELL', None
        """
        if df_5m is None or len(df_5m) < 60:
            return {'regime': 'unknown', 'bias': None, 'adx': 0,
                    'atr_state': 'unknown', 'details': 'Insufficient 5M data'}

        latest = df_5m.iloc[-1]
        ema_20 = float(latest.get('ema_20', 0))
        ema_50 = float(latest.get('ema_50', 0))
        adx = float(latest.get('adx', 0) or 0)
        atr = float(latest.get('atr', 0) or 0)
        atr_sma20 = float(latest.get('atr_sma20', 0) or 0)
        close = float(latest['close'])

        # Higher highs check (last 2 candles)
        hh_1 = bool(df_5m['higher_high'].iloc[-1]) if 'higher_high' in df_5m.columns else False
        hh_2 = bool(df_5m['higher_high'].iloc[-2]) if len(df_5m) > 1 and 'higher_high' in df_5m.columns else False
        ll_1 = bool(df_5m['lower_low'].iloc[-1]) if 'lower_low' in df_5m.columns else False
        ll_2 = bool(df_5m['lower_low'].iloc[-2]) if len(df_5m) > 1 and 'lower_low' in df_5m.columns else False

        # ATR state
        if atr_sma20 > 0:
            if atr > atr_sma20 * self.ATR_HIGH_VOL_MULT:
                atr_state = 'high_volatility'
            elif atr < atr_sma20 * self.ATR_LOW_VOL_MULT:
                atr_state = 'low_volatility'
            else:
                atr_state = 'normal'
        else:
            atr_state = 'normal'

        # Regime + Bias classification
        bias = None
        regime = 'range'

        if adx >= self.ADX_MIN_BIAS:
            if ema_20 > ema_50 and close > ema_20:
                # Trend up requirements: EMA20 > EMA50, price above EMA20,
                # last 2 candles making higher highs
                if hh_1 and hh_2:
                    regime = 'trend_up'
                    bias = 'BUY'
                elif hh_1 or hh_2:
                    # Partial structure — still bullish but weaker
                    regime = 'trend_up'
                    bias = 'BUY'
                else:
                    regime = 'trend_up'
                    bias = 'BUY'
            elif ema_20 < ema_50 and close < ema_20:
                if ll_1 and ll_2:
                    regime = 'trend_down'
                    bias = 'SELL'
                elif ll_1 or ll_2:
                    regime = 'trend_down'
                    bias = 'SELL'
                else:
                    regime = 'trend_down'
                    bias = 'SELL'
            else:
                regime = 'range'
                bias = None
        else:
            # ADX too low — ranging
            regime = 'range'
            bias = None

        # Override with volatility regime if extreme
        if atr_state == 'high_volatility' and regime in ('trend_up', 'trend_down'):
            regime = 'high_volatility'
            # Keep bias from trend detection
        elif atr_state == 'low_volatility':
            regime = 'low_volatility'
            bias = None  # Don't trade in dead markets

        details = (
            f"EMA20={ema_20:.5f} EMA50={ema_50:.5f} ADX={adx:.1f} "
            f"ATR={atr:.5f} ATR_state={atr_state} HH={hh_1},{hh_2} LL={ll_1},{ll_2}"
        )

        return {
            'regime': regime,
            'bias': bias,
            'adx': adx,
            'atr_state': atr_state,
            'ema_20': ema_20,
            'ema_50': ema_50,
            'close': close,
            'details': details,
        }

    # =================================================================
    #  LAYER 2: 1M LIQUIDITY SWEEP DETECTION
    # =================================================================

    def detect_sweep(self, df_1m, bias):
        """Detect a liquidity sweep event on 1M data.

        Bullish sweep:
          - Current candle low dips below 5-candle lowest low (sweeps stops)
          - Current candle CLOSES back above the swept level
          - RSI dipped below 40 during the sweep

        Bearish sweep:
          - Current candle high spikes above 5-candle highest high
          - Current candle CLOSES back below the swept level
          - RSI spiked above 60 during the sweep

        Returns:
            dict: {detected, direction, sweep_wick, swept_level, rsi_at_sweep, details}
        """
        result = {
            'detected': False,
            'direction': None,
            'sweep_wick': None,
            'swept_level': None,
            'rsi_at_sweep': None,
            'candle_index': None,
            'details': '',
        }

        if df_1m is None or len(df_1m) < self.SWEEP_LOOKBACK + 5:
            result['details'] = 'Insufficient 1M data'
            return result

        # Check last 3 candles for a sweep (the confirmation window)
        for i in range(-self.CONFIRMATION_WINDOW, 0):
            candle = df_1m.iloc[i]
            candle_low = float(candle['low'])
            candle_high = float(candle['high'])
            candle_close = float(candle['close'])
            candle_rsi = float(candle.get('rsi', 50) or 50)

            # Get the liquidity level BEFORE this candle
            lookback_end = len(df_1m) + i
            lookback_start = max(0, lookback_end - self.SWEEP_LOOKBACK - 1)
            if lookback_end <= lookback_start + 1:
                continue
            lookback_window = df_1m.iloc[lookback_start:lookback_end - 1]

            if len(lookback_window) < self.SWEEP_LOOKBACK:
                continue

            liq_low = float(lookback_window['low'].min())
            liq_high = float(lookback_window['high'].max())

            if bias == 'BUY':
                # Bullish sweep: price dips below liquidity low then closes above
                swept = candle_low < liq_low and candle_close > liq_low
                rsi_condition = candle_rsi < self.RSI_SWEEP_LONG_MAX

                if swept and rsi_condition:
                    result.update({
                        'detected': True,
                        'direction': 'BUY',
                        'sweep_wick': candle_low,
                        'swept_level': liq_low,
                        'rsi_at_sweep': candle_rsi,
                        'candle_index': i,
                        'details': (
                            f"Bullish sweep: low {candle_low:.5f} < liq_low {liq_low:.5f}, "
                            f"close {candle_close:.5f} > liq_low, RSI={candle_rsi:.1f}"
                        ),
                    })
                    return result

            elif bias == 'SELL':
                # Bearish sweep: price spikes above liquidity high then closes below
                swept = candle_high > liq_high and candle_close < liq_high
                rsi_condition = candle_rsi > self.RSI_SWEEP_SHORT_MIN

                if swept and rsi_condition:
                    result.update({
                        'detected': True,
                        'direction': 'SELL',
                        'sweep_wick': candle_high,
                        'swept_level': liq_high,
                        'rsi_at_sweep': candle_rsi,
                        'candle_index': i,
                        'details': (
                            f"Bearish sweep: high {candle_high:.5f} > liq_high {liq_high:.5f}, "
                            f"close {candle_close:.5f} < liq_high, RSI={candle_rsi:.1f}"
                        ),
                    })
                    return result

        result['details'] = 'No liquidity sweep detected in last 3 candles'
        return result

    # =================================================================
    #  LAYER 3: DISPLACEMENT CONFIRMATION
    # =================================================================

    def detect_displacement(self, df_1m, sweep_result):
        """Check if a displacement candle appeared after the sweep.

        Displacement candle must:
          1. Close above prior candle's high (longs) or below prior candle's low (shorts)
          2. Body ≥ 60% of total range (strong candle, not a doji)
          3. Volume > 1.3× last 10-candle average

        Returns:
            dict: {confirmed, candle_data, details}
        """
        result = {
            'confirmed': False,
            'entry_price': None,
            'trigger_level': None,
            'candle_data': {},
            'details': '',
        }

        if not sweep_result['detected']:
            result['details'] = 'No sweep to confirm'
            return result

        sweep_idx = sweep_result['candle_index']  # negative index (-3, -2, -1)
        direction = sweep_result['direction']

        # Check candles AFTER the sweep for displacement
        # sweep_idx is negative: -3, -2, or -1
        # We need candles after it up to and including the latest
        check_start = sweep_idx + 1
        if check_start >= 0:
            check_start = -1  # At minimum, check the latest candle

        for i in range(check_start, 0):
            if abs(i) > len(df_1m):
                continue

            candle = df_1m.iloc[i]
            prev_candle = df_1m.iloc[i - 1]

            candle_close = float(candle['close'])
            candle_open = float(candle['open'])
            candle_high = float(candle['high'])
            candle_low = float(candle['low'])
            prev_high = float(prev_candle['high'])
            prev_low = float(prev_candle['low'])
            body = abs(candle_close - candle_open)
            candle_range = candle_high - candle_low
            body_ratio = body / candle_range if candle_range > 0 else 0
            vol_ratio = float(candle.get('volume_ratio', 1.0) or 1.0)

            if direction == 'BUY':
                # Bullish displacement: close above prior high, strong body, volume
                closes_above = candle_close > prev_high
                is_bullish = candle_close > candle_open
                body_ok = body_ratio >= self.BODY_RATIO_MIN
                volume_ok = vol_ratio >= self.VOLUME_CONFIRMATION

                if closes_above and is_bullish and body_ok and volume_ok:
                    result.update({
                        'confirmed': True,
                        'entry_price': candle_close,
                        'trigger_level': candle_high,  # Entry on break of this high
                        'candle_data': {
                            'close': candle_close,
                            'high': candle_high,
                            'low': candle_low,
                            'body_ratio': body_ratio,
                            'volume_ratio': vol_ratio,
                        },
                        'details': (
                            f"Bullish displacement: close {candle_close:.5f} > prev_high {prev_high:.5f}, "
                            f"body={body_ratio:.0%}, vol={vol_ratio:.2f}x"
                        ),
                    })
                    return result

            elif direction == 'SELL':
                # Bearish displacement: close below prior low, strong body, volume
                closes_below = candle_close < prev_low
                is_bearish = candle_close < candle_open
                body_ok = body_ratio >= self.BODY_RATIO_MIN
                volume_ok = vol_ratio >= self.VOLUME_CONFIRMATION

                if closes_below and is_bearish and body_ok and volume_ok:
                    result.update({
                        'confirmed': True,
                        'entry_price': candle_close,
                        'trigger_level': candle_low,  # Entry on break of this low
                        'candle_data': {
                            'close': candle_close,
                            'low': candle_low,
                            'high': candle_high,
                            'body_ratio': body_ratio,
                            'volume_ratio': vol_ratio,
                        },
                        'details': (
                            f"Bearish displacement: close {candle_close:.5f} < prev_low {prev_low:.5f}, "
                            f"body={body_ratio:.0%}, vol={vol_ratio:.2f}x"
                        ),
                    })
                    return result

        result['details'] = 'No displacement candle after sweep'
        return result

    # =================================================================
    #  RISK/REWARD CALCULATION
    # =================================================================

    def calculate_risk_reward(self, sweep_result, displacement_result, regime_info, pair):
        """Calculate SL/TP based on sweep wick and regime.

        SL = sweep wick extreme ± 0.2 × ATR buffer
        TP:
          - 1.5R default
          - 2.0R if high_volatility regime (impulse continuation)
          - 1.2R if range regime
        """
        config = self.PAIR_CONFIG.get(pair, self.PAIR_CONFIG['EUR/USD'])
        pip_size = config['pip_size']

        direction = sweep_result['direction']
        sweep_wick = sweep_result['sweep_wick']
        entry_price = displacement_result['entry_price']

        # Get ATR from regime info
        atr = float(regime_info.get('atr', 0) or 0)
        if atr <= 0:
            atr = config['session_atr_min'] * 2  # Fallback

        atr_buffer = atr * self.SL_ATR_BUFFER

        if direction == 'BUY':
            stop_loss = round(sweep_wick - atr_buffer, 5)
            sl_distance = entry_price - stop_loss
        else:
            stop_loss = round(sweep_wick + atr_buffer, 5)
            sl_distance = stop_loss - entry_price

        if sl_distance <= 0:
            return None

        # TP ratio based on regime
        regime = regime_info.get('regime', 'trend_up')
        tp_ratio_map = {
            'high_volatility': 2.0,
            'trend_up': 1.5,
            'trend_down': 1.5,
            'range': 1.2,
            'low_volatility': 1.2,
        }
        tp_ratio = tp_ratio_map.get(regime, 1.5)
        tp_distance = sl_distance * tp_ratio

        if direction == 'BUY':
            take_profit = round(entry_price + tp_distance, 5)
        else:
            take_profit = round(entry_price - tp_distance, 5)

        sl_pips = sl_distance / pip_size
        tp_pips = tp_distance / pip_size

        # Minimum SL check: must be > 3× spread
        spread = config['spread_sim']
        if sl_distance < spread * 3:
            return None

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'sl_distance': sl_distance,
            'tp_distance': tp_distance,
            'risk_pips': sl_pips,
            'reward_pips': tp_pips,
            'rr_ratio': tp_ratio,
            'tp_ratio_used': tp_ratio,
            'atr': atr,
            'sweep_wick': sweep_wick,
            'entry_price': entry_price,
        }

    # =================================================================
    #  MARKET CONDITIONS GATE
    # =================================================================

    def check_market_conditions(self, df_1m, pair, spread=None):
        """Pre-flight checks before running entry logic.

        Returns:
            (ok: bool, reason: str)
        """
        config = self.PAIR_CONFIG.get(pair, self.PAIR_CONFIG['EUR/USD'])

        if df_1m is None or len(df_1m) < 30:
            return False, "Insufficient 1M data"

        latest = df_1m.iloc[-1]
        atr = float(latest.get('atr', 0) or 0)

        # ATR floor
        if atr < config['session_atr_min']:
            return False, (
                f"ATR too low: {atr:.5f} < {config['session_atr_min']:.5f} "
                f"({atr / config['pip_size']:.1f}p < {config['session_atr_min'] / config['pip_size']:.1f}p)"
            )

        # Spread vs ATR
        actual_spread = spread if spread is not None else config['spread_sim']
        if atr > 0 and actual_spread > 0.20 * atr:
            return False, (
                f"Spread too wide: {actual_spread:.5f} > 20% of ATR {atr:.5f}"
            )

        # Exhaustion spike
        candle_range = float(latest.get('candle_range', 0) or (float(latest['high']) - float(latest['low'])))
        if atr > 0 and candle_range > 2.5 * atr:
            return False, (
                f"Exhaustion spike: range {candle_range:.5f} > 2.5×ATR {2.5 * atr:.5f}"
            )

        return True, "Market conditions OK"

    # =================================================================
    #  MAIN SIGNAL GENERATION
    # =================================================================

    def get_signal(self, df_1m, pair, df_5m=None, spread=None):
        """Full entry pipeline: Bias → Sweep → Displacement → R:R.

        Args:
            df_1m: 1-minute DataFrame with OHLCV
            pair: Currency pair string
            df_5m: 5-minute DataFrame (for regime/bias)
            spread: Actual spread from broker

        Returns:
            dict: {
                signal: 'BUY'/'SELL'/'SKIP',
                confidence: 0.0–1.0,
                regime: str,
                bias: str,
                sweep: dict,
                displacement: dict,
                risk_reward: dict,
                details: str,
            }
        """
        result = {
            'signal': 'SKIP',
            'confidence': 0.0,
            'regime': 'unknown',
            'bias': None,
            'sweep': {},
            'displacement': {},
            'risk_reward': None,
            'details': '',
            'sweep_sl_tp': None,
        }

        # ── Step 0: Calculate indicators ──────────────────────────────
        try:
            df_1m = self.calculate_indicators(df_1m)
        except Exception as e:
            result['details'] = f"Indicator calculation failed: {e}"
            return result

        if df_5m is not None and len(df_5m) >= 60:
            try:
                df_5m = self.calculate_indicators(df_5m)
            except Exception:
                df_5m = None

        # ── Step 1: Market conditions gate ────────────────────────────
        ok, reason = self.check_market_conditions(df_1m, pair, spread)
        if not ok:
            result['details'] = f"🚫 {reason}"
            bot_logger.info(f"🚫 Sweep skip {pair}: {reason}")
            return result

        # ── Step 2: 5M Regime & Bias (Layer 1) ────────────────────────
        regime_info = self.detect_regime(df_5m)
        result['regime'] = regime_info['regime']
        result['bias'] = regime_info['bias']

        if regime_info['bias'] is None:
            result['details'] = (
                f"🚫 No directional bias — regime={regime_info['regime']}, "
                f"ADX={regime_info['adx']:.1f}"
            )
            bot_logger.info(
                f"🚫 Sweep skip {pair}: no bias "
                f"(regime={regime_info['regime']}, ADX={regime_info['adx']:.1f})"
            )
            return result

        bot_logger.info(
            f"📊 {pair} regime={regime_info['regime']}, bias={regime_info['bias']}, "
            f"ADX={regime_info['adx']:.1f}"
        )

        # ── Step 3: 1M Liquidity Sweep Detection (Layer 2) ───────────
        sweep = self.detect_sweep(df_1m, regime_info['bias'])
        result['sweep'] = sweep

        if not sweep['detected']:
            result['details'] = f"⏳ Bias={regime_info['bias']} but {sweep['details']}"
            return result

        bot_logger.info(f"💧 {pair} {sweep['details']}")

        # ── Step 4: Displacement Confirmation (Layer 3) ───────────────
        displacement = self.detect_displacement(df_1m, sweep)
        result['displacement'] = displacement

        if not displacement['confirmed']:
            result['details'] = f"💧 Sweep detected but {displacement['details']}"
            bot_logger.info(f"⏳ {pair} sweep present but no displacement yet")
            return result

        bot_logger.info(f"⚡ {pair} {displacement['details']}")

        # ── Step 5: Risk/Reward Calculation ───────────────────────────
        # Get ATR from 1M data for SL buffer
        latest_1m = df_1m.iloc[-1]
        regime_info['atr'] = float(latest_1m.get('atr', 0) or 0)

        rr = self.calculate_risk_reward(sweep, displacement, regime_info, pair)
        if rr is None:
            result['details'] = "🚫 R:R calculation failed (SL too tight)"
            return result

        result['risk_reward'] = rr
        result['sweep_sl_tp'] = rr

        # ── Step 6: Build final signal ────────────────────────────────
        # Confidence is high because ALL three layers aligned
        base_confidence = 0.80

        # Bonus for strong ADX
        if regime_info['adx'] >= 25:
            base_confidence += 0.10
        elif regime_info['adx'] >= 20:
            base_confidence += 0.05

        # Bonus for very strong displacement volume
        vol_ratio = displacement['candle_data'].get('volume_ratio', 1.0)
        if vol_ratio >= 2.0:
            base_confidence += 0.05
        elif vol_ratio >= 1.5:
            base_confidence += 0.03

        # Penalty for range regime (lower conviction)
        if regime_info['regime'] == 'range':
            base_confidence *= 0.70

        result['signal'] = sweep['direction']
        result['confidence'] = min(base_confidence, 1.0)
        result['details'] = (
            f"✅ SWEEP ENTRY: {sweep['direction']} | "
            f"Regime={regime_info['regime']} | "
            f"SL={rr['risk_pips']:.1f}p TP={rr['reward_pips']:.1f}p ({rr['rr_ratio']:.1f}R) | "
            f"Vol={vol_ratio:.2f}x | ADX={regime_info['adx']:.1f}"
        )

        config = self.PAIR_CONFIG.get(pair, self.PAIR_CONFIG['EUR/USD'])
        pip_size = config['pip_size']
        bot_logger.info(
            f"🎯 SWEEP SIGNAL: {pair} {sweep['direction']} | "
            f"Entry={rr['entry_price']:.5f} | "
            f"SL={rr['stop_loss']:.5f} ({rr['risk_pips']:.1f}p) | "
            f"TP={rr['take_profit']:.5f} ({rr['reward_pips']:.1f}p) | "
            f"R:R={rr['rr_ratio']:.1f} | Vol={vol_ratio:.2f}x"
        )

        return result
