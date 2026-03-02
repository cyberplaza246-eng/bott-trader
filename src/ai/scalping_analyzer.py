"""
ATR-Centric Volatility-Adaptive Scalping Strategy

Core Philosophy:
  - ATR drives EVERYTHING: entry filtering, SL/TP, regime adaptation
  - 5M EMA 20/50 provides directional bias
  - 1M pullback to EMA 20 + micro-structure break + volume confirmation
  - Dynamic SL = 0.8 x ATR(14), TP = 1.2-1.8R based on volatility regime

Entry Rules:
  LONG:
    1. 5M bias bullish (EMA20 > EMA50), ADX > 18 preferred
    2. 1M pullback toward EMA20
    3. RSI(14) between 40-60 (pre-expansion zone)
    4. Break of 5-candle micro-structure (close > high of last 5 candles)
    5. Bullish engulfing OR momentum candle confirmation
    6. Volume > 1.2x 20-period average
  SHORT: Mirror logic

Risk Management (fully dynamic - no fixed pips):
  SL = 0.8 x ATR(14) on entry timeframe
  TP = 1.4 x SL (base, modified by ATR regime: 1.2R contracting, 1.8R expanding)

Market Conditions Filter:
  - ATR must be above session minimum threshold
  - Spread <= 20% of ATR
  - ATR must be above its own 5-period rolling average
  - Current candle range must NOT exceed 2 x ATR (exhaustion)
  - Session = London Open (7-12 UTC) or NY Open (13-17 UTC)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from src.utils.logger import bot_logger


class ScalpingAnalyzer:
    """ATR-centric volatility-adaptive scalping analyzer."""

    # -- Pair-specific configuration (ATR-based, no fixed pips) ------
    PAIR_CONFIG = {
        'GBP/USD': {
            'session_atr_min': 0.00055,   # Min ATR to trade (~5.5 pips)
            'spread_sim': 0.00020,        # Simulated spread (2 pips)
            'pip_size': 0.0001,
            'pip_value_label': 'pips',
        },
        'EUR/USD': {
            'session_atr_min': 0.00040,   # Min ATR to trade (~4 pips)
            'spread_sim': 0.00015,        # Simulated spread (1.5 pips)
            'pip_size': 0.0001,
            'pip_value_label': 'pips',
        },
        'USD/JPY': {
            'session_atr_min': 0.060,     # Min ATR to trade (~6 pips in JPY)
            'spread_sim': 0.020,          # Simulated spread (2 pips)
            'pip_size': 0.01,
            'pip_value_label': 'pips',
        },
    }

    # -- ATR risk parameters (universal) -----------------------------
    SL_ATR_MULT = 0.8          # SL = 0.8 x ATR (grid-search optimal)
    TP_BASE_RATIO = 1.3        # TP = 1.3 x SL (flat across all regimes)
    TP_EXPANDING = 1.3         # Same as base — regime TP variance removed
    TP_CONTRACTING = 1.3       # Same as base — contracting regime skipped
    MIN_SL_SPREAD_MULT = 3     # Reject if SL < spread x 3
    MAX_SL_MEDIAN_MULT = 2.0   # Reject if SL > 2x rolling median SL

    # -- Session windows (UTC) ---------------------------------------
    LONDON_OPEN = {'start': 7, 'end': 12}
    NY_OPEN = {'start': 13, 'end': 17}

    # -- Entry parameters --------------------------------------------
    RSI_ENTRY_LOW = 40         # RSI pre-expansion zone lower bound
    RSI_ENTRY_HIGH = 60        # RSI pre-expansion zone upper bound
    VOLUME_SPIKE_THRESHOLD = 1.2  # Volume must be > 1.2x average
    MICRO_STRUCTURE_LOOKBACK = 5  # Candles for structure break
    ENTRY_THRESHOLD = 0.70     # Minimum confidence (grid-search optimal)

    def __init__(self, profit_mode='quick_wins', timeframe='5m'):
        """Initialize ATR-centric scalping analyzer.

        Args:
            profit_mode: 'quick_wins' or 'normal' (affects trailing, not SL/TP)
            timeframe: '1m' or '5m' - determines candle context
        """
        self.timeframe = timeframe
        self.profit_mode = profit_mode

        # Indicator periods
        self.rsi_period = 14       # RSI(14) per strategy spec
        self.atr_period = 14       # ATR(14)
        self.atr_sma_period = 5    # Rolling ATR average for filtering
        self.volume_period = 20    # Volume SMA for spike detection
        self.ema_periods = {'short': 20, 'medium': 50, 'long': 200}
        self.adx_period = 14       # ADX for trend strength

        mode_label = "QUICK_WINS" if profit_mode == 'quick_wins' else "NORMAL"
        bot_logger.info(
            f"🔪 Scalping Analyzer initialized (ATR-adaptive, {timeframe}) [{mode_label} mode]"
        )

    # =================================================================
    #  INDICATOR CALCULATION
    # =================================================================

    def calculate_indicators(self, df, timeframe=None):
        """Calculate all indicators needed for the strategy.

        Adds: rsi, ema_20, ema_50, ema_200, atr, atr_sma5,
              volume_sma, volume_ratio, candle_range, adx, di_plus, di_minus

        Args:
            df: DataFrame with open, high, low, close, volume
            timeframe: Override instance timeframe (optional)

        Returns:
            DataFrame with indicator columns added
        """
        df = df.copy()

        if len(df) < self.ema_periods['long']:
            return df

        close = df['close']
        high = df['high']
        low = df['low']

        # -- RSI(14) -------------------------------------------------
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss.replace(0, 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # -- EMAs ----------------------------------------------------
        df['ema_20'] = close.ewm(span=self.ema_periods['short'], adjust=False).mean()
        df['ema_50'] = close.ewm(span=self.ema_periods['medium'], adjust=False).mean()
        df['ema_200'] = close.ewm(span=self.ema_periods['long'], adjust=False).mean()

        # -- ATR(14) - True Range then rolling mean ------------------
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=self.atr_period).mean()

        # Rolling ATR average (for "ATR above its own average" filter)
        df['atr_sma5'] = df['atr'].rolling(window=self.atr_sma_period).mean()

        # ATR trend: count consecutive rising/falling bars
        atr_diff = df['atr'].diff()
        df['atr_rising'] = (atr_diff > 0).astype(int)
        df['atr_falling'] = (atr_diff < 0).astype(int)

        # Consecutive ATR rise/fall count (for regime detection)
        atr_rise_groups = df['atr_rising'].ne(df['atr_rising'].shift()).cumsum()
        df['atr_rise_streak'] = df.groupby(atr_rise_groups)['atr_rising'].cumsum()
        atr_fall_groups = df['atr_falling'].ne(df['atr_falling'].shift()).cumsum()
        df['atr_fall_streak'] = df.groupby(atr_fall_groups)['atr_falling'].cumsum()

        # -- Candle range (for exhaustion spike detection) -----------
        df['candle_range'] = high - low

        # -- Volume analysis -----------------------------------------
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['volume_sma'] = df['volume'].rolling(window=self.volume_period).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
        else:
            df['volume_sma'] = 0
            df['volume_ratio'] = 1.0  # Neutral if no volume data

        # -- ADX(14) for trend strength ------------------------------
        df = self._calculate_adx(df)

        # Fill NaN
        df = df.bfill().ffill()

        return df

    def _calculate_adx(self, df):
        """Calculate ADX, DI+, DI- manually (no pandas_ta dependency)."""
        high = df['high']
        low = df['low']
        period = self.adx_period

        # Directional movements
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)

        plus_dm_mask = (up_move > down_move) & (up_move > 0)
        minus_dm_mask = (down_move > up_move) & (down_move > 0)
        plus_dm[plus_dm_mask] = up_move[plus_dm_mask]
        minus_dm[minus_dm_mask] = down_move[minus_dm_mask]

        # Use ATR already calculated for TR smoothing
        atr_smooth = df['atr']

        # Smooth DM
        plus_dm_smooth = plus_dm.rolling(window=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period).mean()

        # DI+ and DI-
        df['di_plus'] = (plus_dm_smooth / (atr_smooth + 1e-10)) * 100
        df['di_minus'] = (minus_dm_smooth / (atr_smooth + 1e-10)) * 100

        # DX and ADX
        di_sum = df['di_plus'] + df['di_minus']
        di_diff = (df['di_plus'] - df['di_minus']).abs()
        dx = (di_diff / (di_sum + 1e-10)) * 100
        df['adx'] = dx.rolling(window=period).mean()

        return df

    # =================================================================
    #  MARKET CONDITIONS FILTER
    # =================================================================

    def check_market_conditions(self, df, pair, spread=None):
        """Gate check: should we even look for trades right now?

        Filters:
          1. Session = London Open (7-12 UTC) or NY Open (13-17 UTC)
          2. ATR > session minimum threshold for this pair
          3. ATR > its own 5-period rolling average (volatility not dying)
          4. Spread <= 20% of ATR (cost-effective)
          5. Current candle range <= 2 x ATR (not an exhaustion spike)

        Args:
            df: DataFrame with indicators calculated
            pair: Currency pair
            spread: Actual spread from broker (None = use simulated)

        Returns:
            (ok: bool, reason: str)
        """
        config = self.PAIR_CONFIG.get(pair, self.PAIR_CONFIG['EUR/USD'])
        latest = df.iloc[-1]
        atr = float(latest.get('atr', 0) or 0)

        # 1. Session filter
        now = datetime.now(timezone.utc)
        hour = now.hour
        in_london = self.LONDON_OPEN['start'] <= hour < self.LONDON_OPEN['end']
        in_ny = self.NY_OPEN['start'] <= hour < self.NY_OPEN['end']
        if not (in_london or in_ny):
            return False, f"🚫 Outside trading sessions (UTC {hour}:00) - need London 7-12 or NY 13-17"

        # 2. ATR floor
        session_atr_min = config['session_atr_min']
        if atr < session_atr_min:
            return False, (
                f"🚫 ATR too low: {atr:.5f} < {session_atr_min:.5f} "
                f"({atr / config['pip_size']:.1f} pips < {session_atr_min / config['pip_size']:.1f} pips)"
            )

        # 3. ATR above its 5-period rolling average
        atr_sma5 = float(latest.get('atr_sma5', 0) or 0)
        if atr_sma5 > 0 and atr < atr_sma5:
            return False, (
                f"🚫 ATR below rolling average: {atr:.5f} < SMA5 {atr_sma5:.5f} - volatility declining"
            )

        # 4. Spread <= 20% of ATR
        actual_spread = spread if spread is not None else config['spread_sim']
        if atr > 0 and actual_spread > 0.20 * atr:
            return False, (
                f"🚫 Spread too wide vs ATR: {actual_spread:.5f} > 20% of ATR {atr:.5f} "
                f"({actual_spread / atr * 100:.0f}%)"
            )

        # 5. No exhaustion spike
        candle_range = float(latest.get('candle_range', 0) or 0)
        if atr > 0 and candle_range > 2 * atr:
            return False, (
                f"🚫 Exhaustion spike: candle range {candle_range:.5f} > 2xATR {2 * atr:.5f}"
            )

        return True, "✓ Market conditions passed"

    # =================================================================
    #  5M BIAS DETECTION
    # =================================================================

    def detect_5m_bias(self, df):
        """Detect directional bias from 5-minute EMA alignment.

        Args:
            df: DataFrame with ema_20, ema_50, adx calculated

        Returns:
            dict with bias, strength, adx, direction (for compat)
        """
        if len(df) < 2:
            return {'bias': 'flat', 'strength': 'weak', 'adx': 0, 'direction': 'NONE'}

        latest = df.iloc[-1]
        ema_20 = float(latest.get('ema_20', 0) or 0)
        ema_50 = float(latest.get('ema_50', 0) or 0)
        adx = float(latest.get('adx', 0) or 0)

        if pd.isna(ema_20) or pd.isna(ema_50):
            return {'bias': 'flat', 'strength': 'weak', 'adx': 0, 'direction': 'NONE'}

        # EMA 20 vs EMA 50 determines bias
        if ema_20 > ema_50:
            bias = 'long'
            direction = 'BUY'
        elif ema_20 < ema_50:
            bias = 'short'
            direction = 'SELL'
        else:
            bias = 'flat'
            direction = 'NONE'

        # ADX strength classification (22 = grid-search-optimal minimum)
        if adx > 30:
            strength = 'strong'
        elif adx > 22:
            strength = 'preferred'
        else:
            strength = 'weak'

        return {
            'bias': bias,
            'strength': strength,
            'adx': adx,
            'direction': direction,
            'ema_50_200_alignment': 'bullish' if ema_20 > ema_50 else 'bearish',
            'price_position': 'above_200' if float(latest['close']) > float(latest.get('ema_200', 0) or 0) else 'below_200',
        }

    # =================================================================
    #  ATR REGIME DETECTION
    # =================================================================

    def detect_atr_regime(self, df):
        """Detect ATR volatility regime for TP adjustment.

        Returns:
            dict with regime, tp_ratio, atr_streak
        """
        latest = df.iloc[-1]
        atr_rise_streak = int(latest.get('atr_rise_streak', 0) or 0)
        atr_fall_streak = int(latest.get('atr_fall_streak', 0) or 0)

        if atr_rise_streak >= 5:
            return {
                'regime': 'expanding',
                'tp_ratio': self.TP_EXPANDING,
                'atr_streak': atr_rise_streak,
            }
        elif atr_fall_streak >= 5:
            return {
                'regime': 'contracting',
                'tp_ratio': self.TP_CONTRACTING,
                'atr_streak': -atr_fall_streak,
            }
        else:
            return {
                'regime': 'neutral',
                'tp_ratio': self.TP_BASE_RATIO,
                'atr_streak': atr_rise_streak - atr_fall_streak,
            }

    # =================================================================
    #  1M ENTRY DETECTION
    # =================================================================

    def detect_entry(self, df, bias_direction, pair='EUR/USD'):
        """Detect pullback entry with structure break + candle + volume.

        Scoring (threshold = 0.70):
          - EMA20 pullback zone:       +0.20
          - RSI 40-60 pre-expansion:   +0.15
          - Micro-structure break:     +0.25
          - Candle pattern:            +0.20
          - Volume > 1.2x average:     +0.20

        Args:
            df: DataFrame with indicators
            bias_direction: 'BUY' or 'SELL' (from 5M bias)
            pair: Currency pair

        Returns:
            dict: {ready, confidence, signals, details, direction}
        """
        setup = {
            'ready': False,
            'confidence': 0.0,
            'signals': [],
            'details': {},
            'direction': bias_direction,
        }

        if len(df) < max(30, self.MICRO_STRUCTURE_LOOKBACK + 2):
            return setup

        latest = df.iloc[-1]
        price = float(latest['close'])
        ema_20 = float(latest.get('ema_20', price))
        rsi = float(latest.get('rsi', 50))
        atr = float(latest.get('atr', 0))
        volume_ratio = float(latest.get('volume_ratio', 1.0))
        pip_size = self.PAIR_CONFIG.get(pair, self.PAIR_CONFIG['EUR/USD'])['pip_size']

        setup['details'] = {
            'price': price, 'ema_20': ema_20, 'rsi': rsi,
            'atr': atr, 'volume_ratio': volume_ratio,
        }

        # -- Check 1: Pullback toward EMA 20 zone -------------------
        distance_to_ema20 = abs(price - ema_20)
        pullback_zone = atr * 0.5 if atr > 0 else 0.0010

        if bias_direction == 'BUY':
            in_pullback = distance_to_ema20 <= pullback_zone and price >= (ema_20 - pullback_zone)
        else:
            in_pullback = distance_to_ema20 <= pullback_zone and price <= (ema_20 + pullback_zone)

        if in_pullback:
            pips_dist = distance_to_ema20 / pip_size
            setup['signals'].append(f"✓ Pullback to EMA20 zone ({pips_dist:.1f} pips away)")
            setup['confidence'] += 0.20
        else:
            setup['signals'].append(
                f"✗ Not in EMA20 pullback zone ({distance_to_ema20 / pip_size:.1f}p away, need < {pullback_zone / pip_size:.1f}p)"
            )
            return setup  # Hard requirement

        # -- Check 2: RSI(14) in 40-60 pre-expansion range ----------
        if self.RSI_ENTRY_LOW <= rsi <= self.RSI_ENTRY_HIGH:
            setup['signals'].append(f"✓ RSI {rsi:.1f} in pre-expansion zone ({self.RSI_ENTRY_LOW}-{self.RSI_ENTRY_HIGH})")
            setup['confidence'] += 0.15
        else:
            setup['signals'].append(f"✗ RSI {rsi:.1f} outside entry range ({self.RSI_ENTRY_LOW}-{self.RSI_ENTRY_HIGH})")
            return setup  # Hard requirement

        # -- Check 3: Micro-structure break (5-candle lookback) ------
        lookback = self.MICRO_STRUCTURE_LOOKBACK
        structure_window = df.iloc[-(lookback + 1):-1]

        if bias_direction == 'BUY':
            structure_high = structure_window['high'].max()
            structure_broken = price > structure_high
            if structure_broken:
                setup['signals'].append(
                    f"✓ Micro-structure break: close {price:.5f} > {lookback}-candle high {structure_high:.5f}"
                )
                setup['confidence'] += 0.25
            else:
                setup['signals'].append(
                    f"✗ No structure break: close {price:.5f} <= {lookback}-candle high {structure_high:.5f}"
                )
        else:
            structure_low = structure_window['low'].min()
            structure_broken = price < structure_low
            if structure_broken:
                setup['signals'].append(
                    f"✓ Micro-structure break: close {price:.5f} < {lookback}-candle low {structure_low:.5f}"
                )
                setup['confidence'] += 0.25
            else:
                setup['signals'].append(
                    f"✗ No structure break: close {price:.5f} >= {lookback}-candle low {structure_low:.5f}"
                )

        # -- Check 4: Candle pattern - engulfing or momentum ---------
        candle_confirmed = self._detect_candle_pattern(df, bias_direction)
        if candle_confirmed['detected']:
            setup['signals'].append(f"✓ {candle_confirmed['pattern']}")
            setup['confidence'] += 0.20
        else:
            setup['signals'].append("✗ No engulfing/momentum candle confirmation")

        # -- Check 5: Volume confirmation ----------------------------
        if volume_ratio > self.VOLUME_SPIKE_THRESHOLD:
            setup['signals'].append(
                f"✓ Volume spike: {volume_ratio:.2f}x average (> {self.VOLUME_SPIKE_THRESHOLD}x)"
            )
            setup['confidence'] += 0.20
        else:
            setup['signals'].append(
                f"✗ Volume {volume_ratio:.2f}x average (need > {self.VOLUME_SPIKE_THRESHOLD}x)"
            )

        setup['confidence'] = min(setup['confidence'], 1.0)
        setup['ready'] = setup['confidence'] >= self.ENTRY_THRESHOLD

        return setup

    def _detect_candle_pattern(self, df, direction):
        """Detect bullish/bearish engulfing or momentum candle."""
        if len(df) < 6:
            return {'detected': False, 'pattern': 'none'}

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        curr_open = float(curr['open'])
        curr_close = float(curr['close'])
        curr_high = float(curr['high'])
        curr_low = float(curr['low'])
        prev_open = float(prev['open'])
        prev_close = float(prev['close'])
        prev_high = float(prev['high'])
        prev_low = float(prev['low'])

        curr_body = abs(curr_close - curr_open)
        prev_body = abs(prev_close - prev_open)
        curr_range = curr_high - curr_low

        if direction == 'BUY':
            # Bullish engulfing
            is_engulfing = (
                curr_close > curr_open and
                prev_close < prev_open and
                curr_body > prev_body and
                curr_close > prev_high
            )
            if is_engulfing:
                return {'detected': True, 'pattern': 'Bullish engulfing candle'}

            # Momentum candle: strong close above prior 5-candle high
            if curr_range > 0:
                body_ratio = curr_body / curr_range
                if (curr_close > curr_open and
                        body_ratio > 0.60 and
                        curr_close > df['high'].iloc[-6:-1].max()):
                    return {'detected': True, 'pattern': f'Momentum candle (body {body_ratio:.0%})'}

        else:  # SELL
            is_engulfing = (
                curr_close < curr_open and
                prev_close > prev_open and
                curr_body > prev_body and
                curr_close < prev_low
            )
            if is_engulfing:
                return {'detected': True, 'pattern': 'Bearish engulfing candle'}

            if curr_range > 0:
                body_ratio = curr_body / curr_range
                if (curr_close < curr_open and
                        body_ratio > 0.60 and
                        curr_close < df['low'].iloc[-6:-1].min()):
                    return {'detected': True, 'pattern': f'Momentum candle (body {body_ratio:.0%})'}

        return {'detected': False, 'pattern': 'none'}

    # =================================================================
    #  DYNAMIC RISK/REWARD (ATR-BASED)
    # =================================================================

    def calculate_risk_reward(self, df, direction, pair='EUR/USD',
                              spread=None, tp_ratio_override=None,
                              recent_sl_values=None):
        """Calculate ATR-based SL and TP. Returns None to reject trade.

        SL = 0.8 x ATR(14)
        TP = tp_ratio x SL

        Reject if:
          - SL < spread x 3
          - SL > 2x rolling median SL

        Args:
            df: DataFrame with indicators
            direction: 'BUY' or 'SELL'
            pair: Currency pair
            spread: Actual spread (None = simulated)
            tp_ratio_override: From ATR regime
            recent_sl_values: List of recent SL distances for median check

        Returns:
            dict with SL/TP details, or None to reject
        """
        config = self.PAIR_CONFIG.get(pair, self.PAIR_CONFIG['EUR/USD'])
        latest = df.iloc[-1]
        entry_price = float(latest['close'])
        atr = float(latest.get('atr', 0) or 0)
        pip_size = config['pip_size']

        if atr <= 0:
            bot_logger.warning(f"Cannot calculate R/R: ATR is {atr}")
            return None

        # SL distance = 0.8 x ATR
        sl_distance = atr * self.SL_ATR_MULT

        # Reject: SL < spread x 3
        actual_spread = spread if spread is not None else config['spread_sim']
        min_sl_by_spread = actual_spread * self.MIN_SL_SPREAD_MULT
        if sl_distance < min_sl_by_spread:
            bot_logger.info(
                f"🚫 Trade rejected: SL {sl_distance / pip_size:.1f}p < "
                f"spread x 3 = {min_sl_by_spread / pip_size:.1f}p"
            )
            return None

        # Reject: SL > 2x rolling median SL (abnormal)
        if recent_sl_values and len(recent_sl_values) >= 10:
            median_sl = float(np.median(recent_sl_values))
            if median_sl > 0 and sl_distance > self.MAX_SL_MEDIAN_MULT * median_sl:
                bot_logger.info(
                    f"🚫 Trade rejected: SL {sl_distance / pip_size:.1f}p > "
                    f"2x median {median_sl / pip_size:.1f}p (abnormal volatility)"
                )
                return None

        # TP distance = ratio x SL
        tp_ratio = tp_ratio_override if tp_ratio_override else self.TP_BASE_RATIO
        tp_distance = sl_distance * tp_ratio

        # Calculate price levels
        if direction == 'BUY':
            stop_loss = round(entry_price - sl_distance, 5)
            take_profit = round(entry_price + tp_distance, 5)
        else:
            stop_loss = round(entry_price + sl_distance, 5)
            take_profit = round(entry_price - tp_distance, 5)

        sl_pips = sl_distance / pip_size
        tp_pips = tp_distance / pip_size

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'take_profit_1': take_profit,
            'take_profit_2': take_profit,
            'sl_distance': sl_distance,
            'tp_distance': tp_distance,
            'risk_pips': sl_pips,
            'reward_pips_1': tp_pips,
            'reward_pips_2': tp_pips,
            'rr_ratio': tp_ratio,
            'atr': atr,
            'atr_sl_mult': self.SL_ATR_MULT,
            'tp_ratio_used': tp_ratio,
            'spread': actual_spread,
        }

    # =================================================================
    #  MAIN SIGNAL GENERATION
    # =================================================================

    def get_signal(self, df, pair='EUR/USD', timeframe=None,
                   df_5m=None, spread=None, recent_sl_values=None):
        """Generate scalping buy/sell signal.

        Flow:
          1. Calculate indicators
          2. Check market conditions (ATR floor, spread, exhaustion, session)
          3. Detect directional bias (from 5M or inferred)
          4. Detect entry (pullback + structure + candle + volume)
          5. Calculate ATR-based R/R (may reject)
          6. Return signal

        Args:
            df: DataFrame with OHLCV data (200+ candles)
            pair: Currency pair
            timeframe: Override timeframe
            df_5m: Optional 5M DataFrame for bias detection
            spread: Actual broker spread (None = simulated)
            recent_sl_values: Recent SL values for median check

        Returns:
            dict with signal, confidence, setup, SL, TP, etc.
        """
        result = {
            'signal': 'SKIP',
            'confidence': 0.0,
            'setup': 'none',
            'entry_price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'risk_reward': {},
            'reasons': [],
            'trend': {},
            'atr_regime': 'neutral',
            'atr_tp_ratio': self.TP_BASE_RATIO,
        }

        if df is None or len(df) < 200:
            result['reasons'].append('Insufficient data (need 200+ candles)')
            return result

        # 1. Calculate indicators
        df = self.calculate_indicators(df)

        # 2. Market conditions gate
        ok, reason = self.check_market_conditions(df, pair, spread)
        result['reasons'].append(reason)
        if not ok:
            return result

        # 3. Detect directional bias
        if df_5m is not None and len(df_5m) >= 200:
            bias_df = self.calculate_indicators(df_5m)
        else:
            bias_df = df

        bias = self.detect_5m_bias(bias_df)
        result['trend'] = bias

        if bias['direction'] == 'NONE':
            result['reasons'].append(
                f"No clear bias: EMA20 ~ EMA50 (ADX {bias['adx']:.1f})"
            )
            return result

        result['reasons'].append(
            f"✓ {bias['bias'].upper()} bias (EMA20 {'>' if bias['bias'] == 'long' else '<'} EMA50, "
            f"ADX {bias['adx']:.1f} [{bias['strength']}])"
        )

        if bias['strength'] == 'weak':
            result['reasons'].append(f"⚠ ADX {bias['adx']:.1f} < 22 (weak trend — skipping)")
            return result

        # 4. Detect ATR regime (for TP adjustment & contracting skip)
        atr_regime = self.detect_atr_regime(df)
        result['atr_regime'] = atr_regime['regime']
        result['atr_tp_ratio'] = atr_regime['tp_ratio']

        result['reasons'].append(
            f"ATR regime: {atr_regime['regime']} -> TP ratio {atr_regime['tp_ratio']:.1f}R "
            f"(streak: {atr_regime['atr_streak']})"
        )

        # Skip contracting regime — backtest shows -$641 bleed
        if atr_regime['regime'] == 'contracting':
            result['reasons'].append('🚫 Contracting ATR regime — sitting out')
            return result

        # 5. Detect entry
        entry = self.detect_entry(df, bias['direction'], pair)
        result['reasons'].extend(entry['signals'])

        if not entry['ready']:
            result['reasons'].append(
                f"Entry not ready: confidence {entry['confidence']:.2f} "
                f"< threshold {self.ENTRY_THRESHOLD:.2f}"
            )
            return result

        # 6. Calculate ATR-based R/R
        rr = self.calculate_risk_reward(
            df, bias['direction'], pair,
            spread=spread,
            tp_ratio_override=atr_regime['tp_ratio'],
            recent_sl_values=recent_sl_values,
        )

        if rr is None:
            result['reasons'].append("🚫 R/R calculation rejected trade (SL too small or abnormal)")
            return result

        # 7. Build result
        current_price = float(df['close'].iloc[-1])

        result['signal'] = bias['direction']
        result['confidence'] = entry['confidence']
        result['setup'] = f"pullback_{bias['bias']}"
        result['entry_price'] = current_price
        result['stop_loss'] = rr['stop_loss']
        result['take_profit'] = rr['take_profit']
        result['risk_reward'] = rr

        result['reasons'].append(
            f"{'Buy' if bias['direction'] == 'BUY' else 'Sell'} Entry: "
            f"SL {rr['risk_pips']:.1f}p ({self.SL_ATR_MULT}xATR), "
            f"TP {rr['reward_pips_1']:.1f}p ({rr['tp_ratio_used']:.1f}R)"
        )

        bot_logger.info(
            f"🔪 SCALP SIGNAL: {bias['direction']} {pair} | "
            f"Conf {entry['confidence']:.0%} | "
            f"SL {rr['risk_pips']:.1f}p | TP {rr['reward_pips_1']:.1f}p | "
            f"ATR regime: {atr_regime['regime']}"
        )

        return result
