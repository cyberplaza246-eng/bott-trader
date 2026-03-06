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
import json
import os
from datetime import datetime, timezone
from src.utils.logger import bot_logger


class ScalpingAnalyzer:
    """ATR-centric volatility-adaptive scalping analyzer."""

    # -- Pair-specific configuration (ATR-based, no fixed pips) ------
    PAIR_CONFIG = {
        'GBP/USD': {
            'session_atr_min': 0.00055,   # Min ATR to trade (~5.5 pips)
            'spread_sim': 0.00010,        # 1.0 pip (ECN TradersWay)
            'pip_size': 0.0001,
            'pip_value_label': 'pips',
        },
        'EUR/USD': {
            'session_atr_min': 0.00040,   # Min ATR to trade (~4 pips)
            'spread_sim': 0.00006,        # 0.6 pips (ECN TradersWay)
            'pip_size': 0.0001,
            'pip_value_label': 'pips',
        },
        'USD/JPY': {
            'session_atr_min': 0.080,     # Min ATR to trade (~8 pips in JPY) - increased
            'spread_sim': 0.008,          # 0.8 pips (ECN TradersWay)
            'pip_size': 0.01,
            'pip_value_label': 'pips',
        },
    }

    # -- Structure-based risk parameters (5m scalping) ---------------
    SL_ATR_MULT = 1.2          # Fallback: SL = 1.2 x ATR (wider for noise clearance)
    SL_STRUCTURE_BUFFER = 0.50  # Buffer below swing low/high (50% of ATR) - room for wicks
    SL_MAX_ATR_MULT = 2.0      # Max SL = 2.0x ATR (allow wider for structure)
    SL_MIN_ATR_MULT = 1.0      # Min SL = 1.0x ATR (floor for 5m scalps)
    SL_MIN_SPREAD_MULT = 2     # Realistic: Min SL = 2x spread (reduced from 3x for low-vol periods)
    
    TP_BASE_RATIO = 1.3        # TP = 1.3 x SL — tighter, realistic scalping
    TP_EXPANDING = 1.5         # Wider TP only in expanding volatility
    TP_CONTRACTING = 1.2       # Tight TP in contracting volatility
    
    # Session-aware TP ratios for volatility adaptation
    TP_ASIAN_SESSION = 1.2     # Lower volatility sessions (21-07 UTC)
    TP_LONDON_OPEN = 1.4       # Breakout volatility (07-12 UTC) 
    TP_NY_OVERLAP = 1.5        # Highest volatility (13-17 UTC)
    TP_QUIET_HOURS = 1.1       # Minimal movement periods (17-21 UTC)
    TP_MIN_STRUCTURE_RR = 1.1  # Min 1.1R when targeting structure levels
    TP_MAX_SCALP_RR = 1.6      # Max 1.6R for 5m scalping (realistic)
    
    MIN_SL_SPREAD_MULT = 2     # Reject if SL < spread x 2 (realistic for scalping)
    MAX_SL_MEDIAN_MULT = 2.5   # Reject if SL > 2.5x rolling median SL
    STRUCTURE_LOOKBACK = 20    # Bars to look back for swing highs/lows
    PRICE_DRIFT_TOLERANCE_ATR = 3.0  # Max price drift from signal entry (in ATR units)
    SL_MULTIPLIER_BY_PAIR = {}       # Per-pair SL multipliers (loaded from risk_overrides.json)
    AUTO_LEARNING = {}               # Auto-learning config (loaded from risk_overrides.json)

    # -- Session windows (UTC) ---------------------------------------
    LONDON_OPEN = {'start': 7, 'end': 12}
    NY_OPEN = {'start': 13, 'end': 17}
    ASIAN_SESSION = {'start': 21, 'end': 7}  # Wraps midnight: 21:00-07:00
    QUIET_HOURS = {'start': 17, 'end': 21}   # Post-NY, pre-Asian

    # -- Entry parameters --------------------------------------------
    RSI_ENTRY_LOW = 40         # RSI pre-expansion zone lower bound
    RSI_ENTRY_HIGH = 60        # RSI pre-expansion zone upper bound
    VOLUME_SPIKE_THRESHOLD = 1.05  # Volume must be > 1.05x average (was 1.2 — too strict for forex tick vol)
    MICRO_STRUCTURE_LOOKBACK = 5  # Candles for structure break
    ENTRY_THRESHOLD = 0.45     # Lowered to 0.45 (partial credit removed, honest scoring)
    RISK_OVERRIDES_PATH = 'data/risk_overrides.json'

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

        # Allow safe runtime tuning from file without code edits.
        self._load_risk_overrides()

        mode_label = "QUICK_WINS" if profit_mode == 'quick_wins' else "NORMAL"
        bot_logger.info(
            f"🔪 Scalping Analyzer initialized (ATR-adaptive, {timeframe}) [{mode_label} mode]"
        )

    def _load_risk_overrides(self):
        """Load optional TP/SL overrides from JSON file.

        Expected schema (all keys optional):
            {
              "sl_atr_mult": 1.4,
              "sl_min_atr_mult": 1.0,
              "sl_max_atr_mult": 2.0,
              "sl_multiplier_by_pair": {"EUR/USD": 1.3, "GBP/USD": 1.4, "USD/JPY": 1.5},
              "tp_base_ratio": 1.2,
              "tp_expanding": 1.4,
              "tp_contracting": 1.1,
              "tp_asian_session": 1.1,
              "tp_london_open": 1.3,
              "tp_ny_overlap": 1.35,
              "tp_quiet_hours": 1.05,
              "entry_confidence_threshold": 0.50,
              "price_drift_tolerance_atr": 3.0,
              "auto_learning": {
                "enabled": true,
                "sl_hit_rate_threshold": 0.60,
                "tp_hit_rate_threshold": 0.70,
                "adjustment_interval_trades": 30
              }
            }
        """
        path = self.RISK_OVERRIDES_PATH
        if not os.path.exists(path):
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)

            key_map = {
                'sl_atr_mult': ('SL_ATR_MULT', 0.6, 2.5),
                'sl_min_atr_mult': ('SL_MIN_ATR_MULT', 0.5, 2.0),
                'sl_max_atr_mult': ('SL_MAX_ATR_MULT', 1.0, 3.0),
                'tp_base_ratio': ('TP_BASE_RATIO', 0.8, 3.0),
                'tp_expanding': ('TP_EXPANDING', 0.8, 3.5),
                'tp_contracting': ('TP_CONTRACTING', 0.8, 2.5),
                'tp_asian_session': ('TP_ASIAN_SESSION', 0.8, 3.0),
                'tp_london_open': ('TP_LONDON_OPEN', 0.8, 3.5),
                'tp_ny_overlap': ('TP_NY_OVERLAP', 0.8, 3.5),
                'tp_quiet_hours': ('TP_QUIET_HOURS', 0.8, 2.5),
                'entry_threshold': ('ENTRY_THRESHOLD', 0.2, 0.9),
                'entry_confidence_threshold': ('ENTRY_THRESHOLD', 0.2, 0.9),
                'price_drift_tolerance_atr': ('PRICE_DRIFT_TOLERANCE_ATR', 1.5, 5.0),
            }

            applied = []
            for key, (attr, floor, ceil) in key_map.items():
                if key not in payload:
                    continue
                try:
                    raw_value = float(payload[key])
                except (TypeError, ValueError):
                    continue

                value = max(floor, min(ceil, raw_value))
                setattr(self, attr, value)
                applied.append(f"{attr}={value:.3f}")

            # Load pair-specific SL multipliers
            if 'sl_multiplier_by_pair' in payload and isinstance(payload['sl_multiplier_by_pair'], dict):
                self.SL_MULTIPLIER_BY_PAIR = {}
                for pair, mult in payload['sl_multiplier_by_pair'].items():
                    try:
                        self.SL_MULTIPLIER_BY_PAIR[pair] = max(0.5, min(2.5, float(mult)))
                        applied.append(f"SL_{pair}={mult:.2f}")
                    except (TypeError, ValueError):
                        pass

            # Load auto-learning config
            if 'auto_learning' in payload and isinstance(payload['auto_learning'], dict):
                self.AUTO_LEARNING = payload['auto_learning']
                if self.AUTO_LEARNING.get('enabled'):
                    applied.append("AUTO_LEARNING=ON")

            if applied:
                bot_logger.info(
                    f"🛠️ Applied risk overrides from {path}: " + ", ".join(applied)
                )
        except Exception as e:
            bot_logger.warning(f"Could not load risk overrides from {path}: {e}")

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

    def _get_current_session(self, timestamp=None):
        """Determine current trading session for TP ratio adjustment.
        
        Args:
            timestamp: UTC timestamp, if None uses current time
            
        Returns:
            str: 'london', 'ny_overlap', 'asian', 'quiet'
        """
        import datetime
        if timestamp is None:
            current_hour = datetime.datetime.utcnow().hour
        else:
            current_hour = timestamp.hour if hasattr(timestamp, 'hour') else datetime.datetime.utcfromtimestamp(timestamp).hour
        
        if self.LONDON_OPEN['start'] <= current_hour < self.LONDON_OPEN['end']:
            return 'london'
        elif self.NY_OPEN['start'] <= current_hour < self.NY_OPEN['end']:
            return 'ny_overlap'
        elif self.QUIET_HOURS['start'] <= current_hour < self.QUIET_HOURS['end']:
            return 'quiet'
        else:  # Asian session (wraps midnight)
            return 'asian'
    
    def _get_effective_spread(self, pair, broker_spread, config):
        """Get effective spread using hybrid broker/config approach.
        
        Args:
            pair: Currency pair
            broker_spread: Spread from broker (may be None)
            config: Pair configuration
            
        Returns:
            float: Effective spread to use for calculations
        """
        config_spread = config['spread_sim']
        
        # If no broker spread, use config
        if broker_spread is None:
            return config_spread
            
        # JPY pairs: use hybrid approach with safety limits
        if 'JPY' in pair:
            # Use broker spread if reasonable, otherwise fall back to config
            max_reasonable_jpy_spread = 0.020  # 2.0 pips max for ECN JPY
            if broker_spread <= max_reasonable_jpy_spread:
                return broker_spread
            else:
                return config_spread
        else:
            # Major pairs: use broker spread if reasonable, with safety limits
            max_reasonable_major_spread = config_spread * 1.8  # Max 1.8× config spread
            if broker_spread <= max_reasonable_major_spread:
                return broker_spread
            else:
                return config_spread
    
    def _get_session_tp_ratio(self, base_ratio, session=None):
        """Get session-adaptive TP ratio based on expected volatility.
        
        Args:
            base_ratio: Base ratio from volatility regime
            session: Session override, if None detects current
            
        Returns:
            float: Adjusted TP ratio
        """
        if session is None:
            session = self._get_current_session()
        
        session_multipliers = {
            'london': self.TP_LONDON_OPEN / self.TP_BASE_RATIO,     # 2.0/1.8 = 1.11×
            'ny_overlap': self.TP_NY_OVERLAP / self.TP_BASE_RATIO,  # 2.2/1.8 = 1.22×
            'asian': self.TP_ASIAN_SESSION / self.TP_BASE_RATIO,    # 1.5/1.8 = 0.83×
            'quiet': self.TP_QUIET_HOURS / self.TP_BASE_RATIO,      # 1.3/1.8 = 0.72×
        }
        
        multiplier = session_multipliers.get(session, 1.0)
        return base_ratio * multiplier

    def check_market_conditions(self, df, pair, spread=None):
        """Gate check: should we even look for trades right now?

        Filters:
          1. Session = London through NY close (7-20 UTC)
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

        # 1. Session filter — use candle timestamp when available, else wall clock
        candle_dt = latest.get('datetime', None)
        if candle_dt is not None:
            try:
                candle_dt = pd.to_datetime(candle_dt)
                hour = candle_dt.hour
            except Exception:
                hour = datetime.now(timezone.utc).hour
        else:
            hour = datetime.now(timezone.utc).hour

        # Session filter disabled — trade all forex hours
        # (bot.py handles per-pair session gating via SCALPING_SESSION_WINDOWS)
        # Peak liquidity bonus applied separately via OPTIMAL_HOURS_UTC

        # 2. ATR floor
        session_atr_min = config['session_atr_min']
        if atr < session_atr_min:
            return False, (
                f"🚫 ATR too low: {atr:.5f} < {session_atr_min:.5f} "
                f"({atr / config['pip_size']:.1f} pips < {session_atr_min / config['pip_size']:.1f} pips)"
            ), False

        # 3. ATR above its 5-period rolling average (soft gate — penalty not block)
        atr_sma5 = float(latest.get('atr_sma5', 0) or 0)
        atr_declining = False
        if atr_sma5 > 0 and atr < atr_sma5:
            atr_declining = True
            # Don't return early — apply penalty in entry scoring instead
            pass

        # 4. Spread <= 20% of ATR
        actual_spread = spread if spread is not None else config['spread_sim']
        if atr > 0 and actual_spread > 0.20 * atr:
            return False, (
                f"🚫 Spread too wide vs ATR: {actual_spread:.5f} > 20% of ATR {atr:.5f} "
                f"({actual_spread / atr * 100:.0f}%)"
            ), False

        # 5. No exhaustion spike
        candle_range = float(latest.get('candle_range', 0) or 0)
        if atr > 0 and candle_range > 2 * atr:
            return False, (
                f"🚫 Exhaustion spike: candle range {candle_range:.5f} > 2xATR {2 * atr:.5f}"
            ), False

        return True, "✓ Market conditions passed", atr_declining

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

        # ADX strength classification (15 minimum for any directional move)
        if adx > 30:
            strength = 'strong'
        elif adx > 20:
            strength = 'preferred'
        elif adx > 15:
            strength = 'moderate'
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

        Scoring (threshold = 0.55):  # lowered to allow more setups
          - EMA20 pullback zone:       +0.25
          - Micro-structure break:     +0.30
          - Candle pattern:            +0.25
          - Volume > 1.2x average:     +0.20
          (RSI removed — handled exclusively by LiquiditySweep model)

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
            setup['confidence'] += 0.25
        else:
            setup['signals'].append(
                f"✗ Not in EMA20 pullback zone ({distance_to_ema20 / pip_size:.1f}p away)"
            )

        # -- (RSI filter removed — RSI is handled by LiquiditySweep model only) --

        # -- Check 2: Micro-structure break (5-candle lookback) ------
        lookback = self.MICRO_STRUCTURE_LOOKBACK
        structure_window = df.iloc[-(lookback + 1):-1]

        if bias_direction == 'BUY':
            structure_high = structure_window['high'].max()
            structure_broken = price > structure_high
            if structure_broken:
                setup['signals'].append(
                    f"✓ Micro-structure break: close {price:.5f} > {lookback}-candle high {structure_high:.5f}"
                )
                setup['confidence'] += 0.30
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
                setup['confidence'] += 0.30
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
    #  STRUCTURE-BASED SL/TP METHODS (5M SCALPING)
    # =================================================================

    def _find_structure_stop_loss(self, df, direction, entry_price, atr, pair='EUR/USD'):
        """Find structure-based stop loss using recent swing highs/lows.

        For BUY: Look for recent swing low below entry  
        For SELL: Look for recent swing high above entry
        
        Prioritizes significant levels with stronger patterns and adequate distance.

        Args:
            df: DataFrame with OHLCV
            direction: 'BUY' or 'SELL'
            entry_price: Current price
            atr: Current ATR value
            pair: Currency pair (for JPY-specific logic)

        Returns:
            dict: {distance, level, reason} or None
        """
        if len(df) < self.STRUCTURE_LOOKBACK:
            return None

        lookback_data = df.iloc[-self.STRUCTURE_LOOKBACK:]
        atr_buffer = atr * self.SL_STRUCTURE_BUFFER

        if direction == 'BUY':
            # Enhanced swing low detection with volume and significance weighting
            swing_lows = []
            volume_avg = lookback_data['volume'].rolling(window=5).mean()
            
            for i in range(2, len(lookback_data) - 1):  # 3-candle pattern
                current_row = lookback_data.iloc[i]
                current_low = current_row['low']
                prev_low = lookback_data.iloc[i-1]['low']
                next_low = lookback_data.iloc[i+1]['low']
                current_volume = current_row.get('volume', 1.0)
                avg_volume = volume_avg.iloc[i] if i < len(volume_avg) else 1.0
                
                # Basic swing low: lower than both neighbors
                if current_low < prev_low and current_low < next_low:
                    distance = entry_price - current_low
                    if distance > 0 and distance <= atr * 2.0:  # Within reasonable range
                        
                        # Calculate level significance score
                        significance = 1.0
                        
                        # Volume confirmation: higher volume = more significant
                        volume_ratio = current_volume / max(avg_volume, 0.001)
                        if volume_ratio > 1.2:  # 20% above average
                            significance += 0.3
                        elif volume_ratio > 1.0:
                            significance += 0.1
                        
                        # Multiple-touch validation: look for retests of this level
                        level_tolerance = atr * 0.1  # 10% ATR tolerance
                        touch_count = 0
                        for j in range(max(0, i-15), min(len(lookback_data), i+5)):  # Check ±15 candles
                            if j != i:  # Don't count the original swing
                                low_j = lookback_data.iloc[j]['low']
                                if abs(low_j - current_low) <= level_tolerance:
                                    touch_count += 1
                        
                        # Multiple touches increase significance
                        if touch_count >= 2:
                            significance += 0.4
                        elif touch_count == 1:
                            significance += 0.2
                        
                        # Distance weighting: closer levels preferred but not lowest priority
                        distance_weight = 1.0 / (1.0 + distance / (atr * 0.5))
                        final_score = significance * distance_weight
                        
                        swing_lows.append((current_low, distance, len(lookback_data) - i, final_score, touch_count))

            if swing_lows:
                # Sort by significance score (highest first)
                swing_lows.sort(key=lambda x: x[3], reverse=True)
                structure_level = swing_lows[0][0]
                touch_count = swing_lows[0][4]
                
                sl_level = structure_level - atr_buffer
                sl_distance = entry_price - sl_level
                
                # Ensure reasonable bounds
                min_sl = atr * self.SL_MIN_ATR_MULT
                max_sl = atr * self.SL_MAX_ATR_MULT
                
                sl_distance = max(min_sl, min(sl_distance, max_sl))
                sl_level = entry_price - sl_distance
                
                touch_text = f" ({touch_count} touches)" if touch_count > 0 else ""
                return {
                    'distance': sl_distance,
                    'level': sl_level,
                    'reason': f'significant swing low {structure_level:.5f}{touch_text} + {atr_buffer/atr:.1f}×ATR buffer'
                }

        elif direction == 'SELL':
            # Enhanced swing high detection with volume and significance weighting  
            swing_highs = []
            volume_avg = lookback_data['volume'].rolling(window=5).mean()
            
            for i in range(2, len(lookback_data) - 1):  # 3-candle pattern
                current_row = lookback_data.iloc[i]
                current_high = current_row['high']
                prev_high = lookback_data.iloc[i-1]['high']
                next_high = lookback_data.iloc[i+1]['high']
                current_volume = current_row.get('volume', 1.0)
                avg_volume = volume_avg.iloc[i] if i < len(volume_avg) else 1.0
                
                # Basic swing high: higher than both neighbors
                if current_high > prev_high and current_high > next_high:
                    distance = current_high - entry_price
                    if distance > 0 and distance <= atr * 2.0:  # Within reasonable range
                        
                        # Calculate level significance score
                        significance = 1.0
                        
                        # Volume confirmation: higher volume = more significant
                        volume_ratio = current_volume / max(avg_volume, 0.001)
                        if volume_ratio > 1.2:  # 20% above average
                            significance += 0.3
                        elif volume_ratio > 1.0:
                            significance += 0.1
                        
                        # Multiple-touch validation: look for retests of this level
                        level_tolerance = atr * 0.1  # 10% ATR tolerance
                        touch_count = 0
                        for j in range(max(0, i-15), min(len(lookback_data), i+5)):  # Check ±15 candles
                            if j != i:  # Don't count the original swing
                                high_j = lookback_data.iloc[j]['high']
                                if abs(high_j - current_high) <= level_tolerance:
                                    touch_count += 1
                        
                        # Multiple touches increase significance
                        if touch_count >= 2:
                            significance += 0.4
                        elif touch_count == 1:
                            significance += 0.2
                        
                        # Distance weighting: closer levels preferred but not lowest priority
                        distance_weight = 1.0 / (1.0 + distance / (atr * 0.5))
                        final_score = significance * distance_weight
                        
                        swing_highs.append((current_high, distance, len(lookback_data) - i, final_score, touch_count))

            if swing_highs:
                # Sort by significance score (highest first)
                swing_highs.sort(key=lambda x: x[3], reverse=True)
                structure_level = swing_highs[0][0]
                touch_count = swing_highs[0][4]
                
                sl_level = structure_level + atr_buffer
                sl_distance = sl_level - entry_price
                
                # Ensure reasonable bounds
                min_sl = atr * self.SL_MIN_ATR_MULT
                max_sl = atr * self.SL_MAX_ATR_MULT
                
                sl_distance = max(min_sl, min(sl_distance, max_sl))
                sl_level = entry_price + sl_distance
                
                touch_text = f" ({touch_count} touches)" if touch_count > 0 else ""
                return {
                    'distance': sl_distance,
                    'level': sl_level,
                    'reason': f'significant swing high {structure_level:.5f}{touch_text} + {atr_buffer/atr:.1f}×ATR buffer'
                }

        return None

    def _find_structure_take_profit(self, df, direction, entry_price, sl_distance, base_tp_ratio):
        """Find structure-based take profit using S/R levels.

        Args:
            df: DataFrame with OHLCV
            direction: 'BUY' or 'SELL'
            entry_price: Entry price
            sl_distance: Stop loss distance
            base_tp_ratio: Fallback TP ratio

        Returns:
            dict: {distance, ratio, reason} or None
        """
        if len(df) < self.STRUCTURE_LOOKBACK:
            return None

        lookback_data = df.iloc[-self.STRUCTURE_LOOKBACK:]

        if direction == 'BUY':
            # Find resistance levels (swing highs)
            resistance_levels = []
            for i in range(2, len(lookback_data) - 2):
                current_high = lookback_data.iloc[i]['high']
                prev_high = lookback_data.iloc[i-1]['high']
                next_high = lookback_data.iloc[i+1]['high']
                
                if current_high >= prev_high and current_high >= next_high:
                    resistance_levels.append(float(current_high))

            # Find the best resistance level above entry
            best_tp = None
            for level in [res for res in resistance_levels if res > entry_price]:
                tp_distance = level - entry_price
                rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
                
                # Must meet minimum R:R and be within scalping limits
                if self.TP_MIN_STRUCTURE_RR <= rr_ratio <= self.TP_MAX_SCALP_RR:
                    if best_tp is None or rr_ratio < best_tp['ratio']:  # Prefer closer target
                        best_tp = {
                            'distance': tp_distance,
                            'ratio': rr_ratio,
                            'reason': f'resistance at {level:.5f}'
                        }
                        
            return best_tp

        else:  # SELL
            # Find support levels (swing lows)
            support_levels = []
            for i in range(2, len(lookback_data) - 2):
                current_low = lookback_data.iloc[i]['low']
                prev_low = lookback_data.iloc[i-1]['low']
                next_low = lookback_data.iloc[i+1]['low']
                
                if current_low <= prev_low and current_low <= next_low:
                    support_levels.append(float(current_low))

            # Find the best support level below entry
            best_tp = None
            for level in [sup for sup in support_levels if sup < entry_price]:
                tp_distance = entry_price - level
                rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
                
                # Must meet minimum R:R and be within scalping limits
                if self.TP_MIN_STRUCTURE_RR <= rr_ratio <= self.TP_MAX_SCALP_RR:
                    if best_tp is None or rr_ratio < best_tp['ratio']:  # Prefer closer target
                        best_tp = {
                            'distance': tp_distance,
                            'ratio': rr_ratio,
                            'reason': f'support at {level:.5f}'
                        }
                        
            return best_tp

    # =================================================================
    #  DYNAMIC RISK/REWARD (STRUCTURE + ATR HYBRID)
    # =================================================================

    def calculate_risk_reward(self, df, direction, pair='EUR/USD',
                              spread=None, tp_ratio_override=None,
                              recent_sl_values=None):
        """Calculate structure-based SL and TP for 5m scalping.

        SL Strategy (Structure Priority):
          1. Find recent swing low/high within 20 bars
          2. Place SL below/above with 15% ATR buffer
          3. Cap at 1.2x ATR, floor at 0.5x ATR
          4. Fallback to 0.8x ATR if no structure

        TP Strategy (Structure Priority):
          1. Target nearest swing high/low (resistance/support)
          2. Ensure 1.3R minimum, 2.2R maximum for scalping
          3. Fallback to regime-based ratio if no good structure

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

        # Find structure-based SL level
        structure_sl = self._find_structure_stop_loss(df, direction, entry_price, atr, pair)
        
        if structure_sl:
            sl_distance = structure_sl['distance']
            sl_level = structure_sl['level']
            sl_reason = structure_sl['reason']
            
            # Verify structure SL meets spread requirements (hybrid spread approach)
            actual_spread = self._get_effective_spread(pair, spread, config)
            min_sl_by_spread = actual_spread * self.SL_MIN_SPREAD_MULT
            
            if sl_distance >= min_sl_by_spread:
                bot_logger.info(f"📍 Structure SL: {sl_reason} at {sl_level:.5f} ({sl_distance/pip_size:.1f}p)")
            else:
                bot_logger.info(f"⚠️ Structure SL too tight: {sl_distance/pip_size:.1f}p < {min_sl_by_spread/pip_size:.1f}p req'd → ATR fallback")
                structure_sl = None
        
        if not structure_sl:
            # Fallback to ATR-based SL with spread safety (hybrid spread approach)
            actual_spread = self._get_effective_spread(pair, spread, config)
            min_sl_by_spread = actual_spread * self.MIN_SL_SPREAD_MULT
            
            # Get pair-specific SL multiplier if available
            pair_sl_mult = self.SL_MULTIPLIER_BY_PAIR.get(pair, 1.0)
            base_atr_mult = self.SL_ATR_MULT * pair_sl_mult
            
            # JPY-specific: More generous ATR multiplier due to wider spreads
            if 'JPY' in pair:
                atr_multiplier = max(base_atr_mult, 1.2)  # Min 1.2×ATR for JPY
                safety_buffer = 1.2  # 20% extra safety for JPY
            else:
                atr_multiplier = base_atr_mult
                safety_buffer = 1.1  # 10% extra safety for majors
            
            # Ensure ATR SL meets minimum spread requirement
            atr_sl_distance = max(atr * atr_multiplier, min_sl_by_spread * safety_buffer)
            sl_distance = atr_sl_distance
            sl_reason = f"ATR fallback ({atr_sl_distance/pip_size:.1f}p, {atr_multiplier:.2f}xATR, pair_mult={pair_sl_mult:.2f})"
            bot_logger.info(f"📍 ATR SL: {sl_reason}")

        # Reject: SL < spread x 2 (hybrid spread approach)
        actual_spread = self._get_effective_spread(pair, spread, config)
        min_sl_by_spread = actual_spread * self.MIN_SL_SPREAD_MULT
        
        # ATR-based minimum: ensure SL is at least 0.5×ATR even if spread allows smaller
        min_sl_by_atr = atr * 0.5 if atr > 0 else 0
        min_sl_required = max(min_sl_by_spread, min_sl_by_atr)
        if sl_distance < min_sl_required:
            bot_logger.info(
                f"🚫 Trade rejected: SL {sl_distance / pip_size:.1f}p < "
                f"required {min_sl_required / pip_size:.1f}p (spread={min_sl_by_spread/pip_size:.1f}p, ATR={min_sl_by_atr/pip_size:.1f}p)"
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

        # TP distance = ratio x SL (with session adaptation)
        tp_ratio = tp_ratio_override if tp_ratio_override else self.TP_BASE_RATIO
        session_adapted_tp_ratio = self._get_session_tp_ratio(tp_ratio)
        
        # Try to find structure-based TP (better than ATR-based)
        structure_tp = self._find_structure_take_profit(df, direction, entry_price, sl_distance, session_adapted_tp_ratio)
        
        if structure_tp:
            tp_distance = structure_tp['distance'] 
            tp_ratio = structure_tp['ratio']
            tp_reason = structure_tp['reason']
            bot_logger.info(f"🎯 Structure TP: {tp_reason} (R:R = {tp_ratio:.1f})")
        else:
            tp_distance = sl_distance * session_adapted_tp_ratio
            tp_ratio = session_adapted_tp_ratio

        # ATR-adaptive pullback protection - only apply in low volatility conditions
        session = self._get_current_session()
        session_atr_min = config['session_atr_min']
        is_low_volatility = atr < (session_atr_min * 1.5)  # Apply when ATR < 1.5× session minimum
        
        if is_low_volatility:
            if structure_tp:
                # Structure TP: smaller ATR-adaptive adjustment to preserve good levels
                tp_pullback_buffer = min(atr * 0.10, 1.5 * pip_size)  # Min(10% ATR, 1.5p)
            else:
                # ATR TP: larger adjustment for mathematical targets in low vol
                tp_pullback_buffer = min(atr * 0.15, 3 * pip_size)    # Min(15% ATR, 3p)
        else:
            # High volatility: minimal pullback protection needed
            tp_pullback_buffer = min(atr * 0.05, 1 * pip_size)       # Min(5% ATR, 1p)
            
        tp_distance_adjusted = max(tp_distance - tp_pullback_buffer, sl_distance * 1.2)  # Min 1.2R after adjustment
        
        if tp_distance_adjusted != tp_distance:
            adjustment_pips = (tp_distance - tp_distance_adjusted) / pip_size
            vol_label = "low-vol" if is_low_volatility else "high-vol"
            bot_logger.info(f"📍 TP moved {adjustment_pips:.1f}p closer: {tp_distance/pip_size:.1f}p → {tp_distance_adjusted/pip_size:.1f}p ({vol_label} pullback protection)")

        # Calculate price levels
        if direction == 'BUY':
            stop_loss = round(entry_price - sl_distance, 5)
            take_profit = round(entry_price + tp_distance_adjusted, 5)
        else:
            stop_loss = round(entry_price + sl_distance, 5)
            take_profit = round(entry_price - tp_distance_adjusted, 5)

        sl_pips = sl_distance / pip_size
        tp_pips = tp_distance_adjusted / pip_size
        actual_rr_ratio = tp_distance_adjusted / sl_distance

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'take_profit_1': take_profit,
            'take_profit_2': take_profit,
            'sl_distance': sl_distance,
            'tp_distance': tp_distance_adjusted,
            'risk_pips': sl_pips,
            'reward_pips_1': tp_pips,
            'reward_pips_2': tp_pips,
            'rr_ratio': actual_rr_ratio,
            'atr': atr,
            'atr_sl_mult': self.SL_ATR_MULT,
            'tp_ratio_used': tp_ratio,          # Actual ratio used (structure or session-adapted)
            'tp_ratio_base': session_adapted_tp_ratio,  # Session-adapted base ratio
            'session': session,                 # Current trading session
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
        ok, reason, atr_declining = self.check_market_conditions(df, pair, spread)
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

        if bias['strength'] == 'weak' and bias['adx'] < 12:
            result['reasons'].append(f"⚠ ADX {bias['adx']:.1f} < 12 (very weak trend — skipping)")
            return result
        elif bias['strength'] == 'weak':
            result['reasons'].append(f"⚠ ADX {bias['adx']:.1f} < 15 (weak trend — penalized)")

        # 4. Detect ATR regime (for TP adjustment & contracting skip)
        atr_regime = self.detect_atr_regime(df)
        result['atr_regime'] = atr_regime['regime']
        result['atr_tp_ratio'] = atr_regime['tp_ratio']

        result['reasons'].append(
            f"ATR regime: {atr_regime['regime']} -> TP ratio {atr_regime['tp_ratio']:.1f}R "
            f"(streak: {atr_regime['atr_streak']})"
        )

        # Penalize contracting regime instead of blanket skip — let ensemble decide
        if atr_regime['regime'] == 'contracting':
            result['reasons'].append('⚠️ Contracting ATR regime — entry score penalized ×0.70')
            # Don't return — continue to entry detection with reduced score

        # 5. Detect entry
        entry = self.detect_entry(df, bias['direction'], pair)
        result['reasons'].extend(entry['signals'])

        # Apply penalties for adverse conditions (soft gates, not blocks)
        if atr_regime['regime'] == 'contracting':
            entry['confidence'] *= 0.70
            result['reasons'].append(f"  ⚠️ Contracting penalty: confidence → {entry['confidence']:.2f}")
        if atr_declining:
            entry['confidence'] *= 0.80
            result['reasons'].append(f"  ⚠️ ATR declining penalty: confidence → {entry['confidence']:.2f}")
        if bias['strength'] == 'weak':
            entry['confidence'] *= 0.80
            result['reasons'].append(f"  ⚠️ Weak ADX penalty: confidence → {entry['confidence']:.2f}")

        if not entry['ready'] and entry['confidence'] < self.ENTRY_THRESHOLD:
            result['reasons'].append(
                f"Entry not ready: confidence {entry['confidence']:.2f} "
                f"< threshold {self.ENTRY_THRESHOLD:.2f}"
            )
            return result

        # Re-check readiness after penalties
        entry['ready'] = entry['confidence'] >= self.ENTRY_THRESHOLD
        if not entry['ready']:
            result['reasons'].append(
                f"Entry killed by penalties: confidence {entry['confidence']:.2f} "
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
