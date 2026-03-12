"""
Elite 1M / 5M Structure-Based Liquidity Sweep Entry Model  (v2)

Architecture:
  Layer 1 — 5M True Swing Structure & Bias
             N-bar pivot swing detection → HH/HL = bullish, LH/LL = bearish
  Layer 2 — 1M Liquidity Sweep of swing-based levels + 5M invalidation gate
  Layer 3 — Market Structure Shift (MSS) — break of internal LH/HL after sweep
  Layer 4 — Entry (retest of broken structure *or* aggressive at displacement close)

SL = sweep wick extreme ± 0.2 × ATR buffer
TP = 1.5R default | 2R high-volatility | 1.2R range  (+ liquidity pool TP option)
Time stop = exit if not positive after 3 candles
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from src.utils.logger import bot_logger
from src.instruments import get_instrument, REGISTRY


class LiquiditySweepAnalyzer:
    """Structure-based liquidity sweep entry model (v2).

    Detects proper swing points, validates market structure sequences,
    identifies liquidity sweeps at swing levels, confirms Market Structure
    Shifts, and generates high-probability entry signals.
    """

    # ── Pair Configuration (sourced from instrument registry) ───────
    PAIR_CONFIG = {
        sym: {
            'session_atr_min': spec.atr_minimum,
            'spread_sim': spec.spread_default,
            'pip_size': spec.tick_size,
        }
        for sym, spec in REGISTRY.items()
    }

    # ── Swing Detection ─────────────────────────────────────────────
    SWEEP_LOOKBACK = 3               # N bars each side for pivot (was 5 → more swing points)
    SWING_LOOKBACK = 3               # Alias
    SWING_MIN_POINTS = 2             # Min swing points for structure (was 4)

    # ── Bias / Regime ───────────────────────────────────────────────
    ADX_MIN_BIAS = 10                # ADX floor for directional bias
    ATR_HIGH_VOL_MULT = 1.50         # ATR > 1.5× 20-period mean = high vol (was 1.30)
    ATR_LOW_VOL_MULT = 0.50          # ATR < 0.5× 20-period mean = low vol (was 0.70)

    # ── Sweep Detection ─────────────────────────────────────────────
    SWEEP_TOLERANCE = 0.0005         # Max penetration past swing (was 0.0002)
    SWEEP_WINDOW = 30                # Candles to look back for sweep event (was 15 → 30 = half-hour on 1M)

    # ── MSS / Displacement ──────────────────────────────────────────
    VOLUME_CONFIRMATION = 1.0        # Disabled (forex tick vol unreliable) (was 1.30)
    BODY_RATIO_MIN = 0.30            # Displacement body ≥ 30% of range
    RSI_SWEEP_LONG_MAX = 55          # RSI ≤ 55 at bullish sweep (tighter filter)
    RSI_SWEEP_SHORT_MIN = 45         # RSI ≥ 45 at bearish sweep (tighter filter)
    RSI_SLOPE_WINDOW = 3             # Candles after sweep to check RSI slope (was 2)
    CONFIRMATION_WINDOW = 20         # Candles after sweep to get MSS (was 15)

    # ── Risk Management ─────────────────────────────────────────────
    SL_ATR_BUFFER = 0.50             # SL = sweep wick ± 0.5×ATR (wider buffer for noise)

    # ── Entry Mode ──────────────────────────────────────────────────
    ENTRY_MODE = 'aggressive'        # 'aggressive' → enter at displacement close (more trades)

    # ── Session Windows (UTC) ───────────────────────────────────────
    OPTIMAL_SESSIONS = {
        'london': (7, 12),
        'ny': (13, 17),
    }

    # ── Counter-Trend Filter ───────────────────────────────────────
    CTF_EMA200_ATR_THRESHOLD = 2.0   # Reject sweep if price >2 ATR from EMA200 against direction
    CTF_ADX_THRESHOLD = 15           # Only apply when ADX indicates directional movement

    def __init__(self):
        self.atr_period = 14
        self.rsi_period = 14
        self.ema_short = 20
        self.ema_medium = 50
        self.ema_long = 200
        self.adx_period = 14
        self.volume_avg_period = 10

    # =================================================================
    #  VOLUME PROFILE ANALYSIS
    # =================================================================

    def compute_volume_profile(self, df, num_bins=24):
        """Compute volume profile — price levels where most volume traded.

        Returns list of (price_level, volume) sorted by volume descending.
        High-volume nodes (HVN) act as strong S/R; low-volume nodes (LVN)
        indicate price acceptance zones where moves accelerate.
        """
        if df is None or len(df) < 10 or 'volume' not in df.columns:
            return []

        volume = df['volume'].astype(float).values
        if volume.sum() == 0:
            return []

        # Typical price (HLC/3) weighted by volume
        typical = ((df['high'].astype(float) + df['low'].astype(float) +
                     df['close'].astype(float)) / 3).values
        price_min, price_max = typical.min(), typical.max()
        if price_max - price_min < 1e-10:
            return []

        bin_edges = np.linspace(price_min, price_max, num_bins + 1)
        profile = []
        for i in range(num_bins):
            mask = (typical >= bin_edges[i]) & (typical < bin_edges[i + 1])
            level = (bin_edges[i] + bin_edges[i + 1]) / 2
            vol = float(volume[mask].sum())
            profile.append((level, vol))

        # Sort by volume descending (highest-volume nodes first)
        profile.sort(key=lambda x: -x[1])
        return profile

    def sweep_at_high_volume_node(self, sweep_level, volume_profile, atr):
        """Check if a sweep occurred at/near a high-volume node (HVN).

        HVN sweeps are higher quality — institutional orders cluster there.

        Args:
            sweep_level: Price of the swept swing level.
            volume_profile: Output of compute_volume_profile().
            atr: Current ATR for proximity tolerance.

        Returns:
            (is_hvn: bool, volume_percentile: float 0-1)
        """
        if not volume_profile or atr <= 0:
            return False, 0.0

        total_vol = sum(v for _, v in volume_profile)
        if total_vol == 0:
            return False, 0.0

        tolerance = atr * 1.5  # Search within 1.5 ATR of sweep level
        best_pct = 0.0

        for level, vol in volume_profile:
            if abs(level - sweep_level) <= tolerance:
                pct = vol / total_vol
                best_pct = max(best_pct, pct)

        # Top-quartile of volume nodes = HVN
        is_hvn = best_pct >= 0.06  # Level holds >=6% of total volume
        return is_hvn, best_pct

    def displacement_volume_quality(self, df_1m, sweep_candle_idx):
        """Check if the displacement candle had above-average volume.

        Returns volume ratio (displacement_vol / avg_vol).  Values > 1.5
        indicate strong institutional participation.
        """
        if df_1m is None or 'volume' not in df_1m.columns:
            return 1.0

        abs_idx = len(df_1m) + sweep_candle_idx if sweep_candle_idx < 0 else sweep_candle_idx
        if abs_idx < 0 or abs_idx >= len(df_1m):
            return 1.0

        disp_vol = float(df_1m.iloc[abs_idx]['volume'])
        # Average volume over preceding 20 bars
        start = max(0, abs_idx - 20)
        avg_vol = float(df_1m.iloc[start:abs_idx]['volume'].mean()) if abs_idx > start else 1.0
        if avg_vol <= 0:
            return 1.0

        return disp_vol / avg_vol

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

        # Liquidity levels — swing-based (populated by detect_swing_points)
        # Keep rolling as fallback for indicator tests
        df['liq_low'] = low.rolling(window=self.SWEEP_LOOKBACK).min().shift(1)
        df['liq_high'] = high.rolling(window=self.SWEEP_LOOKBACK).max().shift(1)

        return df

    @staticmethod
    def _get_pip_size(price_or_pair):
        """Get pip/tick size from registry or approximate from price level."""
        if isinstance(price_or_pair, str):
            spec = REGISTRY.get(price_or_pair)
            if spec:
                return spec.tick_size
        # Numeric price heuristic
        p = float(price_or_pair) if not isinstance(price_or_pair, str) else 0
        if p >= 100:      # Futures (MES ~5500, MNQ ~18000) or JPY
            return 0.25
        elif p >= 20:     # JPY pairs
            return 0.01
        return 0.0001     # Standard forex

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
    #  SWING POINT DETECTION (N-bar Pivot Method)
    # =================================================================

    def detect_swing_points(self, df, lookback=None):
        """Identify swing highs and swing lows using N-bar pivot method.

        A swing high is a bar whose high is the highest of (lookback) bars
        on each side.  Similarly for swing low.

        After identifying raw pivots, label each as:
          HH (higher high), LH (lower high), HL (higher low), LL (lower low)
        relative to the previous swing of the same type.

        Args:
            df: DataFrame with 'high' and 'low' columns (indicators already added).
            lookback: N bars each side (default: SWING_LOOKBACK).

        Returns:
            list[dict]: [{index, price, swing_type: 'high'|'low',
                          label: 'HH'|'LH'|'HL'|'LL'|None, bar_idx: int}]
            Sorted chronologically.
        """
        if lookback is None:
            lookback = self.SWING_LOOKBACK

        if df is None or len(df) < lookback * 2 + 1:
            return []

        highs = df['high'].astype(float).values
        lows = df['low'].astype(float).values
        n = len(highs)

        swing_points = []

        for i in range(lookback, n - lookback):
            # Swing High: bar[i] high >= all bars in [i-lookback, i+lookback]
            # Allow ties (uniqueness=1 was too strict for flat/quiet markets)
            window_highs = highs[i - lookback: i + lookback + 1]
            if highs[i] == window_highs.max():
                # If tied, only count if this is the FIRST occurrence (leftmost)
                first_max_pos = np.argmax(window_highs == highs[i])
                if first_max_pos == lookback:  # bar[i] is at position 'lookback' in the window
                    swing_points.append({
                        'bar_idx': i,
                        'price': float(highs[i]),
                        'swing_type': 'high',
                        'label': None,
                    })

            # Swing Low: bar[i] low <= all bars in [i-lookback, i+lookback]
            window_lows = lows[i - lookback: i + lookback + 1]
            if lows[i] == window_lows.min():
                first_min_pos = np.argmax(window_lows == lows[i])
                if first_min_pos == lookback:
                    swing_points.append({
                        'bar_idx': i,
                        'price': float(lows[i]),
                        'swing_type': 'low',
                        'label': None,
                    })

        # Sort by bar index
        swing_points.sort(key=lambda x: x['bar_idx'])

        # Label each swing relative to prior swing of same type
        last_swing_high = None
        last_swing_low = None

        for sp in swing_points:
            if sp['swing_type'] == 'high':
                if last_swing_high is not None:
                    sp['label'] = 'HH' if sp['price'] > last_swing_high else 'LH'
                last_swing_high = sp['price']
            else:  # low
                if last_swing_low is not None:
                    sp['label'] = 'HL' if sp['price'] > last_swing_low else 'LL'
                last_swing_low = sp['price']

        return swing_points

    # =================================================================
    #  LAYER 1: 5M TRUE SWING STRUCTURE & BIAS
    # =================================================================

    def detect_regime(self, df_5m):
        """Classify market regime from 5M data using REAL swing structure.

        Bullish: sequence of HH → HL with last HL not broken.
        Bearish: sequence of LH → LL with last LH not broken.
        Otherwise: range / no trade.

        ADX ≥ 18 used as secondary filter.  ATR volatility can override.

        Returns:
            dict: {regime, bias, adx, atr_state, details,
                   swing_points, last_swing_high, last_swing_low}
        """
        base_result = {
            'regime': 'unknown', 'bias': None, 'adx': 0,
            'atr_state': 'unknown', 'details': 'Insufficient 5M data',
            'swing_points': [], 'last_swing_high': None, 'last_swing_low': None,
        }

        if df_5m is None or len(df_5m) < 20:
            return base_result

        # Ensure indicators exist
        if 'ema_20' not in df_5m.columns:
            try:
                df_5m = self.calculate_indicators(df_5m)
            except Exception:
                return base_result

        latest = df_5m.iloc[-1]
        adx = float(latest.get('adx', 0) or 0)
        atr = float(latest.get('atr', 0) or 0)
        atr_sma20 = float(latest.get('atr_sma20', 0) or 0)
        close = float(latest['close'])

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

        # ── True Swing Detection ────────────────────────────────────
        swing_points = self.detect_swing_points(df_5m, lookback=self.SWING_LOOKBACK)
        bot_logger.debug(f"detect_regime: {len(df_5m)} bars, ADX={adx:.1f}, swings={len(swing_points)}, lookback={self.SWING_LOOKBACK}")

        # Need at least 4 swing points to establish structure
        if len(swing_points) < self.SWING_MIN_POINTS:
            base_result.update({
                'adx': adx, 'atr_state': atr_state,
                'regime': 'range', 'bias': None,
                'swing_points': swing_points,
                'details': f'Only {len(swing_points)} swing points (need {self.SWING_MIN_POINTS})',
            })
            return base_result

        # Extract recent swing highs and lows (last N of each type)
        recent_highs = [sp for sp in swing_points if sp['swing_type'] == 'high'][-4:]
        recent_lows = [sp for sp in swing_points if sp['swing_type'] == 'low'][-4:]

        last_sh = recent_highs[-1] if recent_highs else None
        last_sl = recent_lows[-1] if recent_lows else None

        # Count HH/HL vs LH/LL in recent swings
        hh_count = sum(1 for sp in recent_highs if sp['label'] == 'HH')
        lh_count = sum(1 for sp in recent_highs if sp['label'] == 'LH')
        hl_count = sum(1 for sp in recent_lows if sp['label'] == 'HL')
        ll_count = sum(1 for sp in recent_lows if sp['label'] == 'LL')

        # Determine bias from swing structure
        bias = None
        regime = 'range'

        bullish_score = hh_count + hl_count
        bearish_score = lh_count + ll_count

        # Bullish: HH + HL sequence (relaxed: just need 1+ bullish swing labels)
        if bullish_score >= 1 and bullish_score > bearish_score:
            last_hl_price = None
            for sp in reversed(recent_lows):
                if sp['label'] == 'HL':
                    last_hl_price = sp['price']
                    break
            # If no HL labeled yet, use last swing low
            if last_hl_price is None and recent_lows:
                last_hl_price = recent_lows[-1]['price']

            if last_hl_price is not None and close > last_hl_price:
                regime = 'trend_up'
                bias = 'BUY'
            else:
                # Structure broken but still have bullish lean → allow with reduced conf
                regime = 'range'
                bias = 'BUY'

        # Bearish: LH + LL sequence
        elif bearish_score >= 1 and bearish_score > bullish_score:
            last_lh_price = None
            for sp in reversed(recent_highs):
                if sp['label'] == 'LH':
                    last_lh_price = sp['price']
                    break
            if last_lh_price is None and recent_highs:
                last_lh_price = recent_highs[-1]['price']

            if last_lh_price is not None and close < last_lh_price:
                regime = 'trend_down'
                bias = 'SELL'
            else:
                regime = 'range'
                bias = 'SELL'

        # Tie — use EMA slope as tiebreaker
        elif bullish_score == bearish_score and bullish_score >= 1:
            ema_20 = float(latest.get('ema_20', close))
            ema_50 = float(latest.get('ema_50', close))
            if ema_20 > ema_50:
                regime = 'range'
                bias = 'BUY'
            elif ema_20 < ema_50:
                regime = 'range'
                bias = 'SELL'

        # ADX secondary filter — very low = apply with soft penalty (don't kill)
        if bias is not None and adx < self.ADX_MIN_BIAS:
            # Don't remove bias, just note it's weak
            regime = 'range'

        # Volatility regime modifiers (don't kill bias on low vol)
        if atr_state == 'high_volatility' and regime in ('trend_up', 'trend_down'):
            regime = 'high_volatility'
        elif atr_state == 'low_volatility':
            regime = 'low_volatility'
            # Keep bias — low vol trades just get lower TP ratio

        sh_str = f"{last_sh['price']:.5f}" if last_sh else 'N/A'
        sl_str = f"{last_sl['price']:.5f}" if last_sl else 'N/A'
        details = (
            f"Swings: HH={hh_count} HL={hl_count} LH={lh_count} LL={ll_count} | "
            f"ADX={adx:.1f} ATR_state={atr_state} | "
            f"Last SH={sh_str} Last SL={sl_str}"
        )

        return {
            'regime': regime,
            'bias': bias,
            'adx': adx,
            'atr_state': atr_state,
            'ema_20': float(latest.get('ema_20', 0)),
            'ema_50': float(latest.get('ema_50', 0)),
            'close': close,
            'details': details,
            'swing_points': swing_points,
            'last_swing_high': last_sh,
            'last_swing_low': last_sl,
        }

    # =================================================================
    #  LAYER 2: 1M LIQUIDITY SWEEP AT SWING LEVELS + 5M INVALIDATION
    # =================================================================

    def detect_sweep(self, df_1m, bias, regime_info=None):
        """Detect liquidity sweep at a 1M swing-based level.

        Bullish sweep:
          - Price sweeps below a 1M swing low (identified pivot, not rolling min)
          - Close recovers above the swept level
          - If 5M swing data available, sweep must NOT break the 5M structural level

        Bearish sweep: inverse.

        Args:
            df_1m: 1M DataFrame with indicators.
            bias: 'BUY' or 'SELL' from Layer 1.
            regime_info: dict from detect_regime (contains 5M swing points for invalidation).

        Returns:
            dict: {detected, direction, sweep_wick, swept_level, rsi_at_sweep,
                   fivem_invalidation_held, candle_index, details}
        """
        result = {
            'detected': False,
            'direction': None,
            'sweep_wick': None,
            'swept_level': None,
            'rsi_at_sweep': None,
            'candle_index': None,
            'fivem_invalidation_held': True,
            'details': '',
        }

        attempts = 0
        nearest_gap = None

        if df_1m is None or len(df_1m) < self.SWING_LOOKBACK * 2 + self.SWEEP_WINDOW + 5:
            result['details'] = 'Insufficient 1M data'
            return result

        # ── Identify 1M swing levels (the liquidity pools) ──────────
        # Use data up to SWEEP_WINDOW bars back to find swing points
        # (don't include the last SWEEP_WINDOW bars in swing detection to avoid lookahead)
        detection_end = len(df_1m) - self.SWEEP_WINDOW
        if detection_end < self.SWING_LOOKBACK * 2 + 1:
            result['details'] = 'Not enough data for swing detection'
            return result

        df_for_swings = df_1m.iloc[:detection_end]
        swing_points_1m = self.detect_swing_points(df_for_swings, lookback=self.SWING_LOOKBACK)

        if len(swing_points_1m) < 2:
            result['details'] = 'Insufficient 1M swing points for liquidity levels'
            return result

        # ── Get 5M structural level for invalidation gate ───────────
        fivem_structural_level = None
        if regime_info is not None:
            if bias == 'BUY' and regime_info.get('last_swing_low'):
                fivem_structural_level = regime_info['last_swing_low']['price']
            elif bias == 'SELL' and regime_info.get('last_swing_high'):
                fivem_structural_level = regime_info['last_swing_high']['price']

        # ── Find the most recent relevant swing level ───────────────
        if bias == 'BUY':
            # For bullish sweep: look for recent swing lows (liquidity sits below)
            swing_lows = [sp for sp in swing_points_1m if sp['swing_type'] == 'low']
            if not swing_lows:
                result['details'] = 'No 1M swing lows to sweep'
                return result
            # Use the most recent swing low as the target liquidity level
            target_levels = sorted(swing_lows, key=lambda x: x['bar_idx'], reverse=True)[:3]
        else:
            # For bearish sweep: look for recent swing highs
            swing_highs = [sp for sp in swing_points_1m if sp['swing_type'] == 'high']
            if not swing_highs:
                result['details'] = 'No 1M swing highs to sweep'
                return result
            target_levels = sorted(swing_highs, key=lambda x: x['bar_idx'], reverse=True)[:3]

        # ── Scan last SWEEP_WINDOW candles for sweep event ──────────
        # Only accept sweeps in the last MAX_SWEEP_AGE candles to avoid
        # stale signals re-firing every cycle.  For 1M data, 5 bars = 5 min.
        MAX_SWEEP_AGE = 5
        latest_close_price = float(df_1m.iloc[-1]['close'])
        pip_size = self._get_pip_size(latest_close_price)
        for i in range(-min(self.SWEEP_WINDOW, MAX_SWEEP_AGE), 0):
            candle = df_1m.iloc[i]
            candle_low = float(candle['low'])
            candle_high = float(candle['high'])
            candle_close = float(candle['close'])
            candle_open = float(candle['open'])
            candle_rsi = float(candle.get('rsi', 50) or 50)
            pip_size = self._get_pip_size(candle_close)
            # Tight tolerance: require actual penetration of the swing level.
            # Only allow 1 tick leeway for bid/ask rounding.
            tol = pip_size

            # RSI slope check: was RSI turning in our favour after sweep?
            # Look at 1–2 candles ahead of the sweep candle for slope reversal
            rsi_slope_ok = False
            abs_idx = len(df_1m) + i  # absolute index of sweep candle
            for look_ahead in range(1, self.RSI_SLOPE_WINDOW + 1):
                next_idx = abs_idx + look_ahead
                if next_idx < len(df_1m):
                    next_rsi = float(df_1m.iloc[next_idx].get('rsi', 50) or 50)
                    if bias == 'BUY' and next_rsi > candle_rsi:
                        rsi_slope_ok = True
                        break
                    elif bias == 'SELL' and next_rsi < candle_rsi:
                        rsi_slope_ok = True
                        break
            # If sweep is on the very last candle, slope can't be confirmed yet — accept it
            if abs_idx >= len(df_1m) - 1:
                rsi_slope_ok = True

            for target in target_levels:
                level = target['price']
                attempts += 1

                if bias == 'BUY':
                    # Bullish sweep: wick below swing low, close recovers above
                    swept = candle_low <= (level + tol) and candle_close > level
                    rsi_ok = candle_rsi <= self.RSI_SWEEP_LONG_MAX and rsi_slope_ok
                    gap = abs(candle_low - level)
                    nearest_gap = gap if nearest_gap is None else min(nearest_gap, gap)

                    if swept and rsi_ok:
                        # 5M invalidation: sweep must NOT break 5M structural HL (with scalping tolerance)
                        fivem_held = True
                        if fivem_structural_level is not None:
                            # Add ATR-based tolerance for 1M scalping flexibility
                            regime_atr = float(regime_info.get('atr', 0) if regime_info else 0)
                            tolerance = regime_atr * 0.8 if regime_atr > 0 else 0.0015  # 80% ATR or 1.5p fallback
                            adjusted_fivem_level = fivem_structural_level - tolerance
                            
                            if candle_low < adjusted_fivem_level:
                                fivem_held = False

                        if not fivem_held:
                            result.update({
                                'detected': False,
                                'fivem_invalidation_held': False,
                                'details': (
                                    f"Sweep at {level:.5f} BLOCKED: broke 5M HL "
                                    f"{fivem_structural_level:.5f}"
                                ),
                            })
                            return result

                        result.update({
                            'detected': True,
                            'direction': 'BUY',
                            'sweep_wick': candle_low,
                            'swept_level': level,
                            'rsi_at_sweep': candle_rsi,
                            'rsi_slope_confirmed': True,
                            'candle_index': i,
                            'fivem_invalidation_held': True,
                            'details': (
                                f"Bullish sweep: low {candle_low:.5f} < swing_low "
                                f"{level:.5f}, close {candle_close:.5f} recovered, "
                                f"RSI={candle_rsi:.1f} (slope ↑)"
                            ),
                        })
                        return result

                elif bias == 'SELL':
                    # Bearish sweep: wick above swing high, close recovers below
                    swept = candle_high >= (level - tol) and candle_close < level
                    rsi_ok = candle_rsi >= self.RSI_SWEEP_SHORT_MIN and rsi_slope_ok
                    gap = abs(candle_high - level)
                    nearest_gap = gap if nearest_gap is None else min(nearest_gap, gap)

                    if swept and rsi_ok:
                        # 5M invalidation: sweep must NOT break 5M structural LH (with scalping tolerance)
                        fivem_held = True
                        if fivem_structural_level is not None:
                            # Add ATR-based tolerance for 1M scalping flexibility
                            regime_atr = float(regime_info.get('atr', 0) if regime_info else 0)
                            tolerance = regime_atr * 0.8 if regime_atr > 0 else 0.0015  # 80% ATR or 1.5p fallback
                            adjusted_fivem_level = fivem_structural_level + tolerance
                            
                            if candle_high > adjusted_fivem_level:
                                fivem_held = False

                        if not fivem_held:
                            result.update({
                                'detected': False,
                                'fivem_invalidation_held': False,
                                'details': (
                                    f"Sweep at {level:.5f} BLOCKED: broke 5M LH "
                                    f"{fivem_structural_level:.5f}"
                                ),
                            })
                            return result

                        result.update({
                            'detected': True,
                            'direction': 'SELL',
                            'sweep_wick': candle_high,
                            'swept_level': level,
                            'rsi_at_sweep': candle_rsi,
                            'rsi_slope_confirmed': True,
                            'candle_index': i,
                            'fivem_invalidation_held': True,
                            'details': (
                                f"Bearish sweep: high {candle_high:.5f} > swing_high "
                                f"{level:.5f}, close {candle_close:.5f} recovered, "
                                f"RSI={candle_rsi:.1f}"
                            ),
                        })
                        return result

        # Proximity-sweep fallback removed — require actual level penetration.

        if nearest_gap is not None:
            result['details'] = (
                f"No sweep of 1M swing levels in last {self.SWEEP_WINDOW} candles "
                f"(checked {attempts} candidates, nearest_gap={nearest_gap:.5f})"
            )
        else:
            result['details'] = f'No sweep of 1M swing levels in last {self.SWEEP_WINDOW} candles'
        return result

    # =================================================================
    #  LAYER 3: MARKET STRUCTURE SHIFT (MSS) + DISPLACEMENT
    # =================================================================

    def detect_mss(self, df_1m, sweep_result, regime='range'):
        """Detect Market Structure Shift after a liquidity sweep.

        For a bullish MSS after bullish sweep:
          - Find the last internal LH (lower high) BEFORE the sweep
          - A displacement candle must BREAK above that LH
          - The displacement candle must have bodyRatio ≥ 25%, volume ≥ 1.0×

        For bearish MSS: find last internal HL, displacement breaks below it.

        In RANGE regime (ADX~0): relaxed mode — displacement candle just
        needs right direction + body ratio, doesn't need to close beyond
        the full MSS level (structure is too tight in quiet markets).

        Returns:
            dict: {confirmed, mss_level, entry_price, trigger_level,
                   displacement_candle, details}
        """
        # Always require proper MSS displacement (strict mode)
        relaxed = False
        result = {
            'confirmed': False,
            'mss_level': None,
            'entry_price': None,
            'trigger_level': None,
            'displacement_candle': {},
            'details': '',
        }

        if not sweep_result.get('detected'):
            result['details'] = 'No sweep to confirm'
            return result

        direction = sweep_result['direction']
        sweep_idx = sweep_result['candle_index']  # negative index

        # Absolute index of the sweep candle
        abs_sweep_idx = len(df_1m) + sweep_idx

        # ── Find internal structure level to break ──────────────────
        # Look at candles BEFORE the sweep for the structure level
        pre_sweep_df = df_1m.iloc[max(0, abs_sweep_idx - 40):abs_sweep_idx]

        if len(pre_sweep_df) < 3:
            result['details'] = 'Not enough pre-sweep data for MSS detection'
            return result

        # Find swing points in the pre-sweep region
        pre_swing_points = self.detect_swing_points(pre_sweep_df, lookback=min(2, self.SWING_LOOKBACK))

        mss_level = None

        if direction == 'BUY':
            # For bullish MSS: need to break above last internal LH (or swing high)
            internal_highs = [sp for sp in pre_swing_points if sp['swing_type'] == 'high']
            if internal_highs:
                # Use the most recent swing high as the structure level
                mss_level = internal_highs[-1]['price']
            else:
                # Fallback: use the highest high in the last 10 candles before sweep
                mss_level = float(pre_sweep_df['high'].astype(float).iloc[-10:].max())
        else:
            # For bearish MSS: need to break below last internal HL (or swing low)
            internal_lows = [sp for sp in pre_swing_points if sp['swing_type'] == 'low']
            if internal_lows:
                mss_level = internal_lows[-1]['price']
            else:
                mss_level = float(pre_sweep_df['low'].astype(float).iloc[-10:].min())

        if mss_level is None:
            result['details'] = 'Could not identify MSS level'
            return result

        result['mss_level'] = mss_level

        # ── Check candles AT and AFTER sweep for displacement through MSS ─
        # In relaxed mode, include the sweep candle itself (it already showed
        # conviction by recovering through the swept level).
        check_start = sweep_idx if relaxed else sweep_idx + 1
        if check_start >= 0:
            check_start = -1

        candles_checked = 0
        for i in range(check_start, 0):
            if abs(i) > len(df_1m):
                continue

            candle = df_1m.iloc[i]
            disp = self._check_displacement_candle(
                candle, df_1m.iloc[i - 1], direction, relaxed=relaxed
            )
            candles_checked += 1

            if not disp['is_displacement']:
                bot_logger.debug(
                    f"  MSS candle[{i}]: body={disp['body_ratio']:.0%} "
                    f"vol={disp['volume_ratio']:.2f}x → not displacement"
                )
                continue

            candle_close = float(candle['close'])
            candle_high = float(candle['high'])
            candle_low = float(candle['low'])

            if direction == 'BUY':
                # Full mode: must close ABOVE mss_level
                # Relaxed (range): just need a bullish displacement candle
                mss_ok = candle_close > mss_level or relaxed
                if mss_ok:
                    if self.ENTRY_MODE == 'retest':
                        trigger = mss_level
                        entry = mss_level
                    else:
                        trigger = candle_high
                        entry = candle_close

                    # Wick-side validation: for BUY, entry must be above sweep wick
                    sweep_wick = sweep_result.get('sweep_wick')
                    if sweep_wick is not None and entry <= sweep_wick:
                        bot_logger.debug(f"  MSS BUY entry {entry:.5f} <= wick {sweep_wick:.5f} — skipped")
                        continue

                    label = 'above' if candle_close > mss_level else 'toward'
                    result.update({
                        'confirmed': True,
                        'mss_level': mss_level,
                        'entry_price': entry,
                        'trigger_level': trigger,
                        'displacement_candle': disp,
                        'details': (
                            f"Bullish MSS: displaced {label} {mss_level:.5f}, "
                            f"close={candle_close:.5f}, body={disp['body_ratio']:.0%}, "
                            f"vol={disp['volume_ratio']:.2f}x, "
                            f"mode={self.ENTRY_MODE}"
                            f"{' [relaxed-range]' if relaxed else ''}"
                        ),
                    })
                    return result

            elif direction == 'SELL':
                mss_ok = candle_close < mss_level or relaxed
                if mss_ok:
                    if self.ENTRY_MODE == 'retest':
                        trigger = mss_level
                        entry = mss_level
                    else:
                        trigger = candle_low
                        entry = candle_close

                    # Wick-side validation: for SELL, entry must be below sweep wick
                    sweep_wick = sweep_result.get('sweep_wick')
                    if sweep_wick is not None and entry >= sweep_wick:
                        bot_logger.debug(f"  MSS SELL entry {entry:.5f} >= wick {sweep_wick:.5f} — skipped")
                        continue

                    label = 'below' if candle_close < mss_level else 'toward'
                    result.update({
                        'confirmed': True,
                        'mss_level': mss_level,
                        'entry_price': entry,
                        'trigger_level': trigger,
                        'displacement_candle': disp,
                        'details': (
                            f"Bearish MSS: displaced {label} {mss_level:.5f}, "
                            f"close={candle_close:.5f}, body={disp['body_ratio']:.0%}, "
                            f"vol={disp['volume_ratio']:.2f}x, "
                            f"mode={self.ENTRY_MODE}"
                            f"{' [relaxed-range]' if relaxed else ''}"
                        ),
                    })
                    return result

        # ── Fallback: in relaxed mode, the sweep candle itself IS the
        #    displacement if it recovered through the swept level ──────
        if relaxed and sweep_result.get('detected'):
            sweep_candle = df_1m.iloc[sweep_idx]
            sc_close = float(sweep_candle['close'])
            sc_open = float(sweep_candle['open'])
            sc_high = float(sweep_candle['high'])
            sc_low = float(sweep_candle['low'])
            sc_range = sc_high - sc_low
            sc_body = abs(sc_close - sc_open)
            sc_body_ratio = sc_body / sc_range if sc_range > 0 else 0

            is_bullish_recovery = direction == 'BUY' and sc_close > sc_open and sc_body_ratio >= 0.10
            is_bearish_recovery = direction == 'SELL' and sc_close < sc_open and sc_body_ratio >= 0.10

            # Wick-side validation: entry must be on correct side of sweep wick
            sweep_wick = sweep_result.get('sweep_wick')
            if sweep_wick is not None:
                if direction == 'BUY' and sc_close <= sweep_wick:
                    is_bullish_recovery = False
                if direction == 'SELL' and sc_close >= sweep_wick:
                    is_bearish_recovery = False

            if is_bullish_recovery or is_bearish_recovery:
                entry = sc_close
                result.update({
                    'confirmed': True,
                    'mss_level': mss_level,
                    'entry_price': entry,
                    'trigger_level': entry,
                    'displacement_candle': {
                        'is_displacement': True,
                        'body_ratio': sc_body_ratio,
                        'volume_ratio': float(sweep_candle.get('volume_ratio', 1.0) or 1.0),
                        'close': sc_close,
                        'high': sc_high,
                        'low': sc_low,
                    },
                    'details': (
                        f"{'Bullish' if direction == 'BUY' else 'Bearish'} MSS (sweep-recovery): "
                        f"close={sc_close:.5f}, body={sc_body_ratio:.0%} "
                        f"[relaxed-range, sweep=displacement]"
                    ),
                })
                return result

        result['details'] = (
            f'No MSS displacement through {mss_level:.5f} after sweep '
            f'(checked {candles_checked} candles, sweep_idx={sweep_idx})'
        )
        return result

    def _check_displacement_candle(self, candle, prev_candle, direction, relaxed=False):
        """Check if a candle qualifies as a displacement candle.

        In normal mode:
          1. Body ≥ 25% of total range
          2. Volume ≥ 1.0× average (effectively disabled)
          3. Correct direction (bullish for BUY, bearish for SELL)

        In relaxed mode (range / low_vol regime):
          - Body ≥ 10% — any non-doji directional candle counts
          - Or candle close == candle high/low (momentum candle)

        Returns:
            dict: {is_displacement, body_ratio, volume_ratio, close, high, low}
        """
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

        # In relaxed mode, accept any directional candle with body > 10%
        body_min = 0.10 if relaxed else self.BODY_RATIO_MIN
        is_displacement = False

        if direction == 'BUY':
            is_bullish = candle_close > candle_open
            body_ok = body_ratio >= body_min
            volume_ok = vol_ratio >= self.VOLUME_CONFIRMATION
            is_displacement = is_bullish and body_ok and volume_ok

        elif direction == 'SELL':
            is_bearish = candle_close < candle_open
            body_ok = body_ratio >= body_min
            volume_ok = vol_ratio >= self.VOLUME_CONFIRMATION
            is_displacement = is_bearish and body_ok and volume_ok

        return {
            'is_displacement': is_displacement,
            'body_ratio': body_ratio,
            'volume_ratio': vol_ratio,
            'close': candle_close,
            'high': candle_high,
            'low': candle_low,
        }

    # =================================================================
    #  RISK/REWARD CALCULATION
    # =================================================================

    def calculate_risk_reward(self, sweep_result, displacement_result, regime_info, pair):
        """Validate sweep setup and return wick/entry data for unified SL/TP.

        SL/TP calculation is now handled by src.risk.sl_tp.calculate_sl_tp().
        This method just validates the sweep wick is viable and returns
        the data needed by the unified module.

        Args:
            sweep_result: dict from detect_sweep
            displacement_result: dict from detect_mss
            regime_info: dict from detect_regime
            pair: currency pair string

        Returns:
            dict or None: {sweep_wick, entry_price, atr, direction}
        """
        config = self.PAIR_CONFIG.get(pair, self.PAIR_CONFIG['EUR/USD'])
        pip_size = config['pip_size']

        direction = sweep_result['direction']
        sweep_wick = sweep_result['sweep_wick']
        entry_price = displacement_result.get('entry_price')

        if entry_price is None:
            return None

        atr = float(regime_info.get('atr', 0) or 0)
        if atr <= 0:
            atr = config['session_atr_min'] * 2

        # Basic viability: wick must be on correct side of entry
        if direction == 'BUY' and sweep_wick >= entry_price:
            return None
        if direction == 'SELL' and sweep_wick <= entry_price:
            return None

        # Minimum spread check — wick must be at least 1×spread from entry
        spread = config['spread_sim']
        raw_dist = abs(entry_price - sweep_wick)
        if raw_dist < spread:
            bot_logger.info(
                f"⛔ Sweep wick too close: {raw_dist/pip_size:.1f}p < "
                f"1×spread {spread/pip_size:.1f}p"
            )
            return None

        return {
            'sweep_wick': sweep_wick,
            'entry_price': entry_price,
            'atr': atr,
            'direction': direction,
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
        atr_min = float(config['session_atr_min'])
        atr_eps = max(atr_min * 1e-6, 1e-8)

        # ATR floor
        if atr + atr_eps < atr_min:
            return False, (
                f"ATR too low: {atr:.5f} < {atr_min:.5f} "
                f"({atr / config['pip_size']:.1f}p < {atr_min / config['pip_size']:.1f}p)"
            )

        # Spread check removed — SL/TP already accounts for spread costs
        # Low-ATR quiet hours would block nearly all trades otherwise

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
        """Full 4-layer entry pipeline: Bias → Sweep + 5M Gate → MSS → Entry.

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
                displacement: dict,   # v1 compat — maps to MSS result
                mss: dict,            # v2 MSS result
                risk_reward: dict,
                details: str,
                sweep_sl_tp: dict,
            }
        """
        result = {
            'signal': 'SKIP',
            'confidence': 0.0,
            'regime': 'unknown',
            'bias': None,
            'sweep': {},
            'displacement': {},
            'mss': {},
            'risk_reward': None,
            'details': '',
            'sweep_sl_tp': None,
        }

        # ── Step 0: Calculate indicators (skip if already present) ─────
        try:
            if 'ema_20' not in df_1m.columns:
                df_1m = self.calculate_indicators(df_1m)
        except Exception as e:
            result['details'] = f"Indicator calculation failed: {e}"
            return result

        if df_5m is not None and len(df_5m) >= 30:
            try:
                if 'ema_20' not in df_5m.columns:
                    df_5m = self.calculate_indicators(df_5m)
            except Exception as e:
                bot_logger.warning(f"⚠️ 5M indicator calc failed: {type(e).__name__}: {e}")
                df_5m = None
        elif df_5m is not None:
            bot_logger.warning(f"⚠️ 5M data too short for indicators: {len(df_5m)} rows (need 30)")

        # ── Step 1: Market conditions gate ────────────────────────────
        ok, reason = self.check_market_conditions(df_1m, pair, spread)
        if not ok:
            result['details'] = f"🚫 {reason}"
            bot_logger.info(f"🚫 Sweep skip {pair}: {reason}")
            return result

        # ── Step 2: 5M True Swing Structure & Bias (Layer 1) ─────────
        regime_info = self.detect_regime(df_5m)
        bot_logger.info(f"📊 {pair} detect_regime: df_5m={'None' if df_5m is None else len(df_5m)}, "
                        f"ADX={regime_info.get('adx', '?')}, "
                        f"swings={len(regime_info.get('swing_points', []))}, "
                        f"bias={regime_info.get('bias')}")
        result['regime'] = regime_info['regime']
        result['bias'] = regime_info['bias']

        if regime_info['bias'] is None:
            # Infer bias from 1M EMA instead of blocking
            if df_1m is not None and len(df_1m) >= 20:
                if 'ema_20' not in df_1m.columns:
                    try:
                        df_1m = self.calculate_indicators(df_1m)
                    except Exception:
                        pass
                if 'ema_20' in df_1m.columns and 'ema_50' in df_1m.columns:
                    ema20 = float(df_1m['ema_20'].iloc[-1])
                    ema50 = float(df_1m['ema_50'].iloc[-1])
                    close_1m = float(df_1m['close'].iloc[-1])
                    if close_1m > ema20 > ema50:
                        regime_info['bias'] = 'BUY'
                        regime_info['regime'] = 'range'
                    elif close_1m < ema20 < ema50:
                        regime_info['bias'] = 'SELL'
                        regime_info['regime'] = 'range'
                    else:
                        # Use close vs EMA20 as tiebreaker
                        regime_info['bias'] = 'BUY' if close_1m > ema20 else 'SELL'
                        regime_info['regime'] = 'range'
                    result['regime'] = regime_info['regime']
                    result['bias'] = regime_info['bias']
                    bot_logger.info(
                        f"📊 {pair} no 5M bias → inferred {regime_info['bias']} from 1M EMA"
                    )
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
            f"ADX={regime_info['adx']:.1f}, swings={len(regime_info.get('swing_points', []))}"
        )

        # ── Step 3: 1M Sweep + 5M Invalidation Gate (Layer 2) ────────
        sweep = self.detect_sweep(df_1m, regime_info['bias'], regime_info=regime_info)
        result['sweep'] = sweep

        if not sweep['detected']:
            blocked = "" if sweep.get('fivem_invalidation_held', True) else " [5M INVALIDATION]"
            detail_msg = f"⏳ Bias={regime_info['bias']} but {sweep['details']}{blocked}"
            result['details'] = detail_msg
            bot_logger.info(f"⏭️ {pair} NO SWEEP: {sweep['details']}{blocked}")
            return result

        bot_logger.info(f"💧 {pair} {sweep['details']}")

        # ── Step 3b: Counter-Trend Filter ─────────────────────────────
        # Aggressively reject sweeps that contradict the dominant trend.
        # e.g. bearish sweep in a strong uptrend → most are losers.
        if 'ema_200' in df_1m.columns:
            ema_200_val = float(df_1m['ema_200'].iloc[-1])
            current_close = float(df_1m['close'].iloc[-1])
            local_atr = float(df_1m.iloc[-1].get('atr', 0) or 0)
            if local_atr > 0 and ema_200_val > 0:
                ema_dist_atr = (current_close - ema_200_val) / local_atr
                adx_val = regime_info.get('adx', 0)

                # Price well above EMA200 but sweep says SELL → counter-trend
                if (ema_dist_atr > self.CTF_EMA200_ATR_THRESHOLD
                        and sweep['direction'] == 'SELL'
                        and adx_val >= self.CTF_ADX_THRESHOLD):
                    result['details'] = (
                        f"🚫 Counter-trend: SELL at {ema_dist_atr:.1f}×ATR above "
                        f"EMA200 (ADX={adx_val:.1f})"
                    )
                    bot_logger.info(f"🚫 {pair} counter-trend SELL rejected "
                                    f"(EMA200 dist={ema_dist_atr:.1f}×ATR)")
                    return result

                # Price well below EMA200 but sweep says BUY → counter-trend
                if (ema_dist_atr < -self.CTF_EMA200_ATR_THRESHOLD
                        and sweep['direction'] == 'BUY'
                        and adx_val >= self.CTF_ADX_THRESHOLD):
                    result['details'] = (
                        f"🚫 Counter-trend: BUY at {abs(ema_dist_atr):.1f}×ATR below "
                        f"EMA200 (ADX={adx_val:.1f})"
                    )
                    bot_logger.info(f"🚫 {pair} counter-trend BUY rejected "
                                    f"(EMA200 dist={ema_dist_atr:.1f}×ATR)")
                    return result

        # ── Step 4: Market Structure Shift (Layer 3) — SOFT GATE ─────
        mss = self.detect_mss(df_1m, sweep, regime=regime_info.get('regime', 'range'))
        result['mss'] = mss
        result['displacement'] = mss  # v1 backward compatibility

        mss_confirmed_naturally = mss['confirmed']
        if not mss_confirmed_naturally:
            # Soft gate: proceed with sweep-only entry at reduced confidence
            latest_close = float(df_1m.iloc[-1]['close'])
            mss['confirmed'] = True
            mss['entry_price'] = latest_close
            mss['trigger_level'] = latest_close
            mss['mss_level'] = mss.get('mss_level') or latest_close
            mss['details'] = f"Sweep-only entry (no displacement): {mss['details']}"
            result['mss'] = mss
            result['displacement'] = mss
            bot_logger.info(
                f"⚠️ {pair} sweep detected, soft MSS → entry at {latest_close:.5f} "
                f"({mss['details']})"
            )
        else:
            bot_logger.info(f"⚡ {pair} MSS: {mss['details']}")

        # ── Step 5: Risk/Reward Calculation ───────────────────────────
        latest_1m = df_1m.iloc[-1]
        regime_info['atr'] = float(latest_1m.get('atr', 0) or 0)

        rr = self.calculate_risk_reward(sweep, mss, regime_info, pair)
        if rr is None:
            result['details'] = "🚫 R:R calculation failed (SL too tight)"
            bot_logger.info(
                f"🚫 {pair} R:R failed: entry={mss.get('entry_price')}, "
                f"wick={sweep.get('sweep_wick')}, ATR={regime_info.get('atr')}"
            )
            return result

        result['risk_reward'] = rr
        result['sweep_sl_tp'] = rr

        # ── Step 6: Build final signal ────────────────────────────────
        # Full MSS → high confidence; sweep-only → moderate; proximity → lower
        is_proximity = sweep.get('proximity_sweep', False)
        if mss_confirmed_naturally and not is_proximity:
            base_confidence = 0.70
        elif mss_confirmed_naturally and is_proximity:
            base_confidence = 0.55
        elif not is_proximity:
            base_confidence = 0.38
        else:
            base_confidence = 0.30

        # Bonus for strong ADX
        if regime_info['adx'] >= 25:
            base_confidence += 0.05
        elif regime_info['adx'] >= 20:
            base_confidence += 0.03

        # Bonus for strong displacement volume
        vol_ratio = mss.get('displacement_candle', {}).get('volume_ratio', 1.0)
        if vol_ratio >= 2.0:
            base_confidence += 0.03
        elif vol_ratio >= 1.5:
            base_confidence += 0.01

        # ── Volume Profile Analysis (futures benefit) ─────────────────
        # Sweep at a high-volume node = more significant (institutional orders)
        vp_data = df_5m if df_5m is not None and len(df_5m) >= 30 else df_1m
        vol_profile = self.compute_volume_profile(vp_data)
        sweep_atr = float(regime_info.get('atr', 0) or 0)
        is_hvn, hvn_pct = self.sweep_at_high_volume_node(
            sweep['swept_level'], vol_profile, sweep_atr
        )
        if is_hvn:
            base_confidence += 0.02
            bot_logger.info(
                f"📊 {pair} sweep at HVN ({hvn_pct:.1%} of volume) → +2% conf"
            )

        # Displacement volume quality (real volume on futures)
        # NOTE: kept for logging/tags only; confidence already handled by MSS vol_ratio above
        disp_vol_ratio = self.displacement_volume_quality(
            df_1m, sweep.get('candle_index', -1)
        )

        # Mild penalty for range regime (adaptive learner handles main regime adjustment)
        if regime_info['regime'] == 'range':
            base_confidence *= 0.85

        result['signal'] = sweep['direction']
        result['confidence'] = min(base_confidence, 1.0)
        hvn_tag = f" HVN={hvn_pct:.0%}" if is_hvn else ""
        dvol_tag = f" DispVol={disp_vol_ratio:.1f}x" if disp_vol_ratio >= 1.5 else ""
        result['details'] = (
            f"✅ SWEEP+MSS ENTRY: {sweep['direction']} | "
            f"Regime={regime_info['regime']} | "
            f"MSS={mss['mss_level']:.5f} | "
            f"Wick={rr['sweep_wick']:.5f} Entry={rr['entry_price']:.5f} | "
            f"Vol={vol_ratio:.2f}x{hvn_tag}{dvol_tag} | ADX={regime_info['adx']:.1f} | "
            f"Mode={self.ENTRY_MODE}"
        )

        config = self.PAIR_CONFIG.get(pair, self.PAIR_CONFIG['EUR/USD'])
        pip_size = config['pip_size']
        wick_dist_pips = abs(rr['entry_price'] - rr['sweep_wick']) / pip_size
        bot_logger.info(
            f"🎯 SWEEP+MSS SIGNAL: {pair} {sweep['direction']} | "
            f"Entry={rr['entry_price']:.5f} | "
            f"Wick={rr['sweep_wick']:.5f} ({wick_dist_pips:.1f}p) | "
            f"ATR={rr['atr']:.5f} | MSS@{mss['mss_level']:.5f} | "
            f"Vol={vol_ratio:.2f}x | Mode={self.ENTRY_MODE}"
        )

        return result
