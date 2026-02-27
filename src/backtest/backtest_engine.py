"""
Backtesting Framework — 9-Model Scalping Ensemble (Dual-Timeframe)

Matches the live architecture:
  - 9-model weighted ensemble (ScalpingAnalyzer weight 0.22)
  - 5M primary + 1M confluence (like the live bot)
  - Pip-based SL/TP from SCALPING_PAIRS config
  - Session windows, cooldowns, max-hold timeout
  - Weighted conviction scoring, EMA200 penalty, RSI divergence
  - Regime-aware weight boosts
  - 0.01 min lot

Usage:
  engine = BacktestEngine(initial_balance=10000)
  # Single-timeframe:
  result = engine.run_backtest(df_5m, 'EUR/USD', timeframe_key='5m')
  # Dual-timeframe with confluence:
  result = engine.run_backtest(df_5m, 'EUR/USD', timeframe_key='5m', df_1m=df_1m)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.ai.technical_analyzer import TechnicalAnalyzer
from src.ai.volume_analyzer import VolumeAnalyzer
from src.ai.lstm_predictor import LSTMPredictor, TF_AVAILABLE
from src.ai.ema_crossover import EMACrossoverAnalyzer
from src.ai.candlestick_patterns import CandlestickPatternDetector
from src.ai.support_resistance import SupportResistanceDetector
from src.ai.multi_timeframe import MultiTimeframeAnalyzer
from src.ai.scalping_analyzer import ScalpingAnalyzer
from src.ai.adaptive_learner import AdaptiveLearner
from src.risk.position_manager import RiskManager, PIP_VALUES, DEFAULT_PIP
from config.strategy_config import (
    SCALPING_PAIRS,
    SCALPING_SESSION_WINDOWS,
    ENSEMBLE_CONFIDENCE_THRESHOLD,
    MIN_MODELS_AGREEMENT,
    OPTIMAL_HOURS_UTC,
    OPTIMAL_HOUR_BONUS,
    CONFLUENCE_BONUS,
    DIVERGENCE_PENALTY,
)
from src.utils.logger import bot_logger


# ── Scalping-tuned model weights (must mirror ensemble_trader.py) ────
MODEL_WEIGHTS = {
    'scalping': 0.22,
    'ema_crossover': 0.14,
    'candlestick': 0.13,
    'technical': 0.12,
    'volume': 0.11,
    'multi_tf': 0.10,
    'support_resistance': 0.06,
    'lstm': 0.06,
    'sentiment': 0.06,   # always 0 in backtest—no live API
}

REGIME_BOOSTS = {
    'trending': {
        'scalping': 1.5, 'ema_crossover': 1.4, 'multi_tf': 1.3,
        'technical': 1.2, 'lstm': 1.1, 'support_resistance': 0.7, 'volume': 0.9,
    },
    'ranging': {
        'scalping': 0.6, 'support_resistance': 1.5, 'candlestick': 1.3,
        'technical': 1.2, 'ema_crossover': 0.6, 'multi_tf': 0.8,
    },
    'volatile': {
        'scalping': 0.8, 'volume': 1.4, 'support_resistance': 1.2,
        'candlestick': 1.1, 'ema_crossover': 0.5, 'technical': 0.8,
    },
}


class BacktestEngine:
    """
    9-model scalping backtest engine.

    Matching live architecture:
      - ScalpingAnalyzer as primary model (weight 0.22)
      - Weighted conviction scoring (not simple vote counting)
      - Session window filtering
      - Pip-based SL/TP from SCALPING_PAIRS config
      - EMA200 trend filter with penalty (not hard gate)
      - Regime-aware weight boosts
      - Max-hold timeout
      - 0.01 min lot
    """

    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.equity = initial_balance
        self.trades = []
        self.daily_results = []

        # 9 models (sentiment produces HOLD with 0 confidence in backtest)
        self.technical = TechnicalAnalyzer()
        self.volume = VolumeAnalyzer()
        self.lstm = LSTMPredictor()
        self.lstm_available = TF_AVAILABLE and self.lstm.available
        self.ema_crossover = EMACrossoverAnalyzer()
        self.candlestick = CandlestickPatternDetector()
        self.support_resistance = SupportResistanceDetector()
        self.multi_timeframe = MultiTimeframeAnalyzer()
        self.scalping = ScalpingAnalyzer()
        self.learner = AdaptiveLearner()
        self.risk_manager = RiskManager(initial_balance=initial_balance)

    # ──────────────────────────────────────────────────────────────────
    #  Main backtest loop
    # ──────────────────────────────────────────────────────────────────
    def run_backtest(self, historical_data, pair, confidence_threshold=0.45,
                     min_agreement=2, timeframe_key='5m', df_1m=None,
                     bar_minutes=None):
        """
        Run backtest on historical data using the 9-model scalping ensemble.

        Args:
            historical_data: DataFrame with OHLCV data (primary timeframe)
            pair: Currency pair (e.g. 'EUR/USD')
            confidence_threshold: Min weighted confidence to open a trade
            min_agreement: Min models agreeing (default 2)
            timeframe_key: Scalping config key for SL/TP ('1m' or '5m')
            df_1m: Optional 1M DataFrame for confluence detection.
                   When provided, the engine runs the 9-model ensemble on
                   the 1M data at each 5M candle boundary and applies
                   +CONFLUENCE_BONUS / -DIVERGENCE_PENALTY.
            bar_minutes: Minutes per bar in historical_data.
                         Auto-detected if None.

        Returns:
            dict with backtest results
        """
        data = historical_data.copy()
        data = self.technical.calculate_indicators(data)
        data = self.volume.calculate_volume_profile(data)

        # Auto-detect bar size from timestamps
        if bar_minutes is None:
            if 'datetime' in data.columns and len(data) >= 2:
                d0 = pd.to_datetime(data['datetime'].iloc[0])
                d1 = pd.to_datetime(data['datetime'].iloc[1])
                bar_minutes = max(1, int((d1 - d0).total_seconds() / 60))
            else:
                bar_minutes = 5  # default assumption

        bot_logger.info(f"Starting 9-model scalping backtest for {pair} | "
                        f"tf_key={timeframe_key} | bar={bar_minutes}m | "
                        f"candles={len(data)} | confluence={'1M' if df_1m is not None else 'off'}")

        session = SCALPING_SESSION_WINDOWS.get(pair, {'start': 0, 'end': 24})
        pair_config = SCALPING_PAIRS.get(pair, {}).get(timeframe_key, {})

        # Max-hold in candles (based on actual bar size)
        max_hold_seconds = pair_config.get('max_hold_seconds', 1200)
        max_hold_candles = max(3, max_hold_seconds // (bar_minutes * 60))

        # Cooldown in candles
        cooldown_seconds = pair_config.get('cooldown_seconds', 180)
        cooldown_candles = max(1, cooldown_seconds // (bar_minutes * 60))

        bot_logger.info(f"  max_hold={max_hold_candles} candles ({max_hold_seconds}s), "
                        f"cooldown={cooldown_candles} candles ({cooldown_seconds}s)")

        # Prepare 1M data for confluence if provided
        data_1m = None
        data_1m_enriched = False
        if df_1m is not None and len(df_1m) > 200:
            data_1m = df_1m.copy()
            data_1m = self.technical.calculate_indicators(data_1m)
            data_1m = self.volume.calculate_volume_profile(data_1m)
            data_1m_enriched = True
            if 'datetime' in data_1m.columns:
                data_1m['datetime'] = pd.to_datetime(data_1m['datetime'])
            bot_logger.info(f"  1M confluence data: {len(data_1m)} candles")

        open_position = None
        equity_curve = [self.initial_balance]
        cooldown_until = 0
        lookback = min(200, len(data) - 1)

        for idx in range(lookback, len(data)):
            current_candle = data.iloc[idx]
            current_price = current_candle['close']

            # ── Session window filter ─────────────────────────────
            candle_hour = None
            candle_dt = None
            if 'datetime' in data.columns:
                try:
                    candle_dt = pd.to_datetime(current_candle['datetime'])
                    candle_hour = candle_dt.hour
                except Exception:
                    pass

            in_session = True
            if candle_hour is not None:
                in_session = session['start'] <= candle_hour < session['end']

            # ── Close an open position (SL / TP / timeout) ────────
            if open_position:
                pos_type = open_position['type']
                candles_held = idx - open_position['entry_idx']

                if pos_type == 'BUY':
                    hit_sl = current_candle['low'] <= open_position['stop_loss']
                    hit_tp = current_candle['high'] >= open_position['take_profit']
                else:
                    hit_sl = current_candle['high'] >= open_position['stop_loss']
                    hit_tp = current_candle['low'] <= open_position['take_profit']

                timed_out = candles_held >= max_hold_candles

                exit_price = None
                exit_type = None
                if hit_sl and hit_tp:
                    exit_price = open_position['stop_loss']
                    exit_type = 'STOP_LOSS'
                elif hit_sl:
                    exit_price = open_position['stop_loss']
                    exit_type = 'STOP_LOSS'
                elif hit_tp:
                    exit_price = open_position['take_profit']
                    exit_type = 'TAKE_PROFIT'
                elif timed_out:
                    exit_price = current_price
                    exit_type = 'TIMEOUT'

                if exit_price is not None:
                    self._close_position(open_position, exit_price, exit_type,
                                         idx, pair, equity_curve)
                    open_position = None
                    cooldown_until = idx + cooldown_candles
                    continue

            # ── Cooldown & session gate ───────────────────────────
            if idx < cooldown_until or not in_session:
                continue

            # ── Generate primary ensemble signal ──────────────────
            historical_subset = data.iloc[max(0, idx - lookback):idx + 1]
            signal, confidence, agreement = self._ensemble_signal(
                data, idx, historical_subset, pair
            )

            if signal == 'SKIP':
                continue
            if confidence < confidence_threshold:
                continue
            if agreement < min_agreement:
                continue

            # ── 1M Confluence ─────────────────────────────────────
            if data_1m_enriched and candle_dt is not None:
                conf_signal = self._get_1m_confluence_signal(
                    data_1m, candle_dt, pair
                )
                if conf_signal is not None:
                    if conf_signal == signal:
                        # 1M agrees with 5M → boost
                        confidence = min(1.0, confidence + CONFLUENCE_BONUS)
                    elif conf_signal != 'SKIP':
                        # 1M disagrees → penalty
                        confidence = max(0.0, confidence - DIVERGENCE_PENALTY)

                    # Re-check after modifier
                    if confidence < confidence_threshold:
                        continue

            # ── Optimal-hour bonus ────────────────────────────────
            if candle_hour is not None and candle_hour in OPTIMAL_HOURS_UTC:
                confidence = min(1.0, confidence + OPTIMAL_HOUR_BONUS)

            # ── Open position ─────────────────────────────────────
            entry_price = current_price
            stop_loss = self.risk_manager.calculate_scalping_stop_loss(
                pair, timeframe_key, entry_price, signal
            )
            take_profit = self.risk_manager.calculate_scalping_take_profit(
                pair, timeframe_key, entry_price, stop_loss, signal
            )

            # Widen SL if below minimum
            pip_info = PIP_VALUES.get(pair, DEFAULT_PIP)
            min_sl_pips = pair_config.get('sl_pips_min', 6)
            min_sl_distance = min_sl_pips * pip_info['pip_size']
            sl_distance = abs(entry_price - stop_loss)
            if sl_distance < min_sl_distance:
                sl_distance = min_sl_distance
                if signal == 'BUY':
                    stop_loss = entry_price - sl_distance
                else:
                    stop_loss = entry_price + sl_distance

            position_size = self.risk_manager.calculate_position_size(
                entry_price, stop_loss, pair=pair
            )
            if not position_size:
                continue

            lot_size = max(0.01, position_size['lot_size'])

            open_position = {
                'entry_price': entry_price,
                'entry_idx': idx,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'lot_size': lot_size,
                'type': signal,
                'confidence': confidence,
            }

            bot_logger.info(
                f"[{idx}] {signal} @ {entry_price:.5f} | "
                f"SL: {stop_loss:.5f} | TP: {take_profit:.5f} | "
                f"conf={confidence:.2f} agree={agreement}"
            )

        # Close any still-open position at last price
        if open_position:
            last_price = data.iloc[-1]['close']
            self._close_position(open_position, last_price, 'END_OF_DATA',
                                 len(data) - 1, pair, equity_curve)

        return self._generate_backtest_results(equity_curve, pair)

    # ──────────────────────────────────────────────────────────────────
    #  1M Confluence Signal
    # ──────────────────────────────────────────────────────────────────
    def _get_1m_confluence_signal(self, data_1m, candle_dt, pair):
        """
        Run the 9-model ensemble on the most recent 1M candles
        up to `candle_dt` and return the direction or None.
        """
        try:
            if 'datetime' not in data_1m.columns:
                return None
            mask = data_1m['datetime'] <= candle_dt
            subset_1m = data_1m.loc[mask]
            if len(subset_1m) < 200:
                return None

            tail = subset_1m.iloc[-200:]
            sig, conf, _ = self._ensemble_signal(
                subset_1m, len(subset_1m) - 1, tail, pair
            )
            if sig == 'SKIP' or conf < 0.30:
                return None
            return sig
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────
    #  9-model weighted ensemble (mirrors ensemble_trader.py)
    # ──────────────────────────────────────────────────────────────────
    def _ensemble_signal(self, data, idx, subset, pair):
        """
        Run all models, apply weighted conviction scoring, return
        (signal, confidence, agreement).
        """
        # ── Get signals from each model ───────────────────────────
        scalping_signal = self.scalping.get_signal(subset, pair)
        scalp_type = scalping_signal.get('signal', 'HOLD')
        if scalp_type == 'SKIP':
            scalp_type = 'HOLD'
        scalp_conf = scalping_signal.get('confidence', 0.0)

        if self.lstm_available:
            lstm_signal = self.lstm.predict_direction(subset)
        else:
            lstm_signal = {'signal': 'HOLD', 'confidence': 0.0}

        technical_signal = self.technical.get_signal(subset)
        volume_signal = self.volume.get_volume_signal(subset)
        ema_signal = self.ema_crossover.get_signal(subset)
        candle_signal = self.candlestick.get_pattern_signal(subset)
        sr_signal = self.support_resistance.get_sr_signal(subset)
        mtf_signal = self._get_mtf_signal(data, idx)

        # Sentiment always HOLD in backtest
        sentiment_signal = {'signal': 'HOLD', 'confidence': 0.0}

        # ── Build signal map ──────────────────────────────────────
        all_signals = {
            'scalping': {'signal': scalp_type, 'confidence': scalp_conf},
            'technical': {'signal': technical_signal['signal'],
                          'confidence': technical_signal['confidence']},
            'volume': {'signal': volume_signal['signal'],
                       'confidence': volume_signal['confidence']},
            'ema_crossover': {'signal': ema_signal['signal'],
                              'confidence': ema_signal['confidence']},
            'candlestick': {'signal': candle_signal['signal'],
                            'confidence': candle_signal['confidence']},
            'support_resistance': {'signal': sr_signal['signal'],
                                   'confidence': sr_signal['confidence']},
            'multi_tf': {'signal': mtf_signal['signal'],
                         'confidence': mtf_signal.get('confidence', 0.0)},
            'sentiment': sentiment_signal,
        }
        if self.lstm_available:
            all_signals['lstm'] = {'signal': lstm_signal['signal'],
                                   'confidence': lstm_signal['confidence']}

        # ── Weights: base → adaptive → regime → normalize ─────────
        regime = self.learner.detect_regime(subset)
        weights = dict(MODEL_WEIGHTS)

        # Remove sentiment weight (always HOLD in backtest)
        weights.pop('sentiment', None)

        # Remove LSTM weight if unavailable
        if not self.lstm_available:
            lstm_w = weights.pop('lstm', 0)
            if weights:
                bonus = lstm_w / len(weights)
                weights = {k: v + bonus for k, v in weights.items()}

        # Adaptive weights from learner
        adaptive_w = self.learner.get_adjusted_weights(pair=pair)
        for k, v in adaptive_w.items():
            if k in weights:
                weights[k] = v

        # Regime boosts
        boosts = REGIME_BOOSTS.get(regime, {})
        for model, boost in boosts.items():
            if model in weights:
                weights[model] *= boost

        # Normalize
        w_sum = sum(weights.values())
        if w_sum > 0:
            weights = {k: v / w_sum for k, v in weights.items()}

        # ── Vote counting ─────────────────────────────────────────
        buy_votes = sum(1 for s in all_signals.values() if s['signal'] == 'BUY')
        sell_votes = sum(1 for s in all_signals.values() if s['signal'] == 'SELL')
        agreement = max(buy_votes, sell_votes)

        active_count = sum(1 for s in all_signals.values()
                           if s['signal'] != 'HOLD' or s['confidence'] > 0)
        total = len(all_signals)
        scaled_min = max(2, int(MIN_MODELS_AGREEMENT * active_count / total + 0.5))

        if buy_votes > sell_votes and agreement >= scaled_min:
            direction = 'BUY'
        elif sell_votes > buy_votes and agreement >= scaled_min:
            direction = 'SELL'
        else:
            return 'SKIP', 0.0, 0

        # ── Weighted conviction ───────────────────────────────────
        agreeing = {k: v for k, v in all_signals.items() if v['signal'] == direction}
        opposing = {k: v for k, v in all_signals.items()
                    if v['signal'] != direction and v['signal'] != 'HOLD'}

        agree_conv = sum(weights.get(m, 0) * v['confidence']
                         for m, v in agreeing.items())
        oppose_conv = sum(weights.get(m, 0) * v['confidence']
                          for m, v in opposing.items())
        net = agree_conv - oppose_conv * 0.5

        # Confluence bonus
        agree_ratio = len(agreeing) / max(active_count, 1)
        if agree_ratio >= 0.75:
            net += 0.10
        elif agree_ratio >= 0.60:
            net += 0.05

        # Regime modifier
        regime_mod = self.learner.get_regime_confidence_modifier(regime)
        net *= regime_mod

        # ── EMA200 filter (penalty, not hard gate) ────────────────
        cur_price = subset['close'].iloc[-1]
        ema200 = subset['ema_200'].iloc[-1] if 'ema_200' in subset.columns else None
        if ema200 is not None and not pd.isna(ema200):
            if (direction == 'BUY' and cur_price < ema200) or \
               (direction == 'SELL' and cur_price > ema200):
                net = max(0.0, net - 0.08)
            else:
                net *= 1.05  # trend-aligned bonus

        # ── RSI divergence penalty ────────────────────────────────
        if 'rsi' in subset.columns:
            rsi = subset['rsi'].iloc[-1]
            if direction == 'BUY' and rsi > 70:
                net *= 0.85
            elif direction == 'SELL' and rsi < 30:
                net *= 0.85

        confidence = min(1.0, max(0.0, net))
        return direction, confidence, agreement

    # ──────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────
    def _close_position(self, pos, exit_price, exit_type, idx, pair, eq_curve):
        """Close a position and record the trade."""
        pip_info = PIP_VALUES.get(pair, DEFAULT_PIP)
        pip_size = pip_info['pip_size']
        pip_value = pip_info['pip_value_per_lot']

        if pos['type'] == 'BUY':
            pips = (exit_price - pos['entry_price']) / pip_size
        else:
            pips = (pos['entry_price'] - exit_price) / pip_size

        profit_loss = pips * pos['lot_size'] * pip_value

        self.current_balance += profit_loss
        self.equity = self.current_balance
        eq_curve.append(self.equity)

        self.trades.append({
            'type': pos['type'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'exit_type': exit_type,
            'profit_loss': profit_loss,
            'pips': pips,
            'profit_loss_percent': (profit_loss / self.initial_balance) * 100,
            'candles_held': idx - pos['entry_idx'],
        })

        bot_logger.info(
            f"[{idx}] EXIT ({exit_type}) @ {exit_price:.5f} | "
            f"P/L: ${profit_loss:.2f} ({pips:.1f} pips)"
        )

    def _get_mtf_signal(self, data, idx):
        """Simulate multi-timeframe by resampling to a higher TF."""
        try:
            subset = data.iloc[max(0, idx - 200):idx + 1].copy()
            if 'datetime' not in subset.columns:
                return {'signal': 'HOLD', 'confidence': 0.0}

            subset['datetime'] = pd.to_datetime(subset['datetime'])
            subset = subset.set_index('datetime')

            # Resample to ~4x the bar size (5m → 20m, 1m → 5m, 1h → 4h)
            if len(subset) >= 2:
                gap = (subset.index[1] - subset.index[0]).total_seconds() / 60
            else:
                gap = 5
            if gap <= 1:
                resample_rule = '5min'
            elif gap <= 5:
                resample_rule = '20min'
            elif gap <= 15:
                resample_rule = '1h'
            else:
                resample_rule = '4h'

            resampled = subset.resample(resample_rule).agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum'
            }).dropna()

            if len(resampled) < 30:
                return {'signal': 'HOLD', 'confidence': 0.0}

            resampled = resampled.reset_index()
            resampled = self.technical.calculate_indicators(resampled)
            return self.technical.get_signal(resampled)
        except Exception:
            return {'signal': 'HOLD', 'confidence': 0.0}

    def _generate_backtest_results(self, equity_curve, pair):
        """Generate backtest statistics"""
        if not self.trades:
            return {
                'pair': pair,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'total_profit': 0.0,
                'initial_balance': self.initial_balance,
                'final_balance': self.current_balance,
                'return_percent': 0.0,
                'avg_pips': 0.0,
                'avg_win_pips': 0.0,
                'avg_loss_pips': 0.0,
                'timeout_exits': 0,
            }

        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['profit_loss'] > 0]
        losing_trades = [t for t in self.trades if t['profit_loss'] <= 0]
        timeout_exits = sum(1 for t in self.trades if t['exit_type'] == 'TIMEOUT')

        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        gross_profit = sum(t['profit_loss'] for t in winning_trades)
        gross_loss = abs(sum(t['profit_loss'] for t in losing_trades))

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        total_profit = self.current_balance - self.initial_balance

        avg_pips = np.mean([t['pips'] for t in self.trades])
        avg_win_pips = np.mean([t['pips'] for t in winning_trades]) if winning_trades else 0.0
        avg_loss_pips = np.mean([t['pips'] for t in losing_trades]) if losing_trades else 0.0

        # Max drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100

        # Annualized Sharpe ratio
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = (np.mean(returns) / np.std(returns) *
                        np.sqrt(252)) if np.std(returns) > 0 else 0

        return {
            'pair': pair,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_profit': total_profit,
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'return_percent': (total_profit / self.initial_balance) * 100,
            'avg_pips': avg_pips,
            'avg_win_pips': avg_win_pips,
            'avg_loss_pips': avg_loss_pips,
            'timeout_exits': timeout_exits,
        }
