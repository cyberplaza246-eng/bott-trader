"""
Backtesting Framework — Sweep-Gated Entry (RayAlgo v3)

Matches the live architecture:
  - LiquiditySweep is the SOLE gate (4-layer: Bias → Sweep → MSS → Entry)
  - EMA Crossover + Technical are confirmation boosters (boost/reduce only)
  - All other models run for context logging but have ZERO entry influence
  - 5M primary + 1M confluence (like the live bot)
  - Pip-based SL/TP from SCALPING_PAIRS config
  - Session windows, cooldowns, max-hold timeout
  - EMA200 counter-trend hard block
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
from src.ai.liquidity_sweep import LiquiditySweepAnalyzer
from src.ai.adaptive_learner import AdaptiveLearner
from src.ai.rl_agent import RLTradingAgent
from src.risk.position_manager import RiskManager, PIP_VALUES, DEFAULT_PIP
from src.risk.sl_tp import calculate_sl_tp
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


# ── Sweep-gated confirmation boost/penalty (mirrors ensemble_trader.py) ─
EMA_CONFIRM_BOOST   = 0.05    # EMA aligned with sweep direction
EMA_OPPOSE_PENALTY  = 0.10    # EMA opposes sweep direction
TECH_CONFIRM_BOOST  = 0.03    # Technical momentum matches sweep
TECH_OPPOSE_PENALTY = 0.05    # Technical momentum opposes sweep
LSTM_CONFIRM_BOOST  = 0.08    # LSTM direction agrees with sweep
LSTM_OPPOSE_PENALTY = 0.20    # LSTM direction opposes sweep (strong filter)
RL_SKIP_PENALTY     = 0.08    # RL agent recommends skipping


class BacktestEngine:
    """
    Sweep-gated backtest engine (RayAlgo v3).

    Matching live architecture:
      - LiquiditySweep is the sole gate (4-layer)
      - EMA Crossover + Technical boost/reduce confidence
      - Other models run for context only — no entry influence
      - Session window filtering
      - Pip-based SL/TP from SCALPING_PAIRS config
      - EMA200 counter-trend hard block
      - Max-hold timeout
      - 0.01 min lot
    """

    # ── Default Costs (realistic ECN broker) ────────────────────────
    DEFAULT_COMMISSION_PER_LOT = 7.0    # USD round-trip per standard lot
    DEFAULT_SPREAD_PIPS = {             # Typical average spreads
        'EUR/USD': 1.5,
        'GBP/USD': 2.0,
        'USD/JPY': 2.0,
    }

    def __init__(self, initial_balance=10000, slippage_pips=0.0,
                 commission_per_lot=None, spread_pips=None):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.equity = initial_balance
        self.trades = []
        self.daily_results = []
        self.slippage_pips = slippage_pips  # Slippage per trade in pips

        # Commission: USD per standard lot, round-trip (entry + exit)
        self.commission_per_lot = (
            commission_per_lot if commission_per_lot is not None
            else self.DEFAULT_COMMISSION_PER_LOT
        )

        # Spread: pips to add/subtract at entry.  None → use per-pair defaults.
        self._spread_pips_override = spread_pips  # explicit override (float or None)

        # Cost tracking
        self.total_spread_cost = 0.0
        self.total_commission_cost = 0.0
        self.total_slippage_cost = 0.0

        # Gate + confirmation + context models (mirrors ensemble_trader.py)
        self.sweep = LiquiditySweepAnalyzer()          # GATE
        self.ema_crossover = EMACrossoverAnalyzer()      # Confirmation
        self.technical = TechnicalAnalyzer()             # Confirmation
        self.scalping = ScalpingAnalyzer()               # Context only
        self.volume = VolumeAnalyzer()                   # Context only
        self.lstm = LSTMPredictor()                      # Context only
        self.lstm_available = TF_AVAILABLE and self.lstm.available
        self.candlestick = CandlestickPatternDetector()  # Context only
        self.support_resistance = SupportResistanceDetector()  # Context only
        self.multi_timeframe = MultiTimeframeAnalyzer()  # Context only
        self.learner = AdaptiveLearner()
        self.rl_agent = RLTradingAgent()
        self.rl_available = hasattr(self.rl_agent, 'q_network') or hasattr(self.rl_agent, 'q_table')
        self.risk_manager = RiskManager(initial_balance=initial_balance)

    # ──────────────────────────────────────────────────────────────────
    #  Main backtest loop
    # ──────────────────────────────────────────────────────────────────
    def run_backtest(self, historical_data, pair, confidence_threshold=0.45,
                     min_agreement=2, timeframe_key='5m', df_1m=None,
                     bar_minutes=None, slippage_pips=None):
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
            slippage_pips: Slippage per trade in pips (overrides constructor default).

        Returns:
            dict with backtest results
        """
        data = historical_data.copy()
        data = self.technical.calculate_indicators(data)
        data = self.volume.calculate_volume_profile(data)

        # Apply per-run slippage override if provided
        if slippage_pips is not None:
            self.slippage_pips = slippage_pips

        # Auto-detect bar size from timestamps
        if bar_minutes is None:
            if 'datetime' in data.columns and len(data) >= 2:
                d0 = pd.to_datetime(data['datetime'].iloc[0])
                d1 = pd.to_datetime(data['datetime'].iloc[1])
                bar_minutes = max(1, int((d1 - d0).total_seconds() / 60))
            else:
                bar_minutes = 5  # default assumption

        # Resolve spread for this pair
        if self._spread_pips_override is not None:
            self._pair_spread_pips = self._spread_pips_override
        else:
            # Use config spread_sim (price units → pips) or class default
            pair_cfg = SCALPING_PAIRS.get(pair, {})
            pip_sz = PIP_VALUES.get(pair, DEFAULT_PIP)['pip_size']
            if 'spread_sim' in pair_cfg:
                self._pair_spread_pips = pair_cfg['spread_sim'] / pip_sz
            else:
                self._pair_spread_pips = self.DEFAULT_SPREAD_PIPS.get(pair, 1.5)

        bot_logger.info(f"Starting sweep-gated backtest for {pair} | "
                        f"tf_key={timeframe_key} | bar={bar_minutes}m | "
                        f"candles={len(data)} | confluence={'1M' if df_1m is not None else 'off'} | "
                        f"spread={self._pair_spread_pips:.1f}pip | "
                        f"commission=${self.commission_per_lot:.1f}/lot | "
                        f"slippage={self.slippage_pips:.1f}pip")

        session = SCALPING_SESSION_WINDOWS.get(pair, {'start': 0, 'end': 24})
        # SCALPING_PAIRS is flat per pair (not nested by timeframe)
        pair_config = SCALPING_PAIRS.get(pair, {})

        # Max-hold in candles — prefer direct candle count over seconds
        if 'max_hold_candles' in pair_config:
            max_hold_candles = pair_config['max_hold_candles']
        else:
            max_hold_seconds = pair_config.get('max_hold_seconds', 3600)
            max_hold_candles = max(3, max_hold_seconds // (bar_minutes * 60))
        max_hold_candles = max(3, max_hold_candles)

        # Cooldown in candles
        cooldown_seconds = pair_config.get('cooldown_seconds', 180)
        cooldown_candles = max(1, cooldown_seconds // (bar_minutes * 60))

        bot_logger.info(f"  max_hold={max_hold_candles} candles, "
                        f"cooldown={cooldown_candles} candles ({cooldown_seconds}s)")

        # Prepare 1M data for confluence if provided
        data_1m = None
        data_1m_enriched = False
        if df_1m is not None and len(df_1m) > 200:
            data_1m = df_1m.copy()
            data_1m = self.technical.calculate_indicators(data_1m)
            data_1m = self.volume.calculate_volume_profile(data_1m)
            # Also enrich with sweep analyzer indicators (ADX, body_ratio, etc.)
            data_1m = self.sweep.calculate_indicators(data_1m)
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
                s_start = session.get('start', 0)
                s_end = session.get('end', 24)
                if s_start == s_end:
                    # start == end means 24/7
                    in_session = True
                elif s_start < s_end:
                    in_session = s_start <= candle_hour < s_end
                else:
                    # Wrap-around (e.g. start=22, end=6)
                    in_session = candle_hour >= s_start or candle_hour < s_end

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
                    # Use candle direction to determine which was hit first
                    # Bullish candle (close>open): path ~ open→low→high→close
                    # Bearish candle (close<open): path ~ open→high→low→close
                    is_bullish = current_candle['close'] >= current_candle['open']
                    if pos_type == 'BUY':
                        if is_bullish:
                            # Went to low first → SL hit first
                            exit_price = open_position['stop_loss']
                            exit_type = 'STOP_LOSS'
                        else:
                            # Went to high first → TP hit first
                            exit_price = open_position['take_profit']
                            exit_type = 'TAKE_PROFIT'
                    else:  # SELL
                        if is_bullish:
                            # Went to low first → TP hit first
                            exit_price = open_position['take_profit']
                            exit_type = 'TAKE_PROFIT'
                        else:
                            # Went to high first → SL hit first
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
            signal, confidence, agreement, sweep_sl_tp = self._ensemble_signal(
                data, idx, historical_subset, pair,
                df_1m_all=data_1m, candle_dt=candle_dt
            )

            if signal == 'SKIP':
                continue
            if confidence < confidence_threshold:
                continue
            if agreement < min_agreement:
                continue

            # ── 1M Confluence (only when sweep used 5M fallback) ──
            # When sweep already has 1M data, skip redundant confluence
            if data_1m_enriched and candle_dt is not None and not (data_1m is not None):
                conf_signal = self._get_1m_confluence_signal(
                    data_1m, candle_dt, pair
                )
                if conf_signal is not None:
                    if conf_signal == signal:
                        confidence = min(1.0, confidence + CONFLUENCE_BONUS)
                    elif conf_signal != 'SKIP':
                        confidence = max(0.0, confidence - DIVERGENCE_PENALTY)
                    if confidence < confidence_threshold:
                        continue

            # ── Optimal-hour bonus ────────────────────────────────
            if candle_hour is not None and candle_hour in OPTIMAL_HOURS_UTC:
                confidence = min(1.0, confidence + OPTIMAL_HOUR_BONUS)


            # ── Open position (apply spread to entry) ─────────────
            pip_info = PIP_VALUES.get(pair, DEFAULT_PIP)
            spread_price = self._pair_spread_pips * pip_info['pip_size']
            if signal == 'BUY':
                entry_price = current_price + spread_price / 2   # buy at ask
            else:
                entry_price = current_price - spread_price / 2   # sell at bid

            # ── Unified SL/TP via sl_tp module ─────────────────
            sl_tp_subset = data.iloc[max(0, idx - 200):idx + 1]
            sr_result = self.support_resistance.detect_levels(sl_tp_subset)
            sweep_wick = sweep_sl_tp.get('sweep_wick') if sweep_sl_tp else None

            sl_tp_result = calculate_sl_tp(
                sl_tp_subset, signal, pair, timeframe_key,
                sr_levels=sr_result, sweep_wick=sweep_wick,
            )
            if sl_tp_result is None:
                continue  # No valid SL/TP → skip trade

            stop_loss = sl_tp_result['stop_loss']
            take_profit = sl_tp_result['take_profit']

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
        Run sweep-gated ensemble on the most recent 1M candles
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
            sig, conf, _, _ = self._ensemble_signal(
                subset_1m, len(subset_1m) - 1, tail, pair
            )
            if sig == 'SKIP' or conf < 0.30:
                return None
            return sig
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────
    #  Sweep-gated ensemble (mirrors ensemble_trader.py RayAlgo v3)
    # ──────────────────────────────────────────────────────────────────
    def _ensemble_signal(self, data, idx, subset, pair, df_1m_all=None, candle_dt=None):
        """
        Sweep-gated signal generation:
          1. Run LiquiditySweep → if SKIP/HOLD → final = SKIP (hard gate)
          2. If sweep fires → run EMA + Technical as confirmation boosters
          3. Apply EMA200 counter-trend hard block
          4. Apply regime modifier from adaptive learner
        
        When df_1m_all is provided:
          - 1M data feeds the sweep analyzer as df_1m (sweep detection)
          - 5M data feeds as df_5m (bias/regime detection)
        When df_1m_all is None:
          - Falls back to using 5M subset as both (legacy behavior)
          
        Returns (signal, confidence, agreement, sweep_sl_tp).
        """
        # ── Prepare data for sweep analyzer ───────────────────────
        df_5m_subset = data.iloc[max(0, idx - 250):idx + 1] if len(data) > 0 else None
        
        # If we have actual 1M data, use it properly
        df_1m_subset = None
        if df_1m_all is not None and candle_dt is not None and len(df_1m_all) > 0:
            if 'datetime' in df_1m_all.columns:
                mask = df_1m_all['datetime'] <= candle_dt
                df_1m_subset = df_1m_all.loc[mask].tail(300).copy()
                if len(df_1m_subset) < 30:
                    df_1m_subset = None
        
        # Feed sweep: 1M data for sweep detection, 5M for regime
        if df_1m_subset is not None:
            sweep_signal = self.sweep.get_signal(df_1m_subset, pair, df_5m=df_5m_subset)
        else:
            # Fallback: use 5M subset as "1M" (less accurate but works)
            sweep_signal = self.sweep.get_signal(subset, pair, df_5m=df_5m_subset)
        
        sweep_direction = sweep_signal.get('signal', 'SKIP')
        sweep_conf = sweep_signal.get('confidence', 0.0)
        sweep_sl_tp = sweep_signal.get('sweep_sl_tp')  # Sweep's own SL/TP

        # HARD GATE: no sweep → no trade
        if sweep_direction not in ('BUY', 'SELL'):
            return 'SKIP', 0.0, 0, None

        # ── Confirmation models (boost/reduce only) ───────────────
        ema_signal = self.ema_crossover.get_signal(subset)
        technical_signal = self.technical.get_signal(subset)

        confidence = sweep_conf

        # EMA Crossover confirmation
        if ema_signal['signal'] == sweep_direction:
            confidence += EMA_CONFIRM_BOOST
        elif ema_signal['signal'] != 'HOLD' and ema_signal['signal'] != sweep_direction:
            confidence -= EMA_OPPOSE_PENALTY

        # Technical momentum confirmation
        if technical_signal['signal'] == sweep_direction:
            confidence += TECH_CONFIRM_BOOST
        elif technical_signal['signal'] != 'HOLD' and technical_signal['signal'] != sweep_direction:
            confidence -= TECH_OPPOSE_PENALTY

        # ── EMA200 soft penalty (instead of hard block) ───────────
        cur_price = subset['close'].iloc[-1]
        ema200 = subset['ema_200'].iloc[-1] if 'ema_200' in subset.columns else None
        if ema200 is not None and not pd.isna(ema200):
            if (sweep_direction == 'BUY' and cur_price < ema200) or \
               (sweep_direction == 'SELL' and cur_price > ema200):
                confidence -= 0.10  # penalty instead of hard block

        # ── Regime confidence modifier from learner ───────────────
        regime = self.learner.detect_regime(subset)
        regime_mod = self.learner.get_regime_confidence_modifier(regime)
        confidence *= regime_mod

        # ── Count context models that agree (for logging) ─────────
        context_count = 1  # sweep itself
        if ema_signal['signal'] == sweep_direction:
            context_count += 1
        if technical_signal['signal'] == sweep_direction:
            context_count += 1

        # ── LSTM direction filter ─────────────────────────────────
        if self.lstm_available:
            try:
                lstm_pred = self.lstm.predict_direction(subset)
                pct_change = lstm_pred.get('predicted_change_percent', 0)
                # Use raw prediction (0.02% threshold for 5M candles)
                if abs(pct_change) > 0.02:
                    if (pct_change > 0 and sweep_direction == 'BUY') or \
                       (pct_change < 0 and sweep_direction == 'SELL'):
                        confidence += LSTM_CONFIRM_BOOST
                        context_count += 1
                    else:
                        confidence -= LSTM_OPPOSE_PENALTY
            except Exception:
                pass  # LSTM failure → skip gracefully

        # ── RL quality filter ─────────────────────────────────────
        if self.rl_available:
            try:
                rsi_val = float(subset['rsi'].iloc[-1]) if 'rsi' in subset.columns and not pd.isna(subset['rsi'].iloc[-1]) else 50.0
                adx_val = float(subset['adx'].iloc[-1]) if 'adx' in subset.columns and not pd.isna(subset['adx'].iloc[-1]) else 25.0
                atr_val = float(subset['atr'].iloc[-1]) if 'atr' in subset.columns and not pd.isna(subset['atr'].iloc[-1]) else 0.001
                atr_med = float(subset['atr'].median()) if 'atr' in subset.columns else 0.001
                ema200_val = float(subset['ema_200'].iloc[-1]) if 'ema_200' in subset.columns and not pd.isna(subset['ema_200'].iloc[-1]) else cur_price
                ema200_dist = (cur_price - ema200_val) / (atr_val + 1e-8)
                hour = candle_dt.hour if candle_dt else 12
                vol_ratio = float(subset['volume'].iloc[-1] / (subset['volume'].rolling(20).mean().iloc[-1] + 1)) if 'volume' in subset.columns else 1.0

                rl_state = self.rl_agent.build_state(
                    ensemble_confidence=confidence,
                    model_agreement=context_count,
                    total_models=4,
                    regime='trending' if regime in ('trend_up', 'trend_down') else 'ranging',
                    rsi=rsi_val, adx=adx_val,
                    atr=atr_val, atr_median=atr_med,
                    ema200_dist=ema200_dist,
                    hour=hour, spread=0.00015,
                    volume_ratio=vol_ratio,
                    daily_trades=0, max_daily_trades=30,
                    current_drawdown=0
                )
                rl_action = self.rl_agent.select_action(rl_state, training=False)
                if rl_action == 0:  # SKIP
                    confidence -= RL_SKIP_PENALTY
            except Exception:
                pass  # RL failure → skip gracefully

        confidence = min(1.0, max(0.0, confidence))
        return sweep_direction, confidence, context_count, sweep_sl_tp

    # ──────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────
    def _close_position(self, pos, exit_price, exit_type, idx, pair, eq_curve):
        """Close a position and record the trade with spread, commission, slippage."""
        pip_info = PIP_VALUES.get(pair, DEFAULT_PIP)
        pip_size = pip_info['pip_size']
        pip_value = pip_info['pip_value_per_lot']

        # Apply spread to exit price (exit at worse side of spread)
        spread_price = self._pair_spread_pips * pip_size
        if pos['type'] == 'BUY':
            # Closing a BUY = selling at bid (lower)
            effective_exit = exit_price - spread_price / 2
        else:
            # Closing a SELL = buying at ask (higher)
            effective_exit = exit_price + spread_price / 2

        if pos['type'] == 'BUY':
            pips = (effective_exit - pos['entry_price']) / pip_size
        else:
            pips = (pos['entry_price'] - effective_exit) / pip_size

        # Apply slippage: deduct on both entry and exit (2× slippage_pips)
        slippage_pips_total = 0.0
        if self.slippage_pips > 0:
            slippage_pips_total = self.slippage_pips * 2  # entry + exit
            pips -= slippage_pips_total

        # Gross P/L from pips (spread already embedded)
        gross_pnl = pips * pos['lot_size'] * pip_value

        # Commission: per-lot round-trip
        commission = self.commission_per_lot * pos['lot_size']

        # Net P/L
        profit_loss = gross_pnl - commission

        # Track costs
        spread_cost = self._pair_spread_pips * pos['lot_size'] * pip_value
        slippage_cost = slippage_pips_total * pos['lot_size'] * pip_value
        self.total_spread_cost += spread_cost
        self.total_commission_cost += commission
        self.total_slippage_cost += slippage_cost

        self.current_balance += profit_loss
        self.equity = self.current_balance
        eq_curve.append(self.equity)

        self.trades.append({
            'type': pos['type'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'effective_exit': effective_exit,
            'exit_type': exit_type,
            'gross_pnl': gross_pnl,
            'commission': commission,
            'spread_cost': spread_cost,
            'profit_loss': profit_loss,
            'pips': pips,
            'profit_loss_percent': (profit_loss / self.initial_balance) * 100,
            'candles_held': idx - pos['entry_idx'],
        })

        bot_logger.info(
            f"[{idx}] EXIT ({exit_type}) @ {exit_price:.5f} (eff={effective_exit:.5f}) | "
            f"P/L: ${profit_loss:.2f} ({pips:.1f} pips) | "
            f"costs: spread=${spread_cost:.2f} comm=${commission:.2f}"
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
                'total_spread_cost': 0.0,
                'total_commission_cost': 0.0,
                'total_slippage_cost': 0.0,
                'total_costs': 0.0,
                'spread_pips': getattr(self, '_pair_spread_pips', 0),
                'commission_per_lot': self.commission_per_lot,
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
            # Cost breakdown
            'total_spread_cost': self.total_spread_cost,
            'total_commission_cost': self.total_commission_cost,
            'total_slippage_cost': self.total_slippage_cost,
            'total_costs': (self.total_spread_cost +
                            self.total_commission_cost +
                            self.total_slippage_cost),
            'spread_pips': getattr(self, '_pair_spread_pips', 0),
            'commission_per_lot': self.commission_per_lot,
        }
