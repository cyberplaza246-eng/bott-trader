"""
Dual-Timeframe Scalping Bot — 1M + 5M Confluence System

Runs continuous 1-minute and 5-minute scalping cycles on EUR/USD & GBP/USD.
Uses a 9-model ensemble (including ScalpingAnalyzer as primary signal) with:
  - Adaptive learning, trailing stops, economic calendar
  - S/R-aware exits, cross-pair correlation, LSTM auto-retraining
  - Per-timeframe confluence bonuses/penalties
  - Pip-based SL/TP from scalping config
"""
import os
import time
import threading
from datetime import datetime, time as dt_time, timezone, timedelta
from apscheduler.schedulers.background import BackgroundScheduler

from src.broker.mt5_connector import MT5Connector
from src.core.ensemble_trader import EnsembleTrader
from src.risk.position_manager import RiskManager
from src.core.paper_trading import PaperTradingManager
from src.core.trailing_stop import TrailingStopManager
from src.ai.economic_calendar import EconomicCalendar
from src.ai.lstm_retrainer import LSTMRetrainer
from src.ai.rl_agent import RLTradingAgent
from src.utils.logger import bot_logger, TradeLogger
from config.strategy_config import (
    TRADING_MODE,
    PAIRS,
    TIMEFRAMES,
    AUTOTRADING_ENABLED,
    INITIAL_BALANCE,
    HIGH_CERTAINTY_THRESHOLD,
    MIN_MODELS_AGREEMENT,
    SCALPING_PAIRS,
    SCALPING_SESSION_WINDOWS,
    SCALPING_SPREAD_LIMITS,
    CONFLUENCE_BONUS,
    DIVERGENCE_PENALTY,
    OPTIMAL_HOURS_UTC,
    OPTIMAL_HOUR_BONUS,
)


class TradingBot:
    """Main trading bot orchestrator"""
    
    # ── Trading Session Windows (UTC hours) — Scalping-tight ──────
    PAIR_SESSIONS = SCALPING_SESSION_WINDOWS
    DEFAULT_SESSION = {'start': 7, 'end': 17}

    # ── Correlation Groups ────────────────────────────────────────
    # Pairs that move together — block duplicate directional exposure
    CORRELATED_PAIRS = {
        'EUR/USD': 'GBP/USD',
        'GBP/USD': 'EUR/USD',
        # USD/JPY has negative correlation — no block needed
    }

    # ── Spread Limits (max allowed spread in pips) — Scalping-tight ─
    MAX_SPREAD = {
        'EUR/USD': SCALPING_SPREAD_LIMITS.get('EUR/USD', 2.0) * 0.0001,
        'GBP/USD': SCALPING_SPREAD_LIMITS.get('GBP/USD', 2.5) * 0.0001,
        'USD/JPY': SCALPING_SPREAD_LIMITS.get('USD/JPY', 2.5) * 0.01,
    }
    DEFAULT_MAX_SPREAD = 0.00020

    def __init__(self, newsapi_key=None, enable_dashboard=True):
        self.mode = TRADING_MODE  # 'live', 'paper', 'backtest'
        self.running = False
        self.scheduler = BackgroundScheduler()
        self.enable_dashboard = enable_dashboard
        self.signal_history = []  # For dashboard
        
        # Initialize components
        if self.mode in ['live', 'paper']:
            try:
                self.broker = MT5Connector()
            except Exception as e:
                bot_logger.error(f"Failed to initialize MT5: {e}")
                self.broker = None
        
        self.ensemble = EnsembleTrader(newsapi_key=newsapi_key, broker=self.broker)
        
        # Use actual MT5 balance when connected, fall back to config
        actual_balance = INITIAL_BALANCE
        if self.broker and self.broker.connected:
            try:
                info = self.broker.get_account_info()
                if info and info.get('balance', 0) > 0:
                    actual_balance = info['balance']
                    bot_logger.info(f"💰 Using actual MT5 balance: ${actual_balance:.2f}")
            except Exception:
                pass
        self.risk_manager = RiskManager(initial_balance=actual_balance)
        self.trailing = TrailingStopManager(breakeven_r=1.0, trail_atr_mult=1.5)
        self.calendar = EconomicCalendar()
        self.lstm_retrainer = LSTMRetrainer(
            broker=self.broker,
            predictor=self.ensemble.lstm if self.ensemble.lstm_available else None,
        )
        
        # RL agent for trade decision optimization
        self.rl_agent = RLTradingAgent()

        # Determine active model count
        model_count = 9 if self.ensemble.lstm_available else 8
        
        if self.mode == 'paper':
            self.paper_trader = PaperTradingManager(initial_balance=INITIAL_BALANCE)
        else:
            self.paper_trader = None
        
        self.last_signal_time = {}  # Track last signal per pair per timeframe
        self.max_trade_hold_minutes = int(os.getenv('MAX_TRADE_HOLD_MINUTES', '30'))
        self.reversal_exit_confidence = float(os.getenv('REVERSAL_EXIT_CONFIDENCE', '0.55'))
        self.reversal_exit_min_agreement = int(os.getenv('REVERSAL_EXIT_MIN_AGREEMENT', '2'))
        self.enable_correlation_guard = os.getenv('ENABLE_CORRELATION_GUARD', 'true').lower() == 'true'

        # Dual-timeframe confluence tracking: {pair: {'1m': signal_result, '5m': signal_result}}
        self._last_signals = {pair: {} for pair in PAIRS}

        # Live closure detection: track open tickets so we notice when MT5 closes one
        self._known_tickets = {}   # {ticket: {pair, type, entry_price, open_time}}
        self._processed_closures = set()  # position_ids already recorded
        
        # Backfill past closed trades from MT5 history into adaptive learner
        self._backfill_history()
        
        bot_logger.info(
            f"✅ Scalping Bot initialized in {self.mode.upper()} mode "
            f"({model_count}-model ensemble, 1M+5M dual-timeframe)"
        )

    # ── Session Filter ────────────────────────────────────────────

    def is_pair_in_session(self, pair):
        """Check if the current UTC hour falls within this pair's trading window."""
        current_hour = datetime.now(timezone.utc).hour
        session = self.PAIR_SESSIONS.get(pair, self.DEFAULT_SESSION)
        s, e = session['start'], session['end']
        if s == e:
            return True  # Same start/end means all hours allowed
        if s < e:
            return s <= current_hour < e
        else:
            return current_hour >= s or current_hour < e

    # ── Spread Filter ─────────────────────────────────────────────

    def is_spread_acceptable(self, pair):
        """Return True if the current bid/ask spread is within limits."""
        try:
            price = self.broker.get_latest_price(pair)
            if not price or 'bid' not in price or 'ask' not in price:
                return True  # Can't check → allow
            spread = abs(price['ask'] - price['bid'])
            limit = self.MAX_SPREAD.get(pair, self.DEFAULT_MAX_SPREAD)
            if spread > limit:
                pip_div = 0.01 if 'JPY' in pair else 0.0001
                bot_logger.info(
                    f"⛔ {pair} spread too wide: {spread/pip_div:.1f} pips "
                    f"(max {limit/pip_div:.1f}) — skipping"
                )
                return False
            return True
        except Exception:
            return True  # Fail-open

    # ── Correlation Guard ─────────────────────────────────────────

    def _has_correlated_position(self, pair, direction):
        """Block if a correlated pair already has a position in the same direction."""
        correlated = self.CORRELATED_PAIRS.get(pair)
        if not correlated:
            return False
        try:
            positions = self.broker.get_open_positions()
            if not positions:
                return False
            for pos in positions:
                pos_pair = pos.get('pair', '').replace('/', '')
                corr_clean = correlated.replace('/', '')
                if pos_pair == corr_clean and pos.get('type') == direction:
                    bot_logger.info(
                        f"🔗 Correlation guard: {correlated} already has {direction} open "
                        f"— blocking {pair} {direction}"
                    )
                    return True
        except Exception:
            pass
        return False

    def should_skip_signal(self, pair, timeframe_key='5m'):
        """Prevent over-trading the same pair on the same timeframe"""
        cooldown_key = f"{pair}_{timeframe_key}"
        if cooldown_key not in self.last_signal_time:
            return False
        
        # Use per-pair cooldown from config (ATR-based config has no nested timeframe)
        pair_config = SCALPING_PAIRS.get(pair, {})
        cooldown_seconds = pair_config.get('cooldown_seconds', 30)
        
        elapsed = (datetime.now() - self.last_signal_time[cooldown_key]).total_seconds()
        return elapsed < cooldown_seconds

    @staticmethod
    def _is_opposite(signal_a, signal_b):
        return (signal_a, signal_b) in {('BUY', 'SELL'), ('SELL', 'BUY')}

    def _should_trade_pair(self, pair, signal_result):
        """Determine if a pair should be traded, factoring in adaptive + ML learning."""
        # Log recent win rate for visibility but don't block trades
        recent_wr = self.ensemble.learner.get_recent_win_rate(5)
        if recent_wr < 0.20 and len(self.ensemble.learner.recent_trades_window) >= 5:
            bot_logger.info(
                f"⚠️ Recent win rate {recent_wr:.0%} — tread carefully"
            )

        # ── ML Trade Scorer gate ─────────────────────────────────────
        # If the ML model is trained, use it to predict win probability
        ml_win_prob = self.ensemble.get_ml_win_probability(signal_result, pair)
        ml_status = self.ensemble.ml_scorer.get_status()
        if ml_status['is_trained']:
            # Block trades the ML model predicts will likely lose
            ML_MIN_WIN_PROB = 0.40  # Require at least 40% predicted win prob
            if ml_win_prob < ML_MIN_WIN_PROB:
                bot_logger.info(
                    f"🧠 ML gate BLOCKED {pair}: predicted win prob {ml_win_prob:.1%} "
                    f"< {ML_MIN_WIN_PROB:.0%} (model v{ml_status['model_version']})"
                )
                return False
            else:
                bot_logger.info(
                    f"🧠 ML gate OK {pair}: predicted win prob {ml_win_prob:.1%} "
                    f"(model v{ml_status['model_version']})"
                )

        return self.ensemble.should_trade(signal_result)
    
    def analyze_and_trade(self, timeframe_key='5m'):
        """
        Dual-timeframe scalping loop:
        1. Fetch candles for specified timeframe (1m or 5m)
        2. Run 9-model ensemble analysis
        3. Apply confluence bonus/penalty from the other timeframe
        4. Execute trades if signal is strong
        """
        try:
            self._analyze_and_trade_inner(timeframe_key)
        except Exception as e:
            bot_logger.error(
                f"💥 CRITICAL: analyze_and_trade({timeframe_key}) crashed: {e}",
                exc_info=True,
            )

    def _analyze_and_trade_inner(self, timeframe_key='5m'):
        """Inner analysis loop — wrapped by analyze_and_trade for safety."""
        if not self.broker:
            bot_logger.error("Broker not initialized, skipping analysis")
            return
        # Use recovery-aware check instead of hard self.broker.connected
        # This lets the bot resume after transient relay outages
        if not self.broker.connected:
            if hasattr(self.broker, 'is_connected_or_recover'):
                if not self.broker.is_connected_or_recover():
                    bot_logger.warning("Broker disconnected — waiting for relay recovery")
                    return
                # Recovery succeeded — continue with analysis
            else:
                bot_logger.error("Broker not connected, skipping analysis")
                return
        
        timeframe_minutes = TIMEFRAMES.get(f'scalp_{"fast" if timeframe_key == "1m" else "slow"}', 5)
        
        bot_logger.info("=" * 60)
        bot_logger.info(
            f"🔍 Scalping analysis ({timeframe_key}) at "
            f"{datetime.now().strftime('%H:%M:%S')}"
        )
        
        # Sync balance from broker/paper trader each cycle
        self._sync_balance()
        
        # Detect trades closed by MT5 (TP/SL hit) and record them
        self._detect_closed_trades()
        
        # Sync open trade count from broker (live mode)
        self._sync_open_trades()
        effective_cap, available_slots = self.risk_manager.get_trade_capacity()
        bot_logger.info(
            f"🎛️ Trade slots: {available_slots} available "
            f"({self.risk_manager.open_trades}/{effective_cap} used)"
        )

        # Run trailing stop updates on all tracked positions
        if self.broker:
            try:
                self.trailing.update(self.broker)
            except Exception as e:
                bot_logger.warning(f"Trailing stop update failed: {e}")

            # ── 3-Candle Time Stop ────────────────────────────────────
            # If a trade hasn't moved in our favor after 3 candle cycles,
            # close it at market to prevent slow bleeds.
            try:
                self._check_time_stop()
            except Exception as e:
                bot_logger.warning(f"Time stop check failed: {e}")

        # NOTE: Adaptive hour skip disabled — too easily poisoned by breakeven/early trades
        # if self.ensemble.learner.should_skip_hour():
        #     ...
        #     return

        # Drawdown protection — if in active drawdown, log it clearly
        if self.ensemble.learner.in_drawdown_protection:
            bot_logger.warning(
                f"🛡️ Drawdown protection ACTIVE — confidence threshold raised to "
                f"{self.ensemble.learner.confidence_threshold:.2f}"
            )

        # Log upcoming economic events
        try:
            upcoming_events = self.calendar.get_upcoming_events(hours_ahead=2)
            if upcoming_events:
                for ev in upcoming_events[:3]:
                    bot_logger.info(
                        f"📅 Upcoming: {ev['name']} at {ev['time']} "
                        f"({', '.join(ev['currencies'])}) — {ev['hours_away']}h away"
                    )
        except Exception:
            pass

        # Current UTC hour for optimal-hour bonus
        current_utc_hour = datetime.now(timezone.utc).hour

        for pair in PAIRS:
            try:
                # Skip if outside scalping session for this pair
                if not self.is_pair_in_session(pair):
                    session = self.PAIR_SESSIONS.get(pair, self.DEFAULT_SESSION)
                    bot_logger.info(
                        f"💤 {pair} outside scalping session "
                        f"(UTC {session['start']:02d}:00–{session['end']:02d}:00) — skipping"
                    )
                    continue

                # Skip if cooldown active for this timeframe
                if self.should_skip_signal(pair, timeframe_key):
                    continue

                # Adaptive session skip — skip pair+session combos with terrible win rates
                if self.ensemble.learner.should_skip_session(pair):
                    bot_logger.info(f"📉 Adaptive session skip: {pair} losing in current session")
                    continue

                # Loss pattern guard — skip if current conditions match a known heavy-loss pattern
                if self.ensemble.learner.should_skip_loss_pattern(pair):
                    bot_logger.info(f"🚫 Loss pattern guard: {pair} matches a recurring loss pattern — skipping")
                    continue

                # Economic event filter — avoid trading during high-impact news
                event_blocked, event_name = self.calendar.is_event_blocked(pair)
                if event_blocked:
                    bot_logger.info(f"📰 Event filter: {pair} blocked — {event_name} window active")
                    continue
                
                # Fetch candles for this timeframe
                num_candles = 250 if timeframe_key == '5m' else 300
                df = self.broker.get_candles(
                    pair,
                    timeframe_minutes=timeframe_minutes,
                    num_candles=num_candles
                )
                
                if df is None or len(df) < 50:
                    bot_logger.warning(f"Insufficient data for {pair} ({timeframe_key})")
                    continue
                
                # Get ensemble signal
                signal_result = self.ensemble.get_trading_signal(df, pair)

                # Store signal for confluence tracking (COPY to avoid mutation)
                self._last_signals[pair][timeframe_key] = {
                    'signal': signal_result['signal'],
                    'confidence': signal_result['confidence'],
                    'models_agreement': signal_result.get('models_agreement', 0),
                }

                # ── Confluence modifier ──────────────────────────────
                other_tf = '5m' if timeframe_key == '1m' else '1m'
                other_signal = self._last_signals[pair].get(other_tf)
                if other_signal and other_signal.get('signal') not in ('SKIP', None):
                    if signal_result['signal'] == other_signal['signal']:
                        # Both timeframes agree → boost confidence
                        old_conf = signal_result['confidence']
                        signal_result['confidence'] = min(1.0, old_conf + CONFLUENCE_BONUS)
                        bot_logger.info(
                            f"🎯 Confluence bonus: {timeframe_key}+{other_tf} both {signal_result['signal']} "
                            f"→ confidence {old_conf:.2f} → {signal_result['confidence']:.2f} (+{CONFLUENCE_BONUS:.0%})"
                        )
                    elif self._is_opposite(signal_result['signal'], other_signal['signal']):
                        # Timeframes diverge → reduce confidence
                        old_conf = signal_result['confidence']
                        signal_result['confidence'] = max(0.0, old_conf - DIVERGENCE_PENALTY)
                        bot_logger.info(
                            f"⚠️ Divergence penalty: {timeframe_key}={signal_result['signal']} vs "
                            f"{other_tf}={other_signal['signal']} "
                            f"→ confidence {old_conf:.2f} → {signal_result['confidence']:.2f}"
                        )

                # ── Optimal hour bonus ───────────────────────────────
                if current_utc_hour in OPTIMAL_HOURS_UTC and signal_result['signal'] != 'SKIP':
                    old_conf = signal_result['confidence']
                    signal_result['confidence'] = min(1.0, old_conf + OPTIMAL_HOUR_BONUS)
                    bot_logger.info(
                        f"⏰ Optimal hour bonus (UTC {current_utc_hour:02d}) "
                        f"→ +{OPTIMAL_HOUR_BONUS:.0%} confidence"
                    )

                # Feed price data to cross-pair analyzer for correlation tracking
                self.ensemble.cross_pair.update_prices(pair, df)
                
                # Log detailed analysis
                regime = signal_result.get('regime', 'unknown')
                bot_logger.info(f"\n{pair} [{timeframe_key}] Analysis:")
                bot_logger.info(f"  Signal: {signal_result['signal']}")
                bot_logger.info(f"  Confidence: {signal_result['confidence']:.1%}")
                sweep_model = signal_result.get('models', {}).get('sweep', {})
                bot_logger.info(
                    f"  Sweep: {sweep_model.get('signal', 'N/A')} "
                    f"(MSS={'\u2713' if sweep_model.get('mss_confirmed') else '\u2717'}) | "
                    f"Context: {signal_result['models_agreement']} models aligned"
                )
                bot_logger.info(f"  Market Regime: {regime}")
                bot_logger.info(f"  Details: {signal_result['detailed_reason']}")

                # Active position management: exit stale/reversal trades to free slots
                self._manage_open_position(pair, signal_result)
                
                # Track for dashboard
                self.signal_history.append({
                    'pair': pair,
                    'timeframe': timeframe_key,
                    'signal': signal_result['signal'],
                    'confidence': signal_result['confidence'],
                    'agreement': signal_result['models_agreement'],
                    'time': datetime.now().strftime('%H:%M:%S'),
                })
                # Keep last 100
                if len(self.signal_history) > 100:
                    self.signal_history = self.signal_history[-100:]
                
                # Execute trade if signal is strong enough
                threshold = self.ensemble.learner.get_adjusted_threshold()
                sweep_m = signal_result.get('models', {}).get('sweep', {})
                bot_logger.info(
                    f"  🔑 Trade gate: confidence={signal_result['confidence']:.2%} vs threshold={threshold:.2%} | "
                    f"sweep={'✓' if sweep_m.get('signal') in ('BUY','SELL') else '✗'} | "
                    f"signal={signal_result['signal']} | drawdown_prot={'ON' if self.ensemble.learner.in_drawdown_protection else 'OFF'}"
                )

                # ── Signal Confirmation (2-cycle persistence) ─────────
                # Only act on a signal if the SAME direction appeared on the
                # previous cycle too. This filters single-bar noise.
                if not hasattr(self, '_signal_confirmation'):
                    self._signal_confirmation = {}   # {pair: {direction, count, time}}
                confirm_key = pair
                cur_dir = signal_result['signal']
                prev = self._signal_confirmation.get(confirm_key, {})
                if cur_dir in ('BUY', 'SELL'):
                    if prev.get('direction') == cur_dir:
                        self._signal_confirmation[confirm_key] = {
                            'direction': cur_dir,
                            'count': prev.get('count', 1) + 1,
                            'time': datetime.now(),
                        }
                    else:
                        # Direction changed — reset counter
                        self._signal_confirmation[confirm_key] = {
                            'direction': cur_dir,
                            'count': 1,
                            'time': datetime.now(),
                        }
                    # Scalping: immediate execution for both 5m and 1m (markets move too fast for multi-cycle confirmation)
                    required_count = 1  # Was: 1 if timeframe_key == '5m' else 2
                    if self._signal_confirmation[confirm_key]['count'] < required_count:
                        bot_logger.info(
                            f"  ⏳ Signal confirmation: {pair} {cur_dir} — "
                            f"wait 1 more cycle (count={self._signal_confirmation[confirm_key]['count']})"
                        )
                        self._update_positions(pair)
                        continue
                    else:
                        bot_logger.info(
                            f"  ✅ Signal confirmed: {pair} {cur_dir} — "
                            f"{'5m immediate' if timeframe_key == '5m' else '2 consecutive cycles'}"
                        )
                else:
                    # SKIP — reset confirmation
                    self._signal_confirmation[confirm_key] = {}

                if self._should_trade_pair(pair, signal_result):
                    # Cross-timeframe cooldown: prevent 1M and 5M from placing duplicate orders
                    if not hasattr(self, '_last_trade_time'):
                        self._last_trade_time = {}
                    last_trade = self._last_trade_time.get(pair)
                    if last_trade and (datetime.now() - last_trade).total_seconds() < 15:
                        bot_logger.info(f"⏭️ {pair}: cross-timeframe cooldown active — skipping")
                        continue

                    if not self.risk_manager.can_trade_with_market(signal_result):
                        cap, available = self.risk_manager.get_trade_capacity(signal_result)
                        bot_logger.warning(f"Risk limits prevent trading {pair}")
                        bot_logger.warning(
                            f"No free trade slot for {pair}: "
                            f"{self.risk_manager.open_trades}/{cap} in use (available: {available})"
                        )
                        continue

                    # Adaptive learner: skip chronically losing pairs
                    if self.ensemble.learner.should_skip_pair(pair):
                        bot_logger.info(f"📉 Adaptive skip: {pair} win rate too low")
                        continue

                    # Spread filter (tighter for scalping)
                    if not self.is_spread_acceptable(pair):
                        continue

                    # Correlation guard (optional)
                    if self.enable_correlation_guard and self._has_correlated_position(pair, signal_result['signal']):
                        continue

                    # ── TEMPORARY: Bypass RL Agent for High Confidence Signals ──
                    # Allow excellent signals to trade immediately without RL veto
                    if signal_result['confidence'] >= 0.40:
                        bot_logger.info(f"🚀 High confidence signal ({signal_result['confidence']:.1%}) — bypassing RL agent")
                        # Set default RL values for position sizing
                        signal_result['_rl_lot_mult'] = 1.0
                        # Proceed directly to trade execution
                        self._execute_trade_signal(pair, signal_result, df)
                        continue

                    # ── RL Agent Gate ──────────────────────────────────
                    # Build state and let the RL agent decide action
                    try:
                        enriched = signal_result.get('enriched_df', df)
                        latest_bar = enriched.iloc[-1]
                        rl_state = self.rl_agent.build_state(
                            ensemble_confidence=signal_result['confidence'],
                            model_agreement=signal_result.get('models_agreement', 0),
                            total_models=signal_result.get('total_models', 8),
                            regime=signal_result.get('regime', 'ranging'),
                            rsi=float(latest_bar.get('rsi', 50)),
                            adx=float(latest_bar.get('adx', 20)),
                            atr=float(latest_bar.get('atr', 0)),
                            atr_median=float(enriched['atr'].median()) if 'atr' in enriched.columns else 0.001,
                            ema200_dist=(float(latest_bar['close']) - float(latest_bar.get('ema_200', latest_bar['close']))) / float(latest_bar['close']) if 'ema_200' in latest_bar.index else 0,
                            hour=current_utc_hour,
                            spread=float(self.broker.get_spread(pair)) if hasattr(self.broker, 'get_spread') else 0,
                            volume_ratio=float(latest_bar.get('volume_ratio', 1.0)),
                            daily_trades=self.risk_manager.open_trades,
                            max_daily_trades=effective_cap,
                            current_drawdown=getattr(self.ensemble.learner, 'current_drawdown_pct', 0),
                        )
                        # Training mode only during first 500 trades, then exploit
                        rl_training = self.rl_agent.total_trades < 500
                        rl_action = self.rl_agent.select_action(rl_state, training=rl_training)
                        rl_action_name = self.rl_agent.get_action_name(rl_action)

                        if rl_action == 0:  # SKIP
                            bot_logger.info(f"  🤖 RL Agent: SKIP trade on {pair} (ε={self.rl_agent.epsilon:.3f})")
                            # Store state for later reward computation
                            if not hasattr(self, '_rl_pending_skips'):
                                self._rl_pending_skips = {}
                            self._rl_pending_skips[f"{pair}_{timeframe_key}"] = {
                                'state': rl_state, 'action': rl_action,
                                'signal': signal_result,
                            }
                            continue
                        else:
                            lot_mult = self.rl_agent.get_lot_multiplier(rl_action)
                            bot_logger.info(
                                f"  🤖 RL Agent: {rl_action_name} on {pair} "
                                f"(lot×{lot_mult:.1f}, ε={self.rl_agent.epsilon:.3f})"
                            )
                            # Store RL info for post-trade reward recording
                            signal_result['_rl_state'] = rl_state
                            signal_result['_rl_action'] = rl_action
                            signal_result['_rl_lot_mult'] = lot_mult
                    except Exception as e:
                        bot_logger.warning(f"RL agent error: {e} — proceeding with full lot")
                        signal_result['_rl_lot_mult'] = 1.0

                    self._execute_trade(pair, signal_result, df, timeframe_key)
                    # Set cross-timeframe cooldown (prevents 1M and 5M racing)
                    self._last_trade_time[pair] = datetime.now()
                    cooldown_key = f"{pair}_{timeframe_key}"
                    self.last_signal_time[cooldown_key] = datetime.now()
                    # Track regime for this pair's trade (used when recording trade result)
                    if not hasattr(self, '_last_regime'):
                        self._last_regime = {}
                    self._last_regime[pair] = signal_result.get('regime', 'unknown')
                    # Re-sync free margin after placing a trade so next pair
                    # sees updated margin availability
                    self._sync_balance()
                else:
                    # Log why the signal didn't pass
                    if signal_result['signal'] == 'SKIP':
                        bot_logger.info(f"  ❌ {pair}: sweep gate did not fire — no trade")
                    elif signal_result['confidence'] < threshold:
                        bot_logger.info(f"  ❌ {pair}: confidence {signal_result['confidence']:.2%} < threshold {threshold:.2%}")
                
                # Update open positions
                self._update_positions(pair)
                
            except Exception as e:
                bot_logger.error(f"Error analyzing {pair} [{timeframe_key}]: {str(e)}", exc_info=True)
        
        # Log risk status with tier info
        try:
            daily_status = self.risk_manager.get_daily_status()
            bot_logger.info(f"\n📊 Daily Status:")
            bot_logger.info(f"  Balance: ${daily_status['current_balance']:.2f}")
            bot_logger.info(f"  Tier: {daily_status.get('tier_description', 'N/A')}")
            bot_logger.info(f"  Max Lot: {daily_status.get('max_lot_size', 'N/A')} | Max Trades: {daily_status.get('max_concurrent_trades', 'N/A')}")
            bot_logger.info(f"  Growth: {daily_status.get('account_growth', 0):.1f}% | Next tier: {daily_status.get('next_tier', 'N/A')} (${daily_status.get('balance_to_next', 0):.0f} away)")
            bot_logger.info(f"  Daily Loss: ${daily_status['daily_loss']:.2f} ({daily_status['daily_loss_percent']:.1f}%)")
            bot_logger.info(f"  Open Trades: {daily_status['open_trades']}")
            bot_logger.info(
                f"  Slots Available: {daily_status.get('available_trade_slots', 0)} "
                f"/ {daily_status.get('effective_trade_cap', daily_status.get('max_concurrent_trades', 0))}"
            )
            bot_logger.info(f"  Can Trade: {daily_status['can_trade']}")

            # ML Scorer status
            ml_status = self.ensemble.ml_scorer.get_status()
            if ml_status['is_trained']:
                bot_logger.info(
                    f"  🧠 ML Scorer: v{ml_status['model_version']} | "
                    f"accuracy={ml_status['last_accuracy']:.1%} | "
                    f"cv={ml_status['last_cv_score']:.1%} | "
                    f"samples={ml_status['training_samples']} | "
                    f"next retrain in {ml_status['retrain_interval'] - ml_status['trades_since_retrain']} trades"
                )
            else:
                bot_logger.info(
                    f"  🧠 ML Scorer: collecting data ({ml_status['training_samples']}/{ml_status['min_required']} trades)"
                )
        except Exception as e:
            bot_logger.warning(f"Daily status logging failed: {e}")
    
    def _execute_trade(self, pair, signal_result, df, timeframe_key='5m'):
        """Execute a scalping trade with ATR-based SL/TP from signal"""
        trade_type = signal_result['signal']
        # Use the enriched df (with indicators) from the ensemble
        enriched_df = signal_result.get('enriched_df', df)
        latest = enriched_df.iloc[-1]
        entry_price = latest['close']
        atr = latest.get('atr', entry_price * 0.001)  # Fallback: 0.1% of price

        # ── SL/TP Priority: Sweep > Scalping > ATR Fallback ──────────
        sweep_rr = signal_result.get('sweep_sl_tp')
        scalping_rr = signal_result.get('scalping_risk_reward', {})

        if sweep_rr and sweep_rr.get('stop_loss') and sweep_rr.get('take_profit'):
            # PREFERRED: Use sweep-wick anchored SL/TP
            stop_loss = sweep_rr['stop_loss']
            take_profit = sweep_rr['take_profit']
            sl_distance = sweep_rr.get('sl_distance', abs(entry_price - stop_loss))
            tp_ratio = sweep_rr.get('tp_ratio_used', 1.5)

            bot_logger.info(
                f"💧 Sweep SL/TP: SL={sweep_rr.get('risk_pips', 0):.1f}p "
                f"(wick+0.2×ATR), "
                f"TP={sweep_rr.get('reward_pips', 0):.1f}p ({tp_ratio:.1f}R)"
            )
        elif scalping_rr and scalping_rr.get('stop_loss') and scalping_rr.get('take_profit'):
            # Use ATR-derived SL/TP from the scalping analyzer
            stop_loss = scalping_rr['stop_loss']
            take_profit = scalping_rr['take_profit']
            sl_distance = scalping_rr.get('sl_distance', abs(entry_price - stop_loss))
            tp_ratio = scalping_rr.get('tp_ratio_used', 1.4)

            bot_logger.info(
                f"📐 ATR-based SL/TP: SL={scalping_rr.get('risk_pips', 0):.1f}p "
                f"({scalping_rr.get('atr_sl_mult', 0.8)}×ATR), "
                f"TP={scalping_rr.get('reward_pips_1', 0):.1f}p ({tp_ratio:.1f}R)"
            )
        else:
            # Fallback: calculate ATR-based SL/TP if signal didn't provide it
            pair_config = SCALPING_PAIRS.get(pair, {})
            pip_size = pair_config.get('pip_size', 0.0001)
            sl_mult = 0.8  # Match ScalpingAnalyzer.SL_ATR_MULT (was 1.2 — too wide for scalping)
            sl_distance = atr * sl_mult

            if trade_type == 'BUY':
                stop_loss = round(entry_price - sl_distance, 5)
            else:
                stop_loss = round(entry_price + sl_distance, 5)

            # TP from ATR regime
            atr_regime = signal_result.get('atr_regime', 'neutral')
            tp_ratio_map = {'expanding': 2.0, 'contracting': 1.5, 'neutral': 1.8}
            tp_ratio = tp_ratio_map.get(atr_regime, 1.8)
            tp_distance = sl_distance * tp_ratio

            if trade_type == 'BUY':
                take_profit = round(entry_price + tp_distance, 5)
            else:
                take_profit = round(entry_price - tp_distance, 5)

            bot_logger.info(
                f"📐 ATR fallback SL/TP: SL={sl_distance/pip_size:.1f}p "
                f"({sl_mult}×ATR), TP={tp_distance/pip_size:.1f}p ({tp_ratio:.1f}R)"
            )

        # Record SL value for adaptive learner median tracking
        try:
            sl_dist = abs(entry_price - stop_loss)
            self.ensemble.learner.record_sl_outcome(pair, sl_dist, True)  # outcome recorded on close
        except Exception:
            pass

        # S/R-based dynamic TP: use nearest S/R level ONLY if it improves R:R
        sr_levels = signal_result.get('sr_levels', {})
        risk_distance = abs(entry_price - stop_loss)
        original_rr = abs(take_profit - entry_price) / risk_distance if risk_distance > 0 else 0
        max_rr = 3.0  # Scalping cap
        if sr_levels and risk_distance > 0:
            if trade_type == 'BUY':
                resistances = sr_levels.get('resistance_levels', [])
                for level in sorted(resistances):
                    reward = level - entry_price
                    rr = reward / risk_distance
                    if 1.2 <= rr <= max_rr and rr >= original_rr * 0.9:
                        sr_tp = round(level, 5)
                        bot_logger.info(
                            f"🎯 S/R TP upgrade: {take_profit:.5f} → {sr_tp:.5f} "
                            f"(resistance level, R:R = {rr:.1f} vs original {original_rr:.1f})"
                        )
                        take_profit = sr_tp
                        break
            elif trade_type == 'SELL':
                supports = sr_levels.get('support_levels', [])
                for level in sorted(supports, reverse=True):
                    reward = entry_price - level
                    rr = reward / risk_distance
                    if 1.2 <= rr <= max_rr and rr >= original_rr * 0.9:
                        sr_tp = round(level, 5)
                        bot_logger.info(
                            f"🎯 S/R TP upgrade: {take_profit:.5f} → {sr_tp:.5f} "
                            f"(support level, R:R = {rr:.1f} vs original {original_rr:.1f})"
                        )
                        take_profit = sr_tp
                        break

        position_size = self.risk_manager.calculate_position_size(entry_price, stop_loss, pair=pair)
        
        if not position_size:
            bot_logger.error(f"Failed to calculate position size for {pair}")
            return
        
        lot_size = position_size['lot_size']

        # Apply RL agent lot multiplier if available
        rl_lot_mult = signal_result.get('_rl_lot_mult', 1.0)
        if rl_lot_mult != 1.0:
            lot_size = round(lot_size * rl_lot_mult, 2)
            bot_logger.info(f"  🤖 RL lot adjustment: ×{rl_lot_mult:.1f} → {lot_size}")

        # Scalping: minimum lot 0.01
        lot_size = max(0.01, lot_size)
        
        bot_logger.info(f"\n🔪 SCALPING {trade_type} ({timeframe_key}):")
        bot_logger.info(f"  Pair: {pair}")
        bot_logger.info(f"  Entry: {entry_price:.5f}")
        bot_logger.info(f"  Stop Loss: {stop_loss:.5f}")
        bot_logger.info(f"  Take Profit: {take_profit:.5f}")
        bot_logger.info(f"  Lot Size: {lot_size}")
        bot_logger.info(f"  Risk: ${position_size['risk_amount']:.2f}")

        # Store RL state for reward computation on trade close
        rl_state = signal_result.get('_rl_state')
        rl_action = signal_result.get('_rl_action')
        if rl_state is not None:
            if not hasattr(self, '_pending_rl_info'):
                self._pending_rl_info = {}
            pair_config_pip = SCALPING_PAIRS.get(pair, {}).get('pip_size', 0.0001)
            self._pending_rl_info[pair] = {
                'state': rl_state,
                'action': rl_action,
                'sl_pips': abs(entry_price - stop_loss) / pair_config_pip,
            }

        # VALIDATION: Log exactly what's being sent to broker
        sl_correct = (trade_type == 'BUY' and stop_loss < entry_price) or (trade_type == 'SELL' and stop_loss > entry_price)
        tp_correct = (trade_type == 'BUY' and take_profit > entry_price) or (trade_type == 'SELL' and take_profit < entry_price)
        
        bot_logger.info(f"🔍 ORDER VALIDATION:")
        bot_logger.info(f"  Direction: {trade_type}")
        bot_logger.info(f"  Entry: {entry_price:.5f}")
        bot_logger.info(f"  SL: {stop_loss:.5f} {'✅' if sl_correct else '❌ WRONG SIDE'}")
        bot_logger.info(f"  TP: {take_profit:.5f} {'✅' if tp_correct else '❌ WRONG SIDE'}")

        # Execute based on mode
        if self.mode == 'live' and AUTOTRADING_ENABLED:
            order_id = self.broker.place_order(
                pair=pair,
                order_type=trade_type,
                lot_size=lot_size,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if order_id:
                bot_logger.info(f"✅ Order placed - Ticket: {order_id}")
                self.risk_manager.on_trade_opened()
                # Register for trailing stop management (scalping mode)
                self.trailing.register(
                    ticket=order_id,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    direction=trade_type,
                    atr=atr,
                    pair=pair,
                    take_profit=take_profit,
                    volume=lot_size,
                    scalping_mode=True,
                )
        
        elif self.mode == 'paper':
            trade_id = self.paper_trader.execute_trade(
                pair=pair,
                trade_type=trade_type,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=lot_size
            )
            bot_logger.info(f"✅ Paper trade executed - ID: {trade_id}")
            self.risk_manager.on_trade_opened()

        # Store model signals for adaptive learning when trade closes
        # (works for BOTH live and paper mode so the learner always knows
        #  which models contributed to each trade)
        if not hasattr(self, '_pending_trade_signals'):
            self._pending_trade_signals = {}
        self._pending_trade_signals[pair] = signal_result.get('models', {})

        # ── ML Scorer: capture feature snapshot at entry time ────────
        if not hasattr(self, '_pending_ml_features'):
            self._pending_ml_features = {}
        try:
            ml_features = self.ensemble.capture_ml_features(signal_result, pair)
            self._pending_ml_features[pair] = ml_features
        except Exception as e:
            bot_logger.warning(f"🧠 ML feature capture failed: {e}")
    
    def _update_positions(self, pair):
        """Sync open positions from broker (live + paper mode)"""
        if self.mode == 'paper':
            latest_price = self.broker.get_latest_price(pair)
            if not latest_price:
                return
            
            update = self.paper_trader.update_position(pair, latest_price['bid'])
            
            if update and update['exit_signal']:
                exit_type = update['exit_signal']
                exit_price = update['exit_price']
                
                closed_trade = self.paper_trader.close_position(pair, exit_price, exit_type)
                
                if closed_trade:
                    self.risk_manager.on_trade_closed(closed_trade['profit_loss'])
                    bot_logger.info(
                        f"✅ Position closed ({exit_type}) - P/L: ${closed_trade['profit_loss']:.2f}"
                    )
                    
                    # Record with adaptive learner
                    model_signals = getattr(self, '_pending_trade_signals', {}).pop(pair, {})
                    self.ensemble.record_trade_result({
                        'pair': pair,
                        'signal': closed_trade.get('type', 'UNKNOWN'),
                        'profit_loss': closed_trade['profit_loss'],
                        'entry_price': closed_trade.get('entry_price', 0),
                        'exit_price': closed_trade.get('exit_price', 0),
                        'exit_type': exit_type,
                        'model_signals': model_signals,
                        'regime': getattr(self, '_last_regime', {}).get(pair, 'unknown'),
                    })

                    # Record with ML Trade Scorer
                    self._record_ml_outcome(pair, closed_trade['profit_loss'])

                    # Record with RL Agent
                    self._record_rl_outcome(pair, closed_trade['profit_loss'], exit_type)

    def _check_time_stop(self):
        """Close positions that haven't moved into profit after 3 candle cycles.

        For sweep-based entries this prevents slow bleed losses.
        Tracked via _trade_cycle_counts: {ticket: {entry_price, direction, cycles}}.
        """
        if not hasattr(self, '_trade_cycle_counts'):
            self._trade_cycle_counts = {}

        TIME_STOP_CYCLES = 3  # Close after 3 cycles with no profit

        try:
            positions = self.broker.get_positions()
        except Exception:
            return

        if not positions:
            # Clean stale tracking
            self._trade_cycle_counts.clear()
            return

        active_tickets = set()
        for pos in positions:
            ticket = pos.get('ticket')
            if not ticket:
                continue
            active_tickets.add(ticket)

            entry_price = float(pos.get('price_open', 0) or pos.get('entry_price', 0))
            current_price = float(pos.get('price_current', 0) or 0)
            direction = 'BUY' if pos.get('type', 0) == 0 else 'SELL'
            pair = pos.get('symbol', '')

            if ticket not in self._trade_cycle_counts:
                self._trade_cycle_counts[ticket] = {
                    'entry_price': entry_price,
                    'direction': direction,
                    'pair': pair,
                    'cycles': 0,
                }

            info = self._trade_cycle_counts[ticket]
            info['cycles'] += 1

            # Check if trade is in profit
            if direction == 'BUY':
                in_profit = current_price > entry_price
            else:
                in_profit = current_price < entry_price

            if in_profit:
                # Reset cycle counter — trade is working
                info['cycles'] = 0
            elif info['cycles'] >= TIME_STOP_CYCLES:
                # Time stop triggered — close at market
                pnl = pos.get('profit', 0)
                bot_logger.warning(
                    f"⏰ TIME STOP: {pair} ticket {ticket} — "
                    f"no profit after {TIME_STOP_CYCLES} cycles "
                    f"(entry={entry_price:.5f}, current={current_price:.5f}, P/L=${pnl:.2f})"
                )
                try:
                    result = self.broker.close_position(ticket)
                    if result:
                        bot_logger.info(f"⏰ TIME STOP closed ticket {ticket}")
                        self.risk_manager.on_trade_closed(pnl)
                        self._record_ml_outcome(pair, pnl)
                        self._record_rl_outcome(pair, pnl, 'time_stop')
                except Exception as e:
                    bot_logger.warning(f"Time stop close failed for {ticket}: {e}")

        # Clean tracking for positions that no longer exist
        stale = [t for t in self._trade_cycle_counts if t not in active_tickets]
        for t in stale:
            del self._trade_cycle_counts[t]

    @staticmethod
    def _normalize_pair(pair):
        """Normalize pair format (EUR/USD and EURUSD -> EURUSD)."""
        return str(pair or '').replace('/', '').upper()

    def _record_ml_outcome(self, pair: str, pnl: float):
        """Feed a closed trade's features + outcome to the ML Trade Scorer."""
        try:
            if pnl == 0:
                return  # Skip breakeven trades — not informative
            features = getattr(self, '_pending_ml_features', {}).pop(pair, None)
            if features is not None:
                is_win = pnl > 0
                self.ensemble.record_ml_trade(features, is_win)
                bot_logger.info(
                    f"🧠 ML recorded: {pair} {'WIN' if is_win else 'LOSS'} "
                    f"(${pnl:+.2f}) — "
                    f"{self.ensemble.ml_scorer.get_status()['training_samples']} total samples"
                )
        except Exception as e:
            bot_logger.warning(f"🧠 ML outcome recording failed: {e}")

    def _record_rl_outcome(self, pair: str, pnl: float, exit_type: str = ''):
        """Feed trade outcome to the RL agent for reward learning."""
        try:
            rl_info = getattr(self, '_pending_rl_info', {}).pop(pair, None)
            if rl_info is None:
                return

            state = rl_info['state']
            action = rl_info['action']
            pips = rl_info.get('pips', pnl)
            trade_result = {'pips': pips, 'exit_type': exit_type}

            reward = self.rl_agent.compute_reward(action, trade_result=trade_result)
            # Use current state as next_state (simplified)
            next_state = state  # In practice, market state at close
            won = pnl > 0
            rr = abs(pips / max(rl_info.get('sl_pips', 1), 0.1)) if pips else 0

            self.rl_agent.record_outcome(
                state, action, reward, next_state, done=False,
                trade_info={'won': won, 'pips': pips, 'rr': rr}
            )

            bot_logger.info(
                f"🤖 RL reward: {pair} reward={reward:+.2f} "
                f"(action={self.rl_agent.get_action_name(action)}, ε={self.rl_agent.epsilon:.3f})"
            )

            # Periodically save RL state
            if self.rl_agent.total_trades % 25 == 0:
                self.rl_agent.save_state()
        except Exception as e:
            bot_logger.warning(f"🤖 RL outcome recording failed: {e}")

    def _find_open_position(self, pair):
        """Return the currently open position for a pair, if any."""
        pair_key = self._normalize_pair(pair)

        if self.mode == 'paper' and self.paper_trader:
            for p in self.paper_trader.open_positions.values():
                if self._normalize_pair(p.get('pair', '')) == pair_key:
                    return p
            return None

        if not self.broker:
            return None

        # Only look at bot-placed positions (magic=234000)
        positions = self.broker.get_bot_positions() or []
        for p in positions:
            if self._normalize_pair(p.get('pair', '')) == pair_key:
                return p
        return None

    def _has_bot_position(self, pair):
        """Return True if the bot already has an open position on this pair."""
        return self._find_open_position(pair) is not None

    @staticmethod
    def _position_age_minutes(position):
        """Estimate position age in minutes from known time fields."""
        raw_open_time = position.get('entry_time') or position.get('open_time')
        if not raw_open_time:
            return None

        if isinstance(raw_open_time, datetime):
            open_time = raw_open_time
        elif isinstance(raw_open_time, str):
            try:
                open_time = datetime.fromisoformat(raw_open_time.replace('Z', '+00:00'))
            except Exception:
                return None
        else:
            return None

        if open_time.tzinfo is None:
            open_time = open_time.replace(tzinfo=timezone.utc)

        now_utc = datetime.now(timezone.utc)
        return max(0.0, (now_utc - open_time).total_seconds() / 60.0)

    def _manage_open_position(self, pair, signal_result):
        """Actively close stale or strongly reversed positions to free trade slots."""
        position = self._find_open_position(pair)
        if not position:
            return

        position_type = str(position.get('type', '')).upper()
        signal = signal_result.get('signal', 'SKIP')
        confidence = signal_result.get('confidence', 0.0)
        agreement = signal_result.get('models_agreement', 0)
        min_required = signal_result.get('min_agreement_required', 2)

        opposite_signal = signal in ('BUY', 'SELL') and signal != position_type
        strong_reversal = (
            opposite_signal and
            confidence >= self.reversal_exit_confidence and
            agreement >= max(self.reversal_exit_min_agreement, min_required)
        )

        age_minutes = self._position_age_minutes(position)
        stale_trade = (
            age_minutes is not None and
            age_minutes >= self.max_trade_hold_minutes and
            (signal == 'SKIP' or confidence < self.reversal_exit_confidence)
        )

        if not strong_reversal and not stale_trade:
            return

        if strong_reversal:
            exit_reason = 'REVERSAL_SIGNAL'
        else:
            exit_reason = 'STALE_SIGNAL'

        if self.mode == 'paper' and self.paper_trader:
            latest_price = self.broker.get_latest_price(pair) if self.broker else None
            exit_price = position.get('current_price', position.get('entry_price', 0))
            if latest_price and latest_price.get('bid'):
                exit_price = latest_price['bid']

            closed_trade = self.paper_trader.close_position(pair, exit_price, exit_reason)
            if not closed_trade:
                return

            self.risk_manager.on_trade_closed(closed_trade.get('profit_loss', 0.0))
            bot_logger.info(
                f"🔄 Active exit {pair}: {position_type} closed ({exit_reason}) "
                f"after {age_minutes:.0f}m"
                if age_minutes is not None
                else f"🔄 Active exit {pair}: {position_type} closed ({exit_reason})"
            )

            model_signals = getattr(self, '_pending_trade_signals', {}).pop(pair, {})
            self.ensemble.record_trade_result({
                'pair': pair,
                'signal': closed_trade.get('type', position_type),
                'profit_loss': closed_trade.get('profit_loss', 0.0),
                'entry_price': closed_trade.get('entry_price', 0),
                'exit_price': closed_trade.get('exit_price', 0),
                'exit_type': exit_reason,
                'model_signals': model_signals,
                'regime': getattr(self, '_last_regime', {}).get(pair, 'unknown'),
            })

            # Record with ML Trade Scorer
            self._record_ml_outcome(pair, closed_trade.get('profit_loss', 0.0))
            return

        if self.mode == 'live' and self.broker:
            volume = float(position.get('volume', 0.01) or 0.01)
            ticket = position.get('ticket')
            closed = self.broker.close_position(pair=pair, volume=volume, ticket=ticket)
            if closed:
                bot_logger.info(
                    f"🔄 Active exit requested for {pair} ticket={ticket} ({exit_reason})"
                )
    
    def start(self):
        """Start the trading bot"""
        if self.running:
            bot_logger.warning("Bot is already running")
            return
        
        bot_logger.info(f"🚀 Starting Trading Bot in {self.mode.upper()} mode")
        self.running = True
        
        # Start dashboard in background
        if self.enable_dashboard:
            try:
                from src.dashboard.app import start_dashboard
                dashboard_thread = threading.Thread(
                    target=start_dashboard,
                    kwargs={
                        'bot': self,
                        'learner': self.ensemble.learner,
                        'port': int(os.environ.get('DASHBOARD_PORT', 5000)),
                    },
                    daemon=True,
                )
                dashboard_thread.start()
            except Exception as e:
                bot_logger.warning(f"Dashboard failed to start: {e}")
        
        # Schedule dual-timeframe scalping cycles
        # 1-minute analysis every 60 seconds
        self.scheduler.add_job(
            self.analyze_and_trade,
            'interval',
            seconds=60,
            args=['1m'],
            id='scalp_1m_cycle',
            replace_existing=True
        )
        # 5-minute analysis every 5 minutes
        self.scheduler.add_job(
            self.analyze_and_trade,
            'interval',
            minutes=5,
            args=['5m'],
            id='scalp_5m_cycle',
            replace_existing=True
        )
        
        # Schedule daily reset at 00:00 UTC
        self.scheduler.add_job(
            self.reset_daily_limits,
            'cron',
            hour=0,
            minute=0,
            id='daily_reset',
            timezone='UTC'
        )
        
        # Schedule LSTM auto-retrain every 24 hours at 03:00 UTC (low-activity period)
        if self.lstm_retrainer.available:
            self.scheduler.add_job(
                self.lstm_retrainer.retrain,
                'cron',
                hour=3,
                minute=0,
                id='lstm_retrain',
                timezone='UTC',
                replace_existing=True,
            )
            bot_logger.info("🧠 LSTM auto-retraining scheduled (daily 03:00 UTC)")

        # Add listener to log APScheduler job errors visibly
        def _job_error_listener(event):
            if event.exception:
                bot_logger.error(
                    f"💥 Scheduler job {event.job_id} crashed: {event.exception}",
                    exc_info=True,
                )

        try:
            from apscheduler.events import EVENT_JOB_ERROR
            self.scheduler.add_listener(_job_error_listener, EVENT_JOB_ERROR)
        except Exception:
            pass

        self.scheduler.start()
        
        # Run first analysis immediately (5m first, then 1m will follow on schedule)
        bot_logger.info("Running initial scalping analysis...")
        self.analyze_and_trade('5m')
        
        bot_logger.info("Scalping bot running (1M every 60s, 5M every 5min). Press Ctrl+C to stop.")
        
        try:
            while self.running:
                time.sleep(60)  # Keep running
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        bot_logger.info("⏹️  Stopping Trading Bot")
        self.running = False
        
        if self.scheduler.running:
            self.scheduler.shutdown()
        
        if self.broker:
            self.broker.shutdown()
        
        # Print paper trading summary if applicable
        if self.mode == 'paper':
            summary = self.paper_trader.get_summary()
            bot_logger.info(f"\n📈 Paper Trading Summary:")
            bot_logger.info(f"  Total Trades: {summary['total_trades']}")
            bot_logger.info(f"  Win Rate: {summary['win_rate']:.1f}%")
            bot_logger.info(f"  Total Profit: ${summary['total_profit']:.2f}")
            bot_logger.info(f"  Initial Balance: ${summary['initial_balance']:.2f}")
            bot_logger.info(f"  Current Balance: ${summary['current_balance']:.2f}")
            bot_logger.info(f"  Return: {summary['return_percent']:.2f}%")
    
    def _detect_closed_trades(self):
        """Detect trades closed by MT5 (TP/SL) and record them with adaptive learner.
        
        Compares current open tickets against _known_tickets.
        Any ticket that disappeared was closed — look it up in deal history.
        
        NOTE: Does NOT call risk_manager.on_trade_closed() for balance adjustment
        because _sync_balance() already synced the authoritative broker balance.
        Only decrements the open_trades counter and records the trade for the learner.
        """
        if self.mode != 'live' or not self.broker:
            return

        try:
            positions = self.broker.get_open_positions()
            if positions is None:
                return

            current_tickets = {p['ticket'] for p in positions}

            # Update known tickets with any new positions
            for p in positions:
                if p['ticket'] not in self._known_tickets:
                    self._known_tickets[p['ticket']] = {
                        'pair': p.get('pair', ''),
                        'type': p.get('type', ''),
                        'entry_price': p.get('open_price', 0),
                        'open_time': p.get('open_time', ''),
                    }

            # Find tickets that vanished (closed by MT5)
            closed_tickets = set(self._known_tickets.keys()) - current_tickets
            if not closed_tickets:
                return

            # Query deal history to get P/L for closed trades
            history = self.broker.get_trade_history(hours=24)

            for ticket in closed_tickets:
                info = self._known_tickets.pop(ticket, {})
                pair = info.get('pair', 'UNKNOWN')
                trade_type = info.get('type', 'UNKNOWN')
                entry_price = info.get('entry_price', 0)

                # Find matching deal in history (match by position_id = ticket)
                deal = None
                for d in history:
                    pos_id = d.get('position_id', d.get('ticket', 0))
                    if pos_id == ticket:
                        deal = d
                        break

                if deal:
                    profit = deal.get('profit', 0) + deal.get('swap', 0) + deal.get('commission', 0)
                    exit_price = deal.get('price', 0)
                    is_win = profit > 0
                    exit_type = 'TAKE_PROFIT' if is_win else 'STOP_LOSS'
                else:
                    # No deal found — estimate from last known state
                    profit = 0
                    exit_price = 0
                    exit_type = 'UNKNOWN'

                bot_logger.info(
                    f"📊 Closed trade detected: {pair} {trade_type} | "
                    f"P/L: ${profit:+.2f} | Exit: {exit_type}"
                )

                # Decrement open_trades counter (balance is synced from broker
                # by _sync_balance — do NOT double-count P/L here)
                self.risk_manager.open_trades = max(0, self.risk_manager.open_trades - 1)

                # Record with adaptive learner (for weight adaptation only)
                model_signals = getattr(self, '_pending_trade_signals', {}).pop(pair, {})
                self.ensemble.record_trade_result({
                    'pair': pair,
                    'signal': trade_type,
                    'profit_loss': profit,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_type': exit_type,
                    'model_signals': model_signals,
                    'regime': getattr(self, '_last_regime', {}).get(pair, 'unknown'),
                })

                # Record with ML Trade Scorer
                self._record_ml_outcome(pair, profit)

                # Clean from trailing stop tracker too
                if hasattr(self, 'trailing'):
                    self.trailing._tracking.pop(ticket, None)

        except Exception as e:
            bot_logger.warning(f"Closed trade detection failed: {e}")

    def _backfill_history(self):
        """One-time backfill: load closed deals from MT5 history into the adaptive learner.
        
        Only imports deals that aren't already in the learner's trade_history,
        using position_id to deduplicate.
        
        Set SKIP_HISTORY_BACKFILL=true in .env to disable (recommended when starting fresh).
        """
        # Backfill disabled by default — old trades from a different strategy
        # should not influence the adaptive learner.  Set SKIP_HISTORY_BACKFILL=false to enable.
        if os.getenv('SKIP_HISTORY_BACKFILL', 'true').lower() not in ('false', '0', 'no'):
            bot_logger.info("📊 History backfill skipped (set SKIP_HISTORY_BACKFILL=false to enable)")
            return
        
        if self.mode != 'live' or not self.broker:
            return
        try:
            history = self.broker.get_trade_history(hours=168)  # Last 7 days
            if not history:
                return

            # Build set of already-known position IDs
            existing = set()
            for t in self.ensemble.learner.trade_history:
                pid = t.get('position_id') or t.get('ticket', 0)
                if pid:
                    existing.add(pid)

            imported = 0
            for deal in history:
                pos_id = deal.get('position_id', deal.get('ticket', 0))
                if pos_id in existing or pos_id in self._processed_closures:
                    continue

                profit = deal.get('profit', 0) + deal.get('swap', 0) + deal.get('commission', 0)
                # Normalise pair format: EURUSD → EUR/USD
                raw_pair = deal.get('pair', '')
                if len(raw_pair) == 6 and '/' not in raw_pair:
                    pair = f"{raw_pair[:3]}/{raw_pair[3:]}"
                else:
                    pair = raw_pair
                
                # Skip trades for pairs we're not currently trading
                if pair not in PAIRS:
                    continue
                # The closing deal's type is the *exit* direction — flip for the original signal
                exit_dir = deal.get('type', '')
                trade_type = 'SELL' if exit_dir == 'BUY' else 'BUY'

                self.ensemble.record_trade_result({
                    'pair': pair,
                    'signal': trade_type,
                    'profit_loss': profit,
                    'entry_price': 0,
                    'exit_price': deal.get('price', 0),
                    'exit_type': 'TAKE_PROFIT' if profit > 0 else 'STOP_LOSS',
                    'model_signals': {},
                    'position_id': pos_id,
                    'timestamp': deal.get('time', ''),
                })
                self._processed_closures.add(pos_id)
                imported += 1

            if imported:
                bot_logger.info(f"📊 Backfilled {imported} past trades from MT5 history")
        except Exception as e:
            bot_logger.warning(f"History backfill failed: {e}")

    def _sync_balance(self):
        """Sync balance from broker or paper trader into the risk manager.
        
        If relay fails, keep the last known good values instead of
        falling back to INITIAL_BALANCE ($50).
        """
        try:
            if self.mode == 'paper' and self.paper_trader:
                self.risk_manager.sync_balance(self.paper_trader.current_balance)
            elif self.broker:
                acct = self.broker.get_account_info()
                if acct and acct.get('balance', 0) > 0:
                    self.risk_manager.sync_balance(
                        acct['balance'],
                        leverage=acct.get('leverage'),
                        free_margin=acct.get('margin_free'),
                    )
                else:
                    bot_logger.warning(
                        f"Balance sync returned None — keeping last known "
                        f"${self.risk_manager.current_balance:.2f}"
                    )
        except Exception as e:
            bot_logger.warning(f"Balance sync failed (keeping ${self.risk_manager.current_balance:.2f}): {e}")

    def _sync_open_trades(self):
        """Sync open trade count from broker so the counter stays accurate.
        
        In live mode the risk manager's open_trades counter can drift
        (e.g., MT5 closes a trade via SL/TP while bot is sleeping).
        Only counts bot-placed trades (magic=234000) so manual trades
        don't consume the bot's trade slots.
        """
        if self.mode != 'live' or not self.broker:
            return
        try:
            bot_positions = self.broker.get_bot_positions()
            all_positions = self.broker.get_open_positions()
            if bot_positions is not None:
                bot_count = len(bot_positions)
                total_count = len(all_positions) if all_positions else 0
                manual_count = total_count - bot_count
                if bot_count != self.risk_manager.open_trades:
                    bot_logger.info(
                        f"Position sync: risk_manager had {self.risk_manager.open_trades} "
                        f"bot trades, broker has {bot_count} bot + {manual_count} manual — corrected"
                    )
                    self.risk_manager.open_trades = bot_count
        except Exception as e:
            bot_logger.warning(f"Position sync failed: {e}")

    def reset_daily_limits(self):
        """Reset daily trading limits and decay loss patterns"""
        self._sync_balance()
        self.risk_manager.reset_daily_limits(self.risk_manager.current_balance)
        # Decay loss pattern counts so old losses don't block forever
        if hasattr(self, 'adaptive_learner') and self.adaptive_learner:
            self.adaptive_learner.decay_loss_patterns(factor=0.85)


# Entry point
if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    bot = TradingBot(
        newsapi_key=os.getenv('NEWSAPI_KEY'),
        enable_dashboard=True,
    )
    bot.start()
