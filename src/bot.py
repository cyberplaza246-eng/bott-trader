"""
Dual-Timeframe Scalping Bot — 1M + 5M Confluence System

Runs continuous 1-minute and 5-minute scalping cycles.
Supports both forex (MT5) and futures (TradersPost → Tradovate → Lucid).
Uses a 9-model ensemble (including ScalpingAnalyzer as primary signal) with:
  - Adaptive learning, trailing stops, economic calendar
  - S/R-aware exits, cross-pair correlation, LSTM auto-retraining
  - Per-timeframe confluence bonuses/penalties
  - Tick-based SL/TP from instrument registry
  - Prop firm guard & evaluation passing algorithm (futures mode)
"""
import os
import time
import threading
from datetime import datetime, time as dt_time, timezone, timedelta
from apscheduler.schedulers.background import BackgroundScheduler

from src.broker.broker_factory import create_broker
from src.core.ensemble_trader import EnsembleTrader
from src.risk.position_manager import RiskManager
from src.core.paper_trading import PaperTradingManager
from src.core.trailing_stop import TrailingStopManager
from src.ai.economic_calendar import EconomicCalendar
from src.ai.lstm_retrainer import LSTMRetrainer
from src.ai.rl_agent import RLTradingAgent
from src.risk.prop_guard import PropGuard
from src.risk.eval_algorithm import EvalAlgorithm, EvalConfig
from src.instruments import get_instrument, is_futures, is_maintenance_window, REGISTRY
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
    ASSET_CLASS,
    BROKER_TYPE,
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
        # Futures: MES and ES are same underlying, MNQ and NQ likewise
        'MES': 'ES',
        'ES': 'MES',
        'MNQ': 'NQ',
        'NQ': 'MNQ',
    }

    # ── Spread Limits — built from instrument registry ─────────────
    MAX_SPREAD = {}
    for _sym, _limit in SCALPING_SPREAD_LIMITS.items():
        try:
            _spec = get_instrument(_sym)
            MAX_SPREAD[_sym] = _limit * _spec.tick_size
        except KeyError:
            pass
    DEFAULT_MAX_SPREAD = 0.00020

    def __init__(self, newsapi_key=None, enable_dashboard=True):
        self.mode = TRADING_MODE  # 'live', 'paper', 'backtest'
        self.running = False
        self.scheduler = BackgroundScheduler()
        self.enable_dashboard = enable_dashboard
        self.signal_history = []  # For dashboard
        self.asset_class = ASSET_CLASS
        
        # Initialize broker via factory (MT5 for forex, TradersPost for futures)
        if self.mode in ['live', 'paper']:
            try:
                self.broker = create_broker(BROKER_TYPE)
            except Exception as e:
                bot_logger.error(f"Failed to initialize broker ({BROKER_TYPE}): {e}")
                self.broker = None
        
        self.ensemble = EnsembleTrader(newsapi_key=newsapi_key, broker=self.broker)
        
        # Use actual broker balance when connected, fall back to config
        actual_balance = INITIAL_BALANCE
        if self.broker and self.broker.connected:
            try:
                info = self.broker.get_account_info()
                if info and info.get('balance', 0) > 0:
                    actual_balance = info['balance']
                    bot_logger.info(f"💰 Using actual broker balance: ${actual_balance:.2f}")
            except Exception:
                pass
        self.risk_manager = RiskManager(initial_balance=actual_balance)
        self.trailing = TrailingStopManager(breakeven_r=1.0, trail_atr_mult=1.5)
        self.calendar = EconomicCalendar()
        self.lstm_retrainer = LSTMRetrainer(
            broker=self.broker,
            predictor=self.ensemble.lstm if self.ensemble.lstm_available else None,
        )

        # ── Prop Firm Guard & Eval Algorithm (futures mode) ──────
        if self.asset_class == 'futures':
            self.prop_guard = PropGuard(
                account_size=actual_balance,
                max_drawdown=float(os.getenv('MAX_DRAWDOWN', '2000')),
                daily_loss_limit=float(os.getenv('DAILY_LOSS_LIMIT', '400')),
                max_contracts=int(os.getenv('MAX_CONTRACTS', '2')),
                max_positions=int(os.getenv('MAX_POSITIONS', '3')),
            )
            self.eval_algo = EvalAlgorithm(EvalConfig(
                account_size=actual_balance,
                profit_target=actual_balance + float(os.getenv('PROFIT_TARGET', '3000')),
                daily_target=float(os.getenv('DAILY_TARGET', '500')),
                daily_stop_loss=float(os.getenv('DAILY_STOP_LOSS', '400')),
            ))
        else:
            self.prop_guard = None
            self.eval_algo = None
        
        # RL agent for trade decision optimization
        self.rl_agent = RLTradingAgent()

        # Determine active model count
        model_count = 9 if self.ensemble.lstm_available else 8
        
        if self.mode == 'paper':
            self.paper_trader = PaperTradingManager(initial_balance=INITIAL_BALANCE)
        else:
            self.paper_trader = None
        
        # TRADE EXECUTION LOCK - prevents 1m and 5m from placing trades simultaneously
        self._trade_lock = threading.Lock()
        # CLOSED TRADE DETECTION LOCK - prevents duplicate closure processing
        self._closure_lock = threading.Lock()
        
        self.last_signal_time = {}  # Track last signal per pair per timeframe
        # For scalping, use much longer hold time - let TP/SL do the work
        self.max_trade_hold_minutes = int(os.getenv('MAX_TRADE_HOLD_MINUTES', '180'))
        self.reversal_exit_confidence = float(os.getenv('REVERSAL_EXIT_CONFIDENCE', '0.85'))
        self.reversal_exit_min_agreement = int(os.getenv('REVERSAL_EXIT_MIN_AGREEMENT', '3'))
        self.enable_correlation_guard = os.getenv('ENABLE_CORRELATION_GUARD', 'true').lower() == 'true'

        # Dual-timeframe confluence tracking: {pair: {'1m': signal_result, '5m': signal_result}}
        self._last_signals = {pair: {} for pair in PAIRS}

        # Live closure detection: track open tickets so we notice when MT5 closes one
        self._known_tickets = {}   # {ticket: {pair, type, entry_price, open_time}}
        self._processed_closures = set()  # position_ids already recorded
        
        # Backfill past closed trades from MT5 history into adaptive learner
        self._backfill_history()
        
        # STARTUP SAFETY: Check and close excess positions
        self._enforce_max_positions_on_startup()
        
        bot_logger.info(
            f"✅ Scalping Bot initialized in {self.mode.upper()} mode "
            f"({model_count}-model ensemble, 1M+5M dual-timeframe)"
        )

        # ── Learning System Status ────────────────────────────────
        self._log_learning_status()

    # ── Startup Position Enforcement ──────────────────────────────

    MAX_ALLOWED_POSITIONS = 3  # ABSOLUTE MAXIMUM - never more than this
    
    def _enforce_max_positions_on_startup(self):
        """Close excess positions on startup to enforce max limit."""
        if self.mode != 'live' or not self.broker:
            return
        try:
            positions = self.broker.get_open_positions()
            if positions is None:
                bot_logger.warning("Startup check skipped: could not fetch positions")
                return
            if not positions:
                bot_logger.info(f"✅ Startup check: 0 positions open")
                return
            
            count = len(positions)
            bot_logger.info(f"🔍 Startup check: {count} positions open (max {self.MAX_ALLOWED_POSITIONS})")
            
            if count > self.MAX_ALLOWED_POSITIONS:
                excess = count - self.MAX_ALLOWED_POSITIONS
                bot_logger.warning(f"🚨 CLOSING {excess} EXCESS POSITIONS!")
                
                # Sort by profit (close worst ones first)
                sorted_pos = sorted(positions, key=lambda p: p.get('profit', 0))
                
                for i in range(excess):
                    pos = sorted_pos[i]
                    result = self.broker.close_position(
                        pair=pos.get('pair'),
                        volume=pos.get('volume'),
                        ticket=pos.get('ticket'),
                    )
                    if result:
                        bot_logger.info(
                            f"✅ Closed excess position {pos.get('ticket')} "
                            f"({pos.get('pair')} P/L: {pos.get('profit', 0):.2f})"
                        )
                    else:
                        bot_logger.error(f"❌ Failed to close {pos.get('ticket')}")
                
                # Verify
                import time
                time.sleep(1)
                remaining = self.broker.get_open_positions() or []
                bot_logger.info(f"✅ After cleanup: {len(remaining) if remaining else 0} positions")
        except Exception as e:
            bot_logger.error(f"Startup position check failed: {e}")

    def _enforce_max_positions(self):
        """Periodic check to close excess positions. Called each cycle."""
        if self.mode != 'live' or not self.broker:
            return
        try:
            positions = self.broker.get_open_positions()
            if positions is None:
                bot_logger.warning("Enforcement skipped: could not fetch positions")
                return
            count = len(positions)
            bot_logger.info(f"🛡️ Enforcement check: {count}/{self.MAX_ALLOWED_POSITIONS} positions")
            if not positions:
                return
            if count > self.MAX_ALLOWED_POSITIONS:
                bot_logger.error(f"🚨 ENFORCEMENT: {count} positions detected! Max is {self.MAX_ALLOWED_POSITIONS}. Closing excess.")
                sorted_pos = sorted(positions, key=lambda p: p.get('profit', 0))
                excess = count - self.MAX_ALLOWED_POSITIONS
                for i in range(excess):
                    pos = sorted_pos[i]
                    result = self.broker.close_position(
                        pair=pos.get('pair'),
                        volume=pos.get('volume'),
                        ticket=pos.get('ticket'),
                    )
                    if result:
                        bot_logger.info(f"✅ Enforced close: {pos.get('ticket')} ({pos.get('pair')})")
                    else:
                        bot_logger.error(f"❌ Enforcement close failed: {pos.get('ticket')} ({pos.get('pair')})")
        except Exception as e:
            bot_logger.warning(f"Position enforcement check failed: {e}")

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
                _spec = REGISTRY.get(pair)
                pip_div = _spec.tick_size if _spec else (0.01 if 'JPY' in pair else 0.0001)
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
        
        # ENFORCE MAX POSITIONS every cycle
        self._enforce_max_positions()
        
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

        # Hourly learning status log (only on 5m cycle, once per hour)
        if timeframe_key == '5m':
            now_min = datetime.now().minute
            if now_min < 5 and not getattr(self, '_last_learning_log_hour', -1) == datetime.now().hour:
                self._last_learning_log_hour = datetime.now().hour
                self._log_learning_status()
        
        # Detect trades closed by MT5 (TP/SL hit) and record them
        self._detect_closed_trades()

        # Evaluate RL skip decisions (did skipping avoid a loss?)
        try:
            self._process_rl_pending_skips()
        except Exception as e:
            bot_logger.debug(f"RL skip processing: {e}")
        
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

            # ── Time Stop ─────────────────────────────────────────────
            # Close trades not in profit after MAX_FLAT_MINUTES (real elapsed time).
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

                    # ── RL Agent ──────────────────────────────────────
                    # Run ALL signals through RL so it learns from everything.
                    # High-confidence signals proceed regardless of RL action
                    # (RL is advisory, not gating), but RL still records the
                    # outcome for training.
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
                            # Store state for later reward computation
                            if not hasattr(self, '_rl_pending_skips'):
                                self._rl_pending_skips = {}
                            self._rl_pending_skips[f"{pair}_{timeframe_key}"] = {
                                'state': rl_state, 'action': rl_action,
                                'signal': signal_result,
                                'timestamp': datetime.now(),
                            }
                            # Advisory mode: let high-confidence signals through
                            # regardless of RL SKIP so RL learns from all outcomes
                            if signal_result['confidence'] >= 0.40:
                                bot_logger.info(
                                    f"  🤖 RL: SKIP overridden — high confidence "
                                    f"({signal_result['confidence']:.1%}) (ε={self.rl_agent.epsilon:.3f})"
                                )
                                signal_result['_rl_state'] = rl_state
                                signal_result['_rl_action'] = rl_action
                                signal_result['_rl_lot_mult'] = 1.0
                            else:
                                bot_logger.info(f"  🤖 RL Agent: SKIP trade on {pair} (ε={self.rl_agent.epsilon:.3f})")
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
            # Get actual position count from broker to ensure accurate display
            actual_positions = 0
            if self.mode == 'live' and self.broker:
                positions = self.broker.get_open_positions()
                if positions is not None:
                    actual_positions = len(positions)
                    # Sync internal counter with reality
                    self.risk_manager.sync_open_trades(actual_positions)
            
            daily_status = self.risk_manager.get_daily_status(actual_open_trades=actual_positions)
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
        """Execute a scalping trade with unified SL/TP from S/R levels.
        
        Uses a lock to ensure only one trade can be evaluated and placed at a time,
        preventing race conditions between 1m and 5m timeframe threads.
        """
        with self._trade_lock:
            return self._execute_trade_locked(pair, signal_result, df, timeframe_key)
    
    def _execute_trade_locked(self, pair, signal_result, df, timeframe_key='5m'):
        """Internal trade execution (called with _trade_lock held)."""
        from src.risk.sl_tp import calculate_sl_tp

        trade_type = signal_result['signal']
        enriched_df = signal_result.get('enriched_df', df)
        latest = enriched_df.iloc[-1]
        entry_price = latest['close']
        atr = latest.get('atr', entry_price * 0.001)

        # ── Price drift check ────────────────────────────────────────
        sweep_rr = signal_result.get('sweep_sl_tp')
        scalping_rr = signal_result.get('scalping_risk_reward', {})
        signal_entry = None
        if sweep_rr:
            signal_entry = sweep_rr.get('entry_price')
        elif scalping_rr:
            signal_entry = scalping_rr.get('entry_price')

        # Use configurable drift tolerance from risk_overrides.json (default 8.0×ATR)
        # 1M ATR is small (2-3 pips), so signal entries from a few candles ago
        # can easily drift 10-20 pips by the time the bot confirms the signal.
        drift_tolerance_atr = 8.0
        try:
            import json
            overrides_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'risk_overrides.json')
            if os.path.exists(overrides_path):
                with open(overrides_path) as f:
                    overrides = json.load(f)
                    drift_tolerance_atr = overrides.get('price_drift_tolerance_atr', 8.0)
        except Exception:
            pass
        if signal_entry and abs(entry_price - signal_entry) > atr * drift_tolerance_atr:
            pip_mult = 100 if 'JPY' in pair else 10000
            drift_pips = abs(entry_price - signal_entry) * pip_mult
            bot_logger.warning(
                f"❌ {pair} price drifted {drift_pips:.1f}p from signal entry "
                f"({signal_entry:.5f} → {entry_price:.5f}) — trade SKIPPED (tolerance: {drift_tolerance_atr}×ATR)"
            )
            return

        # ── Unified SL/TP calculation ────────────────────────────────
        sweep_wick = signal_result.get('sweep_wick')
        sr_levels = signal_result.get('sr_levels', {})

        sl_tp = calculate_sl_tp(
            df=enriched_df,
            direction=trade_type,
            pair=pair,
            timeframe=timeframe_key,
            sr_levels=sr_levels,
            sweep_wick=sweep_wick,
        )

        if sl_tp is None:
            return

        stop_loss = sl_tp['stop_loss']
        take_profit = sl_tp['take_profit']

        # Apply adaptive SL multiplier (auto-tuned per pair based on hit rate)
        try:
            sl_mult = self.ensemble.learner.get_sl_multiplier(pair)
            if sl_mult != 0.8:  # 0.8 is the default — only log if changed
                old_sl = stop_loss
                sl_distance = abs(entry_price - stop_loss)
                new_sl_distance = sl_distance * (sl_mult / 0.8)  # Adjust relative to default
                if trade_type == 'BUY':
                    stop_loss = entry_price - new_sl_distance
                else:
                    stop_loss = entry_price + new_sl_distance
                bot_logger.info(
                    f"📐 Adaptive SL: {pair} ×{sl_mult:.2f} "
                    f"({old_sl:.5f} → {stop_loss:.5f})"
                )
        except Exception:
            pass

        # Record SL for adaptive learner median tracking
        try:
            self.ensemble.learner.record_sl_outcome(pair, sl_tp['sl_distance'], True)
        except Exception:
            pass

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
            self._pending_rl_info[pair] = {
                'state': rl_state,
                'action': rl_action,
                'sl_pips': sl_tp['sl_pips'],
            }

        # VALIDATION: sanity check SL/TP sides
        sl_correct = (trade_type == 'BUY' and stop_loss < entry_price) or (trade_type == 'SELL' and stop_loss > entry_price)
        tp_correct = (trade_type == 'BUY' and take_profit > entry_price) or (trade_type == 'SELL' and take_profit < entry_price)

        if not sl_correct or not tp_correct:
            bot_logger.warning(
                f"❌ {pair} SL/TP wrong side — trade BLOCKED "
                f"(SL={'✅' if sl_correct else '❌'}, TP={'✅' if tp_correct else '❌'})"
            )
            return

        bot_logger.info(f"🔍 ORDER VALIDATION:")
        bot_logger.info(f"  Direction: {trade_type}")
        bot_logger.info(f"  Entry: {entry_price:.5f}")
        bot_logger.info(f"  SL: {stop_loss:.5f} ✅")
        bot_logger.info(f"  TP: {take_profit:.5f} ✅")

        # HARD BROKER CHECK: Get actual open positions from MT5 before placing order
        # This prevents drift/race conditions from causing over-trading
        # FAIL-CLOSED: If we can't verify position count, BLOCK the trade
        if self.mode == 'live' and self.broker:
            try:
                # Get ALL positions (not just bot positions) to ensure we never exceed limit
                all_positions = self.broker.get_open_positions()
                
                if all_positions is None:
                    # MT5 query failed - BLOCK trade for safety
                    bot_logger.error(f"🚫 SAFETY BLOCK: Could not verify position count — blocking {pair} trade")
                    return
                
                actual_count = len(all_positions)
                max_trades = self.risk_manager.get_tier_info()['max_concurrent_trades']
                
                bot_logger.info(f"📊 Position check: {actual_count}/{max_trades} positions open")
                
                if actual_count >= max_trades:
                    bot_logger.warning(
                        f"🚫 HARD LIMIT: {actual_count}/{max_trades} positions open — "
                        f"blocking new trade for {pair}"
                    )
                    return
            except Exception as e:
                # Exception during check - BLOCK trade for safety
                bot_logger.error(f"🚫 SAFETY BLOCK: Position check error ({e}) — blocking {pair} trade")
                return

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
                
                # Immediately add to _known_tickets so we can detect when it closes
                # (prevents race condition where TP/SL hits before next poll)
                self._known_tickets[order_id] = {
                    'pair': pair,
                    'type': trade_type,
                    'entry_price': entry_price,
                    'open_time': datetime.now().isoformat(),
                }
                
                bot_logger.info(f"🔍 Checking SL/TP attachment...")
                
                # Wait briefly then check if SL/TP were actually set 
                import time
                time.sleep(1.0)  
                
                # Check the actual position for SL/TP
                positions = self.broker.get_open_positions(pair)
                position_found = False
                for pos in positions:
                    if pos.get('ticket') == order_id:
                        position_found = True
                        actual_sl = pos.get('sl', 0)
                        actual_tp = pos.get('tp', 0)
                        if actual_sl == 0 and actual_tp == 0:
                            bot_logger.warning(f"⚠️ SL/TP NOT SET! Retrying modify for {order_id}...")
                            # Retry SL/TP via modify_position
                            for retry in range(3):
                                mod = self.broker.modify_position(order_id, sl=stop_loss, tp=take_profit)
                                if mod:
                                    bot_logger.info(f"✅ SL/TP set on retry {retry+1}: SL={stop_loss:.5f}, TP={take_profit:.5f}")
                                    break
                                time.sleep(0.5)
                            else:
                                bot_logger.error(f"❌ CRITICAL: SL/TP FAILED after 3 retries for {order_id}! SL={stop_loss:.5f}, TP={take_profit:.5f}")
                        else:
                            bot_logger.info(f"✅ SL/TP confirmed: SL={actual_sl:.5f}, TP={actual_tp:.5f}")
                        break
                
                if not position_found:
                    # Position ticket might differ — try modify anyway
                    bot_logger.warning(f"⚠️ Could not find position {order_id} — attempting SL/TP modify anyway")
                    self.broker.modify_position(order_id, sl=stop_loss, tp=take_profit)
                
                self.risk_manager.on_trade_opened()
                # Register for trailing stop management (aggressive profit protection)
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
                    quick_wins=True,  # ULTRA aggressive profit protection for 5m scalping
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

        # Store model signals and ML features keyed by TICKET (not pair)
        # so concurrent trades on the same pair don't overwrite each other.
        trade_ticket = locals().get('order_id') or locals().get('trade_id')
        if not hasattr(self, '_pending_trade_signals'):
            self._pending_trade_signals = {}
        sig_key = trade_ticket if trade_ticket else pair
        self._pending_trade_signals[sig_key] = signal_result.get('models', {})

        # ── ML Scorer: capture feature snapshot at entry time ────────
        if not hasattr(self, '_pending_ml_features'):
            self._pending_ml_features = {}
        try:
            ml_features = self.ensemble.capture_ml_features(signal_result, pair)
            self._pending_ml_features[sig_key] = ml_features
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
                    pending_signals = getattr(self, '_pending_trade_signals', {})
                    model_signals = pending_signals.pop(pair, {})
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
        """Close positions that haven't moved into profit after MIN_HOLD minutes.

        Uses real elapsed wall-clock time instead of cycle counting to avoid
        double-counting from 1m + 5m schedulers both calling this method.
        """
        MIN_HOLD_MINUTES = 10   # Never close a trade younger than 10 minutes
        # Use adaptive learner's time-exit candles (4-8) mapped to minutes
        # Each candle ≈ 5 min on the 5m timeframe, so candles × 5 = minutes
        try:
            learned_candles = self.ensemble.learner.get_time_exit_candles()
            MAX_FLAT_MINUTES = max(10, learned_candles * 5)
        except Exception:
            MAX_FLAT_MINUTES = 15   # Default: 15 minutes

        try:
            positions = self.broker.get_open_positions()
        except Exception:
            return

        if not positions:
            return

        now = datetime.now(timezone.utc)

        for pos in positions:
            ticket = pos.get('ticket')
            if not ticket:
                continue

            entry_price = float(pos.get('price_open', 0) or pos.get('entry_price', 0))
            current_price = float(pos.get('price_current', 0) or 0)
            direction = 'BUY' if pos.get('type', 0) == 0 else 'SELL'
            pair = pos.get('symbol', '')

            # Skip positions with invalid/missing price data — can't verify profit status
            if entry_price <= 0 or current_price <= 0:
                bot_logger.debug(f"⏰ TIME STOP skipped ticket {ticket}: invalid prices (entry={entry_price}, current={current_price})")
                continue

            # Calculate real age of the trade
            age_minutes = self._position_age_minutes(pos)
            if age_minutes is None or age_minutes < MIN_HOLD_MINUTES:
                continue  # Too young — never time-stop before MIN_HOLD_MINUTES

            # Check if trade is in profit
            if direction == 'BUY':
                in_profit = current_price > entry_price
            else:
                in_profit = current_price < entry_price

            if in_profit:
                continue  # Trade is working — leave it alone

            if age_minutes >= MAX_FLAT_MINUTES:
                # Time stop triggered — close at market
                pnl = pos.get('profit', 0)
                bot_logger.warning(
                    f"⏰ TIME STOP: {pair} ticket {ticket} — "
                    f"no profit after {age_minutes:.0f}min "
                    f"(entry={entry_price:.5f}, current={current_price:.5f}, P/L=${pnl:.2f})"
                )
                try:
                    result = self.broker.close_position(ticket=ticket)
                    if result:
                        bot_logger.info(f"⏰ TIME STOP closed ticket {ticket}")
                        self.risk_manager.on_trade_closed(pnl)

                        # Record with adaptive learner
                        pending_signals = getattr(self, '_pending_trade_signals', {})
                        model_signals = pending_signals.pop(ticket, None) or pending_signals.pop(pair, {})
                        self.ensemble.record_trade_result({
                            'pair': pair,
                            'signal': direction,
                            'profit_loss': pnl,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'exit_type': 'time_stop',
                            'model_signals': model_signals,
                            'regime': getattr(self, '_last_regime', {}).get(pair, 'unknown'),
                        })

                        self._record_ml_outcome(pair, pnl, ticket=ticket)
                        self._record_rl_outcome(pair, pnl, 'time_stop')

                        # Remove from known tickets so _detect_closed_trades doesn't double-record
                        self._known_tickets.pop(ticket, None)
                except Exception as e:
                    bot_logger.warning(f"Time stop close failed for {ticket}: {e}")

    def _log_learning_status(self):
        """Log current state of all learning systems."""
        try:
            learner = self.ensemble.learner
            trade_count = len(learner.trade_history)
            threshold = learner.get_adjusted_threshold()
            weights = learner.model_weights
            drawdown = learner.in_drawdown_protection

            bot_logger.info("=" * 50)
            bot_logger.info("📚 LEARNING SYSTEM STATUS")
            bot_logger.info(f"  Adaptive Learner: {trade_count} trades recorded")
            bot_logger.info(f"  Confidence threshold: {threshold:.2%}")
            bot_logger.info(f"  Drawdown protection: {'ACTIVE' if drawdown else 'off'}")

            # Model weights
            sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            top_models = ", ".join(f"{m}={w:.2f}" for m, w in sorted_w[:4])
            bot_logger.info(f"  Top model weights: {top_models}")

            # Pair stats
            for pair_key, stats in learner.pair_stats.items():
                total = stats.get('wins', 0) + stats.get('losses', 0)
                if total > 0:
                    wr = stats['wins'] / total * 100
                    bot_logger.info(f"  {pair_key}: {total} trades, {wr:.0f}% win rate")

            # ML scorer
            ml_status = self.ensemble.ml_scorer.get_status()
            samples = ml_status.get('training_samples', 0)
            trained = ml_status.get('is_trained', False)
            bot_logger.info(
                f"  ML Scorer: {samples}/50 samples"
                f" {'(TRAINED ✓)' if trained else ''}"
            )

            # RL agent
            rl_trades = self.rl_agent.total_trades
            rl_eps = self.rl_agent.epsilon
            bot_logger.info(f"  RL Agent: {rl_trades} experiences, ε={rl_eps:.3f}")

            # Loss pattern blocks
            blocked = []
            for pair_name in ['EURUSD', 'GBPUSD', 'USDJPY']:
                if learner.should_skip_loss_pattern(pair_name):
                    blocked.append(pair_name)
            if blocked:
                bot_logger.info(f"  ⚠️ Loss pattern blocks: {', '.join(blocked)}")

            bot_logger.info("=" * 50)
        except Exception as e:
            bot_logger.warning(f"Learning status log failed: {e}")

    @staticmethod
    def _normalize_pair(pair):
        """Normalize pair format (EUR/USD and EURUSD -> EURUSD)."""
        return str(pair or '').replace('/', '').upper()

    def _record_ml_outcome(self, pair: str, pnl: float, ticket=None, trade_type: str = ''):
        """Feed a closed trade's features + outcome to the ML Trade Scorer."""
        try:
            pending = getattr(self, '_pending_ml_features', {})
            # Try ticket-keyed first, then pair-keyed (legacy fallback)
            features = None
            if ticket:
                features = pending.pop(ticket, None)
            if features is None:
                features = pending.pop(pair, None)

            # If features are missing (e.g., trade opened before restart),
            # reconstruct a minimal feature vector so we still count the trade.
            if features is None:
                try:
                    # Build a synthetic signal_result from what we know
                    direction = trade_type if trade_type in ('BUY', 'SELL') else 'BUY'
                    synthetic_result = {
                        'signal': direction,
                        'confidence': 0.5,
                        'models_agreement': 1,
                        'total_models': 4,
                        'models': {},
                        'regime': getattr(self, '_last_regime', {}).get(pair, 'unknown'),
                    }
                    features = self.ensemble.capture_ml_features(synthetic_result, pair)
                    bot_logger.info(f"🧠 ML features reconstructed for {pair} (post-restart)")
                except Exception as e:
                    bot_logger.debug(f"🧠 ML feature reconstruction failed: {e}")
                    return

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

    def _process_rl_pending_skips(self):
        """Evaluate RL skip decisions: did skipping avoid a loss or miss a win?

        Checks price movement since the skip to estimate what would have happened,
        then records the appropriate reward (+2.0 for good skip, -0.5 for missed win).
        Skips older than 30 minutes are expired.
        """
        pending = getattr(self, '_rl_pending_skips', {})
        if not pending:
            return

        expired_keys = []
        now = datetime.now()

        for key, info in list(pending.items()):
            skip_time = info.get('timestamp')
            if skip_time is None:
                # Legacy entry without timestamp — add one and skip this cycle
                info['timestamp'] = now
                continue

            age_minutes = (now - skip_time).total_seconds() / 60.0

            # Evaluate after 10 minutes (enough time for a trade to play out)
            if age_minutes < 10:
                continue

            # Expire after 30 minutes
            if age_minutes > 30:
                expired_keys.append(key)
                continue

            state = info.get('state')
            action = info.get('action', 0)
            signal = info.get('signal', {})
            pair = key.split('_')[0]

            # Check what price did since the skip
            try:
                latest_price_data = self.broker.get_latest_price(pair) if self.broker else None
                if not latest_price_data:
                    continue

                current_price = latest_price_data.get('bid', 0) or latest_price_data.get('ask', 0)
                signal_entry = signal.get('enriched_df', {})
                if hasattr(signal_entry, 'iloc'):
                    entry_price = float(signal_entry.iloc[-1]['close'])
                else:
                    continue

                direction = signal.get('signal', 'SKIP')
                if direction == 'BUY':
                    would_have_won = current_price > entry_price
                elif direction == 'SELL':
                    would_have_won = current_price < entry_price
                else:
                    expired_keys.append(key)
                    continue

                # Compute skip reward
                trade_result = {'pips': 0, 'exit_type': 'hypothetical'}
                reward = self.rl_agent.compute_reward(action, trade_result=trade_result)
                # Override with skip-specific reward
                reward = -0.5 if would_have_won else 2.0

                next_state = state
                self.rl_agent.record_outcome(
                    state, action, reward, next_state, done=False,
                    trade_info={'won': False, 'pips': 0, 'rr': 0, 'skip_eval': True}
                )
                bot_logger.info(
                    f"🤖 RL skip eval: {pair} reward={reward:+.1f} "
                    f"({'good skip' if reward > 0 else 'missed win'})"
                )
                expired_keys.append(key)
            except Exception:
                expired_keys.append(key)

        for key in expired_keys:
            pending.pop(key, None)

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
        """Actively close stale or strongly reversed positions to free trade slots.
        
        CONSERVATIVE: Only exits on very strong reversals or truly old stale trades.
        Let TP/SL do the heavy lifting for normal exits.
        """
        position = self._find_open_position(pair)
        if not position:
            return

        position_type = str(position.get('type', '')).upper()
        signal = signal_result.get('signal', 'SKIP')
        confidence = signal_result.get('confidence', 0.0)
        agreement = signal_result.get('models_agreement', 0)
        min_required = signal_result.get('min_agreement_required', 2)
        pnl = float(position.get('profit', 0) or 0)

        # Strong reversal: opposite signal with high confidence
        opposite_signal = signal in ('BUY', 'SELL') and signal != position_type
        strong_reversal = (
            opposite_signal and
            confidence >= self.reversal_exit_confidence and
            agreement >= max(self.reversal_exit_min_agreement, min_required) and
            pnl < -0.10  # Only exit if losing money
        )

        age_minutes = self._position_age_minutes(position)
        
        # Stale trade: very old AND consistently failing signals AND in loss
        # Require ALL conditions to prevent premature exits
        stale_trade = (
            age_minutes is not None and
            age_minutes >= self.max_trade_hold_minutes and  # Default 180 min
            age_minutes <= 1440 and  # Sanity check: max 24 hours (prevent timezone bugs)
            pnl < -0.20 and  # Must be losing money
            (signal == 'SKIP' or confidence < 0.50)  # No strong signal in either direction
        )

        if not strong_reversal and not stale_trade:
            return
        
        # Log why we're exiting
        exit_reason = 'REVERSAL_SIGNAL' if strong_reversal else 'STALE_SIGNAL'
        bot_logger.info(
            f"⚠️ {exit_reason} exit triggered: {pair} | "
            f"age={age_minutes:.0f}m | P/L=${pnl:+.2f} | "
            f"signal={signal} conf={confidence:.1%} | "
            f"reversal={strong_reversal} stale={stale_trade}"
        )

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

            pending_signals = getattr(self, '_pending_trade_signals', {})
            model_signals = pending_signals.pop(pair, {})
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
            entry_price = float(position.get('open_price', 0) or position.get('price_open', 0) or 0)
            current_price = float(position.get('current_price', 0) or position.get('price_current', 0) or 0)
            pnl = float(position.get('profit', 0) or 0)
            closed = self.broker.close_position(pair=pair, volume=volume, ticket=ticket)
            if closed:
                bot_logger.info(
                    f"🔄 Active exit: {pair} ticket={ticket} ({exit_reason}) P/L=${pnl:+.2f}"
                )

                # Record with adaptive learner
                pending_signals = getattr(self, '_pending_trade_signals', {})
                model_signals = pending_signals.pop(ticket, None) or pending_signals.pop(pair, {})
                self.ensemble.record_trade_result({
                    'pair': pair,
                    'signal': position_type,
                    'profit_loss': pnl,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'exit_type': exit_reason,
                    'model_signals': model_signals,
                    'regime': getattr(self, '_last_regime', {}).get(pair, 'unknown'),
                })

                # Record with ML Trade Scorer and RL Agent
                self._record_ml_outcome(pair, pnl, ticket=ticket)
                self._record_rl_outcome(pair, pnl, exit_reason)

                # Remove from known tickets so _detect_closed_trades doesn't double-record
                self._known_tickets.pop(ticket, None)
    
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
        
        When trades are closed and slots become free, triggers immediate scout
        for new setups.
        """
        if self.mode != 'live' or not self.broker:
            return

        # Prevent both 1m and 5m threads from processing the same closure
        if not self._closure_lock.acquire(blocking=False):
            return  # Other thread is already handling closures

        trades_closed = 0  # Track closures for immediate re-scan

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

            trades_closed = len(closed_tickets)  # Track for re-scan trigger

            # Snapshot balance BEFORE processing closures (for fallback P/L estimation)
            pre_balance = getattr(self.risk_manager, 'current_balance', None)

            # Query deal history to get P/L for closed trades
            history = self.broker.get_trade_history(hours=24)
            bot_logger.info(f"🔍 Deal history returned {len(history) if history else 0} deals")

            for ticket in closed_tickets:
                info = self._known_tickets.pop(ticket, {})
                pair = info.get('pair', 'UNKNOWN')
                trade_type = info.get('type', 'UNKNOWN')
                entry_price = info.get('entry_price', 0)

                bot_logger.info(f"🔍 Looking for closed ticket {ticket} ({pair} {trade_type})")
                
                # Find matching deal in history (match by position_id = ticket)
                deal = None
                for d in history:
                    pos_id = d.get('position_id', d.get('ticket', 0))
                    if pos_id == ticket:
                        deal = d
                        bot_logger.info(f"✅ Found deal for ticket {ticket}: profit={d.get('profit')}")
                        break

                # Progressive retry with backoff — MT5 history can lag a few seconds
                if not deal:
                    import time
                    retry_delays = [1.5, 2.5, 4.0]  # seconds
                    for attempt, delay in enumerate(retry_delays, 1):
                        bot_logger.warning(
                            f"⚠️ Deal for ticket {ticket} not in history — "
                            f"retry {attempt}/{len(retry_delays)} after {delay}s"
                        )
                        time.sleep(delay)
                        fresh_history = self.broker.get_trade_history(
                            hours=2, include_all=True
                        )
                        for d in (fresh_history or []):
                            pos_id = d.get('position_id', d.get('ticket', 0))
                            if pos_id == ticket:
                                deal = d
                                bot_logger.info(
                                    f"✅ Found deal on retry {attempt}: "
                                    f"profit={d.get('profit')} magic={d.get('magic')}"
                                )
                                break
                        if deal:
                            break

                if deal:
                    profit = deal.get('profit', 0) + deal.get('swap', 0) + deal.get('commission', 0)
                    exit_price = deal.get('price', 0)
                    is_win = profit > 0
                    exit_type = 'TAKE_PROFIT' if is_win else 'STOP_LOSS'
                    bot_logger.info(f"✅ Deal resolved: profit=${profit:.2f}")
                else:
                    # Fallback: estimate P/L from broker balance change
                    bot_logger.warning(
                        f"⚠️ Could not find deal for ticket {ticket} after retries — "
                        f"estimating P/L from balance"
                    )
                    acct = self.broker.get_account_info() if self.broker else None
                    post_balance = acct.get('balance', 0) if acct else 0
                    if pre_balance and post_balance and pre_balance > 0:
                        estimated_pnl = round(post_balance - pre_balance, 2)
                        # Sanity check: P/L should be reasonable for micro lot
                        if abs(estimated_pnl) <= 50:
                            profit = estimated_pnl
                            bot_logger.info(
                                f"📊 Balance-delta P/L estimate: ${profit:+.2f} "
                                f"(${pre_balance:.2f} → ${post_balance:.2f})"
                            )
                        else:
                            profit = 0.0
                            bot_logger.warning(
                                f"📊 Balance delta ${estimated_pnl:+.2f} too large — recording $0"
                            )
                    else:
                        profit = 0.0
                    exit_price = 0
                    is_win = profit > 0
                    exit_type = 'ESTIMATED_CLOSE'

                bot_logger.info(
                    f"📊 Closed trade detected: {pair} {trade_type} | "
                    f"P/L: ${profit:+.2f} | Exit: {exit_type}"
                )

                # Decrement open_trades counter (balance is synced from broker
                # by _sync_balance — do NOT double-count P/L here)
                self.risk_manager.open_trades = max(0, self.risk_manager.open_trades - 1)

                # Record with adaptive learner (for weight adaptation only)
                # Try ticket-keyed first (new), then fall back to pair-keyed (legacy)
                pending_signals = getattr(self, '_pending_trade_signals', {})
                model_signals = pending_signals.pop(ticket, None) or pending_signals.pop(pair, {})
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
                self._record_ml_outcome(pair, profit, ticket=ticket, trade_type=trade_type)

                # Record with RL Agent
                self._record_rl_outcome(pair, profit, exit_type)

                # Clean from trailing stop tracker too
                if hasattr(self, 'trailing'):
                    self.trailing._tracking.pop(ticket, None)

        except Exception as e:
            bot_logger.warning(f"Closed trade detection failed: {e}")
        finally:
            self._closure_lock.release()

            # ── Immediate Re-Scan: Scout for new setups when slots freed ──
            if trades_closed > 0:
                effective_cap, available_slots = self.risk_manager.get_trade_capacity()
                if available_slots > 0:
                    bot_logger.info(
                        f"🔍 {trades_closed} position(s) closed — "
                        f"{available_slots} slot(s) now free, scouting for new setups..."
                    )
                    # Schedule immediate re-scan in background thread to avoid blocking
                    self._trigger_immediate_scan()

    def _trigger_immediate_scan(self):
        """Trigger an immediate scan for new trading setups when slots become free.
        
        Runs the 5M analysis immediately in a separate thread to scout for
        new entries. Uses a cooldown to prevent spamming if multiple closures
        happen in quick succession.
        """
        import threading
        
        # Cooldown check: don't spam scans (min 30s between triggered scans)
        now = time.time()
        last_triggered = getattr(self, '_last_immediate_scan', 0)
        if now - last_triggered < 30:
            bot_logger.debug("Immediate scan skipped — cooldown active")
            return
        self._last_immediate_scan = now
        
        def _run_scan():
            try:
                # Small delay to let deal history populate
                time.sleep(2)
                bot_logger.info("🚀 IMMEDIATE SCAN: Looking for new setups...")
                self.analyze_and_trade('5m')  # Run 5M analysis for higher-quality setups
            except Exception as e:
                bot_logger.warning(f"Immediate scan failed: {e}")
        
        # Run in background thread to not block the current cycle
        scan_thread = threading.Thread(target=_run_scan, daemon=True)
        scan_thread.start()

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
