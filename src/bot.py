"""
Main Auto-Trading Bot Loop
Runs continuously to monitor, analyze, and execute trades.
Now with: up to 8-model ensemble (LSTM optional), adaptive learning, dashboard, S/R-aware exits
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
from src.utils.logger import bot_logger, TradeLogger
from config.strategy_config import (
    TRADING_MODE,
    PAIRS,
    TIMEFRAMES,
    AUTOTRADING_ENABLED,
    INITIAL_BALANCE,
    HIGH_CERTAINTY_THRESHOLD,
    USDJPY_TUNING_ENABLED,
    USDJPY_MIN_CONFIDENCE,
    USDJPY_MIN_MODELS_AGREEMENT,
    USDJPY_MIN_ADX,
)


class TradingBot:
    """Main trading bot orchestrator"""
    
    # ── Trading Session Windows (UTC hours) ────────────────────────
    # EUR/USD & GBP/USD: London open → NY close  (08:00–22:00 UTC = 3AM–5PM ET)
    # ALL PAIRS:         Asian open → NY close    (00:00–22:00 UTC = 7PM–5PM ET)
    # ALL PAIRS BLOCKED: Daily rollover dead zone  (22:00–00:00 UTC = 5PM–7PM ET)
    # Spread filter + ADX filter protect against low-liquidity overnight entries
    PAIR_SESSIONS = {
        'EUR/USD': {'start': 0, 'end': 22},
        'GBP/USD': {'start': 0, 'end': 22},
        'USD/JPY': {'start': 0, 'end': 22},
    }
    DEFAULT_SESSION = {'start': 0, 'end': 22}

    # ── Correlation Groups ────────────────────────────────────────
    # Pairs that move together — block duplicate directional exposure
    CORRELATED_PAIRS = {
        'EUR/USD': 'GBP/USD',
        'GBP/USD': 'EUR/USD',
    }

    # ── Spread Limits (max allowed spread in price units) ─────────
    MAX_SPREAD = {
        'EUR/USD': 0.00030,   # 3 pips
        'GBP/USD': 0.00035,   # 3.5 pips
        'USD/JPY': 0.030,     # 3 pips (JPY)
    }
    DEFAULT_MAX_SPREAD = 0.00030

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
        self.risk_manager = RiskManager(initial_balance=INITIAL_BALANCE)
        self.trailing = TrailingStopManager(breakeven_r=1.0, trail_atr_mult=1.5)
        
        # Determine active model count
        model_count = 8 if self.ensemble.lstm_available else 7
        
        if self.mode == 'paper':
            self.paper_trader = PaperTradingManager(initial_balance=INITIAL_BALANCE)
        else:
            self.paper_trader = None
        
        self.last_signal_time = {}  # Track last signal per pair
        self.signal_cooldown_minutes = int(os.getenv('SIGNAL_COOLDOWN_MINUTES', '4'))
        self.max_trade_hold_minutes = int(os.getenv('MAX_TRADE_HOLD_MINUTES', '90'))
        self.reversal_exit_confidence = float(os.getenv('REVERSAL_EXIT_CONFIDENCE', '0.55'))
        self.reversal_exit_min_agreement = int(os.getenv('REVERSAL_EXIT_MIN_AGREEMENT', '2'))
        self.enable_correlation_guard = os.getenv('ENABLE_CORRELATION_GUARD', 'false').lower() == 'true'
        self.usdjpy_tuning_enabled = USDJPY_TUNING_ENABLED
        self.usdjpy_min_confidence = USDJPY_MIN_CONFIDENCE
        self.usdjpy_min_models_agreement = USDJPY_MIN_MODELS_AGREEMENT
        self.usdjpy_min_adx = USDJPY_MIN_ADX

        # Live closure detection: track open tickets so we notice when MT5 closes one
        self._known_tickets = {}   # {ticket: {pair, type, entry_price, open_time}}
        self._processed_closures = set()  # position_ids already recorded
        
        # Backfill past closed trades from MT5 history into adaptive learner
        self._backfill_history()
        
        bot_logger.info(f"✅ Trading Bot initialized in {self.mode.upper()} mode ({model_count}-model ensemble)")

    # ── Session Filter ────────────────────────────────────────────

    def is_pair_in_session(self, pair):
        """Check if the current UTC hour falls within this pair's trading window."""
        current_hour = datetime.now(timezone.utc).hour
        session = self.PAIR_SESSIONS.get(pair, self.DEFAULT_SESSION)
        s, e = session['start'], session['end']
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

    def should_skip_signal(self, pair):
        """Prevent over-trading the same pair"""
        if pair not in self.last_signal_time:
            return False
        
        elapsed = (datetime.now() - self.last_signal_time[pair]).total_seconds() / 60
        return elapsed < self.signal_cooldown_minutes

    @staticmethod
    def _is_usdjpy(pair):
        return str(pair or '').replace('/', '').upper() == 'USDJPY'

    @staticmethod
    def _is_opposite(signal_a, signal_b):
        return (signal_a, signal_b) in {('BUY', 'SELL'), ('SELL', 'BUY')}

    def _passes_usdjpy_quality_filter(self, pair, signal_result):
        """Extra quality checks for USD/JPY to reduce false positives."""
        if not (self.usdjpy_tuning_enabled and self._is_usdjpy(pair)):
            return True

        models = signal_result.get('models', {})
        direction = signal_result.get('signal', 'SKIP')
        mtf_signal = models.get('multi_tf', {}).get('signal', 'HOLD')
        if self._is_opposite(direction, mtf_signal):
            bot_logger.info("USD/JPY filter: skipped due to opposite multi-timeframe signal")
            return False

        price_zone = signal_result.get('sr_levels', {}).get('price_zone', '')
        if direction == 'BUY' and price_zone == 'AT_RESISTANCE':
            bot_logger.info("USD/JPY filter: skipped BUY at resistance")
            return False
        if direction == 'SELL' and price_zone == 'AT_SUPPORT':
            bot_logger.info("USD/JPY filter: skipped SELL at support")
            return False

        enriched_df = signal_result.get('enriched_df')
        if enriched_df is not None and len(enriched_df) > 0 and 'adx' in enriched_df.columns:
            adx = float(enriched_df.iloc[-1].get('adx', 0) or 0)
            if adx < self.usdjpy_min_adx:
                bot_logger.info(
                    f"USD/JPY filter: ADX {adx:.1f} below minimum {self.usdjpy_min_adx:.1f}"
                )
                return False

        return True

    def _should_trade_pair(self, pair, signal_result):
        """Determine if a pair should be traded, including USD/JPY-specific override."""
        if self.ensemble.should_trade(signal_result):
            return True

        if not (self.usdjpy_tuning_enabled and self._is_usdjpy(pair)):
            return False

        signal = signal_result.get('signal', 'SKIP')
        confidence = signal_result.get('confidence', 0.0)
        agreement = signal_result.get('models_agreement', 0)
        if signal == 'SKIP':
            return False

        return (
            confidence >= self.usdjpy_min_confidence and
            agreement >= self.usdjpy_min_models_agreement
        )
    
    def analyze_and_trade(self):
        """
        Main trading loop:
        1. Fetch latest data
        2. Run ensemble analysis
        3. Execute trades if signal is strong
        """
        if not self.broker or not self.broker.connected:
            bot_logger.error("Broker not connected, skipping analysis")
            return
        
        bot_logger.info("=" * 60)
        bot_logger.info(f"🔍 Starting analysis cycle at {datetime.now().strftime('%H:%M:%S')}")
        
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

        for pair in PAIRS:
            try:
                # Skip if outside trading session for this pair
                if not self.is_pair_in_session(pair):
                    session = self.PAIR_SESSIONS.get(pair, self.DEFAULT_SESSION)
                    bot_logger.info(
                        f"💤 {pair} outside session (UTC {session['start']:02d}:00–{session['end']:02d}:00) — skipping"
                    )
                    continue

                # Skip if cooldown active
                if self.should_skip_signal(pair):
                    continue
                
                # Fetch latest candle data
                df = self.broker.get_candles(
                    pair,
                    timeframe_minutes=TIMEFRAMES['fast'],
                    num_candles=100
                )
                
                if df is None or len(df) < 50:
                    bot_logger.warning(f"Insufficient data for {pair}")
                    continue
                
                # Get ensemble signal
                signal_result = self.ensemble.get_trading_signal(df, pair)
                
                # Log detailed analysis
                bot_logger.info(f"\n{pair} Analysis:")
                bot_logger.info(f"  Signal: {signal_result['signal']}")
                bot_logger.info(f"  Confidence: {signal_result['confidence']:.1%}")
                bot_logger.info(f"  Models Agreement: {signal_result['models_agreement']}/{signal_result.get('total_models', 7)}")
                bot_logger.info(f"  Details: {signal_result['detailed_reason']}")

                # Active position management: exit stale/reversal trades to free slots
                self._manage_open_position(pair, signal_result)
                
                # Track for dashboard
                self.signal_history.append({
                    'pair': pair,
                    'signal': signal_result['signal'],
                    'confidence': signal_result['confidence'],
                    'agreement': signal_result['models_agreement'],
                    'time': datetime.now().strftime('%H:%M:%S'),
                })
                # Keep last 100
                if len(self.signal_history) > 100:
                    self.signal_history = self.signal_history[-100:]
                
                # Execute trade if signal is strong enough
                if self._should_trade_pair(pair, signal_result):
                    if not self._passes_usdjpy_quality_filter(pair, signal_result):
                        continue

                    # Block duplicate: don't open another trade if bot already has one on this pair
                    if self._has_bot_position(pair):
                        bot_logger.info(f"⏭️ {pair}: already have open bot position — skipping")
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

                    # Spread filter
                    if not self.is_spread_acceptable(pair):
                        continue

                    # Correlation guard (optional)
                    if self.enable_correlation_guard and self._has_correlated_position(pair, signal_result['signal']):
                        continue

                    self._execute_trade(pair, signal_result, df)
                    self.last_signal_time[pair] = datetime.now()
                    # Re-sync free margin after placing a trade so next pair
                    # sees updated margin availability
                    self._sync_balance()
                
                # Update open positions
                self._update_positions(pair)
                
            except Exception as e:
                bot_logger.error(f"Error analyzing {pair}: {str(e)}", exc_info=True)
        
        # Log risk status with tier info
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
    
    def _execute_trade(self, pair, signal_result, df):
        """Execute actual trade"""
        trade_type = signal_result['signal']
        # Use the enriched df (with indicators) from the ensemble
        enriched_df = signal_result.get('enriched_df', df)
        latest = enriched_df.iloc[-1]
        entry_price = latest['close']
        atr = latest.get('atr', entry_price * 0.001)  # Fallback: 0.1% of price
        
        # Calculate risk parameters
        stop_loss = self.risk_manager.calculate_stop_loss(entry_price, atr, trade_type, pair=pair)
        take_profit = self.risk_manager.calculate_take_profit(entry_price, stop_loss, trade_type, pair=pair)
        # S/R-based dynamic TP: use nearest S/R level within tier R:R limits
        sr_levels = signal_result.get('sr_levels', {})
        risk_distance = abs(entry_price - stop_loss)
        # Tier-based R:R cap — micro accounts need tighter TPs
        tier_name = self.risk_manager._current_tier_name or 'micro'
        if 'micro' in tier_name:
            max_rr = 3.0
        elif 'mini' in tier_name:
            max_rr = 4.0
        else:
            max_rr = 5.0
        if sr_levels and risk_distance > 0:
            if trade_type == 'BUY':
                resistances = sr_levels.get('resistance_levels', [])
                for level in sorted(resistances):
                    reward = level - entry_price
                    rr = reward / risk_distance
                    if 1.5 <= rr <= max_rr:
                        digits = 3 if 'JPY' in pair else 5
                        sr_tp = round(level, digits)
                        bot_logger.info(
                            f"🎯 S/R TP: {take_profit:.{digits}f} → {sr_tp:.{digits}f} "
                            f"(resistance level, R:R = {rr:.1f}, max {max_rr:.0f}:1)"
                        )
                        take_profit = sr_tp
                        break
            elif trade_type == 'SELL':
                supports = sr_levels.get('support_levels', [])
                for level in sorted(supports, reverse=True):
                    reward = entry_price - level
                    rr = reward / risk_distance
                    if 1.5 <= rr <= max_rr:
                        digits = 3 if 'JPY' in pair else 5
                        sr_tp = round(level, digits)
                        bot_logger.info(
                            f"🎯 S/R TP: {take_profit:.{digits}f} → {sr_tp:.{digits}f} "
                            f"(support level, R:R = {rr:.1f}, max {max_rr:.0f}:1)"
                        )
                        take_profit = sr_tp
                        break        
        position_size = self.risk_manager.calculate_position_size(entry_price, stop_loss, pair=pair)
        
        if not position_size:
            bot_logger.error(f"Failed to calculate position size for {pair}")
            return
        
        lot_size = position_size['lot_size']

        # Micro-account lot policy: always use at least 0.02
        tier_name = self.risk_manager._current_tier_name or 'micro'
        if 'micro' in tier_name:
            lot_size = max(0.02, lot_size)
        
        bot_logger.info(f"\n🎯 EXECUTING {trade_type} TRADE:")
        bot_logger.info(f"  Pair: {pair}")
        bot_logger.info(f"  Entry: {entry_price:.5f}")
        bot_logger.info(f"  Stop Loss: {stop_loss:.5f}")
        bot_logger.info(f"  Take Profit: {take_profit:.5f}")
        bot_logger.info(f"  Lot Size: {lot_size}")
        bot_logger.info(f"  Risk: ${position_size['risk_amount']:.2f}")
        
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
                # Register for trailing stop management (with TP & volume for partial close)
                self.trailing.register(
                    ticket=order_id,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    direction=trade_type,
                    atr=atr,
                    pair=pair,
                    take_profit=take_profit,
                    volume=lot_size,
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
            if not hasattr(self, '_pending_trade_signals'):
                self._pending_trade_signals = {}
            self._pending_trade_signals[pair] = signal_result.get('models', {})
    
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
                    })

    @staticmethod
    def _normalize_pair(pair):
        """Normalize pair format (EUR/USD and EURUSD -> EURUSD)."""
        return str(pair or '').replace('/', '').upper()

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
            })
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
        
        # Schedule analysis every 5 minutes
        self.scheduler.add_job(
            self.analyze_and_trade,
            'interval',
            minutes=5,
            id='trading_cycle',
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
        
        self.scheduler.start()
        
        # Run first analysis immediately
        bot_logger.info("Running initial analysis...")
        self.analyze_and_trade()
        
        bot_logger.info("Bot running in event-driven mode. Press Ctrl+C to stop.")
        
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

                # Record with adaptive learner
                self.risk_manager.on_trade_closed(profit)
                model_signals = getattr(self, '_pending_trade_signals', {}).pop(pair, {})
                self.ensemble.record_trade_result({
                    'pair': pair,
                    'signal': trade_type,
                    'profit_loss': profit,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_type': exit_type,
                    'model_signals': model_signals,
                })

                # Clean from trailing stop tracker too
                if hasattr(self, 'trailing'):
                    self.trailing._tracking.pop(ticket, None)

        except Exception as e:
            bot_logger.warning(f"Closed trade detection failed: {e}")

    def _backfill_history(self):
        """One-time backfill: load closed deals from MT5 history into the adaptive learner.
        
        Only imports deals that aren't already in the learner's trade_history,
        using position_id to deduplicate.
        """
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
        """Reset daily trading limits"""
        self._sync_balance()
        self.risk_manager.reset_daily_limits(self.risk_manager.current_balance)


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
