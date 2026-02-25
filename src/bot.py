"""
Main Auto-Trading Bot Loop
Runs continuously to monitor, analyze, and execute trades.
Now with: 7-model ensemble, adaptive learning, dashboard, S/R-aware exits
"""
import os
import time
import threading
from datetime import datetime, time as dt_time
from apscheduler.schedulers.background import BackgroundScheduler

from src.broker.mt5_connector import MT5Connector
from src.core.ensemble_trader import EnsembleTrader
from src.risk.position_manager import RiskManager
from src.core.paper_trading import PaperTradingManager
from src.utils.logger import bot_logger, TradeLogger
from config.strategy_config import (
    TRADING_MODE,
    PAIRS,
    TIMEFRAMES,
    AUTOTRADING_ENABLED,
    INITIAL_BALANCE
)


class TradingBot:
    """Main trading bot orchestrator"""
    
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
        
        if self.mode == 'paper':
            self.paper_trader = PaperTradingManager(initial_balance=INITIAL_BALANCE)
        else:
            self.paper_trader = None
        
        self.last_signal_time = {}  # Track last signal per pair
        self.signal_cooldown_minutes = 15  # Don't trade same pair more than every 15 mins
        
        bot_logger.info(f"✅ Trading Bot initialized in {self.mode.upper()} mode (8-model ensemble)")
    
    def should_skip_signal(self, pair):
        """Prevent over-trading the same pair"""
        if pair not in self.last_signal_time:
            return False
        
        elapsed = (datetime.now() - self.last_signal_time[pair]).total_seconds() / 60
        return elapsed < self.signal_cooldown_minutes
    
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
        
        for pair in PAIRS:
            try:
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
                if self.ensemble.should_trade(signal_result):
                    if not self.risk_manager.can_trade():
                        bot_logger.warning(f"Risk limits prevent trading {pair}")
                        continue
                    
                    self._execute_trade(pair, signal_result, df)
                    self.last_signal_time[pair] = datetime.now()
                
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
        stop_loss = self.risk_manager.calculate_stop_loss(entry_price, atr, trade_type)
        take_profit = self.risk_manager.calculate_take_profit(entry_price, stop_loss, trade_type)
        
        position_size = self.risk_manager.calculate_position_size(entry_price, stop_loss, pair=pair)
        
        if not position_size:
            bot_logger.error(f"Failed to calculate position size for {pair}")
            return
        
        lot_size = position_size['lot_size']
        
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
        """Check and close positions if stop-loss or take-profit hit"""
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
    
    def _sync_balance(self):
        """Sync balance from broker or paper trader into the risk manager."""
        try:
            if self.mode == 'paper' and self.paper_trader:
                self.risk_manager.sync_balance(self.paper_trader.current_balance)
            elif self.broker:
                balance = self.broker.get_balance()
                if balance:
                    self.risk_manager.sync_balance(balance)
        except Exception as e:
            bot_logger.warning(f"Balance sync failed: {e}")

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
