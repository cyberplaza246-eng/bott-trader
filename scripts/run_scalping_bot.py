#!/usr/bin/env python3
"""
5-Minute Scalping Bot — Standalone Entry Point

Runs continuous scalping on GBPUSD and EURUSD with:
  - RSI(9), EMA 20/50/200 indicators
  - Trend-aligned pullback entries
  - Micro risk management (6-12 pips SL, 1-2R TP)
  - Automatic trade closure on time limits

Usage:
    python scripts/run_scalping_bot.py
"""
import os
import sys
import time
import signal
from datetime import datetime, timezone
from apscheduler.schedulers.background import BackgroundScheduler

# Add workspace root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.broker.mt5_connector import MT5Connector
from src.core.scalping_trader import ScalpingTrader
from src.risk.position_manager import RiskManager
from src.utils.logger import bot_logger
from config.strategy_config import TRADING_MODE, INITIAL_BALANCE, PAIRS


class ScalpingBot:
    """Main scalping bot orchestrator"""
    
    # Preferred pairs for scalping
    SCALPING_PAIRS = ['GBP/USD', 'EUR/USD']
    
    # 5-minute candle refresh (update every candle close)
    CANDLE_REFRESH_INTERVAL = 300  # seconds
    
    def __init__(self, mode='paper'):
        """Initialize scalping bot.
        
        Args:
            mode: 'live' or 'paper' trading mode
        """
        self.mode = mode
        self.running = False
        self.scheduler = BackgroundScheduler()
        
        bot_logger.info(f"{'='*60}")
        bot_logger.info(f"🔪 SCALPING BOT v1.0 — 5-Minute Pullback Strategy")
        bot_logger.info(f"{'='*60}")
        bot_logger.info(f"Mode: {mode.upper()}")
        bot_logger.info(f"Pairs: {', '.join(self.SCALPING_PAIRS)}")
        bot_logger.info(f"Update Interval: {self.CANDLE_REFRESH_INTERVAL}s (5-min candles)")
        bot_logger.info(f"{'='*60}\n")
        
        # Initialize components
        try:
            self.broker = MT5Connector()
            if not self.broker.is_connected:
                bot_logger.error("Failed to connect to MT5 broker!")
                raise ConnectionError("MT5 connection failed")
        except Exception as e:
            bot_logger.error(f"Broker initialization failed: {e}")
            self.broker = None
        
        self.risk_manager = RiskManager(initial_balance=INITIAL_BALANCE)
        self.scalper = ScalpingTrader(broker=self.broker, risk_manager=self.risk_manager)
        
        bot_logger.info(f"Initial Balance: ${INITIAL_BALANCE:.2f}")
        bot_logger.info(f"Risk Mode: {mode}")
        bot_logger.info("Bot initialized and ready.\n")
    
    def fetch_candles(self, pair, timeframe='M5', count=200):
        """Fetch recent candles for a pair.
        
        Args:
            pair: Currency pair
            timeframe: Timeframe code ('M5' for 5-minute)
            count: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            if self.broker is None:
                return None
            
            df = self.broker.fetch_historical_data(pair, timeframe, count)
            return df
        except Exception as e:
            bot_logger.error(f"Error fetching candles for {pair}: {e}")
            return None
    
    def process_candles(self):
        """Process 5-minute candles and generate scalping signals."""
        try:
            bot_logger.debug(f"\n⏰ Processing candles at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            
            # Fetch data for both pairs
            candles_gbp = self.fetch_candles('GBP/USD')
            candles_eur = self.fetch_candles('EUR/USD')
            
            if candles_gbp is None or candles_eur is None:
                bot_logger.warning("Failed to fetch candle data, skipping this cycle")
                return
            
            # Process and generate trades
            new_trades = self.scalper.process_candle(candles_gbp, candles_eur)
            
            if new_trades:
                bot_logger.info(f"✅ {len(new_trades)} new scalp trade(s) opened")
                for trade in new_trades:
                    bot_logger.info(
                        f"   {trade['pair']} {trade['direction']} @ "
                        f"{trade['entry']:.5f} | SL: {trade['sl']:.5f} | TP: {trade['tp']:.5f}"
                    )
            
            # Log current status
            summary = self.scalper.get_summary()
            if summary['active_scalp_trades'] > 0:
                bot_logger.info(f"📊 Active Scalp Trades: {summary['active_scalp_trades']}")
                for trade in summary['trades']:
                    bot_logger.info(
                        f"   Ticket {trade['ticket']}: {trade['pair']} {trade['direction']} "
                        f"({trade['hold_minutes']:.1f}min, conf {trade['confidence']:.2f})"
                    )
        
        except Exception as e:
            bot_logger.error(f"Error processing candles: {e}", exc_info=True)
    
    def balance_update_scheduled(self):
        """Periodic balance update (every minute)"""
        try:
            if self.broker:
                balance = self.broker.get_balance()
                if balance:
                    self.risk_manager.update_balance(balance)
                    bot_logger.debug(f"💰 Balance: ${balance:.2f}")
        except Exception as e:
            bot_logger.debug(f"Error updating balance: {e}")
    
    def start(self):
        """Start the scalping bot"""
        if self.running:
            bot_logger.warning("Bot is already running!")
            return
        
        self.running = True
        
        # Schedule candle processing every 5 minutes
        self.scheduler.add_job(
            self.process_candles,
            'interval',
            seconds=self.CANDLE_REFRESH_INTERVAL,
            id='process_candles',
            name='Process 5-min candles for scalping signals',
        )
        
        # Schedule balance updates every minute
        self.scheduler.add_job(
            self.balance_update_scheduled,
            'interval',
            seconds=60,
            id='balance_update',
            name='Update account balance',
        )
        
        self.scheduler.start()
        
        bot_logger.info("🚀 Scalping Bot STARTED")
        bot_logger.info(f"Next candle processing: in {self.CANDLE_REFRESH_INTERVAL}s")
        
        # Keep the main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            bot_logger.info("\n⏹️  Shutdown signal received...")
            self.stop()
    
    def stop(self):
        """Stop the scalping bot cleanly"""
        if not self.running:
            return
        
        self.running = False
        
        bot_logger.info("Shutting down scalping bot...")
        
        # Stop scheduler
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)
        
        # Close any open positions (optional for paper trading)
        if self.mode == 'paper':
            bot_logger.info("Closing all open positions...")
            try:
                if self.broker:
                    positions = self.broker.get_open_positions()
                    for pos in positions or []:
                        self.broker.close_position(pos['ticket'])
            except Exception as e:
                bot_logger.error(f"Error closing positions: {e}")
        
        bot_logger.info("✅ Scalping Bot STOPPED")
    
    def signal_handler(self, signum, frame):
        """Handle SIGINT for graceful shutdown"""
        self.stop()
        sys.exit(0)


def main():
    """Entry point"""
    # Determine mode from environment or argument
    mode = os.getenv('TRADING_MODE', 'paper').lower()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    if mode not in ['live', 'paper']:
        print(f"Invalid mode '{mode}'. Use 'live' or 'paper'.")
        sys.exit(1)
    
    # Create and start bot
    bot = ScalpingBot(mode=mode)
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, bot.signal_handler)
    signal.signal(signal.SIGTERM, bot.signal_handler)
    
    # Start the bot
    bot.start()


if __name__ == '__main__':
    main()
