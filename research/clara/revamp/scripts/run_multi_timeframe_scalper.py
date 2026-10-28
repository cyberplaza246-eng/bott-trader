#!/usr/bin/env python3
"""
Multi-Timeframe Scalping Bot — 1M & 5M Strategy

Runs simultaneous scalping on 1-minute and 5-minute timeframes for GBPUSD/EURUSD.
Uses confluence checking: prioritizes trades where both timeframes agree.

Features:
  - 1M scalps: Tight 4-8 pips SL, 0.8-1.5R TP, max 5-8 min hold
  - 5M scalps: Standard 6-12 pips SL, 1-2R TP, max 15-20 min hold
  - Confluence filtering: Prefer trades where 1M & 5M timeframes align
  - Separate position management per timeframe
"""
import os
import sys
import time
import signal
from datetime import datetime, timezone
from apscheduler.schedulers.background import BackgroundScheduler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.broker.mt5_connector import MT5Connector
from src.core.multi_timeframe_scalper import MultiTimeframeScalpingAnalyzer, MultiTimeframeScalpingTrader
from src.risk.position_manager import RiskManager
from src.utils.logger import bot_logger
from config.strategy_config import TRADING_MODE, INITIAL_BALANCE
from config.scalping_config_1m_5m import MultiTimeframeScalpingConfig


class MultiTimeframeScalpingBot:
    """Bot running 1M and 5M scalping simultaneously"""
    
    SCALPING_PAIRS = ['GBP/USD', 'EUR/USD']
    
    # Update intervals
    M1_REFRESH_INTERVAL = 60    # Update every minute
    M5_REFRESH_INTERVAL = 300   # Update every 5 minutes
    
    def __init__(self, mode='paper'):
        """Initialize multi-timeframe bot.
        
        Args:
            mode: 'live' or 'paper' trading mode
        """
        self.mode = mode
        self.running = False
        self.scheduler = BackgroundScheduler()
        
        bot_logger.info(f"{'='*70}")
        bot_logger.info(f"🔪 MULTI-TIMEFRAME SCALPING BOT v2.0")
        bot_logger.info(f"{'='*70}")
        bot_logger.info(f"Mode: {mode.upper()}")
        bot_logger.info(f"Pairs: {', '.join(self.SCALPING_PAIRS)}")
        bot_logger.info(f"Timeframes: 1-minute & 5-minute")
        bot_logger.info(f"1M Update: Every {self.M1_REFRESH_INTERVAL}s")
        bot_logger.info(f"5M Update: Every {self.M5_REFRESH_INTERVAL}s")
        bot_logger.info(f"{'='*70}\n")
        
        try:
            self.broker = MT5Connector()
            if not self.broker.is_connected:
                bot_logger.error("Failed to connect to MT5 broker!")
                raise ConnectionError("MT5 connection failed")
        except Exception as e:
            bot_logger.error(f"Broker initialization failed: {e}")
            self.broker = None
        
        self.risk_manager = RiskManager(initial_balance=INITIAL_BALANCE)
        
        # Get profit mode from config (default: 'quick_wins')
        profit_mode = getattr(MultiTimeframeScalpingConfig, 'PROFIT_MODE', 'quick_wins')
        self.analyzer = MultiTimeframeScalpingAnalyzer(profit_mode=profit_mode)
        self.trader = MultiTimeframeScalpingTrader(broker=self.broker, risk_manager=self.risk_manager, profit_mode=profit_mode)
        
        bot_logger.info(f"Initial Balance: ${INITIAL_BALANCE:.2f}")
        bot_logger.info(f"Risk Mode: {mode}")
        bot_logger.info(f"Profit Mode: {profit_mode.upper()}")
        bot_logger.info("Bot initialized and ready.\n")
    
    def fetch_candles(self, pair, timeframe, count):
        """Fetch candles for a pair and timeframe.
        
        Args:
            pair: Currency pair
            timeframe: 'M1' or 'M5'
            count: Number of candles
            
        Returns:
            DataFrame or None
        """
        try:
            if self.broker is None:
                return None
            
            timeframe_code = timeframe
            df = self.broker.fetch_historical_data(pair, timeframe_code, count)
            return df
        except Exception as e:
            bot_logger.error(f"Error fetching {timeframe} candles for {pair}: {e}")
            return None
    
    def process_1m_candles(self):
        """Process 1-minute candles (runs every minute)"""
        try:
            now = datetime.now(timezone.utc)
            minute = now.strftime('%Y-%m-%d %H:%M UTC')
            
            bot_logger.debug(f"\n⏰ Processing 1M candles at {minute}")
            
            # Fetch 1M data
            candles_gbp_1m = self.fetch_candles('GBP/USD', 'M1', 100)
            candles_eur_1m = self.fetch_candles('EUR/USD', 'M1', 100)
            
            if candles_gbp_1m is None or candles_eur_1m is None:
                bot_logger.warning("Failed to fetch 1M candles")
                return
            
            # Analyze on 1M timeframe (for insight, not direct trading necessarily)
            signal_gbp_1m = self.analyzer.get_signal(candles_gbp_1m, 'GBP/USD', 'M1')
            signal_eur_1m = self.analyzer.get_signal(candles_eur_1m, 'EUR/USD', 'M1')
            
            if signal_gbp_1m['signal'] in ['BUY', 'SELL']:
                bot_logger.debug(
                    f"1M Signal: GBP/USD {signal_gbp_1m['signal']} "
                    f"(confidence {signal_gbp_1m['confidence']:.2f})"
                )
            
            if signal_eur_1m['signal'] in ['BUY', 'SELL']:
                bot_logger.debug(
                    f"1M Signal: EUR/USD {signal_eur_1m['signal']} "
                    f"(confidence {signal_eur_1m['confidence']:.2f})"
                )
        
        except Exception as e:
            bot_logger.error(f"Error processing 1M candles: {e}", exc_info=True)
    
    def process_5m_candles(self):
        """Process 5-minute candles (runs every 5 minutes)"""
        try:
            now = datetime.now(timezone.utc)
            time_str = now.strftime('%Y-%m-%d %H:%M UTC')
            
            bot_logger.debug(f"\n⏰ Processing 5M candles at {time_str}")
            
            # Fetch both 1M and 5M data
            gbp_1m = self.fetch_candles('GBP/USD', 'M1', 100)
            eur_1m = self.fetch_candles('EUR/USD', 'M1', 100)
            gbp_5m = self.fetch_candles('GBP/USD', 'M5', 200)
            eur_5m = self.fetch_candles('EUR/USD', 'M5', 200)
            
            if any(df is None for df in [gbp_1m, eur_1m, gbp_5m, eur_5m]):
                bot_logger.warning("Failed to fetch candles, skipping this cycle")
                return
            
            # Process multi-timeframe analysis
            results = self.trader.process_candles_multi_tf(
                gbp_1m, eur_1m, gbp_5m, eur_5m
            )
            
            # Log results
            gbp_conf = results['gbp_analysis']['confluence']
            eur_conf = results['eur_analysis']['confluence']
            
            if gbp_conf['both_buy'] or gbp_conf['both_sell']:
                bot_logger.info(
                    f"🎯 GBP/USD Confluence: Both 1M & 5M {gbp_conf['both_buy'] and 'BUY' or 'SELL'}"
                )
            elif gbp_conf['divergent']:
                bot_logger.debug(
                    f"⚠️ GBP/USD Divergence: 1M {results['gbp_analysis']['signal_1m']['signal']} "
                    f"vs 5M {results['gbp_analysis']['signal_5m']['signal']}"
                )
            
            if eur_conf['both_buy'] or eur_conf['both_sell']:
                bot_logger.info(
                    f"🎯 EUR/USD Confluence: Both 1M & 5M {eur_conf['both_buy'] and 'BUY' or 'SELL'}"
                )
            elif eur_conf['divergent']:
                bot_logger.debug(
                    f"⚠️ EUR/USD Divergence: 1M {results['eur_analysis']['signal_1m']['signal']} "
                    f"vs 5M {results['eur_analysis']['signal_5m']['signal']}"
                )
            
            # Show summary
            summary = self.trader.get_summary()
            if summary['active_total'] > 0:
                bot_logger.info(
                    f"📊 Active Trades: {summary['active_1m']} (1M) + "
                    f"{summary['active_5m']} (5M) = {summary['active_total']} total"
                )
        
        except Exception as e:
            bot_logger.error(f"Error processing 5M candles: {e}", exc_info=True)
    
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
        """Start the multi-timeframe bot"""
        if self.running:
            bot_logger.warning("Bot is already running!")
            return
        
        self.running = True
        
        # Schedule 1M processing
        self.scheduler.add_job(
            self.process_1m_candles,
            'interval',
            seconds=self.M1_REFRESH_INTERVAL,
            id='process_1m',
            name='Process 1-minute candles',
        )
        
        # Schedule 5M processing
        self.scheduler.add_job(
            self.process_5m_candles,
            'interval',
            seconds=self.M5_REFRESH_INTERVAL,
            id='process_5m',
            name='Process 5-minute candles',
        )
        
        # Schedule balance updates
        self.scheduler.add_job(
            self.balance_update_scheduled,
            'interval',
            seconds=60,
            id='balance_update',
            name='Update account balance',
        )
        
        self.scheduler.start()
        
        bot_logger.info("🚀 Multi-Timeframe Scalping Bot STARTED")
        bot_logger.info(f"Next 1M update: in {self.M1_REFRESH_INTERVAL}s")
        bot_logger.info(f"Next 5M update: in {self.M5_REFRESH_INTERVAL}s")
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            bot_logger.info("\n⏹️ Shutdown signal received...")
            self.stop()
    
    def stop(self):
        """Stop the bot cleanly"""
        if not self.running:
            return
        
        self.running = False
        
        bot_logger.info("Shutting down multi-timeframe scalping bot...")
        
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)
        
        bot_logger.info("✅ Multi-Timeframe Scalping Bot STOPPED")
    
    def signal_handler(self, signum, frame):
        """Handle SIGINT for graceful shutdown"""
        self.stop()
        sys.exit(0)


def main():
    """Entry point"""
    mode = os.getenv('TRADING_MODE', 'paper').lower()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    if mode not in ['live', 'paper']:
        print(f"Invalid mode '{mode}'. Use 'live' or 'paper'.")
        sys.exit(1)
    
    bot = MultiTimeframeScalpingBot(mode=mode)
    
    signal.signal(signal.SIGINT, bot.signal_handler)
    signal.signal(signal.SIGTERM, bot.signal_handler)
    
    bot.start()


if __name__ == '__main__':
    main()
