#!/usr/bin/env python
"""
Quick-start live trading with relay mode forced ON.
Usage: python start_live.py
"""
import os
import sys

# Force environment
os.environ['TRADING_MODE'] = 'live'

# Auto-detect relay URL
if not os.getenv('MT5_RELAY_URL'):
    os.environ['MT5_RELAY_URL'] = 'http://127.0.0.1:5555'

print(f"🔧 TRADING_MODE = {os.environ['TRADING_MODE']}")
print(f"🔧 MT5_RELAY_URL = {os.environ['MT5_RELAY_URL']}")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Monkey-patch MT5Connector to always use relay when URL is set
import src.broker.mt5_connector as mt5mod

_OrigInit = mt5mod.MT5Connector.__init__

def _patched_init(self):
    _OrigInit(self)
    relay_url = os.getenv('MT5_RELAY_URL', '').rstrip('/')
    if relay_url and not self.connected:
        # Force relay mode
        self.relay_mode = True
        self.simulation_mode = False
        mt5mod.MT5_RELAY_URL = relay_url
        mt5mod.RELAY_AVAILABLE = True
        print(f"🔌 Forcing relay mode -> {relay_url}")
        self.initialize()

mt5mod.MT5Connector.__init__ = _patched_init

# Now start the bot
from src.bot import TradingBot
from src.utils.logger import bot_logger

def main():
    bot_logger.info("=" * 70)
    bot_logger.info("🚀 AI-Powered Forex Trading Bot Starting (start_live.py)...")
    bot_logger.info("=" * 70)

    newsapi_key = os.getenv('NEWSAPI_KEY')
    bot = TradingBot(newsapi_key=newsapi_key, enable_dashboard=True)

    try:
        bot.start()
    except KeyboardInterrupt:
        bot_logger.info("\nShutting down...")
        bot.stop()
    except Exception as e:
        bot_logger.error(f"Fatal error: {str(e)}", exc_info=True)
        bot.stop()

if __name__ == '__main__':
    main()
