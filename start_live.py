#!/usr/bin/env python
"""
Quick-start live trading with relay mode forced ON.
Usage: python start_live.py

This script forces relay mode regardless of what mt5_connector.py does locally.
No git pull needed — it patches everything at runtime.
"""
import os
import sys
import requests

# Force environment BEFORE any imports
os.environ['TRADING_MODE'] = 'live'
if not os.getenv('MT5_RELAY_URL'):
    os.environ['MT5_RELAY_URL'] = 'http://127.0.0.1:5555'
os.environ.setdefault('ENSEMBLE_CONFIDENCE_THRESHOLD', '0.12')
os.environ.setdefault('HIGH_CERTAINTY_THRESHOLD', '0.20')

RELAY_URL = os.environ['MT5_RELAY_URL'].rstrip('/')

print(f"🔧 TRADING_MODE = live")
print(f"🔧 MT5_RELAY_URL = {RELAY_URL}")
print(f"🔧 QUICK_WINS mode — aggressive thresholds active")

# Test relay BEFORE starting bot
print(f"\n🔌 Testing relay at {RELAY_URL}...")
try:
    r = requests.get(f"{RELAY_URL}/ping", timeout=5)
    data = r.json()
    if data.get('mt5_connected'):
        print(f"✅ Relay OK — MT5 connected")
    else:
        print(f"❌ Relay responded but MT5 not connected: {data}")
        sys.exit(1)
except Exception as e:
    print(f"❌ Cannot reach relay at {RELAY_URL}: {e}")
    print(f"   Make sure relay_server.py is running in another terminal!")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and completely replace MT5Connector.__init__ to force relay
import src.broker.mt5_connector as mt5mod
from src.utils.logger import bot_logger, trades_logger, error_logger
from datetime import datetime

class RelayMT5Connector(mt5mod.MT5Connector):
    """MT5Connector that ALWAYS uses relay mode."""
    
    def __init__(self):
        self.account = None
        self.password = None
        self.server = None
        self.connected = False
        self.relay_mode = True
        self.relay_url = RELAY_URL
        self.simulation_mode = False
        self.sim_balance = 50.0
        self.sim_equity = 50.0
        self.sim_positions = []
        
        # Test and connect via relay
        result = self._relay_get("/ping")
        if result and result.get("mt5_connected"):
            self.connected = True
            acct = self._relay_get("/account")
            if acct and "balance" in acct:
                bot_logger.info(
                    f"✅ MT5 Relay Connected | Account: {acct.get('login')} | "
                    f"Balance: {acct['balance']} | Server: {acct.get('server')}"
                )
            else:
                bot_logger.info(f"✅ MT5 Relay Connected at {RELAY_URL}")
        else:
            raise Exception(f"Relay at {RELAY_URL} not reachable")
    
    def _relay_get(self, path, params=None, timeout=10):
        headers = {}
        token = os.getenv('MT5_RELAY_TOKEN', '')
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            r = requests.get(f"{RELAY_URL}{path}", params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            error_logger.error(f"Relay GET {path} failed: {e}")
            return None

    def _relay_post(self, path, data, timeout=10):
        headers = {}
        token = os.getenv('MT5_RELAY_TOKEN', '')
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            r = requests.post(f"{RELAY_URL}{path}", json=data, headers=headers, timeout=timeout)
            if r.status_code != 200:
                try:
                    body = r.json()
                except Exception:
                    body = r.text
                error_logger.error(f"Relay POST {path} failed ({r.status_code}): {body}")
                return None
            return r.json()
        except Exception as e:
            error_logger.error(f"Relay POST {path} failed: {e}")
            return None

# Replace the class entirely so TradingBot uses our relay version
mt5mod.MT5Connector = RelayMT5Connector

# Now start the bot
from src.bot import TradingBot

def main():
    bot_logger.info("=" * 70)
    bot_logger.info("🚀 AI-Powered Forex Trading Bot Starting (RELAY MODE)...")
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
