#!/usr/bin/env python
"""
Quick-start live trading with relay mode forced ON.
Usage: python start_live.py

Automatically starts the relay server as a subprocess, then starts the bot.
Single terminal needed — no manual relay setup required.
"""
import os
import sys
import time
import subprocess
import requests

# Force environment BEFORE any imports
os.environ['TRADING_MODE'] = 'live'
if not os.getenv('MT5_RELAY_URL'):
    os.environ['MT5_RELAY_URL'] = 'http://127.0.0.1:5555'
os.environ.setdefault('ENSEMBLE_CONFIDENCE_THRESHOLD', '0.12')
os.environ.setdefault('HIGH_CERTAINTY_THRESHOLD', '0.20')

RELAY_URL = os.environ['MT5_RELAY_URL'].rstrip('/')
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

print(f"🔧 TRADING_MODE = live")
print(f"🔧 MT5_RELAY_URL = {RELAY_URL}")
print(f"🔧 QUICK_WINS mode — aggressive thresholds active")

# ── Auto-start relay if not already running ──────────────────────────
def is_relay_running():
    try:
        r = requests.get(f"{RELAY_URL}/ping", timeout=3)
        return r.json().get('mt5_connected', False)
    except:
        return False

relay_proc = None
if not is_relay_running():
    print(f"\n🔌 Relay not running — starting it automatically...")
    relay_script = os.path.join(PROJECT_ROOT, 'mt5_relay', 'relay_server.py')
    relay_proc = subprocess.Popen(
        [sys.executable, relay_script],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0),
    )
    # Wait for relay to start
    for i in range(15):
        time.sleep(1)
        if is_relay_running():
            print(f"✅ Relay started (PID {relay_proc.pid})")
            break
        print(f"   Waiting for relay... ({i+1}s)")
    else:
        # Read output to show any errors
        try:
            output = relay_proc.stdout.read(2000).decode('utf-8', errors='replace')
            print(f"❌ Relay failed to start. Output:\n{output}")
        except:
            print(f"❌ Relay failed to start after 15 seconds")
        sys.exit(1)
else:
    print(f"\n✅ Relay already running at {RELAY_URL}")

# Test relay connection
print(f"🔌 Testing relay...")
try:
    r = requests.get(f"{RELAY_URL}/ping", timeout=5)
    data = r.json()
    if data.get('mt5_connected'):
        acct = requests.get(f"{RELAY_URL}/account", timeout=5).json()
        print(f"✅ MT5 Connected | Account: {acct.get('login')} | Balance: {acct.get('balance')} | Leverage: {acct.get('leverage')}:1")
    else:
        print(f"❌ Relay running but MT5 not connected: {data}")
        sys.exit(1)
except Exception as e:
    print(f"❌ Cannot reach relay: {e}")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, PROJECT_ROOT)

# Import and replace MT5Connector with relay-only version
import src.broker.mt5_connector as mt5mod
from src.utils.logger import bot_logger, trades_logger, error_logger

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
                except:
                    body = r.text
                error_logger.error(f"Relay POST {path} failed ({r.status_code}): {body}")
                return None
            return r.json()
        except Exception as e:
            error_logger.error(f"Relay POST {path} failed: {e}")
            return None

# Replace the class
mt5mod.MT5Connector = RelayMT5Connector

# Start the bot
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
    finally:
        if relay_proc:
            print("Stopping relay server...")
            relay_proc.terminate()

if __name__ == '__main__':
    main()
