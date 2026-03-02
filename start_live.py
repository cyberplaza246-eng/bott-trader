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
import threading
import subprocess
import requests

# Force environment BEFORE any imports
os.environ['TRADING_MODE'] = 'live'
if not os.getenv('MT5_RELAY_URL'):
    os.environ['MT5_RELAY_URL'] = 'http://127.0.0.1:5555'
os.environ.setdefault('ENSEMBLE_CONFIDENCE_THRESHOLD', '0.12')
os.environ.setdefault('HIGH_CERTAINTY_THRESHOLD', '0.20')
# Match the relay server's default token
os.environ.setdefault('MT5_RELAY_TOKEN', 'change-me-to-a-secret')

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
RELAY_TOKEN = os.environ.get('MT5_RELAY_TOKEN', 'change-me-to-a-secret')
AUTH_HEADERS = {"Authorization": f"Bearer {RELAY_TOKEN}"}
print(f"🔌 Testing relay...")
try:
    r = requests.get(f"{RELAY_URL}/ping", timeout=5)
    data = r.json()
    if data.get('mt5_connected'):
        acct = requests.get(f"{RELAY_URL}/account", headers=AUTH_HEADERS, timeout=5).json()
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
    """MT5Connector that ALWAYS uses relay mode with circuit-breaker protection."""

    # Circuit-breaker states
    CB_CLOSED = 'closed'       # Normal operation
    CB_OPEN = 'open'           # Relay down — skip calls, return None instantly
    CB_HALF_OPEN = 'half_open' # Probing — allow one test call

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

        # ── Circuit-breaker state ─────────────────────────────────
        self._consecutive_failures = 0
        self._max_failures_before_reconnect = 3
        self._cb_state = self.CB_CLOSED
        self._cb_lock = threading.Lock()
        self._cb_cooldown = 60          # seconds before retrying after OPEN
        self._cb_max_cooldown = 300     # max backoff cap
        self._cb_last_fail_time = 0.0   # monotonic timestamp of last failed reconnect
        self._cb_backoff_multiplier = 1 # doubles on each consecutive failure cycle
        self._cb_suppressed_calls = 0   # count of calls short-circuited while open

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

    # ── Circuit-breaker helpers ───────────────────────────────────

    def _cb_current_cooldown(self):
        """Cooldown with exponential back-off, capped."""
        return min(self._cb_cooldown * self._cb_backoff_multiplier, self._cb_max_cooldown)

    def _cb_should_skip(self):
        """Return True if the circuit is open and cooldown hasn't elapsed."""
        if self._cb_state == self.CB_CLOSED:
            return False
        if self._cb_state == self.CB_OPEN:
            elapsed = time.monotonic() - self._cb_last_fail_time
            cooldown = self._cb_current_cooldown()
            if elapsed >= cooldown:
                # Transition to half-open: allow one probe call
                self._cb_state = self.CB_HALF_OPEN
                bot_logger.info(
                    f"🔌 Circuit half-open — probing relay after {int(elapsed)}s cooldown"
                )
                return False
            # Still in cooldown — suppress silently
            self._cb_suppressed_calls += 1
            if self._cb_suppressed_calls % 50 == 1:
                remaining = int(cooldown - elapsed)
                bot_logger.warning(
                    f"⏳ Relay circuit OPEN — skipping calls "
                    f"({self._cb_suppressed_calls} suppressed, ~{remaining}s until retry)"
                )
            return True
        # HALF_OPEN — let the call through
        return False

    def _cb_record_success(self):
        """Reset circuit to closed on a successful relay call."""
        was_disconnected = not self.connected or self._cb_state != self.CB_CLOSED
        self._cb_state = self.CB_CLOSED
        self._consecutive_failures = 0
        self._cb_backoff_multiplier = 1
        self._cb_suppressed_calls = 0
        self.connected = True  # Always restore connected flag
        if was_disconnected:
            bot_logger.info("✅ Relay back online — circuit CLOSED, broker marked connected")

    def _cb_record_failure(self):
        """Track failure; open circuit if threshold reached."""
        self._consecutive_failures += 1
        if self._cb_state == self.CB_HALF_OPEN:
            # Probe failed — go straight back to open with increased backoff
            self._cb_backoff_multiplier = min(self._cb_backoff_multiplier * 2, 8)
            self._cb_state = self.CB_OPEN
            self._cb_last_fail_time = time.monotonic()
            cooldown = self._cb_current_cooldown()
            bot_logger.warning(
                f"⚡ Relay probe failed — circuit OPEN (next retry in {cooldown}s)"
            )
        elif self._consecutive_failures >= self._max_failures_before_reconnect:
            self._try_reconnect()

    # ── Reconnect (with lock to prevent concurrent attempts) ──────

    def _try_reconnect(self):
        """Attempt reconnect; only one thread may run this at a time."""
        if not self._cb_lock.acquire(blocking=False):
            # Another thread is already reconnecting — don't pile up
            return
        try:
            if self._cb_state == self.CB_OPEN:
                return  # Already open, cooldown handles retries
            bot_logger.warning("🔄 Attempting relay reconnect (3 quick probes)...")
            for attempt in range(1, 4):
                time.sleep(2)
                try:
                    r = requests.get(f"{RELAY_URL}/ping", timeout=5)
                    data = r.json()
                    if data.get('mt5_connected'):
                        self.connected = True
                        self._cb_record_success()
                        bot_logger.info(f"✅ Relay reconnected (attempt {attempt})")
                        return
                except Exception:
                    pass
            # All probes failed — open the circuit
            self._cb_state = self.CB_OPEN
            self._cb_last_fail_time = time.monotonic()
            self._cb_backoff_multiplier = min(self._cb_backoff_multiplier * 2, 8)
            cooldown = self._cb_current_cooldown()
            bot_logger.error(
                f"❌ Relay reconnect failed — circuit OPEN "
                f"(will retry in {cooldown}s)"
            )
        finally:
            self._cb_lock.release()

    # ── Relay HTTP wrappers ───────────────────────────────────────

    def _relay_get(self, path, params=None, timeout=10):
        if self._cb_should_skip():
            return None
        headers = {"Authorization": f"Bearer {RELAY_TOKEN}"}
        try:
            r = requests.get(f"{RELAY_URL}{path}", params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            self._cb_record_success()
            return r.json()
        except Exception as e:
            self._cb_record_failure()
            # Log at reduced frequency when circuit is open
            if self._consecutive_failures <= self._max_failures_before_reconnect:
                error_logger.error(f"Relay GET {path} failed ({self._consecutive_failures}x): {e}")
            return None

    def _relay_post(self, path, data, timeout=10):
        if self._cb_should_skip():
            return None
        headers = {"Authorization": f"Bearer {RELAY_TOKEN}"}
        try:
            r = requests.post(f"{RELAY_URL}{path}", json=data, headers=headers, timeout=timeout)
            if r.status_code != 200:
                try:
                    body = r.json()
                except Exception:
                    body = r.text
                error_logger.error(f"Relay POST {path} failed ({r.status_code}): {body}")
                return None
            self._cb_record_success()
            return r.json()
        except Exception as e:
            self._cb_record_failure()
            if self._consecutive_failures <= self._max_failures_before_reconnect:
                error_logger.error(f"Relay POST {path} failed ({self._consecutive_failures}x): {e}")
            return None

    def is_connected_or_recover(self):
        """Check if connected; if not, try a quick probe to recover.
        
        Called by bot.analyze_and_trade instead of a hard self.connected check.
        This prevents the bot from being permanently stuck after a transient outage.
        """
        if self.connected and self._cb_state == self.CB_CLOSED:
            return True
        # Circuit is open or half-open — check if cooldown has elapsed
        if self._cb_state == self.CB_OPEN:
            elapsed = time.monotonic() - self._cb_last_fail_time
            cooldown = self._cb_current_cooldown()
            if elapsed < cooldown:
                return False  # Still in cooldown, skip gracefully
            # Cooldown elapsed — try a probe
            self._cb_state = self.CB_HALF_OPEN
        # Attempt a lightweight probe
        try:
            r = requests.get(f"{RELAY_URL}/ping", timeout=5)
            data = r.json()
            if data.get('mt5_connected'):
                self._cb_record_success()
                bot_logger.info("✅ Relay probe succeeded — resuming trading")
                return True
        except Exception:
            pass
        # Probe failed
        self._cb_state = self.CB_OPEN
        self._cb_last_fail_time = time.monotonic()
        self._cb_backoff_multiplier = min(self._cb_backoff_multiplier * 2, 8)
        return False

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

    # Reset adaptive learner stats so old sim data doesn't block trades
    if hasattr(bot, 'ensemble') and hasattr(bot.ensemble, 'learner'):
        learner = bot.ensemble.learner
        learner.pair_stats.clear()
        learner.session_pair_stats.clear()
        learner.hourly_stats.clear()
        learner.trade_history.clear()
        learner.consecutive_losses = 0
        learner.consecutive_wins = 0
        learner.in_drawdown_protection = False
        learner.confidence_threshold = 0.12
        bot_logger.info("🔄 Adaptive learner RESET — no skip rules from old data")

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
