"""
MT5 Broker Connection and Order Management

On Windows: Uses real MetaTrader5 API
On Linux/Mac: Connects via HTTP relay to a Windows machine running relay_server.py
Fallback: Simulation mode for paper trading and backtesting
"""
import pandas as pd
import numpy as np
import requests
import os
import sys
import time
import subprocess
from datetime import datetime, timedelta
from src.utils.logger import bot_logger, error_logger, trades_logger
from config.strategy_config import MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER, PAIRS, TRADING_MODE

# Try to import MetaTrader5 (Windows only)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

# Check for relay URL (Linux → Windows bridge, or forced relay on Windows)
MT5_RELAY_URL = os.getenv('MT5_RELAY_URL', '').rstrip('/')
MT5_RELAY_TOKEN = os.getenv('MT5_RELAY_TOKEN', '')
RELAY_AVAILABLE = bool(MT5_RELAY_URL)
# Force relay mode when URL is set (even on Windows where MT5 is available)
FORCE_RELAY = RELAY_AVAILABLE and os.getenv('MT5_RELAY_URL')  # Explicitly set = use relay

if not MT5_AVAILABLE and not RELAY_AVAILABLE:
    bot_logger.warning("MetaTrader5 not available (Linux/Mac) and no MT5_RELAY_URL set. Using simulation mode.")


class MT5Connector:
    def __init__(self):
        self.account = MT5_ACCOUNT
        self.password = MT5_PASSWORD
        self.server = MT5_SERVER
        self.connected = False
        
        # Re-check env var at runtime (not module load time)
        self.relay_url = os.getenv('MT5_RELAY_URL', '').rstrip('/')
        relay_available = bool(self.relay_url)
        
        # Prefer relay when explicitly configured, even on Windows
        self.relay_mode = relay_available
        self.simulation_mode = TRADING_MODE in ('paper', 'backtest')
        
        # Track whether we fell back from relay so we can auto-reconnect
        self._relay_fallback = False
        self._original_relay_url = self.relay_url  # remember for reconnect
        self._last_reconnect_attempt = 0  # timestamp of last retry
        self._reconnect_interval = 30  # seconds between retries
        
        bot_logger.info(f"🔌 Broker mode: relay_url={self.relay_url}, relay_mode={self.relay_mode}, sim_mode={self.simulation_mode}")
        
        self.sim_balance = 50.0
        self.sim_equity = 50.0
        self.sim_positions = []
        self.initialize()

    # ── Relay helpers ─────────────────────────────────────────────────

    def _relay_get(self, path, params=None, timeout=10):
        """GET request to the relay server."""
        url = getattr(self, 'relay_url', MT5_RELAY_URL)
        headers = {"Authorization": f"Bearer {MT5_RELAY_TOKEN}"} if MT5_RELAY_TOKEN else {}
        try:
            r = requests.get(f"{url}{path}", params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            error_logger.error(f"Relay GET {path} failed: {e}")
            return None

    def _relay_post(self, path, data, timeout=10):
        """POST request to the relay server."""
        url = getattr(self, 'relay_url', MT5_RELAY_URL)
        headers = {"Authorization": f"Bearer {MT5_RELAY_TOKEN}"} if MT5_RELAY_TOKEN else {}
        try:
            r = requests.post(f"{url}{path}", json=data, headers=headers, timeout=timeout)
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

    def _attempt_relay_autostart(self):
        """Try to start relay server automatically when configured but unreachable."""
        if os.getenv('MT5_RELAY_AUTOSTART', 'true').lower() in ('0', 'false', 'no', 'off'):
            return False

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        relay_script = os.path.join(project_root, 'mt5_relay', 'relay_server.py')
        if not os.path.exists(relay_script):
            return False

        try:
            bot_logger.warning(f"🔌 Relay unreachable, attempting auto-start: {relay_script}")
            creationflags = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
            # Detach relay process and suppress inherited stdio noise.
            subprocess.Popen(
                [sys.executable, relay_script],
                cwd=project_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )

            # Give relay time to bind and initialize MT5 session.
            for _ in range(12):
                time.sleep(1)
                ping = self._relay_get('/ping', timeout=3)
                if ping and ping.get('mt5_connected'):
                    bot_logger.info("✅ Relay auto-started successfully")
                    return True
        except Exception as e:
            bot_logger.warning(f"Relay auto-start failed: {e}")

        return False
    
    def initialize(self):
        """Initialize MT5 connection, relay, or simulation mode"""
        if self.relay_mode and not self.simulation_mode:
            # Test relay connection
            result = self._relay_get("/ping")
            if not result:
                # Fail-safe: try auto-starting relay once before simulation fallback.
                if self._attempt_relay_autostart():
                    result = self._relay_get("/ping")
            if result and result.get("mt5_connected"):
                self.connected = True
                self._relay_fallback = False
                acct = self._relay_get("/account")
                if acct and "balance" in acct:
                    bot_logger.info(
                        f"✅ MT5 Relay Connected | Account: {acct.get('login')} | "
                        f"Balance: ${acct['balance']:.2f} | Server: {acct.get('server')}"
                    )
                else:
                    bot_logger.info(f"✅ MT5 Relay Connected at {self.relay_url}")
                return True
            else:
                bot_logger.warning(
                    f"MT5 Relay at {self.relay_url} not reachable — falling back to simulation "
                    f"(will auto-reconnect every {self._reconnect_interval}s)"
                )
                self._relay_fallback = True
                self.relay_mode = False
                self.simulation_mode = True

        if self.simulation_mode:
            self.connected = True
            bot_logger.info(
                f"✅ Simulation Mode Active | Balance: ${self.sim_balance:.2f} | "
                f"Mode: {TRADING_MODE}"
            )
            return True
        
        if not mt5.initialize(login=int(self.account), password=self.password, server=self.server):
            error_logger.error(f"Failed to initialize MT5: {mt5.last_error()}")
            raise Exception(f"MT5 initialization failed: {mt5.last_error()}")
        
        self.connected = True
        account_info = mt5.account_info()
        if account_info:
            bot_logger.info(
                f"✅ MT5 Connected | Account: {account_info.login} | "
                f"Balance: {account_info.balance} | Equity: {account_info.equity}"
            )
        return True
    
    def _try_relay_reconnect(self):
        """Periodically retry connecting to relay when in fallback simulation mode."""
        if not self._relay_fallback or not self._original_relay_url:
            return False
        
        import time as _time
        now = _time.time()
        if now - self._last_reconnect_attempt < self._reconnect_interval:
            return False
        self._last_reconnect_attempt = now
        
        try:
            headers = {"Authorization": f"Bearer {MT5_RELAY_TOKEN}"} if MT5_RELAY_TOKEN else {}
            r = requests.get(f"{self._original_relay_url}/ping", headers=headers, timeout=3)
            data = r.json()
            if data.get("mt5_connected"):
                # Relay is back! Switch out of simulation
                self.relay_url = self._original_relay_url
                self.relay_mode = True
                self.simulation_mode = False
                self._relay_fallback = False
                acct = self._relay_get("/account")
                bal = acct.get('balance', '?') if acct else '?'
                bot_logger.info(
                    f"🔄 MT5 Relay reconnected! | Balance: ${bal} | "
                    f"Switching from simulation to LIVE relay mode"
                )
                return True
        except Exception:
            pass
        return False

    def get_balance(self):
        """Get current account balance"""
        self._try_relay_reconnect()
        if self.relay_mode:
            r = self._relay_get("/balance")
            return r.get("balance") if r else None
        if self.simulation_mode:
            return self.sim_balance
        account_info = mt5.account_info()
        return account_info.balance if account_info else None

    def get_account_info(self):
        """Get full account info: balance, equity, leverage, free_margin, margin."""
        self._try_relay_reconnect()
        if self.relay_mode:
            r = self._relay_get("/account")
            if r:
                return {
                    'balance': r.get('balance', 0),
                    'equity': r.get('equity', 0),
                    'leverage': r.get('leverage', 100),
                    'margin_free': r.get('margin_free', 0),
                    'margin': r.get('margin', 0),
                    'profit': r.get('profit', 0),
                }
            return None
        if self.simulation_mode:
            return {
                'balance': self.sim_balance,
                'equity': self.sim_equity,
                'leverage': 100,
                'margin_free': self.sim_balance * 0.95,
                'margin': 0,
                'profit': 0,
            }
        account_info = mt5.account_info()
        if account_info:
            return {
                'balance': account_info.balance,
                'equity': account_info.equity,
                'leverage': account_info.leverage,
                'margin_free': account_info.margin_free,
                'margin': account_info.margin,
                'profit': account_info.profit,
            }
        return None
    
    def get_equity(self):
        """Get current account equity"""
        self._try_relay_reconnect()
        if self.relay_mode:
            r = self._relay_get("/equity")
            return r.get("equity") if r else None
        if self.simulation_mode:
            return self.sim_equity
        account_info = mt5.account_info()
        return account_info.equity if account_info else None
    
    def get_candles(self, pair, timeframe_minutes, num_candles=100):
        """
        Fetch OHLCV candle data
        
        In simulation mode: generates realistic synthetic data
        In live mode: fetches from MT5
        
        Automatically tries both 'EUR/USD' and 'EURUSD' symbol formats.
        """
        self._try_relay_reconnect()
        if self.relay_mode:
            r = self._relay_get("/candles", {
                "pair": pair, "timeframe": timeframe_minutes, "count": num_candles
            }, timeout=15)
            if r and "candles" in r:
                df = pd.DataFrame(r["candles"])
                df["datetime"] = pd.to_datetime(df["datetime"])
                return df
            error_logger.error(f"Relay candles failed for {pair}: {r}")
            return None

        if self.simulation_mode:
            return self._generate_simulated_candles(pair, timeframe_minutes, num_candles)
        
        timeframe_map = {
            1: mt5.TIMEFRAME_M1,
            5: mt5.TIMEFRAME_M5,
            15: mt5.TIMEFRAME_M15,
            60: mt5.TIMEFRAME_H1,
            240: mt5.TIMEFRAME_H4,
            1440: mt5.TIMEFRAME_D1,
        }
        
        tf = timeframe_map.get(timeframe_minutes)
        if not tf:
            error_logger.error(f"Unsupported timeframe: {timeframe_minutes}")
            return None
        
        # Try multiple symbol formats (brokers differ)
        symbol_variants = [pair, pair.replace('/', '')]
        
        for symbol in symbol_variants:
            try:
                # Ensure symbol is visible in Market Watch
                if not mt5.symbol_select(symbol, True):
                    continue
                
                candles = mt5.copy_rates_from_pos(symbol, tf, 0, num_candles)
                if candles is not None and len(candles) > 0:
                    df = pd.DataFrame(candles)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'real_volume']]
                    df.rename(columns={
                        'time': 'datetime',
                        'tick_volume': 'volume',
                        'real_volume': 'real_volume'
                    }, inplace=True)
                    return df
            except Exception:
                continue
        
        error_logger.error(f"Failed to fetch candles for {pair}: {mt5.last_error()}")
        return None
    
    def _generate_simulated_candles(self, pair, timeframe_minutes, num_candles):
        """Generate realistic simulated OHLCV candles for paper trading/backtesting"""
        base_prices = {
            'EURUSD': 1.0850, 'EUR/USD': 1.0850,
            'GBPUSD': 1.2650, 'GBP/USD': 1.2650,
            'USDJPY': 150.50, 'USD/JPY': 150.50,
            'AUDUSD': 0.6550, 'AUD/USD': 0.6550,
        }
        
        pair_clean = pair.replace('/', '')
        base_price = base_prices.get(pair, base_prices.get(pair_clean, 1.1000))
        
        is_jpy = 'JPY' in pair
        volatility = 0.0008 if is_jpy else 0.0006
        
        tf_scale = np.sqrt(timeframe_minutes / 1440)
        candle_vol = volatility * tf_scale
        
        now = datetime.now()
        timestamps = [now - timedelta(minutes=timeframe_minutes * (num_candles - i)) for i in range(num_candles)]
        
        np.random.seed(int(datetime.now().timestamp()) % 10000)
        
        prices = [base_price]
        for i in range(1, num_candles):
            mean_reversion = (base_price - prices[-1]) * 0.01
            change = np.random.normal(mean_reversion, base_price * candle_vol)
            prices.append(prices[-1] + change)
        
        data = []
        for i in range(num_candles):
            open_price = prices[i]
            intra_vol = base_price * candle_vol * 0.5
            high = open_price + abs(np.random.normal(0, intra_vol))
            low = open_price - abs(np.random.normal(0, intra_vol))
            close = open_price + np.random.normal(0, intra_vol * 0.7)
            high = max(high, open_price, close) + abs(np.random.normal(0, intra_vol * 0.1))
            low = min(low, open_price, close) - abs(np.random.normal(0, intra_vol * 0.1))
            volume = max(100, int(np.random.lognormal(8, 1)))
            
            data.append({
                'datetime': timestamps[i],
                'open': round(open_price, 5 if not is_jpy else 3),
                'high': round(high, 5 if not is_jpy else 3),
                'low': round(low, 5 if not is_jpy else 3),
                'close': round(close, 5 if not is_jpy else 3),
                'volume': volume,
                'real_volume': volume
            })
        
        return pd.DataFrame(data)
    
    def _resolve_symbol(self, pair):
        """Find the correct symbol name in MT5 (handles EUR/USD vs EURUSD)."""
        if self.simulation_mode:
            return pair
        for variant in [pair, pair.replace('/', '')]:
            info = mt5.symbol_info(variant)
            if info is not None:
                mt5.symbol_select(variant, True)
                return variant
        return pair  # fallback

    def get_latest_price(self, pair):
        """Get latest bid/ask price"""
        if self.relay_mode:
            r = self._relay_get("/price", {"pair": pair})
            if r and "bid" in r:
                return {
                    "bid": r["bid"],
                    "ask": r["ask"],
                    "last": r.get("last", r["bid"]),
                    "time": datetime.now(),
                }
            return None
        if self.simulation_mode:
            df = self.get_candles(pair, 1, 1)
            if df is not None and len(df) > 0:
                close = df.iloc[-1]['close']
                spread = 0.00015 if 'JPY' not in pair else 0.015
                return {
                    'bid': close,
                    'ask': close + spread,
                    'last': close,
                    'time': datetime.now()
                }
            return None
        
        symbol = self._resolve_symbol(pair)
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return {
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'last': tick.last,
                    'time': datetime.fromtimestamp(tick.time)
                }
            return None
        except Exception as e:
            error_logger.error(f"Error getting price for {pair}: {str(e)}")
            return None

    def get_spread(self, pair):
        """Get current spread (ask-bid) for a pair.

        Returns None when price is unavailable so callers can decide fallback behavior.
        """
        price = self.get_latest_price(pair)
        if not price:
            return None
        bid = price.get('bid')
        ask = price.get('ask')
        if bid is None or ask is None:
            return None
        return abs(float(ask) - float(bid))
    
    def place_order(self, pair, order_type, lot_size, entry_price, stop_loss, take_profit):
        """Place a trading order"""
        if self.relay_mode:
            r = self._relay_post("/order", {
                "pair": pair,
                "type": order_type,
                "lot_size": lot_size,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            })
            if r and "ticket" in r:
                trades_logger.info(
                    f"ORDER_PLACED (RELAY) | Pair: {pair} | Type: {order_type} | "
                    f"Lot: {lot_size} | SL: {stop_loss:.5f} | TP: {take_profit:.5f} | "
                    f"Ticket: {r['ticket']}"
                )
                return r["ticket"]
            error_logger.error(f"Relay order failed for {pair}: {r}")
            return None

        if self.simulation_mode:
            ticket = int(datetime.now().timestamp() * 1000) % 1000000
            self.sim_positions.append({
                'ticket': ticket,
                'pair': pair,
                'type': order_type,
                'volume': lot_size,
                'open_price': entry_price,
                'current_price': entry_price,
                'sl': stop_loss,
                'tp': take_profit,
                'profit': 0.0,
                'open_time': datetime.now()
            })
            trades_logger.info(
                f"ORDER_PLACED (SIM) | Pair: {pair} | Type: {order_type} | "
                f"Lot: {lot_size} | SL: {stop_loss:.5f} | TP: {take_profit:.5f} | "
                f"Ticket: {ticket}"
            )
            return ticket
        
        try:
            symbol = self._resolve_symbol(pair)
            action = mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL
            
            # Don't include SL/TP in initial order (TradersWay and some brokers reject it)
            # We'll add them via modify_position after the order fills
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": action,
                "price": entry_price,
                "deviation": 20,
                "magic": 234000,
                "comment": "AI Trading Bot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_logger.error(
                    f"Order failed for {pair} {order_type}: {result.comment} (Code: {result.retcode})"
                )
                return None
            
            bot_logger.info(f"📝 Order filled: {pair} {order_type} | Order ticket: {result.order} | Deal: {result.deal}")
            
            # Get the position ticket (may differ from order ticket)
            # Wait for position to register, then look it up
            import time
            position_ticket = None
            position_obj = None
            
            # Retry position lookup - wait for MT5 to register the position
            for attempt in range(8):  # More attempts
                time.sleep(0.4)  # Wait 400ms between attempts
                positions = mt5.positions_get(symbol=symbol)
                bot_logger.debug(f"Position lookup attempt {attempt+1}: found {len(positions) if positions else 0} positions for {symbol}")
                if positions:
                    # Find most recent position with our magic number
                    bot_positions = [p for p in positions if p.magic == 234000]
                    if bot_positions:
                        # Get the newest one (highest ticket)
                        position_obj = max(bot_positions, key=lambda p: p.ticket)
                        position_ticket = position_obj.ticket
                        bot_logger.info(f"✓ Found position ticket: {position_ticket}")
                        break
            
            # Fallback to deal ticket from result if position lookup fails
            if not position_ticket:
                # Try result.deal first (deal ticket often works for SLTP)
                position_ticket = result.deal if result.deal else result.order
                bot_logger.warning(f"⚠️ Position lookup failed, using ticket: {position_ticket}")
            
            # Modify position to add SL/TP with aggressive retry
            if stop_loss or take_profit:
                modify_success = False
                bot_logger.info(f"🔧 Setting SL={stop_loss:.5f}, TP={take_profit:.5f} on ticket {position_ticket}")
                
                for retry in range(5):  # More retry attempts
                    time.sleep(0.6 + retry * 0.2)  # Increasing delay: 0.6s, 0.8s, 1.0s, 1.2s, 1.4s
                    
                    # Build SLTP request directly (bypass modify_position for more control)
                    sltp_request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position_ticket,
                        "symbol": symbol,
                        "sl": stop_loss if stop_loss else 0.0,
                        "tp": take_profit if take_profit else 0.0,
                    }
                    
                    modify_result = mt5.order_send(sltp_request)
                    
                    if modify_result and modify_result.retcode == mt5.TRADE_RETCODE_DONE:
                        bot_logger.info(f"✅ SL/TP modification successful on attempt {retry+1}")
                        modify_success = True
                        break
                    else:
                        retcode = modify_result.retcode if modify_result else "None"
                        comment = modify_result.comment if modify_result else "No result"
                        bot_logger.warning(f"⚠️ SL/TP attempt {retry+1}/5 failed: {comment} (code={retcode})")
                
                # VERIFY SL/TP were actually set
                if modify_success:
                    time.sleep(0.3)
                    verify_positions = mt5.positions_get(ticket=position_ticket)
                    if verify_positions:
                        vp = verify_positions[0]
                        if vp.sl == 0.0 or vp.tp == 0.0:
                            bot_logger.error(f"❌ VERIFICATION FAILED: Position {position_ticket} has SL={vp.sl}, TP={vp.tp}")
                            modify_success = False
                        else:
                            bot_logger.info(f"✓ Verified: SL={vp.sl:.5f}, TP={vp.tp:.5f}")
                
                if not modify_success:
                    error_logger.error(f"❌ CRITICAL: Failed to set SL/TP on position {position_ticket}!")
                    # Close the position to prevent unprotected trade
                    bot_logger.warning(f"🚨 Closing unprotected position {position_ticket} for safety")
                    close_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": lot_size,
                        "type": mt5.ORDER_TYPE_SELL if order_type == 'BUY' else mt5.ORDER_TYPE_BUY,
                        "position": position_ticket,
                        "deviation": 50,
                        "magic": 234000,
                        "comment": "Safety close - no SL/TP",
                    }
                    close_result = mt5.order_send(close_request)
                    if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
                        bot_logger.info(f"✓ Unprotected position closed")
                    else:
                        error_logger.error(f"❌ FAILED to close unprotected position! Manual intervention needed!")
                    return None  # Don't return ticket if we closed it
            
            trades_logger.info(
                f"ORDER_PLACED | Pair: {pair} | Type: {order_type} | "
                f"Lot: {lot_size} | SL: {stop_loss:.5f} | TP: {take_profit:.5f} | "
                f"Ticket: {position_ticket}"
            )
            return position_ticket
        
        except Exception as e:
            error_logger.error(f"Error placing order for {pair}: {str(e)}")
            return None
    
    def close_position(self, pair, volume, ticket=None):
        """Close an open position by ticket or pair."""
        if self.relay_mode:
            payload = {"volume": volume}
            if ticket:
                payload["ticket"] = ticket
            else:
                payload["pair"] = pair
            r = self._relay_post("/close", payload)
            if r and "closed" in r:
                trades_logger.info(f"POSITION_CLOSED (RELAY) | Pair: {pair} | Closed ticket: {r['closed']}")
                return r["closed"]
            error_logger.error(f"Relay close failed for {pair}: {r}")
            return None

        elif self.simulation_mode:
            for i, pos in enumerate(self.sim_positions):
                if pos['pair'] == pair:
                    closed = self.sim_positions.pop(i)
                    trades_logger.info(f"POSITION_CLOSED (SIM) | Pair: {pair} | P/L: {closed['profit']:.2f}")
                    return closed['ticket']
            return None

        else:
            try:
                symbol = self._resolve_symbol(pair)
                positions = mt5.positions_get(symbol=symbol)
                if not positions:
                    bot_logger.warning(f"No open position for {pair}")
                    return None

                position = positions[0]
                close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume,
                    "type": close_type,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "AI Bot Position Close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                result = mt5.order_send(request)

                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    error_logger.error(f"Failed to close {pair}: {result.comment}")
                    return None
                return result.order

            except Exception as e:
                error_logger.error(f"Error closing position for {pair}: {str(e)}")
                return None

    def close_all_positions(self):
        """Close ALL open positions via relay or native MT5."""
        if self.relay_mode:
            r = self._relay_post("/close_all", {})
            if r and "results" in r:
                for res in r["results"]:
                    if res.get("closed"):
                        trades_logger.info(f"POSITION_CLOSED (RELAY) | Ticket: {res['ticket']}")
                    else:
                        error_logger.error(f"Failed to close ticket {res['ticket']}: {res.get('error')}")
                return r["results"]
            error_logger.error(f"Relay close_all failed: {r}")
            return None

    def modify_position(self, ticket, sl=None, tp=None):
        """Modify SL and/or TP on an existing position.
        
        Args:
            ticket: Position ticket number
            sl: New stop-loss price (or None to keep current)
            tp: New take-profit price (or None to keep current)
        
        Returns:
            dict with modified ticket info, or None on failure
        """
        if self.relay_mode:
            payload = {"ticket": ticket}
            if sl is not None:
                payload["sl"] = sl
            if tp is not None:
                payload["tp"] = tp
            r = self._relay_post("/modify", payload)
            if r and "modified" in r:
                bot_logger.info(
                    f"Position modified (RELAY) | Ticket: {ticket} | "
                    f"SL: {r.get('sl')} | TP: {r.get('tp')}"
                )
                return r
            error_logger.error(f"Relay modify failed for ticket {ticket}: {r}")
            return None

        if self.simulation_mode:
            for pos in self.sim_positions:
                if pos.get('ticket') == ticket:
                    if sl is not None:
                        pos['sl'] = sl
                    if tp is not None:
                        pos['tp'] = tp
                    return {"modified": ticket, "sl": pos['sl'], "tp": pos['tp']}
            return None

        # Native MT5
        try:
            req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
            }
            # Look up the position to get symbol and fill in current SL/TP
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                error_logger.error(f"No position with ticket {ticket}")
                return None
            pos = positions[0]
            req["symbol"] = pos.symbol
            req["sl"] = sl if sl is not None else pos.sl
            req["tp"] = tp if tp is not None else pos.tp

            result = mt5.order_send(req)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                return {"modified": ticket, "sl": req["sl"], "tp": req["tp"]}
            error_logger.error(f"Modify failed for ticket {ticket}: {result.comment if result else 'None'}")
            return None
        except Exception as e:
            error_logger.error(f"Error modifying position {ticket}: {e}")
            return None
    
    # Magic number used by the bot to tag its own orders
    BOT_MAGIC = 234000

    def get_open_positions(self, pair=None):
        """Get ALL open positions (including manual trades)"""
        if self.relay_mode:
            params = {"pair": pair} if pair else {}
            r = self._relay_get("/positions", params)
            if r and "positions" in r:
                return r["positions"]
            # Return None (not []) on relay failure so callers can
            # distinguish "relay down" from "no open positions"
            return None if r is None else []

        elif self.simulation_mode:
            positions = self.sim_positions
            if pair:
                positions = [p for p in positions if p['pair'] == pair]
            return [
                {
                    'ticket': p['ticket'],
                    'pair': p['pair'],
                    'type': p['type'],
                    'volume': p['volume'],
                    'open_price': p['open_price'],
                    'current_price': p['current_price'],
                    'profit': p['profit'],
                    'open_time': p['open_time'],
                }
                for p in positions
            ]

        else:
            try:
                if pair:
                    positions = mt5.positions_get(symbol=pair)
                else:
                    positions = mt5.positions_get()

                if not positions:
                    return []

                return [
                    {
                        'ticket': p.ticket,
                        'pair': p.symbol,
                        'type': 'BUY' if p.type == 0 else 'SELL',
                        'volume': p.volume,
                        'open_price': p.price_open,
                        'current_price': p.price_current,
                        'profit': p.profit,
                        'open_time': datetime.fromtimestamp(p.time),
                        'sl': p.sl,
                        'tp': p.tp,
                        'magic': p.magic,  # Include magic number for bot position filtering
                    }
                    for p in positions
                ]
            except Exception as e:
                error_logger.error(f"Error getting positions: {str(e)}")
                return []

    def get_bot_positions(self, pair=None):
        """Get only positions placed by this bot (magic=234000).

        Manual trades are excluded so the bot's slot counter is independent.
        Falls back to ALL positions if the relay doesn't provide the magic field
        (to avoid thinking there are 0 open trades and over-trading).
        """
        all_positions = self.get_open_positions(pair)
        if all_positions is None:
            return None  # Relay failure — propagate None, don't return []
        if not all_positions:
            return all_positions  # genuinely empty list

        # Check if the relay provides the magic field
        has_magic = any('magic' in p for p in all_positions)
        if not has_magic:
            # Relay hasn't been updated — treat ALL positions as bot positions
            # to prevent over-trading
            return all_positions

        return [
            p for p in all_positions
            if p.get('magic') == self.BOT_MAGIC
        ]
    
    def shutdown(self):
        """Shutdown MT5 connection"""
        if not self.simulation_mode and MT5_AVAILABLE:
            mt5.shutdown()
        self.connected = False
        bot_logger.info("MT5 connection closed")

    def get_trade_history(self, hours=24, include_all=False):
        """
        Get recently closed deals from MT5.
        Returns list of dicts with position_id, pair, profit, etc.
        
        Args:
            hours: How far back to look
            include_all: If True, include ALL deals (not just bot's magic number)
        """
        # ── Relay mode ──
        # ── Relay mode ──
        if self.relay_mode:
            try:
                result = self._relay_get("/history", params={"hours": hours})
                if result:
                    return result.get("deals", [])
                return []
            except Exception as e:
                bot_logger.warning(f"Relay /history failed: {e}")
                return []

        # ── Simulation mode ──
        if self.simulation_mode:
            return []  # no real history in sim

        # ── Native MT5 ──
        if not MT5_AVAILABLE:
            return []
        try:
            from datetime import timedelta
            now = datetime.now()
            from_date = now - timedelta(hours=hours)
            deals = mt5.history_deals_get(from_date, now)
            if not deals:
                return []
            closed = []
            for d in deals:
                if d.entry != 1:
                    continue
                # Only include trades placed by this bot unless include_all is True
                if not include_all and d.magic != self.BOT_MAGIC:
                    continue
                closed.append({
                    "ticket": d.ticket,
                    "order": d.order,
                    "position_id": d.position_id,
                    "pair": d.symbol,
                    "type": "BUY" if d.type == 0 else "SELL",
                    "volume": d.volume,
                    "price": d.price,
                    "profit": d.profit,
                    "commission": d.commission,
                    "swap": d.swap,
                    "time": datetime.fromtimestamp(d.time).isoformat(),
                    "magic": d.magic,  # Include magic for debugging
                })
            return closed
        except Exception as e:
            bot_logger.warning(f"MT5 history_deals_get failed: {e}")
            return []
