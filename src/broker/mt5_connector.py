"""
MT5 Broker Connection and Order Management

On Windows: Uses real MetaTrader5 API
On Linux/Mac: Uses simulation mode for paper trading and backtesting
"""
import pandas as pd
import numpy as np
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
    bot_logger.warning("MetaTrader5 not available (Linux/Mac). Using simulation mode.")


class MT5Connector:
    def __init__(self):
        self.account = MT5_ACCOUNT
        self.password = MT5_PASSWORD
        self.server = MT5_SERVER
        self.connected = False
        self.simulation_mode = not MT5_AVAILABLE or TRADING_MODE in ('paper', 'backtest')
        self.sim_balance = 50.0
        self.sim_equity = 50.0
        self.sim_positions = []
        self.initialize()
    
    def initialize(self):
        """Initialize MT5 connection or simulation mode"""
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
    
    def get_account_info(self):
        """Get current account information"""
        if self.simulation_mode:
            return {
                'login': self.account or 'SIM_ACCOUNT',
                'balance': self.sim_balance,
                'equity': self.sim_equity,
                'margin_free': self.sim_balance * 0.95,
                'margin': self.sim_balance * 0.05,
                'profit': self.sim_equity - self.sim_balance,
                'leverage': 100,
                'currency': 'USD',
                'server': 'Simulation'
            }
        if not self.connected:
            return None
        return mt5.account_info()._asdict()
    
    def get_balance(self):
        """Get current account balance"""
        if self.simulation_mode:
            return self.sim_balance
        account_info = mt5.account_info()
        return account_info.balance if account_info else None
    
    def get_equity(self):
        """Get current account equity"""
        if self.simulation_mode:
            return self.sim_equity
        account_info = mt5.account_info()
        return account_info.equity if account_info else None
    
    def get_candles(self, pair, timeframe_minutes, num_candles=100):
        """
        Fetch OHLCV candle data
        
        In simulation mode: generates realistic synthetic data
        In live mode: fetches from MT5
        """
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
        
        try:
            candles = mt5.copy_rates_from_pos(pair, tf, 0, num_candles)
            if candles is None:
                error_logger.error(f"Failed to fetch candles for {pair}: {mt5.last_error()}")
                return None
            
            df = pd.DataFrame(candles)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'real_volume']]
            df.rename(columns={
                'time': 'datetime',
                'tick_volume': 'volume',
                'real_volume': 'real_volume'
            }, inplace=True)
            
            return df
        except Exception as e:
            error_logger.error(f"Error fetching candles for {pair}: {str(e)}")
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
    
    def get_latest_price(self, pair):
        """Get latest bid/ask price"""
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
        
        try:
            tick = mt5.symbol_info_tick(pair)
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
    
    def place_order(self, pair, order_type, lot_size, entry_price, stop_loss, take_profit):
        """Place a trading order"""
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
            action = mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pair,
                "volume": lot_size,
                "type": action,
                "price": entry_price,
                "sl": stop_loss,
                "tp": take_profit,
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
            
            trades_logger.info(
                f"ORDER_PLACED | Pair: {pair} | Type: {order_type} | "
                f"Lot: {lot_size} | SL: {stop_loss:.5f} | TP: {take_profit:.5f} | "
                f"Ticket: {result.order}"
            )
            return result.order
        
        except Exception as e:
            error_logger.error(f"Error placing order for {pair}: {str(e)}")
            return None
    
    def close_position(self, pair, volume):
        """Close an open position"""
        if self.simulation_mode:
            for i, pos in enumerate(self.sim_positions):
                if pos['pair'] == pair:
                    closed = self.sim_positions.pop(i)
                    trades_logger.info(f"POSITION_CLOSED (SIM) | Pair: {pair} | P/L: {closed['profit']:.2f}")
                    return closed['ticket']
            return None
        
        try:
            positions = mt5.positions_get(symbol=pair)
            if not positions:
                bot_logger.warning(f"No open position for {pair}")
                return None
            
            position = positions[0]
            close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pair,
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
    
    def get_open_positions(self, pair=None):
        """Get open positions"""
        if self.simulation_mode:
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
                }
                for p in positions
            ]
        except Exception as e:
            error_logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def shutdown(self):
        """Shutdown MT5 connection"""
        if not self.simulation_mode and MT5_AVAILABLE:
            mt5.shutdown()
        self.connected = False
        bot_logger.info("MT5 connection closed")
