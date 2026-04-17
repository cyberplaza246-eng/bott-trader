"""
Scalping Trade Executor — 5-Minute Scalping Strategy Manager

Orchestrates scalping entries, exits, and micro risk management.
Supports both forex (EUR/USD, GBP/USD) and futures (MES, MNQ).
Integrates with broker and risk management systems.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.ai.scalping_analyzer import ScalpingAnalyzer
from src.risk.position_manager import PIP_VALUES, DEFAULT_PIP
from src.utils.logger import bot_logger, TradeLogger
from config.strategy_config import PAIRS as CONFIG_PAIRS


class ScalpingTrader:
    """Execute and manage 5-minute scalping trades."""
    
    # Pair configuration: sourced from strategy_config (auto-selects forex or futures)
    SCALPING_PAIRS = CONFIG_PAIRS
    
    # Hold time limits (in minutes) - scalping is very short-term
    MAX_HOLD_MINUTES = {
        'GBP/USD': 15,
        'EUR/USD': 20,
        'USD/JPY': 20,
        'MES': 10,      # Futures scalps — tighter hold times
        'NQ': 10,
    }
    DEFAULT_HOLD_MINUTES = 15
    
    # Quick wins mode multiplier (0.6 = 60% of normal hold time)
    QUICK_WINS_HOLD_MULTIPLIER = 0.6
    
    # Min time between signals for same pair (cooldown)
    SIGNAL_COOLDOWN_MINUTES = 2
    
    def __init__(self, broker=None, risk_manager=None, profit_mode='quick_wins'):
        """Initialize scalping trader.
        
        Args:
            broker: MT5Connector instance
            risk_manager: RiskManager instance
            profit_mode: 'quick_wins' for small fast wins, 'normal' for standard targets
        """
        self.broker = broker
        self.risk_manager = risk_manager
        self.profit_mode = profit_mode
        self.analyzer = ScalpingAnalyzer(profit_mode=profit_mode)
        self.trade_logger = TradeLogger()
        
        # Track recent signals per pair (to avoid overtrading)
        self.last_signal_time = {}
        
        # Track active scalp trades
        self.active_scalp_trades = {}  # {ticket: trade_data}
        
        mode_label = "QUICK_WINS" if profit_mode == 'quick_wins' else "NORMAL"
        bot_logger.info(f"🔪 Scalping Trader initialized ({', '.join(self.SCALPING_PAIRS)}) [{mode_label} mode]")
    
    def set_profit_mode(self, mode):
        """Switch profit mode at runtime.
        
        Args:
            mode: 'quick_wins' or 'normal'
        """
        if mode not in ['quick_wins', 'normal']:
            bot_logger.warning(f"Invalid profit mode '{mode}', using 'quick_wins'")
            mode = 'quick_wins'
        
        old_mode = self.profit_mode
        self.profit_mode = mode
        self.analyzer.profit_mode = mode
        
        mode_label = "QUICK_WINS" if mode == 'quick_wins' else "NORMAL"
        bot_logger.info(f"🔄 Profit mode changed: {old_mode} → {mode} [{mode_label}]")
        return mode
    
    def analyze_pair(self, df, pair):
        """Analyze a pair for scalping setup.
        
        Args:
            df: DataFrame with OHLCV data
            pair: Currency pair
            
        Returns:
            dict: Trading signal result
        """
        if df is None or len(df) < 200:
            return None
        
        signal = self.analyzer.get_signal(df, pair)
        
        # Apply cooldown filter
        now = datetime.now()
        last_time = self.last_signal_time.get(pair)
        
        if last_time and (now - last_time).total_seconds() < (self.SIGNAL_COOLDOWN_MINUTES * 60):
            remaining = self.SIGNAL_COOLDOWN_MINUTES - (now - last_time).total_seconds() / 60
            signal['skip_reason'] = f"Cooldown in effect ({remaining:.1f}m remaining)"
            signal['signal'] = 'SKIP'
            return signal
        
        return signal
    
    def validate_trade_conditions(self, signal, pair):
        """Additional filters before executing a trade.
        
        Args:
            signal: Signal dictionary from analyzer
            pair: Currency pair
            
        Returns:
            dict: {'valid': bool, 'reason': str}
        """
        validation = {'valid': True, 'reasons': []}
        
        # 1. Confidence threshold
        min_conf = 0.50 if self.profit_mode == 'quick_wins' else 0.70
        if signal['confidence'] < min_conf:
            validation['valid'] = False
            validation['reasons'].append(f"Confidence {signal['confidence']:.2f} below {min_conf}")
            return validation
        
        validation['reasons'].append(f"✓ Confidence {signal['confidence']:.2f} meets threshold")
        
        # 2. Risk/reward ratio acceptable
        min_rr = 1.2 if self.profit_mode == 'quick_wins' else 1.5
        if signal['risk_reward']:
            rr = signal['risk_reward']
            ratio = rr['reward_pips_1'] / (rr['risk_pips'] + 0.001)
            if ratio < min_rr:
                validation['valid'] = False
                validation['reasons'].append(f"Risk/Reward {ratio:.2f} below {min_rr}")
                return validation
            validation['reasons'].append(f"✓ Risk/Reward ratio {ratio:.2f}R acceptable")
        
        # 3. Trend is established
        trend = signal.get('trend', {})
        if trend.get('direction') == 'NONE':
            validation['valid'] = False
            validation['reasons'].append("No clear trend established")
            return validation
        
        validation['reasons'].append(f"✓ Strong {trend['direction']} trend (strength {trend.get('strength', 0):.2f})")
        
        # 4. Position limits
        if self.risk_manager:
            max_trades = self.risk_manager._current_tier['max_concurrent_trades']
            if self.risk_manager.open_trades >= max_trades:
                validation['valid'] = False
                validation['reasons'].append(
                    f"Max concurrent trades ({max_trades}) reached"
                )
                return validation
        
        validation['reasons'].append("✓ Position limits within threshold")
        
        return validation
    
    def calculate_lot_size(self, pair, stop_loss_pips):
        """Calculate lot size based on risk management rules.
        
        Args:
            pair: Currency pair
            stop_loss_pips: Stop loss in pips
            
        Returns:
            float: Lot size to use
        """
        if not self.risk_manager:
            # Default small lot size for scalping
            return 0.01
        
        # Get account balance and risk percent
        balance = self.risk_manager.current_balance
        tier = self.risk_manager._current_tier
        risk_percent = tier['risk_percent']
        
        # Risk amount in USD
        risk_amount = balance * (risk_percent / 100)
        
        # Pair config for pip value
        pip_info = PIP_VALUES.get(pair, DEFAULT_PIP)
        pip_value_per_lot = pip_info['pip_value_per_lot']
        
        # Lot size = Risk amount / (SL pips × pip value per lot)
        lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
        
        # Cap to max for account tier
        max_lot = tier['max_lot_size']
        lot_size = min(lot_size, max_lot)
        
        # Minimum lot for scalping
        lot_size = max(lot_size, 0.01)
        
        return round(lot_size, 2)
    
    def execute_trade(self, signal, pair):
        """Execute a scalping trade if all conditions met.
        
        Args:
            signal: Signal from analyzer
            pair: Currency pair
            
        Returns:
            dict: {ticket: int, entry: float, sl: float, tp: float} or None
        """
        if signal['signal'] not in ['BUY', 'SELL']:
            return None
        
        # Validate conditions
        validation = self.validate_trade_conditions(signal, pair)
        if not validation['valid']:
            bot_logger.warning(f"🚫 Trade validation failed for {pair}: {validation['reasons']}")
            return None
        
        if not self.broker:
            bot_logger.warning("No broker connection available for scalping trade")
            return None
        
        # Calculate position size
        sl_pips = signal['risk_reward'].get('risk_pips', 10)
        lot_size = self.calculate_lot_size(pair, sl_pips)
        
        # Prepare trade parameters
        entry = signal['entry_price']
        sl = signal['stop_loss']
        tp = signal['take_profit']
        direction = signal['signal']
        
        try:
            # Execute trade via broker
            ticket = self.broker.place_order(
                pair=pair,
                order_type=direction,
                lot_size=lot_size,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp
            )
            
            if ticket:
                # Log trade
                self.trade_logger.log_trade({
                    'ticket': ticket,
                    'pair': pair,
                    'direction': direction,
                    'entry': entry,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'volume': lot_size,
                    'risk_pips': sl_pips,
                    'risk_reward_ratio': signal['risk_reward'].get('reward_pips_1', 0) / (sl_pips + 0.001),
                    'confidence': signal['confidence'],
                    'setup_type': signal.get('setup', 'unknown'),
                })
                
                # Track the trade
                self.active_scalp_trades[ticket] = {
                    'pair': pair,
                    'direction': direction,
                    'entry_time': datetime.now(),
                    'entry_price': entry,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'volume': lot_size,
                    'confidence': signal['confidence'],
                }
                
                # Update signal cooldown
                self.last_signal_time[pair] = datetime.now()
                
                bot_logger.info(
                    f"✅ Scalp trade opened: {pair} {direction} @ {entry:.5f} "
                    f"SL={sl:.5f} TP={tp:.5f} ({sl_pips:.0f}p risk)"
                )
                
                return {
                    'ticket': ticket,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'pair': pair,
                    'direction': direction,
                }
            else:
                bot_logger.error(f"Failed to open scalping position for {pair}")
                return None
        
        except Exception as e:
            bot_logger.error(f"Error executing scalp trade: {e}")
            return None
    
    def check_active_trades(self):
        """Monitor active scalp trades for closure conditions.
        
        Returns:
            list: Closed trade info
        """
        if not self.broker:
            return []
        
        closed_trades = []
        now = datetime.now()
        
        # Get current positions from broker
        positions = self.broker.get_open_positions() or []
        open_tickets = {p['ticket']: p for p in positions}
        
        # Check each tracked scalp trade
        for ticket in list(self.active_scalp_trades.keys()):
            trade = self.active_scalp_trades[ticket]
            
            # If position is closed in broker, clean up
            if ticket not in open_tickets:
                hold_time = (now - trade['entry_time']).total_seconds() / 60
                closed_trades.append({
                    'ticket': ticket,
                    'pair': trade['pair'],
                    'hold_minutes': hold_time,
                    'status': 'closed'
                })
                del self.active_scalp_trades[ticket]
                continue
            
            # Check hold time limit (shorter in quick_wins mode)
            hold_time = (now - trade['entry_time']).total_seconds() / 60
            max_hold = self.MAX_HOLD_MINUTES.get(trade['pair'], 15)
            
            # Apply quick_wins multiplier (60% of normal hold time)
            if self.profit_mode == 'quick_wins':
                max_hold = max_hold * self.QUICK_WINS_HOLD_MULTIPLIER
            
            if hold_time > max_hold:
                bot_logger.warning(
                    f"⏱️ Scalp trade {ticket} ({trade['pair']}) held for {hold_time:.1f}min "
                    f"(max {max_hold}min) - force closing"
                )
                try:
                    self.broker.close_position(ticket)
                    closed_trades.append({
                        'ticket': ticket,
                        'pair': trade['pair'],
                        'reason': 'max_hold_time_exceeded',
                        'hold_minutes': hold_time,
                    })
                    del self.active_scalp_trades[ticket]
                except Exception as e:
                    bot_logger.error(f"Error force-closing scalp trade {ticket}: {e}")
        
        return closed_trades
    
    def process_candle(self, candle_data_by_pair=None, **kwargs):
        """Process new candle data and generate scalping signals.
        
        Args:
            candle_data_by_pair: dict mapping pair -> DataFrame
            **kwargs: legacy positional args (df_gbpusd, df_eurusd) for backward compat
            
        Returns:
            list: New trades opened (if any)
        """
        new_trades = []
        
        # Support legacy call signature: process_candle(df_gbp, df_eur)
        if candle_data_by_pair is None:
            candle_data_by_pair = {}
            legacy_pairs = ['GBP/USD', 'EUR/USD']
            for i, key in enumerate(['df_gbpusd', 'df_eurusd']):
                if key in kwargs and kwargs[key] is not None:
                    candle_data_by_pair[legacy_pairs[i]] = kwargs[key]
        
        for pair, df in candle_data_by_pair.items():
            if df is not None and len(df) > 0:
                signal = self.analyze_pair(df, pair)
                if signal and signal['signal'] in ['BUY', 'SELL']:
                    trade = self.execute_trade(signal, pair)
                    if trade:
                        new_trades.append(trade)
        
        # Monitor active trades
        closed = self.check_active_trades()
        
        return new_trades
    
    def get_summary(self):
        """Get current scalping stats.
        
        Returns:
            dict: Summary of active scalping trades
        """
        return {
            'active_scalp_trades': len(self.active_scalp_trades),
            'trades': [
                {
                    'ticket': t,
                    'pair': self.active_scalp_trades[t]['pair'],
                    'direction': self.active_scalp_trades[t]['direction'],
                    'hold_minutes': (datetime.now() - self.active_scalp_trades[t]['entry_time']).total_seconds() / 60,
                    'confidence': self.active_scalp_trades[t]['confidence'],
                }
                for t in self.active_scalp_trades
            ]
        }
