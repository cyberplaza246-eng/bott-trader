"""
Paper Trading Mode - Run live signals with virtual money
"""
import pandas as pd
from datetime import datetime
from src.utils.logger import TradeLogger, bot_logger
from src.instruments import REGISTRY, get_instrument


# ── Pip / tick value lookup (sourced from instrument registry) ──────
_PIP_VALUES = {
    sym: {'pip_size': spec.tick_size, 'pip_value_per_lot': spec.tick_value_usd}
    for sym, spec in REGISTRY.items()
}
_DEFAULT_PIP = {'pip_size': 0.0001, 'pip_value_per_lot': 10.0}

# Simulated spread per pair (in pips / ticks)
_SPREAD_PIPS = {
    sym: spec.spread_default / spec.tick_size  # Convert price-unit spread to tick/pip count
    for sym, spec in REGISTRY.items()
}


def _pip_params(pair):
    """Return (pip_size, pip_value_per_lot) for a currency pair."""
    info = _PIP_VALUES.get(pair, _DEFAULT_PIP)
    return info['pip_size'], info['pip_value_per_lot']


class PaperTradingManager:
    """Simulate trades with virtual balance"""
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.equity = initial_balance
        self.open_positions = {}
        self.closed_trades = []
        self.trade_id = 0
    
    def execute_trade(self, pair, trade_type, entry_price, stop_loss, take_profit, lot_size):
        """
        Execute a paper trade
        
        Returns:
            Trade ID
        """
        self.trade_id += 1
        current_time = datetime.now()
        
        position = {
            'id': self.trade_id,
            'pair': pair,
            'type': trade_type,
            'entry_price': entry_price,
            'current_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'lot_size': lot_size,
            'entry_time': current_time,
            'entry_balance': self.current_balance
        }
        
        self.open_positions[pair] = position
        
        # Calculate risk (spread-adjusted)
        pip_size, pip_value = _pip_params(pair)
        spread_pips = _SPREAD_PIPS.get(pair, 1.5)
        spread_cost = spread_pips * pip_value * lot_size
        pip_distance = abs(entry_price - stop_loss)
        risk_per_lot = (pip_distance / pip_size) * pip_value
        risk_amount = risk_per_lot * lot_size
        
        TradeLogger.log_trade_entry(
            pair=pair,
            trade_type=trade_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            lot_size=lot_size
        )
        
        bot_logger.info(f"PAPER TRADE: {pair} {trade_type} @ {entry_price} | Risk: ${risk_amount:.2f}")
        
        return self.trade_id
    
    def update_position(self, pair, current_price):
        """
        Update position P/L with current price
        
        Returns:
            Dictionary with position status and any exit signal
        """
        if pair not in self.open_positions:
            return None
        
        position = self.open_positions[pair]
        position['current_price'] = current_price
        
        # Calculate profit/loss (spread-adjusted)
        pip_size, pip_value = _pip_params(position.get('pair', ''))
        spread_pips = _SPREAD_PIPS.get(position.get('pair', ''), 1.5)
        spread_cost = spread_pips * pip_value * position['lot_size']
        if position['type'] == 'BUY':
            pips_move = (current_price - position['entry_price']) / pip_size
            profit_loss = pips_move * position['lot_size'] * pip_value - spread_cost
        else:  # SELL
            pips_move = (position['entry_price'] - current_price) / pip_size
            profit_loss = pips_move * position['lot_size'] * pip_value - spread_cost
        
        position['profit_loss'] = profit_loss
        position['profit_loss_percent'] = (profit_loss / position['entry_balance']) * 100
        
        # Check exit conditions
        exit_type = None
        
        if position['type'] == 'BUY':
            if current_price <= position['stop_loss']:
                exit_type = 'STOP_LOSS'
                exit_price = position['stop_loss']
            elif current_price >= position['take_profit']:
                exit_type = 'TAKE_PROFIT'
                exit_price = position['take_profit']
        else:  # SELL
            if current_price >= position['stop_loss']:
                exit_type = 'STOP_LOSS'
                exit_price = position['stop_loss']
            elif current_price <= position['take_profit']:
                exit_type = 'TAKE_PROFIT'
                exit_price = position['take_profit']
        
        return {
            'position': position,
            'exit_signal': exit_type,
            'exit_price': exit_price if exit_type else None
        }
    
    def close_position(self, pair, exit_price, exit_type):
        """Close a position and record P/L"""
        if pair not in self.open_positions:
            return None
        
        position = self.open_positions.pop(pair)
        
        pip_size, pip_value = _pip_params(pair)
        if position['type'] == 'BUY':
            pips_move = (exit_price - position['entry_price']) / pip_size
        else:
            pips_move = (position['entry_price'] - exit_price) / pip_size
        
        profit_loss = pips_move * position['lot_size'] * pip_value
        profit_loss_percent = (profit_loss / position['entry_balance']) * 100
        
        # Update balance
        self.current_balance += profit_loss
        self.equity = self.current_balance
        
        # Record trade
        trade_record = {
            'trade_id': position['id'],
            'pair': pair,
            'type': position['type'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'lot_size': position['lot_size'],
            'profit_loss': profit_loss,
            'profit_loss_percent': profit_loss_percent,
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'exit_type': exit_type
        }
        
        self.closed_trades.append(trade_record)
        
        TradeLogger.log_trade_exit(
            pair=pair,
            exit_type=exit_type,
            exit_price=exit_price,
            profit_loss=profit_loss,
            profit_loss_percent=profit_loss_percent
        )
        
        return trade_record
    
    def get_summary(self):
        """Get paper trading summary statistics"""
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'initial_balance': self.initial_balance,
                'current_balance': self.current_balance,
                'return_percent': 0.0
            }
        
        total_trades = len(self.closed_trades)
        winning_trades = sum(1 for t in self.closed_trades if t['profit_loss'] > 0)
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        total_profit = sum(t['profit_loss'] for t in self.closed_trades)
        return_percent = (total_profit / self.initial_balance) * 100
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'return_percent': return_percent,
            'open_positions': len(self.open_positions)
        }
