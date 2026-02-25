"""
Backtesting Framework
Test trading strategy on historical data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.ai.technical_analyzer import TechnicalAnalyzer
from src.ai.sentiment_analyzer import SentimentAnalyzer
from src.ai.volume_analyzer import VolumeAnalyzer
from src.ai.lstm_predictor import LSTMPredictor
from src.risk.position_manager import RiskManager
from src.utils.logger import bot_logger


class BacktestEngine:
    """Backtest trading strategy on historical data"""
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.equity = initial_balance
        self.trades = []
        self.daily_results = []
        
        self.technical = TechnicalAnalyzer()
        self.volume = VolumeAnalyzer()
        self.lstm = LSTMPredictor()
        self.risk_manager = RiskManager(initial_balance=initial_balance)
    
    def run_backtest(self, historical_data, pair, confidence_threshold=0.75, min_agreement=3):
        """
        Run backtest on historical data
        
        Args:
            historical_data: DataFrame with OHLCV data
            pair: Currency pair
            confidence_threshold: Min confidence for trading
            min_agreement: Min models agreeing
        
        Returns:
            Backtest results
        """
        bot_logger.info(f"Starting backtest for {pair}...")
        
        data = historical_data.copy()
        data = self.technical.calculate_indicators(data)
        data = self.volume.calculate_volume_profile(data)
        
        open_position = None
        equity_curve = [self.initial_balance]
        
        for idx in range(100, len(data)):  # Skip first 100 for indicators
            current_candle = data.iloc[idx]
            
            # Get signals from ensemble
            historical_subset = data.iloc[max(0, idx-60):idx]
            
            lstm_signal = self.lstm.predict_direction(historical_subset)
            technical_signal = self.technical.get_signal(historical_subset)
            volume_signal = self.volume.get_volume_signal(historical_subset)
            
            # Count agreement
            buy_votes = 0
            if lstm_signal['signal'] == 'BUY':
                buy_votes += 1
            if technical_signal['signal'] == 'BUY':
                buy_votes += 1
            if volume_signal['signal'] == 'BUY':
                buy_votes += 1
            
            # Open position
            if not open_position and buy_votes >= min_agreement:
                entry_price = current_candle['close']
                atr = current_candle['atr']
                
                stop_loss = self.risk_manager.calculate_stop_loss(entry_price, atr, 'BUY')
                take_profit = self.risk_manager.calculate_take_profit(entry_price, stop_loss, 'BUY')
                
                position_size = self.risk_manager.calculate_position_size(entry_price, stop_loss, pair=pair)
                
                if position_size:
                    open_position = {
                        'entry_price': entry_price,
                        'entry_idx': idx,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'lot_size': position_size['lot_size'],
                        'type': 'BUY'
                    }
                    bot_logger.info(
                        f"[{idx}] BUY @ {entry_price:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}"
                    )
            
            # Close position
            if open_position:
                current_price = current_candle['close']
                
                if current_price <= open_position['stop_loss']:
                    # Hit stop loss
                    exit_price = open_position['stop_loss']
                    exit_type = 'STOP_LOSS'
                elif current_price >= open_position['take_profit']:
                    # Hit take profit
                    exit_price = open_position['take_profit']
                    exit_type = 'TAKE_PROFIT'
                else:
                    exit_price = None
                
                if exit_price:
                    # Calculate P/L
                    pips = (exit_price - open_position['entry_price']) / 0.0001
                    profit_loss = pips * open_position['lot_size'] * 10
                    
                    self.current_balance += profit_loss
                    self.equity = self.current_balance
                    equity_curve.append(self.equity)
                    
                    trade_record = {
                        'entry_price': open_position['entry_price'],
                        'exit_price': exit_price,
                        'exit_type': exit_type,
                        'profit_loss': profit_loss,
                        'profit_loss_percent': (profit_loss / open_position['entry_price']) * 100,
                        'candles_held': idx - open_position['entry_idx']
                    }
                    
                    self.trades.append(trade_record)
                    
                    bot_logger.info(
                        f"[{idx}] EXIT ({exit_type}) @ {exit_price:.5f} | P/L: ${profit_loss:.2f}"
                    )
                    
                    open_position = None
        
        # Generate results
        return self._generate_backtest_results(equity_curve, pair)
    
    def _generate_backtest_results(self, equity_curve, pair):
        """Generate backtest statistics"""
        if not self.trades:
            return {
                'pair': pair,
                'total_trades': 0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'total_profit': 0.0
            }
        
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['profit_loss'] > 0]
        losing_trades = [t for t in self.trades if t['profit_loss'] <= 0]
        
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        gross_profit = sum(t['profit_loss'] for t in winning_trades)
        gross_loss = abs(sum(t['profit_loss'] for t in losing_trades))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        total_profit = self.current_balance - self.initial_balance
        
        # Calculate max drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100
        
        # Sharpe ratio (simplified)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        return {
            'pair': pair,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_profit': total_profit,
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'return_percent': (total_profit / self.initial_balance) * 100
        }
