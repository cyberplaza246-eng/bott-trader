"""
Backtesting Framework
Test trading strategy on historical data — matches the live 8-model ensemble
(Sentiment is skipped because it needs a live API).
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.ai.technical_analyzer import TechnicalAnalyzer
from src.ai.volume_analyzer import VolumeAnalyzer
from src.ai.lstm_predictor import LSTMPredictor
from src.ai.ema_crossover import EMACrossoverAnalyzer
from src.ai.candlestick_patterns import CandlestickPatternDetector
from src.ai.support_resistance import SupportResistanceDetector
from src.ai.multi_timeframe import MultiTimeframeAnalyzer
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
        self.ema_crossover = EMACrossoverAnalyzer()
        self.candlestick = CandlestickPatternDetector()
        self.support_resistance = SupportResistanceDetector()
        self.multi_timeframe = MultiTimeframeAnalyzer()
        self.risk_manager = RiskManager(initial_balance=initial_balance)
    
    def run_backtest(self, historical_data, pair, confidence_threshold=0.55, min_agreement=3):
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
        total_models = 7  # LSTM, Tech, Vol, EMA, Candle, S/R, MTF (no Sentiment)
        
        for idx in range(200, len(data)):  # Skip first 200 for EMA200
            current_candle = data.iloc[idx]
            current_price = current_candle['close']
            
            # Get signals from all available models
            historical_subset = data.iloc[max(0, idx-200):idx+1]
            
            lstm_signal = self.lstm.predict_direction(historical_subset)
            technical_signal = self.technical.get_signal(historical_subset)
            volume_signal = self.volume.get_volume_signal(historical_subset)
            ema_signal = self.ema_crossover.get_signal(historical_subset)
            candle_signal = self.candlestick.get_pattern_signal(historical_subset)
            sr_signal = self.support_resistance.get_sr_signal(historical_subset)
            
            # MTF: resample 1h data to simulate higher timeframes
            mtf_signal = self._get_mtf_signal(data, idx)
            
            all_signals = [
                lstm_signal, technical_signal, volume_signal,
                ema_signal, candle_signal, sr_signal, mtf_signal
            ]
            
            # Count agreement like the live ensemble
            buy_votes = sum(1 for s in all_signals if s.get('signal') == 'BUY')
            sell_votes = sum(1 for s in all_signals if s.get('signal') == 'SELL')
            
            # Determine direction
            if buy_votes >= min_agreement and buy_votes > sell_votes:
                direction = 'BUY'
                agreement = buy_votes
            elif sell_votes >= min_agreement and sell_votes > buy_votes:
                direction = 'SELL'
                agreement = sell_votes
            else:
                direction = None
                agreement = 0
            
            # EMA200 hard gate
            ema200 = current_candle.get('ema_200', None)
            if direction and ema200 is not None:
                if direction == 'BUY' and current_price < ema200:
                    direction = None  # Block counter-trend BUY
                elif direction == 'SELL' and current_price > ema200:
                    direction = None  # Block counter-trend SELL
            
            # Open position
            if not open_position and direction:
                entry_price = current_price
                atr = current_candle.get('atr', 0.001)
                
                stop_loss = self.risk_manager.calculate_stop_loss(entry_price, atr, direction)
                take_profit = self.risk_manager.calculate_take_profit(entry_price, stop_loss, direction)
                
                position_size = self.risk_manager.calculate_position_size(entry_price, stop_loss, pair=pair)
                
                if position_size:
                    open_position = {
                        'entry_price': entry_price,
                        'entry_idx': idx,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'lot_size': position_size['lot_size'],
                        'type': direction
                    }
                    bot_logger.info(
                        f"[{idx}] {direction} @ {entry_price:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}"
                    )
            
            # Close position
            if open_position:
                pos_type = open_position['type']
                
                if pos_type == 'BUY':
                    hit_sl = current_price <= open_position['stop_loss']
                    hit_tp = current_price >= open_position['take_profit']
                else:  # SELL
                    hit_sl = current_price >= open_position['stop_loss']
                    hit_tp = current_price <= open_position['take_profit']
                
                if hit_sl:
                    exit_price = open_position['stop_loss']
                    exit_type = 'STOP_LOSS'
                elif hit_tp:
                    exit_price = open_position['take_profit']
                    exit_type = 'TAKE_PROFIT'
                else:
                    exit_price = None
                
                if exit_price:
                    # Calculate P/L (direction-aware)
                    pip_size = 0.01 if 'JPY' in pair else 0.0001
                    if pos_type == 'BUY':
                        pips = (exit_price - open_position['entry_price']) / pip_size
                    else:  # SELL
                        pips = (open_position['entry_price'] - exit_price) / pip_size
                    profit_loss = pips * open_position['lot_size'] * (1000 if 'JPY' in pair else 10)
                    
                    self.current_balance += profit_loss
                    self.equity = self.current_balance
                    equity_curve.append(self.equity)
                    
                    trade_record = {
                        'type': pos_type,
                        'entry_price': open_position['entry_price'],
                        'exit_price': exit_price,
                        'exit_type': exit_type,
                        'profit_loss': profit_loss,
                        'pips': pips,
                        'profit_loss_percent': (profit_loss / self.initial_balance) * 100,
                        'candles_held': idx - open_position['entry_idx']
                    }
                    
                    self.trades.append(trade_record)
                    
                    bot_logger.info(
                        f"[{idx}] EXIT ({exit_type}) @ {exit_price:.5f} | P/L: ${profit_loss:.2f}"
                    )
                    
                    open_position = None
        
        # Generate results
        return self._generate_backtest_results(equity_curve, pair)
    
    def _get_mtf_signal(self, data, idx):
        """Simulate multi-timeframe by resampling 1h data to 4h and daily"""
        try:
            # Use last 200 bars, resample to 4h
            subset = data.iloc[max(0, idx - 200):idx + 1].copy()
            if 'datetime' not in subset.columns:
                return {'signal': 'HOLD', 'confidence': 0.0}
            
            subset = subset.set_index('datetime')
            
            # 4h resample
            h4 = subset.resample('4h').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum'
            }).dropna()
            
            if len(h4) < 30:
                return {'signal': 'HOLD', 'confidence': 0.0}
            
            h4 = h4.reset_index()
            h4 = self.technical.calculate_indicators(h4)
            return self.technical.get_signal(h4)
        except Exception:
            return {'signal': 'HOLD', 'confidence': 0.0}
    
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
