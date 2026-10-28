"""
Intelligent Trader Module — Intelligent Trading System

High-level trading interface that integrates with the existing bot:
  - Provides signal generation compatible with EnsembleTrader interface
  - Manages model training and persistence
  - Handles real-time predictions
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from .analyzer import IntelligentAnalyzer
from .feature_generator import FeatureGenerator
from .label_generator import LabelGenerator
from .backtesting import BacktestEngine
from .model_store import ModelStore

from src.utils.logger import bot_logger


class IntelligentTrader:
    """
    Intelligent trading system that plugs into the existing bot.
    
    Compatible with EnsembleTrader interface:
      - get_trading_signal(df, pair) -> Dict with signal, confidence, etc.
    
    Additional features:
      - Auto-training on historical data
      - Backtesting functionality
      - Model persistence
      - Feature importance analysis
    """
    
    def __init__(
        self,
        config: Dict = None,
        auto_train: bool = True,
        broker = None
    ):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.analyzer = IntelligentAnalyzer(self.config)
        self.feature_gen = FeatureGenerator()
        self.label_gen = LabelGenerator()
        self.backtest = BacktestEngine()
        self.model_store = ModelStore()
        self.broker = broker
        
        # State
        self.is_trained = False
        self.training_history = []
        self.last_train_time = None
        self.retrain_interval = timedelta(hours=int(self.config.get('retrain_hours', 24)))
        
        # Performance tracking
        self.prediction_history = []
        self.trade_history = []
        
        bot_logger.info("🧠 Intelligent Trader initialized")
        
        # Auto-train if requested and data available
        if auto_train:
            self._attempt_initial_training()
            
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'feature_sets': [
                {
                    'generator': 'talib',
                    'config': {
                        'columns': ['close'],
                        'functions': ['SMA', 'EMA', 'RSI', 'MACD'],
                        'windows': [5, 10, 20, 50]
                    }
                },
                {
                    'generator': 'itbstats',
                    'config': {
                        'columns': ['close'],
                        'functions': ['skew', 'kurtosis', 'slope'],
                        'windows': [10, 20, 50]
                    }
                }
            ],
            'algorithms': [
                {'name': 'gb', 'algo': 'gb', 'params': {'learning_rate': 0.05}},
                {'name': 'svc', 'algo': 'svc', 'params': {'C': 1.0}},
            ],
            'signal_config': {
                'buy_threshold': 0.08,
                'sell_threshold': -0.08
            },
            'features_horizon': 200,
            'label_horizon': 60,
            'retrain_hours': 24,
            'min_training_samples': 1000
        }
        
    def _attempt_initial_training(self):
        """Attempt to load pre-trained models or train on available data."""
        # Try to load existing models
        models = self.model_store.list_models()
        if models:
            bot_logger.info(f"📚 Found {len(models)} pre-trained models")
            self.is_trained = True
            return
            
        bot_logger.info("⏳ No pre-trained models found, will train on first data")
        
    def get_trading_signal(
        self,
        df: pd.DataFrame,
        pair: str = None,
        timeframe: str = '1m'
    ) -> Dict[str, Any]:
        """
        Get trading signal from intelligent analysis.
        
        Compatible with EnsembleTrader interface.
        
        Args:
            df: OHLCV DataFrame
            pair: Trading pair symbol
            timeframe: Data timeframe
            
        Returns:
            Dict with signal, confidence, and analysis details
        """
        # Check if training needed
        if not self.is_trained:
            if len(df) >= self.config.get('min_training_samples', 1000):
                self.train(df, pair)
                
        # Check if retrain needed
        if self._needs_retrain():
            self._schedule_retrain(df, pair)
            
        # Get analysis
        result = self.analyzer.analyze(df, pair, timeframe)
        
        # Enhance result with additional info
        result['pair'] = pair
        result['timeframe'] = timeframe
        result['analysis_type'] = 'intelligent'
        
        # Add signal strength category
        result['signal_strength'] = self.analyzer.get_signal_strength(result)
        
        # Track prediction
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'pair': pair,
            'signal': result.get('signal'),
            'confidence': result.get('confidence'),
            'price': result.get('price')
        })
        
        # Limit history size
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-500:]
            
        return result
        
    def train(
        self,
        df: pd.DataFrame,
        pair: str = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Train models on historical data.
        
        Args:
            df: Historical OHLCV DataFrame
            pair: Trading pair for model naming
            force: Force training even if already trained
            
        Returns:
            Training results
        """
        if self.is_trained and not force:
            return {'status': 'skipped', 'reason': 'Already trained'}
            
        label_horizon = self.config.get('label_horizon', 60)
        
        bot_logger.info(f"🎓 Training intelligent models for {pair or 'default'}...")
        
        result = self.analyzer.train(df, pair, label_horizon)
        
        if result.get('status') == 'success':
            self.is_trained = True
            self.last_train_time = datetime.now()
            self.training_history.append({
                'timestamp': self.last_train_time,
                'pair': pair,
                'samples': result.get('samples'),
                'results': result.get('model_results')
            })
            
            bot_logger.info(f"✅ Training complete: {result.get('samples')} samples, "
                           f"{len(result.get('model_results', {}))} models")
        else:
            bot_logger.warning(f"⚠️ Training issue: {result.get('error', 'Unknown')}")
            
        return result
        
    def _needs_retrain(self) -> bool:
        """Check if models need retraining."""
        if self.last_train_time is None:
            return False
            
        return datetime.now() - self.last_train_time > self.retrain_interval
        
    def _schedule_retrain(self, df: pd.DataFrame, pair: str):
        """Schedule model retraining (async in production)."""
        # For now, just retrain synchronously
        # In production, this could be done in background thread
        try:
            self.train(df, pair, force=True)
        except Exception as e:
            bot_logger.warning(f"Retrain failed: {e}")
            
    def backtest_strategy(
        self,
        df: pd.DataFrame,
        pair: str = None,
        train_ratio: float = 0.7,
        direction: str = 'long'
    ) -> Dict[str, Any]:
        """
        Backtest the intelligent trading strategy.
        
        Args:
            df: Historical OHLCV DataFrame
            pair: Trading pair
            train_ratio: Ratio of data for training
            direction: Trading direction
            
        Returns:
            Backtest results
        """
        n = len(df)
        train_end = int(n * train_ratio)
        
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:]
        
        # Train on training data
        train_result = self.train(train_df, pair, force=True)
        
        if train_result.get('status') != 'success':
            return {'error': 'Training failed', 'details': train_result}
            
        # Generate signals on test data
        signals_df = test_df.copy()
        buy_signals = []
        sell_signals = []
        
        window_size = self.config.get('features_horizon', 200)
        
        for i in range(window_size, len(test_df)):
            window_df = test_df.iloc[max(0, i-window_size):i+1]
            result = self.analyzer.analyze(window_df, pair)
            
            buy_signals.append(1 if result.get('signal') == 'BUY' else 0)
            sell_signals.append(1 if result.get('signal') == 'SELL' else 0)
            
        # Pad with zeros for initial rows
        buy_signals = [0] * window_size + buy_signals
        sell_signals = [0] * window_size + sell_signals
        
        signals_df['buy_signal'] = buy_signals
        signals_df['sell_signal'] = sell_signals
        
        # Run backtest
        performance = self.backtest.simulate_trades(
            signals_df,
            buy_signal_col='buy_signal',
            sell_signal_col='sell_signal',
            price_col='close',
            direction=direction
        )
        
        # Generate equity curve
        if performance.get('trades'):
            equity_df = self.backtest.generate_equity_curve(performance['trades'])
        else:
            equity_df = pd.DataFrame()
            
        return {
            'training': train_result,
            'performance': performance,
            'equity_curve': equity_df,
            'test_period': {
                'start': str(test_df.index[0]),
                'end': str(test_df.index[-1]),
                'bars': len(test_df)
            }
        }
        
    def optimize_parameters(
        self,
        df: pd.DataFrame,
        pair: str = None,
        param_grid: Dict = None
    ) -> Dict[str, Any]:
        """
        Optimize signal thresholds using grid search.
        
        Args:
            df: Historical data
            pair: Trading pair
            param_grid: Grid of parameters to search
            
        Returns:
            Optimization results
        """
        if param_grid is None:
            param_grid = {
                'buy_signal_threshold': [0.04, 0.06, 0.08, 0.10, 0.12],
                'sell_signal_threshold': [-0.04, -0.06, -0.08, -0.10, -0.12]
            }
            
        # First train on the data
        self.train(df, pair, force=True)
        
        # Generate scores for all data
        scores = []
        window_size = self.config.get('features_horizon', 200)
        
        for i in range(window_size, len(df)):
            window_df = df.iloc[max(0, i-window_size):i+1]
            result = self.analyzer.analyze(window_df, pair)
            scores.append(result.get('trade_score', 0))
            
        # Create dataframe with scores
        score_df = df.iloc[window_size:].copy()
        score_df['trade_score'] = scores
        
        # Grid search
        best_params, all_results = self.backtest.grid_search_parameters(
            score_df,
            score_col='trade_score',
            param_grid=param_grid,
            price_col='close',
            direction='long',
            metric='profit_pct'
        )
        
        # Update config with best parameters
        if best_params:
            self.config['signal_config'].update(best_params)
            bot_logger.info(f"🎯 Optimal parameters: {best_params}")
            
        return {
            'best_params': best_params,
            'all_results': sorted(all_results, key=lambda x: x.get('profit_pct', 0), reverse=True)[:10]
        }
        
    def get_feature_analysis(self) -> Dict[str, Any]:
        """Get feature importance analysis."""
        importance = self.analyzer.get_feature_importance()
        
        # Aggregate across models
        all_features = {}
        for model_name, model_importance in importance.items():
            for feature, value in model_importance.items():
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append(abs(value))
                
        # Average importance
        avg_importance = {
            f: np.mean(values) for f, values in all_features.items()
        }
        
        # Sort by importance
        sorted_features = sorted(
            avg_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'top_features': sorted_features[:20],
            'per_model': importance,
            'total_features': len(all_features)
        }
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from prediction history."""
        if not self.prediction_history:
            return {'error': 'No prediction history'}
            
        predictions_df = pd.DataFrame(self.prediction_history)
        
        return {
            'total_predictions': len(predictions_df),
            'signal_distribution': predictions_df['signal'].value_counts().to_dict(),
            'avg_confidence': predictions_df['confidence'].mean(),
            'last_prediction': predictions_df.iloc[-1].to_dict() if len(predictions_df) > 0 else None
        }
        
    def save_models(self):
        """Save all trained models."""
        self.analyzer.save_state()
        bot_logger.info("💾 Models saved")
        
    def load_models(self):
        """Load saved models."""
        self.analyzer.load_state()
        self.is_trained = True
        bot_logger.info("📂 Models loaded")
        
    def record_trade_outcome(
        self,
        prediction: Dict,
        entry_price: float,
        exit_price: float,
        direction: str
    ) -> None:
        """
        Record trade outcome for learning.
        
        Args:
            prediction: Original prediction dict
            entry_price: Trade entry price
            exit_price: Trade exit price
            direction: 'long' or 'short'
        """
        if direction == 'long':
            profit_pct = (exit_price / entry_price - 1) * 100
            was_correct = profit_pct > 0
        else:
            profit_pct = (entry_price / exit_price - 1) * 100
            was_correct = profit_pct > 0
            
        outcome = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,
            'profit_pct': profit_pct,
            'was_correct': was_correct
        }
        
        self.trade_history.append(outcome)
        
        # Update analyzer for adaptive learning
        self.analyzer.update_with_outcome(prediction, outcome)
        
    def get_trading_summary(self) -> str:
        """Get a text summary of trading performance."""
        stats = self.get_performance_stats()
        
        lines = [
            "=" * 40,
            "INTELLIGENT TRADER SUMMARY",
            "=" * 40,
            f"Models Trained: {self.is_trained}",
            f"Total Predictions: {stats.get('total_predictions', 0)}",
            f"Signal Distribution: {stats.get('signal_distribution', {})}",
            f"Avg Confidence: {stats.get('avg_confidence', 0):.2%}",
        ]
        
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            win_rate = trades_df['was_correct'].mean() * 100
            avg_profit = trades_df['profit_pct'].mean()
            
            lines.extend([
                "",
                "TRADE OUTCOMES:",
                f"  Total Trades: {len(trades_df)}",
                f"  Win Rate: {win_rate:.1f}%",
                f"  Avg Profit: {avg_profit:.2f}%",
            ])
            
        return "\n".join(lines)
