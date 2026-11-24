"""
Signal Generator Module — Intelligent Trading System

Generates trading signals from ML predictions:
  - Score smoothing and combining
  - Threshold-based signal generation
  - Multi-score combination strategies
  - Signal filtering and validation
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from src.utils.logger import bot_logger


class SignalGenerator:
    """
    Generate trading signals from ML scores and predictions.
    
    Signal generation flow:
      1. Combine multiple model scores (if multiple models)
      2. Smooth scores with moving average
      3. Apply threshold rules to generate buy/sell signals
      4. Filter signals based on additional criteria
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Default thresholds
        self.buy_threshold = self.config.get('buy_threshold', 0.08)
        self.sell_threshold = self.config.get('sell_threshold', -0.08)
        
    def generate_signals(
        self,
        df_or_series,
        signal_sets: List[Dict] = None,
        method: str = None,
        buy_threshold: float = None,
        sell_threshold: float = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Generate signals according to signal set specifications.
        
        Args:
            df_or_series: DataFrame with prediction scores or Series of scores
            signal_sets: List of signal generator configurations (optional)
            method: Simple signal method ('threshold', 'crossover', etc.)
            buy_threshold: Simple buy threshold
            sell_threshold: Simple sell threshold
            
        Returns:
            Tuple of (df_with_signals, signal_column_names)
        """
        # Handle Series input
        if isinstance(df_or_series, pd.Series):
            df = pd.DataFrame({'score': df_or_series})
            score_col = 'score'
        else:
            df = df_or_series.copy()
            score_col = None
            
        # Handle simple method parameter
        if method is not None and signal_sets is None:
            bt = buy_threshold if buy_threshold is not None else self.buy_threshold
            st = sell_threshold if sell_threshold is not None else self.sell_threshold
            
            if method == 'threshold':
                signal_sets = [{
                    'generator': 'threshold_rule',
                    'config': {
                        'columns': [score_col] if score_col else list(df.columns),
                        'buy_threshold': bt,
                        'sell_threshold': st,
                        'names': 'trade_signal'
                    }
                }]
            elif method == 'crossover':
                signal_sets = [{
                    'generator': 'crossover',
                    'config': {
                        'fast_col': score_col if score_col else df.columns[0],
                        'slow_col': None,
                        'names': 'crossover_signal'
                    }
                }]
            else:
                signal_sets = [{
                    'generator': method,
                    'config': {
                        'columns': [score_col] if score_col else list(df.columns),
                        'buy_threshold': bt,
                        'sell_threshold': st
                    }
                }]
        elif signal_sets is None:
            # Default signal sets
            signal_sets = [{
                'generator': 'threshold_rule',
                'config': {
                    'columns': list(df.columns),
                    'buy_threshold': self.buy_threshold,
                    'sell_threshold': self.sell_threshold
                }
            }]
            
        all_signals = []
        
        for ss in signal_sets:
            generator = ss.get('generator', 'threshold_rule')
            config = ss.get('config', {})
            
            if generator == 'combine':
                signals = self._generate_combine_signals(df, config)
            elif generator == 'smoothen':
                signals = self._generate_smoothen_signals(df, config)
            elif generator == 'threshold_rule':
                signals = self._generate_threshold_signals(df, config)
            elif generator == 'multi_threshold':
                signals = self._generate_multi_threshold_signals(df, config)
            elif generator == 'crossover':
                signals = self._generate_crossover_signals(df, config)
            elif generator == 'divergence':
                signals = self._generate_divergence_signals(df, config)
            else:
                bot_logger.warning(f"Unknown signal generator: {generator}")
                signals = []
                
            all_signals.extend(signals)
            
        return df, all_signals
    
    def _generate_combine_signals(
        self,
        df: pd.DataFrame,
        config: Dict
    ) -> List[str]:
        """
        Combine multiple scores into one combined score.
        
        Config:
        {
            "columns": ["high_score", "low_score"],
            "names": "trade_score",
            "combine": "difference",  # or "relative", "no_combine"
            "coefficient": 1.0,
            "constant": 0.0
        }
        """
        signals = []
        
        columns = config.get('columns', [])
        if not columns or len(columns) < 2:
            bot_logger.warning("Combine requires at least 2 columns")
            return signals
            
        out_name = config.get('names', 'combined_score')
        combine_method = config.get('combine', 'difference')
        coefficient = config.get('coefficient', 1.0)
        constant = config.get('constant', 0.0)
        
        up_col = columns[0]
        down_col = columns[1]
        
        if up_col not in df.columns or down_col not in df.columns:
            bot_logger.warning(f"Columns {up_col} or {down_col} not found")
            return signals
            
        if combine_method == 'difference':
            # score = up - down, positive = buy, negative = sell
            df[out_name] = df[up_col] - df[down_col]
        elif combine_method == 'relative':
            # score in [-1, +1], based on proportion
            sum_scores = df[up_col] + df[down_col]
            df[out_name] = ((df[up_col] / (sum_scores + 1e-10)) * 2) - 1
        elif combine_method == 'no_combine':
            # Just use the first column directly
            df[out_name] = df[up_col]
        else:
            # Default: use higher value with sign
            df[out_name] = df[[up_col, down_col]].apply(
                lambda x: x[0] if x[0] >= x[1] else -x[1], axis=1
            )
            
        # Apply coefficient and constant
        df[out_name] = df[out_name] * coefficient + constant
        
        signals.append(out_name)
        return signals
    
    def _generate_smoothen_signals(
        self,
        df: pd.DataFrame,
        config: Dict
    ) -> List[str]:
        """
        Smooth score columns with moving average.
        
        Config:
        {
            "columns": ["score1", "score2"],  # Will average these
            "names": "smoothed_score",
            "window": 5,  # int for SMA, float for EMA span
            "point_threshold": null  # Optional threshold before smoothing
        }
        """
        signals = []
        
        columns = config.get('columns', [])
        if isinstance(columns, str):
            columns = [columns]
            
        if not columns:
            return signals
            
        out_name = config.get('names', 'smoothed_score')
        window = config.get('window', 5)
        point_threshold = config.get('point_threshold', None)
        
        # Average the columns
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return signals
            
        out_series = df[valid_cols].mean(axis=1, skipna=True)
        
        # Apply point threshold if specified
        if point_threshold is not None:
            out_series = (out_series >= point_threshold).astype(float)
            
        # Apply smoothing
        if isinstance(window, int):
            # Simple moving average
            out_series = out_series.rolling(window, min_periods=window // 2).mean()
        elif isinstance(window, float):
            # Exponential moving average
            out_series = out_series.ewm(span=int(window), min_periods=int(window) // 2, adjust=False).mean()
            
        df[out_name] = out_series
        signals.append(out_name)
        
        return signals
    
    def _generate_threshold_signals(
        self,
        df: pd.DataFrame,
        config: Dict
    ) -> List[str]:
        """
        Generate buy/sell signals based on score thresholds.
        
        Config:
        {
            "columns": "trade_score",
            "names": ["buy_signal", "sell_signal"],
            "parameters": {
                "buy_signal_threshold": 0.08,
                "sell_signal_threshold": -0.08
            }
        }
        """
        signals = []
        
        columns = config.get('columns', 'trade_score')
        if isinstance(columns, list):
            columns = columns[0]
            
        names = config.get('names', ['buy_signal', 'sell_signal'])
        params = config.get('parameters', {})
        
        buy_threshold = params.get('buy_signal_threshold', 0.08)
        sell_threshold = params.get('sell_signal_threshold', -0.08)
        
        if columns not in df.columns:
            bot_logger.warning(f"Column {columns} not found")
            return signals
            
        # Generate buy signal
        buy_name = names[0] if len(names) > 0 else 'buy_signal'
        df[buy_name] = (df[columns] >= buy_threshold).astype(int)
        signals.append(buy_name)
        
        # Generate sell signal
        sell_name = names[1] if len(names) > 1 else 'sell_signal'
        df[sell_name] = (df[columns] <= sell_threshold).astype(int)
        signals.append(sell_name)
        
        return signals
    
    def _generate_multi_threshold_signals(
        self,
        df: pd.DataFrame,
        config: Dict
    ) -> List[str]:
        """
        Generate signals using multiple thresholds (for multiple scores).
        
        Config:
        {
            "columns": ["score1", "score2"],
            "names": ["buy_signal", "sell_signal"],
            "parameters": {
                "buy_signal_threshold": 0.08,
                "buy_signal_threshold_2": 0.05,
                "sell_signal_threshold": -0.08,
                "sell_signal_threshold_2": -0.05
            }
        }
        """
        signals = []
        
        columns = config.get('columns', [])
        if isinstance(columns, str):
            columns = [columns]
            
        if len(columns) < 2:
            return self._generate_threshold_signals(df, config)
            
        names = config.get('names', ['buy_signal', 'sell_signal'])
        params = config.get('parameters', {})
        
        score1 = columns[0]
        score2 = columns[1]
        
        if score1 not in df.columns or score2 not in df.columns:
            return signals
            
        buy_thresh1 = params.get('buy_signal_threshold', 0.08)
        buy_thresh2 = params.get('buy_signal_threshold_2', 0.05)
        sell_thresh1 = params.get('sell_signal_threshold', -0.08)
        sell_thresh2 = params.get('sell_signal_threshold_2', -0.05)
        
        # Buy signal: both scores above their thresholds
        buy_name = names[0] if len(names) > 0 else 'buy_signal'
        df[buy_name] = (
            (df[score1] >= buy_thresh1) & (df[score2] >= buy_thresh2)
        ).astype(int)
        signals.append(buy_name)
        
        # Sell signal: both scores below their thresholds
        sell_name = names[1] if len(names) > 1 else 'sell_signal'
        df[sell_name] = (
            (df[score1] <= sell_thresh1) & (df[score2] <= sell_thresh2)
        ).astype(int)
        signals.append(sell_name)
        
        return signals
    
    def _generate_crossover_signals(
        self,
        df: pd.DataFrame,
        config: Dict
    ) -> List[str]:
        """
        Generate signals based on score crossovers.
        
        Config:
        {
            "columns": "trade_score",
            "names": ["buy_signal", "sell_signal"],
            "parameters": {
                "fast_window": 5,
                "slow_window": 20
            }
        }
        """
        signals = []
        
        columns = config.get('columns', 'trade_score')
        if isinstance(columns, list):
            columns = columns[0]
            
        names = config.get('names', ['buy_signal', 'sell_signal'])
        params = config.get('parameters', {})
        
        fast_window = params.get('fast_window', 5)
        slow_window = params.get('slow_window', 20)
        
        if columns not in df.columns:
            return signals
            
        # Calculate fast and slow EMAs of the score
        fast_ema = df[columns].ewm(span=fast_window, adjust=False).mean()
        slow_ema = df[columns].ewm(span=slow_window, adjust=False).mean()
        
        # Generate crossover signals
        buy_name = names[0] if len(names) > 0 else 'buy_signal'
        sell_name = names[1] if len(names) > 1 else 'sell_signal'
        
        # Buy: fast crosses above slow
        df[buy_name] = ((fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))).astype(int)
        
        # Sell: fast crosses below slow
        df[sell_name] = ((fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))).astype(int)
        
        signals.extend([buy_name, sell_name])
        
        return signals
    
    def _generate_divergence_signals(
        self,
        df: pd.DataFrame,
        config: Dict
    ) -> List[str]:
        """
        Generate signals based on price-score divergence.
        
        Config:
        {
            "columns": ["close", "trade_score"],
            "names": ["buy_signal", "sell_signal"],
            "parameters": {
                "lookback": 14
            }
        }
        """
        signals = []
        
        columns = config.get('columns', ['close', 'trade_score'])
        if len(columns) < 2:
            return signals
            
        price_col = columns[0]
        score_col = columns[1]
        
        if price_col not in df.columns or score_col not in df.columns:
            return signals
            
        names = config.get('names', ['buy_signal', 'sell_signal'])
        params = config.get('parameters', {})
        lookback = params.get('lookback', 14)
        
        price = df[price_col]
        score = df[score_col]
        
        buy_name = names[0] if len(names) > 0 else 'buy_signal'
        sell_name = names[1] if len(names) > 1 else 'sell_signal'
        
        # Initialize signals
        df[buy_name] = 0
        df[sell_name] = 0
        
        # Detect divergence
        for i in range(lookback, len(df)):
            window_price = price.iloc[i-lookback:i+1]
            window_score = score.iloc[i-lookback:i+1]
            
            # Bullish divergence: price lower low, score higher low
            price_at_low = price.iloc[i] <= window_price.min() * 1.001
            score_at_higher = score.iloc[i] > window_score.min()
            
            if price_at_low and score_at_higher:
                df.loc[df.index[i], buy_name] = 1
                
            # Bearish divergence: price higher high, score lower high
            price_at_high = price.iloc[i] >= window_price.max() * 0.999
            score_at_lower = score.iloc[i] < window_score.max()
            
            if price_at_high and score_at_lower:
                df.loc[df.index[i], sell_name] = 1
                
        signals.extend([buy_name, sell_name])
        
        return signals
    
    def apply_signal_rules(
        self,
        df: pd.DataFrame,
        buy_score: str,
        sell_score: str,
        buy_threshold: float = 0.08,
        sell_threshold: float = -0.08
    ) -> pd.DataFrame:
        """
        Apply standard signal rules to scores.
        
        Quick method for generating signals from two score columns.
        """
        df = df.copy()
        
        # Combine scores
        df['trade_score'] = df[buy_score] - df[sell_score]
        
        # Generate signals
        df['buy_signal'] = (df['trade_score'] >= buy_threshold).astype(int)
        df['sell_signal'] = (df['trade_score'] <= sell_threshold).astype(int)
        
        return df
    
    def filter_signals(
        self,
        df: pd.DataFrame,
        buy_signal_col: str = 'buy_signal',
        sell_signal_col: str = 'sell_signal',
        min_bars_between_signals: int = 5,
        confirm_with_trend: bool = True,
        trend_col: str = 'ema_200'
    ) -> pd.DataFrame:
        """
        Filter signals to reduce noise and false triggers.
        
        Args:
            df: DataFrame with signals
            buy_signal_col: Column name for buy signals
            sell_signal_col: Column name for sell signals
            min_bars_between_signals: Minimum bars between consecutive signals
            confirm_with_trend: Require trend confirmation
            trend_col: Column to use for trend (price should be above for buy, below for sell)
        """
        df = df.copy()
        
        # Filter by minimum bars between signals
        last_signal_idx = -min_bars_between_signals
        
        for i in range(len(df)):
            if df[buy_signal_col].iloc[i] == 1 or df[sell_signal_col].iloc[i] == 1:
                if i - last_signal_idx < min_bars_between_signals:
                    # Too close to previous signal, filter out
                    df.loc[df.index[i], buy_signal_col] = 0
                    df.loc[df.index[i], sell_signal_col] = 0
                else:
                    last_signal_idx = i
                    
        # Confirm with trend
        if confirm_with_trend and trend_col in df.columns:
            # Buy signals only allowed when above EMA200
            df.loc[df['close'] < df[trend_col], buy_signal_col] = 0
            # Sell signals only allowed when below EMA200
            df.loc[df['close'] > df[trend_col], sell_signal_col] = 0
            
        return df
    
    def get_signal_strength(
        self,
        score: float,
        thresholds: List[Tuple[float, str]] = None
    ) -> Tuple[str, float]:
        """
        Convert score to signal strength category.
        
        Args:
            score: The trade score
            thresholds: List of (threshold, description) tuples
            
        Returns:
            Tuple of (description, absolute_score)
        """
        if thresholds is None:
            thresholds = [
                (0.15, 'STRONG_BUY'),
                (0.08, 'BUY'),
                (0.03, 'WEAK_BUY'),
                (-0.03, 'NEUTRAL'),
                (-0.08, 'WEAK_SELL'),
                (-0.15, 'SELL'),
            ]
            
        for threshold, description in thresholds:
            if score >= threshold:
                return description, abs(score)
                
        return 'STRONG_SELL', abs(score)
    
    def get_default_signal_sets(self) -> List[Dict]:
        """Get default signal set configurations."""
        return [
            {
                'generator': 'combine',
                'config': {
                    'columns': ['high_score', 'low_score'],
                    'names': 'trade_score',
                    'combine': 'difference',
                    'coefficient': 1.0,
                    'constant': 0.0
                }
            },
            {
                'generator': 'threshold_rule',
                'config': {
                    'columns': 'trade_score',
                    'names': ['buy_signal', 'sell_signal'],
                    'parameters': {
                        'buy_signal_threshold': 0.08,
                        'sell_signal_threshold': -0.08
                    }
                }
            }
        ]
