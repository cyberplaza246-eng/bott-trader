"""
Label Generator Module — Intelligent Trading System

Generates training labels from future price data:
  - High/Low threshold labels (price crosses threshold)
  - Top/Bottom extremum labels (local maxima/minima detection)
  - Future aggregation labels (max high, min low in horizon)
  - First crossing labels (first time price crosses threshold)
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

from src.utils.logger import bot_logger


class LabelGenerator:
    """
    Generate training labels from OHLCV data using future price information.
    
    Label types:
      - 'highlow': Binary labels based on future high/low thresholds
      - 'topbot': Labels for local extremums (tops and bottoms)
      - 'direction': Simple up/down direction labels
      - 'first_cross': Time to first threshold crossing
    """
    
    def generate_labels(
        self,
        df: pd.DataFrame,
        label_sets: List[Dict] = None,
        horizon: int = 60,
        label_type: str = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Generate labels according to label set specifications.
        
        Args:
            df: DataFrame with OHLCV data
            label_sets: List of label generator configurations (optional)
            horizon: Default future horizon for label computation
            label_type: Simple label type ('highlow', 'topbot', 'direction', 'first_cross')
                       If provided, creates a default label_set
            
        Returns:
            Tuple of (df_with_labels, label_column_names)
        """
        # Handle simple label_type parameter
        if label_type is not None and label_sets is None:
            label_sets = [{
                'generator': label_type,
                'config': {
                    'columns': ['close', 'high', 'low'],
                    'horizon': horizon,
                    'thresholds': [1.0],
                    'tolerance': 0.15
                }
            }]
        elif label_sets is None:
            # Default label sets
            label_sets = [{
                'generator': 'direction',
                'config': {
                    'columns': ['close'],
                    'horizon': horizon
                }
            }]
        
        df = df.copy()
        all_labels = []
        
        for ls in label_sets:
            generator = ls.get('generator', 'highlow')
            config = ls.get('config', {})
            
            if 'horizon' not in config:
                config['horizon'] = horizon
                
            if generator == 'highlow' or generator == 'highlow2':
                labels_df, labels = self._generate_highlow_labels(df, config)
            elif generator == 'topbot' or generator == 'topbot2':
                labels_df, labels = self._generate_topbot_labels(df, config)
            elif generator == 'direction':
                labels_df, labels = self._generate_direction_labels(df, config)
            elif generator == 'first_cross':
                labels_df, labels = self._generate_first_cross_labels(df, config)
            else:
                bot_logger.warning(f"Unknown label generator: {generator}")
                continue
                
            # Merge labels into main dataframe
            for col in labels:
                df[col] = labels_df[col]
            all_labels.extend(labels)
            
        return df, all_labels
    
    def _generate_highlow_labels(
        self,
        df: pd.DataFrame,
        config: Dict
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Generate high/low threshold crossing labels.
        
        Labels indicate whether price will cross a threshold within the horizon.
        
        Config:
        {
            "columns": ["close", "high", "low"],  # reference, high, low columns
            "function": "high" or "low",
            "thresholds": [1.0, 1.5, 2.0],  # percentage thresholds
            "tolerance": 0.2,  # tolerance factor
            "horizon": 60,  # future candles to check
            "names": ["high_10", "high_15", "high_20"]  # output column names
        }
        """
        labels_df = pd.DataFrame(index=df.index)
        labels = []
        
        columns = config.get('columns', ['close', 'high', 'low'])
        close_col = columns[0] if len(columns) > 0 else 'close'
        high_col = columns[1] if len(columns) > 1 else 'high'
        low_col = columns[2] if len(columns) > 2 else 'low'
        
        function = config.get('function', 'high')
        thresholds = config.get('thresholds', [1.0, 1.5, 2.0])
        if not isinstance(thresholds, list):
            thresholds = [thresholds]
            
        tolerance = config.get('tolerance', 0.2)
        horizon = config.get('horizon', 60)
        names = config.get('names', [f"{function}_{int(t*10)}" for t in thresholds])
        
        if len(names) != len(thresholds):
            names = [f"{function}_{int(t*10)}" for t in thresholds]
            
        for i, threshold in enumerate(thresholds):
            out_name = names[i]
            
            if function == 'high':
                # Check if future high exceeds threshold above close
                labels_df[out_name] = self._compute_high_cross_label(
                    df, close_col, high_col, low_col,
                    threshold, tolerance, horizon
                )
            else:  # function == 'low'
                # Check if future low falls below threshold under close
                labels_df[out_name] = self._compute_low_cross_label(
                    df, close_col, high_col, low_col,
                    threshold, tolerance, horizon
                )
                
            labels.append(out_name)
            
        return labels_df, labels
    
    def _compute_high_cross_label(
        self,
        df: pd.DataFrame,
        close_col: str,
        high_col: str,
        low_col: str,
        threshold: float,
        tolerance: float,
        horizon: int
    ) -> pd.Series:
        """
        Compute label for high threshold crossing.
        
        Returns 1 if price goes up >= threshold% before going down >= tolerance*threshold%
        """
        label = pd.Series(data=np.nan, index=df.index, dtype=float)
        
        close = df[close_col].values
        high = df[high_col].values
        low = df[low_col].values
        n = len(df)
        
        high_threshold_pct = threshold / 100.0
        low_threshold_pct = -threshold * tolerance / 100.0
        
        for i in range(n - horizon):
            ref_price = close[i]
            
            # Find max high and min low in future window
            future_highs = high[i+1:i+1+horizon]
            future_lows = low[i+1:i+1+horizon]
            
            max_high_pct = (np.max(future_highs) - ref_price) / ref_price
            min_low_pct = (np.min(future_lows) - ref_price) / ref_price
            
            # Find first crossing times
            high_cross_idx = np.where((future_highs - ref_price) / ref_price >= high_threshold_pct)[0]
            low_cross_idx = np.where((future_lows - ref_price) / ref_price <= low_threshold_pct)[0]
            
            first_high_cross = high_cross_idx[0] if len(high_cross_idx) > 0 else horizon + 1
            first_low_cross = low_cross_idx[0] if len(low_cross_idx) > 0 else horizon + 1
            
            # Label is 1 if price crosses high threshold before crossing low threshold
            if first_high_cross < first_low_cross and first_high_cross < horizon:
                label.iloc[i] = 1.0
            else:
                label.iloc[i] = 0.0
                
        return label
    
    def _compute_low_cross_label(
        self,
        df: pd.DataFrame,
        close_col: str,
        high_col: str,
        low_col: str,
        threshold: float,
        tolerance: float,
        horizon: int
    ) -> pd.Series:
        """
        Compute label for low threshold crossing.
        
        Returns 1 if price goes down >= threshold% before going up >= tolerance*threshold%
        """
        label = pd.Series(data=np.nan, index=df.index, dtype=float)
        
        close = df[close_col].values
        high = df[high_col].values
        low = df[low_col].values
        n = len(df)
        
        low_threshold_pct = -threshold / 100.0
        high_threshold_pct = threshold * tolerance / 100.0
        
        for i in range(n - horizon):
            ref_price = close[i]
            
            future_highs = high[i+1:i+1+horizon]
            future_lows = low[i+1:i+1+horizon]
            
            # Find first crossing times
            low_cross_idx = np.where((future_lows - ref_price) / ref_price <= low_threshold_pct)[0]
            high_cross_idx = np.where((future_highs - ref_price) / ref_price >= high_threshold_pct)[0]
            
            first_low_cross = low_cross_idx[0] if len(low_cross_idx) > 0 else horizon + 1
            first_high_cross = high_cross_idx[0] if len(high_cross_idx) > 0 else horizon + 1
            
            # Label is 1 if price crosses low threshold before crossing high threshold
            if first_low_cross < first_high_cross and first_low_cross < horizon:
                label.iloc[i] = 1.0
            else:
                label.iloc[i] = 0.0
                
        return label
    
    def _generate_topbot_labels(
        self,
        df: pd.DataFrame,
        config: Dict
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Generate top/bottom extremum labels.
        
        Labels indicate if the current price is near a local top or bottom.
        
        Config:
        {
            "columns": "close",
            "function": "top" or "bot",
            "level": 0.01,  # minimum jump level (1%)
            "tolerances": [0.25, 0.5, 0.75],  # tolerance as fraction of level
            "names": ["top1_025", "top1_05", "top1_075"]
        }
        """
        labels_df = pd.DataFrame(index=df.index)
        labels = []
        
        column = config.get('columns', 'close')
        if isinstance(column, list):
            column = column[0]
            
        function = config.get('function', 'top')
        level = config.get('level', 0.01)
        
        tolerances = config.get('tolerances', [0.25, 0.5, 0.75])
        if not isinstance(tolerances, list):
            tolerances = [tolerances]
            
        names = config.get('names', [f"{function}_{int(t*100)}" for t in tolerances])
        
        if len(names) != len(tolerances):
            names = [f"{function}_{int(t*100)}" for t in tolerances]
            
        prices = df[column].values
        
        # Find all extremums at the specified level
        if function == 'top':
            extremums = self._find_extremums(prices, True, level)
        else:
            extremums = self._find_extremums(prices, False, level)
            
        for i, tolerance in enumerate(tolerances):
            out_name = names[i]
            tolerance_frac = abs(level) * tolerance
            
            # Create label column
            label = pd.Series(data=0.0, index=df.index, dtype=float)
            
            for ext_idx in extremums:
                ext_price = prices[ext_idx]
                
                # Mark all points within tolerance of this extremum
                if function == 'top':
                    # Points close to top (within tolerance_frac below)
                    threshold_price = ext_price * (1 - tolerance_frac)
                    in_zone = np.where(prices >= threshold_price)[0]
                else:
                    # Points close to bottom (within tolerance_frac above)
                    threshold_price = ext_price * (1 + tolerance_frac)
                    in_zone = np.where(prices <= threshold_price)[0]
                    
                # Mark only points near this extremum (within local window)
                window = int(len(prices) * 0.05)  # 5% of data
                mask = (in_zone >= ext_idx - window) & (in_zone <= ext_idx + window)
                relevant_indices = in_zone[mask]
                
                for idx in relevant_indices:
                    label.iloc[idx] = 1.0
                    
            labels_df[out_name] = label
            labels.append(out_name)
            
        return labels_df, labels
    
    def _find_extremums(
        self,
        prices: np.ndarray,
        find_max: bool,
        level_frac: float
    ) -> List[int]:
        """
        Find all local extremums (maxima or minima) at the specified level.
        
        Level fraction determines the minimum required jump on both sides.
        """
        extremums = []
        n = len(prices)
        
        if n < 10:
            return extremums
            
        level = abs(level_frac)
        
        for i in range(1, n - 1):
            price = prices[i]
            
            if find_max:
                # For maximum: need price to be <= price * (1 - level) on both sides
                threshold = price * (1 - level)
                
                # Look left for lower price
                left_ok = False
                for j in range(i - 1, max(0, i - 100), -1):
                    if prices[j] <= threshold:
                        left_ok = True
                        break
                        
                # Look right for lower price
                right_ok = False
                for j in range(i + 1, min(n, i + 100)):
                    if prices[j] <= threshold:
                        right_ok = True
                        break
                        
                if left_ok and right_ok:
                    extremums.append(i)
            else:
                # For minimum: need price to be >= price * (1 + level) on both sides
                threshold = price * (1 + level)
                
                left_ok = False
                for j in range(i - 1, max(0, i - 100), -1):
                    if prices[j] >= threshold:
                        left_ok = True
                        break
                        
                right_ok = False
                for j in range(i + 1, min(n, i + 100)):
                    if prices[j] >= threshold:
                        right_ok = True
                        break
                        
                if left_ok and right_ok:
                    extremums.append(i)
                    
        return extremums
    
    def _generate_direction_labels(
        self,
        df: pd.DataFrame,
        config: Dict
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Generate simple direction labels (up/down/flat).
        
        Config:
        {
            "columns": "close",
            "horizon": 60,
            "threshold": 0.001,  # minimum change for up/down (0.1%)
            "names": ["direction_60"]
        }
        """
        labels_df = pd.DataFrame(index=df.index)
        labels = []
        
        column = config.get('columns', 'close')
        if isinstance(column, list):
            column = column[0]
            
        horizon = config.get('horizon', 60)
        threshold = config.get('threshold', 0.001)
        names = config.get('names', [f"direction_{horizon}"])
        out_name = names[0] if names else f"direction_{horizon}"
        
        close = df[column]
        
        # Calculate future returns
        future_return = close.shift(-horizon) / close - 1
        
        # Create direction labels: 1 = up, 0 = flat, -1 = down
        label = pd.Series(data=0.0, index=df.index, dtype=float)
        label[future_return > threshold] = 1.0
        label[future_return < -threshold] = -1.0
        
        labels_df[out_name] = label
        labels.append(out_name)
        
        # Also create binary labels
        up_name = f"{out_name}_up"
        down_name = f"{out_name}_down"
        
        labels_df[up_name] = (label == 1.0).astype(float)
        labels_df[down_name] = (label == -1.0).astype(float)
        
        labels.extend([up_name, down_name])
        
        return labels_df, labels
    
    def _generate_first_cross_labels(
        self,
        df: pd.DataFrame,
        config: Dict
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Generate first crossing time labels.
        
        Returns the number of candles until price first crosses the threshold.
        """
        labels_df = pd.DataFrame(index=df.index)
        labels = []
        
        column = config.get('columns', 'close')
        if isinstance(column, list):
            column = column[0]
            
        horizon = config.get('horizon', 60)
        thresholds = config.get('thresholds', [0.5, 1.0, 1.5])
        if not isinstance(thresholds, list):
            thresholds = [thresholds]
            
        names = config.get('names', [f"first_cross_{int(t*10)}" for t in thresholds])
        
        close = df[column].values
        n = len(close)
        
        for i, threshold in enumerate(thresholds):
            out_name = names[i] if i < len(names) else f"first_cross_{int(threshold*10)}"
            threshold_pct = threshold / 100.0
            
            label = pd.Series(data=np.nan, index=df.index, dtype=float)
            
            for j in range(n - horizon):
                ref_price = close[j]
                future_prices = close[j+1:j+1+horizon]
                
                # Find first crossing
                pct_changes = (future_prices - ref_price) / ref_price
                
                up_cross = np.where(pct_changes >= threshold_pct)[0]
                down_cross = np.where(pct_changes <= -threshold_pct)[0]
                
                first_up = up_cross[0] if len(up_cross) > 0 else horizon + 1
                first_down = down_cross[0] if len(down_cross) > 0 else horizon + 1
                
                first_cross = min(first_up, first_down)
                
                if first_cross <= horizon:
                    # Normalize to [0, 1] where 0 = immediate, 1 = at horizon
                    label.iloc[j] = first_cross / horizon
                else:
                    # No crossing within horizon
                    label.iloc[j] = 1.0
                    
            labels_df[out_name] = label
            labels.append(out_name)
            
        return labels_df, labels
    
    def get_default_label_sets(
        self,
        timeframe: str = '1m',
        horizon: Optional[int] = None
    ) -> List[Dict]:
        """Get default label set configurations."""
        if horizon is None:
            if timeframe == '1m':
                horizon = 60  # 1 hour
            elif timeframe == '5m':
                horizon = 24  # 2 hours
            elif timeframe == '1h':
                horizon = 24  # 1 day
            else:
                horizon = 60
                
        return [
            {
                'generator': 'highlow',
                'config': {
                    'columns': ['close', 'high', 'low'],
                    'function': 'high',
                    'thresholds': [0.5, 1.0, 1.5],
                    'tolerance': 0.2,
                    'horizon': horizon,
                    'names': ['high_05', 'high_10', 'high_15']
                }
            },
            {
                'generator': 'highlow',
                'config': {
                    'columns': ['close', 'high', 'low'],
                    'function': 'low',
                    'thresholds': [0.5, 1.0, 1.5],
                    'tolerance': 0.2,
                    'horizon': horizon,
                    'names': ['low_05', 'low_10', 'low_15']
                }
            },
            {
                'generator': 'direction',
                'config': {
                    'columns': 'close',
                    'horizon': horizon,
                    'threshold': 0.001,
                    'names': [f'direction_{horizon}']
                }
            }
        ]
