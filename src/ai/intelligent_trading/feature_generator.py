"""
Feature Generator Module — Intelligent Trading System

Generates derived features from OHLCV data using:
  - TA-Lib technical indicators (SMA, RSI, MACD, etc.)
  - Statistical features (skewness, kurtosis, autocorrelation)
  - Rolling aggregations (mean, std, slope, area)
  - Custom ITB statistical features
  - Relative and percentage transformations
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional
import importlib
import sys

from src.utils.logger import bot_logger


class FeatureGenerator:
    """
    Advanced feature generator supporting multiple generation strategies.
    
    Supported generators:
      - 'talib': TA-Lib technical indicators
      - 'itbstats': Custom statistical features 
      - 'pandas_ta': pandas-ta indicators (fallback if talib unavailable)
      - 'rolling': Rolling window aggregations
    """
    
    def __init__(self):
        self.talib_available = self._check_talib()
        self.tsfresh_available = self._check_tsfresh()
        
    def _check_talib(self) -> bool:
        """Check if TA-Lib is available."""
        try:
            import talib
            return True
        except ImportError:
            bot_logger.warning("TA-Lib not available, using pandas_ta fallback")
            return False
    
    def _check_tsfresh(self) -> bool:
        """Check if tsfresh is available."""
        try:
            import tsfresh
            return True
        except ImportError:
            return False
            
    def _get_default_feature_sets(self) -> List[Dict]:
        """Get default feature set configurations."""
        return [
            {
                'generator': 'talib',
                'config': {
                    'columns': ['close'],
                    'functions': ['SMA', 'EMA', 'RSI'],
                    'windows': [5, 10, 20, 50]
                }
            },
            {
                'generator': 'itbstats',
                'config': {
                    'columns': ['close'],
                    'functions': ['zscore', 'slope'],
                    'windows': [10, 20]
                }
            },
            {
                'generator': 'rolling',
                'config': {
                    'columns': ['close', 'volume'],
                    'functions': ['mean', 'std'],
                    'windows': [5, 10, 20]
                }
            }
        ]
            
    def generate_features(
        self, 
        df: pd.DataFrame, 
        feature_sets: List[Dict] = None, 
        last_rows: int = 0
    ) -> tuple:
        """
        Generate features according to feature set specifications.
        
        Args:
            df: DataFrame with OHLCV data
            feature_sets: List of feature generator configurations (uses defaults if None)
            last_rows: Only compute for this many last rows (0 = all)
            
        Returns:
            Tuple of (df_with_features, feature_column_names)
        """
        if feature_sets is None:
            feature_sets = self._get_default_feature_sets()
            
        df = df.copy()
        all_features = []
        
        for fs in feature_sets:
            generator = fs.get('generator', 'talib')
            config = fs.get('config', {})
            
            if generator == 'talib':
                features = self._generate_talib_features(df, config, last_rows)
            elif generator == 'itbstats':
                features = self._generate_statistical_features(df, config, last_rows)
            elif generator == 'rolling':
                features = self._generate_rolling_features(df, config, last_rows)
            elif generator == 'pandas_ta':
                features = self._generate_pandas_ta_features(df, config, last_rows)
            else:
                bot_logger.warning(f"Unknown feature generator: {generator}")
                features = []
                
            all_features.extend(features)
            
        return df, all_features
    
    def _generate_talib_features(
        self,
        df: pd.DataFrame,
        config: Dict,
        last_rows: int = 0
    ) -> List[str]:
        """
        Generate TA-Lib technical indicator features.
        
        Config format:
        {
            "columns": ["close"],
            "functions": ["SMA", "RSI", "MACD"],
            "windows": [5, 10, 20],
            "parameters": {
                "rel_base": "last",    # relative to last value
                "rel_func": "rel_diff", # relative difference
                "percentage": true      # multiply by 100
            }
        }
        """
        features = []
        
        # Parse config
        columns = config.get('columns', ['close'])
        if isinstance(columns, str):
            columns = [columns]
            
        functions = config.get('functions', ['SMA'])
        if isinstance(functions, str):
            functions = [functions]
            
        windows = config.get('windows', [14])
        if isinstance(windows, int):
            windows = [windows]
            
        params = config.get('parameters', {})
        rel_base = params.get('rel_base', False)
        rel_func = params.get('rel_func', 'rel_diff')
        percentage = params.get('percentage', False)
        
        # Get the library to use
        if self.talib_available:
            import talib
            lib = talib
        else:
            import pandas_ta as pta
            lib = pta
            
        for func_name in functions:
            func_outputs = []
            func_names = []
            
            for col in columns:
                if col not in df.columns:
                    bot_logger.warning(f"Column {col} not found in DataFrame")
                    continue
                    
                series = df[col].interpolate()
                
                for w in windows:
                    try:
                        if self.talib_available:
                            # TA-Lib function call
                            out = self._call_talib_function(lib, func_name, series, w, df)
                        else:
                            # pandas_ta fallback
                            out = self._call_pandas_ta_function(func_name, df, col, w)
                            
                        out_name = f"{col}_{func_name}_{w}"
                        if out is not None:
                            df[out_name] = out
                            func_outputs.append(out)
                            func_names.append(out_name)
                            features.append(out_name)
                            
                    except Exception as e:
                        bot_logger.warning(f"Error generating {func_name}_{w}: {e}")
                        continue
                        
            # Convert to relative values if requested
            if rel_base and len(func_outputs) > 1:
                func_outputs = self._convert_to_relative(
                    func_outputs, func_names, rel_base, rel_func, percentage
                )
                for i, out in enumerate(func_outputs):
                    if isinstance(out, pd.Series):
                        df[func_names[i]] = out
                        
        return features
    
    def _call_talib_function(
        self, 
        lib, 
        func_name: str, 
        series: pd.Series, 
        window: int,
        df: pd.DataFrame
    ) -> Optional[pd.Series]:
        """Call a TA-Lib function with appropriate parameters."""
        try:
            func = getattr(lib, func_name)
        except AttributeError:
            bot_logger.warning(f"TA-Lib function {func_name} not found")
            return None
            
        # Handle functions with different signatures
        try:
            if func_name in ['RSI', 'SMA', 'EMA', 'DEMA', 'TEMA', 'WMA', 
                             'TRIMA', 'KAMA', 'MOM', 'ROC', 'ROCP', 'ROCR']:
                return func(series, timeperiod=window)
            elif func_name in ['LINEARREG', 'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT',
                               'LINEARREG_SLOPE', 'TSF']:
                return func(series, timeperiod=window)
            elif func_name in ['STDDEV', 'VAR']:
                return func(series, timeperiod=window, nbdev=1)
            elif func_name in ['ATR', 'NATR', 'TRANGE']:
                return func(df['high'], df['low'], df['close'], timeperiod=window)
            elif func_name in ['ADX', 'ADXR', 'CCI', 'DX', 'MINUS_DI', 'PLUS_DI',
                               'WILLR', 'MINUS_DM', 'PLUS_DM']:
                return func(df['high'], df['low'], df['close'], timeperiod=window)
            elif func_name in ['MACD', 'MACDEXT', 'MACDFIX']:
                macd, macd_signal, macd_hist = func(series, fastperiod=12, 
                                                     slowperiod=26, signalperiod=9)
                return macd  # Return main MACD line
            elif func_name in ['BBANDS']:
                upper, middle, lower = func(series, timeperiod=window)
                return middle  # Return middle band
            elif func_name in ['STOCH', 'STOCHF', 'STOCHRSI']:
                slowk, slowd = func(df['high'], df['low'], df['close'],
                                    fastk_period=window)
                return slowk
            else:
                # Try generic single-input function
                return func(series, timeperiod=window)
        except Exception as e:
            bot_logger.debug(f"Error calling {func_name}: {e}")
            return None
    
    def _call_pandas_ta_function(
        self,
        func_name: str,
        df: pd.DataFrame,
        col: str,
        window: int
    ) -> Optional[pd.Series]:
        """Call pandas_ta function as fallback."""
        import pandas_ta as pta
        
        func_name_lower = func_name.lower()
        
        try:
            if func_name_lower == 'sma':
                return pta.sma(df[col], length=window)
            elif func_name_lower == 'ema':
                return pta.ema(df[col], length=window)
            elif func_name_lower == 'rsi':
                return pta.rsi(df[col], length=window)
            elif func_name_lower in ['linearreg_slope', 'slope']:
                return pta.slope(df[col], length=window)
            elif func_name_lower == 'stddev':
                return df[col].rolling(window).std()
            elif func_name_lower == 'mom':
                return pta.mom(df[col], length=window)
            elif func_name_lower == 'roc':
                return pta.roc(df[col], length=window)
            elif func_name_lower == 'willr':
                return pta.willr(df['high'], df['low'], df['close'], length=window)
            elif func_name_lower == 'cci':
                return pta.cci(df['high'], df['low'], df['close'], length=window)
            elif func_name_lower == 'atr':
                return pta.atr(df['high'], df['low'], df['close'], length=window)
            elif func_name_lower == 'adx':
                adx_df = pta.adx(df['high'], df['low'], df['close'], length=window)
                if adx_df is not None and len(adx_df.columns) > 0:
                    return adx_df.iloc[:, 0]
            elif func_name_lower == 'macd':
                # MACD returns multiple columns, take the main MACD line
                macd_df = pta.macd(df[col], fast=12, slow=26, signal=9)
                if macd_df is not None and len(macd_df.columns) > 0:
                    # Return the MACD line (first column typically MACD_12_26_9)
                    return macd_df.iloc[:, 0]
            elif func_name_lower == 'bbands':
                # Bollinger Bands returns multiple columns, take the middle band
                bb_df = pta.bbands(df[col], length=window)
                if bb_df is not None and len(bb_df.columns) >= 2:
                    # BBM is typically the middle band
                    mid_col = [c for c in bb_df.columns if 'BBM' in c]
                    if mid_col:
                        return bb_df[mid_col[0]]
                    return bb_df.iloc[:, 1]  # Fallback to second column
            elif func_name_lower == 'stoch':
                # Stochastic returns K and D lines, take K
                stoch_df = pta.stoch(df['high'], df['low'], df['close'], k=window)
                if stoch_df is not None and len(stoch_df.columns) > 0:
                    return stoch_df.iloc[:, 0]
            else:
                # Try generic call
                func = getattr(pta, func_name_lower, None)
                if func:
                    result = func(df[col], length=window)
                    # Handle DataFrame result (multi-column output)
                    if isinstance(result, pd.DataFrame):
                        if len(result.columns) > 0:
                            return result.iloc[:, 0]
                        return None
                    return result
        except Exception as e:
            bot_logger.debug(f"pandas_ta error for {func_name}: {e}")
            
        return None
    
    def _generate_statistical_features(
        self,
        df: pd.DataFrame,
        config: Dict,
        last_rows: int = 0
    ) -> List[str]:
        """
        Generate statistical features (ITB stats).
        
        Includes: skewness, kurtosis, mean_second_derivative, 
                  longest_strike_below_mean, first_location_of_maximum,
                  mean, std, slope, area
        """
        from scipy import stats
        
        features = []
        
        columns = config.get('columns', ['close'])
        if isinstance(columns, str):
            columns = [columns]
            
        windows = config.get('windows', [14])
        if isinstance(windows, int):
            windows = [windows]
            
        functions = config.get('functions', [
            'skew', 'kurtosis', 'mean', 'std', 'slope', 'area'
        ])
        
        for col in columns:
            if col not in df.columns:
                continue
                
            series = df[col].interpolate()
            
            for w in windows:
                ro = series.rolling(window=w, min_periods=max(1, w // 2))
                
                for func in functions:
                    out_name = f"{col}_{func}_{w}"
                    
                    try:
                        if func == 'skew':
                            df[out_name] = ro.skew()
                        elif func == 'kurtosis':
                            df[out_name] = ro.kurt()
                        elif func == 'mean':
                            df[out_name] = ro.mean()
                        elif func == 'std':
                            df[out_name] = ro.std()
                        elif func == 'slope':
                            df[out_name] = self._rolling_slope(series, w)
                        elif func == 'area':
                            df[out_name] = ro.sum()
                        elif func == 'zscore':
                            mean_val = ro.mean()
                            std_val = ro.std()
                            df[out_name] = (series - mean_val) / (std_val + 1e-10)
                        elif func == 'lsbm':
                            # Longest strike below mean
                            df[out_name] = self._rolling_lsbm(series, w)
                        elif func == 'fmax':
                            # First location of maximum
                            df[out_name] = self._rolling_fmax(series, w)
                        elif func == 'msdc':
                            # Mean second derivative central
                            df[out_name] = self._rolling_msdc(series, w)
                        else:
                            continue
                            
                        features.append(out_name)
                        
                    except Exception as e:
                        bot_logger.debug(f"Error generating {out_name}: {e}")
                        
        return features
    
    def _rolling_slope(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling linear regression slope."""
        from scipy import stats as scipy_stats
        
        def slope_fn(x):
            if len(x) < 2:
                return np.nan
            x_vals = np.arange(len(x))
            slope, _, _, _, _ = scipy_stats.linregress(x_vals, x)
            return slope
            
        return series.rolling(window, min_periods=max(2, window // 2)).apply(slope_fn, raw=True)
    
    def _rolling_lsbm(self, series: pd.Series, window: int) -> pd.Series:
        """Longest strike below mean."""
        def lsbm_fn(x):
            mean = np.mean(x)
            below = x < mean
            longest = 0
            current = 0
            for b in below:
                if b:
                    current += 1
                    longest = max(longest, current)
                else:
                    current = 0
            return longest
            
        return series.rolling(window, min_periods=window // 2).apply(lsbm_fn, raw=True)
    
    def _rolling_fmax(self, series: pd.Series, window: int) -> pd.Series:
        """First location of maximum (as fraction of window)."""
        def fmax_fn(x):
            if len(x) == 0:
                return np.nan
            return np.argmax(x) / len(x)
            
        return series.rolling(window, min_periods=window // 2).apply(fmax_fn, raw=True)
    
    def _rolling_msdc(self, series: pd.Series, window: int) -> pd.Series:
        """Mean second derivative central."""
        def msdc_fn(x):
            if len(x) < 3:
                return np.nan
            return (x[-1] - x[-2] - x[-2] + x[-3]) / 2.0
            
        return series.rolling(window, min_periods=3).apply(msdc_fn, raw=True)
    
    def _generate_rolling_features(
        self,
        df: pd.DataFrame,
        config: Dict,
        last_rows: int = 0
    ) -> List[str]:
        """Generate rolling window aggregation features."""
        features = []
        
        columns = config.get('columns', ['close'])
        if isinstance(columns, str):
            columns = [columns]
            
        windows = config.get('windows', [5, 10, 20])
        if isinstance(windows, int):
            windows = [windows]
            
        functions = config.get('functions', ['mean', 'std', 'min', 'max'])
        
        for col in columns:
            if col not in df.columns:
                continue
                
            series = df[col].interpolate()
            
            for w in windows:
                ro = series.rolling(window=w, min_periods=max(1, w // 2))
                
                for func in functions:
                    out_name = f"{col}_{func}_{w}"
                    
                    try:
                        if func == 'mean':
                            df[out_name] = ro.mean()
                        elif func == 'std':
                            df[out_name] = ro.std()
                        elif func == 'min':
                            df[out_name] = ro.min()
                        elif func == 'max':
                            df[out_name] = ro.max()
                        elif func == 'sum':
                            df[out_name] = ro.sum()
                        elif func == 'median':
                            df[out_name] = ro.median()
                        elif func == 'var':
                            df[out_name] = ro.var()
                        elif func == 'range':
                            df[out_name] = ro.max() - ro.min()
                        elif func == 'pct_change':
                            df[out_name] = series.pct_change(periods=w) * 100
                        else:
                            continue
                            
                        features.append(out_name)
                        
                    except Exception as e:
                        bot_logger.debug(f"Error generating {out_name}: {e}")
                        
        return features
    
    def _generate_pandas_ta_features(
        self,
        df: pd.DataFrame,
        config: Dict,
        last_rows: int = 0
    ) -> List[str]:
        """Generate features using pandas_ta directly."""
        import pandas_ta as pta
        
        features = []
        
        # Get full technical analysis strategy results
        strategy = config.get('strategy', 'momentum')
        
        if strategy == 'all':
            # Generate all available indicators
            df.ta.strategy("All")
            features = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
        elif strategy == 'momentum':
            df.ta.strategy("Momentum")  
            features = [c for c in df.columns if c.startswith(('RSI', 'MOM', 'ROC', 'STOCH', 'WILLR'))]
        elif strategy == 'volatility':
            df.ta.strategy("Volatility")
            features = [c for c in df.columns if c.startswith(('ATR', 'BB', 'KC', 'NATR'))]
        elif strategy == 'trend':
            df.ta.strategy("Trend")
            features = [c for c in df.columns if c.startswith(('SMA', 'EMA', 'ADX', 'AROON'))]
            
        return features
    
    def _convert_to_relative(
        self,
        outputs: List[pd.Series],
        names: List[str],
        rel_base: str,
        rel_func: str,
        percentage: bool
    ) -> List[pd.Series]:
        """Convert feature values to relative values."""
        converted = []
        size = len(outputs)
        
        for i, feature in enumerate(outputs):
            if rel_base == 'next' and i == size - 1:
                converted.append(feature)
            elif rel_base == 'prev' and i == 0:
                converted.append(feature)
            elif rel_base in ['next', 'last']:
                base = outputs[i + 1] if rel_base == 'next' else outputs[-1]
                
                if rel_func == 'rel':
                    result = feature / (base + 1e-10)
                elif rel_func == 'diff':
                    result = feature - base
                elif rel_func == 'rel_diff':
                    result = (feature - base) / (base + 1e-10)
                else:
                    result = feature
                    
                if percentage:
                    result = result * 100
                    
                result.name = names[i]
                converted.append(result)
            elif rel_base in ['prev', 'first']:
                base = outputs[i - 1] if rel_base == 'prev' else outputs[0]
                
                if rel_func == 'rel':
                    result = feature / (base + 1e-10)
                elif rel_func == 'diff':
                    result = feature - base
                elif rel_func == 'rel_diff':
                    result = (feature - base) / (base + 1e-10)
                else:
                    result = feature
                    
                if percentage:
                    result = result * 100
                    
                result.name = names[i]
                converted.append(result)
            else:
                converted.append(feature)
                
        return converted
    
    def get_default_feature_sets(self, timeframe: str = '1m') -> List[Dict]:
        """Get default feature set configurations for a timeframe."""
        if timeframe == '1m':
            windows = [5, 10, 20, 50, 100]
        elif timeframe == '5m':
            windows = [6, 12, 24, 48, 96]
        elif timeframe == '1h':
            windows = [6, 12, 24, 168, 672]
        else:
            windows = [5, 10, 20, 50, 100]
            
        return [
            {
                'generator': 'talib',
                'config': {
                    'columns': ['close'],
                    'functions': ['SMA', 'EMA', 'RSI'],
                    'windows': windows
                }
            },
            {
                'generator': 'talib',
                'config': {
                    'columns': ['close'],
                    'functions': ['LINEARREG_SLOPE', 'MOM', 'ROC'],
                    'windows': windows[:3]
                }
            },
            {
                'generator': 'talib',
                'config': {
                    'columns': ['close'],
                    'functions': ['STDDEV'],
                    'windows': windows
                }
            },
            {
                'generator': 'itbstats',
                'config': {
                    'columns': ['close'],
                    'functions': ['skew', 'kurtosis', 'slope'],
                    'windows': windows[:3]
                }
            },
            {
                'generator': 'rolling',
                'config': {
                    'columns': ['volume'],
                    'functions': ['mean', 'std'],
                    'windows': windows[:3]
                }
            }
        ]
