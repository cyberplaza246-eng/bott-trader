"""
Backtesting Engine Module — Intelligent Trading System

Trade simulation and performance measurement:
  - Simulated trade execution
  - Long/short performance tracking
  - Grid search for optimal parameters
  - Rolling prediction backtesting
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from itertools import product

from src.utils.logger import bot_logger


class BacktestEngine:
    """
    Backtesting engine for trade simulation and performance analysis.
    
    Features:
      - Simulate trades based on buy/sell signals
      - Track long and short performance separately
      - Grid search for optimal signal parameters
      - Rolling window backtesting with model retraining
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Default settings
        self.commission = self.config.get('commission', 0.0001)  # 0.01% per trade
        self.slippage = self.config.get('slippage', 0.0001)  # 0.01% slippage
        self.initial_capital = self.config.get('initial_capital', 10000)
        
    def simulate_trades(
        self,
        df: pd.DataFrame,
        buy_signal_col: str = 'buy_signal',
        sell_signal_col: str = 'sell_signal',
        price_col: str = 'close',
        direction: str = 'both'  # 'long', 'short', or 'both'
    ) -> Dict[str, Any]:
        """
        Simulate trades over the time series.
        
        Args:
            df: DataFrame with signals and prices
            buy_signal_col: Column for buy signals
            sell_signal_col: Column for sell signals
            price_col: Column for execution price
            direction: Trading direction ('long', 'short', 'both')
            
        Returns:
            Dict with performance metrics
        """
        df = df[[sell_signal_col, buy_signal_col, price_col]].copy()
        
        # Track trades
        is_long = False
        is_short = False
        
        long_trades = []
        short_trades = []
        
        long_profit = 0.0
        long_profit_pct = 0.0
        long_count = 0
        long_wins = 0
        
        short_profit = 0.0
        short_profit_pct = 0.0
        short_count = 0
        short_wins = 0
        
        entry_price = 0.0
        entry_time = None
        
        for idx, row in df.iterrows():
            price = row[price_col]
            buy_signal = row[buy_signal_col]
            sell_signal = row[sell_signal_col]
            
            if pd.isna(price) or price <= 0:
                continue
                
            # Long trading logic
            if direction in ['long', 'both']:
                if buy_signal and not is_long:
                    # Enter long
                    is_long = True
                    entry_price = price * (1 + self.slippage + self.commission)
                    entry_time = idx
                    
                elif sell_signal and is_long:
                    # Exit long
                    exit_price = price * (1 - self.slippage - self.commission)
                    profit = exit_price - entry_price
                    profit_pct = (profit / entry_price) * 100
                    
                    long_trades.append({
                        'entry_time': entry_time,
                        'exit_time': idx,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'type': 'long'
                    })
                    
                    long_profit += profit
                    long_profit_pct += profit_pct
                    long_count += 1
                    if profit > 0:
                        long_wins += 1
                        
                    is_long = False
                    
            # Short trading logic
            if direction in ['short', 'both']:
                if sell_signal and not is_short:
                    # Enter short
                    is_short = True
                    entry_price = price * (1 - self.slippage - self.commission)
                    entry_time = idx
                    
                elif buy_signal and is_short:
                    # Exit short
                    exit_price = price * (1 + self.slippage + self.commission)
                    profit = entry_price - exit_price
                    profit_pct = (profit / entry_price) * 100
                    
                    short_trades.append({
                        'entry_time': entry_time,
                        'exit_time': idx,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'type': 'short'
                    })
                    
                    short_profit += profit
                    short_profit_pct += profit_pct
                    short_count += 1
                    if profit > 0:
                        short_wins += 1
                        
                    is_short = False
                    
        # Calculate performance metrics
        total_trades = long_count + short_count
        total_profit = long_profit + short_profit
        total_profit_pct = long_profit_pct + short_profit_pct
        total_wins = long_wins + short_wins
        
        performance = {
            'total': {
                'trades': total_trades,
                'profit': round(total_profit, 4),
                'profit_pct': round(total_profit_pct, 2),
                'wins': total_wins,
                'win_rate': round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0,
                'profit_per_trade': round(total_profit / total_trades, 4) if total_trades > 0 else 0,
                'profit_pct_per_trade': round(total_profit_pct / total_trades, 2) if total_trades > 0 else 0,
            },
            'long': {
                'trades': long_count,
                'profit': round(long_profit, 4),
                'profit_pct': round(long_profit_pct, 2),
                'wins': long_wins,
                'win_rate': round(long_wins / long_count * 100, 1) if long_count > 0 else 0,
            },
            'short': {
                'trades': short_count,
                'profit': round(short_profit, 4),
                'profit_pct': round(short_profit_pct, 2),
                'wins': short_wins,
                'win_rate': round(short_wins / short_count * 100, 1) if short_count > 0 else 0,
            },
            'trades': long_trades + short_trades
        }
        
        return performance
    
    def grid_search_parameters(
        self,
        df: pd.DataFrame,
        score_col: str,
        param_grid: Dict[str, List[float]],
        price_col: str = 'close',
        direction: str = 'long',
        metric: str = 'profit_pct'
    ) -> Tuple[Dict, List[Dict]]:
        """
        Grid search for optimal signal parameters.
        
        Args:
            df: DataFrame with scores and prices
            score_col: Column containing trade scores
            param_grid: Dict of parameter names to lists of values to try
            price_col: Column for execution price
            direction: Trading direction
            metric: Metric to optimize ('profit_pct', 'win_rate', 'sharpe')
            
        Returns:
            Tuple of (best_params, all_results)
        """
        results = []
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            
            # Apply parameters to generate signals
            df_test = df.copy()
            
            buy_thresh = params.get('buy_signal_threshold', 0.08)
            sell_thresh = params.get('sell_signal_threshold', -0.08)
            
            df_test['buy_signal'] = (df_test[score_col] >= buy_thresh).astype(int)
            df_test['sell_signal'] = (df_test[score_col] <= sell_thresh).astype(int)
            
            # Run simulation
            perf = self.simulate_trades(
                df_test,
                buy_signal_col='buy_signal',
                sell_signal_col='sell_signal',
                price_col=price_col,
                direction=direction
            )
            
            # Record result
            result = {
                'params': params.copy(),
                **perf['total']
            }
            results.append(result)
            
        # Find best parameters
        if not results:
            return {}, results
            
        # Sort by metric
        valid_results = [r for r in results if r['trades'] > 0]
        if not valid_results:
            return {}, results
            
        best_result = max(valid_results, key=lambda x: x.get(metric, 0))
        
        return best_result['params'], results
    
    def rolling_backtest(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        classifier,
        train_size: int,
        test_size: int,
        step_size: int,
        signal_generator=None,
        price_col: str = 'close'
    ) -> Dict[str, Any]:
        """
        Rolling window backtest with periodic model retraining.
        
        Args:
            df: DataFrame with features and labels
            feature_cols: Feature column names
            label_col: Label column name
            classifier: Classifier instance
            train_size: Number of rows for training
            test_size: Number of rows for testing
            step_size: Number of rows to step forward
            signal_generator: Optional signal generator
            price_col: Price column for simulation
            
        Returns:
            Dict with backtest results
        """
        results = []
        all_predictions = []
        
        n_rows = len(df)
        current_idx = train_size
        
        while current_idx + test_size <= n_rows:
            # Get train and test data
            train_start = max(0, current_idx - train_size)
            train_end = current_idx
            test_start = current_idx
            test_end = current_idx + test_size
            
            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]
            
            X_train = train_df[feature_cols].dropna()
            y_train = train_df.loc[X_train.index, label_col]
            
            X_test = test_df[feature_cols].dropna()
            y_test = test_df.loc[X_test.index, label_col]
            
            # Train model
            try:
                classifier.train(X_train, y_train)
                
                # Get predictions
                y_pred_proba = classifier.predict_proba(X_test)
                
                # Store predictions
                pred_df = pd.DataFrame({
                    'y_true': y_test,
                    'y_pred_proba': y_pred_proba
                }, index=X_test.index)
                pred_df['y_pred'] = (y_pred_proba >= 0.5).astype(int)
                
                all_predictions.append(pred_df)
                
                # If we have a signal generator, generate signals and simulate trades
                if signal_generator is not None:
                    # Create a dataframe for signal generation
                    signal_df = test_df.loc[X_test.index].copy()
                    signal_df['score'] = y_pred_proba
                    
                    # Generate signals
                    signal_df['buy_signal'] = (y_pred_proba >= 0.6).astype(int)
                    signal_df['sell_signal'] = (y_pred_proba <= 0.4).astype(int)
                    
                    # Simulate trades for this window
                    perf = self.simulate_trades(
                        signal_df,
                        buy_signal_col='buy_signal',
                        sell_signal_col='sell_signal',
                        price_col=price_col,
                        direction='long'
                    )
                    
                    results.append({
                        'period_start': test_df.index[0],
                        'period_end': test_df.index[-1],
                        **perf['total']
                    })
                    
            except Exception as e:
                bot_logger.warning(f"Error in rolling backtest window: {e}")
                
            current_idx += step_size
            
        # Combine all predictions
        if all_predictions:
            combined_predictions = pd.concat(all_predictions)
            
            # Calculate overall metrics
            from sklearn.metrics import accuracy_score, roc_auc_score
            
            overall_metrics = {
                'accuracy': accuracy_score(
                    combined_predictions['y_true'],
                    combined_predictions['y_pred']
                ),
            }
            
            try:
                overall_metrics['auc'] = roc_auc_score(
                    combined_predictions['y_true'],
                    combined_predictions['y_pred_proba']
                )
            except:
                overall_metrics['auc'] = 0.0
        else:
            overall_metrics = {}
            combined_predictions = pd.DataFrame()
            
        # Aggregate results
        if results:
            total_trades = sum(r.get('trades', 0) for r in results)
            total_profit = sum(r.get('profit', 0) for r in results)
            total_profit_pct = sum(r.get('profit_pct', 0) for r in results)
            total_wins = sum(r.get('wins', 0) for r in results)
            
            aggregate = {
                'total_trades': total_trades,
                'total_profit': total_profit,
                'total_profit_pct': total_profit_pct,
                'win_rate': total_wins / total_trades * 100 if total_trades > 0 else 0,
                'num_periods': len(results)
            }
        else:
            aggregate = {}
            
        return {
            'period_results': results,
            'aggregate': aggregate,
            'ml_metrics': overall_metrics,
            'predictions': combined_predictions
        }
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Sharpe ratio from returns series."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
            
        excess_returns = returns - (risk_free_rate / periods_per_year)
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
        
        return sharpe
    
    def calculate_max_drawdown(
        self,
        equity_curve: pd.Series
    ) -> Tuple[float, datetime, datetime]:
        """
        Calculate maximum drawdown.
        
        Returns:
            Tuple of (max_drawdown_pct, peak_date, trough_date)
        """
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        
        max_drawdown = drawdown.min()
        trough_idx = drawdown.idxmin()
        peak_idx = equity_curve.loc[:trough_idx].idxmax()
        
        return max_drawdown * 100, peak_idx, trough_idx
    
    def calculate_calmar_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        equity = (1 + returns).cumprod()
        
        annualized_return = returns.mean() * periods_per_year
        max_dd, _, _ = self.calculate_max_drawdown(equity)
        
        if max_dd == 0:
            return 0.0
            
        return annualized_return / abs(max_dd)
    
    def generate_equity_curve(
        self,
        trades: List[Dict],
        initial_capital: float = None
    ) -> pd.DataFrame:
        """
        Generate equity curve from list of trades.
        
        Args:
            trades: List of trade dictionaries
            initial_capital: Starting capital
            
        Returns:
            DataFrame with equity curve
        """
        if initial_capital is None:
            initial_capital = self.initial_capital
            
        if not trades:
            return pd.DataFrame()
            
        # Sort trades by exit time
        sorted_trades = sorted(trades, key=lambda x: x['exit_time'])
        
        equity = initial_capital
        equity_points = []
        
        for trade in sorted_trades:
            equity += trade['profit']
            equity_points.append({
                'time': trade['exit_time'],
                'equity': equity,
                'profit': trade['profit'],
                'profit_pct': trade['profit_pct'],
                'trade_type': trade['type']
            })
            
        return pd.DataFrame(equity_points).set_index('time')
    
    def generate_report(
        self,
        performance: Dict,
        equity_df: pd.DataFrame = None
    ) -> str:
        """Generate a text report of backtest results."""
        lines = []
        lines.append("=" * 60)
        lines.append("BACKTEST REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        total = performance.get('total', {})
        lines.append(f"Total Trades: {total.get('trades', 0)}")
        lines.append(f"Total Profit: ${total.get('profit', 0):.2f}")
        lines.append(f"Total Profit %: {total.get('profit_pct', 0):.2f}%")
        lines.append(f"Win Rate: {total.get('win_rate', 0):.1f}%")
        lines.append(f"Profit per Trade: ${total.get('profit_per_trade', 0):.4f}")
        lines.append("")
        
        long = performance.get('long', {})
        lines.append("LONG TRADES:")
        lines.append(f"  Trades: {long.get('trades', 0)}")
        lines.append(f"  Profit: ${long.get('profit', 0):.2f}")
        lines.append(f"  Win Rate: {long.get('win_rate', 0):.1f}%")
        lines.append("")
        
        short = performance.get('short', {})
        lines.append("SHORT TRADES:")
        lines.append(f"  Trades: {short.get('trades', 0)}")
        lines.append(f"  Profit: ${short.get('profit', 0):.2f}")
        lines.append(f"  Win Rate: {short.get('win_rate', 0):.1f}%")
        lines.append("")
        
        if equity_df is not None and len(equity_df) > 0:
            returns = equity_df['equity'].pct_change().dropna()
            sharpe = self.calculate_sharpe_ratio(returns)
            max_dd, peak, trough = self.calculate_max_drawdown(equity_df['equity'])
            
            lines.append("RISK METRICS:")
            lines.append(f"  Sharpe Ratio: {sharpe:.2f}")
            lines.append(f"  Max Drawdown: {max_dd:.2f}%")
            lines.append(f"  Final Equity: ${equity_df['equity'].iloc[-1]:.2f}")
            
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
