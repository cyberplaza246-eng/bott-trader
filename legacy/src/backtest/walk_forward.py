"""
Walk-Forward Backtesting Engine

Implements walk-forward analysis to prevent overfitting:
  - Splits data into rolling train/test windows
  - Optimizes parameters on train, validates on test
  - Produces out-of-sample performance metrics
  - Supports anchored and rolling window modes

Usage:
    wf = WalkForwardEngine(initial_balance=10000)
    results = wf.run_walk_forward(
        df_5m, 'EUR/USD',
        train_pct=0.70,
        n_splits=5,
        mode='rolling',       # 'rolling' or 'anchored'
        df_1m=df_1m,
    )
"""
import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import datetime
from src.backtest.backtest_engine import BacktestEngine
from src.utils.logger import bot_logger


class WalkForwardEngine:
    """
    Walk-forward analysis engine.

    Splits historical data into sequential train/test windows and runs
    the full 9-model ensemble on each out-of-sample test fold.  This
    prevents curve-fitting by ensuring every reported metric comes from
    data the models have never seen during parameter tuning.

    Two modes:
      - **anchored**: Training window always starts at bar 0 and expands.
      - **rolling**: Training window slides forward (constant size).
    """

    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance

    # ──────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────
    def run_walk_forward(
        self,
        data: pd.DataFrame,
        pair: str,
        train_pct: float = 0.70,
        n_splits: int = 5,
        mode: str = 'rolling',
        confidence_threshold: float = 0.45,
        min_agreement: int = 2,
        timeframe_key: str = '5m',
        df_1m: pd.DataFrame = None,
        bar_minutes: int = None,
        slippage_pips: float = 0.0,
    ) -> dict:
        """
        Run walk-forward analysis.

        Args:
            data:       Primary OHLCV DataFrame (5M or 1M).
            pair:       Currency pair, e.g. 'EUR/USD'.
            train_pct:  Fraction of each window used for training (0.5–0.9).
            n_splits:   Number of train/test folds.
            mode:       'rolling' or 'anchored'.
            confidence_threshold: Min confidence to open a trade.
            min_agreement: Min models agreeing.
            timeframe_key: Config key for SL/TP ('1m' or '5m').
            df_1m:      Optional 1M DataFrame for confluence.
            bar_minutes: Minutes per bar (auto-detect if None).
            slippage_pips: Slippage to add per trade (pips).

        Returns:
            dict with per-fold and aggregate results.
        """
        n = len(data)
        if n < 500:
            bot_logger.warning("Walk-forward: not enough data (need ≥500 bars)")
            return self._empty_result(pair)

        # ── Build fold boundaries ─────────────────────────────────
        folds = self._build_folds(n, train_pct, n_splits, mode)
        bot_logger.info(
            f"Walk-forward: {pair} | mode={mode} | splits={n_splits} "
            f"| train_pct={train_pct} | total_bars={n}"
        )

        fold_results = []
        combined_trades = []
        equity_curve = [self.initial_balance]
        running_balance = self.initial_balance

        for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
            bot_logger.info(
                f"  Fold {i+1}/{len(folds)}: "
                f"train [{train_start}:{train_end}] ({train_end - train_start} bars) | "
                f"test [{test_start}:{test_end}] ({test_end - test_start} bars)"
            )

            train_data = data.iloc[train_start:train_end].copy()
            test_data = data.iloc[test_start:test_end].copy()

            # Prepare 1M confluence data for the test window
            test_1m = None
            if df_1m is not None and 'datetime' in data.columns and 'datetime' in df_1m.columns:
                test_1m = self._slice_1m_for_window(
                    df_1m, data, test_start, test_end
                )

            # ── Train phase: warm up the adaptive learner ─────────
            # Run backtest on training data (results are discarded;
            # the purpose is to let the AdaptiveLearner calibrate).
            train_engine = BacktestEngine(initial_balance=running_balance)
            _train_result = train_engine.run_backtest(
                train_data, pair,
                confidence_threshold=confidence_threshold,
                min_agreement=min_agreement,
                timeframe_key=timeframe_key,
                df_1m=None,  # no 1M confluence during training to save time
                bar_minutes=bar_minutes,
            )
            bot_logger.info(
                f"    Train: {_train_result['total_trades']} trades, "
                f"WR={_train_result['win_rate']:.1f}%, "
                f"PF={_train_result['profit_factor']:.2f}"
            )

            # ── Test phase: out-of-sample evaluation ──────────────
            test_engine = BacktestEngine(initial_balance=running_balance)
            # Transfer learned state from training
            test_engine.learner = deepcopy(train_engine.learner)

            test_result = test_engine.run_backtest(
                test_data, pair,
                confidence_threshold=confidence_threshold,
                min_agreement=min_agreement,
                timeframe_key=timeframe_key,
                df_1m=test_1m,
                bar_minutes=bar_minutes,
            )

            # Apply slippage post-hoc to test trades
            if slippage_pips > 0:
                test_result = self._apply_slippage(
                    test_result, test_engine.trades, pair, slippage_pips
                )

            fold_results.append({
                'fold': i + 1,
                'train_bars': train_end - train_start,
                'test_bars': test_end - test_start,
                'train_trades': _train_result['total_trades'],
                'train_win_rate': _train_result['win_rate'],
                'train_profit_factor': _train_result['profit_factor'],
                'test_trades': test_result['total_trades'],
                'test_win_rate': test_result['win_rate'],
                'test_profit_factor': test_result['profit_factor'],
                'test_sharpe': test_result['sharpe_ratio'],
                'test_max_drawdown': test_result['max_drawdown'],
                'test_return_pct': test_result['return_percent'],
                'test_total_profit': test_result['total_profit'],
            })

            # Accumulate equity and trades
            running_balance = test_result['final_balance']
            combined_trades.extend(test_engine.trades)
            equity_curve.append(running_balance)

            bot_logger.info(
                f"    Test:  {test_result['total_trades']} trades, "
                f"WR={test_result['win_rate']:.1f}%, "
                f"PF={test_result['profit_factor']:.2f}, "
                f"Return={test_result['return_percent']:.2f}%"
            )

        # ── Aggregate out-of-sample statistics ────────────────────
        aggregate = self._aggregate_results(
            fold_results, combined_trades, equity_curve, pair
        )
        aggregate['fold_results'] = fold_results

        bot_logger.info(
            f"Walk-forward complete: {pair} | "
            f"OOS trades={aggregate['total_trades']} | "
            f"OOS WR={aggregate['win_rate']:.1f}% | "
            f"OOS PF={aggregate['profit_factor']:.2f} | "
            f"OOS Return={aggregate['return_percent']:.2f}% | "
            f"Max DD={aggregate['max_drawdown']:.2f}%"
        )

        return aggregate

    # ──────────────────────────────────────────────────────────────
    #  Fold construction
    # ──────────────────────────────────────────────────────────────
    def _build_folds(self, n: int, train_pct: float, n_splits: int, mode: str):
        """
        Build train/test index boundaries.

        Returns list of (train_start, train_end, test_start, test_end).
        """
        folds = []

        if mode == 'anchored':
            # Anchored: training always starts at 0, test window slides
            # Total usable = n, split into n_splits test chunks after
            # an initial training block
            min_train = int(n * train_pct * 0.5)  # minimum train size
            remaining = n - min_train
            test_size = remaining // n_splits

            for i in range(n_splits):
                train_end = min_train + i * test_size
                test_start = train_end
                test_end = min(test_start + test_size, n)
                if test_end <= test_start:
                    break
                folds.append((0, train_end, test_start, test_end))

        else:  # rolling
            # Rolling: fixed window size that slides forward
            window_size = n // n_splits
            train_size = int(window_size * train_pct)
            test_size = window_size - train_size

            # Ensure minimum sizes
            train_size = max(train_size, 200)
            test_size = max(test_size, 50)

            for i in range(n_splits):
                train_start = i * test_size
                train_end = train_start + train_size
                test_start = train_end
                test_end = min(test_start + test_size, n)

                if train_end >= n or test_end <= test_start:
                    break
                folds.append((train_start, train_end, test_start, test_end))

        return folds

    # ──────────────────────────────────────────────────────────────
    #  1M slicing helper
    # ──────────────────────────────────────────────────────────────
    def _slice_1m_for_window(self, df_1m, df_primary, test_start, test_end):
        """Slice 1M data to match the test window's time range."""
        try:
            dt_start = pd.to_datetime(df_primary['datetime'].iloc[test_start])
            dt_end = pd.to_datetime(df_primary['datetime'].iloc[min(test_end - 1, len(df_primary) - 1)])

            df_1m_copy = df_1m.copy()
            df_1m_copy['datetime'] = pd.to_datetime(df_1m_copy['datetime'])

            # Include extra lookback for indicator warmup
            warmup = pd.Timedelta(hours=4)
            mask = (df_1m_copy['datetime'] >= dt_start - warmup) & \
                   (df_1m_copy['datetime'] <= dt_end)
            sliced = df_1m_copy.loc[mask].reset_index(drop=True)

            return sliced if len(sliced) > 200 else None
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────
    #  Slippage modeling
    # ──────────────────────────────────────────────────────────────
    def _apply_slippage(self, result: dict, trades: list, pair: str,
                        slippage_pips: float) -> dict:
        """
        Post-hoc slippage adjustment.

        Deducts slippage_pips from every trade's P/L (both entry and exit).
        """
        from src.risk.position_manager import PIP_VALUES, DEFAULT_PIP
        pip_info = PIP_VALUES.get(pair, DEFAULT_PIP)
        pip_value = pip_info['pip_value_per_lot']

        adjusted_pl = 0.0
        for t in trades:
            slip_cost = slippage_pips * 2 * t.get('lot_size', 0.01) * pip_value
            t['profit_loss'] -= slip_cost
            t['pips'] -= slippage_pips * 2
            adjusted_pl += t['profit_loss']

        # Recalculate summary stats
        winning = [t for t in trades if t['profit_loss'] > 0]
        losing = [t for t in trades if t['profit_loss'] <= 0]
        total = len(trades)

        result['win_rate'] = (len(winning) / total * 100) if total else 0
        gross_profit = sum(t['profit_loss'] for t in winning)
        gross_loss = abs(sum(t['profit_loss'] for t in losing))
        result['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0
        result['total_profit'] = adjusted_pl
        result['final_balance'] = result['initial_balance'] + adjusted_pl
        result['return_percent'] = (adjusted_pl / result['initial_balance']) * 100

        return result

    # ──────────────────────────────────────────────────────────────
    #  Aggregate results across folds
    # ──────────────────────────────────────────────────────────────
    def _aggregate_results(self, fold_results, trades, equity_curve, pair):
        """Compute aggregate out-of-sample metrics."""
        if not trades:
            return self._empty_result(pair)

        total_trades = len(trades)
        winning = [t for t in trades if t['profit_loss'] > 0]
        losing = [t for t in trades if t['profit_loss'] <= 0]

        win_rate = (len(winning) / total_trades * 100) if total_trades else 0
        gross_profit = sum(t['profit_loss'] for t in winning)
        gross_loss = abs(sum(t['profit_loss'] for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        total_profit = sum(t['profit_loss'] for t in trades)

        avg_pips = np.mean([t['pips'] for t in trades]) if trades else 0
        avg_win_pips = np.mean([t['pips'] for t in winning]) if winning else 0
        avg_loss_pips = np.mean([t['pips'] for t in losing]) if losing else 0

        # Equity curve metrics
        eq = np.array(equity_curve)
        running_max = np.maximum.accumulate(eq)
        drawdown = (eq - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100

        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) \
            if len(returns) > 1 and np.std(returns) > 0 else 0

        # Train vs test consistency (overfitting detection)
        train_wrs = [f['train_win_rate'] for f in fold_results if f['train_trades'] > 0]
        test_wrs = [f['test_win_rate'] for f in fold_results if f['test_trades'] > 0]
        overfit_ratio = (np.mean(train_wrs) / np.mean(test_wrs)) if test_wrs and np.mean(test_wrs) > 0 else 0

        return {
            'pair': pair,
            'mode': 'walk_forward',
            'total_trades': total_trades,
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_profit': total_profit,
            'initial_balance': self.initial_balance,
            'final_balance': equity_curve[-1] if equity_curve else self.initial_balance,
            'return_percent': (total_profit / self.initial_balance) * 100,
            'avg_pips': avg_pips,
            'avg_win_pips': avg_win_pips,
            'avg_loss_pips': avg_loss_pips,
            'equity_curve': equity_curve,
            'overfit_ratio': overfit_ratio,
            'n_folds': len(fold_results),
            'avg_train_win_rate': np.mean(train_wrs) if train_wrs else 0,
            'avg_test_win_rate': np.mean(test_wrs) if test_wrs else 0,
        }

    def _empty_result(self, pair):
        return {
            'pair': pair,
            'mode': 'walk_forward',
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_profit': 0.0,
            'initial_balance': self.initial_balance,
            'final_balance': self.initial_balance,
            'return_percent': 0.0,
            'avg_pips': 0.0,
            'avg_win_pips': 0.0,
            'avg_loss_pips': 0.0,
            'equity_curve': [self.initial_balance],
            'overfit_ratio': 0.0,
            'n_folds': 0,
            'fold_results': [],
        }
