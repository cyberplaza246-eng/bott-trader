#!/usr/bin/env python3
"""Walk-Forward Validation — 70/30 split with parameter optimization.

1. Splits data into 70% train / 30% test by date
2. Sweeps key parameters on train set to find best combo
3. Validates on held-out test set
4. Reports both train and test metrics
"""
import logging
logging.disable(logging.CRITICAL)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
import pandas as pd
from src.backtest.backtest_engine import BacktestEngine
from config.strategy_config import PAIRS as CONFIG_PAIRS, INITIAL_BALANCE
from src.instruments import REGISTRY

PAIRS = CONFIG_PAIRS
BALANCE = INITIAL_BALANCE
TRAIN_RATIO = 0.70

# Parameter grid to sweep on training set
PARAM_GRID = {
    'confidence': [0.50, 0.55, 0.60],
    'agreement': [2, 3],
}

def split_data(df5, df1, train_ratio=0.70):
    """Split by date, keeping 1M aligned to 5M window."""
    n = len(df5)
    split_idx = int(n * train_ratio)
    split_dt = df5.iloc[split_idx]['datetime']

    train_5m = df5.iloc[:split_idx].reset_index(drop=True)
    test_5m = df5.iloc[split_idx:].reset_index(drop=True)

    train_1m, test_1m = None, None
    if df1 is not None and len(df1) > 0:
        train_1m = df1[df1['datetime'] < split_dt].reset_index(drop=True)
        test_1m = df1[df1['datetime'] >= split_dt].reset_index(drop=True)

    return train_5m, test_5m, train_1m, test_1m, split_dt


def run_single(df5, df1, pair, confidence, agreement):
    """Run one backtest with given params, return results dict."""
    spec = REGISTRY.get(pair)
    slippage = 0.5 if spec and spec.asset_class == 'futures' else 0.3
    commission = spec.commission_rt if spec else 7.0

    engine = BacktestEngine(
        initial_balance=BALANCE,
        slippage_pips=slippage,
        commission_per_lot=commission,
    )
    return engine.run_backtest(
        df5, pair,
        confidence_threshold=confidence,
        min_agreement=agreement,
        timeframe_key='5m',
        df_1m=df1,
    )


def score_result(r):
    """Score a backtest result for optimization. Higher = better."""
    trades = r['total_trades']
    if trades < 5:
        return -999  # Not enough data
    pf = r['profit_factor']
    wr = r['win_rate'] / 100.0
    dd = abs(r['max_drawdown'])
    ret = r['return_percent']

    # Composite: reward PF, penalize drawdown, bonus for trade count
    trade_bonus = min(trades / 30, 1.0) * 0.2  # up to +0.2 for 30+ trades
    return (pf * 0.5) + (ret * 0.3) + trade_bonus - (dd * 0.1)


def format_row(label, r):
    """Format a result row for display."""
    t = r['total_trades']
    if t == 0:
        return f"  {label}: 0 trades"
    return (
        f"  {label}: {t}T | WR {r['win_rate']:.1f}% | PF {r['profit_factor']:.2f} "
        f"| Ret {r['return_percent']:+.2f}% | DD {r['max_drawdown']:.2f}% "
        f"| ${r['total_profit']:+.0f}"
    )


def main():
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD VALIDATION")
    print(f"  Balance: ${BALANCE} | Train: {TRAIN_RATIO:.0%} | Test: {1-TRAIN_RATIO:.0%}")
    print(f"{'='*70}\n")

    combos = list(itertools.product(
        PARAM_GRID['confidence'],
        PARAM_GRID['agreement'],
    ))

    for pair in PAIRS:
        csv5 = f'data/{pair.replace("/","_")}_5m.csv'
        csv1 = f'data/{pair.replace("/","_")}_1m.csv'
        if not os.path.exists(csv5):
            print(f"  {pair}: No data"); continue

        df5 = pd.read_csv(csv5)
        df5['datetime'] = pd.to_datetime(df5['datetime'])
        df1 = None
        if os.path.exists(csv1):
            df1 = pd.read_csv(csv1)
            df1['datetime'] = pd.to_datetime(df1['datetime'])

        train_5m, test_5m, train_1m, test_1m, split_dt = split_data(df5, df1)

        d0 = df5['datetime'].iloc[0].strftime('%Y-%m-%d')
        d_split = split_dt.strftime('%Y-%m-%d')
        d1 = df5['datetime'].iloc[-1].strftime('%Y-%m-%d')
        train_days = (split_dt - df5['datetime'].iloc[0]).days
        test_days = (df5['datetime'].iloc[-1] - split_dt).days

        print(f"{'─'*70}")
        print(f"  {pair}")
        print(f"  Train: {d0} → {d_split} ({train_days}d, {len(train_5m)} 5M bars)")
        print(f"  Test:  {d_split} → {d1} ({test_days}d, {len(test_5m)} 5M bars)")
        print(f"{'─'*70}")

        # ── Optimize on training set
        print(f"\n  TRAINING SET — Parameter Sweep:")
        best_score = -999
        best_params = None
        best_train = None

        for conf, agr in combos:
            r = run_single(train_5m, train_1m, pair, conf, agr)
            sc = score_result(r)
            label = f"conf={conf:.2f} agr={agr}"
            print(format_row(f"    {label}", r))
            if sc > best_score:
                best_score = sc
                best_params = (conf, agr)
                best_train = r

        print(f"\n  ★ Best Train: conf={best_params[0]:.2f} agr={best_params[1]}")
        print(format_row("    TRAIN", best_train))

        # ── Validate on test set
        r_test = run_single(test_5m, test_1m, pair, best_params[0], best_params[1])
        print(f"\n  TEST SET (out-of-sample):")
        print(format_row("    TEST", r_test))

        # ── Robustness check
        if r_test['total_trades'] >= 5:
            train_pf = best_train['profit_factor']
            test_pf = r_test['profit_factor']
            degradation = (train_pf - test_pf) / max(train_pf, 0.01) * 100
            print(f"\n  Robustness: Train PF={train_pf:.2f} → Test PF={test_pf:.2f} "
                  f"(degradation: {degradation:+.0f}%)")
            if test_pf >= 1.3 and r_test['total_trades'] >= 15:
                print(f"  ✅ PASS — Test PF ≥ 1.3 with {r_test['total_trades']}+ trades")
            elif test_pf >= 1.0:
                print(f"  ⚠️  MARGINAL — Test PF > 1.0 but below 1.3 target")
            else:
                print(f"  ❌ FAIL — Test PF < 1.0")
        else:
            print(f"  ⚠️  Insufficient test trades ({r_test['total_trades']}) for validation")
        print()

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
