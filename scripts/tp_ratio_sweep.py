#!/usr/bin/env python3
"""TP Ratio Sweep — find optimal R:R for max Profit Factor.

Tests multiple TP ratio maps while keeping everything else constant (v6b config).
Uses monkey-patching to swap the tp_ratio_map without modifying source files.
"""
import logging
logging.disable(logging.CRITICAL)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.backtest.backtest_engine import BacktestEngine
from src.ai import liquidity_sweep as lsmod

PAIRS = ['EUR/USD', 'GBP/USD']
BALANCE = 50
CONFIDENCE = 0.45
AGREEMENT = 1
MAX_5M_CANDLES = 3500

# TP ratio configs to test (regime → ratio)
CONFIGS = {
    'v6b_baseline':   {'high_volatility': 2.5, 'trend_up': 2.0, 'trend_down': 2.0, 'range': 1.5, 'low_volatility': 1.2},
    'tight_1.5':      {'high_volatility': 2.0, 'trend_up': 1.5, 'trend_down': 1.5, 'range': 1.2, 'low_volatility': 1.0},
    'tight_1.2':      {'high_volatility': 1.5, 'trend_up': 1.2, 'trend_down': 1.2, 'range': 1.0, 'low_volatility': 0.8},
    'wide_2.5':       {'high_volatility': 3.0, 'trend_up': 2.5, 'trend_down': 2.5, 'range': 2.0, 'low_volatility': 1.5},
    'trend_aggressive': {'high_volatility': 2.0, 'trend_up': 2.5, 'trend_down': 2.5, 'range': 1.0, 'low_volatility': 1.0},
    'uniform_1.8':    {'high_volatility': 1.8, 'trend_up': 1.8, 'trend_down': 1.8, 'range': 1.8, 'low_volatility': 1.8},
}

# Load data once
data_cache = {}
for pair in PAIRS:
    csv5 = f'data/{pair.replace("/","_")}_5m.csv'
    csv1 = f'data/{pair.replace("/","_")}_1m.csv'
    if not os.path.exists(csv5):
        continue
    df5 = pd.read_csv(csv5)
    df5['datetime'] = pd.to_datetime(df5['datetime'])
    df5 = df5.tail(MAX_5M_CANDLES).reset_index(drop=True)
    df1 = None
    if os.path.exists(csv1):
        df1 = pd.read_csv(csv1)
        df1['datetime'] = pd.to_datetime(df1['datetime'])
        dt_start = df5['datetime'].iloc[0]
        df1 = df1[df1['datetime'] >= dt_start].reset_index(drop=True)
    data_cache[pair] = (df5, df1)

print(f"\n{'='*70}")
print(f"  TP RATIO SWEEP — {len(CONFIGS)} configs × {len(PAIRS)} pairs")
print(f"{'='*70}\n")

results = []
for cfg_name, tp_map in CONFIGS.items():
    print(f"--- {cfg_name}: {tp_map} ---")
    
    combined_trades = 0
    combined_wins = 0
    combined_profit = 0
    combined_costs = 0
    pair_results = []
    
    for pair in PAIRS:
        if pair not in data_cache:
            continue
        df5, df1 = data_cache[pair]
        
        engine = BacktestEngine(
            initial_balance=BALANCE,
            slippage_pips=0.3,
            commission_per_lot=7.0,
        )
        
        # Monkey-patch the TP ratio map
        orig_calc = engine.sweep.calculate_risk_reward
        def patched_calc(sweep_result, displacement_result, regime_info, pair_arg, _map=tp_map, _orig=orig_calc):
            # Temporarily override
            import types
            old_map = {'high_volatility': 2.5, 'trend_up': 2.0, 'trend_down': 2.0, 'range': 1.5, 'low_volatility': 1.2}
            # Patch the method's local logic by overriding on the analyzer object
            return _orig(sweep_result, displacement_result, regime_info, pair_arg)
        
        # Direct patch: replace the tp_ratio_map lookup in calculate_risk_reward
        original_method = engine.sweep.calculate_risk_reward
        def make_patched(tp_m, orig_method):
            def patched(sweep_result, displacement_result, regime_info, pair_arg):
                # Override regime to use our tp_ratio
                regime = regime_info.get('regime', 'trend_up')
                custom_ratio = tp_m.get(regime, 1.5)
                # Temporarily patch the class constant
                import types
                # Call original but we need to intercept the tp_ratio_map
                # Simplest: just modify regime_info to embed our ratio
                # Actually, let's just monkey-patch at class level
                return orig_method(sweep_result, displacement_result, regime_info, pair_arg)
            return patched
        
        # Better approach: patch at class level
        orig_crr = lsmod.LiquiditySweepAnalyzer.calculate_risk_reward
        def make_class_patch(tp_m, orig_fn):
            def patched_crr(self, sweep_result, displacement_result, regime_info, pair_arg):
                # Store original, patch, call, restore
                old_code = None
                result = orig_fn(self, sweep_result, displacement_result, regime_info, pair_arg)
                # Result uses original tp_ratio_map. We need to recalculate with our map.
                if result is not None:
                    regime = regime_info.get('regime', 'trend_up')
                    new_ratio = tp_m.get(regime, 1.5)
                    old_ratio = result.get('rr_ratio', 1.5)
                    if abs(old_ratio - new_ratio) > 0.01:
                        # Recalculate TP based on new ratio
                        entry = result['entry_price']
                        sl_dist = result['sl_distance']
                        new_tp_dist = sl_dist * new_ratio
                        direction = 'BUY' if result['take_profit'] > entry else 'SELL'
                        if direction == 'BUY':
                            result['take_profit'] = round(entry + new_tp_dist, 5)
                        else:
                            result['take_profit'] = round(entry - new_tp_dist, 5)
                        result['tp_distance'] = new_tp_dist
                        pip_size = 0.0001 if 'JPY' not in pair_arg else 0.01
                        result['reward_pips'] = new_tp_dist / pip_size
                        result['rr_ratio'] = new_ratio
                        result['tp_ratio_used'] = new_ratio
                return result
            return patched_crr
        
        lsmod.LiquiditySweepAnalyzer.calculate_risk_reward = make_class_patch(tp_map, orig_crr)
        
        r = engine.run_backtest(
            df5, pair,
            confidence_threshold=CONFIDENCE,
            min_agreement=AGREEMENT,
            timeframe_key='5m',
            df_1m=df1,
        )
        
        # Restore
        lsmod.LiquiditySweepAnalyzer.calculate_risk_reward = orig_crr
        
        combined_trades += r['total_trades']
        combined_wins += r['winning_trades']
        combined_profit += r['total_profit']
        combined_costs += r.get('total_costs', 0)
        pair_results.append((pair, r))
    
    wr = (combined_wins / combined_trades * 100) if combined_trades > 0 else 0
    
    # Calculate combined PF
    all_gross_wins = sum(r['total_profit'] + r.get('total_costs', 0) for _, r in pair_results if r['total_profit'] > 0)
    
    print(f"  Trades: {combined_trades} | WR: {wr:.1f}% | P/L: ${combined_profit:+.2f} | Costs: ${combined_costs:.2f}")
    for pair, r in pair_results:
        print(f"    {pair}: {r['total_trades']}t WR={r['win_rate']:.1f}% PF={r['profit_factor']:.2f} P/L=${r['total_profit']:+.2f}")
    print()
    
    results.append({
        'config': cfg_name,
        'trades': combined_trades,
        'win_rate': wr,
        'profit': combined_profit,
        'costs': combined_costs,
    })

print(f"\n{'='*70}")
print(f"  SWEEP RESULTS (sorted by profit)")
print(f"{'='*70}")
sorted_results = sorted(results, key=lambda x: x['profit'], reverse=True)
for r in sorted_results:
    marker = '✅' if r['profit'] > 0 else '❌'
    print(f"  {marker} {r['config']:20s} | {r['trades']}t | WR={r['win_rate']:.1f}% | P/L=${r['profit']:+.2f} | Costs=${r['costs']:.2f}")

best = sorted_results[0]
print(f"\n  🏆 BEST: {best['config']} → ${best['profit']:+.2f}")
