#!/usr/bin/env python3
"""Sweep TP ratio map to find optimal R:R for profitability."""
import logging
logging.disable(logging.CRITICAL)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.ai.liquidity_sweep import LiquiditySweepAnalyzer
from src.backtest.backtest_engine import BacktestEngine
from config.strategy_config import PAIRS as CONFIG_PAIRS, INITIAL_BALANCE

PAIRS = CONFIG_PAIRS
BALANCE = INITIAL_BALANCE
CONFIDENCE = 0.45
MAX_5M_CANDLES = 3500

# TP ratio configs to test (trend_up/down, high_vol, range, low_vol)
CONFIGS = [
    {'label': 'current',   'trend': 2.0, 'high_vol': 2.5, 'range': 1.5, 'low_vol': 1.2},
    {'label': 'lower',     'trend': 1.5, 'high_vol': 2.0, 'range': 1.2, 'low_vol': 1.0},
    {'label': 'flat_1.5',  'trend': 1.5, 'high_vol': 1.5, 'range': 1.5, 'low_vol': 1.5},
    {'label': 'flat_2.0',  'trend': 2.0, 'high_vol': 2.0, 'range': 2.0, 'low_vol': 2.0},
    {'label': 'aggressive','trend': 2.5, 'high_vol': 3.0, 'range': 1.8, 'low_vol': 1.5},
    {'label': 'tight',     'trend': 1.2, 'high_vol': 1.5, 'range': 1.0, 'low_vol': 1.0},
]

# Load data once
data_cache = {}
for pair in PAIRS:
    csv5 = f'data/{pair.replace("/","_")}_5m.csv'
    csv1 = f'data/{pair.replace("/","_")}_1m.csv'
    df5 = pd.read_csv(csv5)
    df5['datetime'] = pd.to_datetime(df5['datetime'])
    df5 = df5.tail(MAX_5M_CANDLES).reset_index(drop=True)
    df1 = pd.read_csv(csv1)
    df1['datetime'] = pd.to_datetime(df1['datetime'])
    dt_start = df5['datetime'].iloc[0]
    df1 = df1[df1['datetime'] >= dt_start].reset_index(drop=True)
    data_cache[pair] = (df5, df1)

print(f"{'Config':<12} | {'Trades':>6} | {'WR':>6} | {'PF':>6} | {'P/L':>8} | {'AvgWin':>7} | {'AvgLoss':>7} | {'Costs':>6}")
print("-" * 80)

for cfg in CONFIGS:
    # Monkey-patch the TP ratio map
    original_calc = LiquiditySweepAnalyzer.calculate_risk_reward
    
    def make_patched(c):
        def patched_calc(self, sweep_result, displacement_result, regime_info, pair):
            # Override regime TP ratios
            regime = regime_info.get('regime', 'trend_up')
            override_map = {
                'high_volatility': c['high_vol'],
                'trend_up': c['trend'],
                'trend_down': c['trend'],
                'range': c['range'],
                'low_volatility': c['low_vol'],
            }
            # Temporarily patch the method's internal map
            old_regime = regime_info.get('regime')
            result = original_calc(self, sweep_result, displacement_result, regime_info, pair)
            if result is not None:
                # Re-calculate TP with our ratio
                tp_ratio = override_map.get(regime, 1.5)
                sl_dist = result['sl_distance']
                tp_dist = sl_dist * tp_ratio
                entry = result['entry_price']
                direction = sweep_result.get('direction', 'BUY')
                if direction == 'BUY':
                    result['take_profit'] = round(entry + tp_dist, 5)
                else:
                    result['take_profit'] = round(entry - tp_dist, 5)
                result['tp_distance'] = tp_dist
                result['reward_pips'] = tp_dist / self.PAIR_CONFIG.get(pair, {}).get('pip_size', 0.0001)
                result['rr_ratio'] = tp_ratio
                result['tp_ratio_used'] = tp_ratio
            return result
        return patched_calc
    
    LiquiditySweepAnalyzer.calculate_risk_reward = make_patched(cfg)
    
    total_trades = 0
    total_wins = 0
    total_profit = 0
    total_costs = 0
    all_win_pips = []
    all_loss_pips = []
    
    for pair in PAIRS:
        df5, df1 = data_cache[pair]
        engine = BacktestEngine(initial_balance=BALANCE, slippage_pips=0.3, commission_per_lot=7.0)
        r = engine.run_backtest(df5, pair, confidence_threshold=CONFIDENCE, min_agreement=1,
                                timeframe_key='5m', df_1m=df1)
        total_trades += r['total_trades']
        total_wins += r['winning_trades']
        total_profit += r['total_profit']
        total_costs += r.get('total_costs', 0)
        if r['avg_win_pips']: all_win_pips.append(r['avg_win_pips'])
        if r['avg_loss_pips']: all_loss_pips.append(r['avg_loss_pips'])
    
    # Restore
    LiquiditySweepAnalyzer.calculate_risk_reward = original_calc
    
    wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    avg_w = sum(all_win_pips) / len(all_win_pips) if all_win_pips else 0
    avg_l = sum(all_loss_pips) / len(all_loss_pips) if all_loss_pips else 0
    # Approximate PF
    if total_trades > 0 and total_wins > 0:
        gross_win = total_wins * avg_w
        gross_loss = (total_trades - total_wins) * abs(avg_l)
        pf = gross_win / gross_loss if gross_loss > 0 else 0
    else:
        pf = 0
    
    print(f"{cfg['label']:<12} | {total_trades:>6} | {wr:>5.1f}% | {pf:>5.2f} | ${total_profit:>+7.2f} | {avg_w:>6.1f}p | {avg_l:>6.1f}p | ${total_costs:>5.2f}")
