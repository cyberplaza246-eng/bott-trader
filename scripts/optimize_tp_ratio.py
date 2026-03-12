#!/usr/bin/env python3
"""Quick TP ratio optimization — sweep tp_ratio_map values to find PF max."""
import logging, os, sys
logging.disable(logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.backtest.backtest_engine import BacktestEngine
from src.ai.liquidity_sweep import LiquiditySweepAnalyzer
from config.strategy_config import PAIRS as CONFIG_PAIRS, INITIAL_BALANCE

PAIRS = CONFIG_PAIRS
MAX_CANDLES = 3500
BALANCE = INITIAL_BALANCE
CONFIDENCE = 0.45

# Test different TP ratios for trend regimes
tp_ratios_to_test = [1.5, 1.8, 2.0, 2.2, 2.5, 3.0]

print(f"{'TR_RATIO':>8} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'PnL':>7} | {'AvgWin':>7} | {'AvgLoss':>7} | {'R:R':>5}")
print("-" * 75)

best_pf = 0
best_ratio = 0

for tr in tp_ratios_to_test:
    # Monkey-patch the tp_ratio_map in LiquiditySweepAnalyzer
    # We modify calculate_risk_reward's tp_ratio_map dynamically
    total_trades = 0
    total_wins = 0
    total_pnl = 0
    total_win_pips = []
    total_loss_pips = []
    
    for pair in PAIRS:
        csv5 = f'data/{pair.replace("/","_")}_5m.csv'
        csv1 = f'data/{pair.replace("/","_")}_1m.csv'
        if not os.path.exists(csv5):
            continue
        
        df5 = pd.read_csv(csv5)
        df5['datetime'] = pd.to_datetime(df5['datetime'])
        df5 = df5.tail(MAX_CANDLES).reset_index(drop=True)
        
        df1 = None
        if os.path.exists(csv1):
            df1 = pd.read_csv(csv1)
            df1['datetime'] = pd.to_datetime(df1['datetime'])
            dt_start = df5['datetime'].iloc[0]
            df1 = df1[df1['datetime'] >= dt_start].reset_index(drop=True)
        
        engine = BacktestEngine(initial_balance=BALANCE, slippage_pips=0.3, commission_per_lot=7.0)
        
        # Patch TP ratio for this run
        orig_calc = engine.sweep.calculate_risk_reward
        def patched_calc(sweep_result, displacement_result, regime_info, pair_name, _tr=tr, _orig=orig_calc):
            # Temporarily override tp_ratio_map
            result = _orig(sweep_result, displacement_result, regime_info, pair_name)
            if result is not None:
                # Recalculate TP with new ratio
                regime = regime_info.get('regime', 'trend_up')
                new_tp_map = {
                    'high_volatility': _tr + 0.5,
                    'trend_up': _tr,
                    'trend_down': _tr,
                    'range': max(1.2, _tr - 0.5),
                    'low_volatility': max(1.0, _tr - 0.8),
                }
                new_ratio = new_tp_map.get(regime, _tr)
                sl_dist = result['sl_distance']
                new_tp_dist = sl_dist * new_ratio
                entry = result['entry_price']
                direction = sweep_result.get('direction', 'BUY')
                if direction == 'BUY':
                    result['take_profit'] = round(entry + new_tp_dist, 5)
                else:
                    result['take_profit'] = round(entry - new_tp_dist, 5)
                result['tp_distance'] = new_tp_dist
                result['reward_pips'] = new_tp_dist / result.get('atr', 0.0001) * 10 if result.get('atr', 0) > 0 else 0
                result['rr_ratio'] = new_ratio
                result['tp_ratio_used'] = new_ratio
            return result
        engine.sweep.calculate_risk_reward = patched_calc
        
        r = engine.run_backtest(df5, pair, confidence_threshold=CONFIDENCE, min_agreement=1, timeframe_key='5m', df_1m=df1)
        
        total_trades += r['total_trades']
        total_wins += r['winning_trades']
        total_pnl += r['total_profit']
        if r['avg_win_pips'] > 0: total_win_pips.append(r['avg_win_pips'])
        if r['avg_loss_pips'] < 0: total_loss_pips.append(abs(r['avg_loss_pips']))
    
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    avg_w = sum(total_win_pips) / len(total_win_pips) if total_win_pips else 0
    avg_l = sum(total_loss_pips) / len(total_loss_pips) if total_loss_pips else 1
    rr = avg_w / avg_l if avg_l > 0 else 0
    
    # Calculate PF from actual trades
    gross_profit = sum(t['profit_loss'] for e_eng in [engine] for t in e_eng.trades if t['profit_loss'] > 0)
    gross_loss = sum(abs(t['profit_loss']) for e_eng in [engine] for t in e_eng.trades if t['profit_loss'] <= 0)
    # This only captures last pair, but gives approximate PF
    pf = total_pnl / max(abs(total_pnl - total_pnl), 0.01) if total_pnl != 0 else 1.0
    # Simpler: use the combined P/L as indicator
    
    indicator = "✅" if total_pnl > 0 else "❌"
    print(f"{tr:>8.1f} | {total_trades:>6} | {wr:>5.1f} | {'':>5} | ${total_pnl:>+6.2f} | {avg_w:>6.1f}p | {avg_l:>6.1f}p | {rr:>5.2f} {indicator}")
    
    if total_pnl > best_pf:
        best_pf = total_pnl
        best_ratio = tr

print(f"\nBest TP ratio: {best_ratio} (PnL: ${best_pf:+.2f})")
