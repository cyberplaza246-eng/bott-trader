#!/usr/bin/env python3
"""Quick sweep-gated backtest — minimal output, all pairs."""
import logging
logging.disable(logging.CRITICAL)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.backtest.backtest_engine import BacktestEngine
from config.strategy_config import PAIRS as CONFIG_PAIRS, ASSET_CLASS, INITIAL_BALANCE
from src.instruments import REGISTRY

PAIRS = CONFIG_PAIRS
BALANCE = INITIAL_BALANCE
CONFIDENCE = 0.55
AGREEMENT = 2
MAX_5M_CANDLES = 0   # 0 = use all data

print(f"\n{'='*60}")
print(f"  SWEEP-GATED BACKTEST (RayAlgo v3)")
print(f"  Balance: ${BALANCE} | Confidence: {CONFIDENCE} | Agreement: {AGREEMENT}")
print(f"{'='*60}\n")

all_results = {}
for pair in PAIRS:
    csv5 = f'data/{pair.replace("/","_")}_5m.csv'
    csv1 = f'data/{pair.replace("/","_")}_1m.csv'
    if not os.path.exists(csv5):
        print(f"  {pair}: No 5M data found")
        continue
    
    df5 = pd.read_csv(csv5)
    df5['datetime'] = pd.to_datetime(df5['datetime'])
    if MAX_5M_CANDLES > 0:
        df5 = df5.tail(MAX_5M_CANDLES).reset_index(drop=True)
    
    df1 = None
    if os.path.exists(csv1):
        df1 = pd.read_csv(csv1)
        df1['datetime'] = pd.to_datetime(df1['datetime'])
        # Keep 1M data that matches the 5M window
        dt_start = df5['datetime'].iloc[0]
        df1 = df1[df1['datetime'] >= dt_start].reset_index(drop=True)
    
    spec = REGISTRY.get(pair)
    slippage = 0.5 if spec and spec.asset_class == 'futures' else 0.3
    commission = spec.commission_rt if spec else 7.0

    engine = BacktestEngine(
        initial_balance=BALANCE,
        slippage_pips=slippage,
        commission_per_lot=commission,
    )
    
    r = engine.run_backtest(
        df5, pair,
        confidence_threshold=CONFIDENCE,
        min_agreement=AGREEMENT,
        timeframe_key='5m',
        df_1m=df1,
    )
    all_results[pair] = r
    
    print(f"--- {pair} ({len(df5)} candles) ---")
    print(f"  Trades: {r['total_trades']}")
    if r['total_trades'] > 0:
        # Calculate days from data
        if 'datetime' in df5.columns:
            d0 = pd.to_datetime(df5['datetime'].iloc[0])
            d1 = pd.to_datetime(df5['datetime'].iloc[-1])
            days = max(1, (d1 - d0).days)
        else:
            days = max(1, len(df5) * 5 / (60 * 24))
        trades_per_day = r['total_trades'] / days
        print(f"  Trades/Day: {trades_per_day:.1f} (target: 8+)")
        print(f"  W/L: {r['winning_trades']}/{r['losing_trades']}")
        print(f"  Win Rate: {r['win_rate']:.1f}%")
        print(f"  Profit Factor: {r['profit_factor']:.2f}")
        print(f"  Balance: ${r['initial_balance']:.2f} -> ${r['final_balance']:.2f}")
        print(f"  Return: {r['return_percent']:+.2f}%")
        print(f"  Max Drawdown: {r['max_drawdown']:.2f}%")
        print(f"  Avg Pips/Trade: {r['avg_pips']:.1f}")
        print(f"  Avg Win: {r['avg_win_pips']:.1f} pips | Avg Loss: {r['avg_loss_pips']:.1f} pips")
        print(f"  Sharpe Ratio: {r['sharpe_ratio']:.2f}")
        print(f"  Timeouts: {r['timeout_exits']}")
        print(f"  Costs: ${r['total_costs']:.2f} (spread=${r['total_spread_cost']:.2f} comm=${r['total_commission_cost']:.2f})")
    else:
        print(f"  No trades — sweep gate too selective for this data")
    print()

# Combined summary
if len(all_results) > 1:
    total_trades = sum(r['total_trades'] for r in all_results.values())
    total_wins = sum(r['winning_trades'] for r in all_results.values())
    total_profit = sum(r['total_profit'] for r in all_results.values())
    total_costs = sum(r.get('total_costs', 0) for r in all_results.values())
    
    print(f"{'='*60}")
    print(f"  COMBINED")
    print(f"{'='*60}")
    print(f"  Total Trades: {total_trades}")
    if total_trades > 0:
        print(f"  Combined Win Rate: {total_wins/total_trades*100:.1f}%")
        print(f"  Combined Profit: ${total_profit:+.2f}")
        print(f"  Combined Costs: ${total_costs:.2f}")
    print()

# Assessment
print("ASSESSMENT:")
for pair, r in all_results.items():
    if r['total_trades'] == 0:
        print(f"  {pair}: ❌ 0 trades — sweep gate too restrictive for historical data")
    elif r['total_trades'] < 10:
        print(f"  {pair}: ⚠️  Only {r['total_trades']} trades — not statistically significant")
    elif r['win_rate'] < 40:
        print(f"  {pair}: ❌ Win rate {r['win_rate']:.1f}% < 40% — needs tuning")
    elif r['profit_factor'] >= 1.5 and r['win_rate'] >= 50:
        print(f"  {pair}: ✅ Excellent — PF {r['profit_factor']:.2f}, WR {r['win_rate']:.1f}%")
    elif r['profit_factor'] >= 1.0:
        print(f"  {pair}: ⚠️  Break-even — PF {r['profit_factor']:.2f}")
    else:
        print(f"  {pair}: ❌ Negative expectancy — PF {r['profit_factor']:.2f}")
print()
