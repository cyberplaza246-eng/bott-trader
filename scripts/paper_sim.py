#!/usr/bin/env python
"""
Paper Trading Simulator - No Rithmic connection required

Runs the clean scalper strategy on historical data in real-time simulation mode.
Great for testing the strategy before connecting to live markets.

Usage:
    python scripts/paper_sim.py              # Simulate both MES and MNQ
    python scripts/paper_sim.py MES          # Simulate MES only
    python scripts/paper_sim.py --speed 10   # Run 10x faster
"""

import sys
import time
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.clean_scalper import CleanScalper


def run_paper_simulation(pairs: list = None, speed: float = 1.0):
    """
    Simulate live trading by walking through historical data bar by bar.
    
    Args:
        pairs: List of instruments to trade
        speed: Simulation speed multiplier (1.0 = real-time, 10.0 = 10x faster)
    """
    pairs = pairs or ['MES', 'MNQ']
    scalper = CleanScalper()
    
    # Load data
    data = {}
    for pair in pairs:
        try:
            df_5m = pd.read_csv(f'data/{pair}_5m.csv', parse_dates=['datetime'])
            df_1m = pd.read_csv(f'data/{pair}_1m.csv', parse_dates=['datetime'])
            
            # Pre-calculate indicators
            df_5m = scalper.calculate_indicators(df_5m)
            df_1m = scalper.calculate_indicators(df_1m)
            
            data[pair] = {'5m': df_5m, '1m': df_1m}
            print(f"Loaded {pair}: {len(df_5m)} 5m bars, {len(df_1m)} 1m bars")
        except FileNotFoundError:
            print(f"Data not found for {pair}, skipping...")
    
    if not data:
        print("No data loaded!")
        return
    
    # Trading state
    positions: Dict[str, dict] = {}
    balance = 50000.0
    initial_balance = balance
    trades = []
    cooldown_until: Dict[str, int] = {p: 0 for p in pairs}
    
    # Find common date range
    all_5m = [data[p]['5m'] for p in data.keys()]
    min_date = max(df['datetime'].min() for df in all_5m)
    max_date = min(df['datetime'].max() for df in all_5m)
    
    print(f"\nSimulation period: {min_date} to {max_date}")
    print(f"Speed: {speed}x real-time")
    print("=" * 70)
    print()
    
    # Warmup period
    warmup = 250
    
    # Process bar by bar
    bar_count = 0
    
    for pair in data.keys():
        df_5m = data[pair]['5m']
        df_1m = data[pair]['1m']
        
        print(f"\n--- Starting {pair} simulation ---")
        
        for idx in range(warmup, len(df_1m)):
            bar_count += 1
            row = df_1m.iloc[idx]
            candle_dt = row['datetime']
            candle_hour = candle_dt.hour if hasattr(candle_dt, 'hour') else 12
            
            # Check open positions for SL/TP
            if pair in positions:
                pos = positions[pair]
                high = row['high']
                low = row['low']
                
                exit_price = None
                exit_type = None
                
                if pos['direction'] == 'BUY':
                    if low <= pos['stop_loss']:
                        exit_price = pos['stop_loss']
                        exit_type = 'SL'
                    elif high >= pos['take_profit']:
                        exit_price = pos['take_profit']
                        exit_type = 'TP'
                else:
                    if high >= pos['stop_loss']:
                        exit_price = pos['stop_loss']
                        exit_type = 'SL'
                    elif low <= pos['take_profit']:
                        exit_price = pos['take_profit']
                        exit_type = 'TP'
                
                if exit_price:
                    config = scalper.INSTRUMENT_CONFIG.get(pair, {})
                    tick_size = config.get('tick_size', 0.25)
                    tick_value = config.get('tick_value', 1.25)
                    
                    if pos['direction'] == 'BUY':
                        ticks = (exit_price - pos['entry_price']) / tick_size
                    else:
                        ticks = (pos['entry_price'] - exit_price) / tick_size
                    
                    pnl = ticks * tick_value * pos['contracts'] - 0.62 * pos['contracts']
                    balance += pnl
                    
                    trades.append({
                        'pair': pair,
                        'direction': pos['direction'],
                        'entry': pos['entry_price'],
                        'exit': exit_price,
                        'exit_type': exit_type,
                        'pnl': pnl,
                    })
                    
                    emoji = "✅" if exit_type == 'TP' else "❌"
                    print(f"{emoji} {candle_dt} | {pair} {exit_type} @ {exit_price:.2f} | "
                          f"P&L: ${pnl:+.2f} | Balance: ${balance:.2f}")
                    
                    del positions[pair]
                    cooldown_until[pair] = idx + 5
            
            # Skip if in cooldown or has position
            if idx < cooldown_until[pair] or pair in positions:
                continue
            
            # Every 5th 1m bar (roughly aligned with 5m), check for signal
            if idx % 5 != 0:
                continue
            
            # Get data subsets
            df_1m_sub = df_1m.iloc[max(0, idx - 250):idx + 1]
            
            # Get 5m subset
            approx_5m_idx = min(idx // 5, len(df_5m) - 1)
            df_5m_sub = df_5m.iloc[max(0, approx_5m_idx - 250):approx_5m_idx + 1]
            
            if len(df_5m_sub) < 210:
                continue
            
            # Generate signal
            signal = scalper.get_signal(
                df_1m_sub, pair,
                df_5m=df_5m_sub,
                candle_hour=candle_hour,
                precalculated=True
            )
            
            if signal['signal'] in ('BUY', 'SELL'):
                sl_tp = signal['sl_tp']
                positions[pair] = {
                    'direction': signal['signal'],
                    'entry_price': sl_tp['entry_price'],
                    'stop_loss': sl_tp['stop_loss'],
                    'take_profit': sl_tp['take_profit'],
                    'contracts': 1,
                }
                
                print(f"🎯 {candle_dt} | {pair} {signal['signal']} @ {sl_tp['entry_price']:.2f} | "
                      f"SL={sl_tp['stop_loss']:.2f} TP={sl_tp['take_profit']:.2f}")
            
            # Simulate real-time delay (60 seconds per bar / speed)
            if speed < 100:
                time.sleep(0.1 / speed)  # Small delay for visual effect
    
    # Print summary
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    
    total_trades = len(trades)
    winners = [t for t in trades if t['pnl'] > 0]
    losers = [t for t in trades if t['pnl'] <= 0]
    
    print(f"Total Trades: {total_trades}")
    print(f"Winners: {len(winners)} ({len(winners)/total_trades*100:.1f}%)" if total_trades > 0 else "")
    print(f"Losers: {len(losers)}")
    print(f"Net P&L: ${balance - initial_balance:.2f}")
    print(f"Final Balance: ${balance:.2f}")
    print(f"Return: {(balance - initial_balance) / initial_balance * 100:.2f}%")


def main():
    pairs = ['MES', 'MNQ']
    speed = 10.0  # Default 10x speed
    
    args = sys.argv[1:]
    for arg in args:
        if arg in ('MES', 'MNQ'):
            pairs = [arg]
        elif arg.startswith('--speed'):
            if '=' in arg:
                speed = float(arg.split('=')[1])
    
    print("=" * 70)
    print("Paper Trading Simulator")
    print("=" * 70)
    print()
    print(f"Strategy: Clean Scalper")
    print(f"Pairs: {pairs}")
    print(f"Min Confirmations: {CleanScalper.MIN_CONFIRMATIONS}")
    print()
    
    run_paper_simulation(pairs, speed)


if __name__ == '__main__':
    main()
