#!/usr/bin/env python3
"""
MNQ Futures Backtest
Tests the same strategy on Micro Nasdaq futures
"""

import pandas as pd
import numpy as np
import sys
import os

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    """Calculate RSI"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def backtest_futures_strategy(df, symbol, initial_balance=50000):
    """Backtest EMA crossover strategy"""

    # Calculate indicators
    df['ema_fast'] = calculate_ema(df['close'], 9)
    df['ema_slow'] = calculate_ema(df['close'], 21)
    df['rsi'] = calculate_rsi(df['close'], 14)

    # Generate signals
    df['signal'] = 0
    df.loc[(df['ema_fast'] > df['ema_slow']) & (df['rsi'] > 30), 'signal'] = 1   # Buy signal
    df.loc[(df['ema_fast'] < df['ema_slow']) & (df['rsi'] < 70), 'signal'] = -1  # Sell signal

    # Remove consecutive signals
    df['signal_change'] = df['signal'].diff()
    df = df[df['signal_change'] != 0].copy()

    trades = []
    balance = initial_balance
    position = 0
    entry_price = 0

    # Contract specifications
    if symbol == 'MES':
        tick_value = 1.25  # $1.25 per point
        commission_per_contract = 2.50
    elif symbol == 'MNQ':
        tick_value = 0.25  # $0.25 per point (1/5 of NQ)
        commission_per_contract = 2.50
    else:
        tick_value = 1.0
        commission_per_contract = 2.50

    for idx, row in df.iterrows():
        signal = row['signal']

        # Close existing position if signal changes
        if position != 0 and signal != 0 and np.sign(signal) != np.sign(position):
            exit_price = row['close']
            if position > 0:  # Closing long
                pnl = (exit_price - entry_price) * tick_value * abs(position)
            else:  # Closing short
                pnl = (entry_price - exit_price) * tick_value * abs(position)

            pnl -= commission_per_contract * abs(position)
            balance += pnl

            trades.append({
                'entry_time': entry_price_time,
                'exit_time': idx,
                'direction': 'LONG' if position > 0 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'bars_held': len(df.loc[entry_price_time:idx]) - 1
            })

            position = 0

        # Open new position
        if position == 0 and signal != 0:
            position = signal
            entry_price = row['close']
            entry_price_time = idx
            balance -= commission_per_contract * abs(position)

    # Close remaining position
    if position != 0:
        exit_price = df['close'].iloc[-1]
        if position > 0:
            pnl = (exit_price - entry_price) * tick_value * abs(position)
        else:
            pnl = (entry_price - exit_price) * tick_value * abs(position)

        pnl -= commission_per_contract * abs(position)
        balance += pnl

        trades.append({
            'entry_time': entry_price_time,
            'exit_time': df.index[-1],
            'direction': 'LONG' if position > 0 else 'SHORT',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'bars_held': len(df.loc[entry_price_time:]) - 1
        })

    return trades, balance

def print_results(symbol, trades, final_balance, initial_balance):
    """Print backtest results"""
    if not trades:
        print(f"❌ No trades generated for {symbol}")
        return

    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]

    total_trades = len(trades)
    win_rate = len(winning_trades) / total_trades * 100
    total_pnl = sum(t['pnl'] for t in trades)
    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
    profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')
    avg_bars_held = np.mean([t['bars_held'] for t in trades])

    # Max drawdown calculation
    balance_history = [initial_balance]
    for trade in trades:
        balance_history.append(balance_history[-1] + trade['pnl'])

    peak = initial_balance
    max_drawdown = 0
    for balance in balance_history:
        if balance > peak:
            peak = balance
        drawdown = (peak - balance) / peak * 100
        max_drawdown = max(max_drawdown, drawdown)

    print(f"\n📊 {symbol} BACKTEST RESULTS")
    print("-" * 40)
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Average Bars Held: {avg_bars_held:.1f}")
    print(f"Max Drawdown: {max_drawdown:.1f}%")
    print(f"Final Balance: ${final_balance:,.2f}")
    print(f"Return: {(final_balance/initial_balance - 1)*100:+.1f}%")

def main():
    print("🚀 Futures Backtest Comparison: MES vs MNQ")
    print("="*50)

    symbols = ['MES', 'MNQ']
    results = {}

    for symbol in symbols:
        data_file = f'data/{symbol}_5m.csv'
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            continue

        print(f"\n📊 Loading {symbol} 5-minute data...")
        df = pd.read_csv(data_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')

        print(f"✅ Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")

        # Run backtest
        print(f"🤖 Running EMA crossover strategy on {symbol}...")
        initial_balance = 50000
        trades, final_balance = backtest_futures_strategy(df.copy(), symbol, initial_balance)

        results[symbol] = {
            'trades': trades,
            'final_balance': final_balance,
            'initial_balance': initial_balance
        }

        print_results(symbol, trades, final_balance, initial_balance)

    # Comparison summary
    if len(results) == 2:
        print(f"\n{'='*50}")
        print("🎯 COMPARISON SUMMARY")
        print(f"{'='*50}")

        for symbol, data in results.items():
            trades = data['trades']
            if trades:
                win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100
                total_pnl = sum(t['pnl'] for t in trades)
                ret = (data['final_balance']/data['initial_balance'] - 1) * 100
                print(f"{symbol}: {len(trades)} trades, {win_rate:.1f}% win rate, ${total_pnl:,.2f} P&L, {ret:+.1f}% return")

if __name__ == "__main__":
    main()