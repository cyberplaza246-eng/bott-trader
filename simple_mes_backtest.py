#!/usr/bin/env python3
"""
Simple MES Futures Backtest
Tests basic trend-following strategy on MES futures data
"""

import pandas as pd
import numpy as np
import sys
import os

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

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

def backtest_mes_strategy(df, initial_balance=50000):
    """Backtest a simple EMA crossover strategy on MES"""

    # Calculate indicators
    df['ema_fast'] = calculate_ema(df['close'], 9)
    df['ema_slow'] = calculate_ema(df['close'], 21)
    df['rsi'] = calculate_rsi(df['close'], 14)

    # Generate signals
    df['signal'] = 0
    df.loc[(df['ema_fast'] > df['ema_slow']) & (df['rsi'] > 30), 'signal'] = 1   # Buy signal
    df.loc[(df['ema_fast'] < df['ema_slow']) & (df['rsi'] < 70), 'signal'] = -1  # Sell signal

    # Remove consecutive signals (avoid overtrading)
    df['signal_change'] = df['signal'].diff()
    df = df[df['signal_change'] != 0].copy()

    trades = []
    balance = initial_balance
    position = 0
    entry_price = 0

    # Commission per contract (simplified)
    commission_per_contract = 2.50  # $2.50 per contract
    tick_value = 1.25  # $1.25 per point (MES = 1/10 of ES)

    for idx, row in df.iterrows():
        signal = row['signal']

        # Close existing position if signal changes
        if position != 0 and signal != 0 and np.sign(signal) != np.sign(position):
            # Calculate P&L
            exit_price = row['close']
            if position > 0:  # Closing long
                pnl = (exit_price - entry_price) * tick_value * abs(position)
            else:  # Closing short
                pnl = (entry_price - exit_price) * tick_value * abs(position)

            # Subtract commission
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
            position = signal  # +1 for long, -1 for short
            entry_price = row['close']
            entry_price_time = idx

            # Subtract entry commission
            balance -= commission_per_contract * abs(position)

    # Close any remaining position at the end
    if position != 0:
        exit_price = df['close'].iloc[-1]
        if position > 0:  # Closing long
            pnl = (exit_price - entry_price) * tick_value * abs(position)
        else:  # Closing short
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

def main():
    print("🚀 MES Futures Backtest")
    print("="*50)

    # Load MES data
    data_file = 'data/MES_5m.csv'
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return

    print("📊 Loading MES 5-minute data...")
    df = pd.read_csv(data_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    print(f"✅ Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")

    # Run backtest
    print("\n🤖 Running EMA crossover strategy backtest...")
    initial_balance = 50000
    trades, final_balance = backtest_mes_strategy(df.copy(), initial_balance)

    # Calculate statistics
    if trades:
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]

        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades * 100
        total_pnl = sum(t['pnl'] for t in trades)
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')
        avg_bars_held = np.mean([t['bars_held'] for t in trades])

        # Calculate max drawdown
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

        print("\n📊 BACKTEST RESULTS")
        print("-" * 40)
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total P&L: ${total_pnl:,.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Average Win: ${avg_win:,.2f}")
        print(f"Average Loss: ${avg_loss:,.2f}")
        print(f"Average Bars Held: {avg_bars_held:.1f}")
        print(f"Max Drawdown: {max_drawdown:.1f}%")
        print(f"Initial Balance: ${initial_balance:,.2f}")
        print(f"Final Balance: ${final_balance:,.2f}")
        print(f"Return: {(final_balance/initial_balance - 1)*100:+.1f}%")

        # Show sample trades
        print(f"\n📋 SAMPLE TRADES (first 5):")
        for i, trade in enumerate(trades[:5]):
            print(f"{i+1}. {trade['direction']} | Entry: ${trade['entry_price']:,.2f} | Exit: ${trade['exit_price']:,.2f} | P&L: ${trade['pnl']:,.2f} | Bars: {trade['bars_held']}")

    else:
        print("❌ No trades were generated by the strategy")

if __name__ == "__main__":
    main()