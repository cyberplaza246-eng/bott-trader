#!/usr/bin/env python
"""
Backtest using live Rithmic data.

Downloads historical candles from Rithmic and runs the full backtest.

Usage:
    python scripts/backtest_rithmic_data.py --symbol MES --days 30
    python scripts/backtest_rithmic_data.py --symbol MNQ --days 14
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


async def fetch_rithmic_data(symbol: str, days: int = 30, timeframe: int = 1) -> pd.DataFrame:
    """Fetch historical data from Rithmic."""
    from src.broker.rithmic_connector import RithmicConnector
    
    print(f"Connecting to Rithmic...")
    broker = RithmicConnector()
    
    if not broker._connected:
        print("❌ Failed to connect to Rithmic")
        return None
    
    print(f"✅ Connected! Fetching {days} days of {timeframe}m data for {symbol}...")
    
    # Fetch candles (Rithmic limits per request, so we may need multiple)
    candles_needed = days * 24 * 60 // timeframe  # Rough estimate
    
    # Rithmic can fetch about 10000 bars per request
    df = broker.get_candles(symbol, timeframe_minutes=timeframe, num_candles=min(candles_needed, 10000))
    
    if df is None or df.empty:
        print(f"❌ No data returned for {symbol}")
        return None
    
    print(f"✅ Fetched {len(df)} candles")
    return df


def run_backtest(df: pd.DataFrame, symbol: str):
    """Run backtest on the data."""
    from src.ai.technical_analyzer import TechnicalAnalyzer
    
    print(f"\n{'='*60}")
    print(f"  BACKTEST — {symbol} ({len(df)} bars)")
    print(f"{'='*60}\n")
    
    # Add indicators
    ta = TechnicalAnalyzer()
    df = ta.calculate_indicators(df)
    
    # Strategy parameters
    LOOKBACK = 10
    ATR_MULT = 1.5
    TP_MULT = 2.0
    EMA_LEN = 50
    
    # Symbol specs
    SPECS = {
        'MES': {'point_value': 5.0, 'tick_size': 0.25},
        'MNQ': {'point_value': 2.0, 'tick_size': 0.25},
    }
    spec = SPECS.get(symbol, SPECS['MES'])
    
    # Results tracking
    trades = []
    balance = 50000.0
    position = None
    
    # Ensure we have enough data
    if len(df) < 200:
        print(f"❌ Insufficient data: {len(df)} bars (need 200+)")
        return
    
    # Calculate indicators if not present
    if 'atr' not in df.columns:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
    
    if 'ema_50' not in df.columns:
        df['ema_50'] = df['close'].ewm(span=EMA_LEN, adjust=False).mean()
    
    df['highest'] = df['high'].rolling(LOOKBACK).max()
    df['lowest'] = df['low'].rolling(LOOKBACK).min()
    
    print("Running backtest...")
    
    for i in range(200, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['close']
        atr = row['atr']
        ema = row['ema_50']
        
        # Check exit if in position
        if position:
            exit_reason = None
            exit_price = None
            
            if position['direction'] == 'long':
                if row['low'] <= position['sl']:
                    exit_price = position['sl']
                    exit_reason = 'SL'
                elif row['high'] >= position['tp']:
                    exit_price = position['tp']
                    exit_reason = 'TP'
            else:  # short
                if row['high'] >= position['sl']:
                    exit_price = position['sl']
                    exit_reason = 'SL'
                elif row['low'] <= position['tp']:
                    exit_price = position['tp']
                    exit_reason = 'TP'
            
            if exit_reason:
                if position['direction'] == 'long':
                    pnl = (exit_price - position['entry']) * spec['point_value']
                else:
                    pnl = (position['entry'] - exit_price) * spec['point_value']
                
                balance += pnl
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': row.name if hasattr(row, 'name') else i,
                    'direction': position['direction'],
                    'entry': position['entry'],
                    'exit': exit_price,
                    'pnl': pnl,
                    'reason': exit_reason
                })
                position = None
        
        # Check entry (no position)
        if position is None and not pd.isna(atr) and atr > 0:
            signal = None
            
            # Bullish breakout
            if prev['close'] < prev['highest'] and price >= row['highest']:
                if price > ema:  # EMA filter
                    signal = 'long'
            
            # Bearish breakout
            elif prev['close'] > prev['lowest'] and price <= row['lowest']:
                if price < ema:  # EMA filter
                    signal = 'short'
            
            if signal:
                if signal == 'long':
                    sl = price - (atr * ATR_MULT)
                    tp = price + (atr * ATR_MULT * TP_MULT)
                else:
                    sl = price + (atr * ATR_MULT)
                    tp = price - (atr * ATR_MULT * TP_MULT)
                
                position = {
                    'direction': signal,
                    'entry': price,
                    'sl': sl,
                    'tp': tp,
                    'entry_time': row.name if hasattr(row, 'name') else i
                }
    
    # Calculate results
    if not trades:
        print("❌ No trades generated")
        return
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    
    total_pnl = sum(t['pnl'] for t in trades)
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    
    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Max drawdown
    equity_curve = [50000.0]
    for t in trades:
        equity_curve.append(equity_curve[-1] + t['pnl'])
    
    peak = equity_curve[0]
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    print(f"\n{'='*60}")
    print(f"  RESULTS — {symbol}")
    print(f"{'='*60}")
    print(f"\n📊 OVERVIEW")
    print(f"   Total Trades:    {len(trades)}")
    print(f"   Wins:            {len(wins)}")
    print(f"   Losses:          {len(losses)}")
    print(f"\n💰 PERFORMANCE")
    print(f"   Win Rate:        {win_rate:.1f}%")
    print(f"   Profit Factor:   {profit_factor:.2f}")
    print(f"   Total P&L:       ${total_pnl:.2f}")
    print(f"   Final Balance:   ${balance:.2f}")
    print(f"\n📉 RISK")
    print(f"   Max Drawdown:    {max_dd:.2f}%")
    if wins:
        print(f"   Avg Win:         ${sum(t['pnl'] for t in wins)/len(wins):.2f}")
    if losses:
        print(f"   Avg Loss:        ${sum(t['pnl'] for t in losses)/len(losses):.2f}")
    
    print(f"\n✅ VALIDATION")
    print(f"   PF ≥ 1.3:        {'✅' if profit_factor >= 1.3 else '❌'} ({profit_factor:.2f})")
    print(f"   DD < 10%:        {'✅' if max_dd < 10 else '❌'} ({max_dd:.2f}%)")
    print(f"   Win Rate 45-55%: {'✅' if 45 <= win_rate <= 55 else '❌'} ({win_rate:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Backtest with Rithmic data')
    parser.add_argument('--symbol', type=str, default='MES', choices=['MES', 'MNQ'])
    parser.add_argument('--days', type=int, default=30, help='Days of history')
    parser.add_argument('--timeframe', type=int, default=5, help='Timeframe in minutes')
    parser.add_argument('--full', action='store_true', help='Use full AI Ensemble (not just breakout)')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  Rithmic Data Backtest")
    print(f"  Symbol: {args.symbol} | Days: {args.days} | TF: {args.timeframe}m")
    print(f"{'='*60}\n")
    
    # Fetch data
    df = asyncio.run(fetch_rithmic_data(args.symbol, args.days, args.timeframe))
    
    if df is None or df.empty:
        print("❌ Failed to fetch data. Using cached data...")
        # Fallback to CSV - try exact match first, then resample from 1m
        csv_path = f"data/{args.symbol}_{args.timeframe}m.csv"
        csv_1m_path = f"data/{args.symbol}_1m.csv"
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(df)} rows from {csv_path}")
        elif os.path.exists(csv_1m_path):
            print(f"📊 Resampling from 1m data...")
            df_1m = pd.read_csv(csv_1m_path)
            
            # Ensure datetime column
            if 'timestamp' in df_1m.columns:
                df_1m['datetime'] = pd.to_datetime(df_1m['timestamp'])
            elif 'datetime' in df_1m.columns:
                df_1m['datetime'] = pd.to_datetime(df_1m['datetime'])
            
            df_1m = df_1m.set_index('datetime')
            
            # Resample to desired timeframe
            df = df_1m.resample(f'{args.timeframe}min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna().reset_index()
            
            # Filter to requested days
            if args.days and len(df) > 0:
                cutoff = df['datetime'].max() - pd.Timedelta(days=args.days)
                df = df[df['datetime'] >= cutoff]
            
            print(f"✅ Resampled to {len(df)} {args.timeframe}m bars from {csv_1m_path}")
        else:
            print(f"❌ No data found at {csv_path} or {csv_1m_path}")
            return
    
    # Run backtest
    if args.full:
        run_ensemble_backtest(df, args.symbol)
    else:
        run_backtest(df, args.symbol)


def run_ensemble_backtest(df: pd.DataFrame, symbol: str):
    """Run backtest using full AI Ensemble system."""
    from src.core.ensemble_trader import EnsembleTrader
    from src.ai.technical_analyzer import TechnicalAnalyzer
    
    print(f"\n{'='*60}")
    print(f"  FULL AI ENSEMBLE BACKTEST — {symbol} ({len(df)} bars)")
    print(f"{'='*60}\n")
    
    # Initialize ensemble (no broker for backtest)
    ensemble = EnsembleTrader(broker=None)
    ta = TechnicalAnalyzer()
    
    # Backtest mode: lower threshold (live uses 0.40 floor)
    ensemble.learner.backtest_mode = True
    ensemble.learner.confidence_threshold = 0.20
    print(f"📊 Backtest threshold: 20% (live uses 40%)")
    
    # Symbol specs
    SPECS = {
        'MES': {'point_value': 5.0, 'tick_size': 0.25},
        'MNQ': {'point_value': 2.0, 'tick_size': 0.25},
    }
    spec = SPECS.get(symbol, SPECS['MES'])
    
    # Track results
    trades = []
    balance = 50000.0
    position = None
    
    # Add indicators
    df = ta.calculate_indicators(df)
    
    # Ensure we have enough data
    if len(df) < 200:
        print(f"❌ Insufficient data: {len(df)} bars (need 200+)")
        return
    
    # Limit to last 1000 bars for speed (full ensemble is slow)
    if len(df) > 1200:
        print(f"📊 Limiting to last 1000 bars (from {len(df)}) for speed...")
        df = df.tail(1200).reset_index(drop=True)
    
    print("Running ensemble backtest...")
    
    # Debug counters
    signal_counts = {'BUY': 0, 'SELL': 0, 'SKIP': 0, 'HOLD': 0, 'passed': 0, 'blocked': 0}
    
    for i in range(200, len(df)):
        row = df.iloc[i]
        price = row['close']
        
        # Check exit if in position
        if position:
            exit_reason = None
            exit_price = None
            
            if position['direction'] == 'long':
                if row['low'] <= position['sl']:
                    exit_price = position['sl']
                    exit_reason = 'SL'
                elif row['high'] >= position['tp']:
                    exit_price = position['tp']
                    exit_reason = 'TP'
            else:  # short
                if row['high'] >= position['sl']:
                    exit_price = position['sl']
                    exit_reason = 'SL'
                elif row['low'] <= position['tp']:
                    exit_price = position['tp']
                    exit_reason = 'TP'
            
            if exit_reason:
                if position['direction'] == 'long':
                    pnl = (exit_price - position['entry']) * spec['point_value']
                else:
                    pnl = (position['entry'] - exit_price) * spec['point_value']
                
                balance += pnl
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': row.name if hasattr(row, 'name') else i,
                    'direction': position['direction'],
                    'entry': position['entry'],
                    'exit': exit_price,
                    'pnl': pnl,
                    'reason': exit_reason,
                    'confidence': position.get('confidence', 0)
                })
                position = None
        
        # Check entry via Ensemble (no position)
        if position is None:
            # Get slice for ensemble (last 100 bars)
            start_idx = max(0, i - 100)
            df_slice = df.iloc[start_idx:i+1].copy()
            
            try:
                # Generate signal from full AI ensemble
                signal_result = ensemble.get_trading_signal(df_slice, symbol)
                
                if signal_result:
                    sig = signal_result.get('signal', 'SKIP')
                    if sig in signal_counts:
                        signal_counts[sig] += 1
                
                if signal_result and signal_result.get('signal') in ('BUY', 'SELL'):
                    # Check if we should trade
                    if ensemble.should_trade(signal_result):
                        signal_counts['passed'] += 1
                        direction = 'long' if signal_result['signal'] == 'BUY' else 'short'
                        confidence = signal_result.get('confidence', 0)
                        
                        # Get ATR for SL/TP
                        atr = row.get('atr', 5.0)
                        if pd.isna(atr) or atr <= 0:
                            atr = 5.0
                        
                        # Calculate SL/TP
                        ATR_MULT = 1.5
                        TP_MULT = 2.0
                        
                        if direction == 'long':
                            sl = price - (atr * ATR_MULT)
                            tp = price + (atr * ATR_MULT * TP_MULT)
                        else:
                            sl = price + (atr * ATR_MULT)
                            tp = price - (atr * ATR_MULT * TP_MULT)
                        
                        position = {
                            'direction': direction,
                            'entry': price,
                            'sl': sl,
                            'tp': tp,
                            'entry_time': row.name if hasattr(row, 'name') else i,
                            'confidence': confidence
                        }
                    else:
                        signal_counts['blocked'] += 1
            except Exception as e:
                # Track exceptions
                if 'errors' not in signal_counts:
                    signal_counts['errors'] = 0
                    signal_counts['last_error'] = ''
                signal_counts['errors'] += 1
                signal_counts['last_error'] = str(e)[:100]
    
    # Print signal debug info
    print(f"\n📊 Signal Summary:")
    print(f"   BUY signals:  {signal_counts.get('BUY', 0)}")
    print(f"   SELL signals: {signal_counts.get('SELL', 0)}")
    print(f"   SKIP/HOLD:    {signal_counts.get('SKIP', 0) + signal_counts.get('HOLD', 0)}")
    print(f"   Passed:       {signal_counts.get('passed', 0)}")
    print(f"   Blocked:      {signal_counts.get('blocked', 0)}")
    print(f"   Errors:       {signal_counts.get('errors', 0)}")
    if signal_counts.get('last_error'):
        print(f"   Last Error:   {signal_counts.get('last_error')}")
    
    # Calculate results
    if not trades:
        print("❌ No trades generated")
        return
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    
    total_pnl = sum(t['pnl'] for t in trades)
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    
    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Avg confidence
    avg_conf = sum(t.get('confidence', 0) for t in trades) / len(trades) * 100 if trades else 0
    
    # Max drawdown
    equity_curve = [50000.0]
    for t in trades:
        equity_curve.append(equity_curve[-1] + t['pnl'])
    
    peak = equity_curve[0]
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    print(f"\n{'='*60}")
    print(f"  FULL AI ENSEMBLE RESULTS — {symbol}")
    print(f"{'='*60}")
    print(f"\n📊 OVERVIEW")
    print(f"   Total Trades:    {len(trades)}")
    print(f"   Wins:            {len(wins)}")
    print(f"   Losses:          {len(losses)}")
    print(f"   Avg Confidence:  {avg_conf:.1f}%")
    print(f"\n💰 PERFORMANCE")
    print(f"   Win Rate:        {win_rate:.1f}%")
    print(f"   Profit Factor:   {profit_factor:.2f}")
    print(f"   Total P&L:       ${total_pnl:.2f}")
    print(f"   Final Balance:   ${balance:.2f}")
    print(f"\n📉 RISK")
    print(f"   Max Drawdown:    {max_dd:.2f}%")
    if wins:
        print(f"   Avg Win:         ${sum(t['pnl'] for t in wins)/len(wins):.2f}")
    if losses:
        print(f"   Avg Loss:        ${sum(t['pnl'] for t in losses)/len(losses):.2f}")
    
    print(f"\n✅ VALIDATION")
    print(f"   PF ≥ 1.3:        {'✅' if profit_factor >= 1.3 else '❌'} ({profit_factor:.2f})")
    print(f"   DD < 10%:        {'✅' if max_dd < 10 else '❌'} ({max_dd:.2f}%)")
    print(f"   Win Rate 45-55%: {'✅' if 45 <= win_rate <= 55 else '❌'} ({win_rate:.1f}%)")
    
    print(f"\n🤖 ENSEMBLE COMPONENTS")
    print(f"   - Sweep-Gate Entry (MSS + Liquidity sweep)")
    print(f"   - IntelligentTrader (ML classifiers)")
    print(f"   - AdvancedStrategies (8 strategies)")
    print(f"   - Dynamic SL/TP with trailing stops")
    print(f"   - Adaptive confidence penalties")


if __name__ == '__main__':
    main()
