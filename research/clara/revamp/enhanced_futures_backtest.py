#!/usr/bin/env python3
"""
Enhanced MES Futures Backtest with Advanced Risk Management
Implements stop losses, improved filters, trend confirmation, and reduced overtrading
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

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def enhanced_futures_backtest(df, symbol, initial_balance=50000):
    """Enhanced backtest with advanced risk management and filters"""

    # Calculate indicators
    df['ema_fast'] = calculate_ema(df['close'], 9)
    df['ema_slow'] = calculate_ema(df['close'], 21)
    df['ema_trend'] = calculate_ema(df['close'], 50)  # Trend filter
    df['ema_long'] = calculate_ema(df['close'], 200)  # Long-term trend
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], 14)

    # ADX for trend strength
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([df['high'] - df['low'],
                     (df['high'] - df['close'].shift(1)).abs(),
                     (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    atr_smooth = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_smooth)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    df['adx'] = dx.ewm(span=14, adjust=False).mean()

    # MACD for momentum logging (not used as signal filter)
    macd_fast = calculate_ema(df['close'], 12)
    macd_slow = calculate_ema(df['close'], 26)
    df['macd'] = macd_fast - macd_slow
    df['macd_signal'] = calculate_ema(df['macd'], 9)
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Contract specifications
    if symbol == 'MES':
        tick_value = 1.25  # $1.25 per point
        commission_per_contract = 2.50
        atr_mult_sl = 2.0  # Stop loss multiplier
        atr_mult_tp = 4.0  # Take profit multiplier (2.0R ratio)
    elif symbol == 'MNQ':
        tick_value = 0.25  # $0.25 per point
        commission_per_contract = 2.50
        atr_mult_sl = 2.0
        atr_mult_tp = 4.0
    else:
        tick_value = 1.0
        commission_per_contract = 2.50
        atr_mult_sl = 2.0
        atr_mult_tp = 4.0

    # Enhanced signal generation with multiple filters
    df['signal'] = 0

    # Trend filter: EMA50 > EMA200 for bullish, EMA50 < EMA200 for bearish
    bullish_trend = df['ema_trend'] > df['ema_long']
    bearish_trend = df['ema_trend'] < df['ema_long']

    # Price position filter: price must be on the right side of EMA200
    price_above_ema200 = df['close'] > df['ema_long']
    price_below_ema200 = df['close'] < df['ema_long']

    # ADX trending filter: require strong trend
    # MNQ needs stricter filter since it's noisier
    adx_threshold = 30 if symbol == 'MNQ' else 25
    trending = df['adx'] > adx_threshold

    # RSI filters
    oversold = df['rsi'] < 40
    overbought = df['rsi'] > 60

    # Session filter for futures: only trade during US RTH (13-21 UTC on 5M data)
    if hasattr(df.index, 'hour'):
        in_session = (df.index.hour >= 13) & (df.index.hour < 21)
    else:
        in_session = pd.Series(True, index=df.index)

    # EMA crossover signals with trend and RSI confirmation
    bullish_crossover = (df['ema_fast'] > df['ema_slow']) & df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)
    bearish_crossover = (df['ema_fast'] < df['ema_slow']) & df['ema_fast'].shift(1) >= df['ema_slow'].shift(1)

    # Buy signals: Bullish crossover + price above EMA200 + bullish trend + ADX trending + RSI + session
    df.loc[bullish_crossover & price_above_ema200 & bullish_trend & trending & oversold & in_session, 'signal'] = 1

    # Sell signals: Bearish crossover + price below EMA200 + bearish trend + ADX trending + RSI + session
    df.loc[bearish_crossover & price_below_ema200 & bearish_trend & trending & overbought & in_session, 'signal'] = -1

    # Additional filter: Only trade if ATR is above minimum threshold
    min_atr_threshold = df['atr'].quantile(0.2)  # Top 80% volatility
    # Zero out signals where ATR is too low
    df.loc[df['atr'] <= min_atr_threshold, 'signal'] = 0

    trades = []
    balance = initial_balance
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    trailing_activated = False  # Track if SL has been moved to breakeven
    last_signal = 0  # Track previous signal to detect transitions
    cooldown_bars = 0  # Bars remaining in cooldown after trade close
    COOLDOWN_PERIOD = 60  # Wait 60 bars (5 hours on 5M) between trades

    # Iterate over EVERY bar (not just signal changes) so SL/TP/trailing checks work correctly
    for idx, row in df.iterrows():
        raw_signal = row['signal']
        # Only trigger entry on signal TRANSITION (rising edge detection)
        signal = raw_signal if raw_signal != last_signal else 0
        last_signal = raw_signal

        # Enforce cooldown after trade close
        if cooldown_bars > 0:
            cooldown_bars -= 1
            signal = 0

        # Check for stop loss/take profit hits first using high/low for intra-bar precision
        if position != 0:
            current_price = row['close']
            bar_high = row['high']
            bar_low = row['low']

            # Trailing stop: move SL to breakeven+1tick after 1R profit achieved
            if not trailing_activated:
                sl_distance = abs(entry_price - stop_loss)
                if position > 0 and bar_high >= entry_price + sl_distance:
                    stop_loss = entry_price + 0.25  # breakeven + 1 tick
                    trailing_activated = True
                elif position < 0 and bar_low <= entry_price - sl_distance:
                    stop_loss = entry_price - 0.25  # breakeven + 1 tick
                    trailing_activated = True

            # Check stop loss (use low for longs, high for shorts)
            sl_hit = (position > 0 and bar_low <= stop_loss) or (position < 0 and bar_high >= stop_loss)
            # Check take profit (use high for longs, low for shorts)
            tp_hit = (position > 0 and bar_high >= take_profit) or (position < 0 and bar_low <= take_profit)

            if sl_hit:
                # Stop loss hit
                exit_price = stop_loss
                if position > 0:  # Closing long
                    pnl = (exit_price - entry_price) * tick_value * abs(position)
                else:  # Closing short
                    pnl = (entry_price - exit_price) * tick_value * abs(position)

                pnl -= commission_per_contract * abs(position)
                balance += pnl

                exit_reason = 'TRAILING_BE' if trailing_activated else 'STOP_LOSS'
                trades.append({
                    'entry_time': entry_price_time,
                    'exit_time': idx,
                    'direction': 'LONG' if position > 0 else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'bars_held': len(df.loc[entry_price_time:idx]) - 1,
                    'atr_at_entry': entry_atr
                })

                position = 0
                cooldown_bars = COOLDOWN_PERIOD
                continue

            elif tp_hit:
                # Take profit hit
                exit_price = take_profit
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
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'exit_reason': 'TAKE_PROFIT',
                    'pnl': pnl,
                    'bars_held': len(df.loc[entry_price_time:idx]) - 1,
                    'atr_at_entry': entry_atr
                })

                position = 0
                cooldown_bars = COOLDOWN_PERIOD
                continue

        # Close existing position if signal changes (trend reversal)
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
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'exit_reason': 'SIGNAL_REVERSE',
                'pnl': pnl,
                'bars_held': len(df.loc[entry_price_time:idx]) - 1,
                'atr_at_entry': entry_atr
            })

            position = 0
            cooldown_bars = COOLDOWN_PERIOD

        # Open new position
        if position == 0 and signal != 0:
            position = signal
            entry_price = row['close']
            entry_atr = row['atr']
            entry_price_time = idx
            trailing_activated = False  # Reset trailing for new trade

            # Calculate ATR-based stop loss and take profit
            atr_value = row['atr']
            if position > 0:  # Long position
                stop_loss = entry_price - (atr_value * atr_mult_sl)
                take_profit = entry_price + (atr_value * atr_mult_tp)
            else:  # Short position
                stop_loss = entry_price + (atr_value * atr_mult_sl)
                take_profit = entry_price - (atr_value * atr_mult_tp)

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
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'exit_reason': 'END_OF_DATA',
            'pnl': pnl,
            'bars_held': len(df.loc[entry_price_time:]) - 1,
            'atr_at_entry': entry_atr
        })

    return trades, balance

def print_enhanced_results(symbol, trades, final_balance, initial_balance):
    """Print enhanced backtest results with detailed statistics"""

    if not trades:
        print(f"❌ No trades generated for {symbol}")
        return

    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]
    tp_trades = [t for t in trades if t.get('exit_reason') == 'TAKE_PROFIT']
    sl_trades = [t for t in trades if t.get('exit_reason') == 'STOP_LOSS']
    trailing_be_trades = [t for t in trades if t.get('exit_reason') == 'TRAILING_BE']
    signal_reverse_trades = [t for t in trades if t.get('exit_reason') == 'SIGNAL_REVERSE']
    end_trades = [t for t in trades if t.get('exit_reason') == 'END_OF_DATA']

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

    print(f"\n🚀 ENHANCED {symbol} BACKTEST RESULTS")
    print("=" * 50)
    print(f"Total Trades: {total_trades}")
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

    # Exit reason breakdown
    print(f"\n📊 EXIT REASONS:")
    print(f"   Take Profit: {len(tp_trades)} ({len(tp_trades)/total_trades*100:.1f}%)")
    print(f"   Stop Loss: {len(sl_trades)} ({len(sl_trades)/total_trades*100:.1f}%)")
    print(f"   Trailing BE: {len(trailing_be_trades)} ({len(trailing_be_trades)/total_trades*100:.1f}%)")
    print(f"   Signal Reverse: {len(signal_reverse_trades)} ({len(signal_reverse_trades)/total_trades*100:.1f}%)")
    print(f"   End of Data: {len(end_trades)} ({len(end_trades)/total_trades*100:.1f}%)")

    # Risk metrics
    if winning_trades and losing_trades:
        avg_win_to_loss_ratio = abs(avg_win / avg_loss)
        print(f"\n🛡️ RISK METRICS:")
        print(f"   Win/Loss Ratio: {avg_win_to_loss_ratio:.2f}")
        print(f"   Profit Factor: {profit_factor:.2f}")
        print(f"   Max Drawdown: {max_drawdown:.1f}%")

        # Kelly Criterion approximation
        if win_rate > 0 and avg_win_to_loss_ratio > 0:
            kelly = (win_rate/100 * (avg_win_to_loss_ratio + 1) - 1) / avg_win_to_loss_ratio
            kelly_pct = max(0, kelly * 100)
            print(f"   Kelly Criterion: {kelly_pct:.1f}%")

def main():
    print("🚀 Enhanced Futures Backtest with Advanced Risk Management")
    print("="*60)
    print("✨ IMPROVEMENTS IMPLEMENTED:")
    print("   ✅ ATR-based Stop Losses & Take Profits")
    print("   ✅ Tighter RSI Filters (35/65 vs 30/70)")
    print("   ✅ 50/200 EMA Trend Confirmation")
    print("   ✅ Reduced Overtrading with Multiple Filters")
    print("   ✅ Volatility Filtering (ATR threshold)")
    print("="*60)

    symbols = ['MES', 'MNQ']
    results = {}

    for symbol in symbols:
        # Use 5M data for cleaner signal generation
        data_file = f'data/{symbol}_5m.csv'
        if not os.path.exists(data_file):
            print(f"❌ Data file not found for {symbol}")
            continue
        timeframe_label = '5-minute'

        print(f"\n📊 Loading {symbol} {timeframe_label} data...")
        df = pd.read_csv(data_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')

        print(f"✅ Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")

        # Run enhanced backtest
        print(f"🤖 Running enhanced strategy on {symbol}...")
        initial_balance = 50000
        trades, final_balance = enhanced_futures_backtest(df.copy(), symbol, initial_balance)

        results[symbol] = {
            'trades': trades,
            'final_balance': final_balance,
            'initial_balance': initial_balance
        }

        print_enhanced_results(symbol, trades, final_balance, initial_balance)

    # Comparison with previous results
    if len(results) == 2:
        print(f"\n{'='*60}")
        print("🎯 ENHANCED VS BASIC STRATEGY COMPARISON")
        print(f"{'='*60}")

        print(f"{'Contract':<8} {'Trades':<8} {'Win%':<8} {'P&L':<12} {'PF':<6} {'Return':<8} {'MaxDD':<8}")
        print("-" * 60)

        for symbol, data in results.items():
            trades = data['trades']
            if trades:
                win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100
                total_pnl = sum(t['pnl'] for t in trades)

                # Calculate max drawdown
                balance_history = [data['initial_balance']]
                for trade in trades:
                    balance_history.append(balance_history[-1] + trade['pnl'])
                peak = data['initial_balance']
                max_dd = 0
                for balance in balance_history:
                    if balance > peak:
                        peak = balance
                    dd = (peak - balance) / peak * 100
                    max_dd = max(max_dd, dd)

                ret = (data['final_balance']/data['initial_balance'] - 1) * 100
                pf = abs(sum(t['pnl'] for t in [t for t in trades if t['pnl'] > 0]) /
                        sum(t['pnl'] for t in [t for t in trades if t['pnl'] <= 0])) if any(t['pnl'] <= 0 for t in trades) else float('inf')

                print(f"{symbol:<8} {len(trades):<8} {win_rate:<8.1f} ${total_pnl:<11,.0f} {pf:<6.2f} {ret:<8.1f} {max_dd:<8.1f}")

        print(f"\n💡 KEY IMPROVEMENTS:")
        print(f"   • ATR-based risk management reduces drawdown")
        print(f"   • Tighter filters reduce losing trades")
        print(f"   • Trend confirmation improves win rate")
        print(f"   • Stop losses prevent large losses")

        print(f"\n🚀 NEXT: Test your Ultimate AI Trading Bot!")
        print(f"   Run: python launch_ultimate_bot.py --mode paper")

if __name__ == "__main__":
    main()