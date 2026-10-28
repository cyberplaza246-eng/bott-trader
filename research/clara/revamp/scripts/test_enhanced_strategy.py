"""
Test the enhanced scalping strategy with improved filters.
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/workspaces/Ai-bot')

from src.strategy.enhanced_scalping import EnhancedScalpingStrategy, EnhancedSignal


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add required indicators to dataframe."""
    df = df.copy()
    
    # EMAs
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # ADX
    df['plus_dm'] = np.where(
        (df['high'].diff() > df['low'].diff().abs()) & (df['high'].diff() > 0),
        df['high'].diff(), 0
    )
    df['minus_dm'] = np.where(
        (df['low'].diff().abs() > df['high'].diff()) & (df['low'].diff() < 0),
        df['low'].diff().abs(), 0
    )
    df['plus_di'] = 100 * (df['plus_dm'].rolling(14).mean() / (df['atr'] + 1e-10))
    df['minus_di'] = 100 * (df['minus_dm'].rolling(14).mean() / (df['atr'] + 1e-10))
    dx = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-10)
    df['adx'] = dx.rolling(14).mean()
    
    return df


def backtest_enhanced_strategy(data_5m: pd.DataFrame, data_1m: pd.DataFrame, 
                                pair: str, min_score: float = 0.55) -> dict:
    """Run backtest with enhanced strategy."""
    
    # Add indicators
    data_5m = add_indicators(data_5m)
    data_1m = add_indicators(data_1m)
    
    strategy = EnhancedScalpingStrategy(
        min_score=min_score,
        use_liquidity_sweep=True,
        use_trading_hours=False,  # Disable for backtest (data is historical)
        atr_sl_mult=1.5,
        tp1_ratio=1.5,
        tp2_ratio=2.5
    )
    
    trades = []
    position = None
    
    # Point value for MES
    point_value = 5.0  # $5 per point for MES
    commission = 3.50  # Round-trip commission
    
    print(f"Running enhanced backtest on {len(data_5m)} 5M bars...")
    
    for idx in range(210, len(data_5m)):
        candle_dt = data_5m['datetime'].iloc[idx] if 'datetime' in data_5m.columns else None
        current_price = data_5m['close'].iloc[idx]
        current_high = data_5m['high'].iloc[idx]
        current_low = data_5m['low'].iloc[idx]
        
        # Check for exit if in position
        if position is not None:
            # Check TP2 (full exit)
            hit_tp2 = (position['direction'] == 'long' and current_high >= position['tp2']) or \
                      (position['direction'] == 'short' and current_low <= position['tp2'])
            
            # Check TP1 (partial exit - track but continue)
            hit_tp1 = (position['direction'] == 'long' and current_high >= position['tp1']) or \
                      (position['direction'] == 'short' and current_low <= position['tp1'])
            
            # Check SL
            hit_sl = (position['direction'] == 'long' and current_low <= position['sl']) or \
                     (position['direction'] == 'short' and current_high >= position['sl'])
            
            # Mark partial exit if hit TP1 but not exited yet
            if hit_tp1 and not position.get('partial_exit'):
                position['partial_exit'] = True
                position['partial_pnl'] = abs(position['tp1'] - position['entry']) * point_value * 0.5
            
            exit_price = None
            exit_type = None
            
            if hit_sl:
                exit_price = position['sl']
                exit_type = 'STOP_LOSS'
            elif hit_tp2:
                exit_price = position['tp2']
                exit_type = 'TAKE_PROFIT_2'
            elif idx >= position['entry_idx'] + 20:  # Max hold time
                exit_price = current_price
                exit_type = 'TIMEOUT'
            
            if exit_price is not None:
                # Calculate PnL
                if position['direction'] == 'long':
                    raw_pnl = (exit_price - position['entry']) * point_value
                else:
                    raw_pnl = (position['entry'] - exit_price) * point_value
                
                # If we had partial exit at TP1, add that portion
                if position.get('partial_exit') and exit_type != 'STOP_LOSS':
                    # Partial (50%) at TP1 + remaining (50%) at exit
                    raw_pnl = position['partial_pnl'] + (raw_pnl * 0.5)
                
                pnl = raw_pnl - commission
                
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': candle_dt,
                    'direction': position['direction'],
                    'entry': position['entry'],
                    'exit': exit_price,
                    'exit_type': exit_type,
                    'pnl': pnl,
                    'entry_score': position['entry_score'],
                    'regime': position['regime'],
                    'partial_exit': position.get('partial_exit', False)
                })
                position = None
                continue
        
        # Generate signal if no position
        if position is None:
            # Create subsets for strategy
            subset_5m = data_5m.iloc[max(0, idx - 250):idx + 1].copy()
            
            if candle_dt is not None and 'datetime' in data_1m.columns:
                mask_1m = data_1m['datetime'] <= candle_dt
                subset_1m = data_1m.loc[mask_1m].tail(300).copy()
            else:
                subset_1m = data_1m.tail(300).copy()
            
            if len(subset_1m) < 50:
                continue
            
            signal = strategy.generate_signal(
                data_1m=subset_1m,
                data_5m=subset_5m,
                candle_dt=candle_dt
            )
            
            if signal.direction != 'none' and signal.confidence >= min_score:
                position = {
                    'direction': signal.direction,
                    'entry': signal.entry_price,
                    'sl': signal.stop_loss,
                    'tp1': signal.take_profit1,
                    'tp2': signal.take_profit2,
                    'entry_idx': idx,
                    'entry_time': candle_dt,
                    'entry_score': signal.entry_score,
                    'regime': signal.regime,
                    'partial_exit': False,
                    'partial_pnl': 0
                }
    
    # Calculate stats
    if not trades:
        return {'trades': 0, 'win_rate': 0, 'total_pnl': 0, 'profit_factor': 0}
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    total_pnl = sum(t['pnl'] for t in trades)
    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Exit type breakdown
    sl_exits = len([t for t in trades if t['exit_type'] == 'STOP_LOSS'])
    tp2_exits = len([t for t in trades if t['exit_type'] == 'TAKE_PROFIT_2'])
    timeout_exits = len([t for t in trades if t['exit_type'] == 'TIMEOUT'])
    partial_exits = len([t for t in trades if t.get('partial_exit')])
    
    return {
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'profit_factor': profit_factor,
        'avg_win': gross_profit / len(wins) if wins else 0,
        'avg_loss': gross_loss / len(losses) if losses else 0,
        'sl_exits': sl_exits,
        'tp2_exits': tp2_exits,
        'timeout_exits': timeout_exits,
        'partial_exits': partial_exits,
        'trades_list': trades
    }


def main():
    print("=== ENHANCED STRATEGY BACKTEST ===\n")
    
    # Load data
    try:
        mes_5m = pd.read_csv('/workspaces/Ai-bot/data/MES_5m.csv', parse_dates=['datetime'])
        mes_1m = pd.read_csv('/workspaces/Ai-bot/data/MES_1m.csv', parse_dates=['datetime'])
        print(f"Loaded MES: {len(mes_5m)} 5M bars, {len(mes_1m)} 1M bars")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Test different min_score thresholds
    for min_score in [0.50, 0.55, 0.60, 0.65]:
        print(f"\n--- Min Score: {min_score} ---")
        results = backtest_enhanced_strategy(mes_5m, mes_1m, 'MES', min_score=min_score)
        
        print(f"Trades: {results['trades']}")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Total PnL: ${results['total_pnl']:.2f}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Avg Win: ${results['avg_win']:.2f} | Avg Loss: ${results['avg_loss']:.2f}")
        print(f"Exits - SL: {results['sl_exits']} | TP2: {results['tp2_exits']} | Timeout: {results['timeout_exits']}")
        
        # Show verdict
        if results['profit_factor'] >= 1.3 and results['total_pnl'] > 0:
            print("✅ PROFITABLE")
        elif results['profit_factor'] >= 1.0:
            print("⚠️ BREAKEVEN")
        else:
            print("❌ LOSING")


if __name__ == "__main__":
    main()
