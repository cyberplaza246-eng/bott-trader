"""
Optimized enhanced strategy - fine-tuned for profitability.
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/workspaces/Ai-bot')


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add required indicators to dataframe."""
    df = df.copy()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    return df


def backtest_optimized(data_5m: pd.DataFrame, data_1m: pd.DataFrame, 
                       sl_mult=1.3, tp1_ratio=1.2, tp2_ratio=2.0,
                       min_score=0.50, partial_pct=0.5) -> dict:
    """
    Optimized backtest with adjustable parameters.
    
    Key differences:
    - Tighter SL (1.3x ATR vs 1.5x)
    - Lower TP1 for quicker wins (1.2R vs 1.5R)
    - Partial exit at TP1, trail remaining to TP2
    """
    
    data_5m = add_indicators(data_5m)
    data_1m = add_indicators(data_1m)
    
    trades = []
    position = None
    point_value = 5.0
    commission = 3.50
    
    for idx in range(210, len(data_5m)):
        candle_dt = data_5m['datetime'].iloc[idx] if 'datetime' in data_5m.columns else None
        current_price = data_5m['close'].iloc[idx]
        current_high = data_5m['high'].iloc[idx]
        current_low = data_5m['low'].iloc[idx]
        
        # Exit logic
        if position is not None:
            direction = position['direction']
            
            # Check hits
            hit_sl = (direction == 'long' and current_low <= position['sl']) or \
                     (direction == 'short' and current_high >= position['sl'])
            hit_tp1 = (direction == 'long' and current_high >= position['tp1']) or \
                      (direction == 'short' and current_low <= position['tp1'])
            hit_tp2 = (direction == 'long' and current_high >= position['tp2']) or \
                      (direction == 'short' and current_low <= position['tp2'])
            
            # Partial exit at TP1 - move SL to breakeven
            if hit_tp1 and not position.get('partial_exit'):
                position['partial_exit'] = True
                position['partial_pnl'] = abs(position['tp1'] - position['entry']) * point_value * partial_pct
                # Move SL to breakeven + small profit
                position['sl'] = position['entry'] + (0.1 * (position['tp1'] - position['entry']) if direction == 'long' 
                                                      else -0.1 * (position['entry'] - position['tp1']))
            
            exit_price = None
            exit_type = None
            
            if hit_sl:
                exit_price = position['sl']
                exit_type = 'STOP_LOSS' if not position.get('partial_exit') else 'BREAKEVEN'
            elif hit_tp2:
                exit_price = position['tp2']
                exit_type = 'TAKE_PROFIT_2'
            elif idx >= position['entry_idx'] + 15:  # 15 bars max (75 min)
                exit_price = current_price
                exit_type = 'TIMEOUT'
            
            if exit_price is not None:
                if direction == 'long':
                    raw_pnl = (exit_price - position['entry']) * point_value
                else:
                    raw_pnl = (position['entry'] - exit_price) * point_value
                
                # Combine partial + remaining
                if position.get('partial_exit'):
                    remaining_pct = 1 - partial_pct
                    raw_pnl = position['partial_pnl'] + (raw_pnl * remaining_pct)
                
                pnl = raw_pnl - commission
                
                trades.append({
                    'entry': position['entry'],
                    'exit': exit_price,
                    'exit_type': exit_type,
                    'pnl': pnl,
                    'direction': direction
                })
                position = None
                continue
        
        # Entry logic
        if position is None:
            subset_5m = data_5m.iloc[max(0, idx - 250):idx + 1].copy()
            
            if len(subset_5m) < 210:
                continue
            
            latest = subset_5m.iloc[-1]
            price = latest['close']
            ema200 = latest['ema_200']
            ema50 = latest['ema_50']
            rsi = latest['rsi']
            atr = latest['atr']
            
            if pd.isna(ema200) or pd.isna(ema50) or pd.isna(atr) or atr == 0:
                continue
            
            # Strong trend filter
            bullish = price > ema200 and ema50 > ema200 and rsi > 50
            bearish = price < ema200 and ema50 < ema200 and rsi < 50
            
            if not (bullish or bearish):
                continue
            
            direction = 'long' if bullish else 'short'
            
            # Entry score calculation
            score = 0.0
            
            # Trend alignment (40%)
            if direction == 'long':
                if price > ema200: score += 0.15
                if ema50 > ema200: score += 0.15
                if rsi > 55: score += 0.10
            else:
                if price < ema200: score += 0.15
                if ema50 < ema200: score += 0.15
                if rsi < 45: score += 0.10
            
            # RSI not extreme (20%)
            if direction == 'long' and 35 < rsi < 65:
                score += 0.20
            elif direction == 'short' and 35 < rsi < 65:
                score += 0.20
            
            # Volume (15%)
            if 'volume' in subset_5m.columns:
                vol_avg = subset_5m['volume'].tail(20).mean()
                if latest['volume'] > vol_avg * 0.9:
                    score += 0.15
            else:
                score += 0.10
            
            # ATR adequate (15%)
            atr_avg = subset_5m['atr'].tail(20).mean()
            if atr > atr_avg * 0.7:
                score += 0.15
            
            # Momentum (10%)
            momentum = (subset_5m['close'].iloc[-1] - subset_5m['close'].iloc[-5]) / subset_5m['close'].iloc[-5] * 100
            if (direction == 'long' and momentum > 0) or (direction == 'short' and momentum < 0):
                score += 0.10
            
            if score < min_score:
                continue
            
            # Calculate levels
            sl_dist = atr * sl_mult
            
            if direction == 'long':
                sl = price - sl_dist
                tp1 = price + (sl_dist * tp1_ratio)
                tp2 = price + (sl_dist * tp2_ratio)
            else:
                sl = price + sl_dist
                tp1 = price - (sl_dist * tp1_ratio)
                tp2 = price - (sl_dist * tp2_ratio)
            
            position = {
                'direction': direction,
                'entry': price,
                'sl': sl,
                'tp1': tp1,
                'tp2': tp2,
                'entry_idx': idx,
                'partial_exit': False,
                'partial_pnl': 0
            }
    
    # Stats
    if not trades:
        return {'trades': 0}
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    
    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
    
    return {
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(trades) * 100,
        'total_pnl': sum(t['pnl'] for t in trades),
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0,
        'avg_win': gross_profit / len(wins) if wins else 0,
        'avg_loss': gross_loss / len(losses) if losses else 0,
        'sl_exits': len([t for t in trades if t['exit_type'] == 'STOP_LOSS']),
        'be_exits': len([t for t in trades if t['exit_type'] == 'BREAKEVEN']),
        'tp2_exits': len([t for t in trades if t['exit_type'] == 'TAKE_PROFIT_2']),
        'timeout_exits': len([t for t in trades if t['exit_type'] == 'TIMEOUT']),
    }


def main():
    print("=== OPTIMIZED STRATEGY PARAMETER SWEEP ===\n")
    
    mes_5m = pd.read_csv('/workspaces/Ai-bot/data/MES_5m.csv', parse_dates=['datetime'])
    mes_1m = pd.read_csv('/workspaces/Ai-bot/data/MES_1m.csv', parse_dates=['datetime'])
    print(f"Loaded: {len(mes_5m)} 5M bars\n")
    
    best = {'pf': 0, 'params': {}}
    
    # Parameter combinations to test
    configs = [
        # (sl_mult, tp1_ratio, tp2_ratio, min_score)
        (1.2, 1.0, 1.8, 0.50),  # Tight SL, quick TP1
        (1.3, 1.2, 2.0, 0.50),  # Balanced
        (1.3, 1.0, 2.0, 0.55),  # Higher min score
        (1.5, 1.5, 2.5, 0.50),  # Original ratios
        (1.2, 1.3, 2.2, 0.50),  # Tight SL, better R:R
        (1.0, 1.0, 1.5, 0.55),  # Very tight
    ]
    
    print(f"{'SL':>4} {'TP1':>5} {'TP2':>5} {'Min':>5} || {'Trades':>6} {'WR%':>6} {'PnL':>8} {'PF':>5} | Exit Breakdown")
    print("-" * 95)
    
    for sl, tp1, tp2, ms in configs:
        r = backtest_optimized(mes_5m, mes_1m, sl_mult=sl, tp1_ratio=tp1, tp2_ratio=tp2, min_score=ms)
        
        if r['trades'] > 0:
            status = "✅" if r['profit_factor'] >= 1.0 else "❌"
            print(f"{sl:>4.1f} {tp1:>5.1f} {tp2:>5.1f} {ms:>5.2f} || "
                  f"{r['trades']:>6} {r['win_rate']:>5.1f}% ${r['total_pnl']:>7.0f} {r['profit_factor']:>5.2f} "
                  f"| SL:{r['sl_exits']} BE:{r['be_exits']} TP2:{r['tp2_exits']} TO:{r['timeout_exits']} {status}")
            
            if r['profit_factor'] > best['pf']:
                best = {'pf': r['profit_factor'], 'params': {'sl': sl, 'tp1': tp1, 'tp2': tp2, 'ms': ms}, 'results': r}
    
    print("\n" + "=" * 50)
    if best['pf'] >= 1.0:
        print(f"✅ BEST PROFITABLE CONFIG: PF={best['pf']:.2f}")
        print(f"   SL={best['params']['sl']}, TP1={best['params']['tp1']}, TP2={best['params']['tp2']}")
        print(f"   Win Rate: {best['results']['win_rate']:.1f}% | PnL: ${best['results']['total_pnl']:.0f}")
    else:
        print(f"⚠️ Best PF: {best['pf']:.2f} (not yet profitable)")
        print(f"   Closest config: SL={best['params'].get('sl')}, TP1={best['params'].get('tp1')}")


if __name__ == "__main__":
    main()
