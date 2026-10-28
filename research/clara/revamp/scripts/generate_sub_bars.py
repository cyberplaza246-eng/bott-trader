#!/usr/bin/env python3
"""Generate realistic 1m and 5m data from 1h OHLCV using cubic interpolation.

For each 1h bar:
  - Split into 60 × 1m bars (or 12 × 5m bars)
  - Use cubic spline to create smooth intra-bar price path
  - Ensure OHLC constraints: first open = bar open, last close = bar close,
    max high = bar high, min low = bar low
  - Distribute volume randomly across sub-bars
"""
import os, sys
import pandas as pd
import numpy as np
from scipy import interpolate

np.random.seed(42)

def generate_sub_bars(row, n_sub_bars):
    """Generate n_sub_bars from a single 1h OHLCV row."""
    o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
    vol = int(row.get('volume', 100))
    dt = pd.to_datetime(row['datetime'])
    
    if h == l:  # No range, flat
        sub_bars = []
        for i in range(n_sub_bars):
            sub_dt = dt + pd.Timedelta(minutes=i * (60 // n_sub_bars))
            sub_bars.append({
                'datetime': sub_dt, 'open': o, 'high': h, 'low': l, 'close': c,
                'volume': max(1, vol // n_sub_bars)
            })
        return sub_bars
    
    # Create anchor points for spline: open, random walk touching high/low, close
    n_anchors = min(8, n_sub_bars)
    t_anchors = np.sort(np.random.choice(range(1, n_sub_bars - 1), size=n_anchors - 2, replace=False))
    t_anchors = np.concatenate([[0], t_anchors, [n_sub_bars - 1]])
    
    # Place high and low at random positions
    high_pos = np.random.randint(1, n_sub_bars - 1)
    low_pos = np.random.randint(1, n_sub_bars - 1)
    while low_pos == high_pos:
        low_pos = np.random.randint(1, n_sub_bars - 1)
    
    # Generate random prices at anchor points
    price_range = h - l
    prices_at_anchors = []
    for i, t in enumerate(t_anchors):
        if t == 0:
            prices_at_anchors.append(o)
        elif t == n_sub_bars - 1:
            prices_at_anchors.append(c)
        else:
            # Random price within the bar range
            p = l + np.random.random() * price_range
            prices_at_anchors.append(p)
    
    prices_at_anchors = np.array(prices_at_anchors)
    
    # Create spline through anchor points
    if len(t_anchors) >= 4:
        cs = interpolate.CubicSpline(t_anchors, prices_at_anchors, bc_type='natural')
    else:
        cs = interpolate.interp1d(t_anchors, prices_at_anchors, kind='linear', fill_value='extrapolate')
    
    all_t = np.arange(n_sub_bars)
    mid_prices = cs(all_t)
    
    # Force high and low to appear
    mid_prices[high_pos] = h
    mid_prices[low_pos] = l
    
    # Clip to bar range
    mid_prices = np.clip(mid_prices, l, h)
    mid_prices[0] = o
    mid_prices[-1] = c
    
    # Generate sub-bar OHLCV
    sub_bars = []
    vol_weights = np.random.dirichlet(np.ones(n_sub_bars)) * vol
    vol_weights = np.maximum(vol_weights, 1).astype(int)
    
    interval_min = 60 // n_sub_bars
    
    for i in range(n_sub_bars):
        sub_dt = dt + pd.Timedelta(minutes=i * interval_min)
        
        sub_open = mid_prices[i]
        sub_close = mid_prices[min(i + 1, n_sub_bars - 1)] if i < n_sub_bars - 1 else c
        
        # Add noise for intra-sub-bar high/low
        noise = price_range * 0.05 * np.random.random()
        sub_high = max(sub_open, sub_close) + noise
        sub_low = min(sub_open, sub_close) - noise
        
        # Clip to parent bar range
        sub_high = min(sub_high, h)
        sub_low = max(sub_low, l)
        sub_high = max(sub_high, sub_open, sub_close)
        sub_low = min(sub_low, sub_open, sub_close)
        
        sub_bars.append({
            'datetime': sub_dt,
            'open': round(sub_open, 5 if 'JPY' not in str(row.get('pair', '')) else 3),
            'high': round(sub_high, 5 if 'JPY' not in str(row.get('pair', '')) else 3),
            'low': round(sub_low, 5 if 'JPY' not in str(row.get('pair', '')) else 3),
            'close': round(sub_close, 5 if 'JPY' not in str(row.get('pair', '')) else 3),
            'volume': int(vol_weights[i]),
        })
    
    return sub_bars


def generate_from_1h(input_csv, pair, output_1m, output_5m):
    """Generate 1m and 5m CSVs from 1h data."""
    df = pd.read_csv(input_csv)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"  {pair}: {len(df)} 1h bars → generating 1m and 5m...")
    
    is_jpy = 'JPY' in pair
    decimals = 3 if is_jpy else 5
    
    # Generate 1m bars
    all_1m = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        row_dict['pair'] = pair
        subs = generate_sub_bars(row_dict, 60)
        for s in subs:
            s['open'] = round(s['open'], decimals)
            s['high'] = round(s['high'], decimals)
            s['low'] = round(s['low'], decimals)
            s['close'] = round(s['close'], decimals)
        all_1m.extend(subs)
    
    df_1m = pd.DataFrame(all_1m)
    df_1m = df_1m.sort_values('datetime').reset_index(drop=True)
    # Remove duplicates
    df_1m = df_1m.drop_duplicates(subset='datetime', keep='first')
    df_1m.to_csv(output_1m, index=False)
    print(f"    1m: {len(df_1m)} candles → {output_1m}")
    
    # Generate 5m by resampling 1m
    df_1m['datetime'] = pd.to_datetime(df_1m['datetime'])
    df_1m_indexed = df_1m.set_index('datetime')
    
    df_5m = df_1m_indexed.resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna().reset_index()
    
    for col in ['open', 'high', 'low', 'close']:
        df_5m[col] = df_5m[col].round(decimals)
    
    df_5m.to_csv(output_5m, index=False)
    print(f"    5m: {len(df_5m)} candles → {output_5m}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    generate_from_1h(
        'data/USD_JPY_1h.csv', 'USD/JPY',
        'data/USD_JPY_1m.csv', 'data/USD_JPY_5m.csv'
    )
    print("\nDone!")
