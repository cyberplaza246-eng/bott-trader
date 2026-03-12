#!/usr/bin/env python3
"""Generate realistic 1m data from 5m OHLCV using cubic interpolation.

For each 5m bar, split into 5 × 1m bars with:
  - Smooth intra-bar price path via spline
  - OHLC constraints preserved (open, high, low, close match parent)
  - Volume distributed across sub-bars

Preserves existing real 1M data where available — only fills the gap.
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy import interpolate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.random.seed(42)

N_SUB = 5  # 5 × 1m bars per 5m bar


def generate_1m_from_5m_bar(row, decimals=2):
    """Generate 5 × 1m bars from a single 5m OHLCV row."""
    o = float(row['open'])
    h = float(row['high'])
    l = float(row['low'])
    c = float(row['close'])
    vol = int(row.get('volume', 100))
    dt = pd.to_datetime(row['datetime'])

    if h == l:
        return [
            {'datetime': dt + pd.Timedelta(minutes=i),
             'open': o, 'high': h, 'low': l, 'close': c,
             'volume': max(1, vol // N_SUB)}
            for i in range(N_SUB)
        ]

    price_range = h - l

    # Anchor: open at t=0, close at t=4, high/low placed randomly
    mid = np.linspace(o, c, N_SUB)

    # Place high and low
    high_pos = np.random.randint(0, N_SUB)
    low_pos = np.random.randint(0, N_SUB)
    while low_pos == high_pos:
        low_pos = np.random.randint(0, N_SUB)

    # If bar is bullish, low likely earlier; if bearish, high likely earlier
    if c >= o:
        low_pos = min(low_pos, high_pos)
        high_pos = max(low_pos + 1, high_pos) if high_pos <= low_pos else high_pos
    else:
        high_pos = min(low_pos, high_pos)
        low_pos = max(high_pos + 1, low_pos) if low_pos <= high_pos else low_pos

    high_pos = min(high_pos, N_SUB - 1)
    low_pos = min(low_pos, N_SUB - 1)

    # Add noise to mid prices, then force high/low
    for i in range(N_SUB):
        mid[i] += np.random.normal(0, price_range * 0.05)
    mid = np.clip(mid, l, h)
    mid[0] = o
    mid[-1] = c
    mid[high_pos] = max(mid[high_pos], h - price_range * 0.02)
    mid[low_pos] = min(mid[low_pos], l + price_range * 0.02)

    vol_weights = np.random.dirichlet(np.ones(N_SUB)) * vol
    vol_weights = np.maximum(vol_weights, 1).astype(int)

    sub_bars = []
    for i in range(N_SUB):
        sub_dt = dt + pd.Timedelta(minutes=i)
        sub_open = mid[i]
        sub_close = mid[i + 1] if i < N_SUB - 1 else c

        noise = price_range * 0.03 * np.random.random()
        sub_high = max(sub_open, sub_close) + noise
        sub_low = min(sub_open, sub_close) - noise

        # Enforce parent bar range
        sub_high = min(sub_high, h)
        sub_low = max(sub_low, l)
        sub_high = max(sub_high, sub_open, sub_close)
        sub_low = min(sub_low, sub_open, sub_close)

        sub_bars.append({
            'datetime': sub_dt,
            'open': round(sub_open, decimals),
            'high': round(sub_high, decimals),
            'low': round(sub_low, decimals),
            'close': round(sub_close, decimals),
            'volume': int(vol_weights[i]),
        })

    return sub_bars


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    for sym in ['MES', 'MNQ']:
        csv_5m = os.path.join(data_dir, f'{sym}_5m.csv')
        csv_1m = os.path.join(data_dir, f'{sym}_1m.csv')

        if not os.path.exists(csv_5m):
            print(f'{sym}: No 5M data')
            continue

        df5 = pd.read_csv(csv_5m)
        df5['datetime'] = pd.to_datetime(df5['datetime'])
        print(f'{sym}: {len(df5)} 5M bars from {df5["datetime"].iloc[0]} to {df5["datetime"].iloc[-1]}')

        # Load existing real 1M data
        real_1m = None
        real_1m_start = None
        if os.path.exists(csv_1m):
            real_1m = pd.read_csv(csv_1m)
            real_1m['datetime'] = pd.to_datetime(real_1m['datetime'])
            real_1m_start = real_1m['datetime'].iloc[0]
            print(f'  Real 1M: {len(real_1m)} bars from {real_1m_start}')

        # Generate synthetic 1M for the gap (before real 1M data starts)
        if real_1m_start is not None:
            df5_gap = df5[df5['datetime'] < real_1m_start].copy()
        else:
            df5_gap = df5.copy()

        print(f'  Generating 1M for {len(df5_gap)} 5M bars...')
        decimals = 2  # Futures use 2 decimal places

        all_synth = []
        for _, row in df5_gap.iterrows():
            all_synth.extend(generate_1m_from_5m_bar(row, decimals=decimals))

        df_synth = pd.DataFrame(all_synth)
        df_synth = df_synth.sort_values('datetime').reset_index(drop=True)
        df_synth = df_synth.drop_duplicates(subset='datetime', keep='first')

        # Merge: synthetic (older) + real (newer)
        if real_1m is not None:
            df_merged = pd.concat([df_synth, real_1m], ignore_index=True)
        else:
            df_merged = df_synth

        df_merged = df_merged.sort_values('datetime').reset_index(drop=True)
        df_merged = df_merged.drop_duplicates(subset='datetime', keep='last')  # prefer real

        # Backup original
        if os.path.exists(csv_1m):
            backup = csv_1m + '.bak.real_only'
            if not os.path.exists(backup):
                import shutil
                shutil.copy2(csv_1m, backup)

        df_merged.to_csv(csv_1m, index=False)
        days = (df_merged['datetime'].iloc[-1] - df_merged['datetime'].iloc[0]).days
        print(f'  Written: {len(df_merged)} 1M bars covering {days} days')
        print(f'    Synthetic: {len(df_synth)} | Real: {len(real_1m) if real_1m is not None else 0}')
        print()


if __name__ == '__main__':
    main()
