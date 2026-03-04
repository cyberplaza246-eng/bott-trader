#!/usr/bin/env python3
"""Diagnose which filter layer is killing trades in the sweep pipeline."""
import logging
logging.disable(logging.CRITICAL)
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.ai.liquidity_sweep import LiquiditySweepAnalyzer

analyzer = LiquiditySweepAnalyzer()

for pair in ['EUR/USD', 'GBP/USD', 'USD/JPY']:
    csv5 = f'data/{pair.replace("/","_")}_5m.csv'
    csv1 = f'data/{pair.replace("/","_")}_1m.csv'
    if not os.path.exists(csv5):
        print(f"{pair}: no data"); continue
    
    df5 = pd.read_csv(csv5); df5['datetime'] = pd.to_datetime(df5['datetime'])
    df5 = analyzer.calculate_indicators(df5)
    
    df1 = None
    if os.path.exists(csv1):
        df1 = pd.read_csv(csv1); df1['datetime'] = pd.to_datetime(df1['datetime'])
        df1 = analyzer.calculate_indicators(df1)
    
    # Count filter kills at each layer by sampling every 5 candles on 5M
    total = 0
    kills = {'market_cond': 0, 'no_bias': 0, 'low_vol': 0, 'adx': 0, 
             'no_sweep': 0, 'rsi_filter': 0, '5m_invalid': 0,
             'no_mss': 0, 'no_displace': 0, 'rr_fail': 0, 'passed': 0}
    
    lookback = 200
    step = 5  # Check every 5th 5M candle
    
    for idx in range(lookback, len(df5), step):
        total += 1
        
        # Simulate: take df_5m up to this point
        df5_slice = df5.iloc[max(0, idx-200):idx+1].copy()
        
        # Get corresponding 1M data (approximate: each 5M = 5 1M candles)
        if df1 is not None:
            dt_5m = df5_slice.iloc[-1]['datetime']
            mask = df1['datetime'] <= dt_5m
            df1_slice = df1[mask].tail(300).copy()
        else:
            df1_slice = None
        
        # Layer 0: market conditions
        if df1_slice is not None and len(df1_slice) >= 30:
            ok, reason = analyzer.check_market_conditions(df1_slice, pair)
            if not ok:
                kills['market_cond'] += 1
                continue
        
        # Layer 1: regime/bias
        regime = analyzer.detect_regime(df5_slice)
        
        if regime['regime'] == 'low_volatility':
            kills['low_vol'] += 1
            continue
        
        if regime['bias'] is None:
            # Check if it was ADX-blocked
            # Re-check without ADX filter
            adx = regime.get('adx', 0)
            if adx < analyzer.ADX_MIN_BIAS and regime['regime'] != 'range':
                kills['adx'] += 1
            else:
                kills['no_bias'] += 1
            continue
        
        # Layer 2: sweep detection
        if df1_slice is not None and len(df1_slice) >= 30:
            sweep = analyzer.detect_sweep(df1_slice, regime['bias'], regime_info=regime)
            if not sweep['detected']:
                if not sweep.get('fivem_invalidation_held', True):
                    kills['5m_invalid'] += 1
                elif 'RSI' in sweep.get('details', ''):
                    kills['rsi_filter'] += 1
                else:
                    kills['no_sweep'] += 1
                continue
            
            # Layer 3: MSS
            mss = analyzer.detect_mss(df1_slice, sweep)
            if not mss['confirmed']:
                if 'displacement' in mss.get('details', '').lower():
                    kills['no_displace'] += 1
                else:
                    kills['no_mss'] += 1
                continue
            
            # Layer 4: R:R
            latest_1m = df1_slice.iloc[-1]
            regime['atr'] = float(latest_1m.get('atr', 0) or 0)
            rr = analyzer.calculate_risk_reward(sweep, mss, regime, pair)
            if rr is None:
                kills['rr_fail'] += 1
                continue
            
            kills['passed'] += 1
        else:
            kills['no_sweep'] += 1
    
    print(f"\n{'='*50}")
    print(f"  {pair} — {total} sample points (every 5th 5M candle)")
    print(f"{'='*50}")
    for k, v in kills.items():
        pct = v/total*100 if total > 0 else 0
        bar = '█' * int(pct/2)
        print(f"  {k:15s}: {v:5d} ({pct:5.1f}%) {bar}")
    
    days = total * 5 / (12 * 24)  # rough days estimate
    passed = kills['passed']
    per_day = passed / days if days > 0 else 0
    print(f"  ---")
    print(f"  Signals/day: {per_day:.1f} (target: 8+)")
