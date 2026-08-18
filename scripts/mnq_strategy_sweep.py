#!/usr/bin/env python3
"""
MNQ strategy comparison — Friday trading enabled.
Tests EMA crossover, MTF scalping, enhanced filters, and sweep-gated ensemble.
"""
import json
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rithmic_backtest import SYMBOL_CONFIGS, compute_stats, param_sweep, run_backtest
from enhanced_futures_backtest import enhanced_futures_backtest
from scripts.backtest_mtf_scalping import MultiTimeframeBacktester, load_data


INITIAL_BALANCE = 50000
SYMBOL = 'MNQ'
RESULTS_PATH = 'data/mnq_strategy_sweep_results.json'


def _score(stats: dict) -> float:
    """Rank strategies: PF first, then PnL, then win rate."""
    if not stats or stats.get('trades', 0) < 5:
        return -999.0
    pf = stats.get('profit_factor', 0) or 0
    pnl = stats.get('total_pnl', stats.get('pnl', 0)) or 0
    wr = stats.get('win_rate', 0) or 0
    return pf * 1000 + pnl * 0.01 + wr


def run_ema_sweep(df: pd.DataFrame) -> dict:
    cfg = SYMBOL_CONFIGS[SYMBOL].copy()
    cfg['skip_friday'] = False  # user requires Friday trading

    trades, _ = run_backtest(df.copy(), SYMBOL, INITIAL_BALANCE, cfg)
    baseline = compute_stats(trades, INITIAL_BALANCE)
    best = param_sweep(df.copy(), SYMBOL, INITIAL_BALANCE)

    return {
        'name': 'ema_crossover',
        'baseline': baseline,
        'best_params': best,
        'skip_friday': False,
    }


def run_mtf_scalping() -> dict:
    df_1m, df_5m = load_data(SYMBOL)
    bt = MultiTimeframeBacktester(SYMBOL)
    stats = bt.run(df_1m, df_5m)
    return {'name': 'mtf_scalping', 'stats': stats, 'skip_friday': True}


def run_enhanced(df: pd.DataFrame) -> dict:
    trades, final_balance = enhanced_futures_backtest(df.copy(), SYMBOL, INITIAL_BALANCE)
    stats = compute_stats(trades, INITIAL_BALANCE)
    if stats:
        stats['final_balance'] = final_balance
    return {'name': 'enhanced_ema', 'stats': stats, 'skip_friday': True}


def run_ensemble_subset(df_5m: pd.DataFrame, df_1m: pd.DataFrame) -> list:
    """Sweep-gated ensemble on recent data (full history is too slow bar-by-bar)."""
    from src.backtest.backtest_engine import BacktestEngine

    # Use last ~30 trading days for ensemble (still representative, much faster)
    cutoff = df_5m['datetime'].max() - pd.Timedelta(days=45)
    sub_5m = df_5m[df_5m['datetime'] >= cutoff].copy().reset_index(drop=True)
    sub_1m = df_1m[df_1m['datetime'] >= cutoff].copy().reset_index(drop=True)

    results = []
    for threshold in (0.55, 0.60, 0.65, 0.68):
        for min_agree in (3, 4):
            engine = BacktestEngine(initial_balance=INITIAL_BALANCE, slippage_pips=0.5)
            engine.learner = engine.learner  # fresh state
            out = engine.run_backtest(
                sub_5m,
                SYMBOL,
                timeframe_key='5m',
                df_1m=sub_1m,
                confidence_threshold=threshold,
                min_agreement=min_agree,
            )
            if not out:
                continue
            row = {
                'name': 'sweep_ensemble',
                'confidence': threshold,
                'min_agreement': min_agree,
                'bars': len(sub_5m),
                'trades': out.get('total_trades', 0),
                'win_rate': out.get('win_rate', 0) * 100,
                'profit_factor': out.get('profit_factor', 0),
                'total_pnl': out.get('total_pnl', 0),
                'max_drawdown': out.get('max_drawdown', 0),
            }
            results.append(row)
            print(
                f"  ensemble conf={threshold:.2f} agree={min_agree}: "
                f"{row['trades']} trades PF={row['profit_factor']:.2f} "
                f"PnL=${row['total_pnl']:,.0f} WR={row['win_rate']:.1f}%"
            )
    return results


def pick_winner(all_results: dict) -> dict:
    candidates = []

    ema = all_results.get('ema_crossover', {})
    if ema.get('best_params'):
        b = ema['best_params']
        candidates.append({
            'strategy': 'ema_crossover',
            'profit_factor': b['profit_factor'],
            'win_rate': b['win_rate'],
            'total_pnl': b['total_pnl'],
            'trades': b['trades'],
            'params': {
                'sl_atr_mult': b['sl'],
                'tp_r_mult': b['r_mult'],
                'cooldown_bars': b['cooldown'],
                'adx_min': b['adx'],
                'trailing': b['trail'],
                'bad_hours': [0, 4, 21],
                'skip_friday': False,
            },
        })

    mtf = all_results.get('mtf_scalping', {}).get('stats')
    if mtf and mtf.get('total_trades', 0) >= 5:
        candidates.append({
            'strategy': 'mtf_scalping',
            'profit_factor': mtf['profit_factor'],
            'win_rate': mtf['win_rate'],
            'total_pnl': mtf['total_pnl'],
            'trades': mtf['total_trades'],
            'params': {'tp_mult': 1.5, 'atr_mult': 1.2, 'adx_threshold': 18, 'skip_friday': True},
        })

    enh = all_results.get('enhanced_ema', {}).get('stats')
    if enh and enh.get('trades', 0) >= 5:
        candidates.append({
            'strategy': 'enhanced_ema',
            'profit_factor': enh['profit_factor'],
            'win_rate': enh['win_rate'],
            'total_pnl': enh['total_pnl'],
            'trades': enh['trades'],
            'params': {'adx_threshold': 30, 'skip_friday': True},
        })

    for row in all_results.get('ensemble', []):
        if row.get('trades', 0) >= 5:
            candidates.append({
                'strategy': 'sweep_ensemble',
                'profit_factor': row['profit_factor'],
                'win_rate': row['win_rate'],
                'total_pnl': row['total_pnl'],
                'trades': row['trades'],
                'params': {
                    'confidence': row['confidence'],
                    'min_agreement': row['min_agreement'],
                    'fallback_enabled': False,
                    'skip_friday': False,
                },
            })

    if not candidates:
        return {}

    candidates.sort(key=lambda c: _score(c), reverse=True)
    return candidates[0]


def main():
    print('=' * 72)
    print('  MNQ STRATEGY SWEEP — Friday trading ENABLED')
    print('=' * 72)

    data_5m = f'data/{SYMBOL}_5m.csv'
    data_1m = f'data/{SYMBOL}_1m.csv'
    if not os.path.exists(data_5m):
        print(f'Missing {data_5m}')
        sys.exit(1)

    df = pd.read_csv(data_5m, parse_dates=['datetime'])
    df = df.set_index('datetime')
    print(f'\nData: {len(df)} x 5m bars | {df.index[0].date()} → {df.index[-1].date()}')

    all_results = {}

    print('\n[1/4] EMA crossover + parameter sweep (Friday ON)...')
    all_results['ema_crossover'] = run_ema_sweep(df)

    print('\n[2/4] MTF scalping (1M+5M)...')
    try:
        all_results['mtf_scalping'] = run_mtf_scalping()
        s = all_results['mtf_scalping']['stats']
        print(f"  MTF: {s['total_trades']} trades PF={s['profit_factor']:.2f} "
              f"PnL=${s['total_pnl']:,.0f} WR={s['win_rate']:.1f}%")
    except Exception as e:
        print(f'  MTF failed: {e}')
        all_results['mtf_scalping'] = {'error': str(e)}

    print('\n[3/4] Enhanced EMA filters...')
    all_results['enhanced_ema'] = run_enhanced(df)
    es = all_results['enhanced_ema'].get('stats')
    if es:
        print(f"  Enhanced: {es['trades']} trades PF={es['profit_factor']:.2f} "
              f"PnL=${es['total_pnl']:,.0f} WR={es['win_rate']:.1f}%")

    print('\n[4/4] Sweep-gated ensemble (recent 45 days)...')
    df_5m_raw = pd.read_csv(data_5m, parse_dates=['datetime'])
    df_1m_raw = pd.read_csv(data_1m, parse_dates=['datetime']) if os.path.exists(data_1m) else None
    if df_1m_raw is not None:
        all_results['ensemble'] = run_ensemble_subset(df_5m_raw, df_1m_raw)
    else:
        print('  Skipped ensemble — no 1M data')
        all_results['ensemble'] = []

    winner = pick_winner(all_results)
    all_results['winner'] = winner
    all_results['generated_at'] = datetime.utcnow().isoformat()

    os.makedirs('data', exist_ok=True)
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)

    print('\n' + '=' * 72)
    print('  WINNER')
    print('=' * 72)
    if winner:
        print(f"  Strategy:      {winner['strategy']}")
        print(f"  Profit Factor: {winner['profit_factor']:.2f}")
        print(f"  Win Rate:      {winner['win_rate']:.1f}%")
        print(f"  Total PnL:     ${winner['total_pnl']:,.2f}")
        print(f"  Trades:        {winner['trades']}")
        print(f"  Params:        {json.dumps(winner['params'])}")
    else:
        print('  No strategy met minimum trade count.')
    print(f"\n  Full results: {RESULTS_PATH}")


if __name__ == '__main__':
    main()
