import sys, time, logging
sys.path.insert(0, '.')
logging.disable(logging.CRITICAL)
import pandas as pd
from src.strategies.clean_scalper_verify import run_backtest

WINDOWS = [
    ('2024-01-01', '2024-04-01', 'Q1 2024'),
    ('2024-07-01', '2024-10-01', 'Q3 2024'),
    ('2025-01-01', '2025-04-01', 'Q1 2025'),
    ('2025-10-01', '2026-01-01', 'Q4 2025'),
]

for symbol in ['MNQ', 'NQ']:
    df_5m_full = pd.read_csv(f'data/{symbol}_5m.csv', parse_dates=['datetime'])
    df_1m_full = pd.read_csv(f'data/{symbol}_1m.csv', parse_dates=['datetime'])
    for start, end, label in WINDOWS:
        df_5m = df_5m_full[(df_5m_full['datetime'] >= start) & (df_5m_full['datetime'] < end)].reset_index(drop=True)
        df_1m = df_1m_full[(df_1m_full['datetime'] >= start) & (df_1m_full['datetime'] < end)].reset_index(drop=True)
        t0 = time.time()
        results = run_backtest(df_5m, df_1m, symbol, min_confirmations=4, max_contracts=2, use_advanced_sweep=True)
        elapsed = time.time() - t0
        print(f"{symbol} {label}: elapsed={elapsed:.0f}s trades={results['total_trades']} "
              f"win_rate={results['win_rate']:.1f}% net_pnl={results['net_pnl']:.2f} "
              f"profit_factor={results['profit_factor']:.3f}", flush=True)
