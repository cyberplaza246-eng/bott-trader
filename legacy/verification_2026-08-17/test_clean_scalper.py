import sys, time, logging; sys.path.insert(0,'.')
logging.disable(logging.CRITICAL)
import pandas as pd
from src.strategies.clean_scalper_verify import run_backtest

df_5m = pd.read_csv('data/MES_5m.csv', parse_dates=['datetime'])
df_1m = pd.read_csv('data/MES_1m.csv', parse_dates=['datetime'])
df_5m = df_5m[(df_5m['datetime']>='2024-01-01') & (df_5m['datetime']<'2024-04-01')].reset_index(drop=True)
df_1m = df_1m[(df_1m['datetime']>='2024-01-01') & (df_1m['datetime']<'2024-04-01')].reset_index(drop=True)
print(f'5m: {len(df_5m)}, 1m: {len(df_1m)}', flush=True)

t0 = time.time()
results = run_backtest(df_5m, df_1m, 'MES', min_confirmations=4, max_contracts=2, use_advanced_sweep=True)
print(f'elapsed: {time.time()-t0:.1f}s', flush=True)
print('trades:', results['total_trades'], flush=True)
print('win_rate:', results['win_rate'], flush=True)
print('net_pnl:', results['net_pnl'], flush=True)
print('profit_factor:', results['profit_factor'], flush=True)
