import os, sys
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(H,'code'))
import backtest_revamp_live_stack as B
m1,m5=B.load_bars(os.path.join(H,'data','MNQ_1m.csv'))
d=m5.reset_index().rename(columns={'index':'datetime'})
d.to_csv(os.path.join(H,'data','bars5m_clean.csv'),index=False)
print(len(d),d.datetime.min(),d.datetime.max(),flush=True)
