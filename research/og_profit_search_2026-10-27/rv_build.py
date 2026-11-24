import os, sys, time, pickle
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(H,'code'))
import backtest_revamp_live_stack as B
import pandas as pd, numpy as np
t0=time.time()
m1,m5=B.load_bars(os.path.join(H,'data','MNQ_1m.csv'))
ens=B.ensemble_signals(m5); b=B.featured(m5); adds=B.addon_rows(b)
print('built',len(m1),len(m5),len(ens),{k:len(v) for k,v in adds.items()},round(time.time()-t0),flush=True)
pickle.dump(dict(m1=m1,m5=m5,ens=ens,adds=adds),open(os.path.join(H,'out','revamp_base.pkl'),'wb'))
sig=pd.concat([ens]+list(adds.values()))
t0=time.time()
tr=B.in_window(B.simulate(m1,m5.index,sig,B.Rules(overnight_sleeves='all')))
print('sim sec',round(time.time()-t0),B.stats(tr))
for k,v in B.by_period(tr).items(): print(k,B.fmt(v))
