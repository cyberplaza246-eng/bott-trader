import pandas as pd, numpy as np
import run_nq as R, engine as E
from data import load_1m, resample_5m
from signals import build_signals, live_atr_sma14
d1=load_1m(); df5=resample_5m(d1); sigs=build_signals(df5)
sigs=sigs[(sigs.signal_time>=R.START)&(sigs.signal_time<R.END)]
B=E.Bars(d1,df5,live_atr_sma14(df5))
E.LOCK_MODE="usd_old"; E.PV=20.0
E.LOCK1_TRIG,E.LOCK1,E.LOCK2_TRIG,E.LOCK2=1000.0,150.0,1500.0,500.0
att=E.attempts(B,sigs,"base",False)
rows=[]
for dl in (500, 2000, 1e9):
    cfg=dict(max_conc=2, extra_conf=0.0, cd_win=25, cd_loss=25, daily_loss=dl, sizing=False)
    g=R.gate(att,20.0,14.5,cfg)
    for per,lo,hi in (("full",R.START,R.END),("holdout",R.HOLD,R.END)):
        m=R.metrics(g,lo,hi); m.update(daily_loss=dl,period=per); rows.append(m)
pd.set_option("display.width",250)
print(pd.DataFrame(rows).to_string(index=False))
