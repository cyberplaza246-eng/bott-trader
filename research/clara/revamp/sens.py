import sys,pandas as pd,pickle
sys.path.insert(0,'.');sys.path.insert(0,'scripts')
import backtest_revamp_live_stack as B
m1,m5=B.load_bars('/workspace/clara/MNQ_1m.csv');b=B.featured(m5);adds=B.addon_rows(b)
base=dict(COMM_RT=4.0,SLIP=0.25,LOCK_ACTIVATE=100.0,LOCK_KEEP=20.0,LOCK_TRAIL_START=150.0,LOCK_KEEP_PCT=0.5,TREND_TP_R=1.5,BREAKOUT_TP_R=2.0)
V={'BASE':{},'slip2ticks':{'SLIP':0.5},'lock-20%':{'LOCK_ACTIVATE':80.,'LOCK_KEEP':16.,'LOCK_TRAIL_START':120.},'lock+20%':{'LOCK_ACTIVATE':120.,'LOCK_KEEP':24.,'LOCK_TRAIL_START':180.},
 'keep_pct0.4':{'LOCK_KEEP_PCT':0.4},'keep_pct0.6':{'LOCK_KEEP_PCT':0.6},'TP_R-20%':{'TREND_TP_R':1.2,'BREAKOUT_TP_R':1.6},'TP_R+20%':{'TREND_TP_R':1.8,'BREAKOUT_TP_R':2.4}}
out={}
for k,v in V.items():
    for a,x in {**base,**v}.items(): setattr(B,a,x)
    sig=pd.concat([B.ensemble_signals(m5)]+list(adds.values()))
    tr=B.in_window(B.simulate(m1,m5.index,sig,B.Rules(overnight_sleeves='all')))
    out[k]=pd.DataFrame([t.__dict__ for t in tr]);print(k,len(tr),round(sum(t.pnl for t in tr)),flush=True)
pickle.dump(out,open('sens.pkl','wb'))
