import sys,os,pandas as pd,pickle
sys.path.insert(0,'.');sys.path.insert(0,'scripts')
import backtest_revamp_live_stack as B
m1,m5=B.load_bars('/workspace/clara/MNQ_1m.csv');ens=B.ensemble_signals(m5);b=B.featured(m5);adds=B.addon_rows(b)
sig=pd.concat([ens]+list(adds.values()))
out={}
for nm,R in [('live_all_max3',B.Rules(overnight_sleeves='all')),('live_all_max1',B.Rules(overnight_sleeves='all',max_open=1)),('trend_ovn_max1',B.Rules(overnight_sleeves='trend',max_open=1))]:
    tr=B.in_window(B.simulate(m1,m5.index,sig,R))
    out[nm]=pd.DataFrame([t.__dict__ for t in tr]);print(nm,len(tr),round(sum(t.pnl for t in tr)),flush=True)
pickle.dump(out,open('revamp_trades.pkl','wb'))
