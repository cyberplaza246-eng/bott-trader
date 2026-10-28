import sys,pandas as pd,pickle
sys.path.insert(0,'.');sys.path.insert(0,'scripts')
import backtest_revamp_live_stack as B
m1,m5=B.load_bars('/workspace/clara/MNQ_1m.csv');ens=B.ensemble_signals(m5);b=B.featured(m5);adds=B.addon_rows(b)
sig=pd.concat([ens]+list(adds.values()))
R=B.Rules(overnight_sleeves='trend,breakout',disabled_tags=('addon_onl',))
tr=B.in_window(B.simulate(m1,m5.index,sig,R))
pickle.dump({'P1':pd.DataFrame([t.__dict__ for t in tr])},open('p1.pkl','wb'));print(len(tr),sum(t.pnl for t in tr))
