import sys,os,pandas as pd,pickle
os.chdir('/workspace/clara/revamp');sys.path.insert(0,'.');sys.path.insert(0,'scripts')
import backtest_revamp_live_stack as B
B.COMM_RT=4.0
m1,m5=B.load_bars('/workspace/clara/MNQ_1m.csv');b=B.featured(m5);adds=B.addon_rows(b)
sig=pd.concat([B.ensemble_signals(m5)]+list(adds.values()))
V={'BASE':({},3),'maxopen4':({},4),'maxopen5':({},5),'maxopen6':({},6),
 'noCooldown':({'CD_WIN_BARS':0,'CD_LOSS_BARS':0,'LOSS_CD_MIN':0},3),
 'looseLimits':({'DAY_LOSS':1000.,'OVN_LOSS':600.,'CONSEC_MAX':5},3),
 'maxopen5+noCD+loose':({'CD_WIN_BARS':0,'CD_LOSS_BARS':0,'LOSS_CD_MIN':0,'DAY_LOSS':1000.,'OVN_LOSS':600.,'CONSEC_MAX':5},5)}
d0={k:getattr(B,k) for k in ['CD_WIN_BARS','CD_LOSS_BARS','LOSS_CD_MIN','DAY_LOSS','OVN_LOSS','CONSEC_MAX']}
out={}
for k,(p,mo) in V.items():
    for a,x in {**d0,**p}.items():setattr(B,a,x)
    tr=B.in_window(B.simulate(m1,m5.index,sig,B.Rules(overnight_sleeves='all',max_open=mo)))
    out[k]=pd.DataFrame([t.__dict__ for t in tr]);print(k,len(tr),round(sum(t.pnl for t in tr)),flush=True)
pickle.dump(out,open('next/more.pkl','wb'))
