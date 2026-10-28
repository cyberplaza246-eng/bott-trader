import pandas as pd,numpy as np,json,sys
from sim import run,days
P=dict(wmax=250,k=0.8,lockR=0.5,lockf=0.4,trail=0.6,cut=11,buf=0,smin=20)  # chosen by fidelity to Clara's log on train, not profit
b=pd.read_pickle('b2.pkl')
def mae(t):
    out=[]
    for _,r in t.iterrows():
        g=b[(b.index>r.et)&(b.index<=r.xt)] if 'xt' in r else None
    return out
t=run(P);t['d']=t.et.dt.normalize()
# intraday worst open excursion per trade (approx from 2m bars between entry and exit)
def worst(r):
    g=b[(b.index>r.et)&(b.index<=r.et+pd.Timedelta(hours=6))]
    g=g[g.index.normalize()==r.d]
    lo=(g.low.min()-r.e) if r.s==1 else (r.e-g.high.max())
    return max(min(lo*2, r.usd),-1e9)
c=pd.read_pickle('t.pkl');c['d']=c.et.dt.normalize();c['s']=(c['Market pos.']=='Long')*2-1
c['usd']=c.pts*2-4-1  # Clara's own trades at 1 MNQ, our cost model
c['wo']=-c.MAE/c.Qty  # MAE in $ total; per contract approx
def metrics(t,wo=None,name=''):
    t=t.sort_values('et').reset_index(drop=True)
    eq=t.usd.cumsum().values;prev=np.r_[0,eq[:-1]]
    pk=np.maximum.accumulate(np.r_[0,eq])[1:]
    closed_dd=(pk-eq).max()
    intr=prev+np.minimum(t[wo].values if wo else 0,t.usd.values) if wo else eq
    intr_dd=(np.maximum.accumulate(np.r_[0,eq])[:-1]-intr).max()
    dd=t.groupby('d').usd.sum();deq=dd.cumsum();eod=(deq.cummax().clip(lower=0)-deq)
    # trailing $2k breach (intraday basis): peak equity minus intraday trough
    tr=np.maximum.accumulate(np.r_[0,eq])[:-1]-intr
    br=sorted(set(str(x.date()) for x in t.d[tr>=2000]))
    l=(t.usd<=0).astype(int);mcl=max((len(s) for s in ''.join(map(str,l)).split('0')),default=0)
    w=t.usd[t.usd>0];L=t.usd[t.usd<=0]
    return dict(set=name,n=len(t),days=t.d.nunique(),trades_per_day=round(len(t)/t.d.nunique(),2),wr=round(len(w)/len(t),3),net=round(t.usd.sum()),per_trade=round(t.usd.mean(),1),
      pf=round(w.sum()/-L.sum(),2),avg_win=round(w.mean()),avg_loss=round(L.mean()),max_consec_losses=mcl,
      closed_dd=round(closed_dd),intraday_dd=round(intr_dd),eod_dd=round(eod.max()),
      daily_pnl_pcts={k:round(v) for k,v in dd.quantile([.05,.25,.5,.75,.95]).items()},worst_day=round(dd.min()),
      breach_2k_dates=br)
t['wo']=t.apply(worst,axis=1)
lo,hi=pd.Timestamp('2024-09-25'),pd.Timestamp('2026-09-20')
cc=c[(c.d>=lo)&(c.d<=hi)]
R=[]
for nm,a,bb in [('train',lo,pd.Timestamp('2025-12-31')),('test2026',pd.Timestamp('2026-01-01'),hi),('full',lo,hi)]:
    R.append(metrics(t[(t.d>=a)&(t.d<=bb)],'wo','replica_'+nm))
    R.append(metrics(cc[(cc.d>=a)&(cc.d<=bb)],'wo','clara1ct_'+nm))
R.append(metrics(c,'wo','clara1ct_all_2019_2026'))
# walk-forward: half-year segments
t['h']=t.d.dt.year.astype(str)+'H'+((t.d.dt.month>6)+1).astype(str)
cc['h']=cc.d.dt.year.astype(str)+'H'+((cc.d.dt.month>6)+1).astype(str)
wf=pd.DataFrame({'rep_net':t.groupby('h').usd.sum().round(),'rep_wr':t.groupby('h').usd.apply(lambda x:(x>0).mean()).round(2),'clara_net':cc.groupby('h').usd.sum().round(),'clara_wr':cc.groupby('h').usd.apply(lambda x:(x>0).mean()).round(2)})
# session & regime
t['hr']=t.et.dt.hour+np.where(t.et.dt.minute>=30,.5,0)
ses=t.groupby('hr').usd.agg(['size','sum','mean']).round(1)
orw={}
for d,g in days.items():
    o=g[g.index<=d+pd.Timedelta(minutes=590)];orw[d]=o.high.max()-o.low.min()
    orw[d]=(orw[d],abs(g.close.iloc[-1]-g.open.iloc[0]))
t['orw']=t.d.map(lambda d:orw[d][0]);t['trend']=t.d.map(lambda d:orw[d][1]>orw[d][0]*1.5)
t['vol']=pd.qcut(t.orw,3,labels=['lowvol','midvol','highvol'])
reg=t.groupby(['trend','vol'],observed=True).usd.agg(['size','sum','mean',lambda x:(x>0).mean()]).round(2)
# match
rf=t.groupby('d').head(1).set_index('d');cf=cc.groupby('d').head(1).set_index('d');both=rf.index.intersection(cf.index)
match=dict(clara_days=len(cf),replica_days=len(rf),overlap=len(both),side_agree=round((rf.loc[both,'s']==cf.loc[both,'s']).mean(),3),
  entry_within_6min=round(((rf.loc[both,'et']-cf.loc[both,'et']).abs()<=pd.Timedelta('6min')).mean(),3),
  same_outcome_sign=round(((rf.loc[both,'usd']>0)==(cf.loc[both,'usd']>0)).mean(),3),
  daily_pnl_corr=round(t.groupby('d').usd.sum().reindex(both).corr(cc.groupby('d').usd.sum().reindex(both)),3))
for x in R: print(json.dumps(x))
print(wf.to_string());print(ses.to_string());print(reg.to_string());print(json.dumps(match))
t.drop(columns=['h']).to_csv('replica_trades.csv',index=False)
