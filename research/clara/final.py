import pandas as pd,numpy as np,pickle,json
from report import metrics,worst
A=pickle.load(open('wf_all.pkl','rb'));A.update(pickle.load(open('wf_all2.pkl','rb')))
A={k:v for k,v in A.items() if len(v)}
print('variants tried',len(A))
TR=(pd.Timestamp('2024-09-25'),pd.Timestamp('2025-12-31'));TE=(pd.Timestamp('2026-01-01'),pd.Timestamp('2026-09-20'))
def sl(t,a,b): return t[(t.d>=a)&(t.d<=b)]
def score(t):
    if len(t)<30: return None
    eq=t.usd.cumsum();dd=(eq.cummax().clip(lower=0)-eq).max()
    return None if dd>=2000 else t.usd.sum()
# 1) train selection (no 2026)
tr={k:score(sl(v,*TR)) for k,v in A.items()};tr={k:s for k,s in tr.items() if s is not None}
best=sorted(tr,key=tr.get,reverse=True)[:5];print('train survivors',len(tr),'top',[(k,round(tr[k])) for k in best])
# 2) rolling walk-forward: each quarter from 2025Q2, pick best on prior 2 quarters (DD<2k), trade next quarter
qs=pd.period_range('2025Q2','2026Q3',freq='Q');wf=[];picks=[]
for q in qs:
    a=(q-2).start_time;b=(q-1).end_time
    sc={k:score(sl(v,a,b)) for k,v in A.items()};sc={k:s for k,s in sc.items() if s is not None}
    if not sc: picks.append((str(q),None));continue
    k=max(sc,key=sc.get);picks.append((str(q),k));wf.append(sl(A[k],q.start_time,q.end_time))
wf=pd.concat(wf);print('WF picks',picks)
def full(t,name):
    t=t.copy();t['wo']=t.apply(worst,axis=1) if 'wo' not in t or t.wo.isna().any() else t.wo
    return metrics(t,'wo',name)
R=[]
for k in best[:3]:
    R.append(full(sl(A[k],*TR),f'{k}_train'));R.append(full(sl(A[k],*TE),f'{k}_TEST2026'))
R.append(full(wf,'walkforward_2025Q2-2026Q3'))
s=pd.read_pickle('six.pkl');s['s']=1
R.append(metrics(sl(s,*TR),'wo','6pm_train'));R.append(metrics(sl(s,*TE),'wo','6pm_TEST2026'))
# fusion: best train variant + 6pm, one position at a time (6pm runs 18:00-09:26, morning 09:50-15:30: no overlap)
k=best[0];mo=A[k].copy();mo['wo']=mo.apply(worst,axis=1)
fu=pd.concat([mo[['et','d','usd','wo']],s[['et','d','usd','wo']]]).sort_values('et')
for nm,(a,b) in [('train',TR),('TEST2026',TE)]: R.append(metrics(sl(fu,a,b),'wo',f'fusion_{k[0]}_{nm}'))
dm=mo.groupby('d').usd.sum();ds=s.groupby('d').usd.sum();j=pd.concat([dm,ds],axis=1).fillna(0)
print('daily corr morning vs 6pm',round(j.corr().iloc[0,1],3))
for r in R: r['breach_2k_dates']=r['breach_2k_dates'][:1]+(['...',r['breach_2k_dates'][-1],f"{len(r['breach_2k_dates'])} days"] if len(r['breach_2k_dates'])>1 else []);print(json.dumps(r,default=str))
# ---- combos with the 6 PM "revamp" long ----
m1=pd.read_csv('MNQ_1m.csv');m1['datetime']=pd.to_datetime(m1.datetime,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None);m1=m1.set_index('datetime')
def on_trend(d):
    a=m1.close.asof(d-pd.Timedelta(hours=6));b=m1.close.asof(d+pd.Timedelta(hours=9,minutes=29));return np.sign(b-a)
s6=s.set_index('d').usd
C={}
for k in best[:3]:
    mo=A[k].copy();mo['wo']=mo.apply(worst,axis=1);mo['ont']=mo.d.map(on_trend);mo['six']=mo.d.map(s6)
    C[(k,'base+6pm')]=mo
    C[(k,'only_with_overnight_trend')]=mo[mo.s==mo.ont]
    C[(k,'skip_if_6pm_lost')]=mo[~(mo.six<0)]
    h=mo.copy();f=np.where(h.six<0,.5,1.);h['usd']=h.usd*f;h['wo']=h.wo*f;C[(k,'half_if_6pm_lost(needs 2ct base)')]=h
    C[(k,'longs_only')]=mo[mo.s==1]
print('combo variants',len(C),' total variants incl morning',len(A)+len(C))
rows=[]
for (k,nm),mo in C.items():
    fu=pd.concat([mo[['et','d','usd','wo']],s[['et','d','usd','wo']]]).sort_values('et')
    for per,(a,b) in [('train',TR),('TEST2026',TE)]:
        r=metrics(sl(fu,a,b),'wo',f'{k[0]}|{k[1]}|{k[2]}|{nm}|{per}');r['breaches']=len(r.pop('breach_2k_dates'));rows.append(r)
X=pd.DataFrame(rows);X.to_csv('combos.csv',index=False)
print(X[['set','n','wr','net','per_trade','pf','avg_win','avg_loss','max_consec_losses','intraday_dd','eod_dd','worst_day','breaches']].to_string())
