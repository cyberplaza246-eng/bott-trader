"""One-shot final test of locked open30 momentum rules (data/nq_open30_momentum_rules_locked.json). History only."""
import numpy as np, pandas as pd, json
df=pd.read_csv("data/MNQ_1m.csv",parse_dates=["datetime"]); df["dt"]=df["datetime"].dt.tz_convert("America/New_York")
df=df.set_index("dt").sort_index(); df=df[~df.index.duplicated()]
df=df[(df.index>=pd.Timestamp("2026-04-01",tz="America/New_York"))&(df.index<pd.Timestamp("2026-09-21",tz="America/New_York"))]
T=df.index; O,C=df.open.values,df.close.values; tm=np.array(T.hour*60+T.minute); day=np.array(T.date); S=0.25
ix={k:{day[i]:i for i in np.where(tm==k)[0]} for k in (570,600,900,959)}
res={}
for nm,pt,cm in (("NQ",20,4.5),("MNQ",2,1.24)):
    tr=[]
    for d in sorted(ix[570]):
        if not all(d in ix[k] for k in ix): continue
        i,j,c=ix[570][d],ix[600][d],ix[959][d]; mv=C[j]-O[i]
        if abs(mv)<20: continue
        dd=1 if mv>0 else -1; tr.append((d,(dd*(C[c]-O[j+1])-2*S)*pt-cm))
    p=np.array([x[1] for x in tr]); eq=np.cumsum(p); m=pd.Series(p,index=pd.to_datetime([x[0] for x in tr])).groupby(lambda t:t.strftime("%Y-%m")).sum()
    pf=p[p>0].sum()/-p[p<=0].sum()
    res[nm]={"trades":len(p),"pf":round(pf,3),"net":round(p.sum(),2),"max_dd":round((np.maximum.accumulate(eq)-eq).max(),2),"win_pct":round(100*(p>0).mean(),1),
             "monthly":m.round(2).to_dict(),"best_month_share":round(m.max()/p.sum(),2) if p.sum()>0 else None,
             "passes":bool(pf>=1.2 and p.sum()>0 and m.max()/p.sum()<0.5),"first":str(tr[0][0]),"last":str(tr[-1][0])}
json.dump(res,open("data/nq_open30_final_test.json","w"),indent=1); print(json.dumps(res,indent=1))
