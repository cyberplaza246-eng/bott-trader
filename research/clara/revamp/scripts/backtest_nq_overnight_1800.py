"""Backtest: long at 18:00 ET reopen (Sun-Thu) -> exit 09:30 (or 09:25) ET open next day. History only. New file."""
import numpy as np, pandas as pd, json
S=0.25
df=pd.read_csv("data/MNQ_1m.csv",parse_dates=["datetime"]); df["dt"]=df["datetime"].dt.tz_convert("America/New_York")
df=df.set_index("dt").sort_index(); df=df[~df.index.duplicated()]
T=df.index; O=df.open.values; tm=np.array(T.hour*60+T.minute)
pos={t:i for i,t in enumerate(T)}
def run(exit_hm):
    tr=[]
    for i in np.where(tm==1080)[0]:
        if T[i].weekday() not in (6,0,1,2,3): continue
        tgt=(T[i]+pd.Timedelta(days=1)).normalize()+pd.Timedelta(minutes=exit_hm)
        j=pos.get(tgt)
        if j is None: continue
        tr.append((T[i],O[j]-O[i]))
    return tr
def st(tr,pt,comm):
    p=np.array([(x[1]-2*S)*pt-comm for x in tr]); t=pd.DatetimeIndex([x[0] for x in tr]); cut=int(len(p)*.6)
    pf=lambda q: round(q[q>0].sum()/-q[q<=0].sum(),3); eq=np.cumsum(p); m=pd.Series(p,index=t).groupby(t.strftime("%Y-%m")).sum()
    return {"trades":len(p),"pf":pf(p),"net":round(p.sum()),"max_dd":round((np.maximum.accumulate(eq)-eq).max()),"win_pct":round(100*(p>0).mean(),1),
            "pf_first60":pf(p[:cut]),"pf_last40":pf(p[cut:]),"green_months_pct":round(100*(m>0).mean(),1),"worst_month":[m.idxmin(),round(m.min())],
            "yearly":pd.Series(p,index=t).groupby(t.year).sum().round().to_dict(),"monthly":m.round().to_dict(),
            "robust":bool(pf(p[:cut])>=1.2 and pf(p[cut:])>=1.2 and len(p)>=100 and (m>0).mean()>0.5)}
res={f"1800_to_{h//60:02d}{h%60:02d}":{"NQ":st(run(h),20,4.5),"MNQ":st(run(h),2,1.24)} for h in (570,565)}
for k,v in res.items():
    for s_ in v: print(k,s_,{x:v[s_][x] for x in ("trades","pf","net","max_dd","win_pct","pf_first60","pf_last40","green_months_pct","worst_month","yearly","robust")})
json.dump(res,open("data/nq_overnight_1800_results.json","w"),indent=1,default=str)
