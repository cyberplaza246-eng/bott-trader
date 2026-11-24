"""Published NQ/index strategies, implemented as published (no tuning). History only, no broker. New file."""
import numpy as np, pandas as pd, json
S=0.25
df=pd.read_csv("data/MNQ_1m.csv",parse_dates=["datetime"]); df["dt"]=df["datetime"].dt.tz_convert("America/New_York")
df=df.set_index("dt").sort_index(); df=df[~df.index.duplicated()]
T=df.index; O,H,L,C,V=[df[k].values for k in ("open","high","low","close","volume")]; n=len(C)
tm=np.array(T.hour*60+T.minute); day=np.array(T.date)
rth=(tm>=570)&(tm<960)
g=pd.Series(np.arange(n))[rth].groupby(day[rth]).agg(["min","max"])
g=g[(tm[g["min"].values]==570)&(tm[g["max"].values]==959)]
D=pd.DataFrame({"o":O[g["min"]],"c":C[g["max"]],"h":[H[a:b+1].max() for a,b in g.values],"l":[L[a:b+1].min() for a,b in g.values]},index=pd.to_datetime(g.index))
A=g["min"].values; B=g["max"].values; days=list(g.index)
def at(a,b,m):
    k=a+(m-570); return k if k<=b and tm[k]==m else None
def P(pts,trades=1): return pts  # points per 1 contract; costs added in st
out={}
# 1 Gao et al 2018: r1 = 10:00 price / prev close -1 ; trade sign(r1) 15:30->16:00
def gao(use12):
    tr=[]
    for k in range(1,len(days)):
        a,b=A[k],B[k]; i10=at(a,b,599); i30=at(a,b,929); i00=at(a,b,899)
        if None in (i10,i30,i00): continue
        r1=C[i10]/D.c.iloc[k-1]-1; r12=C[i30]/C[i00]-1; s=np.sign(r1)
        if use12 and np.sign(r12)!=s: continue
        if s==0: continue
        tr.append((days[k],s*(C[b]-C[i30]),2))
    return tr
out["GAO_first30_last30"]=gao(False); out["GAO_first30+12th_agree"]=gao(True)
# 2 Zarattini-Aziz 2023 5-min ORB, 10R target, stop at first 5m candle extreme, EOD exit
tr=[]
for k,(a,b) in enumerate(zip(A,B)):
    o,c5,h5,l5=O[a],C[a+4],H[a:a+5].max(),L[a:a+5].min()
    if c5==o: continue
    d=1 if c5>o else -1; e=O[a+5]; st=l5 if d==1 else h5; R=abs(e-st)
    if R<=0: continue
    tg=e+d*10*R; x=None
    for j in range(a+5,b+1):
        if (d==1 and L[j]<=st) or (d==-1 and H[j]>=st): x=st-e if d==1 else e-st; break
        if (d==1 and H[j]>=tg) or (d==-1 and L[j]<=tg): x=10*R; break
    if x is None: x=d*(C[b]-e)
    tr.append((days[k],x,2))
out["ZA_5minORB_10R"]=tr
# 3 Zarattini-Aziz-Barbon 2024 noise area: 14d sigma, gap-adjusted bounds, check every :00/:30 from 10:00, stop=max(UB,VWAP)/min(LB,VWAP) at checks, flip allowed, EOD exit
mv=np.full((len(days),390),np.nan)
for k,(a,b) in enumerate(zip(A,B)): mv[k,:b-a+1]=np.abs(C[a:b+1]/O[a]-1)
tr=[]
for k in range(15,len(days)):
    a,b=A[k],B[k]; sig=np.nanmean(mv[k-14:k],axis=0); pc=D.c.iloc[k-1]
    ub=max(O[a],pc)*(1+sig); lb=min(O[a],pc)*(1-sig)
    tp_=(H[a:b+1]+L[a:b+1]+C[a:b+1])/3; vv=np.maximum(V[a:b+1],1); vw=np.cumsum(tp_*vv)/np.cumsum(vv)
    pos=0; ent=0; pnl=0; nt=0
    for m in range(30,390,30):
        j=a+m-1
        if j>b: break
        px=C[j]; idx=m-1
        if pos==1 and px<max(ub[idx],vw[idx]): pnl+=px-ent; pos=0; nt+=1
        elif pos==-1 and px>min(lb[idx],vw[idx]): pnl+=ent-px; pos=0; nt+=1
        if pos==0 and m<390-1:
            if px>ub[idx]: pos,ent=1,px; nt+=1
            elif px<lb[idx]: pos,ent=-1,px; nt+=1
    if pos: pnl+=pos*(C[b]-ent); nt+=1
    if nt: tr.append((days[k],pnl,nt))
out["ZAB_noise_area_VWAP"]=tr
# 4 Overnight drift (Cliff, Cooper & Gulen 2008; Kelly & Clark 2011): long prev close -> open
out["OVERNIGHT_close_to_open"]=[(days[k],D.o.iloc[k]-D.c.iloc[k-1],2) for k in range(1,len(days)) if (D.index[k]-D.index[k-1]).days<=4]
# 5 Turn of month (Lakonishok & Smidt 1988): long close of day -2 to close of day +3
mo=D.index.to_period("M"); tr=[]
for p in sorted(set(mo))[:-1]:
    idx=np.where(mo==p)[0]; nx=np.where(mo==p+1)[0]
    if len(nx)<3: continue
    i0,i1=idx[-2],nx[2]; tr.append((D.index[i1].date(),D.c.iloc[i1]-D.c.iloc[i0],2))
out["TURN_OF_MONTH_-1_+3"]=tr
# 6 FOMC pre-announcement drift (Lucca & Moench 2015): long 14:00 day before -> 14:00 FOMC day
F=["2024-09-18","2024-11-07","2024-12-18","2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30","2025-09-17","2025-10-29","2025-12-10","2026-01-28","2026-03-18","2026-04-29","2026-06-17","2026-07-29","2026-09-16"]
tr=[]
for f in F:
    f=pd.Timestamp(f).date()
    if f not in days: continue
    k=days.index(f); i2=at(A[k],B[k],839); i1=at(A[k-1],B[k-1],840)
    if i1 is None or i2 is None: continue
    tr.append((f,C[i2]-O[i1],2))
out["FOMC_preannounce_drift"]=tr
# 7 Connors RSI(2): close>SMA200 & RSI2<5 buy close, exit close>SMA5 (Connors & Alvarez 2009)
c=D.c; dlt=c.diff(); up=dlt.clip(lower=0).ewm(alpha=1/2,adjust=False).mean(); dn=(-dlt.clip(upper=0)).ewm(alpha=1/2,adjust=False).mean()
rsi=100-100/(1+up/dn); s200=c.rolling(200).mean(); s5=c.rolling(5).mean(); tr=[]; pos=None
for k in range(200,len(c)):
    if pos is None and c.iloc[k]>s200.iloc[k] and rsi.iloc[k]<5: pos=c.iloc[k]
    elif pos is not None and c.iloc[k]>s5.iloc[k]: tr.append((D.index[k].date(),c.iloc[k]-pos,2)); pos=None
out["CONNORS_RSI2"]=tr
# 8 IBS (Pagonidis 2014 / QuantifiedStrategies): buy close if IBS<0.2, sell close when close>prev high
ibs=(D.c-D.l)/(D.h-D.l); tr=[]; pos=None
for k in range(1,len(D)):
    if pos is not None and D.c.iloc[k]>D.h.iloc[k-1]: tr.append((D.index[k].date(),D.c.iloc[k]-pos,2)); pos=None
    elif pos is None and ibs.iloc[k]<0.2: pos=D.c.iloc[k]
out["IBS_lt_0.2"]=tr
# 9 Buy & hold RTH benchmark (open->close daily) for reference
out["BENCH_intraday_long"]=[(days[k],D.c.iloc[k]-D.o.iloc[k],2) for k in range(len(days))]
def st(tr,pt,comm):
    if not tr: return {"trades":0}
    p=np.array([x[1]*pt - (x[2]/2)*comm - x[2]*S*pt for x in tr]); t=pd.to_datetime([x[0] for x in tr])
    ntr=int(sum(x[2] for x in tr)/2)
    def pf(q): return round(q[q>0].sum()/-q[q<=0].sum(),3) if (q<=0).any() and q[q<=0].sum()<0 else None
    cut=int(len(p)*0.6); eq=np.cumsum(p); m=pd.Series(p,index=t).groupby(t.strftime("%Y-%m")).sum(); y=pd.Series(p,index=t).groupby(t.year).sum()
    return {"round_trips":ntr,"days":len(p),"pf":pf(p),"net":round(p.sum()),"max_dd":round((np.maximum.accumulate(eq)-eq).max()),
            "win_pct":round(100*(p>0).mean(),1),"pf_first60":pf(p[:cut]),"net_first60":round(p[:cut].sum()),"pf_last40":pf(p[cut:]),"net_last40":round(p[cut:].sum()),
            "green_months_pct":round(100*(m>0).mean(),1),"yearly":y.round().to_dict(),"monthly":m.round().to_dict()}
res={k:{"NQ":st(v,20,4.5),"MNQ":st(v,2,1.24)} for k,v in out.items()}
for k,v in res.items():
    q=v["NQ"]; v["robust"]=bool((q.get("pf_first60") or 0)>=1.2 and (q.get("pf_last40") or 0)>=1.2 and q["round_trips"]>=100 and q.get("green_months_pct",0)>50)
    print(k,{x:q.get(x) for x in ("round_trips","pf","net","max_dd","win_pct","pf_first60","pf_last40","green_months_pct")},"MNQ pf",v["MNQ"].get("pf"),v["MNQ"].get("net"),"robust",v["robust"],flush=True)
print("range",days[0],days[-1])
json.dump(res,open("data/nq_published_strategies_results.json","w"),indent=1,default=str)
