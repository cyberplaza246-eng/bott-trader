"""NQ fixed-target strategy search, 3-way split (dev/val/final). History only, no broker."""
import json, itertools, numpy as np, pandas as pd
PT=20.0; COMM=4.50; S=0.25
DEV_END=pd.Timestamp("2025-12-01",tz="America/New_York"); VAL_END=pd.Timestamp("2026-04-01",tz="America/New_York")
df=pd.read_csv("data/MNQ_1m.csv",parse_dates=["datetime"])
df["dt"]=df["datetime"].dt.tz_convert("America/New_York"); df=df.set_index("dt").sort_index(); df=df[~df.index.duplicated()]
T=df.index; O,H,L,C,V=[df[k].values for k in("open","high","low","close","volume")]
tmin=np.array(T.hour*60+T.minute); day=np.array(T.date); n=len(C)
rth=(tmin>=570)&(tmin<960)
# per-RTH-day index ranges
rd=pd.Series(np.arange(n))[rth]; groups=rd.groupby(day[rth]).agg(["min","max"])
# daily ATR(14) percentile (RTH range), lagged
dr=pd.Series({d:H[a:b+1].max()-L[a:b+1].min() for d,(a,b) in groups.iterrows()})
atr=dr.rolling(14).mean().shift(1); atrp=atr.rolling(120,min_periods=40).rank(pct=True)
pdh=pd.Series({d:H[a:b+1].max() for d,(a,b) in groups.iterrows()}).shift(1)
pdl=pd.Series({d:L[a:b+1].min() for d,(a,b) in groups.iterrows()}).shift(1)
def sim(i,d,e,sl,tp,iend,limit):
    """i=entry bar (fill known at/after this bar). market: starts checking next bar. limit: entry bar checks stop only."""
    j=i
    if limit:
        if (d==1 and L[i]<=e-sl) or (d==-1 and H[i]>=e+sl): return -sl*PT-S*PT-COMM,i
    for j in range(i+1,iend+1):
        if d==1:
            if L[j]<=e-sl: return (-sl-S)*PT-COMM,j
            if H[j]>=e+tp+S: return tp*PT-COMM,j
        else:
            if H[j]>=e+sl: return (-sl-S)*PT-COMM,j
            if L[j]<=e-tp-S: return tp*PT-COMM,j
    return (d*(C[iend]-e)-S)*PT-COMM,iend
def mk(e,mkt): return e+ (S if mkt else 0)
# ---- families: each returns list of (time, pnl_pts_gross? -> pnl$)
def orb(rmin,tp,slmode,atrf):
    out=[]
    for d,(a,b) in groups.iterrows():
        if atrf and not (atrp.get(d,np.nan)>=0.5): continue
        k=a+rmin; 
        if k>=b: continue
        hi,lo=H[a:k].max(),L[a:k].min(); rng=hi-lo
        if rng<=0: continue
        for i in range(k,b):
            dd=1 if H[i]>=hi+S else (-1 if L[i]<=lo-S else 0)
            if dd==0: continue
            e=(hi+S if dd==1 else lo-S)+dd*S   # stop-entry slippage
            sl=min(rng+S, 30) if slmode=="range" else 12
            p,_=sim(i,dd,e,sl,tp,b,True); out.append((T[i],p)); break
    return out
def sweep(tp,sl_cap,atrf):
    out=[]
    for d,(a,b) in groups.iterrows():
        if atrf and not (atrp.get(d,np.nan)>=0.5): continue
        H_,L_=pdh.get(d,np.nan),pdl.get(d,np.nan)
        if np.isnan(H_): continue
        lowx=None; highx=None
        for i in range(a,min(b,a+240)):
            if L[i]<L_: lowx=min(L[i],lowx if lowx else 1e9)
            if H[i]>H_: highx=max(H[i],highx if highx else 0)
            if lowx and C[i]>L_:
                e=C[i]+S; sl=min(e-lowx+S,sl_cap); p,_=sim(i,1,e,sl,tp,b,False); out.append((T[i],p)); break
            if highx and C[i]<H_:
                e=C[i]-S; sl=min(highx-e+S,sl_cap); p,_=sim(i,-1,e,sl,tp,b,False); out.append((T[i],p)); break
    return out
def vwap2(tp,sl,atrf):
    out=[]
    for d,(a,b) in groups.iterrows():
        if atrf and not (atrp.get(d,np.nan)>=0.5): continue
        tp_=(H[a:b+1]+L[a:b+1]+C[a:b+1])/3; v=np.maximum(V[a:b+1],1)
        cv=np.cumsum(v); vw=np.cumsum(tp_*v)/cv; var=np.cumsum(v*tp_**2)/cv-vw**2; sd=np.sqrt(np.maximum(var,0))
        trades=0; i=a+30
        while i<b-5 and trades<2:
            k=i-a; up=vw[k]+2*sd[k]; dn=vw[k]-2*sd[k]
            if C[i]>up: d_=-1
            elif C[i]<dn: d_=1
            else: i+=1; continue
            # limit at band next bar
            lvl=up if d_==-1 else dn; f=None
            for j in range(i+1,min(i+6,b)):
                if (d_==-1 and H[j]>=lvl+S) or (d_==1 and L[j]<=lvl-S): f=j; break
            if f is None: i+=1; continue
            p,x=sim(f,d_,lvl,sl,tp,b,True); out.append((T[f],p)); trades+=1; i=x+1
    return out
def firstpb(thr,tp,sl,atrf):
    out=[]
    for d,(a,b) in groups.iterrows():
        if atrf and not (atrp.get(d,np.nan)>=0.5): continue
        k=a+30
        if k>=b: continue
        mv=C[k-1]-O[a]
        if abs(mv)<thr: continue
        dd=1 if mv>0 else -1
        ema=pd.Series(C[a:b+1]).ewm(span=20,adjust=False).mean().values
        for i in range(k,min(b,a+180)):
            lvl=ema[i-a-1]
            if (dd==1 and L[i]<=lvl-S) or (dd==-1 and H[i]>=lvl+S):
                p,_=sim(i,dd,lvl,sl,tp,b,True); out.append((T[i],p)); break
    return out
# 6PM long proxy: buy 18:00 bar open on Sun-Thu, time exit 21:00 or TP/SL
idx18=np.where((tmin==1080))[0]
def sixpm(tp,sl,hold):
    out=[]
    for i in idx18:
        if T[i].weekday()==4: continue
        iend=i
        while iend+1<n and (T[iend+1]-T[i]).total_seconds()<=hold*3600: iend+=1
        e=O[i]+S
        # entry at open: check bar i itself conservatively (stop first)
        if L[i]<=e-sl: out.append((T[i],(-sl-S)*PT-COMM)); continue
        p,_=sim(i,1,e,sl,tp,iend,False); out.append((T[i],p))
    return out
grid=[]
for r,tp,sm,a in itertools.product([15,30],[10,20,40],["range","12"],[0,1]): grid.append(("ORB",dict(rmin=r,tp=tp,slmode=sm,atrf=a),orb))
for tp,sc,a in itertools.product([10,20,40],[15,25],[0,1]): grid.append(("SWEEP",dict(tp=tp,sl_cap=sc,atrf=a),sweep))
for tp,sl,a in itertools.product([10,20],[10,20],[0,1]): grid.append(("VWAP2",dict(tp=tp,sl=sl,atrf=a),vwap2))
for th,tp,sl,a in itertools.product([20,40],[10,20],[8,15],[0,1]): grid.append(("FIRSTPB",dict(thr=th,tp=tp,sl=sl,atrf=a),firstpb))
for tp,sl,h in itertools.product([10,20,40],[20,40],[3,15]): grid.append(("SIXPM",dict(tp=tp,sl=sl,hold=h),sixpm))
def st(tr,mnq=False):
    if not tr: return {"trades":0}
    p=np.array([x[1] for x in tr]); t=pd.DatetimeIndex([x[0] for x in tr])
    if mnq: p=(p+COMM)/10-1.24
    w,l=p[p>0],p[p<=0]; eq=np.cumsum(p); m=pd.Series(p,index=t).groupby(t.strftime("%Y-%m")).sum()
    nd=len(set(t.date)); ndays=max(1,np.busday_count(t.min().date(),t.max().date()))
    return {"trades":len(p),"trades_per_day":round(len(p)/ndays,2),"win_pct":round(100*len(w)/len(p),1),"pf":round(w.sum()/-l.sum(),3) if l.sum()<0 else None,
            "net":round(p.sum()),"max_dd":round((np.maximum.accumulate(eq)-eq).max()),"green_months_pct":round(100*(m>0).mean(),1),
            "best_month_share_of_net":round(m.max()/p.sum(),2) if p.sum()>0 else None}
rows=[]
for fam,kw,fn in grid:
    tr=fn(**kw); dv=[x for x in tr if x[0]<DEV_END]; va=[x for x in tr if DEV_END<=x[0]<VAL_END]
    r={"family":fam,"params":kw,"dev":st(dv),"val":st(va),"_tr":tr}; rows.append(r)
    print(fam,kw,"DEV",r["dev"].get("pf"),r["dev"].get("net"),r["dev"]["trades"],"VAL",r["val"].get("pf"),r["val"].get("net"),flush=True)
fin=[r for r in rows if (r["dev"].get("pf") or 0)>=1.15 and r["dev"]["trades"]>=100 and (r["val"].get("pf") or 0)>=1.1 and r["val"]["trades"]>=30]
fin=sorted(fin,key=lambda r:-(r["val"]["pf"]))[:3]
for r in fin:
    ft=[x for x in r["_tr"] if x[0]>=VAL_END]
    r["final"]=st(ft); r["final_mnq"]=st(ft,True); r["dev_mnq"]=st([x for x in r["_tr"] if x[0]<DEV_END],True); r["val_mnq"]=st([x for x in r["_tr"] if DEV_END<=x[0]<VAL_END],True)
    r["passes_phase2"]=bool((r["final"].get("pf") or 0)>=1.2 and r["final"]["trades"]>=100 and (r["final"].get("best_month_share_of_net") or 9)<0.5)
    print("FINAL",r["family"],r["params"],r["final"],r["final_mnq"],r["passes_phase2"],flush=True)
for r in rows: r.pop("_tr")
json.dump({"variants_tried":len(rows),"splits":{"dev_end":str(DEV_END.date()),"val_end":str(VAL_END.date())},
  "selection":"dev PF>=1.15 & >=100 trades, val PF>=1.1 & >=30 trades; top3 by val PF see final once",
  "finalists":[{k:v for k,v in r.items()} for r in fin],"all":rows},open("data/nq_fixed_target_search.json","w"),indent=1,default=str)
print("variants",len(rows),"finalists",len(fin))
