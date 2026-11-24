"""NQ small-target + broad strategy search (new file; copied core from search_nq_fixed_target.py), 3-way split (dev/val/final). History only, no broker."""
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
def mom(look,thr,tp,sl,atrf):
    """momentum burst: |C[i]-C[i-look]|>=thr pts within RTH 9:35-15:30, enter market with move, max 3/day"""
    out=[]
    for d,(a,b) in groups.iterrows():
        if atrf and not (atrp.get(d,np.nan)>=0.5): continue
        i=a+max(look,5); k=0
        while i<b-30 and k<3:
            mv=C[i]-C[i-look]
            if abs(mv)>=thr:
                dd=1 if mv>0 else -1; e=C[i]+dd*S; p,x=sim(i,dd,e,sl,tp,min(b,i+60),False); out.append((T[i],p)); k+=1; i=x+1
            else: i+=1
    return out
grid=[]
TPS=[2,3,4,5,6,8]; RAT=[1.0,1.5,2.0]
for tp,r,rm,a in itertools.product(TPS,RAT,[5,15],[0,1]):
    sl=tp*r
    grid.append(("ORB",dict(rmin=rm,tp=tp,sl=sl,atrf=a),lambda rmin,tp,sl,atrf: orb2(rmin,tp,sl,atrf)))
for tp,r,a in itertools.product(TPS,RAT,[0,1]): grid.append(("SWEEP",dict(tp=tp,sl_cap=tp*r,atrf=a),sweep))
for tp,r,a in itertools.product(TPS,RAT,[0,1]): grid.append(("VWAP2",dict(tp=tp,sl=tp*r,atrf=a),vwap2))
for th,tp,r in itertools.product([20,40],TPS,RAT): grid.append(("FIRSTPB",dict(thr=th,tp=tp,sl=tp*r,atrf=0),firstpb))
for (lk,th),tp,r in itertools.product([(3,15),(5,25)],TPS,RAT): grid.append(("MOMBURST",dict(look=lk,thr=th,tp=tp,sl=tp*r,atrf=0),mom))
def orb2(rmin,tp,sl,atrf):
    out=[]
    for d,(a,b) in groups.iterrows():
        if atrf and not (atrp.get(d,np.nan)>=0.5): continue
        k=a+rmin
        if k>=b: continue
        hi,lo=H[a:k].max(),L[a:k].min()
        for i in range(k,b):
            dd=1 if H[i]>=hi+S else (-1 if L[i]<=lo-S else 0)
            if dd==0: continue
            e=(hi+S if dd==1 else lo-S)+dd*S; p,_=sim(i,dd,e,sl,tp,b,True); out.append((T[i],p)); break
    return out
# ---------- broad families (1 NQ, market in/out: 2 ticks slip + comm) ----------
def tr_ret(i,j,dd): return (dd*(C[j]-O[i])-2*S)*PT-COMM
ix={}
for t_,nm in [(570,"o"),(600,"o30"),(900,"c15"),(959,"c")]:
    m=pd.Series(np.where(tmin==t_)[0]); ix[nm]=dict(zip(day[m.values],m.values))
days=[d for d in groups.index if all(d in ix[k] for k in ix)]
dclose=pd.Series({d:C[ix["c"][d]] for d in days})
def b_rth(filt):
    out=[]
    for d in days:
        dd=1
        if filt:
            s=dclose.rolling(filt).mean().shift(1).get(d,np.nan)
            if np.isnan(s): continue
            dd=1 if dclose.shift(1)[d]>s else -1
        out.append((T[ix["o"][d]],tr_ret(ix["o"][d],ix["c"][d],dd)))
    return out
def b_open30(thr):
    out=[]
    for d in days:
        i,j=ix["o"][d],ix["o30"][d]; mv=C[j]-O[i]
        if abs(mv)<thr: continue
        out.append((T[j],tr_ret(j+1,ix["c"][d],1 if mv>0 else -1)))
    return out
def b_lasthr(thr):
    out=[]
    for d in days:
        i,j=ix["o"][d],ix["c15"][d]; mv=C[j]-O[i]
        if abs(mv)<thr: continue
        out.append((T[j],tr_ret(j+1,ix["c"][d],1 if mv>0 else -1)))
    return out
on18=np.where(tmin==1080)[0]
def b_overnight(exit_min,skipfri):
    out=[]
    for i in on18:
        if skipfri and T[i].weekday()==4: continue
        j=i
        while j+1<n and (T[j+1]-T[i]).total_seconds()<70*3600 and not (tmin[j+1]==exit_min): j+=1
        if j+1<n and tmin[j+1]==exit_min: out.append((T[i],tr_ret(i,j+1,1)))
    return out
def b_swing(ma,short):
    """daily close vs SMA(ma): hold long (or short) next day close-to-close; cost charged on position changes"""
    s=dclose.rolling(ma).mean(); out=[]; pos=0
    for k in range(ma,len(dclose)-1):
        newp=1 if dclose.iloc[k]>s.iloc[k] else (-1 if short else 0)
        cost=(abs(newp-pos))*(COMM/2+S*PT); pos=newp
        if pos==0 and cost==0: continue
        pnl=pos*(dclose.iloc[k+1]-dclose.iloc[k])*PT-cost
        out.append((T[ix["c"][dclose.index[k+1]]],pnl))
    return out
def b_donch(nb,trail):
    """daily Donchian breakout long/short, ATR-ish trailing stop of trail*atr(14) on daily closes"""
    hh=dclose.rolling(nb).max().shift(1); ll=dclose.rolling(nb).min().shift(1); out=[]; pos=0; ext=0
    a_=dclose.diff().abs().rolling(14).mean()
    for k in range(nb+14,len(dclose)):
        c=dclose.iloc[k]; d=dclose.index[k]
        if pos:
            ext=max(ext,c) if pos==1 else min(ext,c)
            if (pos==1 and c<ext-trail*a_.iloc[k]) or (pos==-1 and c>ext+trail*a_.iloc[k]):
                out.append((T[ix["c"][d]],pos*(c-ent)*PT-COMM-2*S*PT)); pos=0
        if not pos:
            if c>hh.iloc[k]: pos,ent,ext=1,c,c
            elif c<ll.iloc[k]: pos,ent,ext=-1,c,c
    return out
for f in [0,20,50]: grid.append(("B_RTH",dict(filt=f),b_rth))
for t in [10,20,40]: grid.append(("B_OPEN30",dict(thr=t),b_open30)); grid.append(("B_LASTHR",dict(thr=t),b_lasthr))
for e,s in itertools.product([570,180,0],[0,1]): grid.append(("B_OVERNIGHT",dict(exit_min=e,skipfri=s),b_overnight))
for m,s in itertools.product([10,20,50,100],[0,1]): grid.append(("B_SWING",dict(ma=m,short=s),b_swing))
for nb,tr in itertools.product([10,20,55],[2,3,5]): grid.append(("B_DONCH",dict(nb=nb,trail=tr),b_donch))
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
for r in rows: r["dev_mnq"]=st([x for x in r["_tr"] if x[0]<DEV_END],True); r["val_mnq"]=st([x for x in r["_tr"] if DEV_END<=x[0]<VAL_END],True)
ok=lambda dv,va:(dv.get("pf") or 0)>=1.15 and dv["trades"]>=100 and (va.get("pf") or 0)>=1.1 and va["trades"]>=30
fin=[r for r in rows if ok(r["dev"],r["val"]) or ok(r["dev_mnq"],r["val_mnq"])]
fin=sorted(fin,key=lambda r:-(r["val"]["pf"]))[:3]
for r in fin:
    ft=[x for x in r["_tr"] if x[0]>=VAL_END]
    r["final"]=st(ft); r["final_mnq"]=st(ft,True); r["dev_mnq"]=st([x for x in r["_tr"] if x[0]<DEV_END],True); r["val_mnq"]=st([x for x in r["_tr"] if DEV_END<=x[0]<VAL_END],True)
    r["passes_phase2"]=bool((r["final"].get("pf") or 0)>=1.2 and r["final"]["trades"]>=100 and (r["final"].get("best_month_share_of_net") or 9)<0.5)
    print("FINAL",r["family"],r["params"],r["final"],r["final_mnq"],r["passes_phase2"],flush=True)
for r in rows: r.pop("_tr")
json.dump({"variants_tried":len(rows),"splits":{"dev_end":str(DEV_END.date()),"val_end":str(VAL_END.date())},
  "selection":"dev PF>=1.15 & >=100 trades, val PF>=1.1 & >=30 trades; top3 by val PF see final once",
  "finalists":[{k:v for k,v in r.items()} for r in fin],"all":rows},open("data/nq_small_target_search.json","w"),indent=1,default=str)
print("variants",len(rows),"finalists",len(fin))


