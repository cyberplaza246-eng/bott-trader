import pickle,numpy as np,pandas as pd
d1=pd.read_parquet("/workspace/intrabar/1m_long.parquet"); TS=d1.index.as_unit("ns").asi8
O,H,Lo,C=[d1[c].to_numpy() for c in ("open","high","low","close")]
et=d1.index.tz_convert("America/New_York"); M=np.asarray(et.hour*60+et.minute); TD=np.array([str(x.date()) for x in (et+pd.Timedelta(hours=6))])
f=pickle.load(open("/workspace/intrabar/f_long.pkl","rb")); fi=f.index.as_unit("ns").asi8
def feat(k,col):  # last 5m feature closed at or before TS[k]
    i=np.searchsorted(fi,TS[k],side="right")-1; return f[col].iloc[i] if i>=0 else np.nan
m15=d1.resample("15min",label="right",closed="right").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
tr=pd.concat([m15.high-m15.low,(m15.high-m15.close.shift()).abs(),(m15.low-m15.close.shift()).abs()],axis=1).max(axis=1); a15=tr.ewm(alpha=1/14,adjust=False).mean()
ai=a15.index.as_unit("ns").asi8; av=a15.to_numpy()
m5=d1.close.resample("5min",label="right",closed="right").last().dropna(); dl=m5.diff()
rsi=100-100/(1+dl.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()/(-dl.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean())
ri=rsi.index.as_unit("ns").asi8; rv=rsi.to_numpy()
def rsi_at(k): i=np.searchsorted(ri,TS[k],side="right")-1; return rv[i]
def trade(k,d,src):
    i=np.searchsorted(ai,TS[k],side="right")-1; sld=float(np.clip(1.5*av[i],12,40)); fill=O[k]; sl=fill-d*sld; locked=False; b=k
    if sld*2+4>300: return None
    while b<len(TS):
        if b>k and TS[b]-TS[b-1]>3*3600*10**9: kx,xp=b-1,C[b-1]; break
        if (d>0 and Lo[b]<=sl) or (d<0 and H[b]>=sl): kx,xp=b,(min(sl,O[b]) if d>0 else max(sl,O[b])) if b>k else sl; break
        if not locked and d*((H[b] if d>0 else Lo[b])-fill)*2>=400: locked=True; sl=fill+d*25
        if M[b]==569: kx,xp=b,C[b]; break
        b+=1
    else: return None
    return dict(T=int(TS[k]),k=k,kx=kx,td=TD[k],src="6pm",dir=d,entry=fill,exit=xp,why=src,pnl=d*(xp-fill)*2-4,stream="6pm",g="6pm")
starts=np.where(M==1080)[0]; V={n:[] for n in ["BASE blind","a 1h uptrend","b >PDC&VWAP @18:15","c dip+reclaim","d RSI>55 @18:15","e trend+reclaim","f trend long/short"]}
for k in starts:
    if k+100>=len(TS) or TS[k+90]-TS[k]>2*3600*10**9: continue
    up=feat(k,"h1_up"); o=O[k]
    V["BASE blind"].append(trade(k,1,"base"))
    if up==1: V["a 1h uptrend"].append(trade(k,1,"a"))
    V["f trend long/short"].append(trade(k,1 if up==1 else -1,"f"))
    k15=k+15
    if M[k15]==1095:
        c=C[k15-1]
        if c>feat(k15,"pdc") and c>feat(k15,"vwap"): V["b >PDC&VWAP @18:15"].append(trade(k15,1,"b"))
        if rsi_at(k15)>55 and c>o: V["d RSI>55 @18:15"].append(trade(k15,1,"d"))
    # c: dip below 18:00 open by >=2pts before 19:30, then 5m close back above open
    dipped=False
    for b in range(k,k+90):
        if M[b]>=1170: break
        if Lo[b]<=o-2: dipped=True
        if dipped and (M[b]+1)%5==0 and C[b]>o and b>k:
            V["c dip+reclaim"].append(trade(b+1,1,"c"))
            if up==1: V["e trend+reclaim"].append(trade(b+1,1,"e"))
            break
V={n:pd.DataFrame([r for r in v if r]) for n,v in V.items()}
for n,v in V.items(): print(n,len(v),round(v.pnl.sum()),v.groupby(v.td.str[:4]).pnl.sum().round().to_dict())
pickle.dump(V,open("/workspace/sixpm/streams.pkl","wb"))
