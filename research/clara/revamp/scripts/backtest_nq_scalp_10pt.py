"""NQ 10-pt scalp backtest (historical data only, no broker). Uses data/MNQ_1m.csv; P&L scaled to NQ $20/pt."""
import json, numpy as np, pandas as pd
PT=20.0; COMM=4.50; SLIP=0.25; TICK=0.25; TP=10.0
OOS_START=pd.Timestamp("2026-03-21",tz="America/New_York")
df=pd.read_csv("data/MNQ_1m.csv",parse_dates=["datetime"])
df["dt"]=df["datetime"].dt.tz_convert("America/New_York"); df=df.set_index("dt").sort_index()
df=df[~df.index.duplicated()]
c=df.close
# 1m features
df["ema1"]=c.ewm(span=20,adjust=False).mean()
df["day"]=df.index.date
tp_=(df.high+df.low+df.close)/3
rth=(df.index.time>=pd.Timestamp("09:30").time())&(df.index.time<pd.Timestamp("16:00").time())
g=df[rth].copy(); g["pv"]=tp_[rth]*g.volume
g["vwap"]=g.groupby("day").pv.cumsum()/g.groupby("day").volume.cumsum().replace(0,np.nan)
df["vwap"]=g["vwap"]
# higher TF trend using only completed bars (shift 1 then ffill onto 1m)
def htf(rule):
    r=c.resample(rule,label="right",closed="right").last().dropna()
    e=r.ewm(span=20,adjust=False).mean()
    return r,e
c5,e5=htf("5min"); c15,e15=htf("15min")
up5=(c5>e5).astype(int)-(c5<e5).astype(int)
sl15=np.sign(e15.diff())
df["t5"]=up5.reindex(df.index,method="ffill").shift(1)
df["t15"]=sl15.reindex(df.index,method="ffill").shift(1)
trend=np.where((df.t5>0)&(df.t15>0),1,np.where((df.t5<0)&(df.t15<0),-1,0))
mom=c-c.shift(3)
sigA=np.where((mom>=4)&(c>c.shift(1)),1,np.where((mom<=-4)&(c<c.shift(1)),-1,0))
ph,pl=df.high.shift(1),df.low.shift(1)
touchL=(pl<=df.ema1.shift(1)+0.5)|(pl<=df.vwap.shift(1)+0.5)
touchS=(ph>=df.ema1.shift(1)-0.5)|(ph>=df.vwap.shift(1)-0.5)
sigB=np.where((trend==1)&touchL&(c>ph),1,np.where((trend==-1)&touchS&(c<pl),-1,0))
O,H,L,C=[df[k].values for k in("open","high","low","close")]
T=df.index; tmin=T.hour*60+T.minute; days=np.array(df.day)
def run(sig,sl,s_end,daily_stop):
    trades=[]; i=0; n=len(C)
    inwin=(tmin>=570)&(tmin<s_end)
    wins=loss=0; cur=None
    while i<n-1:
        if days[i]!=cur: cur=days[i]; wins=loss=0
        if not inwin[i] or sig[i]==0 or (daily_stop and (wins>=3 or loss>=2)): i+=1; continue
        d=sig[i]; e=C[i]+d*SLIP; j=i+1; px=None
        while j<n and days[j]==cur and tmin[j]<960:
            if d==1:
                if L[j]<=e-sl: px=e-sl-SLIP; break
                if H[j]>=e+TP+TICK: px=e+TP; break
            else:
                if H[j]>=e+sl: px=e+sl+SLIP; break
                if L[j]<=e-TP-TICK: px=e-TP; break
            j+=1
        if px is None: j=min(j-1,n-1); px=C[j]-d*SLIP
        pnl=d*(px-e)*PT-COMM
        trades.append((T[i],cur,pnl)); wins+=pnl>0; loss+=pnl<=0; i=j+1
    return trades
def stats(tr,mult=1.0):
    if not tr: return {"trades":0}
    p=np.array([t[2] for t in tr]);
    if mult!=1.0: p=(p+COMM)*mult-0.62*2  # MNQ approx comm $1.24 RT
    dd=pd.Series(p,index=[t[1] for t in tr]).groupby(level=0).sum()
    eq=np.cumsum(p); w=p[p>0]; l=p[p<=0]
    aw=w.mean() if len(w) else 0; al=l.mean() if len(l) else 0
    return {"trades":len(p),"trades_per_day":round(len(p)/len(dd),2),"win_pct":round(100*len(w)/len(p),1),
     "avg_win":round(aw,1),"avg_loss":round(al,1),"net":round(p.sum(),0),
     "profit_factor":round(w.sum()/-l.sum(),3) if len(l) and l.sum()<0 else None,
     "max_dd":round((np.maximum.accumulate(eq)-eq).max(),0),"worst_day":round(dd.min(),0),
     "green_days_pct":round(100*(dd>0).mean(),1),"breakeven_win_pct":round(100*-al/(aw-al),1) if aw>0 else None}
res={"costs":{"comm_rt":COMM,"slip_pts_side":SLIP,"tp_through_ticks":1},"data":[str(T[0]),str(T[-1])],"oos_start":str(OOS_START.date()),"results":[]}
grid=[("A",sigA,[8,10,12,15],False),("B",sigB,[8,10,12],False),("C",sigB,[8,10,12],True)]
for name,s,sls,ds in grid:
  for sess,end in(("RTH",960),("first_hour",630)):
    for sl in sls:
      tr=run(s,sl,end,ds); ins=[t for t in tr if t[0]<OOS_START]; oos=[t for t in tr if t[0]>=OOS_START]
      hk=lambda t: str(t[0].year)+"H"+str(1+(t[0].month>6)); byp={k:stats([t for t in tr if hk(t)==k]) for k in sorted({hk(t) for t in tr})}
      r={"variant":name,"session":sess,"sl":sl,"in_sample":stats(ins),"oos":stats(oos),"oos_mnq_1lot":stats(oos,0.1),"by_half_year":{k:{"net":v["net"],"win_pct":v["win_pct"],"trades":v["trades"]} for k,v in byp.items()}}
      res["results"].append(r); print(name,sess,sl,"IS",r["in_sample"].get("net"),r["in_sample"].get("profit_factor"),"OOS",r["oos"].get("net"),r["oos"].get("profit_factor"),r["oos"].get("win_pct"),r["oos"].get("breakeven_win_pct"),flush=True)
json.dump(res,open("data/nq_scalp_10pt_backtest.json","w"),indent=1,default=str)

