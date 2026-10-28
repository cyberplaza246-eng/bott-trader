import numpy as np,pandas as pd
d=pd.read_parquet('/workspace/intrabar/1m_long.parquet');d.index=d.index.tz_convert('America/New_York')
TS=d.index; O,H,L,C=[d[c].to_numpy() for c in 'open high low close'.split()]
M=np.asarray(TS.hour*60+TS.minute)
m15=d.resample('15min',label='right',closed='right').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
tr=pd.concat([m15.high-m15.low,(m15.high-m15.close.shift()).abs(),(m15.low-m15.close.shift()).abs()],axis=1).max(axis=1)
a15=tr.ewm(alpha=1/14,adjust=False).mean()
# 15m ADX
up=m15.high.diff();dn=-m15.low.diff()
pdm=np.where((up>dn)&(up>0),up,0);ndm=np.where((dn>up)&(dn>0),dn,0)
atr=tr.ewm(alpha=1/14,adjust=False).mean()
pdi=100*pd.Series(pdm,m15.index).ewm(alpha=1/14,adjust=False).mean()/atr;ndi=100*pd.Series(ndm,m15.index).ewm(alpha=1/14,adjust=False).mean()/atr
adx=(100*(pdi-ndi).abs()/(pdi+ndi)).ewm(alpha=1/14,adjust=False).mean()
e50_15=m15.close.ewm(span=50,adjust=False).mean()
h1=d.close.resample('1h',label='right',closed='right').last().dropna()
e20h=h1.ewm(span=20,adjust=False).mean();e50h=h1.ewm(span=50,adjust=False).mean()
def at(s,t): return s[s.index<=t].iloc[-1]
rth=d.between_time('09:30','15:59')
day=rth.groupby(rth.index.date).agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last'))
day.index=pd.to_datetime(day.index)
day['ret']=day.c/day.c.shift()-1; day['rv']=np.log(day.c).diff().rolling(20).std()*np.sqrt(252)
day['hi20']=day.h.rolling(20).max(); day['r5']=day.c/day.c.shift(5)-1
rows=[];prev=None
for k in np.where(M==1080)[0]:
    t=TS[k]
    if k+60>=len(TS): continue
    s=float(np.clip(1.5*at(a15,t),12,40)); e=O[k]; sl=e-s; tp=e+400; lock=False; b=k; xp=None
    while b<len(TS):
        if b>k and (TS[b]-TS[b-1]).total_seconds()>3*3600: xp=C[b-1];break
        if L[b]<=sl: xp=min(sl,O[b]) if b>k else sl;break
        if H[b]>=tp: xp=max(tp,O[b]);break
        if not lock and (H[b]-e)*2>=400: lock=True; sl=max(sl,e+25)
        if M[b]==566: xp=C[b];break   # exit at 9:26 bar close~
        b+=1
    if xp is None: continue
    pnl=(xp-e)*2-4
    dd=day[day.index<pd.Timestamp(t.date())]; dd=day[day.index<=pd.Timestamp(t.date())]
    if len(dd)<21: continue
    r=dd.iloc[-1]; samed = r.name.date()==t.date()
    rows.append(dict(t=t,pnl=pnl,dow=t.dayofweek,sunday=not samed,
      dret=r.ret if samed else np.nan, cloc=(r.c-r.l)/(r.h-r.l) if samed else np.nan,
      gap=e/r.c-1, r5=r.r5, rv=r.rv, dhi20=e/r.hi20-1, atr15=at(a15,t),
      adx=at(adx,t), h1up=int(at(e20h,t)>at(e50h,t)), s15=at(e50_15,t)-at(e50_15,t-pd.Timedelta('15min')),
      prev=prev, monthend=(t+pd.offsets.BDay(1)).month!=t.month))
    prev=pnl
df=pd.DataFrame(rows);df.to_pickle('tr.pkl');print(len(df),df.pnl.mean(),(df.pnl>0).mean(),df[df.pnl>0].pnl.mean(),df[df.pnl<=0].pnl.mean())
