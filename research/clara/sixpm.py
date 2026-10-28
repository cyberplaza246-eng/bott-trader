# Shawayne's 6 PM long, re-implemented from summary: buy 18:00 ET, stop clip(1.5*ATR15,12,40), exit 09:26 ET, skip Monday. Trail params unknown -> not modeled (flag).
import pandas as pd,numpy as np,sys
TRAIL=1
m=pd.read_csv('MNQ_1m.csv');m['datetime']=pd.to_datetime(m.datetime,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None);m=m.set_index('datetime')
b15=m.resample('15min').agg({'high':'max','low':'min','close':'last'}).dropna()
tr=np.maximum(b15.high-b15.low,np.maximum(abs(b15.high-b15.close.shift()),abs(b15.low-b15.close.shift())));atr=tr.rolling(14).mean()
out=[]
for d in pd.date_range(m.index[0].normalize(),m.index[-1].normalize()):
    if d.weekday() in (0,4,5): continue   # Mon skip; Fri/Sat 18:00 no next-session
    t0=d+pd.Timedelta(hours=18)
    if t0 not in m.index: continue
    a=atr[atr.index<t0];
    if len(a)==0 or np.isnan(a.iloc[-1]): continue
    st=float(np.clip(1.5*a.iloc[-1],12,40));e=m.open[t0]
    g=m[(m.index>=t0)&(m.index<d+pd.Timedelta(hours=33,minutes=26))]
    A=a.iloc[-1];sp=e-st;hi=e;x=None
    for ti,o_,h_,l_,c_ in zip(g.index,g.open.values,g.high.values,g.low.values,g.close.values):
        if l_<=sp: x=min(sp,o_);xt=ti;break
        hi=max(hi,h_)
        if TRAIL and hi-e>=100: sp=max(sp,hi-3*A)
    if x is None: x=g.close.iloc[-1];xt=g.index[-1]
    w=g[g.index<=xt].low.min()
    out.append(dict(et=t0,xt=xt,d=(d+pd.Timedelta(days=1)).normalize(),e=e,usd=(x-e-0.5)*2-4,wo=(w-e-0.5)*2-4))
pd.DataFrame(out).to_pickle('six.pkl');s=pd.DataFrame(out);print(len(s),(s.usd>0).mean(),s.usd.sum(),s.usd.mean())
