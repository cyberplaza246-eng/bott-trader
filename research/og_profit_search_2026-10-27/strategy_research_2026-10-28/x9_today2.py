import pandas as pd, numpy as np
d=pd.read_csv('x9_nq_1m_today.csv',index_col=0); d.index=pd.to_datetime(d.index,utc=True).tz_convert('America/New_York')
t=d[d.index.strftime('%Y-%m-%d')=='2026-10-28']
def seg(a,b):
    s=t.between_time(a,b); hi=s.High.max(); lo=s.Low.min()
    print(a,b,'open',s.Open.iloc[0],'close',s.Close.iloc[-1],'high',hi,s.High.idxmax().strftime('%H:%M'),'low',lo,s.Low.idxmin().strftime('%H:%M'))
seg('11:30','12:59'); seg('12:00','12:59'); seg('12:45','13:30'); seg('13:00','13:59'); seg('13:55','14:35'); seg('14:00','14:59'); seg('15:00','15:59'); seg('16:00','16:59')
print('RTH close 15:59',t.between_time('15:59','15:59').Close.iloc[0],'| 16:00-16:59 range hi',t.between_time('16:00','16:59').High.max())
# rule triggers today
W=(60,90); 
for X in (80,120,180):
  for Wm in W:
    tt=t.between_time('09:30','15:59'); 
    hit=None
    for ts in t.between_time('11:30','14:00').index:
        win=t.loc[ts-pd.Timedelta(minutes=Wm-1):ts]
        if len(win)>=Wm*0.9 and win.High.max()-t.loc[ts].Close>=X: hit=ts;break
    if hit is None: print(X,Wm,'no trigger'); continue
    e=t.loc[hit+pd.Timedelta(minutes=1)].Open
    res=[]
    for H in (30,60,120):
        j=min(hit+pd.Timedelta(minutes=1+H),pd.Timestamp('2026-10-28 15:55',tz='America/New_York'))
        xo=t.loc[j].Open if j in t.index else np.nan
        res.append((H,round(1*((xo-0.25)-(e+0.25))*2-4,1),round(-1*((xo+0.25)-(e-0.25))*2-4,1)))
    print(X,Wm,'trigger',hit.strftime('%H:%M'),'entry',e,'(H, long$, short$)',res)
