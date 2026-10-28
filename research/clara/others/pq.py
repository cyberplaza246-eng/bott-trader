import pandas as pd,numpy as np
m=pd.read_csv('../MNQ_1m.csv');m['datetime']=pd.to_datetime(m.datetime,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None);m=m.set_index('datetime')
t=pd.read_csv('pq.csv');mo=lambda s:s.astype(str).str.replace(r'[\$,]','',regex=True).str.replace(r'^\((.*)\)$',r'-\1',regex=True).astype(float)
t['P']=mo(t.Profit);t['et']=pd.to_datetime(t['Entry time']);t['xt']=pd.to_datetime(t['Exit time'])
t['s']=np.where(t['Market pos.']=='Long',1,-1);t['pts']=t.s*(t['Exit price']-t['Entry price'])
print(len(t),t.et.min(),t.et.max(),'wr',(t.P>0).mean(),'avg pts',t.pts.mean(),'trades/day',t.groupby(t.et.dt.date).size().mean())
print(t.et.dt.hour.value_counts().sort_index().to_dict());print(t['Exit name'].value_counts().head())
print('hold min',((t.xt-t.et).dt.total_seconds()/60).describe()[['50%','mean']].to_dict())
# timezone & price-offset: daily median diff entry - our 1m close at et+h
for h in [0,1,2,3]:
    x=t.et+pd.Timedelta(hours=h);c=m.close.reindex(x).values;dif=t['Entry price'].values-c
    s=pd.Series(dif).groupby(t.et.dt.to_period('Q').values).std()
    print(h,np.nanmedian(np.abs(dif-pd.Series(dif).groupby(t.et.dt.to_period('Q').values).transform('median'))))
# MNQ-equivalent with costs: pts*2 -5 per trade (1 MNQ)
t['usd']=t.pts*2-5;eq=t.usd.cumsum();print('MNQ1 net',eq.iloc[-1].round(),'DD',(eq.cummax()-eq).max().round(),'per',t.usd.mean().round(2))
# fill realism: does the exit price lie inside the 1m bar of exit (tz to pick)
t.to_pickle('pq.pkl')
