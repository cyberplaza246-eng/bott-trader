import pandas as pd,numpy as np
m=pd.read_csv('../MNQ_1m.csv');m['datetime']=pd.to_datetime(m.datetime,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None);m=m.set_index('datetime')
t=pd.read_pickle('pq.pkl');t['eE']=t.et+pd.Timedelta(hours=3);t['xE']=t.xt+pd.Timedelta(hours=3)
t=t[t.eE>=m.index[0]]
off=(t['Entry price'].values-m.close.reindex(t.eE).values);t['off']=pd.Series(off,index=t.index).groupby(t.eE.dt.to_period('M')).transform('median')
r=[]
for _,x in t.iterrows():
    g=m[(m.index>x.eE)&(m.index<=x.xE)]
    if len(g)==0: r.append((np.nan,np.nan,np.nan));continue
    ep=x['Exit price']-x.off;inside=(g.low.iloc[-1]-.5<=ep<=g.high.iloc[-1]+.5)
    # conservative: entry at next 1m open after entry, exit at close of exit minute
    e2=m.open.reindex([x.eE+pd.Timedelta(minutes=1)]).iloc[0];x2=g.close.iloc[-1]
    r.append((inside,x.s*(x2-e2),(g.high.max()-g.low.min())))
t[['inside','pts_cons','rng']]=r
t['usd_cons']=t.pts_cons*2-5
print(len(t),'exit price inside exit bar',t.inside.mean())
print('reported pts mean',t.pts.mean(),'conservative pts mean',t.pts_cons.mean(),'wr cons',(t.pts_cons>0).mean())
eq=t.usd_cons.cumsum();print('MNQ1 cons net',eq.iloc[-1],'DD',(eq.cummax()-eq).max())
