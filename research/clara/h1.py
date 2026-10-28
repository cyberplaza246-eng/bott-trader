import pandas as pd,numpy as np
b=pd.read_pickle('b2.pkl');t=pd.read_pickle('t.pkl')
t=t[(t.et>'2024-09-25')&(t.et<'2026-09-20')]
first=t.groupby(t.et.dt.date).head(1)
res=[]
for _,r in first.iterrows():
    d=r.et.normalize();day=b[(b.index>d+pd.Timedelta('9h30min'))&(b.index<=d+pd.Timedelta('15h30min'))]
    if len(day)<100: continue
    o=day.open.iloc[0];orb=day[day.index<=d+pd.Timedelta('9h50min')];H,L=orb.high.max(),orb.low.min()
    tp=(day.high+day.low+day.close)/3;vw=(tp*day.volume).cumsum()/day.volume.cumsum()
    after=day[day.index>d+pd.Timedelta('9h50min')]
    s=1 if r['Market pos.']=='Long' else -1
    br=after[(after.close>H)] if s==1 else after[(after.close<L)]
    anyb=after[(after.close>H)|(after.close<L)]
    fb=anyb.index[0] if len(anyb) else None
    fbs=(1 if anyb.close.iloc[0]>H else -1) if len(anyb) else 0
    res.append(dict(et=r.et,s=s,firstbreak=fb,fbs=fbs,lag=(r.et-fb).total_seconds()/60 if fb is not None else np.nan,orw=H-L,win=r.Profit>0,pts=r.pts,exit=r['Exit name']))
x=pd.DataFrame(res);print(len(x))
print('first break same side',(x.fbs==x.s).mean());print(x.lag.describe());print((x.lag==0).mean(),(x.lag.abs()<=4).mean())
print(x.orw.describe())
print(x.lag.value_counts().head(10))
f=pd.read_pickle('f.pkl');t2=pd.read_pickle('t.pkl')
f=f.merge(t2[['et','Exit name','MAE']],on='et')
L=f[(f.pts<-20)&(f['Exit name']=='Stop loss')].copy()
L['opp']=np.where(L.s==1,L.c-L.orl,L.orh-L.c)
L['loss']=-L.pts
print(L[['loss','opp','orw','dvw']].corr().loss);print((L.loss/L.orw).describe());print((L.loss/L.opp).describe())
# all days: days with no trade, orw
b=pd.read_pickle('b2.pkl');days=pd.Series(b.index.normalize().unique())
