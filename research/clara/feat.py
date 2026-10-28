import pandas as pd,numpy as np
m=pd.read_csv('MNQ_1m.csv');m['datetime']=pd.to_datetime(m.datetime,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
m=m.set_index('datetime')
# 2m bars labelled by close time (NT convention)
b=m.resample('2min',label='right',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
b.to_pickle('b2.pkl')
t=pd.read_pickle('t.pkl');t=t[t.et>=b.index[0]+pd.Timedelta('2D')]
t=t[t.et<=b.index[-1]]
rows=[]
for _,r in t.iterrows():
    d=r.et.normalize();day=b[(b.index>d+pd.Timedelta('9h30min'))&(b.index<=d+pd.Timedelta('16h'))]
    if len(day)<50 or r.et not in day.index: rows.append(None);continue
    o=day.iloc[0].open
    orb=day[day.index<=d+pd.Timedelta('9h50min')]
    pre=day[day.index<=r.et]
    tp=(pre.high+pre.low+pre.close)/3;vw=(tp*pre.volume).sum()/pre.volume.sum()
    c=day.loc[r.et].close
    s=1 if r['Market pos.']=='Long' else -1
    prev=b[(b.index>d-pd.Timedelta('4D'))&(b.index<d)]
    pc=prev[prev.index.time<=pd.Timestamp('16:00').time()].close.iloc[-1] if len(prev) else np.nan
    ema=b.close[b.index<=r.et].ewm(span=20).mean().iloc[-1]
    rows.append(dict(et=r.et,s=s,fill=r['Entry price'],c=c,
      dopen=s*(c-o),gap=s*(o-pc),orh=orb.high.max(),orl=orb.low.min(),orw=orb.high.max()-orb.low.min(),
      vsorh=c-orb.high.max(),vsorl=c-orb.low.min(),dvw=s*(c-vw),dema=s*(c-ema),
      hod=c>=pre.high.max()-1, lod=c<=pre.low.min()+1,
      last3=s*(c-pre.close.iloc[-4]) ,bar=s*(pre.close.iloc[-1]-pre.open.iloc[-1]),pts=r.pts,win=r.Profit>0))
f=pd.DataFrame([x for x in rows if x]);f.to_pickle('f.pkl')
print(len(t),len(f),'fill-close diff',(f.fill-f.c).describe())
print(f.describe().T.to_string())
print('long breaks ORH',((f.s==1)&(f.vsorh>0)).sum(),'/',(f.s==1).sum(),' short below ORL',((f.s==-1)&(f.vsorl<0)).sum(),'/',(f.s==-1).sum())
print('side with open move',(f.dopen>0).mean(),'with vwap',(f.dvw>0).mean(),'with ema',(f.dema>0).mean(),'with gap',(f.gap>0).mean(),'last bar with',(f.bar>0).mean(),'last3 with',(f.last3>0).mean())
