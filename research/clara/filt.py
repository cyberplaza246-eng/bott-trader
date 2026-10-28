import pandas as pd,numpy as np,itertools,json
from sim import run,days
from report import metrics, worst
f=pd.read_pickle('dayf.pkl')
BASE=dict(wmax=250,k=0.8,lockR=0.5,lockf=0.4,trail=0.6,cut=11,buf=0,smin=20)
FILTERS={'none':lambda r:True,
 'eff>=.5':lambda r:r.eff>=.5,'eff>=.35':lambda r:r.eff>=.35,
 'orw_rel<=1.0':lambda r:r.orw_rel<=1.0,'orw_rel<=1.3':lambda r:r.orw_rel<=1.3,
 'orw/pdr<=.35':lambda r:r.orw/r.pdr<=.35,'orw/pdr<=.5':lambda r:r.orw/r.pdr<=.5,
 'onr_rel<=1':lambda r:r.onr_rel<=1,'gap_rel<=.5':lambda r:r.gap_rel<=.5,'gap_rel>=.3':lambda r:r.gap_rel>=.3}
CUTS=[10.0,10.5,11]
EXITS={'lock.5_trail.6':dict(lockR=.5,trail=.6),'lock1_notrail':dict(lockR=1.0,trail=9),'lock.5_trail1':dict(lockR=.5,trail=1.0),'k1.0':dict(k=1.0)}
def go(fn,cut,ex,lo,hi):
    dd={d:g for d,g in days.items() if lo<=d<=hi and d in f.index and fn(f.loc[d])}
    P=dict(BASE,cut=cut,**EXITS[ex]);t=run(P,dd)
    if len(t)==0: return t
    t['d']=t.et.dt.normalize();return t
if __name__=='__main__':
    lo,hi=pd.Timestamp('2024-09-25'),pd.Timestamp('2025-12-31')
    res=[];cache={}
    for fn,cut,ex in itertools.product(FILTERS,CUTS,EXITS):
        t=go(FILTERS[fn],cut,ex,lo,hi);cache[(fn,cut,ex)]=t
        if len(t)<40: continue
        t['wo']=0;eq=t.usd.cumsum();dd=(eq.cummax().clip(lower=0)-eq).max()
        res.append(dict(f=fn,cut=cut,ex=ex,n=len(t),wr=round((t.usd>0).mean(),3),net=round(t.usd.sum()),dd=round(dd)))
    r=pd.DataFrame(res);r.to_csv('filt_train.csv',index=False)
    print('variants',len(r));print(r.sort_values('net',ascending=False).head(15).to_string())
    pd.to_pickle(cache,'filt_cache.pkl')
FILTERS2={'ybody<.5':lambda r:r.ybody<.5,'open_in_y':lambda r:bool(r.open_in_y),'noise>=1':lambda r:r.noise>=1,'noise>=1.3':lambda r:r.noise>=1.3}
