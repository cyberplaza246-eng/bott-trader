import pandas as pd,numpy as np
from sim import days
b=pd.read_pickle('b2.pkl')
rows=[];prev=None
for d,g in days.items():
    o=g[g.index<=d+pd.Timedelta(minutes=590)]
    W=o.high.max()-o.low.min();eff=abs(o.close.iloc[-1]-g.open.iloc[0])/W
    on=b[(b.index>d-pd.Timedelta(hours=6))&(b.index<=d+pd.Timedelta(minutes=570))]
    onr=on.high.max()-on.low.min() if len(on) else np.nan
    rows.append(dict(d=d,orw=W,eff=eff,onr=onr,rth_range=g.high.max()-g.low.min(),
      gap=g.open.iloc[0]-(prev.close.iloc[-1] if prev is not None else np.nan),
      daymove=abs(g.close.iloc[-1]-g.open.iloc[0])))
    prev=g
f=pd.DataFrame(rows).set_index('d')
f['pdr']=f.rth_range.shift(1)                       # prior day range (known at entry)
f['orw_rel']=f.orw/f.orw.shift(1).rolling(10).median()  # vs prior 10 days
f['gap_rel']=f.gap.abs()/f.pdr
f['onr_rel']=f.onr/f.pdr
f.to_pickle('dayf.pkl');print(f.describe().T[['mean','25%','50%','75%']])
