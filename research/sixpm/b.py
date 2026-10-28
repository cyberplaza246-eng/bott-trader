import pandas as pd,numpy as np
exec(open('a.py').read().split('for p,g')[0])
df['yr']=df.t.dt.year
print(df.groupby(['yr','dow']).pnl.agg(['count','mean']).round(0).unstack())
rules={'base':df.pnl*0==0,'skipMon':df.dow!=0,'skipADXlo':df.adx>20.56,'skipMon&ADXlo':(df.dow!=0)&(df.adx>20.56)}
for n,m in rules.items():
  for p,g in [('TR',df.tr),('TE',~df.tr)]:
    print(n,p,'SKIP',st(df[g&m]))
    x=df[g].copy();x['pnl']=np.where(m[g],x.pnl,x.pnl*0.5);print(n,p,'HALF*',st(x))
# bootstrap-ish: Monday mean vs others t-stat
from scipy import stats
print(stats.ttest_ind(df[df.dow==0].pnl,df[df.dow!=0].pnl,equal_var=False))
print(stats.ttest_ind(df[df.adx<=20.56].pnl,df[df.adx>20.56].pnl,equal_var=False))
