import pandas as pd,numpy as np
import x11_an as A
for s in (1,4):
    b=A.load('base',s)
    for t in ('f1','f2'):
        v=A.load(t,s); rem=b[~b.key.isin(set(v.key))]; add=v[~v.key.isin(set(b.key))]
        print(s,t,'removed n',len(rem),'sum',round(rem.pnl.sum()),'wins',int((rem.pnl>0).sum()),'| added n',len(add),'sum',round(add.pnl.sum()),'wins',int((add.pnl>0).sum()))
        l3=rem[rem.entry_time>=A.T('2026-06-22')]; a3=add[add.entry_time>=A.T('2026-06-22')]
        print('   last3m removed',len(l3),round(l3.pnl.sum()),'added',len(a3),round(a3.pnl.sum()))
        print('   removed by month green/red:',(rem.groupby('month').pnl.sum()>0).sum(),'/',(rem.groupby('month').pnl.sum()<=0).sum())
        # months with positive delta
        bm=b.groupby('month').pnl.sum(); vm=v.groupby('month').pnl.sum(); d=(vm.reindex(bm.index).fillna(0)-bm)
        print('   months delta>0:',int((d>0).sum()),'<0:',int((d<0).sum()),'=0:',int((d==0).sum()),'of',len(d))
