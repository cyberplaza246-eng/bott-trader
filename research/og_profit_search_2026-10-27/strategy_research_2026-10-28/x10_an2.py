import pandas as pd,numpy as np,json
import sr_x10_lib as L
from x10_an import *
pd.set_option('display.width',330); pd.set_option('display.max_columns',50); pd.set_option('display.max_rows',200)
b=pd.read_pickle('x10_base.pkl')   # has seg, sleeve, side, m, hr, win
b['fold']=sum((b.entry_time>edges[i]).astype(int) for i in range(1,6))
b['AB']=np.where(b.entry_time<pd.Timestamp('2025-11-01',tz='UTC'),'A','B')
b['xday']=b.exit_time.dt.tz_convert(ET).dt.strftime('%Y-%m-%d')
# ---- 1. standalone cell table: strict proof tiers
cells=[]
def cellrow(g,**k):
    p=g.pnl.values; n=len(p); r=dict(k); r['n']=n; r['net']=round(p.sum()); r['avg']=round(p.mean(),1)
    r['t']=round(p.mean()/(p.std(ddof=1)/np.sqrt(n)),2) if n>2 and p.std()>0 else 0
    r['pf']=L.met(g)['pf']
    r['train']=round(g[g.win=='train'].pnl.sum()); r['y2026']=round(g[g.win=='y2026'].pnl.sum())
    r['A']=round(g[g.AB=='A'].pnl.sum()); r['B']=round(g[g.AB=='B'].pnl.sum())
    fn=g.groupby('fold').pnl.sum().reindex(range(6)).fillna(0); r['folds_pos']=int((fn>0).sum()); r['fold_net']=[round(x) for x in fn]
    # day-bootstrap of this cell's per-day pnl: P(sum>0)
    ds=g.groupby('xday').pnl.sum(); r['p_pos']=pb(ds.values,5000)
    return r
for (s,sl,sd),g in b[b.seg.str[0].isin(['2','3','4'])].groupby(['seg','sleeve','side']): cells.append(cellrow(g,seg=s,sleeve=sl,side=sd))
for (s,sl),g in b[b.seg.str[0].isin(['2','3','4'])].groupby(['seg','sleeve']): cells.append(cellrow(g,seg=s,sleeve=sl,side='both'))
for s,g in b[b.seg.str[0].isin(['1','2','3','4'])].groupby('seg'): cells.append(cellrow(g,seg=s,sleeve='ALL',side='both'))
for (s,sd),g in b[b.seg.str[0].isin(['1','3','4'])].groupby(['seg','side']): cells.append(cellrow(g,seg=s,sleeve='ALL',side=sd))
n=b[b.seg.str[0].isin(['3','4'])]; cells.append(cellrow(n,seg='18:00-09:30',sleeve='ALL',side='both'))
for sd,g in n.groupby('side'): cells.append(cellrow(g,seg='18:00-09:30',sleeve='ALL',side=sd))
n2=b[b.seg.str[0].isin(['3'])]; 
C=pd.DataFrame(cells); C['proven']=(C.train>0)&(C.y2026>0)&(C.A>0)&(C.B>0)&(C.folds_pos>=4)&(C.t>=2)
C.to_csv('x10_cells.csv',index=False)
print(C.drop(columns=['fold_net']).to_string())
print(C[['seg','sleeve','side','fold_net']].to_string())

