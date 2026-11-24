import sys,io,contextlib,numpy as np,pandas as pd
sys.dont_write_bytecode=True
with contextlib.redirect_stdout(io.StringIO()): import sr_gap as G
ET='America/New_York'
base=pd.read_csv('trades_base.csv'); 
for c in('entry_time','exit_time'): base[c]=pd.to_datetime(base[c],utc=True)
FOLDS=[('2025-05-01','2025-08-01'),('2025-08-01','2025-11-01'),('2025-11-01','2026-02-01'),('2026-02-01','2026-05-01'),('2026-05-01','2026-08-01'),('2026-08-01','2026-10-01')]
def dd(p): eq=np.cumsum(p); return float((np.maximum.accumulate(np.concatenate([[0],eq]))[1:]-eq).max())
def S(df):
    df=df.sort_values('exit_time'); p=df.pnl.values; mon=df.groupby(df.exit_time.dt.tz_convert(ET).dt.strftime('%Y-%m')).pnl.sum()
    fo=[df[(df.entry_time>=pd.Timestamp(a,tz='UTC'))&(df.entry_time<pd.Timestamp(b,tz='UTC'))].pnl.sum() for a,b in FOLDS]
    dday=df.groupby(df.exit_time.dt.tz_convert(ET).dt.date).pnl.sum()
    return dict(n=len(p),net=round(p.sum()),pf=round(p[p>0].sum()/-p[p<0].sum(),2),dd=round(dd(p)),green=f'{(mon>0).sum()}/{len(mon)}',train=round(df[df.entry_time<'2026-01-01'].pnl.sum()),y26=round(df[df.entry_time>='2026-01-01'].pnl.sum()),wf=round(sum(fo)),folds=[round(x) for x in fo],worst_day=round(dday.min()))
print('base',S(base))
for q,H in((0.5,60),(0.5,90),(0.75,90)):
    for slip in (1,3):
        g=G.run(q,H,slip_ticks=slip)[['entry_time','exit_time','pnl']]
        print('gap',q,H,'slip',slip,S(g))
        print(' base+gap',q,H,'slip',slip,S(pd.concat([base[['entry_time','exit_time','pnl']],g])))
    dg=G.run(q,H).groupby(lambda i:0).pnl.sum()
# correlation of daily pnl
g=G.run(0.5,90); gd=g.groupby(g.exit_time.dt.tz_convert(ET).dt.date).pnl.sum(); bd=base.groupby(base.exit_time.dt.tz_convert(ET).dt.date).pnl.sum()
j=pd.concat([bd,gd],axis=1).fillna(0); print('corr daily',round(j.corr().iloc[0,1],3))
