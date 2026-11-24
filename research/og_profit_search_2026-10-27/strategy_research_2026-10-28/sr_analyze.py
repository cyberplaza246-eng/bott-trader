import os,sys,glob,json
sys.dont_write_bytecode=True
import numpy as np, pandas as pd
R=os.path.dirname(os.path.abspath(__file__)); ET='America/New_York'
cut=pd.Timestamp('2026-01-01',tz='UTC')
FOLDS=[('2025-05-01','2025-08-01'),('2025-08-01','2025-11-01'),('2025-11-01','2026-02-01'),('2026-02-01','2026-05-01'),('2026-05-01','2026-08-01'),('2026-08-01','2026-10-01')]
def load(n):
    df=pd.read_csv(os.path.join(R,f'trades_{n}.csv'),parse_dates=['entry_time','exit_time']); 
    for c in ('entry_time','exit_time'):
        if df[c].dt.tz is None: df[c]=df[c].dt.tz_localize('UTC')
    return df
def daily(df): return df.groupby(df.exit_time.dt.tz_convert(ET).dt.date).pnl.sum()
def maxdd(p): eq=np.cumsum(p); return float((np.maximum.accumulate(np.concatenate([[0],eq]))[1:]-eq).max())
base=load('base'); bd=daily(base)
rng=np.random.default_rng(1)
def row(n):
    df=load(n); p=df.pnl.values; g=p[p>0].sum(); l=-p[p<0].sum()
    mon=df.groupby(df.exit_time.dt.tz_convert(ET).dt.strftime('%Y-%m')).pnl.sum()
    fo=[df[(df.entry_time>=pd.Timestamp(a,tz='UTC'))&(df.entry_time<pd.Timestamp(b,tz='UTC'))].pnl.sum() for a,b in FOLDS]
    bfo=[base[(base.entry_time>=pd.Timestamp(a,tz='UTC'))&(base.entry_time<pd.Timestamp(b,tz='UTC'))].pnl.sum() for a,b in FOLDS]
    d=(daily(df).reindex(bd.index.union(daily(df).index),fill_value=0)-bd.reindex(bd.index.union(daily(df).index),fill_value=0)).values
    bs=[rng.choice(d,len(d)).sum() for _ in range(4000)]
    tr=df[df.entry_time<cut].pnl.sum(); te=df[df.entry_time>=cut].pnl.sum()
    btr=base[base.entry_time<cut].pnl.sum(); bte=base[base.entry_time>=cut].pnl.sum()
    wf=sum(fo)-sum(bfo)
    return dict(name=n,n=len(p),net=round(p.sum()),pf=round(g/l,2),dd=round(maxdd(p)),green=f'{(mon>0).sum()}/{len(mon)}',train=round(tr),y2026=round(te),
      d_net=round(p.sum()-base.pnl.sum()),d_train=round(tr-btr),d_2026=round(te-bte),wf_oos=round(sum(fo)),d_wf=round(wf),folds_beat=f'{sum(a>b for a,b in zip(fo,bfo))}/6',
      boot_p=round(float(np.mean(np.array(bs)>0)),2))
names=[os.path.basename(f)[7:-4] for f in glob.glob(os.path.join(R,'trades_*.csv'))]
names=['base']+sorted(n for n in names if n!='base')
rows=[row(n) for n in names]
T=pd.DataFrame(rows); T.to_csv(os.path.join(R,'REPORT_TABLE.csv'),index=False)
def flag(r):
    if r['name']=='base': return ''
    wins=[r['d_train']>0,r['d_2026']>0,r['d_wf']>0,r['folds_beat'].split('/')[0]>='4']
    return 'ROBUST' if all(wins) and r['boot_p']>=0.8 else ('one-window/noise' if any(wins) else 'worse')
T['verdict']=[flag(r) for r in rows]
open(os.path.join(R,'REPORT_TABLE.md'),'w').write('|'+'|'.join(T.columns)+'|\n|'+'|'.join(['---']*len(T.columns))+'|\n'+'\n'.join('|'+'|'.join(str(v) for v in r)+'|' for r in T.values))
print(T.to_string())
