"""Addendum 11 analysis (reads x11_trades_*.csv; writes x11_results.md / x11_windows.csv). BACKTEST ONLY."""
import pandas as pd,numpy as np,os,sys,json
ET='America/New_York'
def load(tag,s,win='main'):
    d=pd.read_csv(f'x11_trades_{tag}_s{s}_{win}.csv')
    for c in ('entry_time','exit_time'): d[c]=pd.to_datetime(d[c],utc=True)
    d['xday']=d.exit_time.dt.tz_convert(ET).dt.strftime('%Y-%m-%d')
    d['key']=d.entry_time.astype(str)+'|'+d.tag+'|'+d.dir.astype(str)
    return d
def T(x): return pd.Timestamp(x,tz='UTC')
def mdd(p):
    if len(p)==0: return 0.0
    eq=np.cumsum(p); return float((np.maximum.accumulate(np.concatenate([[0],eq]))[1:]-eq).max())
def pf(p):
    g=p[p>0].sum(); l=-p[p<0].sum(); return round(float(g/l),2) if l>0 else 99.0
def pb(x,n=10000,seed=3):
    x=np.asarray(x,float)
    if len(x)==0: return float('nan')
    r=np.random.default_rng(seed); return round(float((r.choice(x,(n,len(x))).sum(1)>0).mean()),3)
def sel(d,a,b): return d[(d.entry_time>=T(a))&(d.entry_time<T(b))]
def cmp(base,v,a,b):
    bb=sel(base,a,b); vv=sel(v,a,b)
    bd=bb.groupby('xday').pnl.sum(); vd=vv.groupby('xday').pnl.sum()
    j=pd.concat([vd,bd],axis=1,keys=['v','b']).fillna(0); dl=(j.v-j.b).values
    rem=len(set(bb.key)-set(vv.key)); add=len(set(vv.key)-set(bb.key))
    return dict(n_base=len(bb),n_filt=len(vv),removed=rem,added=add,base_net=round(bb.pnl.sum()),filt_net=round(vv.pnl.sum()),delta=round(vv.pnl.sum()-bb.pnl.sum()),
                base_pf=pf(bb.pnl.values),filt_pf=pf(vv.pnl.values),base_dd=round(mdd(bb.sort_values('exit_time').pnl.values)),filt_dd=round(mdd(vv.sort_values('exit_time').pnl.values)),p_better=pb(dl))
WINS=[('FULL Nov24-Sep26','2024-11-01','2026-10-01'),('train <2026','2024-11-01','2026-01-01'),('2026 YTD','2026-01-01','2026-10-01'),
      ('last 6m (Apr-Sep26)','2026-04-01','2026-10-01'),('LAST 4m (May22-Sep21)','2026-05-22','2026-10-01'),('LAST 3m (Jun22-Sep21)','2026-06-22','2026-10-01'),('last 2m (Jul22-Sep21)','2026-07-22','2026-10-01'),
      ('Q4-24 (Nov-Dec)','2024-11-01','2025-01-01'),('Q1-25','2025-01-01','2025-04-01'),('Q2-25','2025-04-01','2025-07-01'),('Q3-25','2025-07-01','2025-10-01'),('Q4-25','2025-10-01','2026-01-01'),('Q1-26','2026-01-01','2026-04-01'),('Q2-26','2026-04-01','2026-07-01'),('Q3-26 (Jul-Sep21)','2026-07-01','2026-10-01')]
if __name__=='__main__':
    rows=[]; out=[]
    NAMES={'f1':'F1 off_18_0930_trend_short','f2':'F2 align_htf2880','both':'F1+F2 together'}
    for s,sn in ((1,'1-tick (baseline cost)'),(4,'4-tick slippage stress')):
        base=load('base',s)
        for t in ('f1','f2','both'):
            v=load(t,s)
            for w,a,b in WINS:
                r=cmp(base,v,a,b); r.update(filter=NAMES[t],slip=sn,window=w); rows.append(r)
            # six equal-time folds
            edges=pd.date_range(base.entry_time.min(),base.entry_time.max(),periods=7)
            for i in range(6):
                r=cmp(base,v,edges[i],edges[i+1]) if False else None
            fl=[]
            for i in range(6):
                bb=base[(base.entry_time>=edges[i])&(base.entry_time<=edges[i+1])]; vv=v[(v.entry_time>=edges[i])&(v.entry_time<=edges[i+1])]
                fl.append((edges[i].strftime('%y-%m-%d'),edges[i+1].strftime('%y-%m-%d'),round(vv.pnl.sum()-bb.pnl.sum()),len(bb),len(vv)))
            for i,f in enumerate(fl): rows.append(dict(filter=NAMES[t],slip=sn,window=f'FOLD{i+1} {f[0]}..{f[1]}',n_base=f[3],n_filt=f[4],delta=f[2]))
        # early-window (unused data Oct 2024)
        try:
            eb=load('base',s,'early'); 
            for t in ('f1','f2','both'):
                ev=load(t,s,'early'); r=cmp(eb,ev,'2024-09-30','2024-11-01'); r.update(filter=NAMES[t],slip=sn,window='UNUSED Oct-2024 (Sep30-Oct31)'); rows.append(r)
        except Exception as e: print('early missing',e)
    R=pd.DataFrame(rows); R.to_csv('x11_windows.csv',index=False)
    pd.set_option('display.width',300); pd.set_option('display.max_rows',500); pd.set_option('display.max_columns',30)
    for (f,sl),g in R.groupby(['filter','slip'],sort=False):
        print('\n###',f,'|',sl); print(g.drop(columns=['filter','slip']).to_string(index=False))
    # frozen-forward: delta before vs after cutoff
    print('\nFROZEN-FORWARD (delta before cutoff | after cutoff), 4-tick'); 
    base=load('base',4)
    for t in ('f1','f2'):
        v=load(t,4)
        for c in ('2025-07-01','2025-10-01','2026-01-01','2026-04-01','2026-06-22'):
            a=cmp(base,v,'2024-11-01',c)['delta']; b=cmp(base,v,c,'2026-10-01')['delta']; print(NAMES[t],c,a,'|',b)
