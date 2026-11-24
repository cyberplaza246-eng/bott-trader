"""RTH-gap fade robustness (BACKTEST ONLY). gap = 09:30 open - prior 15:59 close; fade it (short gap-up / long gap-down) for H minutes. Threshold = expanding past |gap| quantile (no lookahead)."""
import os,sys,json
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
R=os.path.dirname(os.path.abspath(__file__))
import sr_engine as S
E=S.E; import numpy as np, pandas as pd
ET=S.ET; d=E.load(); m1=d['m1']; o=m1.open.values; hi=m1.high.values; lo=m1.low.values; ix=m1.index
def idx_at(ts):
    i=ix.searchsorted(ts)
    return i if i<len(ix) and (ix[i]-ts)<=pd.Timedelta(minutes=3) else None
def T(ds,hm): return pd.Timestamp(f'{ds} {hm}',tz=ET).tz_convert('UTC')
days=pd.bdate_range('2024-10-01','2026-09-18')
gaps=[]  # (day, gap, i930, prevclose)
prev=None
for dd in days:
    ds=dd.strftime('%Y-%m-%d'); i=idx_at(T(ds,'09:30'))
    j=ix.searchsorted(T(ds,'15:59'),side='right')-1
    if prev is not None and i is not None: gaps.append((ds,o[i]-prev,i))
    prev=m1.close.values[j] if j>=0 and abs(ix[j]-T(ds,'15:59'))<pd.Timedelta(minutes=5) else None
def run(q,H,slip_ticks=1,stop_mult=None,fade=True,start=20):
    rows=[];past=[]
    for ds,g,i in gaps:
        thr=np.quantile(past,q) if (q>0 and len(past)>=start) else 0.0
        ok=(len(past)>=start) and abs(g)>=thr and g!=0
        past.append(abs(g))
        if not ok: continue
        sgn=-int(np.sign(g)) if fade else int(np.sign(g)); k=0.25*slip_ticks
        j=idx_at(T(ds,'09:30')+pd.Timedelta(minutes=H))
        if j is None: continue
        ent=o[i]+sgn*k; ex=o[j]-sgn*k
        # intratrade adverse excursion & optional stop (stop at stop_mult*|gap| from entry, filled at stop - slippage)
        seg_lo=lo[i:j+1]; seg_hi=hi[i:j+1]
        mae=(ent-seg_lo.min()) if sgn>0 else (seg_hi.max()-ent)
        pnl=sgn*(ex-ent)*2-4.0
        if stop_mult:
            sd=stop_mult*abs(g)
            hit=(mae>=sd)
            if hit: pnl=-sd*2-4.0-2*k*2   # stop filled at stop plus slippage
        rows.append(dict(day=ds,entry_time=ix[i],exit_time=ix[j],gap=g,pnl=pnl,mae=mae))
    return pd.DataFrame(rows)
def stats(df):
    if len(df)==0: return {}
    p=df.pnl.values; cut=pd.Timestamp('2026-01-01',tz='UTC'); tr=df[df.entry_time<cut].pnl; te=df[df.entry_time>=cut].pnl
    eq=np.cumsum(p); dd=float((np.maximum.accumulate(np.concatenate([[0],eq]))[1:]-eq).max())
    g=p[p>0].sum(); l=-p[p<0].sum(); t=p.mean()/(p.std(ddof=1)/np.sqrt(len(p)))
    mon=df.assign(m=df.entry_time.dt.strftime('%Y-%m')).groupby('m').pnl.sum()
    return dict(n=len(p),net=round(p.sum()),pf=round(g/l,2),win=round(100*(p>0).mean(),1),dd=round(dd),worst=round(p.min()),t=round(t,2),train=round(tr.sum()),n_tr=len(tr),y2026=round(te.sum()),n_26=len(te),green=f'{(mon>0).sum()}/{len(mon)}')
res=[]
for q in (0,0.5,0.75):
    for H in (30,60,90,120,180):
        s=stats(run(q,H)); s.update(q=q,H=H,slip=1,stop=None); res.append(s)
for sm in (1.0,1.5,2.0,3.0):
    s=stats(run(0.5,60,stop_mult=sm)); s.update(q=0.5,H=60,slip=1,stop=sm); res.append(s)
for st in (2,3,4):
    s=stats(run(0.5,60,slip_ticks=st)); s.update(q=0.5,H=60,slip=st,stop=None); res.append(s)
df=run(0.5,60); df.to_csv(os.path.join(R,'gapfade_q50_H60_trades.csv'),index=False)
# walk-forward style: quarterly net
df['q']=df.entry_time.dt.tz_convert(ET).dt.to_period('Q').astype(str)
res.append(dict(quarters=df.groupby('q').pnl.agg(['sum','count']).round(0).to_dict()))
# by weekday / by direction
df['dir']=np.where(df.gap>0,'short(gap up)','long(gap down)'); res.append(dict(by_dir=df.groupby('dir').pnl.agg(['sum','count']).round(0).to_dict()))
json.dump(res,open(os.path.join(R,'gapfade_robust.json'),'w'),indent=1,default=str)
for r in res: print(r)
