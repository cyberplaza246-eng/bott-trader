"""Analysis of resistance-target variants vs fixed 3R baseline -> REPORT_TABLE.md / analysis.json (backtest only)."""
import os, sys, json
import numpy as np, pandas as pd
R=os.path.dirname(os.path.abspath(__file__)); ET='America/New_York'
ORDER=['base','swing20','swing50','swing100','day','round50','round100','part_swing20','part_swing50','part_swing100','part_day','part_round50','part_round100']
def load(n):
    p=os.path.join(R,f'trades_{n}.csv')
    if not os.path.exists(p): return None
    d=pd.read_csv(p); d['entry_time']=pd.to_datetime(d.entry_time,utc=True); d['exit_time']=pd.to_datetime(d.exit_time,utc=True)
    return d.sort_values('exit_time').reset_index(drop=True)
def dd(p):
    eq=np.cumsum(p); return float((np.maximum.accumulate(np.concatenate([[0],eq]))[1:]-eq).max()) if len(p) else 0.0
def met(d):
    if d is None or len(d)==0: return dict(n=0,net=0,pf=0,wr=0,dd=0)
    p=d.pnl.values; g=p[p>0].sum(); l=-p[p<0].sum()
    return dict(n=len(p),net=round(float(p.sum())),pf=round(float(g/l),2) if l>0 else 99,wr=round(100*float((p>0).mean()),1),dd=round(dd(p)))
def sl(d,a,b): return d[(d.entry_time>=a)&(d.entry_time<b)]
def folds():
    s=pd.Timestamp('2024-11-01',tz='UTC'); out=[]
    while True:
        tr1=s+pd.DateOffset(months=6); te1=tr1+pd.DateOffset(months=3)
        if tr1>=pd.Timestamp('2026-09-22',tz='UTC'): break
        out.append((tr1,min(te1,pd.Timestamp('2026-10-01',tz='UTC')))); s=s+pd.DateOffset(months=3)
    return out
def daily(d):
    return d.groupby(d.exit_time.dt.tz_convert(ET).dt.date).pnl.sum()
def paired_boot(b,v,N=2000,seed=11):
    a=daily(b); c=daily(v); days=a.index.union(c.index); x=(c.reindex(days).fillna(0)-a.reindex(days).fillna(0)).values
    rng=np.random.default_rng(seed); n=len(x); s=np.array([x[rng.integers(0,n,n)].sum() for _ in range(N)])
    return dict(p5=round(float(np.percentile(s,5))),p50=round(float(np.percentile(s,50))),p95=round(float(np.percentile(s,95))),p_gain=round(float((s>0).mean()),3))
def main():
    D={n:load(n) for n in ORDER if load(n) is not None}; base=D['base']
    T26=pd.Timestamp('2026-01-01',tz='UTC'); F=folds(); rows=[]; out={}
    bb=base[base.tag=='ens_breakout']; bmet=met(bb)
    for n,d in D.items():
        m=met(d); tr=met(d[d.entry_time<T26]); te=met(d[d.entry_time>=T26])
        mon=d.groupby('month').pnl.sum(); bmon=base.groupby('month').pnl.sum().reindex(mon.index).fillna(0)
        bo=d[d.tag=='ens_breakout']; bm=met(bo)
        wf=[met(sl(d,a,b))['net'] for a,b in F]; wfb=[met(sl(base,a,b))['net'] for a,b in F]
        lvtr=int(bo.lvl.sum()) if 'lvl' in bo else 0
        r=dict(name=n,**m,green=f'{int((mon>0).sum())}/{len(mon)}',train=tr['net'],y2026=te['net'],d_total=m['net']-met(base)['net'],d_train=tr['net']-met(base[base.entry_time<T26])['net'],
               d_2026=te['net']-met(base[base.entry_time>=T26])['net'],bo_n=bm['n'],bo_net=bm['net'],d_bo=bm['net']-bmet['net'],bo_lvl_trades=lvtr,bo_tp_pct=round(100*float((bo.reason=='TAKE_PROFIT').mean()),1),
               tp1_hit=int(bo.tp1_hit.sum()) if 'tp1_hit' in bo else 0,
               wf_oos=int(sum(wf)),wf_d=int(sum(wf)-sum(wfb)),wf_folds_better=f'{sum(1 for x,y in zip(wf,wfb) if x>y)}/{len(F)}',
               months_better=f'{int((mon>bmon).sum())}/{len(mon)}')
        if n!='base': r['boot']=paired_boot(base,d)
        rows.append(r); out[n]=r
    json.dump(out,open(os.path.join(R,'analysis.json'),'w'),indent=1,default=str)
    L=['| variant | trades | net $ | PF | win% | maxDD | green mo | train (to 2025) | 2026 | d vs 3R (all) | d train | d 2026 | WF OOS d (folds better) | BO-sleeve trades w/ level | BO sleeve net (d) | paired day-boot p5..p95 d / P(gain) |','|'+'---|'*16]
    for r in rows:
        bt=r.get('boot'); bs=f"{bt['p5']:+d}..{bt['p95']:+d} / {bt['p_gain']}" if bt else '-'
        L.append(f"| {r['name']} | {r['n']} | {r['net']:+,} | {r['pf']} | {r['wr']} | {r['dd']:,} | {r['green']} | {r['train']:+,} | {r['y2026']:+,} | {r['d_total']:+,} | {r['d_train']:+,} | {r['d_2026']:+,} | {r['wf_d']:+,} ({r['wf_folds_better']}) | {r['bo_lvl_trades']}/{r['bo_n']} | {r['bo_net']:+,} ({r['d_bo']:+,}) | {bs} |")
    L.append(''); L.append('Walk-forward folds (OOS test windows, entry date): '+', '.join(f'{a.date()}..{b.date()}' for a,b in F))
    L.append(''); L.append('| variant | '+' | '.join(f'fold{i+1}' for i in range(len(F)))+' |'); L.append('|'+'---|'*(len(F)+1))
    for n,d in D.items(): L.append(f'| {n} | '+' | '.join(f"{met(sl(d,a,b))['net']:+,}" for a,b in F)+' |')
    open(os.path.join(R,'REPORT_TABLE.md'),'w',encoding='utf-8').write('\n'.join(L)); print('\n'.join(L))
if __name__=='__main__': main()