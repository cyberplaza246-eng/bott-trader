"""Analysis of validation runs -> validation/REPORT_VALIDATION.md (+csv/json). Backtest only."""
import os, sys, json
import numpy as np, pandas as pd
H=os.path.dirname(os.path.abspath(__file__)); V=os.path.join(H,'validation'); ET='America/New_York'
def load(name):
    p=os.path.join(V,'trades_'+name+'.csv')
    if not os.path.exists(p): return None
    d=pd.read_csv(p); d['entry_time']=pd.to_datetime(d.entry_time,utc=True); d['exit_time']=pd.to_datetime(d.exit_time,utc=True)
    return d.sort_values('exit_time').reset_index(drop=True)
def ddinfo(p):
    eq=np.cumsum(p); pk=np.maximum.accumulate(np.concatenate([[0],eq]))[1:]; dd=pk-eq
    return float(dd.max()) if len(dd) else 0.0
def met(d):
    if d is None or len(d)==0: return dict(n=0,net=0.0,wr=0.0,pf=0.0,dd=0.0,avg=0.0)
    p=d.pnl.values; g=p[p>0].sum(); l=-p[p<0].sum()
    return dict(n=len(p),net=round(float(p.sum())),wr=round(100*float((p>0).mean()),1),pf=round(float(g/l),2) if l>0 else 99,dd=round(ddinfo(p)),avg=round(float(p.mean()),1))
def seg(d,a,b):
    lo,hi=pd.Timestamp(a,tz='UTC'),pd.Timestamp(b,tz='UTC')
    return d[(d.entry_time>=lo)&(d.entry_time<hi)]
def dd_duration(d):
    """longest time (days) from an equity peak to the next new peak (unrecovered counts to last trade) by exit time."""
    eq=d.pnl.cumsum().values; t=d.exit_time.values; peak=-1e18; pt=t[0]; longest=0; lst=None
    peak=0.0; pt=d.exit_time.iloc[0]
    for e,tt in zip(eq,d.exit_time):
        if e>peak+1e-9:
            dur=(tt-pt).days
            if dur>longest: longest=dur
            peak=e; pt=tt
    dur=(d.exit_time.iloc[-1]-pt).days
    return max(longest,dur), dur
def full_stats(d):
    m=met(d); et=d.exit_time.dt.tz_convert(ET)
    day=d.groupby(et.dt.date).pnl.sum(); wk=d.groupby(et.dt.strftime('%G-W%V')).pnl.sum(); mon=d.groupby(et.dt.strftime('%Y-%m')).pnl.sum(); yr=d.groupby(et.dt.year).pnl.sum()
    ld,cur=dd_duration(d)
    m.update(worst_day=round(float(day.min())),best_day=round(float(day.max())),worst_week=round(float(wk.min())),green_months=int((mon>0).sum()),months=int(len(mon)),worst_month=round(float(mon.min())),
             longest_dd_days=ld,current_dd_days=cur,yearly={int(k):round(float(v)) for k,v in yr.items()},monthly={k:round(float(v)) for k,v in mon.items()},
             green_weeks=f"{int((wk>0).sum())}/{len(wk)}",green_days=f"{int((day>0).sum())}/{len(day)}")
    return m
def folds():
    s=pd.Timestamp('2024-11-01',tz='UTC'); out=[]
    while True:
        tr0=s; tr1=s+pd.DateOffset(months=6); te1=tr1+pd.DateOffset(months=3)
        if tr1>=pd.Timestamp('2026-09-22',tz='UTC'): break
        out.append((tr0,tr1,min(te1,pd.Timestamp('2026-10-01',tz='UTC')))); s=s+pd.DateOffset(months=3)
    return out
def sl(d,a,b): return d[(d.entry_time>=a)&(d.entry_time<b)]
def score(m): return m['net']-0.75*m['dd']
def walk(names,fixed):
    F=folds(); rows=[]; tot={k:0 for k in fixed}; tot['reselect']=0; picks=[]
    D={n:load(n) for n in names if load(n) is not None}
    for (a,b,c) in F:
        r=dict(train=f"{a.date()}..{b.date()}",test=f"{b.date()}..{c.date()}")
        for k in fixed:
            r[k]=met(sl(D[k],b,c)); tot[k]+=r[k]['net']
        tm={n:met(sl(D[n],a,b)) for n in D}; best=max(tm,key=lambda n:score(tm[n]))
        r['pick']=best; r['pick_train']=tm[best]['net']; r['pick_oos']=met(sl(D[best],b,c)); tot['reselect']+=r['pick_oos']['net']; rows.append(r)
    return rows,tot
def bootstrap(d,N=500,seed=7):
    rng=np.random.default_rng(seed); p=d.pnl.values; n=len(p); res=[]
    for _ in range(N):
        keep=rng.random(n)>0.10; q=p[keep]; res.append((q.sum(),ddinfo(q),int(keep.sum())))
    a=np.array(res)
    # block bootstrap by ET exit day (resample days with replacement, same number of days)
    et=d.exit_time.dt.tz_convert(ET).dt.date; days=d.groupby(et).pnl.apply(lambda x:x.values); days=list(days.values); nd=len(days); b=[]
    for _ in range(N):
        idx=rng.integers(0,nd,nd); q=np.concatenate([days[i] for i in idx]); b.append((q.sum(),ddinfo(q)))
    b=np.array(b)
    f=lambda x,q:round(float(np.percentile(x,q)))
    return dict(remove10=dict(net_p5=f(a[:,0],5),net_p50=f(a[:,0],50),net_p95=f(a[:,0],95),dd_p50=f(a[:,1],50),dd_p95=f(a[:,1],95),dd_max=round(float(a[:,1].max())),p_loss=float((a[:,0]<0).mean())),
                dayblock=dict(net_p5=f(b[:,0],5),net_p50=f(b[:,0],50),net_p95=f(b[:,0],95),dd_p50=f(b[:,1],50),dd_p95=f(b[:,1],95),dd_max=round(float(b[:,1].max())),p_loss=float((b[:,0]<0).mean())))
def main():
    R=[json.loads(l) for l in open(os.path.join(V,'results_runs.jsonl'))]; R={r['name']:r for r in R if 'res' in r}
    out={}; L=[]
    cands=[c for c in ('A','D','D2','base') if load(c) is not None]
    for c in cands: out[c]=full_stats(load(c))
    names=[n for n in R if not n.startswith(('s_','x_'))]
    fixed=[c for c in ('A','D','base') if c in names]
    wf,tot=walk(names,fixed); out['walk']=dict(rows=wf,totals=tot,grid=names)
    bs={c:bootstrap(load(c)) for c in cands if c!='base'}; out['bootstrap']=bs
    st={n:R[n]['res'] for n in R if n.startswith(('s_','x_'))}; out['stress']=st
    json.dump(out,open(os.path.join(V,'validation_results.json'),'w'),indent=1,default=str)
    for c in cands:
        d=load(c); et=d.entry_time.dt.tz_convert(ET); pd.DataFrame({'month':list(out[c]['monthly'].keys()),'pnl':list(out[c]['monthly'].values())}).to_csv(os.path.join(V,f'monthly_{c}.csv'),index=False)
    # ---- markdown
    def row(m): return f"{m['net']:+,} | {m['dd']:,} | {m['pf']} | {m['n']} | {m['wr']}% | {m['avg']}"
    M=[]; w=M.append
    w('# REPORT_VALIDATION (backtest only)\n')
    w('Cost model: $4 round trip per contract + 1 tick slippage each side (stops slip 1 tick; overnight same), 1 MNQ, next-1m-open + 1 tick fills, stop-first on same-bar touches, max 3 open, session loss limits. Data MNQ_1m cleaned of roll spikes; trades Nov-2024 .. 2026-09-21. Candidates: **A** = revamp stack + REVAMP_BREAKOUT_TARGET_R=3.0; **D** = A with overnight sleeves dropped (day-only; ONL + 6PM add-ons kept); **D2** = D without 6PM add-on; base = current revamp live (2.0R).\n')
    w('## 1. Headline (all data)\n'); w('| run | net | maxDD | PF | trades | win | avg/trade |'); w('|---|---|---|---|---|---|---|')
    for c in cands: w(f'| {c} | '+row(out[c])+' |')
    w('\n## 2. Yearly / monthly / drawdown shape\n'); w('| run | 2024 | 2025 | 2026 | green months | worst month | worst day | worst week | longest DD (days) | still in DD (days) | green weeks |'); w('|---|---|---|---|---|---|---|---|---|---|---|')
    for c in cands:
        y=out[c]['yearly']; w(f"| {c} | {y.get(2024,0):+,} | {y.get(2025,0):+,} | {y.get(2026,0):+,} | {out[c]['green_months']}/{out[c]['months']} | {out[c]['worst_month']:+,} | {out[c]['worst_day']:+,} | {out[c]['worst_week']:+,} | {out[c]['longest_dd_days']} | {out[c]['current_dd_days']} | {out[c]['green_weeks']} |")
    w('\nMonthly PnL (ET, by exit):\n'); allm=sorted(set(k for c in cands for k in out[c]['monthly'])); w('| month | '+' | '.join(cands)+' |'); w('|---|'+'---|'*len(cands))
    for m in allm: w(f'| {m} | '+' | '.join(f"{out[c]['monthly'].get(m,0):+,}" for c in cands)+' |')
    w('\n## 3. Walk-forward (6-month train / 3-month out-of-sample test, rolled every 3 months)\n')
    w('Fixed params = A, D, base run once over full history and sliced by entry date (signals have no fitted parameters, so the only look-ahead is the choice of params). Re-select = on each train window pick the best of the grid ('+', '.join(names)+f') by net - 0.75*maxDD, then score it on the next 3 months.\n')
    w('| train window | test window | A OOS net (DD) | D OOS net (DD) | base OOS net (DD) | re-selected pick (train net) | pick OOS net (DD) |'); w('|---|---|---|---|---|---|---|')
    for r in wf: w(f"| {r['train']} | {r['test']} | "+' | '.join(f"{r[k]['net']:+,} ({r[k]['dd']:,})" if k in r else '-' for k in ('A','D','base'))+f" | {r['pick']} ({r['pick_train']:+,}) | {r['pick_oos']['net']:+,} ({r['pick_oos']['dd']:,}) |")
    w('\nTotal OOS net: '+', '.join(f'{k}: {v:+,.0f}' for k,v in tot.items())+f"  | OOS folds positive: "+', '.join(f"{k} {sum(1 for r in wf if (r[k]['net'] if k!='reselect' else r['pick_oos']['net'])>0)}/{len(wf)}" for k in list(fixed)+['reselect'])+'\n')
    w('## 4. Stress tests\n'); w('| spec | net | maxDD | PF | trades | train net | 2026 net |'); w('|---|---|---|---|---|---|---|')
    for c in ('A','D'):
        if c in out: w(f"| {c} (reference) | {out[c]['net']:+,} | {out[c]['dd']:,} | {out[c]['pf']} | {out[c]['n']} | | |")
        for n,s in st.items():
            if n.startswith((f's_{c}_',f'x_{c}_')): w(f"| {n} | {s['all']['net']:+,} | {s['all']['dd']:,} | {s['all']['pf']} | {s['all']['n']} | {s['train']['net']:+,} | {s['y2026']['net']:+,} |")
    w('\nslip2/slip3 = 2x/3x slippage; x_*_early = same params but starting 2024-09-25 (all data); tpthru2/tpthru4 = TP needs 2/4 ticks through; tpthru = take-profit fills only if price trades 1 tick THROUGH the limit; ovnslip2 = +2 ticks extra on overnight fills; bo24/bo36 = breakout target 3.0R -/+20%; trend12/18 = trend target 1.5R -/+20%; lock_m20/p20 = $ profit-lock thresholds -/+20%; comm6 = $6 round trip. The 4.0 ATR trail is applied inside the strategy engine and was not re-run at +/-20% (see caveats).\n')
    w('## 5. Bootstrap (500 runs) and account sizing\n'); w('| run | method | net p5 | net p50 | net p95 | maxDD p50 | maxDD p95 | maxDD worst | P(net<0) |'); w('|---|---|---|---|---|---|---|---|---|')
    for c,b in bs.items():
        for k,lab in (('remove10','random 10% of trades removed'),('dayblock','resample trading days with replacement')):
            x=b[k]; w(f"| {c} | {lab} | {x['net_p5']:+,} | {x['net_p50']:+,} | {x['net_p95']:+,} | {x['dd_p50']:,} | {x['dd_p95']:,} | {x['dd_max']:,} | {x['p_loss']:.1%} |")
    w('\nAccount sizing for 1 MNQ (cushion = 1.25 x the larger of observed max DD and bootstrap p95 DD, rounded up to $500):\n'); w('| run | observed maxDD | worst bootstrap p95 DD | suggested min account | equity at bootstrap net p5 (from that start) |'); w('|---|---|---|---|---|')
    for c,b in bs.items():
        dd95=max(b['remove10']['dd_p95'],b['dayblock']['dd_p95']); need=int(np.ceil(1.25*max(out[c]['dd'],dd95)/500)*500); p5=min(b['remove10']['net_p5'],b['dayblock']['net_p5'])
        w(f"| {c} | {out[c]['dd']:,} | {dd95:,} | ${need:,} | ${need+p5:,} (p5 net {p5:+,}) |")
    open(os.path.join(V,'REPORT_BODY.md'),'w',encoding='utf-8').write('\n'.join(M)); print('\n'.join(M))
if __name__=='__main__': main()

