"""Validation runs: simulate specs with the revamp harness, save per-trade CSVs into validation/. Backtest only."""
import os, sys, json, time
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
import multiprocessing as mp
H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,H)
import rv_engine as E
V=os.path.join(H,'validation')
E.LOCKS['base_m20']=(80.,16.,120.,.5); E.LOCKS['base_p20']=(120.,24.,180.,.5)
DAY=['trend_ovn','bo_ovn_0009','bo_ovn_other','mr_ovn']
def specs():
    S={}
    # candidate A: revamp stack, breakout target 3R. candidate D: day-only (overnight sleeves dropped, ONL+E1 kept) + 3R. D2: also drop E1 (6PM long)
    S['base']={}
    S['A']=dict(bo_r=3.0)
    S['D']=dict(bo_r=3.0,drop=DAY)
    S['D2']=dict(bo_r=3.0,drop=DAY+['e1'])
    # re-selection grid
    S['g_A_notrend']=dict(bo_r=3.0,drop=['trend_day'])
    S['g_A_norevamp']=dict(revamp=False)
    S['g_A_no0009']=dict(bo_r=3.0,drop=['bo_ovn_0009'])
    S['g_A_tight2']=dict(bo_r=3.0,lock='tight2')
    S['g_A_wide']=dict(bo_r=3.0,lock='wide')
    S['g_D_notrend']=dict(bo_r=3.0,drop=DAY+['trend_day'])
    S['g_base_notrend']=dict(drop=['trend_day'])
    S['g_bo25']=dict(bo_r=2.5)
    # stress
    for tag,sp in (('A',S['A']),('D',S['D'])):
        S[f's_{tag}_slip2']=dict(sp,slip=0.5)
        S[f's_{tag}_slip3']=dict(sp,slip=0.75)
        S[f's_{tag}_tpthru']=dict(sp,tp_through=0.25)
        S[f's_{tag}_tpthru_slip2']=dict(sp,tp_through=0.25,slip=0.5)
        S[f's_{tag}_ovnslip2']=dict(sp,ovn_slip=2)
        S[f's_{tag}_bo24']=dict(sp,bo_r=2.4)
        S[f's_{tag}_bo36']=dict(sp,bo_r=3.6)
        S[f's_{tag}_trend12']=dict(sp,trend_r=1.2)
        S[f's_{tag}_trend18']=dict(sp,trend_r=1.8)
        S[f's_{tag}_lock_m20']=dict(sp,lock='base_m20')
        S[f's_{tag}_lock_p20']=dict(sp,lock='base_p20')
        S[f's_{tag}_comm6']=dict(sp,comm=6.0)
    return S
def job(item):
    name,sp=item; t0=time.time()
    try:
        sp={k:(tuple(v) if isinstance(v,list) else v) for k,v in sp.items()}
        tr=E.run_spec(sp); df=E.trades_df(tr)
        df.to_csv(os.path.join(V,'trades_'+name+'.csv'),index=False)
        r=dict(name=name,spec={k:(list(v) if isinstance(v,tuple) else v) for k,v in sp.items()},res=E.summarize(df),sec=round(time.time()-t0))
    except Exception:
        import traceback; r=dict(name=name,err=traceback.format_exc()[-800:])
    return r
if __name__=='__main__':
    n=int(sys.argv[1]); S=specs(); outp=os.path.join(V,'results_runs.jsonl'); done=set()
    if os.path.exists(outp):
        for l in open(outp):
            try: done.add(json.loads(l)['name'])
            except Exception: pass
    items=[(k,v) for k,v in S.items() if k not in done]
    print(len(items),'to run',flush=True)
    with mp.Pool(n) as pool, open(outp,'a',buffering=1) as f:
        for r in pool.imap_unordered(job,items):
            f.write(json.dumps(r,default=str)+'\n'); f.flush()
            print(r['name'],r.get('res',{}).get('all'),r.get('err','')[:300],flush=True)
            open(os.path.join(V,'progress.txt'),'a').write(f"{time.strftime('%H:%M:%S')} {r['name']}\n")
