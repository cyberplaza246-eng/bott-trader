"""Addendum 11: holdout re-test of two night filters on the 3R baseline. BACKTEST ONLY. Writes x11_trades_*.csv only."""
import os,sys,time
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
import sr_x10_lib as L
BO,TR='ens_breakout','ens_trend'
F1=[dict(sess='allnight',tags=[TR],dir=-1)]
F2=[dict(sess='allnight',tags=[BO,TR],against='htf2880')]
SP={'base':{}, 'f1':dict(drops=F1), 'f2':dict(drops=F2), 'both':dict(drops=F1+F2)}
def job(it):
    name,extra=it; t0=time.time()
    tag,sl,win=name
    sp=dict(SP[tag]); sp['slip']=sl
    if win=='early': sp['start']='2024-09-30'; sp['end']='2024-11-01'
    try:
        df=L.sim(sp); df.to_csv(f'x11_trades_{tag}_s{int(sl*4)}_{win}.csv',index=False)
        return (name,len(df),round(float(df.pnl.sum()),1),round(time.time()-t0))
    except Exception:
        import traceback; return (name,traceback.format_exc()[-1500:])
if __name__=='__main__':
    import multiprocessing as mp
    items=[((t,s,w),None) for w in ('main','early') for s in (0.25,1.0) for t in SP]
    with mp.Pool(int(sys.argv[1]) if len(sys.argv)>1 else 4) as p:
        for r in p.imap_unordered(job,items): print(r,flush=True)
