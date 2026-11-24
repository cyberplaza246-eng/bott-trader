import os,sys,json,time
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
import sr_x10_lib as L
from x10_variants import V
import pandas as pd
def init_worker():
    try:
        import ctypes; ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    except Exception: pass
def job(item):
    name,spec,tag=item; t0=time.time()
    try:
        df=L.sim(spec); df.to_csv(f'x10_trades_{name}{tag}.csv',index=False)
        return dict(name=name+tag,n=len(df),net=round(float(df.pnl.sum()),1),sec=round(time.time()-t0))
    except Exception:
        import traceback; return dict(name=name+tag,err=traceback.format_exc()[-1500:])
if __name__=='__main__':
    import multiprocessing as mp
    n=int(sys.argv[1]); mode=sys.argv[2]; only=sys.argv[3:]
    items=[]
    if mode=='main':
        for k,(g,sp,d) in V.items():
            if only and k not in only: continue
            if os.path.exists(f'x10_trades_{k}.csv') and not only: continue
            items.append((k,sp,''))
    else:  # stress: mode like slip3 / slip4
        sl={'slip3':0.75,'slip4':1.0}[mode]
        for k in only:
            sp=dict(V[k][1]); sp['slip']=sl; items.append((k,sp,'__'+mode))
    print(len(items),'to run',flush=True); init_worker()
    with mp.Pool(n,initializer=init_worker) as pool:
        for r in pool.imap_unordered(job,items): print(r,flush=True)
