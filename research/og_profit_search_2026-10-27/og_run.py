"""OG bot replay with fallback entries ON/OFF + variants. Unified cost model: comm_rt/contract RT + slip_ticks each side on entry & stops."""
import os, sys, json, glob, time, copy
sys.dont_write_bytecode=True
H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(H,'code'))
import sim2 as S
import pandas as pd, numpy as np
ET='America/New_York'
def coverage():
    ids=set()
    for f in glob.glob(os.path.join(H,'out','fb_*.jsonl')):
        for l in open(f,encoding='utf-8'):
            try: ids.add(json.loads(l)['i'])
            except Exception: pass
    return ids
def split_stats(df,col):
    if df is None or len(df)==0: return {}
    d=df.copy(); et=d.entry_t.dt.tz_convert(ET); d['month']=et.dt.strftime('%Y-%m')
    tr=d[d.entry_t<pd.Timestamp('2026-01-01',tz='UTC')]; te=d[d.entry_t>=pd.Timestamp('2026-01-01',tz='UTC')]
    mon=d.groupby('month')[col].sum()
    return dict(all=S.metrics(d,col),train=S.metrics(tr,col),y2026=S.metrics(te,col),months_green=int((mon>0).sum()),months=int(len(mon)),monthly=mon.round(0).to_dict())
def run(C,F,cfg):
    df,oe=S.simulate(C,F,cfg)
    if len(df)==0: return df,{}
    r={'c4':split_stats(df,'pnl_real'),'c224':split_stats(df,'pnl_224')}
    return df,r
if __name__=='__main__':
    specs=json.load(open(os.path.join(H,sys.argv[1])))
    outp=os.path.join(H,'out',sys.argv[2])
    t0=time.time(); C=S.load_signals(); F=S.load_fine(); print('signals',len(C),'fine',len(F),'t',round(time.time()-t0),flush=True)
    ts=pd.to_datetime([r['t'] for r in C],utc=True); print('signal span',ts.min(),ts.max(),flush=True)
    done=set()
    if os.path.exists(outp) and len(sys.argv)<4:
        for l in open(outp): done.add(json.loads(l)['name'])
    f=open(outp,'a',buffering=1)
    for name,sp in specs.items():
        if name in done: continue
        t1=time.time(); cfg=dict(sp); cfg.pop('_save',None)
        for k in ('bad_hours',):
            if k in cfg and isinstance(cfg[k],list): cfg[k]=tuple(cfg[k])
        df,r=run(C,F,cfg)
        f.write(json.dumps(dict(name=name,spec=sp,res=r,sec=round(time.time()-t1)),default=str)+'\n')
        if sp.get('_save') and len(df): df.to_csv(os.path.join(H,'out','og_trades_'+name+'.csv'),index=False)
        m=r.get('c4',{}).get('all',{}); print(name,m,round(time.time()-t1),flush=True)
