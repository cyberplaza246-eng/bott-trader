"""Addendum 10 library: night-session rule re-simulation of the 3R baseline. BACKTEST ONLY (no repo/live files touched).
Patches an in-memory copy of sr_engine.run_spec3 so a spec can carry 'drops' (list of clauses) and 'extra' (extra signal frames)."""
import os,sys,inspect,json,time
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
import sr_engine as S
import numpy as np, pandas as pd
E=S.E; B=S.B; ET=S.ET
src=inspect.getsource(S.run_spec3).replace('def run_spec3','def run_spec10')
k2="    lk=E.LOCKS[s['lock']]"; assert k2 in src
hook="""    if sp.get('drops') or sp.get('extra'):
        ens,adds=X10_APPLY(sp,ens,adds,m5)
"""
src=src.replace(k2,hook+k2,1)
S.X10_APPLY=None
exec(src,S.__dict__)

SESS={'day':(570,960),'evening':(960,1080),'night':(1080,120),'overnight':(120,570),'allnight':(1080,570)}  # [a,b) ET minutes, wrap if a>b
def inwin(m,a,b):
    return (m>=a)&(m<b) if a<b else (m>=a)|(m<b)
_F={}
def feat(m5):
    if 'f' not in _F: _F['f']=S.features(m5)
    return _F['f']
def mask_clause(cl,df,m5):
    bend=(df.index+pd.Timedelta(minutes=5)).tz_convert(ET); m=np.asarray(bend.hour*60+bend.minute)
    if 'sess' in cl: a,b=SESS[cl['sess']]
    else: a,b=cl['win']
    k=inwin(m,a,b)
    if cl.get('tags'): k&=np.isin(df.tag.values,cl['tags'])
    if cl.get('dir') is not None: k&=(df.dir.values==cl['dir'])
    if cl.get('against'):   # drop when trend-filter sign disagrees with trade direction (sign 0/nan = keep)
        f=feat(m5).reindex(df.index); sg=f[cl['against']].values; ok=(sg==0)|(sg==df.dir.values)|np.isnan(sg)
        k&=~ok
    return k
def apply10(sp,ens,adds,m5):
    drops=sp.get('drops') or []
    def f(df):
        if len(df)==0: return df
        keep=np.ones(len(df),bool)
        for cl in drops: keep&=~mask_clause(cl,df,m5)
        return df[keep]
    ens=f(ens); adds=[f(a) for a in adds]
    for nm in (sp.get('extra') or []):
        x=pd.read_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)),'x10_'+nm+'.pkl')).copy(); x['tp1']=np.nan; x['lvl']=False; adds=adds+[x]
    return ens,adds
S.X10_APPLY=apply10
def sim(spec):
    sp=dict(spec); return S.run_spec10(sp)

def mdd(p):
    if len(p)==0: return 0.0
    eq=np.cumsum(p); return float((np.maximum.accumulate(np.concatenate([[0],eq]))[1:]-eq).max())
def met(df):
    if len(df)==0: return dict(n=0,net=0,pf=0.0,avg=0.0,dd=0,wr=0)
    p=df.sort_values('exit_time').pnl.values; g=p[p>0].sum(); l=-p[p<0].sum()
    return dict(n=int(len(p)),net=round(float(p.sum())),pf=round(float(g/l),2) if l>0 else 99.0,avg=round(float(p.mean()),1),dd=round(mdd(p)),wr=round(100*float((p>0).mean())))


