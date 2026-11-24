"""Shared helpers for the 3 new backtests (BACKTEST ONLY)."""
import sys,os
sys.argv=['x']
import sr_pub as P
from sr_pub import *
N=len(o); TD=pd.Timedelta(minutes=1)
def ok_dt(i,j,m): return j<N and ix[j]-ix[i]==m*TD
A0,A1,B0,B1='2024-11-01','2025-11-01','2025-11-01','2026-10-01'
def ny2(q=0.75,H=90,tk=1,dly=0,stop=300,lo=None,hi=None,qmin=None):
    """NY-open gap fade trades (1 contract) with gap metadata. past = expanding |gap| history (no lookahead)."""
    rows=[];past=[]
    for k,D in enumerate(days):
        if k==0: continue
        pc=sess[days[k-1]]['cl']; s=sess[D]; a=s['a']; g=s['op']-pc
        ok=len(past)>=20 and g!=0 and abs(g)>=np.quantile(past,q)
        if len(past)>=20: rank=float((np.array(past)<=abs(g)).mean()); t90=float(np.quantile(past,0.9)); t75=float(np.quantile(past,0.75)); t85=float(np.quantile(past,0.85)); t95=float(np.quantile(past,0.95))
        past.append(abs(g))
        if not ok: continue
        if lo and D<lo: continue
        if hi and D>=hi: continue
        side=-int(np.sign(g)); i_in=a+dly; e=a+H
        ent=o[i_in]+side*0.25*tk
        if stop: j,px,_=walk(i_in,e,side,ent-side*stop,None,tk)
        else: j,px=e,o[e]-side*0.25*tk
        r=mk(i_in,ent,j,px,side); r.update(d=D,g=g,absg=abs(g),rank=rank,t90=t90,t85=t85,t95=t95,t75=t75,i0=a,pc=pc); rows.append(r)
    return rows
def cb(df):
    a=df.groupby(df.exit_time.dt.tz_convert(ET).dt.date).pnl.sum(); b=P.base.groupby(P.base.exit_time.dt.tz_convert(ET).dt.date).pnl.sum()
    j=pd.concat([a,b],axis=1).fillna(0); return round(float(j.corr().iloc[0,1]),3)
def dayser(df): return df.groupby(df.exit_time.dt.tz_convert(ET).dt.date).pnl.sum()
def worst(df):
    d=dayser(df); p=df.pnl.values
    w=df.set_index(df.exit_time.dt.tz_convert(ET)).pnl.resample('W').sum()
    return dict(worst_trade=round(p.min()),worst_day=round(d.min()),worst_week=round(w.min()),dd=round(P.mdd(df.sort_values('exit_time').pnl.values)))
def pboot(delta_days,n=5000,seed=3):
    """paired day-bootstrap: P(sum of per-day delta >0)"""
    r=np.random.default_rng(seed); x=np.asarray(delta_days,float)
    if len(x)==0: return float('nan')
    return round(float((r.choice(x,(n,len(x))).sum(1)>0).mean()),3)