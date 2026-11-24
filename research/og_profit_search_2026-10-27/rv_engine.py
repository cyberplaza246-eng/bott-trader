"""Revamp variant search engine (backtest only). Spec -> trades -> stats. Uses the repo's own revamp harness (copied to code/)."""
import os, sys, json, pickle, time, math
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(H,'code'))
import backtest_revamp_live_stack as B
import pandas as pd, numpy as np
from src.research_ensemble.ensemble import EnsembleStrategy
from src.research_ensemble.indicators import adx, atr as ens_atr
ET='America/New_York'
D=None
def load():
    global D
    if D is None:
        D=pickle.load(open(os.path.join(H,'out','revamp_base.pkl'),'rb'))
        D['ens_cache']={}
    return D
def ens_signals(m5, revamp=True, bo_r=2.0, trend_r=1.5, mr_target='bb_far'):
    strat=EnsembleStrategy(rth_only=False, revamp_exits=revamp, breakout_target_r=bo_r, mr_target=mr_target)
    sig=strat.generate_signals(m5)
    a=adx(m5['high'],m5['low'],m5['close'],14)
    bo=strat._breakout.generate_signals(m5).entries
    ranging=a<strat.adx_range_max; trending=a>=strat.adx_trend_min
    bo_fires=(bo!=0)&~ranging
    sleeve=np.where(bo_fires,'breakout',np.where(trending,'trend',np.where(ranging,'mean_reversion','trend')))
    out=pd.DataFrame({'dir':sig.entries.astype(int),'sl':sig.stop_price,'tp':sig.target_price,'sleeve':sleeve,'adx':a,
                      'atr':ens_atr(m5['high'],m5['low'],m5['close'],14)},index=m5.index)
    c=m5['close']; trend=(out.sleeve=='trend')&(out.dir!=0)
    out.loc[trend,'tp']=c[trend]+out.dir[trend]*trend_r*(c[trend]-out.sl[trend]).abs()
    out=out[(out.dir!=0)&np.isfinite(out.sl)].copy()
    out['tag']='ens_'+out['sleeve']; out['lock']=True; out['flat_only']=False; out['max_bars']=0
    bend=out.index+pd.Timedelta(minutes=5)
    kinds=[B.session_of(t)[0] for t in bend]
    out['kind']=kinds
    out['et_hour']=bend.tz_convert(ET).hour
    return out
COMPONENTS=['trend_day','trend_ovn','bo_day','bo_ovn_0009','bo_ovn_other','mr_day','mr_ovn','onl','e1']
def comp_of(row_tag,kind,hr):
    if row_tag=='addon_onl': return 'onl'
    if row_tag=='addon_e1': return 'e1'
    sl=row_tag.replace('ens_','')
    if sl=='trend': return 'trend_day' if kind=='day' else 'trend_ovn'
    if sl=='mean_reversion': return 'mr_day' if kind=='day' else 'mr_ovn'
    if sl=='breakout':
        if kind=='day': return 'bo_day'
        return 'bo_ovn_0009' if 0<=hr<9 else 'bo_ovn_other'
    return 'other'
DEFAULT=dict(drop=(),max_open=3,lock='base',trend_r=1.5,bo_r=2.0,revamp=True,mr_target='bb_far',slip=0.25,comm=4.0,start='2024-11-01',end='2026-10-01',
             hours_drop=(),only_dir=None,tp_through=0.0,ovn_slip=0)
LOCKS={'base':(100.,20.,150.,.5),'tight':(60.,15.,100.,.5),'tight2':(80.,20.,120.,.6),'wide':(150.,30.,250.,.5),'off':None}
def run_spec(spec):
    d=load(); s=dict(DEFAULT); s.update(spec)
    key=(s['revamp'],s['bo_r'],s['trend_r'],s['mr_target'])
    if key not in d['ens_cache']: d['ens_cache'][key]=ens_signals(d['m5'],*key)
    ens=d['ens_cache'][key].copy()
    comp=[comp_of(t,k,h) for t,k,h in zip(ens.tag,ens.kind,ens.et_hour)]
    ens['comp']=comp
    ens=ens[~ens.comp.isin(s['drop'])]
    if s['hours_drop']:  # extra ET entry hours to drop for main signals
        ens=ens[~ens.et_hour.isin(s['hours_drop'])]
    if s['only_dir'] is not None: ens=ens[ens.dir==s['only_dir']]
    lk=LOCKS[s['lock']]
    if lk is None: ens['lock']=False
    else: B.LOCK_ACTIVATE,B.LOCK_KEEP,B.LOCK_TRAIL_START,B.LOCK_KEEP_PCT=lk
    adds=[v for k,v in d['adds'].items() if ({'addon_onl':'onl','addon_e1':'e1'}[k]) not in s['drop']]
    cols=['dir','sl','tp','sleeve','tag','lock','flat_only','max_bars']
    sig=pd.concat([ens[cols]]+[a[cols] for a in adds])
    B.COMM_RT=s['comm']; B.SLIP=s['slip']; B.TP_THROUGH=s['tp_through']
    B.PERIODS[0]=(B.PERIODS[0][0],s['start'],B.PERIODS[0][2]); B.PERIODS[-1]=(B.PERIODS[-1][0],B.PERIODS[-1][1],s['end'])
    tr=B.in_window(B.simulate(d['m1'],d['m5'].index,sig,B.Rules(overnight_sleeves='all',max_open=s['max_open'],ovn_extra_slip_ticks=s['ovn_slip'])))
    return tr
def trades_df(tr):
    df=pd.DataFrame([t.__dict__ for t in tr])
    if len(df)==0: return df
    df['month']=df.entry_time.dt.tz_convert(ET).dt.strftime('%Y-%m')
    return df
def met(df,col='pnl'):
    if df is None or len(df)==0: return dict(n=0,net=0.0,wr=0.0,pf=0.0,dd=0.0,avg=0.0)
    d=df.sort_values('exit_time'); p=d[col].values
    g=p[p>0].sum(); l=-p[p<0].sum(); eq=np.cumsum(p); dd=float((np.maximum.accumulate(np.concatenate([[0],eq]))[1:]-eq).max())
    return dict(n=int(len(p)),net=round(float(p.sum()),1),wr=round(100*float((p>0).mean()),1),pf=round(float(g/l),2) if l>0 else 99.0,dd=round(dd,1),avg=round(float(p.mean()),2))
def summarize(df,comm_adj=0.0):
    df=df.copy(); df['pnl']=df.pnl+comm_adj
    tr_=df[df.entry_time<pd.Timestamp('2026-01-01',tz='UTC')]; te=df[df.entry_time>=pd.Timestamp('2026-01-01',tz='UTC')]
    mon=df.groupby('month').pnl.sum()
    per={}
    for nm,a,b in B.PERIODS:
        lo,hi=pd.Timestamp(a,tz='UTC'),pd.Timestamp(b,tz='UTC'); per[nm.split()[0]]=round(float(df[(df.entry_time>=lo)&(df.entry_time<hi)].pnl.sum()),0)
    return dict(all=met(df),train=met(tr_),y2026=met(te),per=per,months_green=int((mon>0).sum()),months=int(len(mon)),worst_month=round(float(mon.min()),0))
def job(item):
    name,spec=item
    t0=time.time()
    try:
        tr=run_spec(spec); df=trades_df(tr)
        res=dict(name=name,spec={k:(list(v) if isinstance(v,tuple) else v) for k,v in spec.items()},res=summarize(df),sec=round(time.time()-t0))
        res['res224']=summarize(df,comm_adj=4.0-2.24)
        mon=df.groupby('month').pnl.sum().round(0).to_dict(); res['monthly']=mon
        df.to_csv(os.path.join(H,'out','trades_'+name.replace('/','_')+'.csv'),index=False) if spec.get('_save') else None
    except Exception as e:
        import traceback; res=dict(name=name,spec=str(spec),err=traceback.format_exc()[-800:])
    return res

