"""Strategy-research add-on / filter tests on the 3R revamp baseline (BACKTEST ONLY; nothing live is touched).
Reuses resistance_test/rt_engine.py (which reuses rv_engine + repo revamp harness). All filters use only data known at signal-bar close."""
import os, sys, json, time, math
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
R=os.path.dirname(os.path.abspath(__file__)); H=os.path.dirname(R)
sys.path.insert(0,H); sys.path.insert(0,os.path.join(H,'resistance_test'))
import rt_engine as RT
E=RT.E; B=RT.B
import numpy as np, pandas as pd
ET='America/New_York'

# ---- public event calendars (BLS 2025/2026 schedule pages; FOMC calendar federalreserve.gov). ET times.
NFP=['2024-11-01','2024-12-06','2025-01-10','2025-02-07','2025-03-07','2025-04-04','2025-05-02','2025-06-06','2025-07-03','2025-08-01','2025-09-05','2025-11-20','2025-12-16',
     '2026-01-09','2026-02-11','2026-03-06','2026-04-03','2026-05-08','2026-06-05','2026-07-02','2026-08-07','2026-09-04']
CPI=['2024-11-13','2024-12-11','2025-01-15','2025-02-12','2025-03-12','2025-04-10','2025-05-13','2025-06-11','2025-07-15','2025-08-12','2025-09-11','2025-10-24','2025-12-18',
     '2026-01-13','2026-02-13','2026-03-11','2026-04-10','2026-05-12','2026-06-10','2026-07-14','2026-08-12','2026-09-11']
FOMC=['2024-11-07','2024-12-18','2025-01-29','2025-03-19','2025-05-07','2025-06-18','2025-07-30','2025-09-17','2025-10-29','2025-12-10',
      '2026-01-28','2026-03-18','2026-04-29','2026-06-17','2026-07-29','2026-09-16']
def ev_utc(dates,hm):
    return pd.DatetimeIndex([pd.Timestamp(f'{d} {hm}',tz=ET).tz_convert('UTC') for d in dates]).sort_values()
EV_MACRO=ev_utc(NFP+CPI,'08:30').append(ev_utc(FOMC,'14:00')).sort_values()
EV_FOMC=ev_utc(FOMC,'14:00')
def blocked(bend,events,pre=30,post=60):
    """True if bar-end time falls in [event-pre, event+post] minutes"""
    ev=events.values; b=bend.values
    i=np.searchsorted(ev,b+np.timedelta64(pre,'m'),side='left')  # first event >= b+pre
    # event e blocks if e-pre <= b <= e+post  -> exists e in [b-post, b+pre]
    lo=np.searchsorted(ev,b-np.timedelta64(post,'m'),side='left'); hi=np.searchsorted(ev,b+np.timedelta64(pre,'m'),side='right')
    return hi>lo

_F=None
def features(m5):
    global _F
    if _F is not None: return _F
    idx=m5.index.tz_convert(ET); c=m5['close']
    f=pd.DataFrame(index=m5.index)
    atr=E.ens_atr(m5['high'],m5['low'],m5['close'],14)
    f['rel_atr']=(atr/atr.rolling(2016,min_periods=500).median()).values
    for sp in (576,2880):
        ema=c.ewm(span=sp,adjust=False).mean(); f[f'htf{sp}']=np.sign((c-ema).values)
    sess=pd.Series((idx-pd.Timedelta(hours=18)).normalize().values,index=m5.index)
    tp=(m5['high']+m5['low']+m5['close'])/3; v=m5['volume'].clip(lower=0)
    cv=(tp*v).groupby(sess).cumsum(); vv=v.groupby(sess).cumsum()
    for _n in (200,500): f[f'sma{_n}']=np.sign((c-c.rolling(_n).mean()).values)
    f['vwap']=(cv/vv.replace(0,np.nan)).values; f['vwap_sign']=np.sign((c-f['vwap']).values)
    mins=np.asarray(idx.hour*60+idx.minute); dts=pd.Series(idx.normalize().values,index=m5.index)
    o930=m5['open'].where(mins==570).groupby(dts).transform('max')
    f['o930_sign']=np.where((mins>=575)&(mins<960),np.sign((c-o930).values),0.0)
    _F=f; return f

DEF=dict(filt=None,max_risk=500.0,day_loss=500.0,ovn_loss=300.0,target=2000.0,max_open=3,stop_k=1.0,news=None,hours_drop=(),adds=True,dow_drop=(),consec=3,loss_cd=10)
def run_spec3(spec):
    d=E.load(); s=dict(E.DEFAULT); s.update(bo_r=3.0); sp=dict(DEF); sp.update(spec); s.update({k:v for k,v in spec.items() if k in s})
    key=(s['revamp'],s['bo_r'],s['trend_r'],s['mr_target'])
    if key not in d['ens_cache']: d['ens_cache'][key]=E.ens_signals(d['m5'],*key)
    ens=d['ens_cache'][key].copy(); m5=d['m5']
    ens['comp']=[E.comp_of(t,k,h) for t,k,h in zip(ens.tag,ens.kind,ens.et_hour)]
    f=features(m5).reindex(ens.index)
    keep=np.ones(len(ens),bool); sl=ens.sleeve.values; dr=ens.dir.values
    trendy=np.isin(sl,['breakout','trend'])
    flt=sp['filt']
    if flt:
        kind,arg=flt
        if kind=='rel_gt': keep&=~(f.rel_atr.values>arg)
        elif kind=='rel_lt': keep&=~(f.rel_atr.values<arg)
        elif kind in ('htf576','htf2880','vwap_sign','o930_sign','sma200','sma500'):
            sg=f[kind].values; ok=(sg==0)|(sg==dr)|np.isnan(sg)
            if arg=='bo': m=(sl=='breakout')
            else: m=trendy
            if kind=='o930_sign': m=m&(ens.kind.values=='day')
            keep&=~(m&~ok)
    if sp['dow_drop']:
        wd=(ens.index+pd.Timedelta(minutes=5)).tz_convert(ET).weekday.values; keep&=~np.isin(wd,list(sp['dow_drop']))
    if sp['hours_drop']: keep&=~ens.et_hour.isin(sp['hours_drop']).values
    ens=ens[keep]
    if sp['stop_k']!=1.0:
        c=m5['close'].reindex(ens.index).values; k=sp['stop_k']
        ens=ens.copy(); ens['sl']=c-ens.dir.values*k*np.abs(c-ens.sl.values)
        fin=np.isfinite(ens.tp.values); tpn=ens.tp.values.copy(); tpn[fin]=c[fin]+ens.dir.values[fin]*k*np.abs(ens.tp.values[fin]-c[fin]); ens['tp']=tpn
    ens['tp1']=np.nan; ens['lvl']=False
    adds=[v.copy() for v in d['adds'].values()] if sp['adds'] else []
    for a in adds: a['tp1']=np.nan; a['lvl']=False
    if sp['news'] is not None:
        evs={'all':EV_MACRO,'fomc':EV_FOMC}[sp['news']]
        ens=ens[~blocked(ens.index+pd.Timedelta(minutes=5),evs)]
        adds=[a[~blocked(a.index+pd.Timedelta(minutes=5),evs)] for a in adds]
    lk=E.LOCKS[s['lock']]; B.LOCK_ACTIVATE,B.LOCK_KEEP,B.LOCK_TRAIL_START,B.LOCK_KEEP_PCT=lk
    cols=['dir','sl','tp','sleeve','tag','lock','flat_only','max_bars','tp1','lvl']
    sig=pd.concat([ens[cols]]+[a[cols] for a in adds])
    B.COMM_RT=s['comm']; B.SLIP=s['slip']; B.TP_THROUGH=s['tp_through']
    B.CONSEC_MAX=sp['consec']; B.LOSS_CD_MIN=sp['loss_cd']
    B.MAX_RISK_USD=sp['max_risk']; B.DAY_LOSS=sp['day_loss']; B.OVN_LOSS=sp['ovn_loss']; B.DAILY_TARGET=sp['target']
    B.PERIODS[0]=(B.PERIODS[0][0],s['start'],B.PERIODS[0][2]); B.PERIODS[-1]=(B.PERIODS[-1][0],B.PERIODS[-1][1],s['end'])
    tr=RT.simulate_p(d['m1'],m5.index,sig,B.Rules(overnight_sleeves='all',max_open=sp['max_open'],ovn_extra_slip_ticks=s['ovn_slip']))
    df=pd.DataFrame(tr)
    lo,hi=pd.Timestamp(B.PERIODS[0][1],tz='UTC'),pd.Timestamp(B.PERIODS[-1][2],tz='UTC')
    df=df[(df.entry_time>=lo)&(df.entry_time<hi)]
    g=df.groupby('pid')
    pos=g.agg(entry_time=('entry_time','first'),exit_time=('exit_time','max'),pnl=('pnl','sum'),tag=('tag','first'),dir=('dir','first'),session=('session','first'),reason=('reason','last')).reset_index(drop=True)
    pos['month']=pos.entry_time.dt.tz_convert(ET).dt.strftime('%Y-%m')
    # risk $ at entry (stop distance x $2) for sizing studies
    sg=pd.concat([ens[['tag','sl']],*[a[['tag','sl']] for a in adds]]); sg['c']=m5['close'].reindex(sg.index).values; sg['risk']=(sg.c-sg.sl).abs()*2.0
    sg=sg.reset_index().rename(columns={sg.index.name or 'index':'t'}).sort_values('t')
    p2=pos.sort_values('entry_time').copy(); p2['k']=p2.entry_time-pd.Timedelta(minutes=1)
    mm=pd.merge_asof(p2,sg[['t','tag','risk']],left_on='k',right_on='t',by='tag',direction='backward',tolerance=pd.Timedelta(minutes=15))
    pos=mm.drop(columns=['k','t']).sort_values('exit_time').reset_index(drop=True)
    return pos

SPECS={
 'base':{},
 'vol_hi15':dict(filt=('rel_gt',1.5)),'vol_hi20':dict(filt=('rel_gt',2.0)),'vol_lo07':dict(filt=('rel_lt',0.7)),'vol_lo05':dict(filt=('rel_lt',0.5)),
 'htf576':dict(filt=('htf576','bt')),'htf2880':dict(filt=('htf2880','bt')),
 'vwap_bt':dict(filt=('vwap_sign','bt')),'vwap_bo':dict(filt=('vwap_sign','bo')),'rthopen':dict(filt=('o930_sign','bt')),
 'news_all':dict(news='all'),'news_fomc':dict(news='fomc'),
 'risk300':dict(max_risk=300.0),'risk200':dict(max_risk=200.0),
 'dl300':dict(day_loss=300.0,ovn_loss=200.0),'dl750':dict(day_loss=750.0,ovn_loss=450.0),'notarget':dict(target=0.0),
 'open1':dict(max_open=1),'open2':dict(max_open=2),
 'nofri':dict(dow_drop=(4,)),'nomon':dict(dow_drop=(0,)),'consec2':dict(consec=2),'losscd30':dict(loss_cd=30),
 'stop125':dict(stop_k=1.25),'stop080':dict(stop_k=0.8),
}
def job(item):
    name,sp=item; t0=time.time()
    try:
        df=run_spec3(sp); df.to_csv(os.path.join(R,'trades_'+name+'.csv'),index=False)
        return dict(name=name,spec=str(sp),res=E.summarize(df),n_risk=int(df.risk.notna().sum()),sec=round(time.time()-t0))
    except Exception:
        import traceback; return dict(name=name,err=traceback.format_exc()[-1800:])
def init_worker():
    try:
        import ctypes; ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    except Exception: pass
if __name__=='__main__':
    import multiprocessing as mp
    n=int(sys.argv[1]); only=sys.argv[2:]; S=dict(SPECS)
    extra=os.path.join(R,'extra_specs.json')
    if os.path.exists(extra):
        for k,v in json.load(open(extra)).items(): S[k]={kk:(tuple(vv) if isinstance(vv,list) else vv) for kk,vv in v.items()}
    if only: S={k:v for k,v in S.items() if k in only}
    outp=os.path.join(R,'results_runs.jsonl'); done=set()
    if os.path.exists(outp) and not only:
        for l in open(outp):
            try:
                r=json.loads(l)
                if 'res' in r: done.add(r['name'])
            except Exception: pass
    items=[(k,v) for k,v in S.items() if k not in done]; print(len(items),'to run',flush=True)
    init_worker()
    with mp.Pool(n,initializer=init_worker) as pool, open(outp,'a',buffering=1) as f:
        for r in pool.imap_unordered(job,items):
            f.write(json.dumps(r,default=str)+'\n'); print(r['name'],r.get('res',{}).get('all'),r.get('err','')[:1500],flush=True)
