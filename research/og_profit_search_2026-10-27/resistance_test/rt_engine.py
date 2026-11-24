"""Resistance/support take-profit test for the revamp BREAKOUT sleeve (backtest only; nothing live is touched).
Reuses rv_engine (revamp harness) from the parent folder. Baseline = candidate A (bo_r=3.0)."""
import os, sys, json, time, math
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
R=os.path.dirname(os.path.abspath(__file__)); H=os.path.dirname(R); sys.path.insert(0,H)
import rv_engine as E
B=E.B
import numpy as np, pandas as pd
ET='America/New_York'; TICK=0.25

# ------------------------------------------------------------------ levels (all known at signal-bar close)
def build_levels(m5):
    idx=m5.index.tz_convert(ET); mins=np.asarray(idx.hour*60+idx.minute); wd=np.asarray(idx.weekday)
    rth=(wd<5)&(mins>=570)&(mins<960)
    d=idx.normalize().tz_localize(None)
    nd=np.where((wd>=5)|(mins>=960), d.values+np.timedelta64(1,'D'), d.values)
    nd=pd.DatetimeIndex(nd); w2=np.asarray(nd.weekday)
    nd=nd+pd.to_timedelta(np.where(w2==5,2,np.where(w2==6,1,0)),unit='D')
    td=pd.Series(nd.values)
    hi=pd.Series(m5['high'].values); lo=pd.Series(m5['low'].values)
    rthm=pd.Series(rth)
    lv=pd.DataFrame(index=range(len(m5)))
    lv['sh']=hi.groupby(td).cummax(); lv['sl']=lo.groupby(td).cummin()
    lv['onh']=hi.where(~rthm).groupby(td).cummax().groupby(td).ffill(); lv['onl']=lo.where(~rthm).groupby(td).cummin().groupby(td).ffill()
    rh=hi.where(rthm).groupby(td).max().shift(1); rl=lo.where(rthm).groupby(td).min().shift(1)
    lv['pdh']=td.map(rh).values; lv['pdl']=td.map(rl).values
    ah=hi.groupby(td).max().shift(1); al=lo.groupby(td).min().shift(1)
    lv['psh']=td.map(ah).values; lv['psl']=td.map(al).values
    # 5m fractal pivots (2 bars each side), usable only once 2 later bars have closed
    h=m5['high'].values; l=m5['low'].values; n=len(h)
    ph=np.full(n,np.nan); pl=np.full(n,np.nan)
    for i in range(2,n-2):
        if h[i]>h[i-1] and h[i]>h[i-2] and h[i]>=h[i+1] and h[i]>=h[i+2]: ph[i]=h[i]
        if l[i]<l[i-1] and l[i]<l[i-2] and l[i]<=l[i+1] and l[i]<=l[i+2]: pl[i]=l[i]
    return lv,ph,pl

def cands_for(mode,s,dr,c,Rp,lv,ph,pl):
    if mode.startswith('swing'):
        L=int(mode[5:]); a=max(0,s-L); seg=(ph if dr>0 else pl)[a:max(a,s-1)]   # pivots at i<=s-2 only
        return seg[np.isfinite(seg)]
    if mode=='day':
        ks=('pdh','psh','onh','sh') if dr>0 else ('pdl','psl','onl','sl')
        v=np.array([lv[k].values[s] for k in ks],dtype=float); return v[np.isfinite(v)]
    if mode.startswith('round'):
        st=float(mode[5:])
        return np.array([math.ceil((c+Rp)/st)*st]) if dr>0 else np.array([math.floor((c-Rp)/st)*st])
    raise ValueError(mode)

def apply_levels(ens,m5,mode,partial):
    lv,ph,pl=build_levels(m5)
    pos=m5.index.get_indexer(ens.index); cl=m5['close'].values
    ens=ens.copy(); ens['tp1']=np.nan; ens['lvl']=False
    isb=(ens.tag=='ens_breakout').values; nlv=0; nb=0
    tp=ens.tp.values.copy(); tp1=ens.tp1.values.copy(); lvf=ens.lvl.values.copy()
    for k in np.where(isb)[0]:
        s=pos[k]; dr=int(ens.dir.values[k]); c=cl[s]; Rp=abs(c-ens.sl.values[k]); nb+=1
        if Rp<=0 or s<0: continue
        cd=cands_for(mode,s,dr,c,Rp,lv,ph,pl)
        if dr>0:
            lo_b,hi_b=c+Rp,c+3*Rp; v=cd[(cd>=lo_b)&(cd<=hi_b)]
            if len(v)==0: continue
            t=max(round((v.min()-TICK)/TICK)*TICK,lo_b)
        else:
            lo_b,hi_b=c-Rp,c-3*Rp; v=cd[(cd<=lo_b)&(cd>=hi_b)]
            if len(v)==0: continue
            t=min(round((v.max()+TICK)/TICK)*TICK,lo_b)
        nlv+=1; lvf[k]=True
        if partial: tp1[k]=t
        else: tp[k]=t
    ens['tp']=tp; ens['tp1']=tp1; ens['lvl']=lvf
    return ens,dict(bo_signals=nb,bo_with_level=nlv)

# ------------------------------------------------------------------ simulator (copy of B.simulate + partial leg + lvl flag)
def simulate_p(m1,m5_index,signals,rules,frac=0.5):
    sig=signals.sort_index(); sig=sig[~sig.tag.isin(rules.disabled_tags)]
    by_bar={t:g for t,g in sig.groupby(level=0)}
    t1=m1.index.values; o1,h1,l1,c1=(m1[k].values for k in ('open','high','low','close'))
    PV=B.PV; COMM=B.COMM_RT; SLIP=B.SLIP; TICKv=B.TICK
    sess_cache={}
    def sess(ts):
        k=ts.floor('5min')
        if k not in sess_cache: sess_cache[k]=B.session_of(k)
        return sess_cache[k]
    open_pos=[]; trades=[]; st=dict(cooldown=0,consec=0,loss_cd=pd.Timestamp.min.tz_localize('UTC'),pause=pd.Timestamp.min.tz_localize('UTC'))
    sess_pnl={}; day_pnl={}; pid=[0]
    def book(p,ts,px,reason,qty,final):
        pnl=qty*p.dir*(px-p.entry)*PV-COMM*qty
        trades.append(dict(pid=p.pid,tag=p.tag,dir=p.dir,entry_time=p.entry_time,exit_time=ts,entry=p.entry,exit=px,pnl=pnl,reason=reason,session=p.session,lvl=p.lvl,final=final))
        sess_pnl[(p.session,p.tday)]=sess_pnl.get((p.session,p.tday),0.0)+pnl
        day_pnl[p.tday]=day_pnl.get(p.tday,0.0)+pnl
        p.realized+=pnl
        if final:
            tot=p.realized
            st['cooldown']=B.CD_WIN_BARS if tot>0 else B.CD_LOSS_BARS
            if tot<0:
                st['consec']+=1; st['loss_cd']=ts+pd.Timedelta(minutes=B.LOSS_CD_MIN)
                if st['consec']>=B.CONSEC_MAX: st['pause']=ts+pd.Timedelta(minutes=B.CONSEC_PAUSE_MIN)
            elif tot>0: st['consec']=0
    def close(p,ts,px,reason):
        book(p,ts,px,reason,p.qty,True)
    idx1=0; n1=len(t1)
    for bi,bt in enumerate(m5_index):
        bend=bt+pd.Timedelta(minutes=5); bt64,bend64=bt.to_datetime64(),bend.to_datetime64()
        while idx1<n1 and t1[idx1]<bt64: idx1+=1
        j=idx1
        while j<n1 and t1[j]<bend64:
            ts=pd.Timestamp(t1[j]).tz_localize('UTC')
            for p in list(open_pos):
                if p.entry_time>ts: continue
                hit_sl=(l1[j]<=p.sl) if p.dir>0 else (h1[j]>=p.sl)
                hit_tp=(h1[j]>=p.tp+B.TP_THROUGH) if p.dir>0 else (l1[j]<=p.tp-B.TP_THROUGH)
                if hit_sl:
                    gap=(o1[j]<p.sl) if p.dir>0 else (o1[j]>p.sl)
                    px=(o1[j] if gap else p.sl)-p.dir*p.slip
                    open_pos.remove(p); close(p,ts,px,'STOP_LOSS'); continue
                if not p.done and np.isfinite(p.tp1):
                    hit1=(h1[j]>=p.tp1+B.TP_THROUGH) if p.dir>0 else (l1[j]<=p.tp1-B.TP_THROUGH)
                    if hit1:
                        book(p,ts,p.tp1,'TP1',frac,False); p.qty-=frac; p.done=True
                if hit_tp:
                    open_pos.remove(p); close(p,ts,p.tp,'TAKE_PROFIT'); continue
                if p.exit_after is not None and ts+pd.Timedelta(minutes=1)>=p.exit_after:
                    open_pos.remove(p); close(p,ts,c1[j]-p.dir*p.slip,'TIME'); continue
                if p.lock:
                    best=h1[j] if p.dir>0 else l1[j]
                    p.peak_pts=max(p.peak_pts,p.dir*(best-p.entry))
                    nsl=B._lock_stop(p,c1[j])
                    if nsl is not None: p.sl=nsl
            j+=1
        idx1=j
        if st['cooldown']>0: cd_block=True; st['cooldown']-=1
        else: cd_block=False
        g=by_bar.get(bt)
        if g is None or j>=n1: continue
        if bend<st['loss_cd'] or bend<st['pause']: continue
        kind,tday=sess(bend)
        lim=B.DAY_LOSS if kind=='day' else B.OVN_LOSS
        if lim and sess_pnl.get((kind,tday),0.0)<=-lim: continue
        if B.DAILY_TARGET and day_pnl.get(tday,0.0)>=B.DAILY_TARGET: continue
        main=g[~g.flat_only]; addons=g[g.flat_only]; chosen=None
        if len(main) and not cd_block and len(open_pos)<rules.max_open:
            r=main.iloc[0]; ok=kind!='overnight' or rules.overnight_ok(r.sleeve)
            if ok and open_pos:
                px=c1[j-1]; ok=all(p.dir==r.dir for p in open_pos) and all(p.dir*(px-p.entry)>0 for p in open_pos)
            if ok: chosen=r
        if chosen is None and len(addons) and not open_pos: chosen=addons.iloc[0]
        if chosen is None: continue
        ref=c1[j-1]
        if abs(ref-chosen.sl)*PV>B.MAX_RISK_USD: continue
        slip=SLIP+(TICKv*rules.ovn_extra_slip_ticks if kind=='overnight' else 0.0)
        fill=o1[j]+chosen.dir*slip
        tp=float(chosen.tp) if np.isfinite(chosen.tp) else fill+chosen.dir*1e6
        if (chosen.dir>0 and (fill<=chosen.sl or fill>=tp)) or (chosen.dir<0 and (fill>=chosen.sl or fill<=tp)): continue
        et=pd.Timestamp(t1[j]).tz_localize('UTC')
        exit_after=bend+pd.Timedelta(minutes=5*int(chosen.max_bars)) if chosen.max_bars else None
        p=B.Pos(chosen.tag,int(chosen.dir),fill,float(chosen.sl),tp,bool(chosen.lock),et,exit_after,kind,tday,slip)
        pid[0]+=1; p.pid=pid[0]; p.qty=1.0; p.realized=0.0; p.done=False; p.lvl=bool(chosen.lvl)
        t1v=chosen.tp1 if 'tp1' in chosen.index else np.nan
        p.tp1=float(t1v) if (t1v is not None and np.isfinite(t1v)) else float('nan')
        open_pos.append(p)
    return trades

def run_spec2(spec):
    d=E.load(); s=dict(E.DEFAULT); s.update(spec)
    key=(s['revamp'],s['bo_r'],s['trend_r'],s['mr_target'])
    if key not in d['ens_cache']: d['ens_cache'][key]=E.ens_signals(d['m5'],*key)
    ens=d['ens_cache'][key].copy()
    ens['comp']=[E.comp_of(t,k,h) for t,k,h in zip(ens.tag,ens.kind,ens.et_hour)]
    ens=ens[~ens.comp.isin(s['drop'])]
    info={}
    mode=spec.get('lvl')
    if mode: ens,info=apply_levels(ens,d['m5'],mode,bool(spec.get('partial')))
    else: ens['tp1']=np.nan; ens['lvl']=False
    lk=E.LOCKS[s['lock']]; B.LOCK_ACTIVATE,B.LOCK_KEEP,B.LOCK_TRAIL_START,B.LOCK_KEEP_PCT=lk
    adds=[v.copy() for k,v in d['adds'].items()]
    for a in adds: a['tp1']=np.nan; a['lvl']=False
    cols=['dir','sl','tp','sleeve','tag','lock','flat_only','max_bars','tp1','lvl']
    sig=pd.concat([ens[cols]]+[a[cols] for a in adds])
    B.COMM_RT=s['comm']; B.SLIP=s['slip']; B.TP_THROUGH=s['tp_through']
    B.PERIODS[0]=(B.PERIODS[0][0],s['start'],B.PERIODS[0][2]); B.PERIODS[-1]=(B.PERIODS[-1][0],B.PERIODS[-1][1],s['end'])
    tr=simulate_p(d['m1'],d['m5'].index,sig,B.Rules(overnight_sleeves='all',max_open=s['max_open'],ovn_extra_slip_ticks=s['ovn_slip']))
    df=pd.DataFrame(tr)
    lo,hi=pd.Timestamp(B.PERIODS[0][1],tz='UTC'),pd.Timestamp(B.PERIODS[-1][2],tz='UTC')
    df=df[(df.entry_time>=lo)&(df.entry_time<hi)]
    g=df.groupby('pid')
    pos=g.agg(entry_time=('entry_time','first'),exit_time=('exit_time','max'),pnl=('pnl','sum'),tag=('tag','first'),dir=('dir','first'),
              session=('session','first'),lvl=('lvl','first'),reason=('reason','last'),legs=('pnl','size')).reset_index(drop=True)
    pos['tp1_hit']=(g.reason.apply(lambda x:'TP1' in set(x))).values
    pos['month']=pos.entry_time.dt.tz_convert(ET).dt.strftime('%Y-%m')
    return pos.sort_values('exit_time').reset_index(drop=True),info

def specs():
    S={'base':{}}
    for L in (20,50,100): S[f'swing{L}']=dict(lvl=f'swing{L}')
    S['day']=dict(lvl='day')
    S['round50']=dict(lvl='round50'); S['round100']=dict(lvl='round100')
    for k in ('swing20','swing50','swing100','day','round50','round100'): S['part_'+k]=dict(lvl=k,partial=True)
    return S

def job(item):
    name,sp=item; t0=time.time()
    try:
        sp=dict(sp,bo_r=3.0)
        df,info=run_spec2(sp)
        df.to_csv(os.path.join(R,'trades_'+name+'.csv'),index=False)
        return dict(name=name,spec=sp,info=info,res=E.summarize(df),sec=round(time.time()-t0))
    except Exception:
        import traceback; return dict(name=name,err=traceback.format_exc()[-1500:])

def init_worker():
    try:
        import ctypes; ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)  # BELOW_NORMAL
    except Exception: pass

if __name__=='__main__':
    import multiprocessing as mp
    n=int(sys.argv[1]); only=sys.argv[2:] ; S=specs()
    if only: S={k:v for k,v in S.items() if k in only}
    outp=os.path.join(R,'results_runs.jsonl')
    done=set()
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
            f.write(json.dumps(r,default=str)+'\n'); print(r['name'],r.get('info'),r.get('res',{}).get('all'),r.get('err','')[:1500],flush=True)