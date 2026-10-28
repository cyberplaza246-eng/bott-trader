"""Trade simulator for the OG MNQ bot (sweeps only). Reads signals made by signal_worker.py (real bot code),
replays entries/exits with the live bot's rules: sizing, risk cap, USD profit lock, cooldowns, loss pauses,
session loss limits, Tokyo block, bad UTC hours, extra-trade rule. No live files touched."""
import os, json, glob, math, copy
import numpy as np, pandas as pd
from zoneinfo import ZoneInfo
ET=ZoneInfo('America/New_York'); TK=ZoneInfo('Asia/Tokyo')
HERE=os.path.dirname(os.path.abspath(__file__)); B=r'C:\Users\shawa\Projects\bott-trader\data'
PV=2.0; TICK=0.25

BASE=dict(
  thr=0.64, rsi_buy=70.0, rsi_sell=30.0, ema_block=True, min_agree=3, use_raw_sweep_conf=False,
  tokyo_min=60, bad_hours=(0,4,21), long_only=False, short_only=False, skip_sunday=False, only_session=None,
  sl_scale=1.0, tp_mode='struct', tp_r=None,   # tp_mode: 'struct' (bot's own TP) or 'r' (tp_r x risk)
  lock=True, lock_act=100., lock_keep=20., lock_trail_start=150., lock_trail_pct=0.5,
  size_mode='conf', conf2=.72, conf3=.78, conf4=.85, max_contracts=4, risk_cap=500.,
  cd_win=1, cd_loss=2, loss_cd_min=10, consec_n=3, consec_pause_min=30,
  max_pos=3, extra_rule='same_dir_profit', session_limits=True, day_lim=500., on_lim=300., day_target=2000.,
  same_bar_tp_first=False, be_after_r=None,
)

def load_signals():
    recs={}
    for f in glob.glob(os.path.join(HERE,'out','sig_*.jsonl')):
        for line in open(f,encoding='utf-8'):
            try: r=json.loads(line)
            except Exception: continue
            if r.get('sweep') in ('BUY','SELL') and 'sig' in r: recs[r['i']]=r
    return [recs[k] for k in sorted(recs)]

def load_fine():
    m=pd.read_csv(B+r'\MNQ_1m.csv'); m['datetime']=pd.to_datetime(m['datetime'],utc=True)
    m=m.drop_duplicates('datetime').sort_values('datetime')
    last1=m.datetime.max()
    b5=pd.read_csv(os.path.join(HERE,'bars5m.csv')); b5['datetime']=pd.to_datetime(b5['datetime'],utc=True)
    late=b5[(b5.datetime>last1)&(b5['src']=='live')] if 'src' in b5 else b5[b5.datetime>last1]
    f1=m[['datetime','open','high','low','close']].copy(); f1['dur']=1
    f5=late[['datetime','open','high','low','close']].copy(); f5['dur']=5
    F=pd.concat([f1,f5]).sort_values('datetime').reset_index(drop=True)
    return F

def usd_lock_stop(is_long, entry, size, peak_pts, live, cur_sl, c):
    usd_per_pt=PV*max(1,size)
    live_pts=(live-entry) if is_long else (entry-live)
    if live_pts<=0: return None
    peak=max(peak_pts,live_pts); peak_usd=peak*usd_per_pt
    if peak_usd+1e-9<c['lock_act']: return None
    lock_usd=c['lock_keep']
    if peak_usd+1e-9>=c['lock_trail_start']: lock_usd=max(lock_usd,c['lock_trail_pct']*peak_usd)
    keep_pts=math.floor((lock_usd/usd_per_pt)/TICK+1e-9)*TICK
    if keep_pts<=0: return None
    new_sl=entry+keep_pts if is_long else entry-keep_pts
    buf=TICK*4
    if is_long and new_sl>live-buf+1e-9: return None
    if (not is_long) and new_sl<live+buf-1e-9: return None
    if cur_sl>0:
        if is_long and new_sl<=cur_sl+1e-9: return None
        if (not is_long) and new_sl>=cur_sl-1e-9: return None
    return new_sl

def session_of(ts_et):
    d=ts_et.date(); hm=ts_et.hour*60+ts_et.minute
    def nextwd(d):
        while d.weekday()>=5: d=d+pd.Timedelta(days=1)
        return d
    if d.weekday()<5 and 570<=hm<960: return 'day',d
    if d.weekday()>=5: return 'overnight',nextwd(d)
    if hm>=960: return 'overnight',nextwd(d+pd.Timedelta(days=1))
    return 'overnight',d

def passes(r,c):
    """should_trade() re-computed from the stored signal fields so gates can be ablated."""
    sig=r['sig']
    if sig not in ('BUY','SELL'): return False
    ema=r['ema']
    if c['ema_block'] and ema in ('BUY','SELL') and ema!=sig: return False
    rsi=r['rsi']
    if sig=='BUY' and rsi>c['rsi_buy']: return False
    if sig=='SELL' and rsi<c['rsi_sell']: return False
    if r['agree']<c['min_agree']: return False
    conf=r['sweep_conf'] if c['use_raw_sweep_conf'] else r['conf']
    if conf<c['thr']: return False
    return True

def simulate(cands,F,cfg=None,costs=('gross','bot','real')):
    c=dict(BASE); c.update(cfg or {})
    ft=F.datetime.values.astype('datetime64[ns]').astype('int64'); fo=F.open.values; fh=F.high.values; fl=F.low.values; fc=F.close.values
    NS5=5*60*10**9
    pos=[]; trades=[]; fp=0
    cd_until=0; loss_until=0; consec=0; consec_until=0
    ledger={}   # (td,kind)->pnl ; td->pnl
    tdp={}
    def book_exit(p,price,reason,t_ns,gap=False):
        nonlocal cd_until,loss_until,consec,consec_until
        pts=(price-p['entry']) if p['long'] else (p['entry']-price)
        raw=pts*PV*p['size']
        t=dict(entry_t=p['t'],exit_t=pd.Timestamp(t_ns,tz='UTC'),dir='long' if p['long'] else 'short',entry=p['entry'],exit=price,size=p['size'],
               pts=pts,reason=reason,raw=raw,conf=p['conf'],sl0=p['sl0'],tp0=p['tp0'],rsi=p['rsi'],regime=p['regime'])
        t['pnl_gross']=raw; t['pnl_bot']=raw-(2.0+TICK*PV); t['pnl_real']=raw-2.24*p['size']
        trades.append(t)
        net=t['pnl_bot']
        # bookkeeping uses bot accounting (what the live bot books)
        k=(p['td'],p['kind']); ledger[k]=ledger.get(k,0)+net; tdp[p['td']]=tdp.get(p['td'],0)+net
        # cooldown (bars, counted from next 5m boundary at/after exit)
        nb=((t_ns+NS5-1)//NS5)*NS5
        n=c['cd_win'] if net>0 else c['cd_loss']
        cd_until=nb+n*NS5
        if net<0:
            consec+=1; loss_until=t_ns+c['loss_cd_min']*60*10**9
            if consec>=c['consec_n']: consec_until=t_ns+c['consec_pause_min']*60*10**9; consec=0
        elif net>0: consec=0
    def advance_to(limit_ns):
        nonlocal fp
        n=len(ft)
        while fp<n and ft[fp]<limit_ns and pos:
            o,h,l,cl=fo[fp],fh[fp],fl[fp],fc[fp]; endt=ft[fp]+(F.dur.iat[fp])*60*10**9
            for p in list(pos):
                if ft[fp]<p['start_ns']: continue
                if p['long']:
                    if o<=p['sl']: book_exit(p,o,'STOP_LOSS(gap)',endt); pos.remove(p); continue
                    if l<=p['sl']:
                        book_exit(p,p['sl'],'STOP_LOSS' if p['sl']<p['entry']-1e-9 else ('LOCK_STOP'),endt); pos.remove(p); continue
                    if h>=p['tp']:
                        book_exit(p,max(p['tp'],o) if o>=p['tp'] else p['tp'],'TAKE_PROFIT',endt); pos.remove(p); continue
                    fav=h-p['entry']
                else:
                    if o>=p['sl']: book_exit(p,o,'STOP_LOSS(gap)',endt); pos.remove(p); continue
                    if h>=p['sl']:
                        book_exit(p,p['sl'],'STOP_LOSS' if p['sl']>p['entry']+1e-9 else 'LOCK_STOP',endt); pos.remove(p); continue
                    if l<=p['tp']:
                        book_exit(p,p['tp'],'TAKE_PROFIT',endt); pos.remove(p); continue
                    fav=p['entry']-l
                p['peak']=max(p['peak'],fav)
                if c['lock']:
                    ns=usd_lock_stop(p['long'],p['entry'],p['size'],p['peak'],cl,p['sl'],c)
                    if ns is not None: p['sl']=ns
            fp+=1
        if not pos:
            fp=max(fp,int(np.searchsorted(ft,limit_ns)))
    for r in cands:
        D=pd.Timestamp(r['t']).value+NS5   # decision time ns (bar close)
        advance_to(D)
        if not pos: fp=max(fp,int(np.searchsorted(ft,D)))
        if not passes(r,c): continue
        long_=r['sig']=='BUY'
        if c['long_only'] and not long_: continue
        if c['short_only'] and long_: continue
        Dts=pd.Timestamp(D,tz='UTC'); Det=Dts.tz_convert(ET)
        if c['skip_sunday'] and Det.weekday()==6: continue
        if Dts.hour in c['bad_hours']: continue
        if c['tokyo_min']>0:
            tk=Dts.tz_convert(TK); st=tk.replace(hour=9,minute=0,second=0,microsecond=0)
            if st<=tk<st+pd.Timedelta(minutes=c['tokyo_min']): continue
        kind,td=session_of(Det)
        if c['only_session'] and kind!=c['only_session']: continue
        if len(pos)>=c['max_pos'] or D<cd_until or D<loss_until or D<consec_until: continue
        if c['session_limits']:
            if c['day_target']>0 and tdp.get(td,0)>=c['day_target']: continue
            sp=ledger.get((td,kind),0)
            if kind=='day' and c['day_lim']>0 and sp<=-c['day_lim']: continue
            if kind=='overnight' and c['on_lim']>0 and sp<=-c['on_lim']: continue
        entry=r['close']
        if 'sl' not in r or 'tp' not in r: continue
        sl=r['sl']; tp=r['tp']
        risk=abs(entry-sl)*c['sl_scale']
        if risk<=0: continue
        sl=entry-risk if long_ else entry+risk
        if c['tp_mode']=='r': tp=entry+c['tp_r']*risk if long_ else entry-c['tp_r']*risk
        sl=round(sl/TICK)*TICK; tp=round(tp/TICK)*TICK; risk=abs(entry-sl)
        if (long_ and (tp<=entry or sl>=entry)) or ((not long_) and (tp>=entry or sl<=entry)): continue
        if pos:
            if c['extra_rule']=='same_dir_profit':
                ok=all(p['long']==long_ and ((entry-p['entry']) if p['long'] else (p['entry']-entry))>0 for p in pos)
            else: ok=r['conf']>=0.70
            if not ok: continue
        conf=r['conf']
        if c['size_mode']=='conf':
            n=4 if conf>=c['conf4'] else 3 if conf>=c['conf3'] else 2 if conf>=c['conf2'] else 1
            n=min(n,c['max_contracts'])
        else: n=int(c['size_mode'])
        rpc=risk*PV
        if c['risk_cap']>0 and rpc*n>c['risk_cap']+1e-9: n=min(n,int((c['risk_cap']+1e-9)//rpc))
        if n<=0: continue
        pos.append(dict(long=long_,entry=entry,sl=sl,tp=tp,size=n,peak=0.0,t=Dts,start_ns=D,conf=conf,sl0=sl,tp0=tp,rsi=r['rsi'],regime=r['regime'],td=td,kind=kind))
    advance_to(int(1e19)) if pos else None
    open_end=len(pos)
    return pd.DataFrame(trades), open_end

def metrics(df,col='pnl_bot'):
    if df is None or len(df)==0: return dict(n=0,net=0.0,wr=0,avg_w=0,avg_l=0,pf=0,maxdd=0,avg=0)
    d=df.sort_values('exit_t'); p=d[col].values
    w=p[p>0]; l=p[p<0]; eq=np.cumsum(p); dd=(np.maximum.accumulate(np.concatenate([[0],eq]))[1:]-eq).max()
    return dict(n=int(len(p)),net=float(p.sum()),wr=float((p>0).mean()*100),avg_w=float(w.mean()) if len(w) else 0.0,avg_l=float(l.mean()) if len(l) else 0.0,
                pf=float(w.sum()/-l.sum()) if len(l) and l.sum()!=0 else float('inf'),maxdd=float(dd),avg=float(p.mean()))
