s=open('code/sim.py',encoding='utf-8').read().replace('\r\n','\n')
# paths
s=s.replace("HERE=os.path.dirname(os.path.abspath(__file__)); B=r'C:\\Users\\shawa\\Projects\\bott-trader\\data'","HERE=os.path.dirname(os.path.abspath(__file__)); B=os.path.join(HERE,'..','data'); OUTD=os.path.join(HERE,'..','out')")
assert "OUTD" in s
s=s.replace("glob.glob(os.path.join(HERE,'out','sig_*.jsonl'))","glob.glob(os.path.join(OUTD,'fb_*.jsonl'))")
s=s.replace("if r.get('sweep') in ('BUY','SELL') and 'sig' in r: recs[r['i']]=r","if r.get('sig') in ('BUY','SELL') and 'sl' in r: recs[r['i']]=r")
# fine bars from cleaned 1m
i0=s.index('def load_fine():'); i1=s.index('def usd_lock_stop')
s=s[:i0]+'''def load_fine():
    import pickle
    d=pickle.load(open(os.path.join(OUTD,'revamp_base.pkl'),'rb')); m=d['m1'].reset_index()
    m['datetime']=pd.to_datetime(m['datetime'],utc=True)
    f1=m[['datetime','open','high','low','close']].copy(); f1['dur']=1
    return f1.sort_values('datetime').reset_index(drop=True)

'''+s[i1:]
# cost model
s=s.replace("same_bar_tp_first=False, be_after_r=None,","same_bar_tp_first=False, be_after_r=None,\n  comm_rt=4.0, slip_ticks=1, fb_mode='on', fb_regimes=None, fb_dirs=None, fb_kind=None, sweep_kind=None, fb_conf=None, fb_rsi=None,")
s=s.replace("t['pnl_gross']=raw; t['pnl_bot']=raw-(2.0+TICK*PV); t['pnl_real']=raw-2.24*p['size']",
"""t['pnl_gross']=raw; t['pnl_bot']=raw-(2.0+TICK*PV)
        slip_usd=c['slip_ticks']*TICK*PV*p['size']*(1+(1 if reason.startswith('STOP') or reason=='LOCK_STOP' else 0))
        t['pnl_real']=raw-c['comm_rt']*p['size']-slip_usd; t['pnl_224']=raw-2.24*p['size']-slip_usd; t['fb']=p['fb']; t['kind']=p['kind']""")
s=s.replace("regime=p['regime'])\n","regime=p['regime'])\n",1)
# passes: fallback handling
s=s.replace('''    sig=r['sig']
    if sig not in ('BUY','SELL'): return False
    ema=r['ema']''','''    sig=r['sig']
    if sig not in ('BUY','SELL'): return False
    if r.get('sweep') not in ('BUY','SELL'):   # fallback (non-sweep) entry
        if c['fb_mode']=='off': return False
        if not r.get('should_trade'): return False
        if c['fb_regimes'] and r.get('regime') not in c['fb_regimes']: return False
        if c['fb_conf'] is not None and r['conf']<c['fb_conf']: return False
        return True
    if c['fb_mode']=='only': return False
    ema=r['ema']''')
s=s.replace("pos.append(dict(long=long_,","pos.append(dict(fb=(r.get('sweep') not in ('BUY','SELL')),long=long_,")
open('code/sim2.py','w',encoding='utf-8').write(s)
print('written')
