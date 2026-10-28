import os,sys,json,logging,importlib.util
os.environ['PYTHONUTF8']='1'
H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,H)
import pandas as pd, numpy as np, sim
SB=os.path.join(H,'sandbox'); sys.path.insert(0,SB); cwd=os.getcwd(); os.chdir(SB)
from src.ai.adaptive_learner import AdaptiveLearner
L=AdaptiveLearner(); os.chdir(cwd); logging.disable(logging.CRITICAL)
oldmod={r:L.get_regime_confidence_modifier(r) for r in ('trending','ranging','volatile')}
print('learner old modifiers',oldmod, 'stats',L.regime_stats,flush=True)
# load trend_15m / regime_modifier_v2 source from live file without heavy imports
src=open(r'C:\Users\shawa\Projects\bott-trader\src\core\ensemble_trader.py',encoding='utf-8').read()
a=src.index('def trend_15m'); b=src.index('\nclass ',a)
ns={'pd':pd,'os':os}; exec(src[a:b],ns)
C=sim.load_signals(); F=sim.load_fine()
bars=pd.read_csv(os.path.join(H,'bars5m.csv'),parse_dates=['datetime']); bars['datetime']=pd.to_datetime(bars['datetime'],utc=True); bars=bars.set_index('datetime')
for r in C:
    w=bars.iloc[max(0,r['i']-99):r['i']+1]; r['tf15']=ns['trend_15m'](w)
def variant(fix,m15):
    out=[]
    for r in C:
        r2=dict(r)
        if fix and r['sig'] in ('BUY','SELL'):
            om=oldmod.get(r['regime'],1.0) or 1.0
            os.environ['OG_REGIME_15M']='1' if m15 else '0'
            class W: pass
            ns['trend_15m_cache']=r['tf15']
            nm,_,_=ns['regime_modifier_v2'](L,r.get('sweep_regime'),r['sig'],r['close'],r.get('ema200'),None)
            if m15 and str(r.get('sweep_regime')) in ('trend_up','trend_down'):
                # recompute with stored 15m direction
                want='up' if r['sig']=='BUY' else 'down'; tdir='BUY' if r['sweep_regime']=='trend_up' else 'SELL'
                ema_ok=r.get('ema200') is None or ((r['close']>r['ema200']) if r['sig']=='BUY' else (r['close']<r['ema200']))
                if r['sig']!=tdir or not ema_ok: nm=0.80
                elif r['tf15']=='neutral': nm=0.95
                elif r['tf15']!=want: nm=0.80
                else: nm=1.0
            r2['conf']=r['conf']/om*nm
        out.append(r2)
    return out
cfg=dict(max_contracts=1)
res={}
for name,fix,m15 in (('OLD (FIX=0)',0,0),('FIX, no 15m',1,0),('FIX + 15m',1,1)):
    d,_=sim.simulate(variant(fix,m15),F,cfg)
    d=d.copy(); d['pnl']=d['pnl_gross']/d['size']-4.0 if 'size' in d else d['pnl_gross']-4.0
    d=d.sort_values('exit_t'); cut=pd.Timestamp('2025-12-31',tz='UTC')
    row={}
    for lab,s in (('ALL',d),('train<=2025',d[d.exit_t<=cut]),('test 2026',d[d.exit_t>cut])):
        m=sim.metrics(s,'pnl'); row[lab]=dict(n=m['n'],wr=round(m['wr'],1),net=round(m['net']),avg=round(m['avg'],1),maxdd=round(m['maxdd']))
    res[name]=row; print(name,json.dumps(row),flush=True)
json.dump(res,open(os.path.join(H,'regime_fix_bt.json'),'w'),indent=1)
