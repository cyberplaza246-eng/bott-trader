"""Addendum 8 part 2: re-simulate the 3R baseline with time-of-day / FOMC-day entry rules (full sim: concurrency, loss limits, cooldowns re-run). BACKTEST ONLY."""
import os,sys,json,time,inspect,textwrap
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
import sr_engine as S
import numpy as np, pandas as pd
from sr_x8_desc import FOMC,ET,mdd,met
RT=S.RT; E=S.E
# --- patch 1: simulate_p gets an optional forced-flat timestamp set (position closed at 1m close - slip) ---
src=inspect.getsource(RT.simulate_p)
key="                if p.exit_after is not None and ts+pd.Timedelta(minutes=1)>=p.exit_after:"
assert key in src
src=src.replace(key,"                if FLAT_TS and ts in FLAT_TS:\n                    open_pos.remove(p); close(p,ts,c1[j]-p.dir*p.slip,'FOMC_FLAT'); continue\n"+key)
RT.FLAT_TS=set(); exec(src,RT.__dict__)
# --- patch 2: run_spec3 applies spec['trule'] (function of signal bar-end ET timestamps -> True = drop) ---
s3=inspect.getsource(S.run_spec3).replace('def run_spec3','def run_spec4')
k2="    lk=E.LOCKS[s['lock']]"; assert k2 in s3
s3=s3.replace(k2,"    tr_=sp.get('trule')\n    if tr_ is not None:\n        ens=ens[~tr_(ens.index+pd.Timedelta(minutes=5))]; adds=[a[~tr_(a.index+pd.Timedelta(minutes=5))] for a in adds]\n"+k2,1)
exec(s3,S.__dict__)
FD=set(FOMC)
def mk_rule(t0,t1,fomc_only):
    a=t0[0]*60+t0[1]; b=t1[0]*60+t1[1]
    def rule(bend):
        e=bend.tz_convert(ET); m=np.asarray(e.hour*60+e.minute); day=np.asarray(e.strftime('%Y-%m-%d'))
        hit=(m>=a)&(m<b)
        if fomc_only: hit&=np.isin(day,list(FD))
        return hit
    return rule
def flat_set(hm):
    return set(pd.Timestamp(f'{d} {hm}',tz=ET).tz_convert('UTC') for d in FOMC)
V={ # name: (spec, flat_hm or None, description)
 'base':({},None,'baseline'),
 'R1_all_1315_1800':(dict(trule=mk_rule((13,15),(18,0),False)),None,'ALL days: no new entries 13:15-18:00 ET (trader rule)'),
 'R1b_all_1315_1600':(dict(trule=mk_rule((13,15),(16,0),False)),None,'ALL days: no new entries 13:15-16:00'),
 'R2_fomc_1315_1600':(dict(trule=mk_rule((13,15),(16,0),True)),None,'FOMC days only: no new entries 13:15-16:00'),
 'R3_fomc_1345_1600_flat':(dict(trule=mk_rule((13,45),(16,0),True)),'13:45','FOMC days: no new entries 13:45-16:00 AND flatten open trades at 13:45'),
 # sensitivity / extra variants (counted in the variant total)
 'R1c_all_1400_1800':(dict(trule=mk_rule((14,0),(18,0),False)),None,'ALL days: no new entries 14:00-18:00'),
 'R1d_all_1500_1800':(dict(trule=mk_rule((15,0),(18,0),False)),None,'ALL days: no new entries 15:00-18:00'),
 'R1e_all_1200_1800':(dict(trule=mk_rule((12,0),(18,0),False)),None,'ALL days: no new entries 12:00-18:00'),
 'R2b_fomc_1315_1800':(dict(trule=mk_rule((13,15),(18,0),True)),None,'FOMC days: no new entries 13:15-18:00'),
 'R2c_fomc_1400_1600':(dict(trule=mk_rule((14,0),(16,0),True)),None,'FOMC days: no new entries 14:00-16:00 (around statement)'),
 'R3b_fomc_1345_1600_noflat':(dict(trule=mk_rule((13,45),(16,0),True)),None,'FOMC days: no new entries 13:45-16:00 (no flatten)'),
 'R3c_fomc_flat1345_only':(dict(),'13:45','FOMC days: flatten open trades at 13:45 only (entries allowed)'),
 'R3d_fomc_1345_1600_flat1400':(dict(trule=mk_rule((13,45),(16,0),True)),'13:55','FOMC days: no entries 13:45-16:00, flatten at 13:55'),
 'R4_fomc_skipday':(dict(trule=mk_rule((0,0),(24*60,0),True)),None,'FOMC calendar days: no entries all day (sanity: FOMC days are good)'),
}
def run(name):
    sp,flat,desc=V[name]; RT.FLAT_TS=flat_set(flat) if flat else set()
    t0=time.time(); df=S.run_spec4(sp); df.to_csv(f'x8_trades_{name}.csv',index=False)
    print(name,len(df),round(df.pnl.sum(),1),round(time.time()-t0),'s',flush=True); return df
if __name__=='__main__':
    names=sys.argv[1:] or list(V)
    for n in names: run(n)
