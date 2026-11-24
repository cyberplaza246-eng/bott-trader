"""Strict double-test of NY-open gap fade: freeze on first 12 months, score on the rest. BACKTEST ONLY."""
import sys,os
sys.argv=['x']
import sr_pub as P
from sr_pub import *
ROLL=['2024-12-16','2025-03-17','2025-03-18','2025-06-16','2025-06-17','2025-09-16','2025-12-16','2026-03-16','2026-09-14','2026-09-15']
rolld=set((pd.Timestamp(r)+pd.Timedelta(days=k)).strftime('%Y-%m-%d') for r in ROLL for k in range(-1,2))
A0,A1,B0,B1='2024-11-01','2025-11-01','2025-11-01','2026-10-01'
def ny(q,H,tk=1,dly=0,stop=None,exroll=False,lo=None,hi=None,gapsrc=None):
    rows=[];past=[]
    for k,D in enumerate(days):
        if k==0: continue
        pc=sess[days[k-1]]['cl']; s=sess[D]; a=s['a']; g=s['op']-pc
        ok=len(past)>=20 and abs(g)>=(np.quantile(past,q) if q>0 else 0) and g!=0
        past.append(abs(g))
        if not ok: continue
        if exroll and D in rolld: continue
        if lo and D<lo: continue
        if hi and D>=hi: continue
        side=-int(np.sign(g)); i_in=a+dly; e=a+H
        ent=o[i_in]+side*0.25*tk
        if stop: j,px,_=walk(i_in,e,side,ent-side*stop,None,tk)
        else: j,px=e,o[e]-side*0.25*tk
        r=mk(i_in,ent,j,px,side); r['d']=D; rows.append(r)
    return rows
def mstat(df,months):
    s=st(df); s['tpm']=round(len(df)/months,1); s['corr']=cb(df); return s
def cb(df):
    a=df.groupby(df.exit_time.dt.tz_convert(ET).dt.date).pnl.sum(); b=P.base.groupby(P.base.exit_time.dt.tz_convert(ET).dt.date).pnl.sum()
    j=pd.concat([a,b],axis=1).fillna(0); return round(float(j.corr().iloc[0,1]),3)
def worst(df):
    d=df.groupby(df.exit_time.dt.tz_convert(ET).dt.date).pnl.sum(); w=df.set_index(df.exit_time.dt.tz_convert(ET)).pnl.resample('W').sum()
    p=df.pnl.values; ls=0;cur=0
    for x in p:
        cur=cur+1 if x<0 else 0; ls=max(ls,cur)
    return dict(worst_trade=round(p.min()),worst_day=round(d.min()),worst_week=round(w.min()),max_dd=round(P.mdd(p)),max_losing_streak=ls)
if __name__=='__main__':
    # --- stage 1: grid on window A only
    grid=[(q,H) for q in (0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9) for H in (30,60,90,120,180,240)]
    ra=[]
    for q,H in grid:
        df=dfm(ny(q,H,lo=A0,hi=A1)); s=st(df) if len(df)>=3 else dict(n=len(df),net=-1e9,pf=0)
        ra.append(dict(q=q,H=H,n=s['n'],net=s['net'],pf=s['pf'],dd=s.get('dd')))
    RA=pd.DataFrame(ra).sort_values('net',ascending=False); print('GRID on window A (',A0,'..',A1,') variants:',len(grid)); print(RA.head(8).to_string()); print('A, q75/H90 row:',RA[(RA.q==0.75)&(RA.H==90)].to_dict('records'))
    q,H=RA.iloc[0].q,int(RA.iloc[0].H); print('FROZEN on A by net:',q,H)
    # also the pre-existing choice
    for lab,(qq,HH) in (('FROZEN_by_A',(q,H)),('ORIGINAL q75/H90',(0.75,90))):
        for stp in (None,300):
            dfA=dfm(ny(qq,HH,stop=stp,lo=A0,hi=A1)); dfB=dfm(ny(qq,HH,stop=stp,lo=B0,hi=B1))
            mA=(pd.Timestamp(A1)-pd.Timestamp(A0)).days/30.4; mB=(pd.Timestamp('2026-09-22')-pd.Timestamp(B0)).days/30.4
            sA=mstat(dfA,mA); sB=mstat(dfB,mB)
            print(f'\n== {lab} q{qq} H{HH} stop={stp}')
            print(' A (in-sample):',{k:sA[k] for k in('n','tpm','net','pf','dd','worst','t','corr')})
            print(' B (out-of-sample):',{k:sB[k] for k in('n','tpm','net','pf','dd','worst','t','boot_p','corr')})
            print('   B worst:',worst(dfB))
            for nmx,kw in (('slip4',dict(tk=4)),('delay1',dict(dly=1)),('delay3',dict(dly=3)),('delay3+slip4',dict(dly=3,tk=4)),('exroll',dict(exroll=True))):
                d_=dfm(ny(qq,HH,stop=stp,lo=B0,hi=B1,**kw)); print(f'   B {nmx}: n {len(d_)} net {round(d_.pnl.sum())} pf {st(d_)["pf"]}')
            cmb=pd.concat([P.base[['entry_time','exit_time','pnl']],dfB[['entry_time','exit_time','pnl']]]); bB=P.base[P.base.entry_time>=pd.Timestamp(B0,tz='UTC')]
            print('   baseline in B net',round(bB.pnl.sum()),'dd',round(P.mdd(bB.sort_values('exit_time').pnl.values)),'| baseline+gap B net',round(bB.pnl.sum()+dfB.pnl.sum()),'dd',round(P.mdd(pd.concat([bB[['exit_time','pnl']],dfB[['exit_time','pnl']]]).sort_values('exit_time').pnl.values)))
    # --- lookahead checks
    # (1) point-in-time: threshold at each event uses only earlier events; re-implement independently with explicit truncation
    ref=dfm(ny(0.75,90)); alld=sorted(sess); trades2=[]; hist=[]
    for k,D in enumerate(alld):
        if k==0: continue
        pc=sess[alld[k-1]]['cl']; a=sess[D]['a']; g=o[a]-pc          # only data up to 09:30 open
        thr=np.quantile(np.array(hist),0.75) if len(hist)>=20 else None
        if thr is not None and abs(g)>=thr and g!=0:
            side=-int(np.sign(g)); trades2.append((D,side,round(side*(o[a+90]-0.25*(-side)*0 -o[a]),2)))
        hist.append(abs(g))
    print('\nLOOKAHEAD (1) point-in-time re-implementation: trades',len(trades2),'vs vectorized',len(ref),'same dates/sides:',[ (t[0],t[1]) for t in trades2]==[(r['d'],int(r['dir'])) for r in ref.sort_values('entry_time').to_dict('records')] if 'd' in ref else 'n/a')
    # (2) placebo: random sign of gap -> should be ~ -costs
    rng2=np.random.default_rng(1); pl=[]
    for rep in range(200):
        tot=0;past=[]
        for k,D in enumerate(days):
            if k==0: continue
            s=sess[D]; a=s['a']; g=s['op']-sess[days[k-1]]['cl']; ok=len(past)>=20 and abs(g)>=np.quantile(past,0.75); past.append(abs(g))
            if not ok: continue
            side=int(rng2.choice([-1,1])); tot+=side*(o[a+90]-o[a]-side*0.5)*2-4
        pl.append(tot)
    pl=np.array(pl); print('LOOKAHEAD (2) placebo (random side, same days) net: mean',round(pl.mean()),'sd',round(pl.std()),'95th pct',round(np.percentile(pl,95)),'| real',round(ref.pnl.sum()),'-> real exceeds placebo p-value',round(float((pl>=ref.pnl.sum()).mean()),3))
    # (3) timing: is minute-label start-of-bar? volume peak at label 09:30 vs 09:29
    mv_=pd.Series(v).groupby(MOD).mean(); print('LOOKAHEAD (3) mean 1m volume at label 09:29 / 09:30 / 09:31:',round(mv_[569]),round(mv_[570]),round(mv_[571]))
    # (4) entry one bar LATE (entry at 09:31 open) already in delay1 results; and entry at the CLOSE of 09:30 bar (not known at open) excluded.