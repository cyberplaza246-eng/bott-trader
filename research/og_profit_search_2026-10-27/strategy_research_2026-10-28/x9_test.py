"""Addendum 9: midday-selloff rules (BACKTEST ONLY). Pre-set grid, frozen before looking at any results:
 trigger: at 1m bar close t in [11:30,14:00] ET, drop = (max 1m high over trailing W min) - close >= X pts; first trigger per day.
 X in {80,120,180} pts, W in {60,90} min -> 6 event definitions. Entry next 1m open +/-1 tick slippage; exit at open of bar entry+H (H in {30,60,120}, capped 15:55) -/+ 1 tick; $4 RT.
 dir: +1 = buy-the-dip/fade, -1 = continuation short. 6 x 2 x 3 = 36 variants. No stop (time exit)."""
import os,sys,json
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
sys.argv=['x']
import sr_pub as P
import numpy as np, pandas as pd
from sr_x8_desc import load as loadb, met, ET
ix,o,h,l,c,DS=P.ix,P.o,P.h,P.l,P.c,P.DS
pos,sess,days=P.pos,P.sess,P.days
XS=(80,120,180); WS=(60,90); HS=(30,60,120); TD=pd.Timedelta(minutes=1)
SPLIT=pd.Timestamp('2026-01-01',tz='UTC')
def pnl_at(i_sig,side,H,cap):
    i=i_sig+1; j=min(i+H,cap)
    if j<=i+4 or ix[j]-ix[i]!=(j-i)*TD or ix[i]-ix[i_sig]!=TD: return None
    pin=o[i]+side*0.25; pout=o[j]-side*0.25
    return side*(pout-pin)*2-4.0, ix[i]
# events
EV={}
for X in XS:
    for W in WS:
        ev=[]
        for D in days:
            a=pos(D,'11:30'); z=pos(D,'14:00')
            if a is None or z is None: continue
            for i in range(a,z+1):
                if i-W<0 or ix[i]-ix[i-W]!=W*TD: continue
                hm=h[i-W+1:i+1].max(); drop=hm-c[i]
                if drop>=X: ev.append((D,i,drop)); break
        EV[(X,W)]=ev
print('events (X,W)->n days:',{k:len(v) for k,v in EV.items()},'of',len(days),'days',flush=True)
# placebo pool: every minute 11:30-14:00 on every day, per (side,H)
caps={D:pos(D,'15:55') for D in days}
pool={}
for side in (1,-1):
    for H in HS:
        arr=[]
        for D in days:
            a=pos(D,'11:30'); z=pos(D,'14:00'); cap=caps[D]
            if a is None or z is None or cap is None: continue
            for i in range(a,z+1):
                r=pnl_at(i,side,H,cap)
                if r: arr.append((r[0],r[1]))
        pool[(side,H)]=(np.array([x[0] for x in arr]),np.array([x[1]>=SPLIT for x in arr]))
        print('pool',side,H,len(arr),'mean',round(float(np.mean([x[0] for x in arr])),2),flush=True)
rng=np.random.default_rng(7)
def pnull(pp,n,obs,nsim=4000):
    if n==0: return float('nan')
    idx=rng.integers(0,len(pp),(nsim,n)); m=pp[idx].mean(1); return float((m>=obs).mean())
base=loadb('trades_base.csv'); base['xday']=base.exit_time.dt.tz_convert(ET).dt.strftime('%Y-%m-%d')
bday=base.groupby('xday').pnl.sum()
rows=[]; alltr={}
for (X,W),ev in EV.items():
    for side in (1,-1):
        for H in HS:
            tr=[]
            for D,i,drop in ev:
                r=pnl_at(i,side,H,caps[D])
                if r: tr.append(dict(day=D,pnl=r[0],t=r[1],drop=drop))
            df=pd.DataFrame(tr); alltr[(X,W,side,H)]=df
            row=dict(X=X,W=W,dir='FADE-long' if side>0 else 'CONT-short',H=H,n=len(df))
            pp,w26=pool[(side,H)]
            for nm,mask in (('all',None),('train',False),('y26',True)):
                d=df if mask is None else df[(df.t>=SPLIT)==mask]
                n=len(d); row[f'n_{nm}']=n
                row[f'avg_{nm}']=round(float(d.pnl.mean()),1) if n else np.nan
                row[f'net_{nm}']=round(float(d.pnl.sum())) if n else 0
                if nm=='all' and n>1:
                    row['t']=round(float(d.pnl.mean()/(d.pnl.std(ddof=1)/np.sqrt(n))),2)
                    g=d.pnl[d.pnl>0].sum(); ls=-d.pnl[d.pnl<0].sum(); row['pf']=round(float(g/ls),2) if ls>0 else 99
                    row['wr']=round(100*float((d.pnl>0).mean()))
                    row['p_null']=round(pnull(pp,n,float(d.pnl.mean())),3)
                    row['p_bonf36']=round(min(1,row['p_null']*36),3)
                    row['placebo_avg']=round(float(pp.mean()),1)
                if nm!='all' and n>1:
                    sub=pp[w26==(nm=='y26')]; row[f'p_{nm}']=round(pnull(sub,n,float(d.pnl.mean())),3)
            rows.append(row)
R=pd.DataFrame(rows); R.to_csv('x9_rules.csv',index=False)
pd.set_option('display.width',300); pd.set_option('display.max_columns',40)
print(R.to_string())
R['both']=(R.avg_train>0)&(R.avg_y26>0)
print('variants',len(R),'positive all',int((R.avg_all>0).sum()),'positive both windows',int(R.both.sum()),'p_null<0.05 (raw)',int((R.p_null<0.05).sum()),'p_bonf<0.05',int((R.p_bonf36<0.05).sum()))
# baseline overlap
print('--- baseline on event days ---')
for (X,W),ev in EV.items():
    dset=[D for D,_,_ in ev]
    on=bday.reindex(dset).fillna(0); off=bday[~bday.index.isin(dset)]
    # baseline trades entered after trigger on event day up to 16:00
    tt=[]
    for D,i,drop in ev:
        g=base[(base.eday==D)&(base.entry_time>ix[i])&(base.entry_time<ix[i]+pd.Timedelta(hours=4)) ] if 'eday' in base else None
        tt.append(g)
    g=pd.concat(tt) if tt else pd.DataFrame(columns=base.columns)
    gm=met(g)
    print((X,W),'event days',len(dset),'| baseline day P&L avg on event days',round(on.mean(),1),'vs other days',round(off.mean(),1),'| sum',round(on.sum()),'| trades entered within 4h after trigger:',gm)
    for w,gg in g.groupby('win'): print('    ',w,met(gg))
json.dump({str(k):[(D,int(i),float(dr)) for D,i,dr in v] for k,v in EV.items()},open('x9_events.json','w'))
