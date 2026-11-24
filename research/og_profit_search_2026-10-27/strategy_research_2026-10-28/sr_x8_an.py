import pandas as pd,numpy as np,json
from sr_x8_desc import *
from sr_x8 import V
base=load('x8_trades_base.csv')
def dayser(d): return d.groupby(d.exit_time.dt.tz_convert(ET).dt.strftime('%Y-%m-%d')).pnl.sum()
def pb(delta,n=10000,seed=3):
    r=np.random.default_rng(seed); x=np.asarray(delta,float); return round(float((r.choice(x,(n,len(x))).sum(1)>0).mean()),3)
bd=dayser(base); t0=base.entry_time.min(); t1=base.entry_time.max()
edges=pd.date_range(t0,t1,periods=7)
def folds(d):
    out=[]
    for i in range(6):
        b=base[(base.entry_time>=edges[i])&(base.entry_time<=edges[i+1])].pnl.sum(); v=d[(d.entry_time>=edges[i])&(d.entry_time<=edges[i+1])].pnl.sum(); out.append(v-b)
    return out
rows=[]
for nm,(sp,fl,desc) in V.items():
    d=load(f'x8_trades_{nm}.csv'); vd=dayser(d); j=pd.concat([vd,bd],axis=1,keys=['v','b']).fillna(0); dl=(j.v-j.b)
    r=dict(name=nm,desc=desc)
    for w in ('all','train','y2026'):
        dd=d if w=='all' else d[d.win==w]; m=met(dd); r.update({f'{w}_{k}':v for k,v in m.items() if k in('n','net','pf','avg','dd')})
    ds=dl.copy(); idx=pd.to_datetime(ds.index)
    r['dnet_all']=round(dl.sum()); r['dnet_train']=round(dl[idx<'2026-01-01'].sum()); r['dnet_2026']=round(dl[idx>='2026-01-01'].sum())
    r['p_better_all']=pb(dl.values); r['p_better_2026']=pb(dl[idx>='2026-01-01'].values); r['p_better_train']=pb(dl[idx<'2026-01-01'].values)
    f=folds(d); r['folds_better']=int(sum(x>0 for x in f)); r['fold_d']=[round(x) for x in f]
    rows.append(r)
R=pd.DataFrame(rows); R.to_csv('x8_rules.csv',index=False)
pd.set_option('display.width',300); pd.set_option('display.max_columns',40)
print(R.drop(columns=['desc','fold_d']).to_string())
print(R[['name','fold_d']].to_string())
# FOMC-day cond: trades dropped by each rule by window
# break the dropped trade P&L? delta equals -dropped(+ re-routed). show baseline trades on FOMC days 13:15-16:00
d=base; g=d[d.fomc&(d.emin>=13*60+15)&(d.emin<16*60)]; print('base FOMC 13:15-16:00 entries',met(g),{w:met(x) for w,x in g.groupby('win')})
g=d[(d.emin>=13*60+15)&(d.emin<16*60)]; print('base all-day 13:15-16:00 entries',met(g),{w:met(x) for w,x in g.groupby('win')})
g=d[(d.emin>=13*60+15)&(d.emin<18*60)]; print('base all-day 13:15-18:00 entries',met(g),{w:met(x) for w,x in g.groupby('win')})
# bootstrap CI on afternoon per-trade vs rest
r=np.random.default_rng(1)
for lab,mask in (('13:15-18:00',(d.emin>=795)&(d.emin<1080)),('14:00-16:00',(d.emin>=840)&(d.emin<960)),('FOMC all',d.fomc)):
    a=d[mask].pnl.values; b=d[~mask].pnl.values
    diffs=[r.choice(a,len(a)).mean()-r.choice(b,len(b)).mean() for _ in range(5000)]
    print(lab,'mean',round(a.mean(),1),'rest',round(b.mean(),1),'95% CI diff',np.round(np.percentile(diffs,[2.5,97.5]),1),'P(worse)',round(float(np.mean(np.array(diffs)<0)),3))
