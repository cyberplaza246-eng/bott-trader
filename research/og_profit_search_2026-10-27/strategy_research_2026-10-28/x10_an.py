import pandas as pd,numpy as np,os,sys,json
import sr_x10_lib as L
from x10_variants import V
ET=L.ET
def load(p):
    d=pd.read_csv(p,parse_dates=['entry_time','exit_time'])
    for c in ('entry_time','exit_time'): d[c]=pd.to_datetime(d[c],utc=True)
    d['win']=np.where(d.entry_time<pd.Timestamp('2026-01-01',tz='UTC'),'train','y2026')
    d['AB']=np.where(d.entry_time<pd.Timestamp('2025-11-01',tz='UTC'),'A','B')
    d['xday']=d.exit_time.dt.tz_convert(ET).dt.strftime('%Y-%m-%d'); return d
def dayser(d): return d.groupby('xday').pnl.sum()
def pb(x,n=10000,seed=3):
    r=np.random.default_rng(seed); x=np.asarray(x,float)
    if len(x)==0: return 0.5
    return round(float((r.choice(x,(n,len(x))).sum(1)>0).mean()),3)
base=load('x10_trades_base.csv'); bd=dayser(base)
edges=pd.date_range(base.entry_time.min(),base.entry_time.max(),periods=7)
def fold_net(d): return np.array([d[(d.entry_time>=edges[i])&(d.entry_time<=edges[i+1])].pnl.sum() for i in range(6)])
bf=fold_net(base)
def row(name,suffix='',bdf=None,bds=None,bff=None):
    bdf=base if bdf is None else bdf; bds=bd if bds is None else bds; bff=bf if bff is None else bff
    d=load(f'x10_trades_{name}{suffix}.csv'); vd=dayser(d)
    j=pd.concat([vd,bds],axis=1,keys=['v','b']).fillna(0); dl=(j.v-j.b); idx=pd.to_datetime(dl.index)
    r=dict(name=name+suffix)
    for w,sel in (('all',None),('train','train'),('y2026','y2026')):
        dd=d if sel is None else d[d.win==sel]; m=L.met(dd); r.update({f'{w}_{k}':m[k] for k in ('n','net','pf','avg','dd')})
    r['d_all']=round(d.pnl.sum()-bdf.pnl.sum()); r['d_train']=round(d[d.win=='train'].pnl.sum()-bdf[bdf.win=='train'].pnl.sum()); r['d_2026']=round(d[d.win=='y2026'].pnl.sum()-bdf[bdf.win=='y2026'].pnl.sum())
    r['d_A']=round(d[d.AB=='A'].pnl.sum()-bdf[bdf.AB=='A'].pnl.sum()); r['d_B']=round(d[d.AB=='B'].pnl.sum()-bdf[bdf.AB=='B'].pnl.sum())
    r['pbet_all']=pb(dl.values); r['pbet_train']=pb(dl[idx<'2026-01-01'].values); r['pbet_2026']=pb(dl[idx>='2026-01-01'].values)
    f=fold_net(d)-bff; r['folds_better']=int((f>0).sum()); r['fold_d']=[round(x) for x in f]
    mo=d.groupby('month').pnl.sum(); r['mo_green']=int((mo>0).sum())
    return r
if __name__=='__main__':
    rows=[]
    for k,(g,sp,desc) in V.items():
        if k=='base': continue
        r=row(k); r['group']=g; r['desc']=desc; rows.append(r)
    R=pd.DataFrame(rows); R.to_csv('x10_results.csv',index=False)
    pd.set_option('display.width',330); pd.set_option('display.max_columns',50); pd.set_option('display.max_rows',200)
    print('BASE',L.met(base),{w:L.met(base[base.win==w]) for w in ('train','y2026')}, 'folds',bf.round().tolist())
    print(R[['name','group','all_n','all_net','all_pf','all_dd','d_all','d_train','d_2026','d_A','d_B','pbet_all','pbet_train','pbet_2026','folds_better']].to_string())
