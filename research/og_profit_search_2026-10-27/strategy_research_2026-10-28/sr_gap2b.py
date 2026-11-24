import sys,io,json,pickle
src=open('sr_gap2.py',encoding='utf-8-sig').read().split('# ---- run ----')[0]
src=src.replace("side=-int(np.sign(mv)); ex=a+Hm","side=MIR*-int(np.sign(mv)); ex=a+Hm").replace("else: i_in=a\n","else: i_in=a+DLY\n")
MIR=1;DLY=0
import contextlib
with contextlib.redirect_stdout(io.StringIO()): exec(src,globals())
T=pickle.load(open('gap2_trades.pkl','rb'))
span_m=(base.entry_time.max()-base.entry_time.min()).days/30.44
def daily(df): return df.groupby(df.exit_time.dt.tz_convert(ET).dt.date).pnl.sum()
bd=daily(base)
def corr(x):
    u=bd.index.union(x.index); return round(float(np.corrcoef(bd.reindex(u).fillna(0),x.reindex(u).fillna(0))[0,1]),3)
out={}
# --- A. delay stress
cands={'ny_q75_H90':lambda:run_ev(EV['ny'],'q',.75,90,1),'ny_q65_H90':lambda:run_ev(EV['ny'],'q',.65,90,1),'ny_q50_H90':lambda:run_ev(EV['ny'],'q',.5,90,1),
 'ny_q50_H180':lambda:run_ev(EV['ny'],'q',.5,180,1),'ny_abs40_H180':lambda:run_ev(EV['ny'],'abs',40,180,1),'ny_atr12_H180':lambda:run_ev(EV['ny'],'atr',.12,180,1),
 'm60_q75_H25':lambda:run_ev(EV['m60'],'q',.75,25,1)}
dl={}
for k,f in cands.items():
    dl[k]={}
    for d_ in(0,1,3):
        DLY=d_; r=dfm(f()); dl[k][d_]=(round(r.pnl.sum()),round(r[r.entry_time<cut].pnl.sum()),round(r[r.entry_time>=cut].pnl.sum()))
    DLY=0
out['delay']=dl; print('delay',json.dumps(dl),flush=True)
# --- B. marginal trades vs q75_H90 reference
ref=T['P1_ny_q75_H90']; 
def marg(name):
    x=T[name]; m=x[~x.entry_time.isin(ref.entry_time)]
    s=st(m) if len(m)>=3 else {}
    return dict(n=len(m),net=round(m.pnl.sum()),pf=s.get('pf'),train=s.get('train'),y26=s.get('y26'),folds=s.get('folds_pos'),boot_p=s.get('boot_p'))
mg={n:marg(n) for n in('P1_ny_q65_H90','P1_ny_q50_H90','P1_ny_q50_H180','P1_ny_q75_H180','P1_ny_abs40_H180','P1_ny_atr12_H180','P1_ny_abs30_H180')}
out['marginal']=mg; print('marginal',json.dumps(mg,default=str),flush=True)
# --- C. portfolios
def port(names,label):
    parts=[base[['entry_time','exit_time','pnl']]]+[T[n][['entry_time','exit_time','pnl']] for n in names]
    x=pd.concat(parts).sort_values('exit_time').reset_index(drop=True); s=st(x)
    sl=pd.concat([T[n][['entry_time','exit_time','pnl']] for n in names]).sort_values('exit_time') if names else None
    d=dict(label=label,n=s['n'],tpm=round(s['n']/span_m,1),net=s['net'],pf=s['pf'],dd=s['dd'],train=s['train'],y26=s['y26'],wf=s['wf'],folds=s['folds_pos'],green=None)
    if names:
        d['sleeve_n']=len(sl); d['sleeve_tpm']=round(len(sl)/span_m,1); d['sleeve_net']=round(sl.pnl.sum()); d['corr']=corr(daily(sl))
    # months green
    mm=x.groupby(x.exit_time.dt.tz_convert(ET).dt.to_period('M')).pnl.sum(); d['green']=f'{(mm>0).sum()}/{len(mm)}'
    # paired boot of daily diff (sleeve daily) vs base
    if names:
        dd_=daily(sl).values; b=np.array([rng.choice(dd_,len(dd_)).sum() for _ in range(3000)]); d['boot_p']=round(float((b>0).mean()),2)
    return d
P_=[port([],'BASE only'),
 port(['P1_ny_q75_H90'],'base+gap q75 H90 (ref)'),
 port(['P1_ny_q65_H90'],'base+gap q65 H90'),
 port(['P1_ny_q50_H90'],'base+gap q50 H90'),
 port(['P1_ny_q50_H180'],'base+gap q50 H180'),
 port(['P1_ny_q75_H180'],'base+gap q75 H180'),
 port(['P1_ny_abs40_H180'],'base+gap abs40pt H180'),
 port(['P1_ny_atr12_H180'],'base+gap 0.12ATR H180'),
 port(['P4_ny_q50_conf1m_H90'],'base+gap q50 1m-confirm H90'),
 port(['P1_ny_q75_H90','P2_m60_q75_H25'],'base+gap q75H90+1530 fade'),
 port(['P1_ny_q65_H90','P2_m60_q75_H25'],'base+gap q65H90+1530 fade'),
 port(['P1_ny_abs40_H180','P2_m60_q75_H25'],'base+gap abs40 H180+1530 fade'),
 port(['P1_ny_q75_H90','P2_m60_q75_H25','P2_asia2_q75_H120'],'base+gap q75H90+1530+asia (asia = one-window)'),
]
out['portfolios']=P_
for p in P_: print(json.dumps(p),flush=True)
# --- D. post-hoc continuation (mirror) of the strongly negative fades -- NOT pre-declared, selected after seeing losses
MIR=-1; mr={}
for nm,f in {'weekend_q0_H120':lambda tk:run_ev(EV['gx_we'],'q',0,120,tk,minpast=10),'weekend_q50_H120':lambda tk:run_ev(EV['gx_we'],'q',.5,120,tk,minpast=10),
             'gxwd_q50_H60':lambda tk:run_ev(EV['gx_wd'],'q',.5,60,tk),'lon3_q50_H60':lambda tk:run_ev(EV['lon3'],'q',.5,60,tk),
             'pl60_q75_H60':lambda tk:run_ev(EV['pl60'],'q',.75,60,tk)}.items():
    r=dfm(f(1)); s=st(r); s['slip4']=round(dfm(f(4)).pnl.sum()); s['tpm']=round(len(r)/span_m,1); s['corr']=corr(daily(r)); mr[nm]=s; print('MIRROR',nm,s,flush=True)
MIR=1
out['mirror']=mr
json.dump(out,open('gap2_stage2.json','w'),indent=1,default=str)
print('DONE2',flush=True)
