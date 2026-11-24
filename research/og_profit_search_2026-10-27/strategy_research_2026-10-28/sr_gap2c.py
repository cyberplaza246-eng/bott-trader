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
out={}
MIR=-1
W={}
for q in(0,.5,.75):
    for H in(60,120,180):
        f=lambda tk:run_ev(EV['gx_we'],'q',q,H,tk,minpast=10)
        r=dfm(f(1)); s=st(r); s['slip4']=round(dfm(f(4)).pnl.sum()); DLY=1; r1=dfm(f(1)); s['delay1']=round(r1.pnl.sum()); DLY=0
        p=np.sort(r.pnl.values)[::-1]; s['top3']=round(p[:3].sum()); W[f'wk_cont_q{int(q*100)}_H{H}']=s; print('WKC',f'q{q} H{H}',{k:s[k] for k in('n','net','pf','dd','train','y26','folds_pos','boot_p','slip4','delay1','top3')},flush=True)
        if q==0 and H==120: wk=r
MIR=1
out['wk']=W
ref=T['P1_ny_q75_H90']; m60=T['P2_m60_q75_H25']
def port(parts,label):
    x=pd.concat([base[['entry_time','exit_time','pnl']]]+[p[['entry_time','exit_time','pnl']] for p in parts]).sort_values('exit_time'); s=st(x)
    sl=pd.concat([p[['entry_time','exit_time','pnl']] for p in parts]); d=daily(sl); u=bd.index.union(d.index)
    mm=x.groupby(x.exit_time.dt.tz_convert(ET).dt.to_period('M')).pnl.sum()
    return dict(label=label,n=s['n'],net=s['net'],pf=s['pf'],dd=s['dd'],train=s['train'],y26=s['y26'],wf=s['wf'],folds=s['folds_pos'],green=f'{(mm>0).sum()}/{len(mm)}',sleeve_tpm=round(len(sl)/span_m,1),corr=round(float(np.corrcoef(bd.reindex(u).fillna(0),d.reindex(u).fillna(0))[0,1]),3))
P_=[port([ref,wk],'base+gap q75H90+weekend continuation q0H120'),port([ref,T['P1_ny_abs40_H180'],wk][::2],'base+gap q75H90+weekend cont (dup check)'),
    port([T['P1_ny_abs40_H180'],wk],'base+gap abs40 H180+weekend cont'),port([ref,wk,m60],'base+gap q75H90+weekend cont+1530 fade')]
for p in P_: print(json.dumps(p),flush=True)
out['ports']=P_
json.dump(out,open('gap2_stage3.json','w'),indent=1,default=str); print('DONE3')
