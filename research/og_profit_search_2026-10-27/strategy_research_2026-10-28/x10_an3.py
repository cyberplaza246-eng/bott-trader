import pandas as pd,numpy as np,json,ast
import sr_x10_lib as L
from x10_an import *
from scipy import stats
pd.set_option('display.width',330); pd.set_option('display.max_columns',50); pd.set_option('display.max_rows',200)
R=pd.read_csv('x10_results.csv'); R['fold_d']=R.fold_d.map(ast.literal_eval)
Rd=R[~R.group.isin(['G7'])].copy()
N=len(Rd); print('variants in family (excl base, excl post-hoc combos):',N)
# ---------- A. frozen-forward + rank persistence
print('\n== FROZEN FORWARD ==')
for a,b_,lab in (('d_A','d_B','A (2024-11..2025-10) -> B (2025-11..2026-09)'),('d_train','d_2026','train (<2026) -> 2026')):
    s=Rd.sort_values(a,ascending=False)
    top=s.head(1).iloc[0]; print(lab,'| best on first window:',top['name'],f'{a}={top[a]} -> {b_}={top[b_]}')
    t3=s.head(3); print('   top3',list(zip(t3.name,t3[a],t3[b_])),'sum later window',int(t3[b_].sum()))
    pos=s[s[a]>0]; print('   positive-in-first-window variants',len(pos),'of',N,'; of those positive in second window',int((pos[b_]>0).sum()),'; sum later',int(pos[b_].sum()))
    rho=stats.spearmanr(Rd[a],Rd[b_]); print('   Spearman first vs later delta across all variants',round(rho.statistic,2),'p',round(rho.pvalue,3))
# ---------- B. proper expanding walk-forward over the variant family (full-sim fold deltas)
print('\n== EXPANDING WALK-FORWARD over variant family (choose best on earlier folds, score next fold) ==')
F=np.array(Rd.fold_d.tolist()); names=Rd.name.values; tot=0; log=[]
for k in range(1,6):
    cum=F[:,:k].sum(1); j=int(np.argmax(cum))
    if cum[j]<=0: log.append((k+1,'none',0)); continue
    log.append((k+1,names[j],int(F[j,k]))); tot+=F[j,k]
print(log,'total OOS delta vs baseline',int(tot),' (folds 2-6)')
# also: always-apply the single best-on-fold1-5 ... baseline reference: sum of base net folds 2-6
print('baseline folds 2-6 net',int(bf[1:].sum()))
# ---------- C. cell-pruning walk-forward on baseline trades (approximate: removed trades not re-routed)
b=pd.read_pickle('x10_base.pkl'); b['xday']=b.exit_time.dt.tz_convert(ET).dt.strftime('%Y-%m-%d'); b['fold']=sum((b.entry_time>edges[i]).astype(int) for i in range(1,6))
nb=b[b.seg.str[0].isin(['2','3','4'])].copy(); nb['cell']=nb.seg.str[0]+'|'+nb.sleeve+'|'+nb.side
print('\n== CELL-PRUNING WALK-FORWARD (approx) ==')
for pol in ('avg<0 & n>=15','t<-1 & n>=15','session avg<0'):
    tot=0; rows=[]
    for k in range(1,6):
        tr=nb[nb.fold<k]; te=nb[nb.fold==k]
        if pol=='session avg<0': g=tr.groupby('seg').pnl.agg(['mean','size']); bad=set(g[(g['mean']<0)&(g['size']>=15)].index); rem=te[te.seg.isin(bad)]
        else:
            g=tr.groupby('cell').pnl.agg(['mean','std','size']); g['t']=g['mean']/(g['std']/np.sqrt(g['size'])); 
            bad=set(g[(g['size']>=15)&((g['mean']<0) if pol.startswith('avg') else (g.t<-1))].index); rem=te[te.cell.isin(bad)]
        d=-rem.pnl.sum(); tot+=d; rows.append((k+1,len(bad),len(rem),round(d)))
    print(pol,rows,'total OOS delta',round(tot))
# ---------- D. null: random removal of k night trades (same size) -> screen pass rate and empirical p
nt=b[b.seg.str[0].isin(['3','4'])].reset_index(drop=True)
days=sorted(set(b.xday)); di={d:i for i,d in enumerate(days)}; nt['di']=nt.xday.map(di); ndays=len(days)
dtrain=np.array([pd.Timestamp(d)<pd.Timestamp('2026-01-01') for d in days])
rng=np.random.default_rng(11); BI=rng.integers(0,ndays,(300,ndays))
pn=nt.pnl.values; trn=(nt.win=='train').values; fo=nt.fold.values; dix=nt.di.values
def null(k,ndraw=1500):
    k=int(max(1,min(k,len(nt)))); pas=0; ds=[]
    for _ in range(ndraw):
        s=rng.choice(len(nt),k,replace=False); d=-pn[s]
        dt=d[trn[s]].sum(); d26=d[~trn[s]].sum(); ds.append(d.sum())
        if dt<=0 or d26<=0: continue
        fb=np.bincount(fo[s],weights=d,minlength=6); 
        if (fb>0).sum()<4: continue
        dv=np.bincount(dix[s],weights=d,minlength=ndays)
        if (dv[BI].sum(1)>0).mean()>=0.9: pas+=1
    return pas/ndraw,np.array(ds)
rows=[]; cache={}
for _,r in Rd.iterrows():
    if r.group in ('G6',) or r['name'].startswith(('off_evening','off_overnight_all','off_night_all','off_18_0930_all')) and False: pass
    k=max(1,int(round(2473-r.all_n)))
    if k not in cache: cache[k]=null(k)
    pr,ds=cache[k]; rows.append((r['name'],k,round(pr,4),round(float((ds>=r.d_all).mean()),4)))
NR=pd.DataFrame(rows,columns=['name','k_removed','null_screen_pass_rate','emp_p_delta']); 
R2=R.merge(NR,on='name',how='left'); R2.to_csv('x10_results2.csv',index=False)
print('\n== NULL ==')
print('expected number of screen passes by chance among',len(NR),'variants:',round(NR.null_screen_pass_rate.sum(),2))
scr=(R2.d_train>0)&(R2.d_2026>0)&(R2.folds_better>=4)&(R2.pbet_all>=0.9)
print('observed screen passes:',R2[scr&~R2.group.isin(['G7'])].name.tolist())
print(R2[R2.group!='G7'].sort_values('emp_p_delta')[['name','d_all','d_train','d_2026','folds_better','pbet_all','k_removed','null_screen_pass_rate','emp_p_delta']].head(12).to_string())
best=R2[R2.group!='G7'].emp_p_delta.min(); print('best emp p',best,' Sidak-adjusted over',len(NR),'variants:',round(1-(1-max(best,1/1500))**len(NR),3))


