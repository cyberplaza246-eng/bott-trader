import sys; sys.argv=['x']
exec(open('sr_x7d.py').read().split("R=pd.DataFrame(rows)")[0])
R=pd.DataFrame(rows).set_index('feat')
print(R.loc[[x for x in ['NQ','ES','QQQ','SPY','XLK','NVDA','AAPL','VIX','VIXY','DXY','GC','CL','ZN','TNX','RSP','IWM'] if x in R.index],['n_big','mean_big','mean_rest','t_diff','mA_big','mA_rest','mB_big','mB_rest']].round(1).to_string())
# composite: number of risk features in big state (past-only thr, q80), tercile test
feats=[c for c in dcols if c not in LEVELS]
cnt=np.zeros(len(bp)); 
for c in feats:
    col=DFs[c].values.astype(float); v,t=tab_move(col,minp=250,q=0.8,win=500); kb=(np.abs(v[ridx])>=t[ridx])&~np.isnan(t[ridx]); cnt+=kb
print('score mean',cnt.mean(),'quantiles',np.quantile(cnt,[.33,.5,.67]))
for lab,m in (('score>=median+',cnt>=np.median(cnt)+1),('score>=q67',cnt>=np.quantile(cnt,.67)),('score>=q80',cnt>=np.quantile(cnt,.8))):
    lo=~m; t=(bp[m].mean()-bp[lo].mean())/np.sqrt(bp[m].var()/m.sum()+bp[lo].var()/lo.sum())
    print(lab,'n_hi',int(m.sum()),'mean hi %.1f lo %.1f t %.2f | A hi %.1f lo %.1f | B hi %.1f lo %.1f | net hi A %.0f B %.0f lo A %.0f B %.0f'%(bp[m].mean(),bp[lo].mean(),t,bp[m&A].mean(),bp[lo&A].mean(),bp[m&B].mean(),bp[lo&B].mean(),bp[m&A].sum(),bp[m&B].sum(),bp[lo&A].sum(),bp[lo&B].sum()))
    # day-level paired bootstrap of "skip low-score trades" = -sum(lo pnl)
    dd=pd.DataFrame(dict(d=bdate,x=np.where(lo,-bp,0.0))).groupby('d').x.sum(); print('   skip-low-score: dnet %.0f, pboot(day) %.3f; dropped %d trades'%(dd.sum(),pboot(dd.values),lo.sum()))
# permutation of session mapping for composite q67
rng=np.random.default_rng(1); obs=None
sess_ids=np.unique(ridx); res=[]
colsd=[]; 
for c in feats:
    col=DFs[c].values.astype(float); v,t=tab_move(col,minp=250,q=0.8,win=500); colsd.append((np.abs(v)>=t)&~np.isnan(t))
BIG=np.array(colsd).sum(0)   # per DF row score
def tstat(score_by_row,perm=None):
    rr=ridx if perm is None else perm[ridx]; s=score_by_row[rr]; m=s>=np.quantile(s,.67); lo=~m
    return (bp[m].mean()-bp[lo].mean())/np.sqrt(bp[m].var()/m.sum()+bp[lo].var()/lo.sum())
real=tstat(BIG); lo_,hi_=ridx.min(),ridx.max(); null=[]
for rep in range(1000):
    perm=np.arange(len(BIG)); perm[lo_:hi_+1]=rng.permutation(np.arange(lo_,hi_+1)); null.append(tstat(BIG,perm))
null=np.array(null); print('composite t real %.2f ; null (session-shuffled) 95%% %.2f 99%% %.2f ; p=%.3f'%(real,np.quantile(null,.95),np.quantile(null,.99),(null>=real).mean()))