import sys; sys.argv=['x']
exec(open('sr_x7d.py').read().split("R=pd.DataFrame(rows)")[0])
rowsK=[]; names=[]
for c in dcols:
    col=DFs[c].values.astype(float)
    if c in LEVELS: v,t=tab_lvl(col,minp=250,q=0.8,win=500); k=(v>=t)&~np.isnan(t)
    else: v,t=tab_move(col,minp=250,q=0.8,win=500); k=(np.abs(v)>=t)&~np.isnan(t)
    rowsK.append(k); names.append(c)
K=np.array(rowsK)   # F x rows
def maxt(perm=None):
    rr=ridx if perm is None else perm[ridx]; ts=[]
    for f in range(K.shape[0]):
        m=K[f][rr]
        if m.sum()<150 or (~m).sum()<150: continue
        ts.append((bp[m].mean()-bp[~m].mean())/np.sqrt(bp[m].var()/m.sum()+bp[~m].var()/(~m).sum()))
    return np.array(ts)
real=maxt(); rng=np.random.default_rng(3); lo_,hi_=ridx.min(),ridx.max(); mx=[];n2=[]
for rep in range(500):
    perm=np.arange(K.shape[1]); perm[lo_:hi_+1]=rng.permutation(np.arange(lo_,hi_+1)); t=maxt(perm); mx.append(t.max()); n2.append((t>=2).sum())
mx=np.array(mx); n2=np.array(n2)
print('real: max t %.2f, #t>=2 = %d of %d'%(real.max(),(real>=2).sum(),len(real)))
print('null: max t median %.2f 95%% %.2f 99%% %.2f ; P(max>=real) = %.3f ; #t>=2 mean %.1f 95%% %.0f ; P(#>=real count)=%.3f'%(np.median(mx),np.quantile(mx,.95),np.quantile(mx,.99),(mx>=real.max()).mean(),n2.mean(),np.quantile(n2,.95),(n2>=(real>=2).sum()).mean()))