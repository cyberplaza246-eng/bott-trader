"""Baseline: does the baseline earn more on big-move days of any symbol (prior session |move|>= past q80)? (info only; live is 1 contract). BACKTEST ONLY."""
import sys; sys.argv=['x']
exec(open('sr_x7c.py').read().split('# =================== PART A')[0])
exec("DFs=DF[DF.index>='2019-01-01']; dcols=[c for c in DFs.columns if DFs[c].notna().sum()>400]")
b=P.base.copy(); b['et']=b.entry_time.dt.tz_convert(ET); bp=b.pnl.values
bdate=b.et.dt.tz_localize(None).dt.normalize(); sessd=np.where(b.et.dt.hour>=16,bdate,bdate-pd.Timedelta(days=1))
ridx=DFs.index.searchsorted(pd.DatetimeIndex(sessd),side='right')-1
A=((bdate>=A0)&(bdate<A1)).values; B=((bdate>=B0)&(bdate<B1)).values
rows=[]
def rowstat(name,kb):
    kb=np.asarray(kb,bool)
    if kb.sum()<150 or (~kb).sum()<150: return
    pa,pb=bp[kb],bp[~kb]; t=(pa.mean()-pb.mean())/np.sqrt(pa.var()/len(pa)+pb.var()/len(pb))
    rows.append(dict(feat=name,n_big=int(kb.sum()),mean_big=pa.mean(),mean_rest=pb.mean(),t_diff=t,netA_big=bp[kb&A].sum(),netB_big=bp[kb&B].sum(),mA_big=bp[kb&A].mean(),mA_rest=bp[~kb&A].mean(),mB_big=bp[kb&B].mean(),mB_rest=bp[~kb&B].mean()))
for c in dcols:
    col=DFs[c].values.astype(float)
    if c in LEVELS: v,t=tab_lvl(col,minp=250,q=0.8,win=500); kb=(v[ridx]>=t[ridx])
    else: v,t=tab_move(col,minp=250,q=0.8,win=500); kb=(np.abs(v[ridx])>=t[ridx])
    rowstat(c,kb&~np.isnan(t[ridx]))
R=pd.DataFrame(rows); R['same_sign_AB']=((R.mA_big-R.mA_rest)*(R.mB_big-R.mB_rest)>0)
R.to_csv('x7d_baseline_bigday.csv',index=False); pd.set_option('display.width',250)
print(len(R),'features; share with t_diff>0:',(R.t_diff>0).mean().round(2),'| |t|>=2:',(R.t_diff.abs()>=2).sum(),'| t>=2 & same sign in A and B:',((R.t_diff>=2)&R.same_sign_AB).sum())
print(R.sort_values('t_diff',ascending=False).head(10).round(2).to_string(index=False))
print(R.sort_values('t_diff').head(5).round(2).to_string(index=False))
print('overall mean pnl/trade %.2f'%bp.mean())