"""Q3: does any symbol act as a filter/direction check on (a) the q75/H90 NY gap fade, (b) the 3R baseline trades? Permutation null. BACKTEST ONLY."""
from sr_x import *
import warnings; warnings.filterwarnings('ignore')
OV=pd.read_csv('x7_ovn.csv',index_col=0,parse_dates=True); DF=pd.read_csv('x7_daily.csv',index_col=0,parse_dates=True)
zs=(OV==0).sum()/OV.notna().sum(); OVS=[s for s in OV.columns if s not in ('MNQ','NQ') and zs[s]<0.5]
mn=OV['MNQ']
def expbeta(x,m,minn=40):
    b=np.full(len(x),np.nan); sxy=sxx=0.;n=0
    for k in range(len(x)):
        if n>=minn and sxx>0: b[k]=sxy/sxx
        if not(np.isnan(x[k]) or np.isnan(m[k])): sxy+=x[k]*m[k]; sxx+=m[k]*m[k]; n+=1
    return b
# ---------- feature tables: value + past-only threshold, keyed by row ----------
def tab_move(v,minp=40,q=0.7,win=None):
    v=np.asarray(v,float); thr=np.full(len(v),np.nan); past=[]
    for k in range(len(v)):
        if len(past)>=minp: thr[k]=np.quantile(past[-win:] if win else past,q)
        if not np.isnan(v[k]): past.append(abs(v[k]))
    return v,thr
def tab_lvl(v,minp=40,q=0.5,win=None):
    v=np.asarray(v,float); thr=np.full(len(v),np.nan); past=[]
    for k in range(len(v)):
        if len(past)>=minp: thr[k]=np.quantile(past[-win:] if win else past,q)
        if not np.isnan(v[k]): past.append(v[k])
    return v,thr
LEVELS=['VIXlvl','VIXterm','VIX9D_VIX']
def filt_masks(val,thr,kind,side):
    """returns dict filtername->keep mask (bool array); NaN -> not kept (cannot evaluate)"""
    ok=~np.isnan(val)&~np.isnan(thr); out={}
    if kind=='lvl':
        out['keep_hi']=ok&(val>=thr); out['keep_lo']=ok&(val<thr)
    else:
        big=ok&(np.abs(val)>=thr); calm=ok&(np.abs(val)<thr)
        out['keep_calm']=calm; out['keep_big']=big
        sg=np.sign(val); out['keep_with_trade']=ok&(sg==side); out['keep_against_trade']=ok&(sg==-side)
    return out
def evaluate(pnl,dkey,inA,inB,tab,side,names,kinds,perm=None,need=None,detail=False):
    """tab: dict name->(val_by_row,thr_by_row); dkey: row index per trade (int, -1 missing)"""
    rows=[]; netA0=pnl[inA].sum(); netB0=pnl[inB].sum(); n0=len(pnl)
    for nm in names:
        val,thr=tab[nm]
        kk=dkey if perm is None else np.where(dkey>=0,perm[np.clip(dkey,0,len(perm)-1)],-1)
        v=np.where(kk>=0,val[np.clip(kk,0,len(val)-1)],np.nan); t=np.where(kk>=0,thr[np.clip(kk,0,len(thr)-1)],np.nan)
        for fn,keep in filt_masks(v,t,kinds[nm],side).items():
            dl=np.where(keep,0.0,-pnl)   # delta of filter vs unfiltered
            nk=int(keep.sum())
            if nk<need: continue
            dA=dl[inA].sum(); dB=dl[inB].sum(); sd=dl.std()
            tt=dl.mean()/(sd/np.sqrt(len(dl))) if sd>0 else 0
            rows.append(dict(feat=nm,filt=fn,kept=nk,net=pnl[keep].sum(),dnet=dl.sum(),dA=dA,dB=dB,t=tt))
    return rows
def ispass(r): return r['dA']>0 and r['dB']>0 and r['t']>=1.64
def run_null(pnl,dkey,inA,inB,tab,side,names,kinds,nperm,need,nrows,seed=5):
    rng=np.random.default_rng(seed); cnt=[]; mx=[]
    lo=dkey[dkey>=0].min(); hi=dkey.max()
    for rep in range(nperm):
        perm=np.arange(nrows); sub=np.arange(lo,hi+1); perm[lo:hi+1]=rng.permutation(sub)
        rr=evaluate(pnl,dkey,inA,inB,tab,side,names,kinds,perm=perm,need=need)
        cnt.append(sum(ispass(r) for r in rr)); mx.append(max([r['t'] for r in rr]) if rr else 0)
    return np.array(cnt),np.array(mx)
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
# =================== PART A: gap fade ===================
rows=ny2(0.75,90,stop=300); G=dfm(rows); G['dd']=pd.to_datetime(G.d); pnl=G.pnl.values
gside=np.array([-int(np.sign(r['g'])) for r in rows]); print('gap fade trades',len(G),'net',round(pnl.sum()))
inA=((G.dd>=A0)&(G.dd<A1)).values; inB=((G.dd>=B0)&(G.dd<B1)).values
# overnight tables by OV row (index = day)
tab={};kinds={};names=[]
for s in OVS:
    x=OV[s].values.astype(float); 
    tab['ov_'+s]=tab_move(x,q=0.7); kinds['ov_'+s]='mv'; names.append('ov_'+s)
    b=expbeta(x,mn.values); xr=x-b*mn.values
    tab['res_'+s]=tab_move(xr,q=0.7); kinds['res_'+s]='mv'; names.append('res_'+s)
# daily prior-session features -> map to OV rows via asof(prior business day)
DFs=DF[DF.index>='2019-01-01']
dcols=[c for c in DFs.columns if DFs[c].notna().sum()>400]
ovidx=OV.index; 
def dailyrow(dts):  # last completed session strictly before D at open => asof(D-1 day)
    return DFs.index.searchsorted(dts-pd.Timedelta(days=1),side='right')-1
rowmap=dailyrow(ovidx)
for c in dcols:
    col=DFs[c].values.astype(float)
    if c in LEVELS: v,t=tab_lvl(col,minp=250,q=0.5,win=500); kinds['d_'+c]='lvl'
    else: v,t=tab_move(col,minp=250,q=0.7,win=500); kinds['d_'+c]='mv'
    tab['d_'+c]=(v[rowmap],t[rowmap]); names.append('d_'+c)
# trade -> OV row
pos_ov={d:i for i,d in enumerate(ovidx)}; dkey=np.array([pos_ov.get(pd.Timestamp(d),-1) for d in G.d])
need=40; USE=int(22.6*4); R1=pd.DataFrame(evaluate(pnl,dkey,inA,inB,tab,gside,names,kinds,need=need))
R1['pass']=[ispass(r) for r in R1.to_dict('records')]; R1['usable']=R1.kept>=USE
print('GAPFADE variants (>=%d trades kept):'%need,len(R1),' pass(real):',int(R1['pass'].sum()),'pass&>=4/mo:',int((R1['pass']&R1.usable).sum()),'| unfiltered A %.0f B %.0f'%(pnl[inA].sum(),pnl[inB].sum()))
cn,mx=run_null(pnl,dkey,inA,inB,tab,gside,names,kinds,200,need,len(ovidx))
print('NULL pass count: mean %.1f 95%% %.0f max %d ; null max t: median %.2f 95%% %.2f ; real max t %.2f'%(cn.mean(),np.quantile(cn,.95),cn.max(),np.median(mx),np.quantile(mx,.95),R1.t.max()))
R1.to_csv('x7c_gapfade.csv',index=False)
print(R1.sort_values('t',ascending=False).head(14).round(2).to_string(index=False))
# slip4/delay3 + bootstrap on the top 6
print('--- top candidates robustness (slip4 / delay3 filtered nets vs unfiltered) ---')
r4=dfm(ny2(0.75,90,stop=300,tk=4)); r3=dfm(ny2(0.75,90,stop=300,dly=3))
for _,r in R1.sort_values('t',ascending=False).head(6).iterrows():
    val,thr=tab[r.feat]; v=np.where(dkey>=0,val[np.clip(dkey,0,len(val)-1)],np.nan); t=np.where(dkey>=0,thr[np.clip(dkey,0,len(thr)-1)],np.nan)
    keep=filt_masks(v,t,kinds[r.feat],gside)[r.filt]
    dl=pd.Series(np.where(keep,0.0,-pnl)); p=pboot(dl.values)
    print(r.feat,r.filt,'kept',r.kept,'dnet',round(r.dnet),'pboot',p,'| slip4 kept net %.0f vs unf %.0f | delay3 kept %.0f vs unf %.0f'%(r4.pnl.values[keep].sum(),r4.pnl.sum(),r3.pnl.values[keep].sum(),r3.pnl.sum()))
# =================== PART B: baseline 3R trades ===================
b=P.base.copy(); b['et']=b.entry_time.dt.tz_convert(ET); bpnl=b.pnl.values; bside=b.dir.values.astype(int)
bdate=b.et.dt.tz_localize(None).dt.normalize()
sessd=np.where(b.et.dt.hour>=16,bdate,bdate-pd.Timedelta(days=1))   # last completed US session date
ridx=DFs.index.searchsorted(pd.DatetimeIndex(sessd),side='right')-1
tabB={};kindsB={};namesB=[]
for c in dcols:
    col=DFs[c].values.astype(float)
    if c in LEVELS: tabB['d_'+c]=tab_lvl(col,minp=250,q=0.5,win=500); kindsB['d_'+c]='lvl'
    else: tabB['d_'+c]=tab_move(col,minp=250,q=0.8,win=500); kindsB['d_'+c]='mv'
    namesB.append('d_'+c)
bA=((bdate>=A0)&(bdate<A1)).values; bB=((bdate>=B0)&(bdate<B1)).values
keyB=ridx; needB=int(0.45*len(bpnl))
R2=pd.DataFrame(evaluate(bpnl,keyB,bA,bB,tabB,bside,namesB,kindsB,need=needB)); R2['pass']=[ispass(r) for r in R2.to_dict('records')]
print('\nBASELINE variants (>=45%% trades kept):',len(R2),' pass(real):',int(R2['pass'].sum()),'| baseline A %.0f B %.0f'%(bpnl[bA].sum(),bpnl[bB].sum()))
cn2,mx2=run_null(bpnl,keyB,bA,bB,tabB,bside,namesB,kindsB,120,needB,len(DFs),seed=9)
print('NULL pass count: mean %.1f 95%% %.0f max %d ; null max t median %.2f 95%% %.2f ; real max t %.2f'%(cn2.mean(),np.quantile(cn2,.95),cn2.max(),np.median(mx2),np.quantile(mx2,.95),R2.t.max()))
R2.to_csv('x7c_baseline.csv',index=False)
print('kept distribution',R2.kept.describe().round(0).to_dict()); print(R2.sort_values('t',ascending=False).head(14).round(2).to_string(index=False))
for _,r in R2.sort_values('t',ascending=False).head(5).iterrows():
    val,thr=tabB[r.feat]; v=np.where(keyB>=0,val[np.clip(keyB,0,len(val)-1)],np.nan); t=np.where(keyB>=0,thr[np.clip(keyB,0,len(thr)-1)],np.nan)
    keep=filt_masks(v,t,kindsB[r.feat],bside)[r.filt]; dd_=pd.DataFrame(dict(d=bdate,x=np.where(keep,0.0,-bpnl))).groupby('d').x.sum()
    print(r.feat,r.filt,'kept',r.kept,'dnet',round(r.dnet),'pboot(day)',pboot(dd_.values))