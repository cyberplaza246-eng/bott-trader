"""Q1: does the overnight move (16:00 ET -> 09:00 ET) in each symbol predict MNQ's next 30/60/120 min from the 09:30 open? Frozen sign on window A, scored on B, permutation null. BACKTEST ONLY."""
from sr_x import *
import warnings; warnings.filterwarnings('ignore')
OV=pd.read_csv('x7_ovn.csv',index_col=0,parse_dates=True)
zs=(OV==0).sum()/OV.notna().sum(); SY=[s for s in OV.columns if s not in ('MNQ','NQ') and zs[s]<0.5]
print('symbols',len(SY),SY)
dts=[d.strftime('%Y-%m-%d') for d in OV.index]
av=np.array([sess[d]['a'] if d in sess else -1 for d in dts])
Hs=(30,60,120); DLY=(0,3)
def ycalc(H,dly):
    y=np.full(len(dts),np.nan)
    for k,a in enumerate(av):
        if a>=0 and ok_dt(a,a+H,H) and ok_dt(a,a+dly,dly): y[k]=o[a+H]-o[a+dly]
    return y
Y={(H,d):ycalc(H,d) for H in Hs for d in DLY}
dd=pd.to_datetime(dts); inA=(dd>=A0)&(dd<A1); inB=(dd>=B0)&(dd<B1); B1h=inB&(dd<'2026-05-01'); B2h=inB&(dd>='2026-05-01')
mn=OV['MNQ'].values
def expbeta(x,m,minn=40):
    b=np.full(len(x),np.nan); sxy=sxx=0.;n=0
    for k in range(len(x)):
        if n>=minn and sxx>0: b[k]=sxy/sxx
        if not(np.isnan(x[k]) or np.isnan(m[k])): sxy+=x[k]*m[k]; sxx+=m[k]*m[k]; n+=1
    return b
cols=[];XR=[];XS=[];meta=[]
for s in SY+['MNQ']:
    x=OV[s].values.astype(float)
    for kind in (('raw','res') if s!='MNQ' else ('raw',)):
        if kind=='res': b=expbeta(x,mn); xv=x-b*mn
        else: xv=x.copy()
        # strong = |x|>= expanding q60 of past |x|
        st_=np.zeros(len(xv),bool); past=[]
        for k in range(len(xv)):
            if len(past)>=40 and not np.isnan(xv[k]) and abs(xv[k])>=np.quantile(past,0.6): st_[k]=True
            if not np.isnan(xv[k]): past.append(abs(xv[k]))
        for mode in ('all','strong'):
            XR.append(xv); XS.append(st_ if mode=='strong' else np.ones(len(xv),bool)); meta.append((s,kind,mode))
XR=np.array(XR).T; XS=np.array(XS).T   # days x V
V=XR.shape[1]; print('variants per H',V,'total',V*len(Hs))
def evalall(XRm,XSm,H,dly,tk,sign=None):
    y=Y[(H,dly)][:,None]; sg=np.sign(XRm)*XSm*(~np.isnan(XRm))*(~np.isnan(y))
    g=np.where(sg!=0,2*sg*np.nan_to_num(y)-tk-4,0.0)  # pnl if s=+1... need sign choice: pnl=2*s*sg*y-tk-4
    return sg,np.nan_to_num(y)
def run(XRm,XSm,full=False):
    out={}
    for H in Hs:
        # choose sign on A with base costs, delay 0
        y0=np.nan_to_num(Y[(H,0)])[:,None]; sg0=np.sign(np.nan_to_num(XRm))*XSm*(~np.isnan(Y[(H,0)])[:,None])
        gA=(sg0*y0)[inA].sum(0); s=np.where(gA>=0,1.,-1.)
        res={}
        for lab,dly,tk in (('base',0,1),('slip4',0,4),('dly3',3,1)):
            y=np.nan_to_num(Y[(H,dly)])[:,None]; ok=(~np.isnan(Y[(H,dly)]))[:,None]
            sg=np.sign(np.nan_to_num(XRm))*XSm*ok
            pnl=np.where(sg!=0,2*s*sg*y-tk-4,0.0); res[lab]=pnl
        out[H]=(s,res)
    return out
def summarize(out):
    rows=[]
    for H in Hs:
        s,res=out[H]; p=res['base']
        for v in range(V):
            pb=p[inB,v]; nb=(pb!=0).sum()
            if nb<30: rows.append(None); continue
            sd=pb[pb!=0].std(); tB=pb.sum()/ (sd*np.sqrt(nb)/ (nb) *nb) if sd>0 else 0  # = mean/ (sd/sqrt n) 
            tB=pb[pb!=0].mean()/(sd/np.sqrt(nb))
            pa=p[inA,v]; 
            rows.append(dict(H=H,v=v,sym=meta[v][0],kind=meta[v][1],mode=meta[v][2],sign=int(s[v]),nA=int((pa!=0).sum()),netA=pa.sum(),nB=int(nb),netB=pb.sum(),tB=tB,
              slip4B=res['slip4'][inB,v].sum(),dly3B=res['dly3'][inB,v].sum(),B1=p[B1h,v].sum(),B2=p[B2h,v].sum()))
    return rows
def passes(r): return r['netB']>0 and r['tB']>=1.28 and r['slip4B']>0 and r['dly3B']>0 and r['B1']>0 and r['B2']>0
real=[r for r in summarize(run(XR,XS)) if r]
R=pd.DataFrame(real); R['pass']=[passes(r) for r in real]
# correlations (all days) raw feature vs gross y
cc=[]
for r in real:
    v=r['v']; y=Y[(r['H'],0)]; x=XR[:,v]; m=~np.isnan(x)&~np.isnan(y)&XS[:,v]
    cc.append(np.corrcoef(x[m],y[m])[0,1] if m.sum()>30 else np.nan)
R['corr']=cc
R['gross_perTrade_B']=(R.netB+ (5)*R.nB)/R.nB   # $ gross per trade ~ (net + cost)/n
R.to_csv('x7a_all.csv',index=False)
print('variants evaluated',len(R),'pass',int(R['pass'].sum()))
# permutation null: permute day-rows jointly
rng=np.random.default_rng(11); null=[]; nullmax=[]
for rep in range(300):
    pm=rng.permutation(len(dts)); out=run(XR[pm],XS[pm])
    rr=[r for r in summarize(out) if r]; null.append(sum(passes(r) for r in rr)); nullmax.append(max(r['netB'] for r in rr))
null=np.array(null); print('NULL pass count per replicate: mean %.1f, 50%% %.0f, 95%% %.0f, max %d ; real %d'%(null.mean(),np.median(null),np.quantile(null,.95),null.max(),R['pass'].sum()))
print('NULL max netB: median %.0f 95%% %.0f ; real max netB %.0f'%(np.median(nullmax),np.quantile(nullmax,.95),R.netB.max()))
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
c=['H','sym','kind','mode','sign','nA','netA','nB','netB','tB','slip4B','dly3B','B1','B2','corr','pass']
print(R.sort_values('netB',ascending=False).head(18)[c].round(2).to_string(index=False))
print('PASSING:'); print(R[R['pass']][c].round(2).to_string(index=False))
# per-symbol summary: mean corr over H for all/raw
s1=R[(R['mode']=='all')].groupby(['sym','kind']).agg(corr_mean=('corr','mean'),netB_mean=('netB','mean'),best_netB=('netB','max')).round(3)
print(s1.sort_values('corr_mean').to_string())