"""Q2: intraday lead/lag of each symbol vs MNQ. (a) hourly bars, 2 years, MNQ from local 1m; (b) 5-minute bars, last ~60 days only (yfinance limit), NQ=F as MNQ proxy. BACKTEST ONLY."""
from sr_x import *
import warnings; warnings.filterwarnings('ignore')
exec(open('sr_x7_data.py').read().split("H1={s:")[0].split('from sr_x import *')[1])  # SYMS, DIFF, loadh
SY=[s for s in SYMS if s not in ('NQ',)]
t_idx=pd.DatetimeIndex(ix)  # MNQ 1m index (tz-aware)
Nn=len(o)
def mnq_pts(t_start,mins):
    """MNQ open-to-open points from t_start over `mins` minutes using exact minute bars; nan if missing"""
    i=ix.searchsorted(t_start); 
    if i>=Nn or ix[i]!=t_start: return np.nan,-1
    j=i+mins
    if j>=Nn or ix[j]-ix[i]!=mins*TD: return np.nan,-1
    return o[j]-o[i],i
rng=np.random.default_rng(7)
# ---------------- hourly ----------------
rows=[]
for s in SY:
    d=loadh(s); d=d[(d.index>=pd.Timestamp('2024-10-01',tz='UTC'))]
    et=d.index.tz_convert(ET); hm=et.hour*60+et.minute; ok=(hm>=600)&(hm<=870)&(et.weekday<5)   # start 10:00..14:30 ET
    dd=d[ok]; ee=et[ok]
    for (st_,r0) in zip(ee,dd.itertuples()):
        x=np.log(r0.Close/r0.Open) if s not in DIFF else (r0.Close-r0.Open)
        m0,i0=mnq_pts(st_,60)
        if i0<0: continue
        m1,i1=mnq_pts(st_+pd.Timedelta(minutes=60),60)
        if i1<0: continue
        rows.append((s,st_.strftime('%Y-%m-%d'),st_.hour*60+st_.minute,x,m0,m1,o[i1],o[i1+60]-0))
H=pd.DataFrame(rows,columns=['sym','d','hm','x','m0','m1','p_e','p_x']); H['dt']=pd.to_datetime(H.d)
print('hourly rows',len(H),'symbols',H.sym.nunique())
def part_corr(x,m0,m1):
    bx=np.polyfit(m0,x,1); b1=np.polyfit(m0,m1,1); ex=x-np.polyval(bx,m0); e1=m1-np.polyval(b1,m0); return np.corrcoef(ex,e1)[0,1]
out=[]
for s,g in H.groupby('sym'):
    g=g.sort_values(['dt','hm']); 
    if len(g)<500 or g.x.std()==0: continue
    cc=np.corrcoef(g.x,g.m0)[0,1]; lc=np.corrcoef(g.x,g.m1)[0,1]; pc=part_corr(g.x.values,g.m0.values,g.m1.values)
    # day-cluster bootstrap of partial lead corr
    gd=[x for _,x in g.groupby('d')]; bs=[]
    for _ in range(300):
        pick=rng.integers(0,len(gd),len(gd)); gg=pd.concat([gd[i] for i in pick]); bs.append(part_corr(gg.x.values,gg.m0.values,gg.m1.values))
    t=pc/np.std(bs)
    # frozen signal: sign chosen on A, scored on B (trade next hour, 1 tick/side + $4)
    A_=g[(g.dt>=A0)&(g.dt<A1)]; B_=g[(g.dt>=B0)&(g.dt<B1)]
    gp=lambda df:(2*np.sign(df.x)*(df.p_x-df.p_e)).values  # placeholder replaced below
    def gross(df): return (df.m1.values*2*np.sign(df.x.values))
    sgn=1 if gross(A_).sum()>=0 else -1
    nB=len(B_); gB=sgn*gross(B_); netB=gB.sum()-5*nB; gpt=gB.mean(); tB=(gB-5).mean()/((gB-5).std()/np.sqrt(nB))
    nA=len(A_); gA=sgn*gross(A_); 
    h1=B_[B_.dt<'2026-05-01']; h2=B_[B_.dt>='2026-05-01']; n1=sgn*gross(h1).sum()-5*len(h1); n2=sgn*gross(h2).sum()-5*len(h2)
    out.append(dict(sym=s,n=len(g),contemp_corr=cc,lead_corr=lc,partial_lead_corr=pc,t_boot=t,sign=sgn,gross_per_trade_B=gpt,netB=netB,tB=tB,netA=gA.sum()-5*nA,B1=n1,B2=n2))
HR=pd.DataFrame(out); HR['pass']=(HR.netB>0)&(HR.tB>=1.28)&(HR.B1>0)&(HR.B2>0)
HR.to_csv('x7b_hourly.csv',index=False); pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print('HOURLY lead (X bar return -> MNQ next-hour points). variants',len(HR),'pass',int(HR['pass'].sum()),' |t_boot|>=2:',int((HR.t_boot.abs()>=2).sum()))
print(HR.sort_values('t_boot',key=abs,ascending=False).head(14).round(3).to_string(index=False))
print('cost per trade $5; gross edge per trade B (top by |t|) above')
# null for hourly signal: shuffle x across bars within same hm slot (break link), count passes
def count_pass(Hn):
    k=0
    for s,g in Hn.groupby('sym'):
        A_=g[(g.dt>=A0)&(g.dt<A1)]; B_=g[(g.dt>=B0)&(g.dt<B1)]
        if len(B_)<100: continue
        sgn=1 if (A_.m1.values*np.sign(A_.x.values)).sum()>=0 else -1
        gB=sgn*B_.m1.values*2*np.sign(B_.x.values)-5; tB=gB.mean()/(gB.std()/np.sqrt(len(gB)))
        h1=B_.dt<'2026-05-01'
        if gB.sum()>0 and tB>=1.28 and gB[h1.values].sum()>0 and gB[~h1.values].sum()>0: k+=1
    return k
nc=[]
for rep in range(100):
    Hn=H.copy(); Hn['x']=Hn.groupby(['sym','hm']).x.transform(lambda v: rng.permutation(v.values)); nc.append(count_pass(Hn))
nc=np.array(nc); print('HOURLY NULL pass count: mean %.1f 95%% %.0f max %d ; real %d'%(nc.mean(),np.quantile(nc,.95),nc.max(),HR['pass'].sum()))
# ---------------- 5-minute ----------------
def load5(s):
    d=pd.read_csv(f'yf7/{s}_5m.csv',index_col=0); d.index=pd.to_datetime(d.index,utc=True); d=d[['Open','Close']].apply(pd.to_numeric,errors='coerce').dropna(); return d[~d.index.duplicated()].sort_index()
nq=load5('NQ'); nqr=np.log(nq.Close/nq.Open)
res=[]; LAGS=(1,2,3,6,12)
for s in SY:
    d=load5(s); xr=(np.log(d.Close/d.Open) if s not in DIFF else (d.Close-d.Open)).rename('x')
    J=pd.concat([xr,nqr.rename('n')],axis=1,join='inner').dropna()
    et=J.index.tz_convert(ET); hm=et.hour*60+et.minute; J=J[(hm>=575)&(hm<=955)]   # RTH starts 09:35..15:55
    J['d']=J.index.tz_convert(ET).date
    if len(J)<1500 or J.x.std()==0: continue
    for L in LAGS:
        fut=J.groupby('d').n.shift(-L)   # NQ return L bars later (same day)
        K=pd.DataFrame(dict(x=J.x,n0=J.n,f=fut,d=J.d)).dropna()
        if len(K)<1000: continue
        bx=np.polyfit(K.n0,K.x,1); bf=np.polyfit(K.n0,K.f,1); ex=K.x-np.polyval(bx,K.n0); ef=K.f-np.polyval(bf,K.n0)
        pc=np.corrcoef(ex,ef)[0,1]; gd=[(ex[K.d==u].values,ef[K.d==u].values) for u in K.d.unique()]; bs=[]
        for _ in range(200):
            pk=rng.integers(0,len(gd),len(gd)); a=np.concatenate([gd[i][0] for i in pk]); b=np.concatenate([gd[i][1] for i in pk]); bs.append(np.corrcoef(a,b)[0,1])
        pts=nq.Close.mean()
        edge=float((np.sign(ex)*K.f).mean()*pts*2)   # $ per MNQ trade gross
        res.append(dict(sym=s,lag_min=5*L,n=len(K),days=K.d.nunique(),partial_corr=pc,t_boot=pc/np.std(bs),gross_usd_per_trade=edge))
R5=pd.DataFrame(res); R5.to_csv('x7b_5m.csv',index=False)
print('\n5-MIN: window',nq.index.min(),nq.index.max(),'| rows',len(R5),'tests',len(R5),'| |t|>=3:',int((R5.t_boot.abs()>=3).sum()),'|t|>=2:',int((R5.t_boot.abs()>=2).sum()),'expected |t|>=2 by chance ~',round(len(R5)*0.05,1))
print(R5.sort_values('t_boot',key=abs,ascending=False).head(14).round(3).to_string(index=False))
print('ES lags:'); print(R5[R5.sym=='ES'].round(3).to_string(index=False))