import numpy as np,pandas as pd,numba as nb,itertools
d=pd.read_parquet('/workspace/intrabar/1m_long.parquet');d.index=d.index.tz_convert('America/New_York')
TS=d.index;O,H,L,C=[d[c].to_numpy().astype(float) for c in 'open high low close'.split()]
M=np.asarray(TS.hour*60+TS.minute)
def atr_s(rule):
    m=d.resample(rule,label='right',closed='right').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    tr=pd.concat([m.high-m.low,(m.high-m.close.shift()).abs(),(m.low-m.close.shift()).abs()],axis=1).max(axis=1)
    a=tr.ewm(alpha=1/14,adjust=False).mean()
    # swing low confirmed: low[i] < lows of 2 bars each side, known at close of bar i+2
    lo=m.low; sw=(lo<lo.shift(1))&(lo<lo.shift(2))&(lo<lo.shift(-1))&(lo<lo.shift(-2))
    swv=lo.where(sw).shift(2).ffill()  # value available at close of bar i+2
    # map to 1m: value known at minute bar whose close >= label -> bar timestamp is open time? use asof on minute end
    tm=TS+pd.Timedelta('1min')
    A=a.reindex(tm,method='ffill').to_numpy(); S=swv.reindex(tm,method='ffill').to_numpy()
    return A,S
A15,S15=atr_s('15min');A5,S5=atr_s('5min')
st_,en_=[],[]
for k in np.where(M==1080)[0]:
    if k+60>=len(TS):continue
    b=k+1
    while b<len(TS) and (TS[b]-TS[b-1]).total_seconds()<=3*3600 and M[b-1]!=566: b+=1
    st_.append(k);en_.append(b-1 if M[b-1]==566 or b>=len(TS) else b-1)
st_=np.array(st_);en_=np.array(en_);T=TS[st_]
@nb.njit
def sim(k,e_,mode,p1,p2,p3,be,tp_on,atrsrc):
    e=O[k];a0=A15[k-1] if True else 0
    s=min(max(1.5*A15[k-1],12.),40.);sl=e-s;tp=e+400.;pk=e;lock=False;exitp=0.;why=0
    ae=A15[k-1] if atrsrc==0 else A5[k-1]
    for b in range(k,e_+1):
        if L[b]<=sl:
            exitp=min(sl,O[b]) if b>k else sl; why=1 if sl<e else 2;break
        if tp_on and H[b]>=tp: exitp=max(tp,O[b]);why=3;break
        if b==e_: exitp=C[b];why=4;break
        if H[b]>pk: pk=H[b]
        g=pk-e
        if be>0 and g>=be: sl=max(sl,e+2.)
        if mode==0:
            if g*2>=400: sl=max(sl,e+25.)
        elif mode==1:
            if g>=p2:
                a=ae if atrsrc<2 else A15[b]
                sl=max(sl,pk-p1*a)
        elif mode==2:
            if g*2>=p2: sl=max(sl,e+(p1*g*2+4)/2) if False else max(sl,e+p1*g)
        elif mode==3:
            if g>=p2:
                sv=S5[b] if p1==5. else S15[b]
                if sv==sv and sv>e-400: sl=max(sl,sv-p3)
    # shakeout: stopped above initial stop (trail exit) and later new high before exit time
    sh=0
    if why==2 or (why==1 and sl>e-s+1e-9):
        for c in range(b+1,e_+1):
            if H[c]>pk: sh=1;break
    return (exitp-e)*2-4,(pk-e)*2,why,sh,(1 if sl>e-s+1e-9 else 0)
@nb.njit
def runall(st,en,mode,p1,p2,p3,be,tp_on,src):
    n=len(st);R=np.zeros((n,5))
    for i in range(n):
        r=sim(st[i],en[i],mode,p1,p2,p3,be,tp_on,src)
        R[i,0]=r[0];R[i,1]=r[1];R[i,2]=r[2];R[i,3]=r[3];R[i,4]=r[4]
    return R
tr=(T>=pd.Timestamp('2024-10-01',tz=T.tz))&(T<pd.Timestamp('2025-10-01',tz=T.tz));te=T>=pd.Timestamp('2025-10-01',tz=T.tz)
mon=np.asarray(T.dayofweek)==0
def stats(R,m):
    p=R[m,0];eq=np.cumsum(p);dd=(eq-np.maximum.accumulate(np.r_[0,eq])[1:]).min()
    trail=(R[m,2]!=1)|(R[m,4]==1); stopped_tr=((R[m,2]<=2)&(R[m,4]==1))
    return dict(n=m.sum(),win=round((p>0).mean()*100,1),avg=round(p.mean(),1),tot=round(p.sum()),dd=round(dd),
      gb=round((R[m,1]-p)[R[m,1]>=200].mean()),trailx=int(stopped_tr.sum()),shake=round(R[m,3][stopped_tr].mean()*100) if stopped_tr.sum() else 0)
V=[('lock',0,0,0,0,0,1,0)]
for tp in(1,0):
  for src in(0,1,2):
    for k in(1.5,2,2.5,3,4):
      for X in(0,20,40,60,100):V.append((f'ch src{src} k{k} X{X} tp{tp}',1,k,X,0,0,tp,src))
  for f in(.4,.5,.6,.7):
    for P in(100,150,200,300):V.append((f'pct{int(f*100)} P{P} tp{tp}',2,f,P,0,0,tp,0))
  for tf in(5,15):
    for X in(20,40,60,100):
      for buf in(0,5):V.append((f'swing{tf} X{X} b{buf} tp{tp}',3,tf,X,buf,0,tp,0))
  for N in(20,40,60):
    for k in(2,3,4):
      for X in(40,60,100):V.append((f'be{N}+ch15 k{k} X{X} tp{tp}',1,k,X,0,N,tp,0))
rows=[]
for v in V:
    R=runall(st_,en_,int(v[1]),float(v[2]),float(v[3]),float(v[4]),float(v[5]),int(v[6]),int(v[7]))
    for filt,fm in(('all',np.ones(len(T),bool)),('noMon',~mon)):
        for per,pm in(('TR',tr),('TE',te)):
            s=stats(R,pm&fm);s.update(v=v[0],f=filt,per=per);rows.append(s)
df=pd.DataFrame(rows);df.to_pickle('trail.pkl')
w=df.pivot_table(index=['v','f'],columns='per',values=['avg','tot','dd','win','shake','gb','n','trailx'],aggfunc='first')
w.columns=[f'{a}_{b}' for a,b in w.columns];w.to_csv('trail.csv')
for f in('all','noMon'):
    x=w.xs(f,level='f').sort_values('tot_TR',ascending=False)
    print('====',f);print(x.loc[['lock']].to_string());print(x.head(25)[['n_TR','win_TR','avg_TR','tot_TR','dd_TR','gb_TR','shake_TR','trailx_TR','avg_TE','tot_TE','dd_TE','shake_TE']].to_string())
