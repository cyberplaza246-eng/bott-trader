"""Gap-fade frequency research (BACKTEST ONLY). Reuses sr_pub infra. Entry convention = base gap fade: market at event open +1tk; time exit at open of bar a+H -1tk."""
import sys,os,json,pickle,bisect
sys.argv=['x']
from sr_pub import *
import sr_pub as P
rngs=[sess[D]['hi']-sess[D]['lo'] for D in days]
def atrd(ds):
    k=bisect.bisect_left(days,ds)
    return float(np.mean(rngs[k-20:k])) if k>=20 else None
N=len(o)
TD=pd.Timedelta(minutes=1)
def ok_dt(i,j,m): return j<N and ix[j]-ix[i]==m*TD
# ---- event builders ----
def ev_ny():
    out=[]
    for k,D in enumerate(days):
        if k==0: continue
        s=sess[D]; out.append(dict(i=s['a'],ds=D,mv=s['op']-sess[days[k-1]]['cl'],dow=pd.Timestamp(D).weekday()))
    return out
alld=[d.strftime('%Y-%m-%d') for d in pd.date_range(pd.Timestamp(days[0])-pd.Timedelta(days=3),pd.Timestamp(days[-1]))]
def ev_globex(weekend):
    out=[]
    for D in alld:
        wd=pd.Timestamp(D).weekday()
        if weekend!=(wd==6): continue
        if wd in(4,5): continue
        a=pos(D,'18:00')
        if a is None or a<1 or not(ix[a]-ix[a-1]>=pd.Timedelta(minutes=55)): continue
        out.append(dict(i=a,ds=D,mv=o[a]-c[a-1],dow=wd))
    return out
def ev_move(hm,W):
    out=[]
    for D in alld:
        wd=pd.Timestamp(D).weekday()
        if wd in(5,6) and hm<'18:00': continue
        a=pos(D,hm)
        if a is None or a<W+1: continue
        if not ok_dt(a-W,a,W): continue
        out.append(dict(i=a,ds=D,mv=o[a]-o[a-W],dow=wd))
    return out
def run_ev(ev,mode,thr,Hm,tk,confirm=None,pr=None,minpast=20):
    """mode q/abs/atr. confirm '1m'. pr=(f,m): retrace >= f of |gap| after m minutes, <0.9."""
    rows=[];past=[]
    for e_ in ev:
        mv=e_['mv']; a=e_['i']; D=e_['ds']
        if mode=='q': ok=len(past)>=minpast and abs(mv)>=(np.quantile(past,thr) if thr>0 else 0)
        elif mode=='abs': ok=abs(mv)>=thr
        else:
            at=atrd(D); ok=at is not None and abs(mv)>=thr*at
        past.append(abs(mv))
        if not ok or mv==0: continue
        side=-int(np.sign(mv)); ex=a+Hm
        if not ok_dt(a,ex,Hm): continue
        if confirm=='1m':
            if not side*(c[a]-o[a])>0: continue
            i_in=a+1
        elif pr is not None:
            f,m=pr; fr=side*(c[a+m-1]-o[a])/abs(mv)
            if not(f<=fr<0.9): continue
            i_in=a+m
        else: i_in=a
        if ex<=i_in+1: continue
        ent=o[i_in]+side*0.25*tk
        rows.append(mk(i_in,ent,ex,o[ex]-side*0.25*tk,side))
    return rows
# ---- intraday extension fades ----
cs_tp=(h+l+c)/3
vwap=np.full(N,np.nan)
for D in days:
    s=sess[D]; a,b=s['a'],s['b']; vv=v[a:b+1].astype(float); vwap[a:b+1]=np.cumsum(cs_tp[a:b+1]*vv)/np.maximum(np.cumsum(vv),1)
def ext_fade(kind,Nn,Lm,Hm,tk,start=10*60,end=14*60+30):
    rows=[]
    for k,D in enumerate(days):
        at=atrd(D)
        if at is None: continue
        s=sess[D]; a=s['a']; 
        idx=np.arange(a,s['b']-Hm-2)
        idx=idx[(MOD[idx]>=start)&(MOD[idx]<=end)]
        if len(idx)==0: continue
        if kind=='move': mv=c[idx]-c[idx-Lm]
        elif kind=='vwap': mv=c[idx]-vwap[idx]
        else: mv=c[idx]-o[a]
        hit=np.where(np.abs(mv)>=Nn*at)[0]
        if len(hit)==0: continue
        j=idx[hit[0]]; side=-int(np.sign(mv[hit[0]])); i_in=j+1; ex=i_in+Hm
        if not ok_dt(i_in,ex,Hm): continue
        ent=o[i_in]+side*0.25*tk
        rows.append(mk(i_in,ent,ex,o[ex]-side*0.25*tk,side))
    return rows
# ---- spec list ----
EV={'ny':ev_ny(),'gx_wd':ev_globex(False),'gx_we':ev_globex(True),'lon3':ev_move('03:00',180),'lon9':ev_move('03:00',540),
    'asia2':ev_move('20:00',120),'pl60':ev_move('13:00',60),'pl90':ev_move('13:00',90),'m30':ev_move('15:30',30),'m60':ev_move('15:30',60)}
print({k:len(v_) for k,v_ in EV.items()},flush=True)
for k in ('ny','gx_wd','gx_we','lon3','asia2','m60'):
    a_=np.abs([e['mv'] for e in EV[k]]); print(k,'|mv| q25/50/75/90',np.round(np.quantile(a_,[.25,.5,.75,.9]),1),flush=True)
print('ATRd median',np.median(rngs),flush=True)
S_={}
def add(name,fn,part): S_[name]=(fn,part)
for q in(0.5,0.6,0.65,0.75):
    for H in(60,90,120,180): add(f'P1_ny_q{int(q*100)}_H{H}',lambda tk,q=q,H=H:run_ev(EV['ny'],'q',q,H,tk),1)
for pts in(10,20,30,40):
    for H in(90,180): add(f'P1_ny_abs{pts}_H{H}',lambda tk,p=pts,H=H:run_ev(EV['ny'],'abs',p,H,tk),1)
for f in(0.04,0.06,0.08,0.12):
    for H in(90,180): add(f'P1_ny_atr{int(f*100):02d}_H{H}',lambda tk,f=f,H=H:run_ev(EV['ny'],'atr',f,H,tk),1)
for q in(0,0.5,0.75):
    for H in(60,120,180): add(f'P2_gxwd_q{int(q*100)}_H{H}',lambda tk,q=q,H=H:run_ev(EV['gx_wd'],'q',q,H,tk),2)
for k_ in('lon3','lon9'):
    for q in(0.5,0.75):
        for H in(60,120): add(f'P2_{k_}_q{int(q*100)}_H{H}',lambda tk,k_=k_,q=q,H=H:run_ev(EV[k_],'q',q,H,tk),2)
for q in(0.5,0.75):
    for H in(60,120): add(f'P2_asia2_q{int(q*100)}_H{H}',lambda tk,q=q,H=H:run_ev(EV['asia2'],'q',q,H,tk),2)
for k_ in('pl60','pl90'):
    for H in(30,60): add(f'P2_{k_}_q75_H{H}',lambda tk,k_=k_,H=H:run_ev(EV[k_],'q',0.75,H,tk),2)
for k_ in('m30','m60'): add(f'P2_{k_}_q75_H25',lambda tk,k_=k_:run_ev(EV[k_],'q',0.75,25,tk),2)
for kind,Lm,Ns in(('move',30,(0.25,0.35,0.5)),('move',60,(0.25,0.35,0.5)),('vwap',0,(0.2,0.3,0.4)),('open',0,(0.3,0.4,0.5))):
    for Nn in Ns:
        for H in(45,90): add(f'P3_{kind}{Lm}_N{int(Nn*100)}_H{H}',lambda tk,kind=kind,Lm=Lm,Nn=Nn,H=H:ext_fade(kind,Nn,Lm,H,tk),3)
for q in(0.5,0.6,0.75): add(f'P4_ny_q{int(q*100)}_conf1m_H90',lambda tk,q=q:run_ev(EV['ny'],'q',q,90,tk,confirm='1m'),4)
for q in(0.5,0.75):
    for f in(0.15,0.3):
        for m in(5,15): add(f'P4_ny_q{int(q*100)}_part{int(f*100)}at{m}_H90',lambda tk,q=q,f=f,m=m:run_ev(EV['ny'],'q',q,90,tk,pr=(f,m)),4)
for q in(0,0.5,0.75):
    for H in(60,120,180): add(f'P5_weekend_q{int(q*100)}_H{H}',lambda tk,q=q,H=H:run_ev(EV['gx_we'],'q',q,H,tk,minpast=10),5)
print('variants',len(S_),flush=True)
# ---- run ----
def daily(df): return df.groupby(df.exit_time.dt.tz_convert(ET).dt.date).pnl.sum()
bd=daily(base)
span_m=(base.entry_time.max()-base.entry_time.min()).days/30.44
res={};trades={}
for nm,(fn,part) in S_.items():
    r=dfm(fn(1))
    if len(r)<8: res[nm]=dict(n=len(r),part=part,flag='n<8'); print(nm,res[nm],flush=True); continue
    s=st(r); s['part']=part; s['tpm']=round(len(r)/span_m,1)
    for tk in(3,4):
        rr=dfm(fn(tk)); s[f'slip{tk}']=round(rr.pnl.sum()) if len(rr) else None
    x=daily(r); u=bd.index.union(x.index); s['corr']=round(float(np.corrcoef(bd.reindex(u).fillna(0),x.reindex(u).fillna(0))[0,1]),3)
    s['with_base']=comb(r)
    tr_,y26=s['train'],s['y26']
    if tr_>0 and y26>0 and int(s['folds_pos'][0])>=4 and s['boot_p']>=0.9 and (s['slip4'] or 0)>0: s['flag']='ROBUST'
    elif (tr_>0)!=(y26>0) and s['net']>0: s['flag']='one-window/noise'
    elif tr_>0 and y26>0: s['flag']='both+ but weak'
    else: s['flag']='fail'
    res[nm]=s; trades[nm]=r; print(nm,{k_:s[k_] for k_ in('n','tpm','net','pf','dd','train','y26','folds_pos','boot_p','slip3','slip4','corr','flag')},flush=True)
json.dump(res,open('gap2_results.json','w'),indent=1,default=str)
pickle.dump(trades,open('gap2_trades.pkl','wb'))
print('DONE',flush=True)
