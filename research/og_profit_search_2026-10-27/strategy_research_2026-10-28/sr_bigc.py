"""Big-candle / exhaustion fade + continuation (BACKTEST ONLY). 1m & 5m signals, entry at open of 1m bar after signal close (+delay), 1 tick slip each side, $4 RT."""
import os,sys,json,time
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
import numpy as np, pandas as pd
import sr_pub as P
ix=P.ix; o=P.o; h=P.h; l=P.l; c=P.c; v=P.v; ET=P.ET; m1=P.m1
N1=len(ix); IXN=ix.asi8; MIN=int(pd.Timedelta(minutes=1).value*0+pd.Timedelta("1min")/pd.Timedelta(1,unit=ix.unit))
def tfdata(tf):
    if tf==1: d=m1
    else: d=P.m5 if tf==5 else None
    t=d.index.as_unit(ix.unit).asi8; O=d.open.values;H=d.high.values;L=d.low.values;C=d.close.values;V=d.volume.values.astype(float)
    pc=np.r_[np.nan,C[:-1]]; tr=np.maximum(H-L,np.maximum(abs(H-pc),abs(L-pc)))
    atr=pd.Series(tr).rolling(14).mean().shift(1).values; va=pd.Series(V).rolling(20).mean().shift(1).values
    n=len(t); k=np.arange(n); ok=np.zeros(n,bool); ok[15:]=(t[15:]-t[:-15])<=(16*tf)*MIN
    s=np.searchsorted(IXN,t+tf*MIN); s=np.minimum(s,N1-1); ok&=(IXN[s]==t+tf*MIN)
    e=d.index.tz_convert(ET); mod=(e.hour*60+e.minute).values; wd=e.weekday.values
    return dict(t=t,O=O,H=H,L=L,C=C,V=V,atr=atr,va=va,ok=ok&~np.isnan(atr)&(atr>0),s=s,mod=mod,wd=wd,rng=H-L,tf=tf)
TF={1:tfdata(1),5:tfdata(5)}
def signals(fam,par,tf):
    D=TF[tf]; rg=D['rng']; ra=rg/np.where(D['atr']>0,D['atr'],np.nan); body=D['C']-D['O']; col=np.sign(body)
    ok=D['ok'].copy()
    if fam=='range':
        m=ok&(ra>=par['N'])&(col!=0); side=col
    elif fam=='vol':
        m=ok&(D['V']>=par['V']*D['va'])&(ra>=1.0)&(col!=0); side=col
    elif fam=='wick':
        up=D['H']-np.maximum(D['O'],D['C']); lo=np.minimum(D['O'],D['C'])-D['L']
        m=ok&(ra>=par['R'])&(rg>0)
        w=np.where(lo>=up,lo,up)/np.where(rg>0,rg,np.nan); m&=(w>=0.6)
        side=np.where(lo>=up,-1.0,1.0)   # move direction that was REJECTED (lower wick -> down move rejected: stored as -1)
    elif fam=='climax':
        n=len(rg); m=np.zeros(n,bool); side=np.zeros(n)
        a=np.arange(n-1)
        cl=ok[:-1]&(ra[:-1]>=par['N'])&(D['V'][:-1]>=2*D['va'][:-1])&(col[:-1]!=0)
        rv=(col[1:]==-col[:-1])&(abs(body[1:])>=0.5*D['atr'][:-1])&(D['t'][1:]-D['t'][:-1]==tf*MIN)
        mm=cl&rv; m[1:]=mm; side[1:]=col[:-1]   # side=climax direction; signal bar = reversal bar (index+1)
    elif fam=='news':
        m=ok&(D['mod']==par['hm'])&(D['wd']<5)&(col!=0)
        if par['R']>0: m&=(ra>=par['R'])
        side=col
    idx=np.where(m)[0]; return idx,side[idx]
def run(idx,sd,tf,mode,ex,delay=0,tk=1):
    """mode fade: trade against sd ; cont: with sd (for wick fade = with rejection => sd stored as rejected move dir -> fade=-sd same). returns df"""
    D=TF[tf]; s=D['s'][idx]+delay; sig=-sd if mode=='fade' else sd
    keep=(s+61<N1); s=s[keep]; sig=sig[keep]; atr=D['atr'][idx][keep]; ti=D['t'][idx][keep]
    # entry bar must exist exactly delay mins after signal close; 60 min contiguity
    e=s; okc=(IXN[e]==ti+(tf+delay)*MIN)
    if ex[0]=='T':
        Hm=int(ex[1:]); okc&=(IXN[np.minimum(e+Hm,N1-1)]-IXN[e]==Hm*MIN)
    else: okc&=(IXN[e+60]-IXN[e]==60*MIN)
    e=e[okc];sig=sig[okc];atr=atr[okc]
    if len(e)==0: return pd.DataFrame(columns=['entry_time','exit_time','dir','pnl'])
    sl=0.25*tk; pin=o[e]+sig*sl
    if ex[0]=='T':
        Hm=int(ex[1:]); xi=e+Hm; pout=o[xi]-sig*sl
    else:
        tm={'R1':1.0,'R2':2.0}[ex]; stop=pin-sig*atr; tgt=pin+sig*tm*atr
        K=np.arange(60); W=e[:,None]+K[None,:]
        Hh=h[W];Ll=l[W];Oo=o[W]
        sh=np.where(sig[:,None]>0,Ll<=stop[:,None],Hh>=stop[:,None]); th=np.where(sig[:,None]>0,Hh>=tgt[:,None]+0.25*sig[:,None]*1,Ll<=tgt[:,None]-0.25)
        ks=np.where(sh.any(1),sh.argmax(1),99); kt=np.where(th.any(1),th.argmax(1),99)
        stopfirst=(ks<=kt)&(ks<99); tgtfirst=(kt<ks)
        pout=np.empty(len(e)); xi=e+60
        pout[:]=o[e+60]-sig*sl
        r=np.where(stopfirst)[0]; kk=ks[r]
        so=Oo[r,kk]; pout[r]=np.where(sig[r]>0,np.minimum(so,stop[r]),np.maximum(so,stop[r]))-sig[r]*sl; xi[r]=e[r]+kk
        r=np.where(tgtfirst)[0]; pout[r]=tgt[r]; xi[r]=e[r]+kt[r]
    pnl=sig*(pout-pin)*2-4.0
    # one position at a time
    order=np.argsort(e,kind='stable'); e=e[order];xi=xi[order];sig=sig[order];pnl=pnl[order]
    keepi=[];last=-1
    for j in range(len(e)):
        if e[j]>last: keepi.append(j); last=xi[j]
    kj=np.array(keepi,int)
    return pd.DataFrame(dict(entry_time=ix[e[kj]],exit_time=ix[np.maximum(xi[kj],e[kj])],dir=sig[kj].astype(int),pnl=pnl[kj]))
EX=['T5','T15','T30','T60','R1','R2']
def variants():
    V=[]
    for tf in (1,5):
        for N in (2,3,4): V.append(('range',dict(N=N),tf))
        for x in (2,3): V.append(('vol',dict(V=x),tf))
        for R in (1.5,2.0): V.append(('wick',dict(R=R),tf))
        for N in (2,3): V.append(('climax',dict(N=N),tf))
        for hm,nm in ((510,'0830'),(600,'1000'),(840,'1400')):
            for R in (0,1.5): V.append(('news',dict(hm=hm,R=R,nm=nm),tf))
    return V
def vname(f,p,tf,mode,ex): return f"{f}|{','.join(f'{k}={x}' for k,x in p.items() if k!='hm')}|{tf}m|{mode}|{ex}"
def corr_base(df):
    a=df.groupby(df.exit_time.dt.tz_convert(ET).dt.date).pnl.sum(); b=P.base.groupby(P.base.exit_time.dt.tz_convert(ET).dt.date).pnl.sum()
    j=pd.concat([a,b],axis=1).fillna(0); return round(float(j.corr().iloc[0,1]),3)
if __name__=='__main__':
    t0=time.time(); rows=[]; cache={}
    for f,p,tf in variants():
        idx,sd=signals(f,p,tf)
        for mode in ('fade','cont'):
            for ex in EX:
                df=run(idx,sd,tf,mode,ex); s=P.st(P.dfm(df.to_dict('records'))) if len(df)>=3 else dict(n=len(df))
                s.update(name=vname(f,p,tf,mode,ex),fam=f,tf=tf,mode=mode,ex=ex,nsig=len(idx)); rows.append(s)
        print(f,p,tf,'signals',len(idx),f'{time.time()-t0:.0f}s',flush=True)
    R=pd.DataFrame(rows); R.to_csv('bigc_all.csv',index=False)
    print('variants',len(R))