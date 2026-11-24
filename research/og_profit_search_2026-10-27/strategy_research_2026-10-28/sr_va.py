"""Value-area / POC / SMA200-500 bounce + 80% rule (BACKTEST ONLY) on 1m and 5m bars. Reuses sr_pub (1m data, sessions, stats)."""
import os,sys,json
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
import sr_pub as P
import numpy as np, pandas as pd
ix=P.ix; o,h,l,c,v=P.o,P.h,P.l,P.c,P.v; sess=P.sess; days=P.days; ET=P.ET
# ---- prior-session RTH volume profile (volume spread uniformly over each 1m bar's range, 1pt bins) ----
prof={}
for D in days:
    s=sess[D]; a,b=s['a'],s['b']; lo=np.floor(h[a:b+1].min()); hi=np.ceil(h[a:b+1].max()); 
    lo=np.floor(l[a:b+1].min()); n=int(hi-lo)+2; vol=np.zeros(n)
    for i in range(a,b+1):
        b0=int(np.floor(l[i]-lo)); b1=int(np.floor(h[i]-lo)); k=b1-b0+1
        vol[b0:b1+1]+=v[i]/k
    tot=vol.sum(); poc=int(vol.argmax()); lo_i=hi_i=poc; acc=vol[poc]
    while acc<0.7*tot:
        up=vol[hi_i+1] if hi_i+1<n else -1; dn=vol[lo_i-1] if lo_i>0 else -1
        if up>=dn: hi_i+=1; acc+=up
        else: lo_i-=1; acc+=dn
    prof[D]=dict(poc=lo+poc+0.5,vah=lo+hi_i+1,val=lo+lo_i,end=ix[b])
# level series on 1m grid: latest COMPLETED RTH session's profile
n1=len(ix); POC=np.full(n1,np.nan); VAH=POC.copy(); VAL=POC.copy()
ends=[prof[D]['end'] for D in days]; 
for k,D in enumerate(days):
    st_=ix.searchsorted(prof[D]['end'],side='right'); en=ix.searchsorted(prof[days[k+1]]['end'],side='right') if k+1<len(days) else n1
    POC[st_:en]=prof[D]['poc']; VAH[st_:en]=prof[D]['vah']; VAL[st_:en]=prof[D]['val']
# 5m SMA context (computed on completed 5m closes; mapped to 1m grid using last completed 5m bar)
m5=P.m5; s200=m5.close.rolling(200).mean(); s500=m5.close.rolling(500).mean()
def to1m(sr): 
    sr=sr.copy(); sr.index=sr.index+pd.Timedelta(minutes=5)   # value known at bar END
    return sr.reindex(ix,method='ffill').values
S200_1=to1m(s200); S500_1=to1m(s500)
# aggregated 5m arrays aligned to bar-start; level arrays = values known at bar start (prior completed state)
ix5=m5.index; o5,h5,l5,c5=m5.open.values,m5.high.values,m5.low.values,m5.close.values
def lvl5(arr1): return pd.Series(arr1,index=ix).reindex(ix5,method='ffill').values
POC5,VAH5,VAL5=lvl5(POC),lvl5(VAH),lvl5(VAL)
S200_5=s200.shift(1).values; S500_5=s500.shift(1).values   # known at bar start
def bounce(tf,levname,Rm,tk=1,minrisk=None,maxrisk=None,hold=None,cool=30):
    if tf==1: O,H,L,C=o,h,l,c; LV={'dPOC':(POC,POC),'SMA200':(S200_1,S200_1),'SMA500':(S500_1,S500_1),'VA':(VAL,VAH)}; dtix=ix; mr,xr,hd=(6,25,120); cd=30
    else: O,H,L,C=o5,h5,l5,c5; LV={'dPOC':(POC5,POC5),'SMA200':(S200_5,S200_5),'SMA500':(S500_5,S500_5),'VA':(VAL5,VAH5)}; dtix=ix5; mr,xr,hd=(12,50,24); cd=6
    minrisk=minrisk or mr; maxrisk=maxrisk or xr; hold=hold or hd
    sl=0.25*tk; rows=[]; busy=-1; lastday={}
    dstr=np.array(dtix.tz_convert(ET).strftime('%Y-%m-%d'))
    names=[levname] if levname!='ANY' else ['dPOC','SMA200','SMA500','VA']
    N=len(O); C3=np.r_[np.full(3,np.nan),C[:-3]]
    cands=[]
    for nm in names:
        LL,LH=LV[nm]
        with np.errstate(invalid='ignore'):
            lg=(L<=LL)&(LL<C)&(C>O)&(C3>LL); sh=(H>=LH)&(LH>C)&(C<O)&(C3<LH)
        for i in np.where(lg)[0]: cands.append((i,nm,1,LL[i]))
        for i in np.where(sh)[0]: cands.append((i,nm,-1,LH[i]))
    cands.sort(key=lambda x:x[0])
    for i,nm,side,lv in cands:
        if i<30 or i>=N-hold-2 or i<=busy: continue
        key=(nm,side,dstr[i])
        if lastday.get(key,-10**9)>i-cd: continue
        ent=O[i+1]+side*sl
        stop=(L[i]-0.25) if side>0 else (H[i]+0.25)
        risk=abs(ent-stop)
        if risk<minrisk: stop=ent-side*minrisk; risk=minrisk
        if risk>maxrisk: continue
        tgt=ent+side*Rm*risk; j=None
        for k in range(i+1,i+1+hold):
            if side>0:
                if L[k]<=stop: j,ex=k,min(O[k] if k>i+1 else stop,stop)-sl; break
                if H[k]>=tgt+0.25: j,ex=k,tgt; break
            else:
                if H[k]>=stop: j,ex=k,max(O[k] if k>i+1 else stop,stop)+sl; break
                if L[k]<=tgt-0.25: j,ex=k,tgt; break
        if j is None: j=i+1+hold; ex=O[j]-side*sl
        rows.append(dict(entry_time=dtix[i+1],exit_time=dtix[j],dir=side,pnl=side*(ex-ent)*2-4.0,lvl=nm)); busy=j; lastday[key]=i
    return rows
def rule80(nconf=2,target='opp',tk=1):
    rows=[];sl=0.25*tk
    for k,D in enumerate(days):
        if k==0: continue
        pf=prof[days[k-1]]; s=sess[D]; a=s['a']; op=s['op']
        if pf['val']<=op<=pf['vah']: continue
        side=-1 if op>pf['vah'] else 1; opp=pf['val'] if side<0 else pf['vah']
        tgt=opp if target=='opp' else pf['poc']
        e=P.pos(D,'15:55'); 
        if e is None: continue
        cnt=0; i=a+29
        while i<e-2:
            cl=c[i]; inside=pf['val']<=cl<=pf['vah']
            cnt=cnt+1 if inside else 0
            if cnt>=nconf:
                ent=o[i+1]+side*sl
                ext=h[a:i+1].max()+0.25 if side<0 else l[a:i+1].min()-0.25
                risk=abs(ent-ext)
                if risk>250 or risk<2 or (side*(tgt-ent))<=0: break
                j,px,_=P.walk(i+1,e,side,ext,tgt,tk)
                rows.append(P.mk(i+1,ent,j,px,side)); break
            i+=30
    return rows
if __name__=='__main__':
    out={}
    def go(nm,f):
        r=P.dfm(f(1)); s=P.st(r)
        if len(r)>=3: s['slip3_net']=P.st(P.dfm(f(3)))['net']; s['with_base']=P.comb(r); r.to_csv(os.path.join(P.R,f'va_{nm}.csv'),index=False)
        out[nm]=s; print(nm,s,flush=True)
    for nconf in (1,2):
        for tg in ('opp','poc'): go(f'rule80_c{nconf}_{tg}',lambda tk,nconf=nconf,tg=tg: rule80(nconf,tg,tk))
    for tf in (5,1):
        for lev in ('dPOC','SMA200','SMA500','VA','ANY'):
            for Rm in (1.5,3.0):
                go(f'bounce_{tf}m_{lev}_{Rm}R',lambda tk,tf=tf,lev=lev,Rm=Rm: bounce(tf,lev,Rm,tk))
    json.dump(out,open(os.path.join(P.R,'va_results.json'),'w'),indent=1,default=str)
