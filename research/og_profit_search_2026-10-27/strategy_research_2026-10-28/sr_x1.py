"""Standalone opening-range-breakout sleeve (BACKTEST ONLY). Rules declared up front, see REPORT Addendum 6."""
from sr_x import *
import time
m5=P.m5; tr5=np.maximum(m5.high-m5.low,np.maximum((m5.high-m5.close.shift()).abs(),(m5.low-m5.close.shift()).abs())); ATR5=tr5.rolling(14).mean().shift(1).values; M5I=m5.index
alld=[d.strftime('%Y-%m-%d') for d in pd.date_range(pd.Timestamp(days[0])-pd.Timedelta(days=2),pd.Timestamp(days[-1]))]
def tm(ds,hm): return pos(ds,hm,2)
EV={}
for D in alld:
    wd=pd.Timestamp(D).weekday()
    if wd<5:
        for key,hm in (('NY','09:30'),('E830','08:30')):
            a=tm(D,hm); bs=tm(D,'15:55')
            if a is not None and bs is not None and bs-a>150: EV.setdefault(key,[]).append((D,a,bs))
    if wd in (6,0,1,2,3):
        a=tm(D,'18:00'); nd=(pd.Timestamp(D)+pd.Timedelta(days=1)).strftime('%Y-%m-%d'); bs=tm(nd,'02:00')
        if a is not None and bs is not None and bs-a>240 and ix[a]-ix[a-1]>=pd.Timedelta(minutes=55): EV.setdefault('GBX',[]).append((D,a,bs))
print({k:len(v) for k,v in EV.items()},flush=True)
def atr_at(i): j=M5I.searchsorted(ix[i],'right')-1; return ATR5[j] if j>=0 else np.nan
def trades(sess_,R,mode,stype,ex,tk=1,dly=0,lo=None,hi=None):
    rows=[]
    for D,a,bs in EV[sess_]:
        if lo and D<lo: continue
        if hi and D>=hi: continue
        if not ok_dt(a,a+R,R) or a+R+2>=bs: continue
        rh=h[a:a+R].max(); rl=l[a:a+R].min(); lim=min(a+R+120,bs-3)
        br=None
        for i in range(a+R,lim):
            if c[i]>rh: br=(i,1);break
            if c[i]<rl: br=(i,-1);break
        if br is None: continue
        i,s_=br
        if mode=='brk':
            side=s_; i_in=i+1+dly; stop_opp=rl if s_>0 else rh
        else:
            j=None
            for k_ in range(i+1,min(i+11,bs-3)):
                if (s_>0 and c[k_]<=rh) or (s_<0 and c[k_]>=rl): j=k_;break
            if j is None: continue
            side=-s_; i_in=j+1+dly; ext=(h[i:j+1].max()+0.25) if s_>0 else (l[i:j+1].min()-0.25); stop_opp=ext
        if i_in>=bs-2: continue
        ent=o[i_in]+side*0.25*tk
        if stype=='opp': stop=stop_opp
        else:
            at=atr_at(i_in)
            if np.isnan(at): continue
            stop=ent-side*at
        Rr=abs(ent-stop)
        if Rr<1.0 or (side>0 and stop>=ent) or (side<0 and stop<=ent): continue
        if ex=='T120': i1=min(i_in+120,bs); tgt=None
        else: i1=bs; tgt=ent+side*int(ex[1])*Rr
        if i1<=i_in+1: continue
        jx,px,_=walk(i_in,i1,side,stop,tgt,tk)
        r=mk(i_in,ent,jx,px,side); r['d']=D; rows.append(r)
    return rows
def specs():
    S=[]
    for sn in ('NY','E830','GBX'):
        for R in (5,15,30,60):
            for mode in ('brk','fade'):
                for stype in ('opp','atr'):
                    for ex in ('R1','R2','R3','T120'): S.append((sn,R,mode,stype,ex))
    return S
def nm(s): return '|'.join(str(x) for x in s)
if __name__=='__main__':
    t0=time.time(); rows=[]; store={}
    for sp in specs():
        df=dfm(trades(*sp)); s=st(df) if len(df)>=3 else dict(n=len(df))
        dA=df[df.d<A1] if len(df) else df; dB=df[df.d>=B0] if len(df) else df
        s['netA']=round(dA.pnl.sum()) if len(dA) else 0; s['netB']=round(dB.pnl.sum()) if len(dB) else 0; s['nA']=len(dA); s['nB']=len(dB)
        s['name']=nm(sp); s['sess']=sp[0]; s['R']=sp[1]; s['mode']=sp[2]; s['stype']=sp[3]; s['ex']=sp[4]; s['sp']=repr(sp); rows.append(s)
    print('time',round(time.time()-t0),flush=True)
    Rr=pd.DataFrame(rows); Rr.to_csv('orb_all.csv',index=False); print('variants',len(Rr))
    Rr['fp']=Rr.folds_pos.str.split('/').str[0].astype(float)
    pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
    print(Rr.groupby(['sess','mode']).agg(n=('net','size'),mean_net=('net','mean'),pos=('net',lambda x:(x>0).sum()),best=('net','max'),mean_trades=('n','mean')).round(0))
    print(Rr.groupby(['R']).agg(mean_net=('net','mean'),pos=('net',lambda x:(x>0).sum())).round(0))
    print(Rr.groupby(['ex','stype']).agg(mean_net=('net','mean'),pos=('net',lambda x:(x>0).sum())).round(0))
    rob=Rr[(Rr.train>0)&(Rr.y26>0)&(Rr.fp>=4)&(Rr.boot_p>=0.9)]
    print('screen pass',len(rob),'| net>0:',(Rr.net>0).sum(),'| t>=2:',(Rr.t>=2).sum(),'| boot_p>=0.9:',(Rr.boot_p>=0.9).sum())
    cols=['name','n','net','pf','dd','t','train','y26','folds_pos','boot_p','netA','netB']
    print(Rr.sort_values('net',ascending=False).head(12)[cols].to_string(index=False))
    # frozen: best on A (n>=30) -> B
    cand=Rr[Rr.nA>=30].sort_values('netA',ascending=False); print('FROZEN best-on-A:',cand.iloc[0][['name','nA','netA','nB','netB']].to_dict())
    print('A-top5 -> B:',cand.head(5)[['name','netA','netB']].to_string(index=False))
    print('rank corr A vs B net:',round(float(Rr[['netA','netB']].corr(method='spearman').iloc[0,1]),2))
    out=[]
    for _,r in rob.iterrows():
        sp=eval(r.sp); rec=dict(name=r['name'],n=r.n,net=r.net,pf=r.pf,dd=r.dd,t=r.t,train=r.train,y26=r.y26,folds=r.folds_pos,boot_p=r.boot_p,netA=r.netA,netB=r.netB)
        for lab,kw in (('slip4',dict(tk=4)),('delay1',dict(dly=1)),('delay3',dict(dly=3))):
            rec[lab]=st(dfm(trades(*sp,**kw))).get('net')
        df=dfm(trades(*sp)); rec['corr']=cb(df); cmb=P.comb(df); rec['port_net']=cmb['net']; rec['port_dd']=cmb['dd']; rec['port_wf']=cmb['wf']; out.append(rec); print(rec)
    pd.DataFrame(out).to_csv('orb_robust.csv',index=False)