"""OSS ORB/VWAP ports, independently re-implemented in Python. Credits:
 S1 ORB5  - Zarattini & Aziz (2023) SSRN 4416622; ref impl github.com/giovannibrusco/zarattini-2023-orb-qqq (MIT)
 S2 NQBO  - github.com/giovannibrusco/nq-intraday-breakout (MIT): 09:30-11:00 range, 100pt stop/200pt tgt, max2/day
 S3 VWAPT - Zarattini & Aziz (2023) SSRN 4631351 'VWAP Holy Grail' (paper rules: long>VWAP short<VWAP, 1m, EOD flat)
 S4 NOISE - Zarattini, Aziz & Barbon (2024) SSRN 4824172; ref github.com/giovannibrusco/zarattini-2024-momentum-spy (MIT)
Costs: 1 MNQ, $2/pt, $4 RT, 0.25pt slippage each side."""
import pandas as pd, numpy as np, json, sys
PT=2.0; COM=4.0; SL=0.25
df=pd.read_csv('/workspace/clara/MNQ_1m.csv',parse_dates=['datetime'])
df['t']=df.datetime.dt.tz_convert('America/New_York'); df=df.set_index('t')
rth=df.between_time('09:30','15:59').copy(); rth['d']=rth.index.date
days={d:g for d,g in rth.groupby('d') if len(g)>300}
def fill(px,side,entry): return px+SL*side if entry else px-SL*side
def trade(day,side,e,x): return dict(date=day,side=side,pnl=(x-e)*side*PT-COM)
def walk(g,i0,side,e,stop,tgt,trail=None):
    """simulate from bar i0 (entry at e); stop/target checked intrabar, stop first."""
    for i in range(i0,len(g)):
        h,l,c=g.high.iat[i],g.low.iat[i],g.close.iat[i]
        if side>0:
            if l<=stop: return fill(min(stop,g.open.iat[i]) if i>i0 else stop,side,0),i
            if tgt and h>=tgt: return fill(tgt,side,0),i
        else:
            if h>=stop: return fill(max(stop,g.open.iat[i]) if i>i0 else stop,side,0),i
            if tgt and l<=tgt: return fill(tgt,side,0),i
        if trail is not None: stop=trail(i,stop)
    return fill(g.close.iat[-1],side,0),len(g)-1
def s1(day,g):
    b=g.between_time('09:30','09:34'); o,c=b.open.iat[0],b.close.iat[-1]
    if o==c: return []
    side=1 if c>o else -1; stop=b.low.min() if side>0 else b.high.max()
    i0=len(b); e=fill(g.open.iat[i0],side,1); R=abs(e-stop)
    x,_=walk(g,i0,side,e,stop,e+10*R*side); return [trade(day,side,e,x)]
def s2(day,g):
    w=g.between_time('09:30','10:59'); hi,lo=w.high.max(),w.low.min(); out=[]; i=len(w); n=0
    last=g.index.get_indexer([g.between_time('14:30','14:30').index[0]])[0] if len(g.between_time('14:30','14:30')) else len(g)-1
    while i<=last and n<2:
        h,l=g.high.iat[i],g.low.iat[i]; side=1 if h>hi else (-1 if l<lo else 0)
        if side:
            lvl=hi if side>0 else lo; e=fill(max(lvl,g.open.iat[i]) if side>0 else min(lvl,g.open.iat[i]),side,1)
            x,j=walk(g,i,side,e,e-100*side,e+200*side); out.append(trade(day,side,e,x)); n+=1; i=j+1
        else: i+=1
    return out
def s3(day,g):
    tp=(g.high+g.low+g.close)/3; vw=(tp*g.volume).cumsum()/g.volume.cumsum().replace(0,np.nan)
    sig=np.sign(g.close-vw).values; out=[]; pos=0; e=0
    for i in range(1,len(g)-1):
        s=sig[i]
        if s and s!=pos:
            px=g.open.iat[i+1]
            if pos: out.append(trade(day,pos,e,fill(px,pos,0)))
            pos=int(s); e=fill(px,pos,1)
    if pos: out.append(trade(day,pos,e,fill(g.close.iat[-1],pos,0)))
    return out
# S4 noise band: sigma(t)=mean over prior 14 days of |close(t)/open-1|; checks on :00/:30
mv={}
for d,g in days.items(): mv[d]=(g.close/g.open.iat[0]-1).abs().groupby(g.index.strftime('%H:%M')).last()
mvdf=pd.DataFrame(mv).T; sig14=mvdf.rolling(14).mean().shift(1)
dl=list(days); prevc={dl[k]:days[dl[k-1]].close.iat[-1] for k in range(1,len(dl))}
def s4(day,g):
    if day not in prevc or sig14.loc[day].isna().all(): return []
    o=g.open.iat[0]; pc=prevc[day]; up=max(o,pc); dn=min(o,pc)
    tp=(g.high+g.low+g.close)/3; vw=((tp*g.volume).cumsum()/g.volume.cumsum().replace(0,np.nan)).values
    s=sig14.loc[day]; out=[]; pos=0; e=0
    for i in range(len(g)-1):
        hm=g.index[i].strftime('%H:%M')
        if g.index[i].minute not in (0,30) or hm<'10:00' or hm not in s or np.isnan(s[hm]): continue
        c=g.close.iat[i]; U=up*(1+s[hm]); D=dn*(1-s[hm])
        if pos>0: want=1 if c>max(U,vw[i]) else 0
        elif pos<0: want=-1 if c<min(D,vw[i]) else 0
        else: want=0
        if c>U: want=1
        elif c<D: want=-1
        if want!=pos:
            px=g.open.iat[i+1]
            if pos: out.append(trade(day,pos,e,fill(px,pos,0)))
            pos=want
            if pos: e=fill(px,pos,1)
    if pos: out.append(trade(day,pos,e,fill(g.close.iat[-1],pos,0)))
    return out
def stats(name,T):
    t=pd.DataFrame(T); 
    if t.empty: return {}
    t['date']=pd.to_datetime(t.date); dp=t.groupby('date').pnl.sum()
    eq=dp.cumsum(); dd=(eq.cummax().clip(lower=0)-eq)
    # intraday approx: trade-by-trade equity
    te=t.pnl.cumsum(); idd=(te.cummax().clip(lower=0)-te).max()
    # breaches of $2k trailing DD (EOD), reset peak after breach to list episodes
    br=[];pk=0;inb=False
    for d,v in eq.items():
        pk=max(pk,v)
        if pk-v>=2000 and not inb: br.append(str(d.date())); inb=True
        if v>=pk: inb=False
    w=t[t.pnl>0].pnl; l=t[t.pnl<=0].pnl; s=(t.pnl<=0).astype(int); streak=(s.groupby((s!=s.shift()).cumsum()).cumsum()).max()
    r=lambda x:round(float(x),1)
    out=dict(strategy=name,trades=len(t),net=r(t.pnl.sum()),PF=round(w.sum()/-l.sum(),2) if len(l) else None,win=round(100*len(w)/len(t),1),
      avgW=r(w.mean()),avgL=r(l.mean()),maxLstreak=int(streak),ddEOD=r(dd.max()),ddTrade=r(idd),
      tpd=round(len(t)/len(days),2),breaches=br)
    for nm,m in [('train',t.date<'2026-01-01'),('test26',t.date>='2026-01-01')]:
        out[nm+'_net']=r(t[m].pnl.sum()); out[nm+'_n']=int(m.sum())
    return out
R=[]
for nm,f in [('S1_ORB5_Zarattini',s1),('S2_NQ_breakout_brusco',s2),('S3_VWAP_trend',s3),('S4_noise_momentum',s4)]:
    T=[x for d,g in days.items() for x in f(d,g)]; pd.DataFrame(T).to_csv(f'{nm}_trades.csv',index=False)
    st=stats(nm,T); R.append(st); print(json.dumps(st),flush=True)
json.dump(R,open('results.json','w'),indent=1)
