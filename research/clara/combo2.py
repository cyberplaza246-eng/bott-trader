import pandas as pd,numpy as np,pickle,itertools,json
from report import metrics
m=pd.read_csv('MNQ_1m.csv');m['datetime']=pd.to_datetime(m.datetime,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None);m=m.set_index('datetime')
LO,HI=m.low.values,m.high.values;IX=m.index
R=pickle.load(open('revamp/revamp_trades.pkl','rb'))
def prep_rev(t):
    t=t.copy()
    for c in ['entry_time','exit_time']: t[c]=pd.to_datetime(t[c]).dt.tz_convert('America/New_York').dt.tz_localize(None)
    adj=np.where(t.reason.astype(str).str.contains('tp|target',case=False),1.5,1.0)
    t['usd']=t.pnl-adj   # his $3+1tick model -> $4 RT + 1 tick each side
    a=IX.searchsorted(t.entry_time);b=IX.searchsorted(t.exit_time)
    wo=[]
    for i,(x,y,d,e) in enumerate(zip(a,b,t.dir,t.entry)):
        if y<=x: wo.append(t.usd.iloc[i]);continue
        w=(LO[x:y+1].min()-e) if d==1 else (e-HI[x:y+1].max()); wo.append(min(w*2-5,t.usd.iloc[i]))
    t['wo']=wo;t['et']=t.entry_time;t['xt']=t.exit_time;t['src']='revamp:'+t.tag.astype(str)
    return t[['et','xt','usd','wo','src']]
A=pickle.load(open('wf_all.pkl','rb'))
from report import worst
def prep_cl(k):
    t=A[k].copy();t['wo']=t.apply(worst,axis=1);t['xt']=t.et+pd.Timedelta(hours=6)  # exit time not stored; conservative block till 15:30
    t['xt']=t.d+pd.Timedelta(hours=15,minutes=32);t['src']='clara';return t[['et','xt','usd','wo','src']]
def combine(parts,dls=None,pause=None,eqf=None):
    t=pd.concat(parts).sort_values('et').reset_index(drop=True)
    t['sess']=(t.et+pd.Timedelta(hours=6)).dt.normalize()
    keep=[];busy=pd.Timestamp(0);day=None;dpnl=0;eq=0;pk=0;pause_until=pd.Timestamp(0);hist=[]
    for r in t.itertuples():
        if r.et<busy: continue                      # one position at a time
        if r.sess!=day: day=r.sess;dpnl=0
        if dls and dpnl<=-dls: continue             # combined daily loss stop
        if pause and r.et<pause_until: continue
        if eqf and len(hist)>=eqf and eq<np.mean(hist[-eqf:]): hist.append(eq);continue  # equity-curve filter (skip, track)
        keep.append(r.Index);busy=r.xt;dpnl+=r.usd;eq+=r.usd;hist.append(eq);pk=max(pk,eq)
        if pause and pk-eq>=pause[0]: pause_until=r.xt+pd.Timedelta(days=pause[1]);pk=eq
    t=t.loc[keep];t['d']=t.sess;return t
rev={k:prep_rev(v) for k,v in R.items()}
CL={'none':None,'clara_eff35':prep_cl(('eff>=.35',10.0,'lock1_notrail')),'clara_base':prep_cl(('none',11,'lock.5_trail.6'))}
TR=(pd.Timestamp('2024-09-25'),pd.Timestamp('2025-12-31'));TE=(pd.Timestamp('2026-01-01'),pd.Timestamp('2026-09-21'))
CTRL={'none':{},'dls300':dict(dls=300),'dls500':dict(dls=500),'pause1000_2d':dict(pause=(1000,2)),'pause1000_5d':dict(pause=(1000,5)),'eqf20':dict(eqf=20)}
rows=[]
for rk,ck,ctk in itertools.product(rev,CL,CTRL):
    parts=[rev[rk]]+([CL[ck]] if CL[ck] is not None else [])
    t=combine(parts,**CTRL[ctk])
    for per,(a,b) in [('train',TR),('test26',TE)]:
        s=t[(t.d>=a)&(t.d<=b)]
        r=metrics(s,'wo',f'{rk}|{ck}|{ctk}|{per}');br=r.pop('breach_2k_dates');r['breaches']=len(br);r['first_breach']=br[0] if br else '';rows.append(r)
X=pd.DataFrame(rows);X.to_csv('combo2.csv',index=False)
X['key']=X.set.str.rsplit('|',n=1).str[0];X['per']=X.set.str.rsplit('|',n=1).str[1]
P=X.pivot(index='key',columns='per',values=['net','intraday_dd','breaches'])
P.columns=[f'{a}_{b}' for a,b in P.columns];print('variants',len(P))
print(P.sort_values('net_train',ascending=False).to_string())
