import pickle,pandas as pd,numpy as np
comp=pickle.load(open('comp.pkl','rb'));R=pickle.load(open('revamp/revamp_trades.pkl','rb'))['live_all_max3']
L=comp['live_all_max3:ALL'].reset_index(drop=True).copy()
L['dir']=R.dir.values;L['setup']=R.tag.values+'@'+R.session.values;L['reason']=R.reason.values
L['d']=(L.et+pd.Timedelta(hours=6)).dt.normalize()  # trading day (18:00 ET roll)
def basic(t):
    w,l=t.usd[t.usd>0],t.usd[t.usd<=0]
    s=(t.usd<=0).astype(int);streak=(s.groupby((s!=s.shift()).cumsum()).cumsum()*s).max()
    return dict(n=len(t),net=round(t.usd.sum()),pf=round(w.sum()/-l.sum(),2),win=round(100*len(w)/len(t),1),avgW=round(w.mean()),avgL=round(l.mean()),exp=round(t.usd.mean(),1),tpd=round(len(t)/t.d.nunique(),2),streak=int(streak))
def dd(t):
    t=t.sort_values('xt');eq=t.usd.cumsum().values;eqi=np.r_[0,eq]
    # intraday: equity before trade + worst open excursion (approx, per trade)
    low=np.r_[0,(np.r_[0,eq[:-1]]+np.minimum(t.wo.values,0))]
    peak=np.maximum.accumulate(eqi);ddi=(peak-np.minimum(eqi,low)).max()
    # EOD
    de=t.groupby('d').usd.sum().cumsum();pk=de.cummax().clip(lower=0);dde=pk-de;tr=dde.idxmax();st=de[:tr][de[:tr]==pk[tr]].index
    st=st[-1] if len(st) else de.index[0];rec=de[tr:][de[tr:]>=pk[tr]].index;rec=rec[0] if len(rec) else 'not recovered'
    # longest underwater
    uw=(dde>0);grp=(~uw).cumsum();lens=[(g.index[0],g.index[-1],(g.index[-1]-g.index[0]).days) for _,g in dde[uw].groupby(grp[uw])]
    lo=max(lens,key=lambda x:x[2]) if lens else None
    return dict(intraday_dd=round(ddi),eod_dd=round(dde.max()),eod_start=str(st)[:10],trough=str(tr)[:10],recovered=str(rec)[:10],longest_underwater=f"{lo[2]}d {str(lo[0])[:10]}->{str(lo[1])[:10]}")
P={'FULL':L,'TRAIN<=2025':L[L.d<'2026-01-01'],'TEST2026':L[L.d>='2026-01-01']}
for k,t in P.items(): print(k,basic(t),dd(t))
day=L.groupby('d').usd.sum();wk=L.groupby(L.d.dt.to_period('W')).usd.sum();mo=L.groupby(L.d.dt.to_period('M')).usd.sum()
print('worst day',day.idxmin().date(),round(day.min()),'| best day',round(day.max()),'| worst week',wk.idxmin(),round(wk.min()),'| worst month',mo.idxmin(),round(mo.min()))
print('MONTHLY');print(mo.round().to_string());print('profitable months',(mo>0).sum(),'/',len(mo))
def grp(col):
    return pd.DataFrame({k:basic(g) for k,g in L.groupby(col)}).T[['n','net','pf','win','exp']]
print(grp('setup').sort_values('net',ascending=False).to_string())
L['side']=np.where(L.dir>0,'long','short');print(grp('side').to_string())
h=L.et.dt.hour;L['tod']=pd.cut(h,[-1,3,8,9,11,13,15,16,17,23],labels=['00-04','04-09','09-10','10-12','12-14','14-16','16-17','17-18','18-24']);print(grp('tod').to_string())
L['dow']=L.d.dt.day_name().str[:3];print(grp('dow').to_string())
print(L.reason.value_counts().to_dict())
