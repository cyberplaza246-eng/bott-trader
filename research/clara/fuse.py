exec(open('combo2.py').read().split("rows=[]")[0])
import warnings;warnings.filterwarnings('ignore')
comp=pickle.load(open('comp.pkl','rb'))
R=pickle.load(open('revamp/revamp_trades.pkl','rb'))['live_all_max3']
L=comp['live_all_max3:ALL'].copy();L['dir']=R.dir.values;L['sess']=R.session.values;L['setup']=R.tag.values
L['etl']=pd.to_datetime(R.entry_time).dt.tz_convert('America/New_York').dt.tz_localize(None).values
L['date']=L.etl.dt.normalize()
r=pd.read_csv('replica_trades.csv',parse_dates=['et','d']);r['et']=r.et+pd.Timedelta(minutes=2);f=r.groupby('d').first()  # first clara signal per day
dayf=pd.read_pickle('dayf.pkl');dayf.index=pd.to_datetime(dayf.index)
L=L.join(f[['et','s']].rename(columns={'et':'cet','s':'cs'}),on='date').join(dayf[['eff','orw_rel']],on='date')
day=L.sess.eq('day')&L.etl.dt.hour.between(9,16)
known=day&L.cet.notna()&(L.etl>=L.cet)
dirn=L.dir.map(lambda x:1 if str(x).lower() in('1','long','buy') else -1)
trendday=(L.eff>=0.35)
CL=comp['clara_eff35'].copy();CLb=comp['clara_base'].copy()
crep=r.copy();crep['etn']=crep.et
V={}
V['BASE']=[L]
V['a1_block_counterbias']=[L[~(known&(dirn!=L.cs))]]
V['a2_withbias_only_day']=[L[~day|(known&(dirn==L.cs))]]
V['b1_trendday_noMR_day']=[L[~(day&trendday&(L.setup=='ens_mean_reversion'))]]
V['b2_chop_noTB_day']=[L[~(day&(L.eff<0.35)&L.setup.isin(['ens_trend','ens_breakout']))]]
V['b3_b1+b2']=[L[~(day&trendday&(L.setup=='ens_mean_reversion'))&~(day&(L.eff<0.35)&L.setup.isin(['ens_trend','ens_breakout']))]]
# c: clara eff35 trade when revamp flat 9:50-10:00 or agrees
def cfilter(C,mode):
    keep=[]
    for i,t in C.iterrows():
        d=t.et.normalize();s=L[(L.etl<=t.et)&(L.xt>t.et)] if 'xt' in L else None
        open_=L[(L.etl<=t.et)&(pd.to_datetime(L.xt)>t.et)]
        side=r.set_index('et').s.get(t.et,0)
        if mode=='flat':keep.append(len(open_)==0)
        else:keep.append(len(open_)==0 or (dirn[open_.index]==side).all())
    return C[keep]
V['c1_BASE+clara35_ifflat']=[L,cfilter(CL,'flat')]
V['c2_BASE+clara35_flat_or_agree']=[L,cfilter(CL,'agree')]
V['c3_BASE+clarabase_flat_or_agree']=[L,cfilter(CLb,'agree')]
# d: skip revamp opposite to open clara position
rr=r.copy();rr['xt']=rr.et+pd.Timedelta(hours=6)
def opp(row):
    m=r[(r.et<=row.etl)&(r.et.dt.normalize()==row.date)]
    if not len(m):return False
    t=m.iloc[-1];return t.s!=(1 if str(row.dir).lower() in('1','long','buy') else -1) and t.why!='x' and row.etl.hour<16
# approximate open clara pos: clara trade entered earlier same day & not stopped before revamp entry unknown -> use exit time unknown; flag approximation
V['d1_skip_opp_open_clara(approx)']=[L[~(day&L.apply(opp,axis=1))]]
V['e1_BASE+clara35_hiconf(eff>=.5&orw_rel<1)']=[L,CL[CL.et.dt.normalize().map(lambda d:(dayf.eff.get(d,0)>=.5)&(dayf.orw_rel.get(d,9)<1)).astype(bool)]]
V['a1+b3+c2']=[L[~(known&(dirn!=L.cs))&~(day&trendday&(L.setup=='ens_mean_reversion'))&~(day&(L.eff<0.35)&L.setup.isin(['ens_trend','ens_breakout']))],cfilter(CL,'agree')]
rows=[]
for k,parts in V.items():
    t=pd.concat([p[['et','xt','usd','wo']] for p in parts]).sort_values('et');t['d']=(t.et+pd.Timedelta(hours=6)).dt.normalize()
    o={'variant':k}
    for per,(a,b) in [('tr',TR),('26',TE)]:
        m=metrics(t[(t.d>=a)&(t.d<=b)],'wo','');o[f'net_{per}']=m['net'];o[f'pf_{per}']=m['pf'];o[f'dd_{per}']=m['intraday_dd'];o[f'p/dd_{per}']=round(m['net']/m['intraday_dd'],2)
    q=t.groupby(t.et.dt.to_period('Q')).usd.sum();o['q+']=f"{(q>0).sum()}/{len(q)}";rows.append(o)
X=pd.DataFrame(rows);print(X.to_string());X.to_csv('fuse.csv',index=False)
print(L.cet.notna().mean(), dirn.value_counts().to_dict())
B=L[known&(dirn!=L.cs)];print('a1 blocked',len(B),B.usd.sum().round());print(B[['etl','cet','cs','dir','setup','sess','usd']].head(12).to_string())
