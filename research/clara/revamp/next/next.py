import os;os.chdir('/workspace/clara')
import sys;sys.path.insert(0,'/workspace/clara')
exec(open('combo2.py').read().split("rows=[]")[0])
import warnings;warnings.filterwarnings('ignore')
exec(open('solo.py').read().split("P={")[0])   # L with setup, dd(), basic()
comp=pickle.load(open('comp.pkl','rb'));S6=comp['6pm_trail_noMon'].copy()
h=L.et.dt.hour+L.et.dt.minute/60
ob=L.setup.eq('ens_breakout@overnight');dt=L.setup.eq('ens_trend@day');db=L.setup.eq('ens_breakout@day')
def win(a,b):return ob&(h>=a)&(h<b)
V={}
V['BASE']=(~L.index.isin([]),None)
V['T1_noDayTrend']=(~dt,None)
for a,b in[(0,9),(0,4),(4,9),(2,8),(0,6)]:V[f'T2_cutOB_{a}-{b}']=(~win(a,b),None)
def mk(keep,mult,extra=None):return(keep,mult,extra)
rows=[];store={}
def run(k,keep,mult=None,extra=None):
    t=L[keep].copy();m=pd.Series(1.0,index=t.index) if mult is None else mult[keep]
    t['usd']*=m;t['wo']*=m;parts=[t[['et','xt','usd','wo','d']]]
    if extra is not None:
        e=extra.copy();e['d']=(e.et+pd.Timedelta(hours=6)).dt.normalize();parts.append(e[['et','xt','usd','wo','d']])
    t=pd.concat(parts).sort_values('et').reset_index(drop=True);store[k]=t
    o={'v':k}
    for nm,(a,b) in [('full',(pd.Timestamp('2000-01-01'),pd.Timestamp('2030-01-01'))),('tr',TR),('26',TE)]:
        x=t[(t.d>=a)&(t.d<=b)];m_=metrics(x,'wo','');o[f'net_{nm}']=m_['net']
        if nm!='26':
            pass
        if nm=='full':
            o.update(pf=m_['pf'],wr=m_['wr'],n=m_['n'],idd=m_['intraday_dd'],eod=m_['eod_dd'],breach=len(m_['breach_2k_dates']),acct=round(1.5*m_['intraday_dd']))
            mo=x.groupby(x.d.dt.to_period('M')).usd.sum();o['mo+']=f"{(mo>0).sum()}/{len(mo)}"
        if nm=='tr':o['idd_tr']=m_['intraday_dd'];o['pdd_tr']=round(m_['net']/m_['intraday_dd'],2)
        if nm=='26':o['idd_26']=m_['intraday_dd']
    q=t.groupby(t.d.dt.to_period('Q')).usd.sum();o['q+']=f"{(q>0).sum()}/{len(q)}";rows.append(o)
for k,(keep,_) in V.items(): run(k,keep)
X=pd.DataFrame(rows);print(X.to_string())
# pick trim on train p/dd then net
tr=X[X.v!='BASE'].sort_values(['net_tr'],ascending=False);print('trim by train net:',tr.v.iloc[0])
rows.clear()
trim=~win(2,8)
one=pd.Series(1.0,index=L.index)
wed=L.d.dt.dayofweek.eq(2)
# train-only check for sleeve strength
print(L[L.d<'2026-01-01'].groupby('setup').usd.agg(['size','sum']).round())
print(L[L.d<'2026-01-01'].groupby(L.d.dt.day_name()).usd.sum().round().to_dict())
D={'BASE':(one.index==one.index,one,None),
 'D1_2x_all':(one.index==one.index,one*2,None),
 'D2_2x_dayBO':(one.index==one.index,one.where(~db,2),None),
 'D3_trim+2x_dayBO':(trim,one.where(~db,2),None),
 'D4_trim+2x_dayBO+6pm1x':(trim,one.where(~db,2),S6),
 'D5_trim+3x_dayBO':(trim,one.where(~db,3),None),
 'D6_trim+2x_all':(trim,one*2,None),
 'D7_trim+2x_dayBO_E1_ONL_MRday':(trim,one.where(~(db|L.setup.isin(['addon_e1@overnight','addon_onl@day','ens_mean_reversion@day'])),2),None),
 'D8_trim+2x_dayBO+2xWed':(trim,one.where(~db,2)*np.where(wed,2,1),None)}
for k,(keep,m,e) in D.items(): run(k,keep,m,e)
X=pd.DataFrame(rows);print(X.to_string());X.to_csv('revamp/next/size.csv',index=False)
