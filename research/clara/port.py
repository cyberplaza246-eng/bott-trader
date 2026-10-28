exec(open('combo2.py').read().split("rows=[]")[0])
import warnings;warnings.filterwarnings('ignore')
comp=pickle.load(open('comp.pkl','rb'))
P1=prep_rev(pickle.load(open('revamp/p1.pkl','rb'))['P1'])
base=comp['live_all_max3:ALL']
PF={'LIVE_current':[base],'P1_drop_train_losers':[P1],'P1+6pm':[P1,comp['6pm_trail_noMon']],'P1+clara_eff35':[P1,comp['clara_eff35']],'LIVE+6pm':[base,comp['6pm_trail_noMon']]}
WF=(pd.Timestamp('2025-01-01'),pd.Timestamp('2026-09-21'))
for k,parts in PF.items():
    t=pd.concat(parts).sort_values('et');t['d']=(t.et+pd.Timedelta(hours=6)).dt.normalize()
    for per,(a,b) in [('train',TR),('test26',TE)]:
        r=metrics(t[(t.d>=a)&(t.d<=b)],'wo',f'{k}|{per}');br=r.pop('breach_2k_dates')
        r['breach_days']=len(br);r['breach_first_last']=(br[0],br[-1]) if br else None
        r['net/dd']=round(r['net']/max(r['intraday_dd'],1),2);r['acct_est']=round(r['intraday_dd']*1.5)
        print(json.dumps(r,default=str))
    q=t.groupby(t.et.dt.to_period('Q')).usd.sum().round();print(k,'quarters',q.to_dict())
