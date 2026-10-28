import sys,os;sys.path.insert(0,'impl');os.environ['REVAMP_CLARA_BIAS']='1'
import pandas as pd,pickle,numpy as np,clara_bias as cb
exec(open('combo2.py').read().split("rows=[]")[0])
m=pd.read_csv('MNQ_1m.csv');tc=[c for c in m.columns if 'time' in c.lower() or c.lower()in('ts','date','datetime')][0]
m.index=pd.to_datetime(m[tc],utc=True);m=m[['open','high','low','close','volume']]
m5=m.resample('2min',label='left',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
m5et=m5.copy();m5et.index=m5.index.tz_convert('America/New_York')
comp=pickle.load(open('comp.pkl','rb'));R=pickle.load(open('revamp/revamp_trades.pkl','rb'))['live_all_max3']
L=comp['live_all_max3:ALL'].copy();et=pd.to_datetime(R.entry_time.values)
et=(et.tz_localize('UTC') if et.tz is None else et).tz_convert('America/New_York')
keep=[]
for e,dr in zip(et,R.dir.values):
    sub=m5[(m5.index>=(e.normalize()+pd.Timedelta(hours=9)).tz_convert('UTC'))&(m5.index<e.tz_convert('UTC'))]
    keep.append(cb.block_reason(sub,'LONG' if str(dr).lower() in('1','long','buy') else 'SHORT',e,2) is None)
keep=np.array(keep);print('blocked',(~keep).sum())
for nm,t in [('BASE',L),('a1_5m_live_module',L[keep])]:
    t=t.copy();t['d']=(t.et+pd.Timedelta(hours=6)).dt.normalize();o=[nm]
    for a,b in [TR,TE]:
        x=metrics(t[(t.d>=a)&(t.d<=b)],'wo','');x.pop('breach_2k_dates');o.append({k:x[k] for k in['n','net','pf','wr','avg_win','avg_loss','intraday_dd','eod_dd','worst_day','max_consec_losses','trades_per_day']})
    q=t.groupby(t.et.dt.to_period('Q')).usd.sum().round();o.append(q.to_dict());print(o)
