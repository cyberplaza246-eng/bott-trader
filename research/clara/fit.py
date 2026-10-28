import pandas as pd,itertools
from sim import run,stats,days
c=pd.read_pickle('t.pkl');c['d']=c.et.dt.normalize();c['s']=(c['Market pos.']=='Long')*2-1
tr={d:g for d,g in days.items() if d<pd.Timestamp('2026-01-01')}
cf=c[(c.d>='2024-09-25')&(c.d<'2026-01-01')].groupby('d').head(1).set_index('d')
res=[]
for wmax,k,lockR,trail,cut in itertools.product([250,400],[0.8,1.0],[0.5,1.0],[0.6,9],[11,14]):
    P=dict(wmax=wmax,k=k,lockR=lockR,lockf=0.4,trail=trail,cut=cut,buf=0,smin=20)
    t=run(P,tr);t['d']=t.et.dt.normalize();rf=t.groupby('d').head(1).set_index('d')
    both=rf.index.intersection(cf.index)
    st=stats(t);st.update(P);st['day_recall']=round(len(both)/len(cf),3);st['day_prec']=round(len(both)/len(rf),3)
    st['side_agree']=round((rf.loc[both,'s']==cf.loc[both,'s']).mean(),3)
    st['time_within6m']=round(((rf.loc[both,'et']-cf.loc[both,'et']).abs()<=pd.Timedelta('6min')).mean(),3)
    st['fid']=st['day_recall']+st['day_prec']+st['side_agree']+st['time_within6m']-abs(st['wr']-0.709)*3
    res.append(st);print(st,flush=True)
pd.DataFrame(res).sort_values('fid',ascending=False).to_csv('fit.csv',index=False)
