exec(open('combo2.py').read().split("rows=[]")[0])
import warnings;warnings.filterwarnings('ignore')
R2=pickle.load(open('revamp/revamp_trades.pkl','rb'))
comp={}
for rk in ['live_all_max3','trend_ovn_max1']:
    t=rev[rk].copy();raw=R2[rk];t['tag']=raw.tag.values+'@'+raw.session.values
    for g,s in t.groupby('tag'): comp[f'{rk}:{g}']=s
    comp[f'{rk}:ALL']=t
s6=pd.read_pickle('six.pkl');s6['xt']=pd.to_datetime(s6.xt);comp['6pm_trail_noMon']=s6[['et','xt','usd','wo']]
comp['clara_eff35']=CL['clara_eff35'];comp['clara_base']=CL['clara_base']
def st(t,a,b):
    t=t.copy();t['d']=(t.et+pd.Timedelta(hours=6)).dt.normalize();t=t[(t.d>=a)&(t.d<=b)]
    if len(t)<10: return None
    r=metrics(t,'wo','');return r
rows=[]
for k,t in comp.items():
    a=st(t,*TR);b=st(t,*TE)
    if a is None or b is None: continue
    # walk-forward stability: quarters positive
    t2=t.copy();q=t2.et.dt.to_period('Q');qs=t2.groupby(q).usd.sum()
    rows.append(dict(comp=k,n_tr=a['n'],net_tr=a['net'],pf_tr=a['pf'],net_26=b['net'],pf_26=b['pf'],dd_26=b['intraday_dd'],q_pos=f"{(qs>0).sum()}/{len(qs)}"))
X=pd.DataFrame(rows).sort_values('net_tr',ascending=False);print(X.to_string());X.to_csv('rank.csv',index=False)
pickle.dump(comp,open('comp.pkl','wb'))
