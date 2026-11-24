import pandas as pd,numpy as np,ast
from x10_an import *
R=pd.read_csv('x10_results.csv'); Rd=R[~R.group.isin(['G7'])].copy()
b=base.copy(); b['xday']=b.exit_time.dt.tz_convert(ET).dt.strftime('%Y-%m-%d')
b['fold']=sum((b.entry_time>edges[i]).astype(int) for i in range(1,6))
b['key']=b.entry_time.astype(str)+'|'+b.tag+'|'+b.dir.astype(str)
e=b.entry_time.dt.tz_convert(ET); b['m']=e.dt.hour*60+e.dt.minute
nt=b[(b.m>=1080)|(b.m<570)].reset_index(drop=True)
days=sorted(set(b.xday)); di={d:i for i,d in enumerate(days)}; nt['di']=nt.xday.map(di); ndays=len(days)
rng=np.random.default_rng(5); BI=rng.integers(0,ndays,(300,ndays))
pn=nt.pnl.values; trn=(nt.win=='train').values; fo=nt.fold.values; dix=nt.di.values
def screen(s):
    d=-pn[s]; dt=d[trn[s]].sum(); d26=d[~trn[s]].sum()
    if dt<=0 or d26<=0: return False
    if (np.bincount(fo[s],weights=d,minlength=6)>0).sum()<4: return False
    dv=np.bincount(dix[s],weights=d,minlength=ndays); return (dv[BI].sum(1)>0).mean()>=0.9
cache={}
def null(k,ndraw=3000):
    if k in cache: return cache[k]
    pas=0; ds=np.empty(ndraw)
    for i in range(ndraw):
        s=rng.choice(len(nt),k,replace=False); ds[i]=-pn[s].sum(); pas+=screen(s)
    cache[k]=(pas/ndraw,ds); return cache[k]
rows=[]
for _,r in Rd.iterrows():
    v=load(f'x10_trades_{r["name"]}.csv'); vk=set(v.entry_time.astype(str)+'|'+v.tag+'|'+v.dir.astype(str))
    rem=b[~b.key.isin(vk)]; rem=rem[(rem.m>=1080)|(rem.m<570)]; k=len(rem)
    if k==0: rows.append((r['name'],0,0,np.nan,np.nan)); continue
    sd=-rem.pnl.sum(); pr,ds=null(min(k,len(nt)-1))
    rows.append((r['name'],k,round(sd),pr,float((ds>=sd).mean())))
NR=pd.DataFrame(rows,columns=['name','k_removed_night','standalone_delta','null_screen_pass_rate','emp_p'])
R2=Rd.merge(NR,on='name'); R2.to_csv('x10_results2.csv',index=False)
pd.set_option('display.width',300); pd.set_option('display.max_rows',100)
print('N variants',len(R2),' expected chance passes of the screen (sum of null pass rates):',round(R2.null_screen_pass_rate.sum(),2))
scr=(R2.d_train>0)&(R2.d_2026>0)&(R2.folds_better>=4)&(R2.pbet_all>=0.9)
print('observed passes:',R2[scr].name.tolist())
print(R2.sort_values('emp_p').head(10)[['name','k_removed_night','standalone_delta','d_all','d_train','d_2026','folds_better','pbet_all','null_screen_pass_rate','emp_p']].to_string())
mp=R2.emp_p.min(); print('min emp p',round(mp,4),'Sidak over',len(R2),'=',round(1-(1-max(mp,1/3000))**len(R2),3))
# same for the candidate in isolation (42 trend shorts)
print(R2[R2.name=='off_18_0930_trend_short'][['k_removed_night','standalone_delta','emp_p']].to_string())
