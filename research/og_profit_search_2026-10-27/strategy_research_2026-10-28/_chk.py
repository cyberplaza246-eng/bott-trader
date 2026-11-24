import sys,io,contextlib,json
src=open('sr_gap2.py',encoding='utf-8-sig').read().split('# ---- run ----')[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src,globals())
mine=json.load(open(r'C:\Users\shawa\gf_seed_check.json'))['past_abs']
ref=[round(abs(e['mv']),2) for e in EV['ny']]
print(len(ref),len(mine),ref[:5],mine[:5],ref[-3:],mine[-3:])
print(EV['ny'][0]['ds'],EV['ny'][-1]['ds'])
import numpy as np
n=0
for k in range(20,len(ref)):
    if ref[k]>=np.quantile(ref[:k],.75): n+=1
print('ref events fire',n)
# trades with ok_dt
r=run_ev(EV['ny'],'q',.75,90,1); print(len(r))

k=0
while k<len(ref) and ref[-1-k]==mine[-1-k]: k+=1
print('tail equal',k,'of',len(ref),'; first diverging idx from end',k, ref[-1-k],mine[-1-k],EV['ny'][len(ref)-1-k]['ds'])
import json,pandas as pd,numpy as np
# recompute my per-date gaps
df=pd.read_csv(r'..\..\..\data\MNQ_1m.csv',usecols=['datetime','open','close']); df['t']=pd.to_datetime(df.datetime,utc=True); df=df.set_index('t').sort_index()
et=df.index.tz_convert('America/New_York'); df['d']=et.strftime('%Y-%m-%d'); df['m']=et.hour*60+et.minute
op={};cl={}
for d,g in df.groupby('d'):
    o=g[(g.m>=570)&(g.m<=572)]; c=g[(g.m>=957)&(g.m<=959)]
    if o.empty or c.empty or (c.index[-1]-o.index[0]).total_seconds()/60<300: continue
    op[d]=float(o.iloc[0].open); cl[d]=float(c[c.m==c.m.max()].iloc[-1].close)
ds=sorted(op); mine={ds[k]:op[ds[k]]-cl[ds[k-1]] for k in range(1,len(ds))}
refd={e['ds']:e['mv'] for e in EV['ny']}
bad=[(d,round(refd[d],2),round(mine[d],2)) for d in refd if d in mine and abs(refd[d]-mine[d])>0.01]
print('common',len([d for d in refd if d in mine]),'mismatch',len(bad)); print(bad[:12]); print([d for d in refd if d not in mine][:5]); print('mine only',len([d for d in mine if d not in refd]),sorted([d for d in mine if d not in refd])[:8])

print([ (d,round(a,1),round(b,1)) for d,a,b in bad])
r=pd.DataFrame(run_ev(EV['ny'],'q',.75,90,1))
r['d']=r.entry_time.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
bd=set(d for d,_,_ in bad)
x=r[r.d.isin(bd)]
print('traded on mismatch dates:',len(x),'pnl',round(x.pnl.sum()),'; rest:',len(r)-len(x),'pnl',round(r[~r.d.isin(bd)].pnl.sum()))
print(x[['d','dir','pnl']].to_string())
