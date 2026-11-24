import pandas as pd,numpy as np,sys
import sr_x10_lib as L
ET=L.ET
d=pd.read_csv('trades_base.csv',parse_dates=['entry_time','exit_time'])
for c in ('entry_time','exit_time'): d[c]=pd.to_datetime(d[c],utc=True)
e=d.entry_time.dt.tz_convert(ET); d['m']=e.dt.hour*60+e.dt.minute; d['hr']=e.dt.hour
d['win']=np.where(d.entry_time<pd.Timestamp('2026-01-01',tz='UTC'),'train','y2026')
def seg(m):
    if 570<=m<960: return '1_day 09:30-16:00'
    if 960<=m<1080: return '2_evening 16:00-18:00'
    if m>=1080 or m<120: return '3_night 18:00-02:00'
    return '4_overnight 02:00-09:30'
d['seg']=d.m.map(seg)
d['sleeve']=d.tag.map({'ens_breakout':'breakout','ens_trend':'trend','ens_mean_reversion':'mean_rev','addon_e1':'6PM long (e1)','addon_onl':'ON-low long (onl)'})
d['side']=np.where(d.dir>0,'long','short')
d.to_pickle('x10_base.pkl')
def row(g,**k):
    r=dict(k); a=L.met(g); r.update({'n':a['n'],'net':a['net'],'pf':a['pf'],'avg':a['avg'],'dd':a['dd']})
    for w in ('train','y2026'):
        b=L.met(g[g.win==w]); r[w+'_n']=b['n']; r[w+'_net']=b['net']; r[w+'_pf']=b['pf']; r[w+'_avg']=b['avg']
    mo=g.groupby('month').pnl.sum(); r['mo_green']=f"{int((mo>0).sum())}/{len(mo)}"
    p=g.pnl.values; r['t']=round(float(p.mean()/(p.std(ddof=1)/np.sqrt(len(p)))),2) if len(p)>2 and p.std()>0 else 0
    return r
pd.set_option('display.width',320); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',500)
out=[]
out.append(row(d,cut='ALL',seg='ALL'))
for s,g in d.groupby('seg'): out.append(row(g,cut='session',seg=s))
for (s,sd),g in d.groupby(['seg','side']): out.append(row(g,cut='session x side',seg=s,side=sd))
for (s,sl),g in d.groupby(['seg','sleeve']): out.append(row(g,cut='session x sleeve',seg=s,sleeve=sl))
for (s,sl,sd),g in d.groupby(['seg','sleeve','side']): out.append(row(g,cut='session x sleeve x side',seg=s,sleeve=sl,side=sd))
n18=d[d.seg.str[0].isin(['2','3','4'])]
out.append(row(n18,cut='ALL 16:00-09:30',seg='eve+night+ovn'))
n18b=d[d.seg.str[0].isin(['3','4'])]
out.append(row(n18b,cut='ALL 18:00-09:30',seg='night+ovn'))
for sd,g in n18b.groupby('side'): out.append(row(g,cut='18:00-09:30 by side',seg='night+ovn',side=sd))
for sl,g in n18b.groupby('sleeve'): out.append(row(g,cut='18:00-09:30 by sleeve',seg='night+ovn',sleeve=sl))
T=pd.DataFrame(out).fillna(''); T.to_csv('x10_part1.csv',index=False); print(T.to_string())
# hourly
hh=[]
for h,g in d[d.seg.str[0].isin(['3','4'])].groupby('hr'):
    hh.append(row(g,cut='hour',hr=h))
H=pd.DataFrame(hh); H.to_csv('x10_hourly.csv',index=False); print(H[['hr','n','net','pf','avg','dd','train_n','train_net','y2026_n','y2026_net','mo_green','t']].to_string())
# breakout by hour
hb=[]
for h,g in d[(d.sleeve=='breakout')&d.seg.str[0].isin(['3','4'])].groupby('hr'): hb.append(row(g,cut='bo hour',hr=h))
Hb=pd.DataFrame(hb); Hb.to_csv('x10_hourly_bo.csv',index=False); print(Hb[['hr','n','net','pf','avg','train_n','train_net','y2026_n','y2026_net','mo_green','t']].to_string())
# e1 timing
e1=d[d.tag=='addon_e1']; print('e1 entry hour/min dist'); print(e1.m.map(lambda m:f'{m//60:02d}:{(m%60)//30*30:02d}').value_counts().sort_index().to_string()); print(e1.reason.value_counts().to_dict(), e1.dir.value_counts().to_dict())
