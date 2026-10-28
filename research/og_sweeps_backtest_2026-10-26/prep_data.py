import pandas as pd, numpy as np
B=r'C:\Users\shawa\Projects\bott-trader\data'
m=pd.read_csv(B+r'\MNQ_1m.csv',parse_dates=['datetime'])
m['datetime']=pd.to_datetime(m['datetime'],utc=True)
m=m.drop_duplicates('datetime').sort_values('datetime').set_index('datetime')
print('1m rows',len(m),m.index.min(),m.index.max())
g=m.resample('5min',label='left',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['open'])
print('5m rows from 1m',len(g))
l=pd.read_csv(B+r'\live_bars_MNQ_5m.csv')
l['datetime']=pd.to_datetime(l['ts_et'],utc=True)
l=l.drop_duplicates('datetime').set_index('datetime')[['open','high','low','close','volume']]
print('live 5m',len(l),l.index.min(),l.index.max())
# per-month counts and gaps
c=g.groupby(g.index.to_period('M')).size(); print(c.to_string())
gaps=g.index.to_series().diff(); big=gaps[gaps>pd.Timedelta(hours=6)]
print('gaps>6h:',len(big)); print(big.tail(40).to_string())
g['src']='1m'; l['src']='live'
allb=pd.concat([g,l[~l.index.isin(g.index)]]).sort_index()
allb.reset_index().to_csv('bars5m.csv',index=False)
print(allb.shape, allb.index.min(), allb.index.max())
print(m['volume'].describe())
