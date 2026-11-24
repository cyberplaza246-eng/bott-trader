import yfinance as yf, pandas as pd
d=yf.download('NQ=F',period='5d',interval='1m',auto_adjust=False,progress=False,prepost=True)
if hasattr(d.columns,'levels'): d.columns=d.columns.get_level_values(0)
d.index=d.index.tz_convert('America/New_York'); d.to_csv('x9_nq_1m_today.csv'); print(len(d),d.index.min(),d.index.max())
t=d[d.index.strftime('%Y-%m-%d')=='2026-10-28']
print(len(t))
r=t.resample('15min').agg({'Open':'first','High':'max','Low':'min','Close':'last'})
r=r[(r.index.hour>=9)&(r.index.hour<=17)]
print(r.round(2).to_string())
