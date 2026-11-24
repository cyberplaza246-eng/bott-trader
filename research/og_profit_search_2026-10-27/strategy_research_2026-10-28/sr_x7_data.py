"""Addendum 7 data layer (BACKTEST ONLY): overnight (16:00 ET prior -> 09:00 ET) features per symbol from free yfinance hourly bars; daily features known pre-open."""
from sr_x import *
SYMS=['ES','NQ','YM','RTY','VIX','VIX3M','VIX9D','VIXY','UVXY','DXY','UUP','EURUSD','USDJPY','TNX','FVX','IRX','ZN','ZB','ZT','GC','CL','BTC','ETH','IWM','RUT','RSP','SPY','QQQ','SOXX','SMH','NVDA','AAPL','MSFT','N225','DAX','FTSE','HSI','STOXX','SPX','NDX','HYG','LQD','TLT','XLK','TSLA','AMZN','META','GOOGL']
DIFF={'TNX','FVX','IRX'}   # yields: level change, not pct
def loadh(s,iv='1h'):
    d=pd.read_csv(f'yf7/{s}_{iv}.csv',index_col=0); d.index=pd.to_datetime(d.index,utc=True); d=d[['Open','High','Low','Close']].apply(pd.to_numeric,errors='coerce').dropna(); return d[~d.index.duplicated()].sort_index()
H1={s:loadh(s) for s in SYMS}
def asof_close(d,t):
    """Close of last bar whose START <= t (t tz-aware ET)"""
    k=d.index.searchsorted(t,side='right')-1
    return (d.Close.iloc[k],d.index[k]) if k>=0 else (np.nan,None)
def ret(s,v1,v0): return (v1-v0) if s in DIFF else (np.log(v1/v0) if v0>0 and v1>0 else np.nan)
rows=[]
for k,D in enumerate(days):
    if k==0: continue
    pD=days[k-1]
    t0=pd.Timestamp(f'{pD} 15:59',tz=ET); t1=pd.Timestamp(f'{D} 08:00',tz=ET)
    r=dict(d=D)
    i9=pos(D,'09:00',2)
    r['MNQ']=np.log(o[i9]/sess[pD]['cl']) if i9 is not None else np.nan
    for s in SYMS:
        d=H1[s]; v0,b0=asof_close(d,t0); v1,b1=asof_close(d,t1)
        if b0 is None or b1 is None or (t0-b0)>pd.Timedelta(hours=30) or (t1-b1)>pd.Timedelta(hours=30): r[s]=np.nan; continue
        r[s]=ret(s,v1,v0)
    rows.append(r)
OV=pd.DataFrame(rows).set_index('d'); OV.index=pd.to_datetime(OV.index)
OV.to_csv('x7_ovn.csv')
cov=pd.DataFrame(dict(cover=OV.notna().sum(),zero_share=(OV==0).sum()/OV.notna().sum(),corr_mnq=OV.corrwith(OV['MNQ']),std_bp=OV.std()*1e4)).round(3)
print(len(OV),'days'); print(cov.sort_values('corr_mnq',ascending=False).to_string())
# daily features: last completed session close/close, levels, term structure
def loadd(s):
    d=pd.read_csv(f'yf7/{s}_1d.csv',index_col=0,parse_dates=True); d.index=pd.to_datetime(d.index,utc=True).tz_localize(None).normalize(); return d['Close'].astype(float).dropna()
D1={s:loadd(s) for s in SYMS}
DF=pd.DataFrame({s:(D1[s].diff() if s in DIFF else np.log(D1[s]).diff()) for s in SYMS})
DF['VIXlvl']=D1['VIX']; DF['VIXterm']=D1['VIX']/D1['VIX3M']; DF['VIX9D_VIX']=D1['VIX9D']/D1['VIX']
DF['RSPSPY']=np.log(D1['RSP']/D1['SPY']).diff(); DF['IWMSPY']=np.log(D1['IWM']/D1['SPY']).diff(); DF['SMHQQQ']=np.log(D1['SMH']/D1['QQQ']).diff()
DF.to_csv('x7_daily.csv'); print(DF.shape,DF.index.min(),DF.index.max())