import yfinance as yf, pandas as pd, sys
T={'ES':'ES=F','NQ':'NQ=F','YM':'YM=F','RTY':'RTY=F','VIX':'^VIX','VIX3M':'^VIX3M','VIX9D':'^VIX9D','VIXY':'VIXY','UVXY':'UVXY','DXY':'DX-Y.NYB','UUP':'UUP','EURUSD':'EURUSD=X','USDJPY':'JPY=X','TNX':'^TNX','FVX':'^FVX','IRX':'^IRX','ZN':'ZN=F','ZB':'ZB=F','ZT':'ZT=F','GC':'GC=F','CL':'CL=F','BTC':'BTC-USD','ETH':'ETH-USD','IWM':'IWM','RUT':'^RUT','RSP':'RSP','SPY':'SPY','QQQ':'QQQ','SOXX':'SOXX','SMH':'SMH','NVDA':'NVDA','AAPL':'AAPL','MSFT':'MSFT','N225':'^N225','DAX':'^GDAXI','FTSE':'^FTSE','HSI':'^HSI','STOXX':'^STOXX50E','SPX':'^GSPC','NDX':'^NDX','HYG':'HYG','LQD':'LQD','TLT':'TLT','XLK':'XLK','TSLA':'TSLA','AMZN':'AMZN','META':'META','GOOGL':'GOOGL'}
for k,tk in T.items():
    for iv,per in (('1h','730d'),('5m','60d'),('1d','max')):
        try:
            d=yf.download(tk,period=per,interval=iv,auto_adjust=False,progress=False,prepost=True)
            if hasattr(d.columns,'levels'): d.columns=d.columns.get_level_values(0)
            d.to_csv(f'yf7/{k}_{iv}.csv'); print(k,iv,len(d),str(d.index.min())[:16],str(d.index.max())[:16],flush=True)
        except Exception as e: print(k,iv,'ERR',str(e)[:100],flush=True)