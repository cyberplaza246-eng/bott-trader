import yfinance as yf, pandas as pd
for tk in ('ES=F','NQ=F'):
    for iv,per in (('1h','730d'),('30m','60d'),('1d','max')):
        try:
            d=yf.download(tk,period=per,interval=iv,auto_adjust=False,progress=False,prepost=True)
            if hasattr(d.columns,'levels'): d.columns=d.columns.get_level_values(0)
            print(tk,iv,len(d),d.index.min(),d.index.max())
            d.to_csv(f'yf_{tk[:2]}_{iv}.csv')
        except Exception as e: print(tk,iv,'ERR',str(e)[:150])