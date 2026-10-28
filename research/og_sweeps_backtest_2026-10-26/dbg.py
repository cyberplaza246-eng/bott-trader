import os, sys, logging
SB=os.path.abspath('sandbox'); os.chdir(SB); sys.path.insert(0,SB)
os.environ.update(dict(FALLBACK_ENTRIES_ENABLED='false',ASSET_CLASS='futures',TRADING_SYMBOLS='MNQ',TRADING_MODE='backtest',BROKER_TYPE='rithmic',NEWSAPI_KEY=''))
import pandas as pd
from src.core.ensemble_trader import EnsembleTrader
from src.ai.technical_analyzer import TechnicalAnalyzer
et=EnsembleTrader(broker=None)
et.sentiment.get_pair_sentiment=lambda pair:{'sentiment_score':0.0,'news_count':0}
b=pd.read_csv('../bars5m.csv',parse_dates=['datetime']); b['datetime']=pd.to_datetime(b['datetime'],utc=True)
i=b.index[b.datetime=='2026-10-26 11:55:00+00:00'][0]
df=b.iloc[i-99:i+1].copy().reset_index(drop=True)
dfe=TechnicalAnalyzer().calculate_indicators(df)
print('=====START')
r=et.get_trading_signal(dfe,'MNQ'); print(r['signal'],r['confidence'])
