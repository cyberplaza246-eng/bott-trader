import os, sys, time
SB=os.path.abspath('sandbox'); os.chdir(SB); sys.path.insert(0,SB)
os.environ.update(dict(PYTHONUTF8='1',FALLBACK_ENTRIES_ENABLED='false',ASSET_CLASS='futures',TRADING_SYMBOLS='MNQ',TRADING_MODE='backtest',BROKER_TYPE='rithmic',NEWSAPI_KEY=''))
import pandas as pd, numpy as np
t=time.time()
from src.core.ensemble_trader import EnsembleTrader
from src.ai.technical_analyzer import TechnicalAnalyzer
print('import',time.time()-t)
et=EnsembleTrader(broker=None)
print('init',time.time()-t, 'intel',et.intelligent_available,'adv',et.advanced_strats_available,'lstm',et.lstm_available,'rl',et.rl_available,'thr',et.learner.get_adjusted_threshold())
class S: 
    pass
et.sentiment.get_pair_sentiment=lambda pair:{'sentiment_score':0.0,'news_count':0}
b=pd.read_csv('../bars5m.csv',parse_dates=['datetime'])
b=b[b.datetime>='2026-03-01'].reset_index(drop=True)
ta=TechnicalAnalyzer()
n=0; t=time.time(); sigs=[]
for i in range(300,360):
    df=b.iloc[i-99:i+1].copy().reset_index(drop=True)
    dfe=ta.calculate_indicators(df)
    r=et.get_trading_signal(dfe,'MNQ')
    sigs.append((r['signal'],r['confidence']))
print('per call',(time.time()-t)/60)
print(sigs[:20])
print(list(dfe.columns))
