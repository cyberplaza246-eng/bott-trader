import os,sys,time,logging,json,glob
sys.dont_write_bytecode=True
H=os.path.dirname(os.path.abspath(__file__)); SB=os.path.join(H,'sandbox'); os.chdir(SB); sys.path.insert(0,SB)
os.environ.update(dict(PYTHONUTF8='1',FALLBACK_ENTRIES_ENABLED='true',ASSET_CLASS='futures',TRADING_SYMBOLS='MNQ',TRADING_MODE='backtest',BROKER_TYPE='rithmic',NEWSAPI_KEY='',RSI_BUY_BLOCK='70',RSI_SELL_BLOCK='30'))
import pandas as pd
import datetime as _real, types
class FDT(_real.datetime):
    _now=None
    @classmethod
    def utcnow(cls): return cls._now.replace(tzinfo=None) if cls._now is not None else _real.datetime.utcnow()
    @classmethod
    def now(cls,tz=None):
        if tz is None or cls._now is None: return _real.datetime.now(tz)
        return cls._now.astimezone(tz)
_shim=types.ModuleType('datetime'); _shim.__dict__.update(_real.__dict__); _shim.datetime=FDT; sys.modules['datetime']=_shim
from src.core.ensemble_trader import EnsembleTrader
from src.ai.technical_analyzer import TechnicalAnalyzer
from src.utils.logger import bot_logger
for h in list(bot_logger.handlers): bot_logger.removeHandler(h)
logging.disable(logging.CRITICAL)
et=EnsembleTrader(broker=None); et.sentiment.get_pair_sentiment=lambda pair:{'sentiment_score':0.0,'news_count':0}
ta=TechnicalAnalyzer()
b=pd.read_csv(os.path.join(H,'data','bars5m_clean.csv'),parse_dates=['datetime']); b['datetime']=pd.to_datetime(b['datetime'],utc=True)

import random,numpy as np
random.seed(3); idxs=random.sample(range(30000,120000),25)
def run(i,reduced):
    df=b.iloc[i-99:i+1].copy().reset_index(drop=True); FDT._now=b.datetime.iloc[i].to_pydatetime()
    dfe=ta.calculate_indicators(df)
    sw=et.sweep.get_signal(dfe,'MNQ',df_5m=dfe.tail(250).copy(),spread=None)
    sf=sw.get('signal') in ('BUY','SELL')
    et.lstm_available=et.intelligent_available=et.advanced_strats_available=(not reduced) or sf
    r=et.get_trading_signal(dfe,'MNQ')
    return sf,r['signal'],r['confidence'],r.get('regime'),r.get('rsi')
t0=time.time(); full=[run(i,False) for i in idxs]; t1=time.time(); red=[run(i,True) for i in idxs]; t2=time.time()
bad=[(i,a,b_) for i,a,b_ in zip(idxs,full,red) if (a[1],round(a[2],4))!=(b_[1],round(b_[2],4)) or a[3]!=b_[3]]
print('full s/bar',(t1-t0)/25,'reduced s/bar',(t2-t1)/25,'mismatch',len(bad),'nonsweep',sum(1 for a in full if not a[0]),'sigs',sum(1 for a in full if a[1] in('BUY','SELL')))
print(bad[:5])
