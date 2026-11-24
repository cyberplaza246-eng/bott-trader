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
et.lstm_available=False
ta=TechnicalAnalyzer()
b=pd.read_csv(os.path.join(H,'data','bars5m_clean.csv'),parse_dates=['datetime']); b['datetime']=pd.to_datetime(b['datetime'],utc=True)
tmap={str(t):i for i,t in enumerate(b.datetime)}
old=[]
for f in sorted(glob.glob(os.path.join(H,'..','og_sweeps_backtest_2026-10-26','out','sig_*.jsonl')))[:6]:
    for l in open(f):
        r=json.loads(l)
        if r.get('sweep') in ('BUY','SELL') and 'sig' in r and r['t'] in tmap: old.append(r)
import random; random.seed(1); old=random.sample(old,12)
t0=time.time()
for r in old:
    i=tmap[r['t']]; df=b.iloc[i-99:i+1].copy().reset_index(drop=True); FDT._now=b.datetime.iloc[i].to_pydatetime()
    dfe=ta.calculate_indicators(df); n=et.get_trading_signal(dfe,'MNQ')
    print(r['t'],r['sig'],round(r['conf'],4),'->',n['signal'],round(n['confidence'],4),flush=True)
print('sec/bar',(time.time()-t0)/len(old))
