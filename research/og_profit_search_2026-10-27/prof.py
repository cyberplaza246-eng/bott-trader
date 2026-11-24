import os,sys,time,cProfile,pstats,logging
sys.dont_write_bytecode=True
H=os.path.dirname(os.path.abspath(__file__)); SB=os.path.join(H,'sandbox'); os.chdir(SB); sys.path.insert(0,SB)
os.environ.update(dict(PYTHONUTF8='1',FALLBACK_ENTRIES_ENABLED='true',ASSET_CLASS='futures',TRADING_SYMBOLS='MNQ',TRADING_MODE='backtest',BROKER_TYPE='rithmic',NEWSAPI_KEY=''))
import pandas as pd
from src.core.ensemble_trader import EnsembleTrader
from src.ai.technical_analyzer import TechnicalAnalyzer
from src.utils.logger import bot_logger
for h in list(bot_logger.handlers): bot_logger.removeHandler(h)
logging.disable(logging.CRITICAL)
et=EnsembleTrader(broker=None); et.sentiment.get_pair_sentiment=lambda pair:{'sentiment_score':0.0,'news_count':0}
ta=TechnicalAnalyzer()
b=pd.read_csv(os.path.join(H,'data','bars5m_clean.csv'),parse_dates=['datetime'])
print('lstm',et.lstm_available,'intel',et.intelligent_available,'adv',et.advanced_strats_available,flush=True)
def one(i):
    df=b.iloc[i-99:i+1].copy().reset_index(drop=True); dfe=ta.calculate_indicators(df); r=et.get_trading_signal(dfe,'MNQ'); return r
one(50100)
pr=cProfile.Profile(); pr.enable()
for i in range(50101,50131): one(i)
pr.disable(); pstats.Stats(pr).sort_stats('cumulative').print_stats(25)
