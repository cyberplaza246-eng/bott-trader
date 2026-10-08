import og_signals as O, time, logging, cProfile, pstats, collections
import numpy as np, pandas as pd, time_machine
logging.disable(logging.CRITICAL)
d1, fr = O.load_frames()
from src.core.ensemble_trader import EnsembleTrader
from src.ai.technical_analyzer import TechnicalAnalyzer
logging.disable(logging.CRITICAL)
b = O.FakeBroker(fr); ens = EnsembleTrader(broker=b)
ens.sentiment.get_pair_sentiment = lambda pair: {"sentiment_score": 0.0, "confidence": 0.0, "news_count": 0}
tech = TechnicalAnalyzer()
idx = fr[5].index
sel = np.where((idx >= pd.Timestamp('2025-03-04 14:30', tz='UTC')) & (idx < pd.Timestamp('2025-03-05 21:00', tz='UTC')))[0]
c = collections.Counter(); rows=[]
pr = cProfile.Profile()
for i in sel[:40]:
    t = idx[i] + pd.Timedelta(minutes=5); b.now_ns = t.value
    with time_machine.travel(t.to_pydatetime(), tick=False):
        df = O.live_calc_indicators(b.get_candles('MNQ', 5, 100), 10, 50)
        pr.enable()
        dfe = tech.calculate_indicators(df); sr = ens.get_trading_signal(dfe, 'MNQ'); ok = ens.should_trade(sr)
        pr.disable()
    m = sr['models']
    rows.append((str(idx[i]), m['sweep']['signal'], m['sweep'].get('bias'), m['sweep'].get('regime'), m['ema_crossover']['signal'], m['technical']['signal'], round(m['technical'].get('confidence',0),2), sr['signal'], round(sr['confidence'],2), sr['models_agreement'], ok, sr['regime']))
df = pd.DataFrame(rows, columns=['bar','sweep','bias','sregime','ema','tech','techc','final','conf','agree','ok','regime'])
print(df['sweep'].value_counts().to_dict(), df['final'].value_counts().to_dict(), df['ok'].sum())
print(df[df.final!='SKIP'].to_string())
print(df.head(15).to_string())
pstats.Stats(pr).sort_stats('cumulative').print_stats(25)
