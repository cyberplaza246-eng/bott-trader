import os, sys, time, json, logging
def main(start_i, end_i, outpath, progress):
    outpath=os.path.abspath(outpath); progress=os.path.abspath(progress)
    SB=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'sandbox')); os.chdir(SB); sys.path.insert(0,SB)
    os.environ.update(dict(PYTHONUTF8='1',FALLBACK_ENTRIES_ENABLED='false',ASSET_CLASS='futures',TRADING_SYMBOLS='MNQ',TRADING_MODE='backtest',BROKER_TYPE='rithmic',NEWSAPI_KEY='',RSI_BUY_BLOCK='70',RSI_SELL_BLOCK='30'))
    import datetime as _dt, types
    import pandas as pd, numpy as np
    import datetime as _real
    class FDT(_real.datetime):
        _now=None
        @classmethod
        def utcnow(cls):
            return cls._now.replace(tzinfo=None) if cls._now is not None else _real.datetime.utcnow()
        @classmethod
        def now(cls, tz=None):
            if tz is None or cls._now is None:
                return _real.datetime.now(tz)
            return cls._now.astimezone(tz)
    _shim=types.ModuleType('datetime'); _shim.__dict__.update(_real.__dict__); _shim.datetime=FDT; sys.modules['datetime']=_shim
    from src.core.ensemble_trader import EnsembleTrader
    from src.ai.technical_analyzer import TechnicalAnalyzer
    from src.utils.logger import bot_logger
    for h in list(bot_logger.handlers): bot_logger.removeHandler(h)
    bot_logger.addHandler(logging.NullHandler()); bot_logger.setLevel(logging.CRITICAL); bot_logger.propagate=False
    logging.disable(logging.CRITICAL)
    et=EnsembleTrader(broker=None)
    et.sentiment.get_pair_sentiment=lambda pair:{'sentiment_score':0.0,'news_count':0}
    ta=TechnicalAnalyzer()
    b=pd.read_csv(os.path.join(SB,'..','bars5m.csv'),parse_dates=['datetime'])
    b['datetime']=pd.to_datetime(b['datetime'],utc=True)
    b=b.reset_index(drop=True)
    done=set()
    if os.path.exists(outpath):
        for line in open(outpath):
            try: done.add(json.loads(line)['i'])
            except Exception: pass
    out=open(outpath,'a',buffering=1)
    t0=time.time(); n=0; nsig=0
    for i in range(max(start_i,99), min(end_i,len(b))):
        if i in done: continue
        # skip bars after a >60min data gap until 100 contiguous-ish bars? live replays whatever bars exist; keep.
        df=b.iloc[i-99:i+1].copy().reset_index(drop=True)
        FDT._now=b.datetime.iloc[i].to_pydatetime()
        rec={'i':i,'t':str(b.datetime.iloc[i]),'close':float(b.close.iloc[i])}
        try:
            dfe=ta.calculate_indicators(df)
            sw=et.sweep.get_signal(dfe,'MNQ',df_5m=dfe.tail(250).copy(),spread=None)
            sd=sw.get('signal','SKIP')
            rec['sweep']=sd
            if sd in ('BUY','SELL'):
                r=et.get_trading_signal(dfe,'MNQ')
                m=r['models']
                rec.update(dict(sig=r['signal'],conf=r['confidence'],sweep_conf=sw.get('confidence',0.0),rsi=r['rsi'],regime=r['regime'],
                    sweep_regime=m['sweep'].get('regime'),agree=r['models_agreement'],ema=m['ema_crossover']['signal'],tech=m['technical']['signal'],
                    intel=m['intelligent']['signal'],adv=m['advanced_strategies']['signal'],atr=float(dfe['atr'].iloc[-1]),
                    ema200=float(dfe['ema_200'].iloc[-1]) if 'ema_200' in dfe.columns else None,
                    zone=(r.get('sr_levels') or {}).get('price_zone')))
                rec['should_trade']=bool(et.should_trade(r)); rec['thr']=float(et.learner.get_adjusted_threshold()); rec['sweep_bias']=sw.get('bias'); rec['mss']=bool((sw.get('mss') or {}).get('confirmed',False))
                try:
                    s=et.get_dynamic_sl_tp(dfe,r['signal'],float(dfe.close.iloc[-1]),symbol='MNQ',sr_levels=r.get('sr_levels'),sweep_wick=r.get('sweep_wick'),timeframe='5m')
                    rec['sl']=float(s['sl_price']); rec['tp']=float(s['tp_price']); rec['sltp_method']=s.get('method')
                except Exception as e:
                    rec['sltp_err']=str(e)[:100]
        except Exception as e:
            rec['err']=str(e)[:200]
        out.write(json.dumps(rec)+'\n'); n+=1
        if rec.get('sweep') in ('BUY','SELL'): nsig+=1
        if n%200==0:
            open(progress,'w').write(json.dumps({'start':start_i,'end':end_i,'done':n,'cur_i':i,'sweeps':nsig,'sec':time.time()-t0}))
    open(progress,'w').write(json.dumps({'start':start_i,'end':end_i,'done':n,'FINISHED':True,'sweeps':nsig,'sec':time.time()-t0}))
if __name__=='__main__':
    main(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],sys.argv[4])


