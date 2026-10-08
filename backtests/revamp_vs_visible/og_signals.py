"""Replay the OG 'VISIBLE' EnsembleTrader (Sweep-Gate + EMA/Tech fallback + ML confirm) on historical 5m bars.
SIMULATION ONLY. Mirrors start_live_rithmic.py non-revamp check_entry_signal:
  100 completed 5m bars -> calculate_indicators -> TechnicalAnalyzer.calculate_indicators ->
  EnsembleTrader.get_trading_signal -> should_trade -> get_dynamic_sl_tp(structure-aware).
Clock frozen at bar close (time_machine). Fake broker serves historical 5m/60m/240m bars for multi-TF context.
News sentiment stubbed neutral (not reproducible historically). LSTM off (no TF in live venv either).
"""
import os, sys, time, json, logging
REPO = "/workspace/og_bt/repo"
ENV = dict(ASSET_CLASS="futures", TRADING_SYMBOLS="MNQ", TRADING_MODE="live", INITIAL_BALANCE="50000",
           ENSEMBLE_CONFIDENCE_THRESHOLD="0.68", MIN_MODELS_AGREEMENT="4", FALLBACK_ENTRIES_ENABLED="true",
           ADAPTIVE_SKIP_ENABLED="false", OVERNIGHT_TRADING="true", SESSION_MODE="extended", SCALP_SESSIONS="off",
           DAILY_LOSS_LIMIT_PERCENT="1.0", TRAIL_ATR_MULT="0.35", TRAIL_ACTIVATION_R="0.25",
           LOCK_MODE="usd", LOCK_ACTIVATE_USD="100", LOCK_KEEP_USD="20", LOCK_TRAIL_START_USD="150",
           LOCK_TRAIL_KEEP_PCT="0.5", NEWSAPI_KEY="", BROKER_TYPE="rithmic")
os.environ.update(ENV)
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np, pandas as pd
import time_machine

CLEAN = "/workspace/neat_plan_out/clean.parquet"


def load_frames():
    d = pd.read_parquet(CLEAN)
    d["datetime"] = pd.to_datetime(d["datetime"], utc=True)
    d = d.set_index("datetime").sort_index()
    d = d[~d.index.duplicated(keep="last")]
    fr = {}
    for tf in (5, 60, 240):
        a = d[["open", "high", "low", "close", "volume"]].resample(f"{tf}min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])
        fr[tf] = a
    return d, fr


class FakeBroker:
    """Historical bars as of the frozen clock (completed bars only)."""
    def __init__(self, frames):
        self.fr = frames
        self.ends = {tf: (f.index + pd.Timedelta(minutes=tf)).asi8 for tf, f in frames.items()}
        self.now_ns = 0

    def get_candles(self, symbol, timeframe_minutes=5, num_candles=100, end_time=None):
        tf = int(timeframe_minutes)
        if tf not in self.fr:
            return None
        k = int(np.searchsorted(self.ends[tf], self.now_ns, side="right"))
        if k <= 0:
            return None
        s = self.fr[tf].iloc[max(0, k - num_candles):k]
        out = s.reset_index().rename(columns={"index": "datetime"})
        return out

    def get_spread(self, symbol):
        return None


def live_calc_indicators(df, lookback=20, ema_len=50):
    """start_live_rithmic.LiveRithmicTrader.calculate_indicators (non-revamp)."""
    df = df.copy()
    df['high_N'] = df['high'].rolling(lookback).max().shift(1)
    df['low_N'] = df['low'].rolling(lookback).min().shift(1)
    df['ema'] = df['close'].ewm(span=ema_len, adjust=False).mean()
    hl = df['high'] - df['low']; hc = (df['high'] - df['close'].shift()).abs(); lc = (df['low'] - df['close'].shift()).abs()
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    return df


def build(start, end, warm_start=None, out_path=None, progress=True):
    logging.disable(logging.CRITICAL)
    d1, fr = load_frames()
    from config.strategy_config import SCALPING_PAIRS  # bad hours (start_live_rithmic uses this first)
    from src.core.ensemble_trader import EnsembleTrader
    from src.ai.technical_analyzer import TechnicalAnalyzer
    from src.utils import logger as _lg
    logging.disable(logging.CRITICAL)
    try:
        _lg.TradeLogger.log_signal = staticmethod(lambda *a, **k: None)
    except Exception:
        pass
    broker = FakeBroker(fr)
    ens = EnsembleTrader(broker=broker)
    ens.sentiment.get_pair_sentiment = lambda pair: {"sentiment_score": 0.0, "confidence": 0.0, "news_count": 0}
    for nm in ("save_state", "_save_state", "save", "_save"):
        if hasattr(ens.learner, nm):
            setattr(ens.learner, nm, lambda *a, **k: None)
    tech = TechnicalAnalyzer()
    lookback, ema_len = 10, 50  # start_live_rithmic.PARAMS
    bad_hours = SCALPING_PAIRS.get('MNQ', {}).get('bad_hours_utc', [0, 4, 21])
    meta = dict(threshold=ens.learner.get_adjusted_threshold(), bad_hours=list(bad_hours),
                intelligent=ens.intelligent_available, advanced=ens.advanced_strats_available,
                lstm=ens.lstm_available, rl=ens.rl_available, sltp=ens.sltp_available,
                lookback=lookback, ema_len=ema_len)
    f5 = fr[5]
    idx = f5.index
    lo = pd.Timestamp(warm_start or start, tz="UTC"); hi = pd.Timestamp(end, tz="UTC"); st0 = pd.Timestamp(start, tz="UTC")
    sel = np.where((idx >= lo) & (idx < hi))[0]
    rows = []
    t0 = time.time()
    for n, i in enumerate(sel):
        bar_start = idx[i]
        close_t = bar_start + pd.Timedelta(minutes=5)
        if close_t.hour in bad_hours:  # live: now.hour (UTC) at bar processing
            continue
        if i < 120:
            continue
        broker.now_ns = close_t.value
        with time_machine.travel(close_t.to_pydatetime() + pd.Timedelta(seconds=2).to_pytimedelta(), tick=False):
            df = broker.get_candles("MNQ", 5, 100)
            if df is None or len(df) < lookback + ema_len + 15:
                continue
            df = live_calc_indicators(df, lookback, ema_len)
            try:
                dfe = tech.calculate_indicators(df)
                sr = ens.get_trading_signal(dfe, "MNQ")
                ok = ens.should_trade(sr)
            except Exception as e:
                rows.append(dict(bar=bar_start, err=str(e)[:120]))
                continue
            if bar_start < st0:
                continue  # warm-up
            sig = sr.get('signal')
            if not ok or sig not in ('BUY', 'SELL'):
                continue
            price = float(df['close'].iloc[-1])
            try:
                st = ens.get_dynamic_sl_tp(dfe, sig, price, symbol="MNQ", sr_levels=sr.get('sr_levels'),
                                           sweep_wick=sr.get('sweep_wick'), timeframe='5m')
                sl, tp, meth = float(st['sl_price']), float(st['tp_price']), st.get('method', '?')
            except Exception as e:
                rows.append(dict(bar=bar_start, err="sltp:" + str(e)[:100]))
                continue
            sw = sr.get('models', {}).get('sweep', {})
            rows.append(dict(bar=bar_start, dir=1 if sig == 'BUY' else -1, ref=price, sl=sl, tp=tp,
                             conf=float(sr.get('confidence', 0)), sweep=sw.get('signal'), method=meth,
                             regime=sr.get('regime'), agree=sr.get('models_agreement'),
                             intel=sr.get('models', {}).get('intelligent', {}).get('signal')))
        if progress and n % 2000 == 0:
            print(f"{os.getpid()} {bar_start} {n}/{len(sel)} {time.time()-t0:.0f}s sigs={len(rows)}", flush=True)
    out = pd.DataFrame(rows)
    if out_path:
        out.to_pickle(out_path)
        json.dump(meta, open(out_path + ".meta.json", "w"), default=str)
    return out, meta


if __name__ == "__main__":
    s, e, w, o = sys.argv[1:5]
    build(s, e, w if w != "-" else None, o)
