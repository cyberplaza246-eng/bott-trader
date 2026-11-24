"""REVAMP PAPER runner - breakout sleeve OFF (trend ADX>=25 + mean reversion ADX<20, MR unfiltered).
PAPER ONLY: no broker import, no Rithmic login, no order code at all. Reads the 5m bars the NEAT kid
keeps in data/neat_kid_bars_MNQ_5m.csv (or data/live_bars_MNQ_5m.csv if newer). Logic mirrors
research/tv_filters/tv_filter_backtest.py variant X_no_breakout (revamp exits: trend 1.5R, MR BB far band,
4.0 ATR trail, 1 contract, $4 round trip). Usage: python tools/revamp_paper_nobo.py [--replay-days N]
"""
import sys, os, time, csv, argparse
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[1]; os.chdir(REPO); sys.path.insert(0, str(REPO))
from src.research_ensemble.indicators import adx, atr, bollinger_bands
from src.research_ensemble.trend_following import TrendFollowingStrategy
from src.research_ensemble.mean_reversion import MeanReversionStrategy

PT, TICK = 2.0, 0.25
COST_RT = 2 * 1.5 + 2 * TICK * PT
TREND_R, TRAIL = 1.5, 4.0
SOURCES = [REPO / "data/neat_kid_bars_MNQ_5m.csv", REPO / "data/live_bars_MNQ_5m.csv"]
TRADES_CSV = REPO / "data/revamp_paper_nobo_trades.csv"
LOG = REPO / "logs/revamp_paper_nobo.log"; LOG.parent.mkdir(exist_ok=True)
_lf = open(LOG, "a", encoding="utf-8")
COLS = ["entry_time_et", "sleeve", "side", "entry", "sl", "tp", "rr", "exit_time_et", "exit", "reason", "pnl", "day_pnl"]

def say(m=""):
    s = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}  {m}"; print(s, flush=True); _lf.write(s + "\n"); _lf.flush()

def et(ts): return pd.Timestamp(ts).tz_convert("America/New_York").strftime("%a %b %d %I:%M %p ET")

def load():
    frames = []
    for p in SOURCES:
        if p.is_file():
            for _ in range(5):  # kid rewrites the file; retry on partial read
                try:
                    d = pd.read_csv(p); d["datetime"] = pd.to_datetime(d["datetime"], utc=True); frames.append(d); break
                except Exception: time.sleep(1)
    df = pd.concat(frames).drop_duplicates("datetime", keep="last").sort_values("datetime").set_index("datetime")
    return df[["open", "high", "low", "close", "volume"]].astype(float)

def signals(df):
    c = df["close"]; a = adx(df["high"], df["low"], c, 14)
    tr, mr = TrendFollowingStrategy().generate_signals(df), MeanReversionStrategy().generate_signals(df)
    up, _, lo = bollinger_bands(c, 20, 2.0)
    n = len(df); e = np.zeros(n, int); sl = np.full(n, np.nan); tp = np.full(n, np.nan); sv = np.array([""] * n, dtype=object)
    for sig, mask, name in ((tr, (a >= 25).values, "trend"), (mr, (a < 20).values, "mr")):
        m = mask & (sig.entries.values != 0)
        e[m] = sig.entries.values[m]; sl[m] = sig.stop_price.values[m]; sv[m] = name
    # breakout sleeve intentionally OFF
    cv = c.values; risk = np.abs(cv - sl)
    m = sv == "trend"; tp[m] = cv[m] + e[m] * TREND_R * risk[m]
    m = sv == "mr"; tp[m] = np.where(e[m] > 0, up.values[m], lo.values[m])
    bad = ((e == 1) & (tp <= cv)) | ((e == -1) & (tp >= cv)); tp[bad] = np.nan
    return e, sl, tp, sv

def simulate(df, start_ts):
    """Same engine as the backtest. Returns closed trades and the open position (if any)."""
    e, sl, tp, sv = signals(df); atr_ = atr(df["high"], df["low"], df["close"], 14).values
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close")); idx = df.index
    i = int(np.searchsorted(idx, start_ts)); out = []; pos = 0; n = len(df)
    while i < n:
        if pos == 0:
            if e[i] != 0 and np.isfinite(sl[i]):
                pos, ent, stop, tgt, ei, slv, best = e[i], c[i], sl[i], tp[i], i, sv[i], c[i]; sl0 = stop
            i += 1; continue
        ex = None
        if pos > 0:
            if l[i] <= stop: ex, why = min(stop, o[i]), "stop"
            elif np.isfinite(tgt) and h[i] >= tgt: ex, why = max(tgt, o[i]), "target"
        else:
            if h[i] >= stop: ex, why = max(stop, o[i]), "stop"
            elif np.isfinite(tgt) and l[i] <= tgt: ex, why = min(tgt, o[i]), "target"
        if ex is None:
            best = max(best, h[i]) if pos > 0 else min(best, l[i])
            ts = best - pos * TRAIL * atr_[i]
            if np.isfinite(ts): stop = max(stop, ts) if pos > 0 else min(stop, ts)
            i += 1; continue
        if why == "stop" and stop != sl0: why = "trail stop"
        out.append(dict(t0=idx[ei], sleeve=slv, side="LONG" if pos > 0 else "SHORT", entry=ent, sl=sl0, tp=tgt,
                        t1=idx[i], exit=ex, reason=why, pnl=(ex - ent) * pos * PT - COST_RT))
        pos = 0; i += 1
    op = dict(t0=idx[ei], sleeve=slv, side="LONG" if pos > 0 else "SHORT", entry=ent, sl=sl0, tp=tgt, stop=stop) if pos else None
    return out, op

def rr(t):
    r = abs(t["entry"] - t["sl"]); return round(abs(t["tp"] - t["entry"]) / r, 2) if r and np.isfinite(t["tp"]) else ""

def show_entry(t):
    tp = f"{t['tp']:.2f}" if np.isfinite(t["tp"]) else "none (trail)"
    say(f">>> PAPER ENTRY {t['side']} [{t['sleeve']}] {et(t['t0'])} @ {t['entry']:.2f}  SL {t['sl']:.2f}  TP {tp}  R:R {rr(t)}")

def show_exit(t, day):
    say(f"<<< PAPER EXIT  {t['side']} [{t['sleeve']}] {et(t['t1'])} @ {t['exit']:.2f} ({t['reason']})  P&L {t['pnl']:+.2f}  | day P&L {day:+.2f}")

def day_key(ts): return pd.Timestamp(ts).tz_convert("America/New_York").date()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--replay-days", type=int, default=0)
    ap.add_argument("--poll", type=int, default=30); a = ap.parse_args()
    say("=" * 72); say("REVAMP PAPER (breakout OFF) - PAPER ONLY. NO BROKER, NO RITHMIC LOGIN, NO ORDERS."); say("=" * 72)
    df = load(); say(f"Bars: {len(df)}  {et(df.index[0])} -> {et(df.index[-1])}  (source: {SOURCES[0].name})")
    if a.replay_days:
        start = df.index[-1] - pd.Timedelta(days=a.replay_days)
        tr, op = simulate(df, start); days = {}
        for t in tr:
            k = day_key(t["t1"]); days[k] = days.get(k, 0) + t["pnl"]; show_entry(t); show_exit(t, days[k])
        say(f"REPLAY last {a.replay_days} days: {len(tr)} trades, net {sum(t['pnl'] for t in tr):+.2f}" + (f", still open {op['side']} {op['sleeve']}" if op else ""))
        return 0
    live_from = df.index[-1] + pd.Timedelta(minutes=1)
    say(f"LIVE: paper-trading bars after {et(df.index[-1])}. Checking every {a.poll}s...")
    seen, shown_open, last = set(), None, df.index[-1]
    while True:
        time.sleep(a.poll)
        try: df = load()
        except Exception as ex: say(f"read error {ex}"); continue
        if df.index[-1] == last: continue
        last = df.index[-1]; tr, op = simulate(df, live_from)
        for t in tr:
            if t["t0"] in seen: continue
            seen.add(t["t0"])
            if shown_open != t["t0"]: show_entry(t)
            k = day_key(t["t1"]); day = sum(x["pnl"] for x in tr if day_key(x["t1"]) == k); show_exit(t, day)
            hdr = not TRADES_CSV.exists()
            with open(TRADES_CSV, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLS); hdr and w.writeheader()
                w.writerow(dict(entry_time_et=et(t["t0"]), sleeve=t["sleeve"], side=t["side"], entry=t["entry"], sl=round(t["sl"], 2),
                                tp=round(t["tp"], 2) if np.isfinite(t["tp"]) else "", rr=rr(t), exit_time_et=et(t["t1"]),
                                exit=t["exit"], reason=t["reason"], pnl=round(t["pnl"], 2), day_pnl=round(day, 2)))
        if op and shown_open != op["t0"]: show_entry(op); shown_open = op["t0"]
        today = sum(x["pnl"] for x in tr if day_key(x["t1"]) == day_key(last))
        st = f"in {op['side']} [{op['sleeve']}] @ {op['entry']:.2f}, stop {op['stop']:.2f}" if op else "flat, waiting for signal"
        say(f"    bar {et(last)} close {df['close'].iloc[-1]:.2f} | {st} | day P&L {today:+.2f}")

if __name__ == "__main__":
    sys.exit(main())
