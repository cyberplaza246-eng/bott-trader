"""READ-ONLY live chart for EVAL PASSER / FUNDED 12K. Never imports the bot, never talks to the broker.
Reads: 1m bar CSV (NEAT export, same file the bot reads), the bot's opt-in JSONL event log (FLEX3_EVENT_LOG),
and the flex3 state JSON (day P&L, eval P&L). Serves http://127.0.0.1:8765 .
  live:   python server.py --bars <MNQ_1m_live.csv> --events logs\\flex3_events.jsonl --state logs\\flex3_state.json --mode eval
  replay: python server.py --replay 2026-09-15 --speed 60
Needs only Python 3 + pandas (already in the bot venv)."""
import argparse, json, os, time, threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import pandas as pd
ET = "America/New_York"; HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--bars"); ap.add_argument("--events"); ap.add_argument("--state")
ap.add_argument("--mode", default="eval", choices=["eval", "funded"])
ap.add_argument("--bar-label", default="open", choices=["open", "close"])
ap.add_argument("--replay"); ap.add_argument("--speed", type=float, default=60.0)
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--daily-stop", type=float, default=float(os.getenv("FLEX3_DAILY_LOSS_STOP", 1500)))
ap.add_argument("--start-balance", type=float, default=float(os.getenv("FUNDED_START_BALANCE", 52200)))
ap.add_argument("--floor", type=float, default=50100.0)
A = ap.parse_args()
if A.replay:
    A.bars = os.path.join(HERE, "replay", f"MNQ_1m_{A.replay}.csv"); A.events = os.path.join(HERE, "replay", f"events_{A.replay}.jsonl")
    A.mode = "eval"; A.state = None
    RP = dict(start=pd.Timestamp(f"{A.replay}", tz=ET) - pd.Timedelta(hours=6, minutes=15), speed=A.speed, wall=time.time(), paused=False, at=None)
    RP["end"] = pd.Timestamp(A.replay, tz=ET) + pd.Timedelta(hours=16, minutes=50)
LOCK = threading.Lock(); CACHE = {}

def now_utc():
    if not A.replay: return pd.Timestamp.now(tz="UTC")
    with LOCK:
        if RP["paused"]: t = RP["at"]
        else: t = RP["start"] + pd.Timedelta(seconds=(time.time() - RP["wall"]) * RP["speed"])
        return min(t, RP["end"]).tz_convert("UTC")

def load_bars():
    mt = os.path.getmtime(A.bars)
    if CACHE.get("mt") != mt:
        h = pd.read_csv(A.bars); h["datetime"] = pd.to_datetime(h["datetime"], utc=True, errors="coerce")
        h = h.dropna(subset=["datetime"]).drop_duplicates("datetime", keep="last").set_index("datetime").sort_index()
        if A.bar_label == "close": h.index = h.index - pd.Timedelta(minutes=1)
        CACHE.update(mt=mt, bars=h[["open", "high", "low", "close"]].astype(float))
    return CACHE["bars"]

def load_events(upto):
    ev = []
    if A.events and os.path.exists(A.events):
        for ln in open(A.events, encoding="utf-8"):
            try: e = json.loads(ln)
            except Exception: continue
            e["_t"] = pd.Timestamp(e["t"]).tz_convert("UTC")
            if e["_t"] <= upto: ev.append(e)
    return ev

def tday(ts):  # trading day rolls 18:00 ET
    e = ts.tz_convert(ET); d = (e + pd.Timedelta(hours=6)).normalize().tz_localize(None)
    return d + pd.Timedelta(days={5: 2, 6: 1}.get(d.weekday(), 0))

def ets(ts):  # chart time: ET wall clock as fake-UTC seconds
    e = ts.tz_convert(ET); return int(e.tz_localize(None).value // 10**9)

def levels(b, now):
    e = b.index.tz_convert(ET); m = e.hour * 60 + e.minute
    d = (e + pd.Timedelta(hours=6)).normalize().tz_localize(None)
    wd = d.weekday; d = d + pd.to_timedelta([{5: 2, 6: 1}.get(w, 0) for w in wd], "D")
    cur = tday(now); out = {}
    rth = (m >= 570) & (m < 960); x = b[rth]; xd = d[rth]
    prev = sorted(set(xd[xd < cur]))
    if prev:
        p = x[xd == prev[-1]]; out["PDH"] = p.high.max(); out["PDL"] = p.low.min()
    nm = now.tz_convert(ET); nmin = nm.hour * 60 + nm.minute; is_cur = d == cur
    if 570 <= nmin < 1080:
        o = b[is_cur & ((m >= 1080) | (m < 570))]
        if len(o): out["ONH"] = o.high.max(); out["ONL"] = o.low.min()
    if 585 <= nmin < 1020:
        o = b[is_cur & (m >= 570) & (m < 585)]
        if len(o): out["ORH"] = o.high.max(); out["ORL"] = o.low.min()
    wk = pd.PeriodIndex(d, freq="W-FRI"); cw = pd.Period(cur, freq="W-FRI"); pw = b[wk == cw - 1]
    if len(pw): out["PWH"] = pw.high.max(); out["PWL"] = pw.low.min()
    return {k: round(float(v), 2) for k, v in out.items()}

def payload(tf):
    now = now_utc(); b = load_bars(); b = b[b.index <= now - pd.Timedelta(minutes=1)]
    lv = levels(b.iloc[-30000:], now) if len(b) else {}
    ref = b.index[-1] if len(b) else now
    view = b[b.index >= ref - pd.Timedelta(hours=24)]
    if tf == 5: view = view.resample("5min", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    candles = [dict(time=ets(t), open=r.open, high=r.high, low=r.low, close=r.close) for t, r in view.iterrows()]
    last = float(b.close.iloc[-1]) if len(b) else None
    ev = load_events(now); trades = {}; order = []
    for e in ev:
        o = e.get("order_id")
        if e["event"] == "entry":
            trades[o] = dict(id=o, dir=e.get("direction"), size=e.get("size", 1), entry=e.get("entry"), setup=e.get("source") or e.get("stream"),
                             stream=e.get("stream"), t_in=ets(e["_t"]), td=str(tday(e["_t"]).date()), sl_path=[[ets(e["_t"]), e.get("sl")]], tp=e.get("tp"), open=True); order.append(o)
        elif e["event"] == "sl_move" and o in trades:
            trades[o]["sl_path"].append([ets(e["_t"]), e.get("sl")])
            if e.get("tp") is not None: trades[o]["tp"] = e.get("tp")
        elif e["event"] == "exit" and o in trades:
            trades[o].update(open=False, t_out=ets(e["_t"]), exit=e.get("exit"), reason=e.get("reason"), pnl=e.get("pnl"))
    T = [trades[o] for o in order if trades[o]["t_in"] >= ets(ref - pd.Timedelta(hours=24))]
    ctd = str(tday(now).date()); today = [t for t in trades.values() if t["td"] == ctd]
    day_pnl = sum(t.get("pnl") or 0 for t in today if not t["open"])
    allpnl = sum(t.get("pnl") or 0 for t in trades.values() if not t["open"])
    st = {}
    if A.state and os.path.exists(A.state):
        try: st = json.load(open(A.state))
        except Exception: st = {}
    if st.get("day_key") == ctd and "day_pnl" in st: day_pnl = float(st["day_pnl"])
    opn = []
    for t in trades.values():
        if t["open"] and last is not None:
            sg = 1 if t["dir"] == "long" else -1; u = sg * (last - t["entry"]) * 2 * t["size"]
            opn.append(dict(setup=t["setup"], dir=t["dir"], size=t["size"], entry=t["entry"], sl=t["sl_path"][-1][1], tp=t["tp"], upnl=round(u, 2)))
    upnl = sum(o["upnl"] for o in opn)
    panel = dict(mode=A.mode, now_et=now.tz_convert(ET).strftime("%a %b %d %H:%M:%S ET"), trading_day=ctd, last=last, open=opn,
                 day_pnl=round(day_pnl, 2), open_pnl=round(upnl, 2), trades_today=len(today),
                 wins_today=sum(1 for t in today if (t.get("pnl") or 0) > 0 and not t["open"]),
                 to_daily_stop=round(A.daily_stop + day_pnl + upnl, 2), daily_stop=A.daily_stop)
    if A.mode == "eval":
        ep = float(st.get("eval_pnl", allpnl)); hw = float(st.get("eval_hwm", max(ep, 0)))
        floor = min(hw - 2000, 100); panel.update(eval_pnl=round(ep, 2), dd_floor=round(floor, 2), to_drawdown=round(ep + upnl - floor, 2), target=3000, to_target=round(3000 - ep, 2))
    else:
        bal = A.start_balance + float(st.get("eq", allpnl)); panel.update(balance=round(bal, 2), dd_floor=A.floor, to_drawdown=round(bal + upnl - A.floor, 2))
    if A.replay: panel.update(replay=A.replay, speed=RP["speed"], paused=RP["paused"])
    return dict(candles=candles, levels=lv, trades=T, panel=panel, now=ets(now))

class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=os.path.join(HERE, "static"), **k)
    def log_message(self, *a): pass
    def _json(self, obj):
        b = json.dumps(obj).encode(); self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == "/api/data": return self._json(payload(int(q.get("tf", ["5"])[0])))
        if u.path == "/api/replay" and A.replay:
            with LOCK:
                cur = RP["start"] + pd.Timedelta(seconds=(time.time() - RP["wall"]) * RP["speed"]) if not RP["paused"] else RP["at"]
                if "restart" in q: cur = RP["start"]
                if "jump" in q: cur = pd.Timestamp(q["jump"][0], tz=ET)
                if "speed" in q: RP["speed"] = max(0.1, float(q["speed"][0]))
                if "pause" in q: RP["paused"] = q["pause"][0] == "1"
                RP["at"] = cur; RP["start"] = cur; RP["wall"] = time.time()
            return self._json({"ok": True})
        if u.path == "/": self.path = "/index.html"
        return super().do_GET()

if __name__ == "__main__":
    print(f"READ-ONLY chart on http://127.0.0.1:{A.port}  mode={A.mode} {'REPLAY '+A.replay if A.replay else 'LIVE'}")
    ThreadingHTTPServer(("127.0.0.1", A.port), H).serve_forever()
