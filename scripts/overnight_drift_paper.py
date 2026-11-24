"""PAPER-ONLY overnight drift runner: long 1 MNQ (or NQ) at the 18:00 ET Globex reopen (Sun-Thu), exit 09:25 ET next day (ODRIFT_EXIT=0930 optional); never held past 16:00 ET.
No broker, no Rithmic, no orders. Live prices: read-only tail of data/live_bars_MNQ_5m.csv (written by the
already-running bot); fallback yfinance MNQ=F 5m if that file is stale. Simulated fills -> logs/overnight_drift_paper.csv.
Rule source: Cliff, Cooper & Gulen (2008); backtest in scripts/test_published_nq_strategies.py."""
import csv, json, os, time, datetime as dt
from zoneinfo import ZoneInfo
ET=ZoneInfo("America/New_York"); ORDERS_ENABLED=False  # hard-coded; this file never sends orders
LIVE=os.environ.get("ODRIFT_LIVE","data/live_bars_MNQ_5m.csv"); LOG="logs/overnight_drift_paper.csv"
STATE="logs/overnight_drift_paper_state.json"; HB="logs/overnight_drift_paper_heartbeat.json"
SYM=os.environ.get("ODRIFT_SYMBOL","MNQ"); PT={"NQ":20.0,"MNQ":2.0}[SYM]; COMM={"NQ":4.5,"MNQ":1.24}[SYM]; TICK=0.25
STALE_MIN=12
def pnl(entry,exit_,pt=None,comm=None): return round((exit_-entry-2*TICK)*(pt or PT)-(comm or COMM),2)
def read_live(path=LIVE,n=300):
    """read-only tail of the 5m bar file -> list of (ts_et, open, close); ts is bar START"""
    with open(path,"rb") as f:
        f.seek(0,2); f.seek(max(0,f.tell()-80*n)); lines=f.read().decode(errors="ignore").splitlines()[1:]
    out=[]
    for ln in lines:
        p=ln.split(",")
        try: out.append((dt.datetime.fromisoformat(p[0]).astimezone(ET),float(p[1]),float(p[4])))
        except Exception: pass
    return out
def read_yf():
    import yfinance as yf
    h=yf.Ticker("MNQ=F").history(period="2d",interval="5m")
    return [(t.tz_convert(ET).to_pydatetime(),float(r.Open),float(r.Close)) for t,r in h.iterrows()]
def market_open(now):
    """CME equity futures: Sun 18:00 - Fri 17:00 ET, daily break 17:00-18:00"""
    wd,hm=now.weekday(),now.hour*60+now.minute
    if wd==5 or (wd==4 and hm>=1020) or (wd==6 and hm<1080): return False
    return not (1020<=hm<1080)
EXIT_HM=int(os.environ.get("ODRIFT_EXIT","0925")[:2])*60+int(os.environ.get("ODRIFT_EXIT","0925")[2:])
def exit_due(entry_iso):
    e=dt.datetime.fromisoformat(entry_iso).astimezone(ET); d=(e+dt.timedelta(days=1)).date()
    return dt.datetime(d.year,d.month,d.day,EXIT_HM//60,EXIT_HM%60,tzinfo=ET)
def step(bars,state):
    """event-driven on bar START timestamps. entry: open of 18:00 bar Sun-Thu. exit: open of first bar at/after
    EXIT_HM (default 09:25) next day; hard backstop: never hold past 16:00 ET (any bar >= 15:55 after entry day)."""
    ev=[]; last=state.get("last_ts")
    for t,o,c in bars:
        if last and t.isoformat()<=last: continue
        hm=t.hour*60+t.minute; p_=state.get("pos")
        if p_ is not None:
            due=exit_due(p_["time"]); hard=dt.datetime(due.year,due.month,due.day,15,55,tzinfo=ET)
            if t>=due or t>=hard:
                p=pnl(p_["entry"],o); state["pos"]=None; ev.append(("PAPER_SELL",t,o,p)); p_=None
        if p_ is None and state.get("pos") is None and hm==1080 and t.weekday() in (6,0,1,2,3) and not any(e[0]=="PAPER_SELL" and e[1]==t for e in ev):
            state["pos"]={"entry":o,"date":t.date().isoformat(),"time":t.isoformat()}; ev.append(("PAPER_BUY",t,o,""))
        state["last_ts"]=t.isoformat()
    return ev
def load():
    try: return json.load(open(STATE))
    except Exception: return {"pos":None,"last_ts":None}
def save(s):
    tmp=STATE+".tmp"; json.dump(s,open(tmp,"w")); os.replace(tmp,STATE)
def log(row):
    new=not os.path.exists(LOG)
    with open(LOG,"a",newline="") as f:
        w=csv.writer(f); new and w.writerow(["time_et","symbol","side","price","pnl_usd","source"]); w.writerow(row)
def main():
    assert not ORDERS_ENABLED
    s=load()
    if s.get("last_ts") is None:  # first start: don't replay history
        b=read_live(); s["last_ts"]=b[-1][0].isoformat() if b else None; save(s)
    print(f"PAPER ONLY overnight drift {SYM}. No orders are ever sent. Log {LOG}. State {STATE}. Position: {s.get('pos')}",flush=True)
    while True:
        now=dt.datetime.now(ET); src="bot_live_bars"; bars=[]
        try: bars=read_live()
        except Exception as e: print(now,"live file read error",e,flush=True)
        age=(now-(bars[-1][0]+dt.timedelta(minutes=5))).total_seconds()/60 if bars else 1e9
        if age>STALE_MIN and market_open(now):
            try: yb=read_yf(); src="yfinance_MNQ=F"; age_y=(now-(yb[-1][0]+dt.timedelta(minutes=5))).total_seconds()/60
            except Exception as e: yb=[]; print(now,"yfinance fallback failed",e,flush=True)
            if yb: bars,age=yb,age_y; print(now,f"bot bars stale; using yfinance, delay {age:.1f} min",flush=True)
        for side,t,px,p in step(bars,s):
            log([now.isoformat(),SYM,side,px,p,src]); print(now,side,px,p,src,flush=True)
        save(s)
        last=bars[-1] if bars else None
        json.dump({"now":now.isoformat(),"source":src,"last_bar":last[0].isoformat() if last else None,"last_price":last[2] if last else None,
                   "age_min":round(age,1),"stale":bool(age>STALE_MIN and market_open(now)),"pos":s.get("pos"),"pid":os.getpid()},open(HB,"w"))
        time.sleep(30)
if __name__=="__main__": main()


