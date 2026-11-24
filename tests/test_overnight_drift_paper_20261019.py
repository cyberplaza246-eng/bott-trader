import ast, datetime as dt, pathlib, sys
from zoneinfo import ZoneInfo
sys.path.insert(0,"scripts"); import overnight_drift_paper as m
ET=ZoneInfo("America/New_York")
def b(y,mo,d,h,mi,o,c): return (dt.datetime(y,mo,d,h,mi,tzinfo=ET),o,c)
def test_orders_disabled(): assert m.ORDERS_ENABLED is False
def test_no_broker_imports():
    tree=ast.parse(pathlib.Path("scripts/overnight_drift_paper.py").read_text(encoding="utf-8-sig"))
    mods={a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}|{n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
    assert not any(x and ("rithmic" in x.lower() or "broker" in x.lower() or x.startswith("src")) for x in mods)
def test_entry_exit_cycle():
    s={"pos":None,"last_ts":None}
    ev=m.step([b(2026,10,20,17,55,1,2),b(2026,10,20,18,0,100.0,101)],s); assert ev[0][0]=="PAPER_BUY" and s["pos"]["entry"]==100.0
    assert m.step([b(2026,10,20,18,0,100.0,101)],s)==[]          # no duplicate on re-read
    assert m.step([b(2026,10,21,9,20,1,1)],s)==[]
    ev=m.step([b(2026,10,21,9,25,110.0,111)],s); assert ev[0][0]=="PAPER_SELL" and ev[0][3]==m.pnl(100,110) and s["pos"] is None
def test_no_friday_or_saturday_entry():
    s={"pos":None,"last_ts":None}; assert m.step([b(2026,10,23,18,0,1,1),b(2026,10,24,18,0,1,1)],s)==[]
def test_sunday_entry_exits_monday():
    s={"pos":None,"last_ts":None}; m.step([b(2026,10,25,18,0,100.0,1)],s)
    assert s["pos"] and m.step([b(2026,10,26,9,25,105.0,1)],s)[0][0]=="PAPER_SELL"
def test_missing_exit_bar_still_exits_and_never_past_1600():
    s={"pos":None,"last_ts":None}; m.step([b(2026,10,20,18,0,100.0,1)],s)
    ev=m.step([b(2026,10,21,10,0,90.0,1)],s); assert ev[0][0]=="PAPER_SELL" and s["pos"] is None
    s={"pos":{"entry":1.0,"date":"2026-10-20","time":"2026-10-20T18:00:00-04:00"},"last_ts":None}
    assert m.exit_due(s["pos"]["time"]).hour*60+m.exit_due(s["pos"]["time"]).minute < 960
def test_no_4pm_entry():
    s={"pos":None,"last_ts":None}; assert m.step([b(2026,10,20,15,55,1,100.0)],s)==[]
def test_restart_safe(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"STATE",str(tmp_path/"st.json")); m.save({"pos":{"entry":1.0,"date":"2026-10-20","time":"2026-10-20T18:00:00-04:00"},"last_ts":"x"})
    assert m.load()["pos"]["entry"]==1.0
def test_market_hours():
    assert m.market_open(dt.datetime(2026,10,20,22,0,tzinfo=ET)); assert not m.market_open(dt.datetime(2026,10,24,12,0,tzinfo=ET))
    assert not m.market_open(dt.datetime(2026,10,20,17,30,tzinfo=ET))
def test_pnl_costs(): assert m.pnl(100,110,20.0,4.5)==round(9.5*20-4.5,2)
def test_reads_live_file():
    bars=m.read_live(); assert bars and bars[-1][2]>0
