r"""OFFLINE test of the gap-fade sleeve: no Rithmic, no network, no real orders.
Run from an EMPTY scratch dir so nothing in the live bot's data/ or logs/ is touched:
    cd C:\Users\shawa\Projects\gf_sandbox
    python C:\Users\shawa\Projects\bott-trader\scripts\gapfade_offline_test.py
Part A: unit scenarios with a fake broker (entry/skip rules, stop, time exit, partial close, failures).
Part B: replay of the last N sessions from data/MNQ_1m.csv through the real bot class (paper mode)."""
import os, sys, json, types, io, contextlib, traceback
REPO = os.environ.get("BOTT_REPO", r"C:\Users\shawa\Projects\bott-trader")
SB = os.getcwd()
assert os.path.abspath(SB) != os.path.abspath(REPO), "run from a scratch dir, not the repo"
os.makedirs("data", exist_ok=True); os.makedirs("logs", exist_ok=True)
os.environ.update(GAPFADE_ENABLED="1", SESSION_PNL_FILE=os.path.join(SB, "data", "session_pnl.json"),
                  GAPFADE_STATE_FILE=os.path.join(SB, "data", "gapfade_state.json"), TRADING_MODE="paper",
                  REVAMP_EXITS="1", MAX_RISK_PER_TRADE_USD="500", MAX_CONCURRENT_TRADES="3",
                  BASE_CONTRACTS="1", MAX_CONTRACTS="1", PYTHONIOENCODING="utf-8")
sys.path.insert(0, REPO)
import numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
ET = "America/New_York"
quiet = io.StringIO()
with contextlib.redirect_stdout(quiet):
    import start_live_rithmic as live
    from src.core import gapfade_sleeve as gf
    gf.install(live.LiveRithmicTrader, live)
assert os.path.abspath(os.getcwd()) == os.path.abspath(SB)
PASS = FAIL = 0
def ok(name, cond, extra=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  PASS {name}")
    else: FAIL += 1; print(f"  FAIL {name} {extra}")

def utc(s):  # ET wall time string -> aware UTC datetime
    return pd.Timestamp(s, tz=ET).tz_convert("UTC").to_pydatetime()

class FakeBroker:
    def __init__(self): self.calls = []; self.net = 0; self._positions = {}; import threading; self._state_lock = threading.Lock(); self.fail_market = False
    def _resolve_symbol(self, s): return ("MNQZ6", "CME")
    def place_order(self, symbol, order_type, size, entry_price, stop_loss, take_profit):
        self.calls.append(("place_order", order_type, size, entry_price, stop_loss, take_profit)); self.net += size if order_type == "buy" else -size
        self._positions[symbol] = {"size": self.net}
        return {"ticket": f"bot_t{len(self.calls)}", "entry_price": entry_price, "native_stop_attached": True, "supports_stop_modify": True, "can_update_stop": True}
    def flatten_symbol(self, symbol, **k): self.calls.append(("flatten_symbol", symbol)); self.net = 0; self._positions[symbol] = {"size": 0}; return {"flat": True}
    def cancel_entry_bracket_legs(self, oid, symbol, reason="", force=True): self.calls.append(("cancel_legs", oid, reason)); return 2
    def _async_submit_flatten_market_order(self, **k): self.calls.append(("market_close", k["net_qty"])); return (not self.fail_market) and self._apply(k["net_qty"])
    def _apply(self, nq): self.net -= nq; self._positions["MNQ"] = {"size": self.net}; return True
    def _run_sync(self, x, timeout=30): return x
    def get_open_positions(self, symbol=None): return []
    def entry_protection_grace_active(self, oid): return True

def mk_trader(live_mode=False, prior_close=29000.0, past=None, px=None):
    for f in ("gapfade_state.json", "session_pnl.json"):
        try: os.remove(os.path.join("data", f))
        except FileNotFoundError: pass
    with contextlib.redirect_stdout(quiet):
        t = live.LiveRithmicTrader(symbol="MNQ", symbols=["MNQ"], paper_mode=not live_mode, skip_confirm=True)
    t.max_positions = 3; t.max_risk_per_trade_usd = 500.0
    t._gf_cfg, t._gf_state, t._gf_addons = gf.Cfg(), gf.State(os.environ["GAPFADE_STATE_FILE"]), __import__("src.core.addon_setups", fromlist=["x"])
    t._gf_base_risk_cap = 500.0
    t._gf_state.d["past_abs"] = list(past if past is not None else [100.0] * 100)   # q75 threshold = 100
    t._gf_state.d["closes"]["2026-10-27"] = {"px": prior_close, "contract": "MNQZ6"}
    t._px = [prior_close + 150.0]
    t.get_price = lambda sym=None: {"bid": t._px[0] - 0.25, "ask": t._px[0] + 0.25, "last": t._px[0]}
    if live_mode:
        t.broker = FakeBroker(); t._write_board_status = lambda *a, **k: None
        t._broker_open_tickets = lambda: {}
    else: t.broker = None
    return t

def run(t, when, quiet_=True):
    with contextlib.redirect_stdout(quiet if quiet_ else sys.stdout):
        t.gapfade_tick(utc(when))

print("A1 gap up above q75 -> SHORT fade, 1 lot, stop 300 pts, exit 11:00")
t = mk_trader(); run(t, "2026-10-28 09:30:08")
p = list(t.positions.values())
ok("one position opened", len(p) == 1)
if p:
    ok("direction short (fade of gap up)", p[0].direction == "short" and p[0].size == 1)
    ok("stop = entry + 300", abs(p[0].sl - (p[0].entry_price + 300)) < 0.5, f"{p[0].entry_price} {p[0].sl}")
    ok("tagged gapfade, no stock addon exit, own exit 11:00 ET", p[0].addon == "gapfade" and p[0].addon_exit_at is None and p[0].gapfade_exit_at == utc("2026-10-28 11:00"))
    ok("stock add-on time-exit leaves it alone", t.check_addon_time_exits("MNQ", None, utc("2026-10-28 12:00")) == 0 and len(t.positions) == 1)
    ok("risk cap restored", t.max_risk_per_trade_usd == 500.0)
    run(t, "2026-10-28 09:30:15"); ok("no second entry same day", len(t.positions) == 1)
    run(t, "2026-10-28 10:59:50"); ok("holds before 11:00", len(t.positions) == 1)
    run(t, "2026-10-28 11:00:06"); ok("time exit at 11:00 closes it (paper)", len(t.positions) == 0 and t.trades and t.trades[-1]["exit_type"] == "TIME_LIMIT")
print("A2 gap down -> LONG; below threshold -> nothing; late -> nothing")
t = mk_trader(); t._px[0] = 29000 - 150; run(t, "2026-10-28 09:30:05"); ok("long on gap down", [x.direction for x in t.positions.values()] == ["long"])
t = mk_trader(); t._px[0] = 29000 + 60; run(t, "2026-10-28 09:30:05"); ok("no trade below q75 (60 < 100)", len(t.positions) == 0 and len(t._gf_state.d["past_abs"]) == 101)
t = mk_trader(); run(t, "2026-10-28 09:34:30"); ok("no entry after the 3-min window", len(t.positions) == 0)
t = mk_trader(); run(t, "2026-10-28 09:29:50"); ok("no entry before 09:30", len(t.positions) == 0)
t = mk_trader(); run(t, "2026-10-31 09:30:05"); ok("no entry on Saturday", len(t.positions) == 0)
t = mk_trader(past=[100.0] * 10); run(t, "2026-10-28 09:30:05"); ok("warm-up (<20 gaps) -> no trade", len(t.positions) == 0)
print("A3 gates")
t = mk_trader(); t._session_limit_block = lambda: "day loss limit -$500"; run(t, "2026-10-28 09:30:05"); ok("session loss limit blocks entry", len(t.positions) == 0)
t = mk_trader()
for i in range(3):   # fill the cap with same-direction (short) baseline trades
    t.positions[f"b{i}"] = live.Position(order_id=f"b{i}", symbol="MNQ", direction="short", entry_price=29100, size=1, sl=29200, tp=29000, entry_time=utc("2026-10-28 08:00"))
run(t, "2026-10-28 09:30:05"); ok("position cap (3/3) blocks entry", len(t.positions) == 3)
t = mk_trader()
t.positions["b0"] = live.Position(order_id="b0", symbol="MNQ", direction="long", entry_price=29100, size=1, sl=29000, tp=29300, entry_time=utc("2026-10-28 08:00"))
run(t, "2026-10-28 09:30:05"); ok("opposite-direction baseline trade blocks (fade would be SHORT)", len(t.positions) == 1)
t = mk_trader()
t.positions["b0"] = live.Position(order_id="b0", symbol="MNQ", direction="short", entry_price=29100, size=1, sl=29200, tp=29000, entry_time=utc("2026-10-28 08:00"))
run(t, "2026-10-28 09:30:05"); ok("same-direction baseline trade open -> gap fade still enters (2/3)", len(t.positions) == 2)
t = mk_trader(live_mode=True); t.broker._positions["MNQ"] = {"size": 2}; t.broker.net = 2
run(t, "2026-10-28 09:30:05"); ok("broker net opposite (untracked) blocks entry", not any(c[0] == "place_order" for c in t.broker.calls))
print("A4 contract roll between close and open")
t = mk_trader(live_mode=True); t._gf_state.d["closes"]["2026-10-27"]["contract"] = "MNQU6"; run(t, "2026-10-28 09:30:05")
ok("roll day skipped, gap not added to history", not t.broker.calls and len(t._gf_state.d["past_abs"]) == 100)
print("A5 LIVE path with fake broker")
t = mk_trader(live_mode=True); run(t, "2026-10-28 09:30:06")
c = t.broker.calls
ok("one bracket order sent: sell 1 with SL +300 and far TP", len(c) == 1 and c[0][0] == "place_order" and c[0][1] == "sell" and c[0][2] == 1 and abs(c[0][4] - (c[0][3] + 300)) < 0.5 and c[0][5] < c[0][3] - 500, str(c))
ok("risk cap lifted only for the call (600 > 500 would otherwise skip)", len(t.positions) == 1 and t.max_risk_per_trade_usd == 500.0)
run(t, "2026-10-28 11:00:07")
ok("alone on the symbol -> flatten_symbol, position removed", ("flatten_symbol", "MNQ") in t.broker.calls and len(t.positions) == 0)
t = mk_trader(live_mode=True); run(t, "2026-10-28 09:30:06")
t.positions["b0"] = live.Position(order_id="b0", symbol="MNQ", direction="short", entry_price=29100, size=1, sl=29200, tp=29000, entry_time=utc("2026-10-28 10:00"))
t.broker.net += 1; t.broker._positions["MNQ"] = {"size": t.broker.net}   # baseline short added -> net -2 (abs 2)
t.broker.net = -2; t.broker._positions["MNQ"] = {"size": -2}
t.broker.calls.clear(); run(t, "2026-10-28 11:00:07")
names = [x[0] for x in t.broker.calls]
ok("others open -> partial close: market BUY 1 FIRST, then cancel only its own legs, no flatten_symbol", names[:2] == ["market_close", "cancel_legs"] and "flatten_symbol" not in names and t.broker.calls[0][1] == -1, str(t.broker.calls))
ok("baseline trade untouched, net now -1", "b0" in t.positions and t.broker.net == -1 and len(t.positions) == 1)
t = mk_trader(live_mode=True); run(t, "2026-10-28 09:30:06")
t.positions["b0"] = live.Position(order_id="b0", symbol="MNQ", direction="short", entry_price=29100, size=1, sl=29200, tp=29000, entry_time=utc("2026-10-28 10:00"))
t.broker.net = -2; t.broker._positions["MNQ"] = {"size": -2}; t.broker.fail_market = True; t.broker.calls.clear(); run(t, "2026-10-28 11:00:07")
ok("market close fails -> NOTHING cancelled, trade still tracked (stop still working)", [x[0] for x in t.broker.calls] == ["market_close"] and len(t.positions) == 2)
t.broker.fail_market = False; run(t, "2026-10-28 11:00:20"); ok("retry next loop succeeds", len(t.positions) == 1)
t = mk_trader(live_mode=True); os.environ["GAPFADE_PARTIAL_CLOSE"] = "0"; t._gf_cfg = gf.Cfg(); run(t, "2026-10-28 09:30:06")
t.positions["b0"] = live.Position(order_id="b0", symbol="MNQ", direction="short", entry_price=29100, size=1, sl=29200, tp=29000, entry_time=utc("2026-10-28 10:00"))
t.broker.net = -2; t.broker._positions["MNQ"] = {"size": -2}; t.broker.calls.clear(); run(t, "2026-10-28 11:00:07")
ok("GAPFADE_PARTIAL_CLOSE=0 -> keeps stop, sends nothing", t.broker.calls == [] and len(t.positions) == 2); os.environ.pop("GAPFADE_PARTIAL_CLOSE")
t = mk_trader(live_mode=True); t.broker.place_order = lambda **k: None; run(t, "2026-10-28 09:30:06"); ok("broker rejects order -> no position, event marked failed, no retry spam", len(t.positions) == 0 and t._gf_state.d["events"]["2026-10-28"]["action"] == "failed")
print("A6 switch OFF")
os.environ["GAPFADE_ENABLED"] = "0"; ok("enabled() False and install() is a no-op", (not gf.enabled()) and (gf.install(live.LiveRithmicTrader, live) is False)); os.environ["GAPFADE_ENABLED"] = "1"

# ---------------- Part B: replay of historical sessions through the real bot class in paper mode ----------------
print("B replay: last sessions of data/MNQ_1m.csv through LiveRithmicTrader(paper) + gap-fade sleeve")
m1 = pd.read_csv(os.path.join(REPO, "data", "MNQ_1m.csv"), usecols=["datetime", "open", "high", "low", "close"])
m1["t"] = pd.to_datetime(m1.datetime, utc=True); m1 = m1.set_index("t").sort_index()
ety = m1.index.tz_convert(ET); m1["d"] = ety.strftime("%Y-%m-%d"); m1["m"] = ety.hour * 60 + ety.minute
sessions = {}
for d, g in m1.groupby("d"):
    o_ = g[(g.m >= 570) & (g.m <= 572)]; c_ = g[(g.m >= 957) & (g.m <= 959)]
    if o_.empty or c_.empty or (c_.index[-1] - o_.index[0]).total_seconds() / 60 < 300: continue
    sessions[d] = (float(o_.iloc[0].open), float(c_[c_.m == c_.m.max()].iloc[-1].close))
days = sorted(sessions)
NREP = int(os.environ.get("GF_REPLAY_DAYS", "260")); test_days = days[-NREP:]
pre = [abs(sessions[days[k]][0] - sessions[days[k - 1]][1]) for k in range(1, len(days) - NREP)]
t = mk_trader(past=[round(x, 2) for x in pre])
state_ref = {"t": None}
def px_at(ts):  # 1m open at/after ts
    i = m1.index.searchsorted(pd.Timestamp(ts)); return float(m1.iloc[i].open)
t.get_price = lambda sym=None: {"bid": state_ref["p"] - 0.25, "ask": state_ref["p"] + 0.25, "last": state_ref["p"]}
def df5_until(ts):
    w = m1[(m1.index < ts) & (m1.index >= ts - pd.Timedelta(hours=60))]
    r = w.resample("5min", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    r = r[r.index + pd.Timedelta(minutes=5) <= ts]
    out = r.reset_index().rename(columns={"t": "datetime"}); out.attrs["bar_time"] = "start"; return out
t.get_candles = lambda sym, count=100: df5_until(state_ref["now"])
t._gf_state.d["closes"] = {}
stops = 0
for d in test_days:
    D = pd.Timestamp(d)
    prev = days[days.index(d) - 1]
    # previous evening: record close (16:05)
    state_ref["now"] = pd.Timestamp(f"{prev} 16:05", tz=ET).tz_convert("UTC"); state_ref["p"] = px_at(state_ref["now"])
    with contextlib.redirect_stdout(quiet): t.gapfade_tick(state_ref["now"].to_pydatetime())
    t0 = pd.Timestamp(f"{d} 09:30:08", tz=ET).tz_convert("UTC")
    state_ref["now"] = t0; state_ref["p"] = px_at(pd.Timestamp(f"{d} 09:30", tz=ET).tz_convert("UTC"))
    with contextlib.redirect_stdout(quiet): t.gapfade_tick(t0.to_pydatetime())
    if t.positions:
        pos = next(iter(t.positions.values()))
        mins = pd.date_range(pd.Timestamp(f"{d} 09:31", tz=ET), pd.Timestamp(f"{d} 11:00", tz=ET), freq="1min")
        for ts in mins:
            ts = ts.tz_convert("UTC"); state_ref["now"] = ts + pd.Timedelta(seconds=8); state_ref["p"] = px_at(ts)
            bar = m1.loc[ts] if ts in m1.index else None
            if bar is not None and t.positions:
                hit = (bar.high >= pos.sl) if pos.direction == "short" else (bar.low <= pos.sl)
                if hit:
                    with contextlib.redirect_stdout(quiet): t.process_exit(pos.order_id, "STOP_LOSS", pos.sl)
                    stops += 1; break
            with contextlib.redirect_stdout(quiet): t.gapfade_tick(state_ref["now"].to_pydatetime())
            if not t.positions: break
        ok_ = not t.positions
        if not ok_: print("   position left open after 11:00 on", d)
tr = pd.DataFrame(t.trades)
print(f"   replayed {len(test_days)} sessions {test_days[0]}..{test_days[-1]}: trades {len(tr)}, stops {stops}, exits {tr.exit_type.value_counts().to_dict() if len(tr) else {}}")
if len(tr):
    print(f"   net ${tr.pnl.sum():.0f}  win% {100*(tr.pnl>0).mean():.0f}  worst ${tr.pnl.min():.0f}  (bot cost model per round trip applied by process_exit)")
    # independent reference: same rule on the same sessions, numpy only (no bot code)
    allgaps = [abs(sessions[days[k]][0] - sessions[days[k - 1]][1]) for k in range(1, len(days))]
    ref = []
    for k in range(1, len(days)):
        if days[k] not in test_days: continue
        g = sessions[days[k]][0] - sessions[days[k - 1]][1]; past = allgaps[:k - 1]
        if len(past) >= 20 and abs(g) >= np.quantile(past, .75) and g != 0: ref.append((days[k], "short" if g > 0 else "long"))
    got = [(d_, e_["side"]) for d_, e_ in sorted(t._gf_state.d["events"].items()) if e_.get("action") == "entered"]
    ok("replay trades == independent rule (dates and sides)", got == ref, f"bot {len(got)} vs ref {len(ref)}; bot-only {sorted(set(got) - set(ref))[:6]} ref-only {sorted(set(ref) - set(got))[:6]}")
ok("every position closed by 11:00 or its stop", not t.positions)
print(f"\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
