"""NEAT kid (W5w_worst) LIVE PAPER trader.  NEVER PLACES ORDERS.

Two bar sources, picked automatically:
  FILE mode   - main bot (start_live_rithmic.py) is running: read closed MNQ 5m bars it appends to
                data/live_bars_MNQ_5m.csv. Zero Rithmic connection (never kicks the main bot).
  SELF mode   - main bot not running: kid logs in to Rithmic itself, ticker + history plants ONLY
                (no order/pnl plant), every order/cancel/modify/flatten call blocked and raising.
If the main bot starts while the kid is in SELF mode, the kid disconnects at once and switches to FILE mode.
Trades -> data/neat_paper_trades.csv   Console copy -> logs/neat_kid_live.log
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KID = Path(__file__).resolve().parent / "neat_kid"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(KID))
os.chdir(REPO)

import pandas as pd  # noqa: E402
import kid_engine_live as K  # noqa: E402

SYMBOL = "MNQ"
BARS_CSV = REPO / "data" / "neat_kid_bars_MNQ_5m.csv"
TRADES_CSV = REPO / "data" / "neat_paper_trades.csv"
LOG = REPO / "logs" / "neat_kid_live.log"
LOCK = REPO / "data" / "neat_kid_live.lock"
HIST_BARS = 4200
MAX_RISK = 500.0
ET = "America/New_York"
COLS = ["entry_time_et", "side", "entry", "sl", "tp", "risk_usd", "reward_usd", "rr",
        "exit_time_et", "exit", "reason", "pnl_usd", "day_pnl_usd"]

LOG.parent.mkdir(exist_ok=True)
TRADES_CSV.parent.mkdir(exist_ok=True)
_logf = open(LOG, "a", encoding="utf-8")


def say(msg: str = "") -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ---------------------------------------------------------------- safety guards
class OrderBlocked(RuntimeError):
    pass


def _blocked(*_a, **_k):
    raise OrderBlocked("NEAT kid is paper-only: order calls are disabled")


ORDER_WORDS = ("order", "cancel", "modify", "flatten", "close_position", "close_all",
               "bracket", "take_profit", "stop_loss", "submit", "exit_position", "set_sl", "set_tp")


def other_bots() -> list[str]:
    """Any other python process in this repo (bot, backtest that logs in, etc.)."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | ? { $_.Name -match 'python' } | % { \"$($_.ProcessId)|$($_.CommandLine)\" }"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        return [f"could not check processes: {e}"]
    me = os.getpid(); parent = os.getppid()
    bad = []
    for ln in out.splitlines():
        if "|" not in ln:
            continue
        pid, cmd = ln.split("|", 1)
        if not pid.strip().isdigit() or int(pid) in (me, parent) or "neat_live_kid" in cmd:
            continue
        if any(k in cmd.lower() for k in ("start_live", "rithmic", "bott-trader", "trading", "bot")):
            bad.append(f"PID {pid.strip()}: {cmd.strip()[:160]}")
    return bad


def make_connector():
    from async_rithmic.enums import SysInfraType
    import async_rithmic
    from src.broker.rithmic_connector import RithmicConnector

    # library-level block too: even a stray call cannot reach the exchange
    for cls_name in dir(async_rithmic):
        cls = getattr(async_rithmic, cls_name)
        if isinstance(cls, type) and cls_name.endswith("Client"):
            for attr in list(vars(cls)):
                if any(w in attr.lower() for w in ("submit_order", "cancel_order", "modify_order",
                                                    "cancel_all_orders", "exit_position", "place_")):
                    setattr(cls, attr, _blocked)

    class DataOnlyConnector(RithmicConnector):
        async def _async_connect(self, url: str) -> None:
            orig = None
            from async_rithmic import RithmicClient
            orig = RithmicClient.connect

            async def data_only_connect(client, *a, **k):
                k["plants"] = [SysInfraType.TICKER_PLANT, SysInfraType.HISTORY_PLANT]
                return await orig(client, **k)

            RithmicClient.connect = data_only_connect
            try:
                # quotes_only path builds the client then calls connect(); we force ticker+history
                self._quotes_only = True
                await RithmicConnector._async_connect(self, url)
            finally:
                RithmicClient.connect = orig
                self._quotes_only = False  # history requests allowed; no order plant exists

    for name in dir(RithmicConnector):
        low = name.lower()
        if name in ("_cancel_pending_async_tasks", "evaluate_order_flow"):
            continue  # internal asyncio cleanup / read-only tick math, not broker orders
        if callable(getattr(RithmicConnector, name, None)) and any(w in low for w in ORDER_WORDS):
            setattr(DataOnlyConnector, name, _blocked)
    DataOnlyConnector._on_exchange_order_notification = lambda self, *a, **k: None
    b = DataOnlyConnector(live_mode=False)
    b._symbols_to_watch = [SYMBOL]
    return b


def fetch(b, n: int) -> pd.DataFrame:
    """n closed 5m bars, paginated backwards. Rithmic stamps bar END; kid uses bar START."""
    b._ensure_market_data()
    parts, end = [], datetime.now(timezone.utc)
    got = 0
    for _ in range(8):
        page = b._run_sync(b._async_get_candles(SYMBOL, 5, min(1500, n - got), end_time=end,
                                                lookback_hours_override=24.0 * 9),
                           timeout=120)
        if page is None or not len(page):
            break
        parts.insert(0, page)
        got += len(page)
        oldest = pd.to_datetime(page["datetime"].iloc[0], utc=True)
        if got >= n or len(page) < 50:
            break
        end = (oldest - timedelta(minutes=1)).to_pydatetime()
    if not parts:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    df = pd.concat(parts, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["datetime"] <= now]  # finished bars only
    df["datetime"] = df["datetime"] - pd.Timedelta(minutes=5)
    return df.drop_duplicates("datetime", keep="last").sort_values("datetime").reset_index(drop=True)


LIVE_CSV = REPO / "data" / "live_bars_MNQ_5m.csv"   # written by the main bot (shadow_bar_export)
STALE_MIN = 15


def read_live() -> pd.DataFrame:
    """Closed 5m bars from the main bot's export file. NO network, NO Rithmic.
    The export stamps bars by END time (ts_et); the kid uses bar START, so shift -5m."""
    if not LIVE_CSV.is_file():
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    df = pd.read_csv(LIVE_CSV)
    df["datetime"] = pd.to_datetime(df["ts_et"], utc=True) - pd.Timedelta(minutes=5)
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    return df.drop_duplicates("datetime", keep="last").sort_values("datetime").reset_index(drop=True)


_last_stale_warn = [0.0]


def is_rth_like() -> bool:
    n = pd.Timestamp.now(tz=ET)
    return n.weekday() < 5 and not (17 <= n.hour < 18)


def warn_stale(age: float) -> None:
    if time.time() - _last_stale_warn[0] > 600:
        _last_stale_warn[0] = time.time()
        say(f"    (no new bar from main bot for {age:.0f} min - is the main bot running/connected?)")




def main_bot_running() -> list[str]:
    """PIDs/cmdlines of a running start_live_rithmic.py (the main bot)."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | % { \"$($_.ProcessId)|$($_.CommandLine)\" }"],
            capture_output=True, text=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except Exception:
        return []
    hits = []
    for ln in out.splitlines():
        if "|" in ln and "start_live_rithmic" in ln.lower():
            pid, cmd = ln.split("|", 1)
            hits.append(f"PID {pid.strip()}: {cmd.strip()[:120]}")
    return hits


def connected_plants(b) -> list[str]:
    try:
        return [n for n, pl in b._client.plants.items() if getattr(pl, "is_connected", False)]
    except Exception:
        return []


def self_connect():
    """Log in data-only. Returns connector or None. Aborts if an order/pnl plant is connected."""
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
    os.environ["CANDLE_HISTORY_MAX_DAYS"] = "25"
    b = make_connector()
    say("SELF mode: logging in to Rithmic (data only: ticker + history)...")
    b.initialize()
    if not b.connected:
        say(f"Login failed: {b._last_connect_error}")
        return None
    live = connected_plants(b)
    say(f"Logged in. system={os.getenv('RITHMIC_SYSTEM')}  plants connected={live}")
    if any(x in live for x in ("order", "pnl")):
        say("STOP: an order/pnl plant connected - kid refuses to run like that. Disconnecting.")
        disconnect(b)
        return None
    try:
        b.place_order(SYMBOL, "BUY", 1)
        say("!! place_order did not raise - refusing to run.")
        disconnect(b)
        return None
    except OrderBlocked:
        say("Safety check passed: order calls are blocked.")
    except AttributeError:
        say("Safety check passed: connector has no place_order.")
    return b


def disconnect(b) -> None:
    try:
        b.shutdown()
    except Exception as e:
        say(f"   (disconnect warning: {e})")


# ---------------------------------------------------------------- display
def et(ts) -> str:
    return pd.Timestamp(ts).tz_convert(ET).strftime("%a %b %d %I:%M %p ET")


def money(x: float) -> str:
    return f"+${x:,.2f}" if x >= 0 else f"-${-x:,.2f}"


def plan(side, entry, sl, tp):
    risk = abs(entry - sl) * K.PV
    reward = abs(tp - entry) * K.PV if tp not in ("", None) else 0.0
    return risk, reward, (reward / risk if risk else 0.0)


def show_entry(side, entry, sl, tp, when):
    risk, reward, rr = plan(side, entry, sl, tp)
    verb = "BUY " if side == "LONG" else "SELL"
    say(f">>> PAPER {verb} 1 MNQ @ {entry:.2f}  ({et(when)})")
    say(f"    SL {sl:.2f}   TP {tp:.2f}   risk ${risk:,.2f}   reward ${reward:,.2f}   R:R 1:{rr:.2f}")
    if risk > MAX_RISK:
        say(f"    !! risk over ${MAX_RISK:.0f} cap - a real bot would skip this")


def main() -> int:
    say("=" * 72)
    say("NEAT KID (W5w_worst) - LIVE PAPER MODE.  NO ORDERS ARE EVER SENT.")
    say("=" * 72)
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().strip())
            alive = str(pid) in subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                                               capture_output=True, text=True).stdout
        except Exception:
            alive = False
        if alive:
            say(f"STOP: kid already running (PID {pid}).")
            return 2
    LOCK.write_text(str(os.getpid()))

    b = None
    mb = main_bot_running()
    if mb:
        mode = "FILE"
        say(f"Main bot is running ({mb[0]}) -> FILE mode: reading {LIVE_CSV.name}, NO Rithmic login.")
    else:
        mode = "SELF"
        say("Main bot is NOT running -> SELF mode: kid logs in to Rithmic itself (paper only).")
        b = self_connect()
        if b is None:
            return 1

    cols, norm = K.load_norm()
    p = json.loads((KID / "w5w_params.json").read_text())
    net = K.load_net(KID / "w5w_net.npz")
    bars = fetch(b, HIST_BARS) if mode == "SELF" else read_live()
    if os.path.isfile(BARS_CSV):
        try:
            old = pd.read_csv(BARS_CSV, parse_dates=["datetime"])
            old["datetime"] = pd.to_datetime(old["datetime"], utc=True)
            bars = pd.concat([old, bars]).drop_duplicates("datetime", keep="last").sort_values("datetime")
        except Exception:
            pass
    bars = bars.tail(6000).reset_index(drop=True)
    if not len(bars):
        say("No bars yet; waiting.")
    else:
        say(f"Got {len(bars)} closed 5m bars: {et(bars.datetime.iloc[0])} -> {et(bars.datetime.iloc[-1])}")
    if len(bars) < 1700:
        say("Not enough history to warm up the kid's indicators yet; will keep collecting.")

    frame, F, _ = K.build_features(bars, cols, norm)
    if len(frame) > 2400:
        rp = K.simulate(frame, F, net, p, cost_mult=1.0, start=len(frame) - 830, liquidate_end=True)
        say(f"Replay of the last ~3 days: {len(rp)} pretend trades, {money(sum(t['pnl'] for t in rp))}")
        for t in rp[-3:]:
            say(f"   {t['side']:5s} {et(t['entry_time'])} @ {t['entry']:.2f} -> {t['reason']} {t['exit']:.2f}  {money(t['pnl'])}")

    live_from = pd.Timestamp.now(tz="UTC").floor("5min")
    seen: set[str] = set()
    if TRADES_CSV.exists():
        with open(TRADES_CSV, newline="") as f:
            for r in csv.DictReader(f):
                seen.add(r["entry_time_et"] + r["side"])
    announced = None
    last_bar = None
    gone = 0
    say(f"LIVE ({mode} mode): kid trades bars from {et(live_from)} on. Checking every 30s...")
    while True:
        try:
            # ---- mode switching: never fight the main bot for the Rithmic session
            mb = main_bot_running()
            if mode == "SELF" and mb:
                say(f"!!! Main bot started ({mb[0]}). Kid DISCONNECTING from Rithmic now -> FILE mode.")
                disconnect(b)
                b = None
                mode = "FILE"
                say("Kid is now in FILE mode (no Rithmic login), reading the main bot's bar file.")
            elif mode == "FILE" and not mb:
                gone += 1
                if gone >= 2:
                    say("Main bot is no longer running -> switching to SELF mode (kid logs in, paper only).")
                    b = self_connect()
                    if b is not None:
                        mode = "SELF"
                    gone = 0
            else:
                gone = 0

            if mode == "SELF":
                new = fetch(b, 30)
            else:
                new = read_live()
                if len(new):
                    age = (pd.Timestamp.now(tz="UTC") - (new.datetime.iloc[-1] + pd.Timedelta(minutes=5))).total_seconds() / 60
                    if age > STALE_MIN and is_rth_like():
                        warn_stale(age)
            bars = pd.concat([bars, new]).drop_duplicates("datetime", keep="last").sort_values("datetime").reset_index(drop=True)
            if len(bars) and (last_bar is None or bars.datetime.iloc[-1] != last_bar):
                last_bar = bars.datetime.iloc[-1]
                bars.to_csv(BARS_CSV, index=False)
                frame, F, _ = K.build_features(bars, cols, norm)
                idx = frame.index[frame.datetime >= live_from]
                trades = []
                if len(idx) and len(frame) > 1600:
                    trades = K.simulate(frame, F, net, p, cost_mult=1.0, start=max(1600, int(idx[0]) - 1),
                                        liquidate_end=False)
                op = K.simulate.last_open if len(idx) else None
                today = pd.Timestamp(last_bar).tz_convert(ET)
                gday = (today + pd.Timedelta(hours=6)).date()
                day_pnl = 0.0
                for t in trades:
                    tday = (pd.Timestamp(t["exit_time"]).tz_convert(ET) + pd.Timedelta(hours=6)).date()
                    if tday == gday:
                        day_pnl += t["pnl"]
                    key = pd.Timestamp(t["entry_time"]).tz_convert(ET).strftime("%Y-%m-%d %H:%M") + t["side"]
                    if key in seen:
                        continue
                    seen.add(key)
                    if announced != key:
                        show_entry(t["side"], t["entry"], t["sl"], t["tp"], t["entry_time"])
                    risk, reward, rr = plan(t["side"], t["entry"], t["sl"], t["tp"])
                    why = {"stop": "STOP LOSS hit", "target": "TAKE PROFIT hit", "flat": "kid closed it / session rule"}.get(t["reason"], t["reason"])
                    say(f"<<< CLOSED {t['side']} @ {t['exit']:.2f}  {why}  P&L {money(t['pnl'])}   day P&L {money(day_pnl)}  ({et(t['exit_time'])})")
                    row = dict(entry_time_et=key[:-len(t['side'])], side=t["side"], entry=round(t["entry"], 2),
                               sl=round(t["sl"], 2), tp=round(float(t["tp"]), 2) if t["tp"] != "" else "", risk_usd=round(risk, 2), reward_usd=round(reward, 2),
                               rr=round(rr, 2), exit_time_et=pd.Timestamp(t["exit_time"]).tz_convert(ET).strftime("%Y-%m-%d %H:%M"),
                               exit=round(t["exit"], 2), reason=t["reason"], pnl_usd=round(t["pnl"], 2), day_pnl_usd=round(day_pnl, 2))
                    hdr = not TRADES_CSV.exists()
                    with open(TRADES_CSV, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=COLS)
                        if hdr:
                            w.writeheader()
                        w.writerow(row)
                c = float(frame.close.iloc[-1])
                if op:
                    key = pd.Timestamp(op["since"]).tz_convert(ET).strftime("%Y-%m-%d %H:%M") + op["side"]
                    if announced != key:
                        show_entry(op["side"], op["entry"], op["sl"], op["tp"] or 0.0, op["since"])
                        announced = key
                    sgn = 1 if op["side"] == "LONG" else -1
                    say(f"    bar {et(last_bar)} close {c:.2f} | in {op['side']} from {op['entry']:.2f}  open P&L {money((c - op['entry']) * sgn * K.PV)} | day P&L {money(day_pnl)}")
                else:
                    say(f"    bar {et(last_bar)} close {c:.2f} | flat, waiting for signal | day P&L {money(day_pnl)}")
                if day_pnl <= -K.CAP:
                    say("    Day loss limit -$500 reached: kid stops for the rest of this session.")
        except OrderBlocked as e:
            say(f"BLOCKED order attempt (none sent): {e}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            say(f"update error (no orders involved): {e}")
        for _ in range(6):
            time.sleep(5)
            if mode == "SELF" and main_bot_running():
                break


if __name__ == "__main__":
    try:
        rc = main()
    except KeyboardInterrupt:
        say("Stopped by you (Ctrl+C). No orders were ever placed.")
        rc = 0
    finally:
        try:
            if LOCK.exists() and LOCK.read_text().strip() == str(os.getpid()):
                LOCK.unlink()
        except Exception:
            pass
    sys.exit(rc)
