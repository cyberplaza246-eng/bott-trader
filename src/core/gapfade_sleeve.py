"""NY-open GAP FADE sleeve for the revamp bot (GAPFADE_ENABLED=1; default OFF).

Rule (research 2026-10-28, REPORT.md addendum 3):
  gap  = price at the 09:30 ET open - the 15:59 ET close of the previous session
  fire = |gap| >= expanding 75th percentile of all PAST |gap| (first 20 events skipped)
  side = FADE the gap (gap up -> SHORT, gap down -> LONG), 1 MNQ, no confirmation
  exit = time exit at 09:30 ET + 90 min (11:00 ET), or the protective stop first.
Live additions (not in the no-stop research run): wide protective stop, far TP (so the broker
bracket is valid), position-cap / session-loss-limit / no-opposite-exposure gates.

Install: run_revamp_gapfade.py calls install(LiveRithmicTrader). With GAPFADE_ENABLED off,
install() does nothing at all, so the stock bot behaves exactly as before.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    ET = None

log = logging.getLogger("gapfade")
LABEL = "NY-open gap fade"


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or str(v).strip() == "" else str(v).strip()


def _envf(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return float(default)


def enabled() -> bool:
    return _env("GAPFADE_ENABLED", "0").lower() in ("1", "true", "yes", "on")


class Cfg:
    """Read at install time (env). Defaults = the researched setup."""
    def __init__(self) -> None:
        self.symbol = _env("GAPFADE_SYMBOL", "MNQ")
        self.quantile = _envf("GAPFADE_QUANTILE", 0.75)
        self.hold_min = int(_envf("GAPFADE_HOLD_MIN", 90))
        self.min_past = int(_envf("GAPFADE_MIN_PAST", 20))
        self.window_sec = int(_envf("GAPFADE_ENTRY_WINDOW_SEC", 180))   # enter within 3 min of 09:30 (delay hurts: 3 min = -25%)
        self.stop_pts = _envf("GAPFADE_STOP_PTS", 300.0)                # 300 pts = $600/MNQ; 240 fits the $500 cap
        self.far_tp_pts = _envf("GAPFADE_FAR_TP_PTS", 800.0)            # never meant to hit; keeps the broker bracket valid
        self.risk_cap_usd = _envf("GAPFADE_RISK_CAP_USD", 650.0)        # lifts MAX_RISK_PER_TRADE_USD for THIS sleeve only
        self.contracts = int(_envf("GAPFADE_CONTRACTS", 1))
        self.partial_close = _env("GAPFADE_PARTIAL_CLOSE", "1").lower() in ("1", "true", "yes", "on")
        self.state_file = _env("GAPFADE_STATE_FILE", os.path.join("data", "gapfade_state.json"))
        self.conf = _envf("GAPFADE_CONF_LABEL", 0.56)


# --------------------------------------------------------------------------- state
class State:
    def __init__(self, path: str) -> None:
        self.path = path
        self.d: Dict[str, Any] = {"version": 1, "past_abs": [], "closes": {}, "events": {}, "open": {}}
        self.load()

    def load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.d.update(json.load(f))
        except FileNotFoundError:
            pass
        except Exception as e:  # corrupt file: keep running with what we have, never crash the bot
            log.warning("gapfade state unreadable (%r) - starting from empty", e)

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path) or ".", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.d, f, indent=1)
            os.replace(tmp, self.path)
        except Exception as e:
            log.warning("gapfade state save failed: %r", e)


# --------------------------------------------------------------------------- pure logic
def threshold(past_abs: List[float], q: float, min_past: int) -> Optional[float]:
    """Expanding quantile of PAST |gap|; None until min_past events exist."""
    if len(past_abs) < min_past:
        return None
    return float(np.quantile(np.asarray(past_abs, dtype=float), q))


def decide(gap: float, past_abs: List[float], q: float, min_past: int) -> Dict[str, Any]:
    thr = threshold(past_abs, q, min_past)
    out: Dict[str, Any] = {"gap": float(gap), "thr": thr, "fire": False, "side": 0}
    if thr is None:
        out["why"] = f"warm-up ({len(past_abs)}/{min_past} past gaps)"
    elif gap == 0 or abs(gap) < thr:
        out["why"] = f"|gap| {abs(gap):.2f} < q{int(q*100)} threshold {thr:.2f}"
    else:
        out["fire"] = True
        out["side"] = -1 if gap > 0 else 1          # fade
        out["why"] = f"|gap| {abs(gap):.2f} >= q{int(q*100)} threshold {thr:.2f}"
    return out


def prior_close_from_5m(df5: Optional[pd.DataFrame], day_et: pd.Timestamp, addons_mod) -> Optional[tuple]:
    """Latest 15:55 ET 5m bar (= the 15:59 close) strictly before day_et's date. Returns (date_str, close)."""
    if df5 is None or len(df5) == 0:
        return None
    b = addons_mod.closed_bars(df5)
    if b is None or b.empty:
        return None
    sel = b[(b.index.hour == 15) & (b.index.minute == 55) & (b.index.normalize().tz_localize(None) < pd.Timestamp(day_et.date()))]
    if sel.empty:
        return None
    return sel.index[-1].strftime("%Y-%m-%d"), float(sel.iloc[-1]["close"])


# --------------------------------------------------------------------------- trader methods
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _et(now: datetime):
    return now.astimezone(ET)


def gf_log(self, msg: str, warn: bool = False) -> None:
    try:
        (log.warning if warn else log.info)(msg)
        print(f"   [GAPFADE] {msg}")
        from src.utils.logger import bot_logger  # type: ignore
        (bot_logger.warning if warn else bot_logger.info)(f"GAPFADE {msg}")
    except Exception:
        pass


def gf_tick(self, now: Optional[datetime] = None) -> None:
    """Called every loop (~5-10 s) from sync_broker_position. Never raises."""
    try:
        _gf_tick(self, now or _now_utc())
    except Exception as e:  # the sleeve must never break the main loop
        try:
            log.exception("gapfade tick error")
            gf_log(self, f"tick error (ignored): {e!r}", warn=True)
        except Exception:
            pass


def _gf_tick(self, now: datetime) -> None:
    cfg: Cfg = self._gf_cfg
    st: State = self._gf_state
    if cfg.symbol not in list(getattr(self, "symbols", []) or [cfg.symbol]):
        return
    et = _et(now)
    D = et.strftime("%Y-%m-%d")
    mins = et.hour * 60 + et.minute
    # 1) manage open gap-fade trades (time exit) - runs on any day/time
    _gf_manage(self, now)
    if et.weekday() >= 5:
        return
    # 2) remember the 15:59 close once the day session is over (needed for Monday / after a restart)
    if 16 * 60 <= mins < 20 * 60 and D not in st.d["closes"]:
        if (now - getattr(self, "_gf_last_close_try", datetime(2000, 1, 1, tzinfo=timezone.utc))).total_seconds() >= 120:
            self._gf_last_close_try = now
            _gf_record_close(self, now, D)
    # 3) entry decision inside the first seconds after 09:30
    open_dt = et.replace(hour=9, minute=30, second=0, microsecond=0)
    el = (et - open_dt).total_seconds()
    ev = st.d["events"].get(D)
    if ev is None and 0 <= el <= cfg.window_sec:
        _gf_try_entry(self, now, D)
    elif ev is not None and ev.get("provisional") and el >= 6 * 60:
        _gf_finalize_gap(self, now, D, ev)


def _gf_contract(self) -> str:
    """Front-month contract code the bot is trading right now ('' when unknown, e.g. paper without a broker)."""
    try:
        if getattr(self, "broker", None) is not None:
            return str(self.broker._resolve_symbol(self._gf_cfg.symbol)[0] or "")
    except Exception:
        pass
    return ""


def _close_px(v) -> float:
    return float(v["px"]) if isinstance(v, dict) else float(v)


def _close_contract(v) -> str:
    return str(v.get("contract", "")) if isinstance(v, dict) else ""


def _gf_record_close(self, now: datetime, D: str) -> None:
    cfg: Cfg = self._gf_cfg
    st: State = self._gf_state
    try:
        df = self.get_candles(cfg.symbol, 320)
        r = prior_close_from_5m(df, pd.Timestamp(now.astimezone(ET).date()) + pd.Timedelta(days=1), self._gf_addons)
    except Exception as e:
        r = None
        gf_log(self, f"could not read the 15:55 bar after the close ({e!r})", warn=True)
    if r and r[0] == D:
        st.d["closes"][D] = {"px": r[1], "contract": _gf_contract(self)}
        st.save()
        gf_log(self, f"saved {D} 15:59 close {r[1]:.2f} ({_gf_contract(self) or 'contract n/a'})")


def _gf_prior_close(self, now: datetime, D: str) -> Optional[tuple]:
    cfg: Cfg = self._gf_cfg
    st: State = self._gf_state
    prev = sorted(k for k in st.d["closes"] if k < D)
    if prev and (pd.Timestamp(D) - pd.Timestamp(prev[-1])).days <= 5:
        return prev[-1], _close_px(st.d["closes"][prev[-1]]), _close_contract(st.d["closes"][prev[-1]])
    try:   # fallback: ~3.5 days of 5m bars (one history request, only when the state has nothing recent)
        df = self.get_candles(cfg.symbol, 1000)
        r = prior_close_from_5m(df, pd.Timestamp(D), self._gf_addons)
        if r:
            st.d["closes"][r[0]] = {"px": r[1], "contract": _gf_contract(self)}
            st.save()
            return r[0], r[1], _gf_contract(self)
    except Exception as e:
        gf_log(self, f"prior-close fallback failed ({e!r})", warn=True)
    return None


def _gf_price(self, symbol: str) -> Optional[float]:
    q = self.get_price(symbol) or {}
    last = float(q.get("last") or 0)
    bid = float(q.get("bid") or 0)
    ask = float(q.get("ask") or 0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0 if last <= 0 else last
    return last if last > 0 else None


def _gf_try_entry(self, now: datetime, D: str) -> None:
    cfg: Cfg = self._gf_cfg
    st: State = self._gf_state
    pc = _gf_prior_close(self, now, D)
    if pc is None:
        gf_log(self, f"{D}: no prior 15:59 close known yet - retrying until the window ends", warn=True)
        return
    px = _gf_price(self, cfg.symbol)
    if px is None or px <= 0:
        return  # retry next loop
    gap = px - pc[1]
    cnow = _gf_contract(self)
    if pc[2] and cnow and pc[2] != cnow:   # contract roll between the close and the open: the "gap" is the calendar spread
        gf_log(self, f"{D}: contract changed {pc[2]} -> {cnow}; gap {gap:+.2f} is a roll artifact - no trade, not added to history")
        st.d["events"][D] = {"gap": round(gap, 2), "action": "skip", "why": "contract roll"}
        st.save()
        return
    if abs(gap) > 3000:
        gf_log(self, f"{D}: implausible gap {gap:.2f} (price {px:.2f}, close {pc[1]:.2f}) - skipping", warn=True)
        st.d["events"][D] = {"gap": gap, "action": "skip", "why": "implausible gap"}
        st.save()
        return
    dec = decide(gap, st.d["past_abs"], cfg.quantile, cfg.min_past)
    ev = {"gap": round(gap, 2), "prior_close": pc[1], "open_px": px, "thr": dec["thr"], "provisional": True,
          "action": "none", "why": dec["why"]}
    # the gap enters the expanding history whether or not we trade it (same as the research)
    st.d["past_abs"].append(round(abs(gap), 2))
    st.d["events"][D] = ev
    if not dec["fire"]:
        gf_log(self, f"{D}: gap {gap:+.2f} vs close {pc[1]:.2f} - no trade ({dec['why']})")
        st.save()
        return
    side = dec["side"]
    direction = "long" if side > 0 else "short"
    block = _gf_entry_block(self, direction)
    if block:
        ev.update(action="skip", why=block)
        gf_log(self, f"{D}: gap {gap:+.2f} would FADE {direction.upper()} but skipped: {block}")
        st.save()
        return
    q = self.get_price(cfg.symbol) or {}
    entry = float(q.get("ask" if side > 0 else "bid") or px)
    sl = entry - side * cfg.stop_pts
    tp = entry + side * cfg.far_tp_pts
    open_dt = _et(now).replace(hour=9, minute=30, second=0, microsecond=0)
    exit_at = (open_dt + timedelta(minutes=cfg.hold_min)).astimezone(timezone.utc)
    sig = {
        "symbol": cfg.symbol, "direction": direction, "entry": entry, "sl": sl, "tp": tp,
        "atr": float(cfg.stop_pts), "confidence": cfg.conf, "ensemble_signal": "BUY" if side > 0 else "SELL",
        "model_signals": {"addon": "gapfade"}, "regime": "addon", "trailing_enabled": False,
        "exit_sleeve": "addon", "addon": "gapfade", "fixed_contracts": cfg.contracts,
        "addon_exit_at": exit_at, "addon_label": LABEL,
    }
    gf_log(self, f"{D}: gap {gap:+.2f} ({dec['why']}) -> FADE {direction.upper()} {cfg.contracts} {cfg.symbol} @ ~{entry:.2f} "
                 f"stop {sl:.2f} ({cfg.stop_pts:g} pts = ${cfg.stop_pts*self._spec(cfg.symbol)['point_value']:.0f}), "
                 f"time exit {exit_at.astimezone(ET).strftime('%H:%M')} ET")
    before = set(self.positions)
    had = "max_risk_per_trade_usd" in self.__dict__
    old_cap = self.__dict__.get("max_risk_per_trade_usd")
    base_cap = float(old_cap if old_cap is not None else getattr(self, "_gf_base_risk_cap", 0.0) or 0.0)
    try:
        if base_cap > 0:   # cap is ON: lift it for THIS order only (a 0 cap = off stays off)
            self.max_risk_per_trade_usd = max(base_cap, cfg.risk_cap_usd)
        ok = bool(self.place_order(sig))
    finally:
        if had:
            self.max_risk_per_trade_usd = old_cap
        else:
            self.__dict__.pop("max_risk_per_trade_usd", None)
    new = [k for k in self.positions if k not in before]
    if not ok or not new:
        ev.update(action="failed", why="order not placed")
        gf_log(self, f"{D}: order was NOT placed (see log)", warn=True)
        st.save()
        return
    oid = new[0]
    pos = self.positions[oid]
    pos.addon = "gapfade"
    pos.gapfade_exit_at = exit_at      # our own exit; the stock add-on time-exit must not touch it
    pos.addon_exit_at = None
    ev.update(action="entered", order_id=oid, side=direction, entry=float(pos.entry_price), sl=sl,
              exit_at=exit_at.isoformat())
    st.d["open"][oid] = {"direction": direction, "size": int(pos.size), "exit_at": exit_at.isoformat(), "day": D}
    st.save()


def _gf_entry_block(self, direction: str) -> Optional[str]:
    cfg: Cfg = self._gf_cfg
    why = self._session_limit_block()
    if why:
        return f"session limit: {why}"
    n = self._effective_open_count()
    if n >= self.max_positions:
        return f"position cap {n}/{self.max_positions}"
    for p in self.positions.values():
        if p.symbol == cfg.symbol and p.direction != direction:
            return "opposite-direction trade open (broker nets per symbol)"
    # broker net sign when known (untracked exposure)
    try:
        raw = None
        if getattr(self, "broker", None) is not None and hasattr(self.broker, "_positions"):
            with self.broker._state_lock:  # type: ignore[attr-defined]
                raw = self.broker._positions.get(cfg.symbol)  # type: ignore[attr-defined]
        if raw:
            sg = int(raw.get("size", 0) or 0)
            if sg != 0 and ((sg > 0) != (direction == "long")):
                return "broker net is opposite to the fade"
    except Exception:
        pass
    return None


def _gf_finalize_gap(self, now: datetime, D: str, ev: Dict[str, Any]) -> None:
    """Replace the provisional (live-price) gap in the history with the true 09:30 bar open when available."""
    st: State = self._gf_state
    cfg: Cfg = self._gf_cfg
    try:
        df = self.get_candles(cfg.symbol, 320)
        b = self._gf_addons.closed_bars(df, now=now)
        if b is None:
            return
        sel = b[(b.index.strftime("%Y-%m-%d") == D) & (b.index.hour == 9) & (b.index.minute == 30)]
        if sel.empty:
            return
        true_gap = float(sel.iloc[0]["open"]) - float(ev["prior_close"])
        old = abs(float(ev["gap"]))
        for i in range(len(st.d["past_abs"]) - 1, -1, -1):
            if abs(st.d["past_abs"][i] - round(old, 2)) < 1e-9:
                st.d["past_abs"][i] = round(abs(true_gap), 2)
                break
        ev["gap_open_bar"] = round(true_gap, 2)
        ev["provisional"] = False
        st.save()
    except Exception as e:
        ev["provisional"] = False
        gf_log(self, f"{D}: could not finalize gap from the 09:30 bar ({e!r}); keeping the live-price gap", warn=True)
        st.save()


# --------------------------------------------------------------------------- exits
def _gf_manage(self, now: datetime) -> None:
    cfg: Cfg = self._gf_cfg
    st: State = self._gf_state
    if not getattr(self, "_gf_boot_checked", False):
        self._gf_boot_checked = True
        for oid, meta in list(st.d["open"].items()):   # left over from a previous run of the bot
            net = self._broker_net_qty(cfg.symbol) if self._live_broker() else 0
            if net:
                gf_log(self, f"WARNING: previous run left a gap-fade {meta.get('direction')} trade ({oid}, exit was due "
                             f"{meta.get('exit_at')}). Broker shows net {net} {cfg.symbol}. It is now an ordinary/adopted "
                             f"position WITHOUT the 11:00 time exit - close it by hand if it is the gap fade.", warn=True)
            st.d["open"].pop(oid, None)
        st.save()
    for oid, pos in list(self.positions.items()):
        ex = getattr(pos, "gapfade_exit_at", None)
        if getattr(pos, "addon", "") != "gapfade" or ex is None:
            continue
        if ex.tzinfo is None:
            ex = ex.replace(tzinfo=timezone.utc)
        if now < ex:
            continue
        _gf_time_exit(self, oid, pos, now)
    # forget closed trades
    for oid in list(st.d["open"]):
        if oid not in self.positions:
            st.d["open"].pop(oid, None)
            st.save()


def _gf_px_for_exit(self, pos) -> float:
    q = self.get_price(pos.symbol) or {}
    px = float(q.get("bid" if pos.direction == "long" else "ask") or q.get("last") or 0)
    return px if px > 0 else float(pos.entry_price)


def _gf_time_exit(self, oid: str, pos, now: datetime) -> bool:
    cfg: Cfg = self._gf_cfg
    px = _gf_px_for_exit(self, pos)
    if not self._live_broker():
        gf_log(self, f"{oid}: 90-min time exit (paper) @ {px:.2f}")
        self.process_exit(oid, "TIME_LIMIT", px)
        return True
    others = [p for k, p in self.positions.items() if k != oid and p.symbol == pos.symbol]
    net = self._broker_net_qty(pos.symbol)
    if net is None:
        gf_log(self, f"{oid}: time exit due but broker net unknown - stop stays working, retry next loop", warn=True)
        return False
    if not others and int(net) == int(pos.size):
        res: Dict[str, Any] = {}
        try:
            res = self.broker.flatten_symbol(pos.symbol) or {}
        except Exception as e:
            gf_log(self, f"{oid}: flatten failed {e!r}", warn=True)
        if res.get("flat"):
            gf_log(self, f"{oid}: 90-min time exit - flattened (only trade on {pos.symbol})")
            self.process_exit(oid, "TIME_LIMIT", px)
            return True
        gf_log(self, f"{oid}: flatten not confirmed ({res}) - stop stays working, retry next loop", warn=True)
        return False
    if not cfg.partial_close:
        gf_log(self, f"{oid}: time exit due but other {pos.symbol} trades are open and GAPFADE_PARTIAL_CLOSE=0 - "
                     f"holding with its protective stop", warn=True)
        return False
    return _gf_partial_close(self, oid, pos, px)


def _gf_partial_close(self, oid: str, pos, px: float) -> bool:
    """Close ONLY this trade's contract(s) with a market order on the closing side, then cancel its own
    bracket legs. Market first, legs second: if the market order fails nothing is cancelled and the
    protective stop is still working."""
    b = self.broker
    try:
        broker_sym, exchange = b._resolve_symbol(pos.symbol)
        net_qty = int(pos.size) if pos.direction == "long" else -int(pos.size)   # +ve -> SELL
        ok = bool(b._run_sync(b._async_submit_flatten_market_order(
            symbol=pos.symbol, broker_sym=broker_sym, exchange=exchange, net_qty=net_qty), timeout=40))
    except Exception as e:
        gf_log(self, f"{oid}: partial close error {e!r} - stop stays working", warn=True)
        return False
    if not ok:
        gf_log(self, f"{oid}: partial close market order not confirmed - stop stays working, retry next loop", warn=True)
        return False
    try:
        n = b.cancel_entry_bracket_legs(oid, pos.symbol, reason="gapfade_time_exit")
        gf_log(self, f"{oid}: 90-min time exit - closed {pos.size} of net via market, cancelled {n} bracket leg(s)")
    except Exception as e:
        gf_log(self, f"{oid}: closed, but cancelling its bracket legs raised {e!r} (orphan sweep will clean up)", warn=True)
    self.process_exit(oid, "TIME_LIMIT", px)
    return True


# --------------------------------------------------------------------------- install
def _entry_line_factory(orig):
    def entry_line(signal, contracts, fill):
        if signal.get("addon") == "gapfade":
            ex = pd.Timestamp(signal["addon_exit_at"]).tz_convert(ET).strftime("%H:%M")
            return (f"{signal['direction'].upper()} {contracts} {signal['symbol']} at {fill:,.2f}. "
                    f"SL {signal['sl']:,.2f}, far TP {signal['tp']:,.2f}. ({LABEL}, time exit {ex} ET)")
        return orig(signal, contracts, fill)
    return entry_line


def install(trader_cls, module=None) -> bool:
    """Patch LiveRithmicTrader in memory (no file is modified). No-op unless GAPFADE_ENABLED=1."""
    if not enabled():
        print("[GAPFADE] OFF (GAPFADE_ENABLED not set) - stock bot behaviour")
        return False
    from src.core import addon_setups as addons
    cfg = Cfg()
    state = State(cfg.state_file)
    base_cap = float(getattr(module, "MAX_RISK_PER_TRADE_USD", 0.0) or 0.0) if module is not None else 0.0
    addons.entry_line = _entry_line_factory(addons.entry_line)
    orig_sync = trader_cls.sync_broker_position

    def sync_with_gapfade(self, *a, **k):
        if not hasattr(self, "_gf_cfg"):
            self._gf_cfg, self._gf_state, self._gf_addons = cfg, state, addons
            self._gf_base_risk_cap = base_cap
        gf_tick(self)
        return orig_sync(self, *a, **k)

    trader_cls.sync_broker_position = sync_with_gapfade
    trader_cls.gapfade_tick = gf_tick
    thr = threshold(state.d["past_abs"], cfg.quantile, cfg.min_past)
    print("=" * 70)
    print(f"[GAPFADE] ON  {cfg.symbol} NY-open gap fade: q{int(cfg.quantile*100)} threshold "
          f"{('%.1f pts' % thr) if thr is not None else 'warming up'} ({len(state.d['past_abs'])} past gaps), "
          f"fade, {cfg.contracts} lot, hold {cfg.hold_min} min, stop {cfg.stop_pts:g} pts "
          f"(${cfg.stop_pts*2:.0f}), enter within {cfg.window_sec}s of 09:30 ET")
    print(f"[GAPFADE] state file: {cfg.state_file} | partial-close when other trades open: {cfg.partial_close}")
    print("=" * 70)
    return True
