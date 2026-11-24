"""FUNDED 12K bot settings + guard (FUNDED_12K_ON=1). Same interface as flex3_eval (the runner imports it as F3).
Streams (research_v2/mid.py "MID A (OG 1 MNQ)" + guard.py):
  og     OG main-stream signals (bot's own ensemble) taken ONLY when A+ (15m/30m/1h EMA20>EMA50 all with the trade,
         1h swing structure with the trade, > 2 ATR5 room to the next level) at 1 MNQ, max 10/day, OLD bot lock (LOCK_* env).
  v2     v2 breakout-retest + level/VWAP pullbacks (no pb_ema), 1 MNQ, +$400 -> +$50 lock.
  addon  ONL / E1 (shares the one-at-a-time v2 slot), 1 MNQ, OLD lock, 2 h time exit.
  6pm    6 PM long only when balance - FLOOR >= $3,000 cushion, 1 MNQ, +$400 -> +$50 lock, out 09:26 ET.
Guard: realized day P&L <= -$500 -> no new entries today; DD from realized high >= $1,000 -> half size (1 MNQ: og paused, others 1);
DD >= $1,500 -> no trades for the rest of today + next 2 trading days, then resume at half with the guard high re-based to
equity + $1,000; full size again when DD < $500. Flat 16:44 ET. Account must be in FUNDED_ACCOUNT_ALLOWLIST (empty -> refuse all)."""
import json, os
from dataclasses import dataclass, asdict, field
from datetime import time, timedelta
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
ET = ZoneInfo("America/New_York")
def _f(n, d): return float(os.getenv(n, d))
def _t(n, d): h, m = os.getenv(n, d).split(":"); return time(int(h), int(m))
ACCOUNT_ALLOWLIST = [a.strip() for a in os.getenv("FUNDED_ACCOUNT_ALLOWLIST", "LFF050-9XP156CR-PRO005").split(",") if a.strip()]
ACCOUNT_ALLOWED = ACCOUNT_ALLOWLIST[0] if ACCOUNT_ALLOWLIST else ""        # paper-mode stand-in only
SIZE_MNQ = int(os.getenv("FUNDED_SIZE", "1"))
DAILY_LOSS_STOP = _f("FUNDED_DAILY_LOSS_STOP", "500")
HALF_DD = _f("FUNDED_HALF_DD", "1000"); PAUSE_DD = _f("FUNDED_PAUSE_DD", "1500"); FULL_DD = _f("FUNDED_FULL_DD", "500")
PAUSE_DAYS = int(os.getenv("FUNDED_PAUSE_DAYS", "2"))
FLOOR = _f("FUNDED_FLOOR", "50100"); START_BALANCE = _f("FUNDED_START_BALANCE", "52200")   # balance when guard state was created
SIX_PM_CUSHION = _f("FUNDED_6PM_CUSHION", "0")   # audit: cushion gate cut 2026 edge; 6pm alone 5.3% bust
OG_MAX_PER_DAY = int(os.getenv("FUNDED_OG_MAX_PER_DAY", "10")); OG_MAIN_ON = os.getenv("FUNDED_OG_ON", "0") == "1"   # audit 2026-10-24: OG off (bust 16.6% -> 10.9%)
FLAT_TIME = _t("FUNDED_FLAT_ET", "16:44"); REOPEN_TIME = _t("FUNDED_REOPEN_ET", "18:00")
V2_LOCK = dict(activate=_f("FUNDED_V2_LOCK_ACT", "400"), keep=_f("FUNDED_V2_LOCK_KEEP", "50"), trail_start=1e12, keep_pct=0.0)
COST_PER_CONTRACT = _f("FUNDED_COST_PER_CONTRACT", "4"); SIX_PM_FAR_TP_PTS = _f("FUNDED_SIX_PM_FAR_TP_PTS", "400")
DAY_PROFIT_CAP = float("inf"); CAP_FLATTEN = False
# sources that lost in the TUNING period (< 2026, live-causal streams); decided on tune data only, 2026 = out-of-sample
EXCLUDE_SRC = tuple(x for x in os.getenv("FUNDED_EXCLUDE_SRC", "pb_ema,pb_pdc,pb_pwh,pb_vwap,pb_zone,E1").split(",") if x)
PLANNER_KW = dict(exclude_src=EXCLUDE_SRC, shared_busy=True, v2_sess_stops=False, max_per_day=None)
STREAMS = ("og", "v2", "addon", "6pm")
def account_ok(acct): return bool(acct) and acct in ACCOUNT_ALLOWLIST
def classify_stream(signal):
    s = str(signal.get("flex3_stream") or "").lower()
    return s if s in STREAMS else "og"           # anything not from the planner is an OG main-stream signal
def lock_params(stream, size):
    if stream not in ("v2", "6pm"): return {}    # og / addon: bot's old lock (LOCK_* env)
    n = max(1, int(size)); p = V2_LOCK
    return dict(activate_usd=p["activate"]*n, keep_usd=p["keep"]*n, trail_start_usd=p["trail_start"]*n, keep_pct=p["keep_pct"])
def trading_day(now_et):
    d = now_et + timedelta(days=1) if now_et.time() >= REOPEN_TIME else now_et
    return d.date().isoformat()
def in_flat_window(t): return FLAT_TIME <= t < REOPEN_TIME

# ---- OG A+ classifier (causal port of research_v2/mine.py features) ----
def _trend_struct(d1, tf, T):
    from src.strategy._flex3_levels import rs, ema
    x = rs(d1, tf); x = x[x.index <= T]                        # index = bar CLOSE time -> closed bars only
    if len(x) < 60: return np.nan, np.nan
    tr = float(np.sign(ema(x.close, 20).iloc[-1] - ema(x.close, 50).iloc[-1])); k = 3
    H, L = x.high.to_numpy(), x.low.to_numpy(); ph, pl = [], []
    for i in range(k, len(x) - k):
        if H[i] == H[i-k:i+k+1].max(): ph.append(H[i])
        if L[i] == L[i-k:i+k+1].min(): pl.append(L[i])
    if len(ph) < 2 or len(pl) < 2: return tr, 0.0
    hh, hl = ph[-1] > ph[-2], pl[-1] > pl[-2]
    return tr, (1.0 if hh and hl else (-1.0 if (not hh and not hl) else 0.0))
def og_quality(d1, T, entry, direction):
    """d1 closed 1m bars (UTC, bar-open index); T entry time (UTC). Returns dict(htf_align, struct1h, room_atr, Aplus, worst)."""
    from src.strategy._flex3_levels import build_features, levels_at
    T = pd.Timestamp(T); d = 1 if str(direction).lower() in ("long", "1", "buy") else -1
    al = 0.0; st = 0.0
    for tf in ("15min", "30min", "1h"):
        tr, s = _trend_struct(d1, tf, T); al += tr * d
        if tf == "1h": st = s * d
    f = build_features(d1[d1.index < T], causal=True); f = f[f.index <= T]
    row = f.iloc[-1]; a = row.atr; lv = [x for x, _ in levels_at(row) if x == x]
    ah = [v for v in ((x - entry) * d / a for x in lv) if v > 0]
    room = float(np.clip(min(ah) if ah else 9, 0, 9))
    return dict(htf_align=al, struct1h=st, room_atr=room, Aplus=bool(al == 3 and st == 1 and room > 2),
                worst=bool((st == -1 or al == -1) and room <= 1))

@dataclass
class Flex3Guard:          # name kept so the runner's hooks are shared with PASS EVAL 3K
    eq: float = 0.0; hwm: float = 0.0; ghwm: float = 0.0; day_pnl: float = 0.0; day_key: str = ""
    half: bool = False; pause_left: int = -1; unknown_exits: int = 0; og_today: int = 0
    days: list = field(default_factory=list)
    eval_pnl: float = 0.0; eval_hwm: float = 0.0; best_day: float = 0.0     # printed by the runner
    def new_day(self, key):
        if key == self.day_key: return
        if self.day_key: self.days = (self.days + [self.day_key])[-30:]
        self.day_key = key; self.day_pnl = 0.0; self.og_today = 0
        if self.pause_left >= 0:
            self.pause_left -= 1
            if self.pause_left < 0:                                            # resume: half size, guard HWM re-based
                self.ghwm = min(self.ghwm, self.eq + HALF_DD); self.half = True
    def on_close(self, pnl, day_key=None):
        if day_key: self.new_day(day_key)
        self.eq += pnl; self.day_pnl += pnl; self.hwm = max(self.hwm, self.eq); self.ghwm = max(self.ghwm, self.eq)
        self.eval_pnl, self.eval_hwm = self.eq, self.hwm
    def on_unknown_exit(self): self.unknown_exits += 1
    def passed(self): return False
    def balance(self): return START_BALANCE + self.eq
    def dd(self): return self.ghwm - self.eq
    def _update_mode(self):
        if self.pause_left < 0 and self.dd() >= PAUSE_DD: self.pause_left = PAUSE_DAYS
        if self.dd() >= HALF_DD: self.half = True
        if self.dd() < FULL_DD: self.half = False
    def allow_entry(self, stream, now_et_time, account, quality=None, **_):
        if not account_ok(account): return False, f"account {account!r} not in FUNDED_ACCOUNT_ALLOWLIST {ACCOUNT_ALLOWLIST}"
        if self.unknown_exits: return False, "unknown-pnl broker exit: reconcile state json then reset unknown_exits"
        if in_flat_window(now_et_time): return False, "flat window"
        self._update_mode()
        if self.pause_left >= 0: return False, f"2-day pause (DD ${self.dd():.0f} >= ${PAUSE_DD:.0f})"
        if self.day_pnl <= -DAILY_LOSS_STOP: return False, "daily loss stop"
        if stream == "og":
            if self.size("og") < 1: return False, "half size: OG stream paused"
            if self.og_today >= OG_MAX_PER_DAY: return False, "OG max/day"
            if quality is None or not quality.get("Aplus"): return False, f"OG not A+ {quality}"
        if stream == "6pm":
            if START_BALANCE <= 0: return False, "6pm: FUNDED_START_BALANCE not set"
            if self.balance() - FLOOR < SIX_PM_CUSHION: return False, f"6pm cushion ${self.balance() - FLOOR:.0f} < ${SIX_PM_CUSHION:.0f}"
        return True, "ok"
    def on_entry(self, stream):
        if stream == "og": self.og_today += 1
    def cap_reached(self, open_pnl): return False
    def size(self, stream):
        if not self.half: return SIZE_MNQ
        k = SIZE_MNQ // 2
        return k if k >= 1 else (0 if stream == "og" else 1)
    def save(self, path):
        tmp = path + ".tmp"; open(tmp, "w").write(json.dumps(asdict(self), indent=1)); os.replace(tmp, path)
    @classmethod
    def load(cls, path):
        try: return cls(**{k: v for k, v in json.load(open(path)).items() if k in cls.__dataclass_fields__})
        except Exception: return cls()
