#!/usr/bin/env python
"""
LIVE TRADING - Full AI Ensemble via Rithmic (Tradesea/Lucid)

Strategy: Sweep-Gated Entry + ML Confirmations (PF 1.16+ validated)
- IntelligentTrader: ML classifiers (Gradient Boost, SVC, Neural Net)
- AdvancedStrategies: 8 strategies (FibMACD, StochRSI, HeikinAshi, etc)
- Dynamic SL/TP: Swing-based stops with trailing
- EMA/Technical confirmation layer

Requirements:
1. Rithmic credentials in .env (already configured for LucidTrading)
2. Active Tradesea/Lucid account with trading permissions

Usage:
    python start_live_rithmic.py                       # MES default
    python start_live_rithmic.py --symbol MNQ          # Trade MNQ
    python start_live_rithmic.py --symbols MES MNQ     # Scan both, trade best setup
    python start_live_rithmic.py --paper               # Paper mode (signals only)
"""

import os
import sys
import time
import re
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.broker.rithmic_connector import RithmicConnector
from src.broker.rithmic_stability import filter_warm_history
from src.broker.rithmic_stability import filter_warm_history_market, stitch_history
from src.core.ensemble_trader import EnsembleTrader
from src.ai.technical_analyzer import TechnicalAnalyzer
from src.utils.logger import bot_logger, trades_logger
from src.core.conf_display import conf_pct, ordinal, trade_status_line, extra_trade_line
# 2nd/3rd trade rule (2026-10-09): same direction + open trade(s) in profit (EXTRA_TRADE_MODE).
from src.core import extra_rule as _extra_rule
# Day-session / overnight loss limits + daily profit target (2026-10-09, REVAMP default on).
from src.core import session_limits as _session_limits
# REVAMP overnight sleeves (2026-10-10, box ovn_fix.py 'trend only'): outside 9:30-16:00 ET the revamp
# takes only trend-sleeve entries (first, extra). REVAMP_OVERNIGHT_SLEEVES=trend (default) | all (old).
_OVN_SLEEVE_LABEL = {'breakout': 'breakout', 'mean_reversion': 'mean-reversion', 'trend': 'trend'}


def revamp_overnight_sleeves() -> str:
    raw = str(os.getenv('REVAMP_OVERNIGHT_SLEEVES', 'trend') or 'trend').strip().lower()
    return 'all' if raw in ('all', 'any', 'off', '0', 'old') else 'trend'


def revamp_overnight_block(sleeve: str, now: Optional[datetime] = None) -> Optional[str]:
    """Reason a REVAMP sleeve entry is skipped overnight, else None. Session is decided by the
    entry time (same rule as session_limits.session_of / box ovn_build.sess). Day session: never."""
    if revamp_overnight_sleeves() == 'all':
        return None
    kind, _td = _session_limits.session_of(now or datetime.now(timezone.utc))
    if kind != 'overnight' or str(sleeve or '') == 'trend':
        return None
    return f"overnight: trend trades only ({_OVN_SLEEVE_LABEL.get(sleeve, sleeve or 'unknown')} skipped)"
from src.core.size_scale import (
    load_size_scale_config,
    compute_sized_contracts,
    format_size_log,
)
from config.strategy_config import SCALPING_PAIRS

# ── flex3 LucidFlex eval gate (FLEX3_ON=1). SIM account lock + daily rules. ──
# EVAL PASSER (internal name flex3): EVAL_PASSER_ON=1 or FLEX3_ON=1. Trades ONLY v2 / ONL+E1 / 6 PM long; OG main stream off.
FLEX3_ON = any(os.getenv(k, '0').strip().lower() in ('1', 'true', 'yes', 'on') for k in ('EVAL_PASSER_ON', 'FLEX3_ON'))
# FUNDED 12K: FUNDED_12K_ON=1 reuses the same hooks with src/strategy/funded12k.py as the rule module (OG main stream stays ON, A+ only).
FUNDED_12K_ON = os.getenv('FUNDED_12K_ON', '0').strip().lower() in ('1', 'true', 'yes', 'on')
if FUNDED_12K_ON and FLEX3_ON:
    raise SystemExit('EVAL_PASSER_ON and FUNDED_12K_ON are mutually exclusive')
FLEX3_ON = FLEX3_ON or FUNDED_12K_ON
FLEX3_1M_CSV = os.getenv('FLEX3_1M_CSV', os.getenv('NEAT_BAR_EXPORT_PATH', 'data/MNQ_1m.csv'))
FLEX3_BAR_LABEL = os.getenv('FLEX3_BAR_LABEL', 'open').strip().lower()   # 'close' if the CSV stamps bars at their close
from src.strategy import flex3_events as F3EV   # opt-in JSONL for live chart (FLEX3_EVENT_LOG)
FLEX3_STATUS = os.getenv('FLEX3_STATUS', '0') == '1'   # read-only bias/setup readout after each closed 5m bar (src/strategy/flex3_status.py)
OG_STATUS = os.getenv('OG_STATUS', '0') == '1'   # read-only OG bias/setup/win-rate readout (src/strategy/og_status.py)
FLEX3_STATE_FILE = os.getenv('FLEX3_STATE_FILE', 'logs/funded12k_state.json' if FUNDED_12K_ON else 'logs/flex3_state.json')
if FUNDED_12K_ON:
    from src.strategy import funded12k as F3
    from src.strategy import flex3_signals as F3S
elif FLEX3_ON:
    from src.strategy import flex3_eval as F3
    from src.strategy import flex3_signals as F3S


def _load_risk_overrides() -> Dict[str, Any]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'risk_overrides.json')
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


RISK_OVERRIDES = _load_risk_overrides()

# Strategy parameters (validated profitable)
PARAMS = {
    'lookback': 10,
    'atr_mult': 1.5,
    'tp_mult': 2.0,
    'ema_len': 50
}

TRAIL_SETTINGS = {
    # Start trailing once price has moved this fraction of TP distance.
    'activation_tp_fraction': float(os.getenv('TRAIL_ACTIVATION_TP_FRACTION', '0.35')),
    # Alternative activation based on initial risk (R-multiple).
    'activation_r': float(os.getenv('TRAIL_ACTIVATION_R', '0.30')),
    # Trail stop behind favorable move using ATR.
    'atr_mult': float(os.getenv('TRAIL_ATR_MULT', '0.6')),
    # Minimum SL movement required before submitting a broker update.
    'min_step_ticks': int(os.getenv('TRAIL_MIN_STEP_TICKS', '1')),
}

# Dollar profit locks (MNQ $2/pt: +$100 uPnL = 50 pts → SL locks +$20 = 10 pts)
PROFIT_LOCK_ENABLED = os.getenv('PROFIT_LOCK_ENABLED', 'true').strip().lower() in (
    '1', 'true', 'yes', 'on',
)
PROFIT_LOCK_TRIGGER_1_USD = float(os.getenv('PROFIT_LOCK_TRIGGER_1_USD', '100'))
PROFIT_LOCK_LEVEL_1_USD = float(os.getenv('PROFIT_LOCK_LEVEL_1_USD', '20'))
PROFIT_LOCK_TRIGGER_2_USD = float(os.getenv('PROFIT_LOCK_TRIGGER_2_USD', '150'))
PROFIT_LOCK_LEVEL_2_USD = float(os.getenv('PROFIT_LOCK_LEVEL_2_USD', '50'))

# R-based profit locks (replace dollar locks). R = initial stop distance (pts).
#   +LOCK_R1 R  -> SL to entry +LOCK_R1_KEEP R
#   +LOCK_R2 R  -> SL to entry +LOCK_R2_KEEP R
#   beyond      -> SL keeps TRAIL_KEEP_PCT of max favorable excursion
LOCK_R1 = float(os.getenv('LOCK_R1', '0.5'))
LOCK_R1_KEEP = float(os.getenv('LOCK_R1_KEEP', '0.1'))
LOCK_R2 = float(os.getenv('LOCK_R2', '1.0'))
LOCK_R2_KEEP = float(os.getenv('LOCK_R2_KEEP', '0.5'))
TRAIL_KEEP_PCT = float(os.getenv('TRAIL_KEEP_PCT', '0.5'))
SL_MOVE_MIN_BUFFER_TICKS = int(os.getenv('SL_MOVE_MIN_BUFFER_TICKS', '4'))

# Dollar profit lock (user 2026-10-07). Whole-trade $ (all contracts):
#   no stop move until open profit reaches LOCK_ACTIVATE_USD
#   then SL locks LOCK_KEEP_USD
#   from LOCK_TRAIL_START_USD peak profit: SL keeps LOCK_TRAIL_KEEP_PCT of peak
#   (never below the lock, tighten-only, only while in profit)
# LOCK_MODE=usd (default) | r (old R-lock: LOCK_R1/LOCK_R2/TRAIL_KEEP_PCT)
LOCK_MODE = os.getenv('LOCK_MODE', 'usd').strip().lower()
LOCK_ACTIVATE_USD = float(os.getenv('LOCK_ACTIVATE_USD', '100'))
LOCK_KEEP_USD = float(os.getenv('LOCK_KEEP_USD', '20'))
LOCK_TRAIL_START_USD = float(os.getenv('LOCK_TRAIL_START_USD', '150'))
LOCK_TRAIL_KEEP_PCT = float(os.getenv('LOCK_TRAIL_KEEP_PCT', '0.5'))

# Per-trade profit lock (user 2026-10-22): once a trade's open profit reaches
# PROFIT_LOCK_TRIGGER dollars (all contracts), move its SL to lock PROFIT_LOCK_AMOUNT
# dollars. Once per trade, tighten-only, TP untouched. Either set to 0 = off.
def _pl_env(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, default)).strip() or default)
    except (TypeError, ValueError):
        return float(default)


PROFIT_LOCK_TRIGGER = _pl_env('PROFIT_LOCK_TRIGGER', 400.0)
PROFIT_LOCK_AMOUNT = _pl_env('PROFIT_LOCK_AMOUNT', 50.0)


def fixed_profit_lock_stop(direction: str, entry: float, size: int, point_value: float,
                           peak_pts: float, live: float, tick: float, cur_sl: float,
                           trigger_usd: float = None, lock_usd: float = None,
                           buffer_ticks: int = None):
    """Pure per-trade lock rule. Returns (status, price).

    status: 'off' | 'wait' (not triggered / stop not placeable yet) |
            'tighter' (current SL already at/above lock -> done) | 'move' (price = new SL)
    """
    import math
    trigger_usd = PROFIT_LOCK_TRIGGER if trigger_usd is None else float(trigger_usd)
    lock_usd = PROFIT_LOCK_AMOUNT if lock_usd is None else float(lock_usd)
    buffer_ticks = SL_MOVE_MIN_BUFFER_TICKS if buffer_ticks is None else buffer_ticks
    if trigger_usd <= 0 or lock_usd <= 0:
        return 'off', None
    if entry <= 0 or live <= 0 or tick <= 0 or point_value <= 0:
        return 'wait', None
    usd_per_pt = float(point_value) * max(1, int(size or 1))
    is_long = direction == 'long'
    live_pts = (live - entry) if is_long else (entry - live)
    peak = max(float(peak_pts or 0), live_pts)
    if peak * usd_per_pt + 1e-9 < trigger_usd:
        return 'wait', None
    # Round AWAY from entry so the lock is never less than lock_usd.
    lock_pts = math.ceil((lock_usd / usd_per_pt) / tick - 1e-9) * tick
    new_sl = round(entry + lock_pts if is_long else entry - lock_pts, 10)
    if cur_sl > 0:
        if is_long and cur_sl >= new_sl - 1e-9:
            return 'tighter', cur_sl
        if (not is_long) and cur_sl <= new_sl + 1e-9:
            return 'tighter', cur_sl
    buf = tick * max(1, int(buffer_ticks))
    if is_long and new_sl > live - buf + 1e-9:
        return 'wait', None  # price fell back under the lock: can't rest a stop there
    if (not is_long) and new_sl < live + buf - 1e-9:
        return 'wait', None
    return 'move', new_sl


# Bars to sit out after a trade closes (was a fixed 5 = 25 min on 5m bars).
COOLDOWN_BARS_AFTER_WIN = max(0, int(os.getenv('COOLDOWN_BARS_AFTER_WIN', '1')))
COOLDOWN_BARS_AFTER_LOSS = max(0, int(os.getenv('COOLDOWN_BARS_AFTER_LOSS', '2')))

# Adopt broker size the bot is not tracking (restart / lost tracking): rebuild
# a tracked trade and make sure it has a stop. Untracked size must be seen on
# two checks >= ADOPT_CONFIRM_SEC apart (Lucid list_positions can lag/flicker).
# 2026-10-11b: ADOPT_MANUAL=0 turns manual-trade takeover OFF completely (no SL/TP placed for
# your manual trades). ADOPT_UNTRACKED is the old name and still works.
ADOPT_UNTRACKED = os.getenv('ADOPT_MANUAL', os.getenv('ADOPT_UNTRACKED', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
ADOPT_CONFIRM_SEC = float(os.getenv('ADOPT_CONFIRM_SEC', '20'))
ADOPT_STOP_ATR_MULT = float(os.getenv('ADOPT_STOP_ATR_MULT', '1.5'))
ADOPT_DEFAULT_STOP_PTS = float(os.getenv('ADOPT_DEFAULT_STOP_PTS', '40'))
# 2026-10-11: adopted trades also get a TP LIMIT. ADOPT_TP_R = target in R (stop distance)
# when the setup is unknown (default = revamp trend TP 1.5R). A logged bot ENTRY within
# ADOPT_PLANNED_TP_MATCH_PTS of the broker avg reuses that trade's own planned TP.
ADOPT_PLANNED_TP_MATCH_PTS = float(os.getenv('ADOPT_PLANNED_TP_MATCH_PTS', '2.0'))
# Per-trade risk cap (2026-10-08): stop risk = |entry - SL| x $/pt x contracts.
# Trim contracts to fit; if even 1 contract exceeds the cap, skip the entry.
# MAX_RISK_PER_TRADE_USD=0 turns the guard off.
def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, default)).strip() or default)
    except (TypeError, ValueError):
        return float(default)


MAX_RISK_PER_TRADE_USD = _env_float('MAX_RISK_PER_TRADE_USD', 500.0)


def cap_contracts_by_risk(entry, sl, point_value, contracts, max_risk_usd):
    """Return (allowed_contracts, risk_per_contract_usd). 0 contracts = skip."""
    contracts = max(0, int(contracts or 0))
    try:
        rpc = abs(float(entry) - float(sl)) * float(point_value)
    except (TypeError, ValueError):
        return 0, float('inf')
    if rpc != rpc or rpc == float('inf'):
        return 0, float('inf')
    try:
        cap = float(max_risk_usd or 0)
    except (TypeError, ValueError):
        cap = 0.0
    if cap <= 0 or rpc <= 0:
        return contracts, rpc
    if rpc * contracts <= cap + 1e-9:
        return contracts, rpc
    return max(0, min(contracts, int((cap + 1e-9) // rpc))), rpc


def usd_lock_stop(direction: str, entry: float, size: int, point_value: float,
                  peak_pts: float, live: float, tick: float, cur_sl: float,
                  activate_usd: float = None, keep_usd: float = None,
                  trail_start_usd: float = None, keep_pct: float = None,
                  buffer_ticks: int = None):
    """Pure dollar-lock rule. Returns new SL price or None (keep current stop)."""
    activate_usd = LOCK_ACTIVATE_USD if activate_usd is None else activate_usd
    keep_usd = LOCK_KEEP_USD if keep_usd is None else keep_usd
    trail_start_usd = LOCK_TRAIL_START_USD if trail_start_usd is None else trail_start_usd
    keep_pct = LOCK_TRAIL_KEEP_PCT if keep_pct is None else keep_pct
    buffer_ticks = SL_MOVE_MIN_BUFFER_TICKS if buffer_ticks is None else buffer_ticks
    if entry <= 0 or live <= 0 or tick <= 0 or point_value <= 0:
        return None
    usd_per_pt = float(point_value) * max(1, int(size or 1))
    is_long = direction == 'long'
    live_pts = (live - entry) if is_long else (entry - live)
    if live_pts <= 0:
        return None  # flat or losing: never move
    peak = max(float(peak_pts or 0), live_pts)
    peak_usd = peak * usd_per_pt
    if peak_usd + 1e-9 < activate_usd:
        return None  # not up enough yet
    lock_usd = float(keep_usd)
    if peak_usd + 1e-9 >= trail_start_usd:
        lock_usd = max(lock_usd, float(keep_pct) * peak_usd)
    import math
    keep_pts = math.floor((lock_usd / usd_per_pt) / tick + 1e-9) * tick
    if keep_pts <= 0:
        return None
    new_sl = entry + keep_pts if is_long else entry - keep_pts
    buf = tick * max(1, int(buffer_ticks))
    if is_long and new_sl > live - buf + 1e-9:
        return None
    if (not is_long) and new_sl < live + buf - 1e-9:
        return None
    if cur_sl > 0:
        if is_long and new_sl <= cur_sl + 1e-9:
            return None
        if (not is_long) and new_sl >= cur_sl - 1e-9:
            return None
    return new_sl


def r_lock_stop(direction: str, entry: float, risk_pts: float, mfe_pts: float,
                live: float, tick: float, cur_sl: float,
                r1: float = None, r1_keep: float = None, r2: float = None,
                r2_keep: float = None, keep_pct: float = None,
                buffer_ticks: int = None):
    """Pure R-lock rule. Returns new SL price or None (keep current stop).

    Only when live price is in profit; only profit-locking; tighten-only;
    must sit >= buffer ticks on the correct side of the live price.
    """
    r1 = LOCK_R1 if r1 is None else r1
    r1_keep = LOCK_R1_KEEP if r1_keep is None else r1_keep
    r2 = LOCK_R2 if r2 is None else r2
    r2_keep = LOCK_R2_KEEP if r2_keep is None else r2_keep
    keep_pct = TRAIL_KEEP_PCT if keep_pct is None else keep_pct
    buffer_ticks = SL_MOVE_MIN_BUFFER_TICKS if buffer_ticks is None else buffer_ticks
    if risk_pts <= 0 or entry <= 0 or live <= 0 or tick <= 0:
        return None
    is_long = direction == 'long'
    live_pts = (live - entry) if is_long else (entry - live)
    if live_pts <= 0:
        return None  # flat or losing: never move
    mfe = max(float(mfe_pts or 0), live_pts)
    mfe_r = mfe / risk_pts
    if mfe_r < r1:
        return None
    keep_pts = r1_keep * risk_pts
    if mfe_r >= r2:
        keep_pts = max(r2_keep * risk_pts, keep_pct * mfe)
    import math
    steps = keep_pts / tick
    keep_pts = (math.floor(steps + 1e-9)) * tick  # round toward entry (conservative)
    if keep_pts <= 0:
        return None
    new_sl = entry + keep_pts if is_long else entry - keep_pts
    buf = tick * max(1, int(buffer_ticks))
    if is_long and new_sl > live - buf + 1e-9:
        return None
    if (not is_long) and new_sl < live + buf - 1e-9:
        return None
    if cur_sl > 0:
        if is_long and new_sl <= cur_sl + 1e-9:
            return None
        if (not is_long) and new_sl >= cur_sl - 1e-9:
            return None
    return new_sl

# ---------------------------------------------------------------------------
# REVAMP exits (overnight research winner). Default OFF = original live path
# (AI ensemble + TRAIL_ATR_MULT=0.6). Enable only via start_mnq_live_revamp.bat
# or env REVAMP_EXITS=1. Does not change start_mnq_live.bat / ema15_eod.
#   trail ATR mult 4.0 | breakout fixed 2R (no trail) | MR target bb_far
# ---------------------------------------------------------------------------
from src.research_ensemble import revamp_exits as _revamp_exits
from src.research_ensemble.live_signals import classify_last_sleeve
from src.research_ensemble.session import format_ts_et
# Re-entry after a profitable lock/trail stop exit (default OFF; REENTRY_ENABLED=1).
from src.core import reentry as _reentry
# Backtested add-on setups (2026-10-10): ONL reclaim LONG + 6 PM momentum LONG, flat-only, 1 MNQ.
# ADDON_ONL_RECLAIM / ADDON_E1_6PM (default 1; 0 = off). Rules: docs/ADDON_SPECS_20261010.md
from src.core import addon_setups as _addons
try:  # display-only readout; a failure here must never stop the bot
    from src.research_ensemble import live_display as _revamp_display
except Exception as _revamp_display_import_err:  # pragma: no cover
    _revamp_display = None

REVAMP_EXITS_ON = _revamp_exits.revamp_enabled()
if REVAMP_EXITS_ON:
    TRAIL_SETTINGS['atr_mult'] = float(_revamp_exits.revamp_trail_mult())
REVAMP_CANDLE_COUNT = int(os.getenv('REVAMP_CANDLE_COUNT', '320'))
# DISPLAY-ONLY: per-bar bias + decision readout in revamp mode.
# REVAMP_VERBOSE unset/1 = compact (now X | needs Y), 2 = detailed 6 lines, 0 = off.
REVAMP_VERBOSE_LEVEL = _revamp_display.revamp_verbose_level() if _revamp_display is not None else 0
REVAMP_VERBOSE = REVAMP_VERBOSE_LEVEL > 0

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == '':
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


# Concurrent open trades: hard business cap is 2. Prefer MAX_CONCURRENT_TRADES,
# then MNQ_MAX_POSITIONS, never above 3 unless explicitly raised later.
_HARD_MAX_CONCURRENT = 3  # user 2026-10-07: up to 3, extras need confidence
ENV_MAX_CONCURRENT_TRADES = max(
    1,
    min(
        _HARD_MAX_CONCURRENT,
        _env_int('MAX_CONCURRENT_TRADES', _env_int('MNQ_MAX_POSITIONS', _HARD_MAX_CONCURRENT)),
    ),
)
INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', '50000'))
DAILY_LOSS_LIMIT_PCT = float(os.getenv('DAILY_LOSS_LIMIT_PERCENT', '3'))
LOSS_COOLDOWN_MINUTES = int(os.getenv('LOSS_COOLDOWN_MINUTES', '10'))
ENSEMBLE_MAX_CONSEC_LOSSES = int(os.getenv('ENSEMBLE_MAX_CONSEC_LOSSES', '3'))
ENSEMBLE_CONSEC_LOSS_PAUSE_MIN = int(os.getenv('ENSEMBLE_CONSEC_LOSS_PAUSE_MINUTES', '30'))
GHOST_FLAT_GRACE_SEC = float(os.getenv('GHOST_FLAT_GRACE_SEC', '30'))
# Orphan working-order sweep (leftover sell/buy limits + stops on the Lucid chart).
ORPHAN_SWEEP_SEC = float(os.getenv('ORPHAN_SWEEP_SEC', '20'))
# Market-flatten open bot positions on shutdown (default off: keep SL/TP on open trades).
SHUTDOWN_FLATTEN = os.getenv('SHUTDOWN_FLATTEN', '0').strip().lower() in ('1', 'true', 'yes', 'on')

# Risk settings
RISK_SETTINGS = {
    'contracts': 1,              # Start with 1 micro
    'daily_loss_limit': INITIAL_BALANCE * (DAILY_LOSS_LIMIT_PCT / 100),
    # 0 = unlimited (user 2026-10-07: no daily trade-count cap; loss limit still applies)
    'max_trades_per_day': _env_int('MAX_TRADES_PER_DAY', 0),
    'cooldown_bars': COOLDOWN_BARS_AFTER_LOSS,  # legacy; process_exit uses win/loss split
    'max_positions': ENV_MAX_CONCURRENT_TRADES,
}

# Confidence lot sizing (pure confidence bands by default; see src/core/size_scale.py)
def _revamp_confidence(hit: dict, sleeve: str) -> float:
    """Revamp signals used a hard-coded 0.80 (always 3 lots). Derive a real score.

    Uses classifier 'confidence' if present; else trend/breakout scale with ADX
    (ADX 20 -> 0.62, 30 -> 0.72, 36 -> 0.78, 43+ -> 0.85+); mean-reversion 0.66
    (1 lot, never opens an extra concurrent slot). Clamped 0.50..0.95.
    """
    try:
        c = hit.get("confidence")
        if c is not None:
            return max(0.5, min(0.95, float(c)))
    except (TypeError, ValueError):
        pass
    try:
        adx = float(hit.get("adx"))
    except (TypeError, ValueError):
        adx = float("nan")
    if sleeve == "mean_reversion" or not np.isfinite(adx):
        return float(os.getenv("REVAMP_MR_CONFIDENCE", "0.66"))
    return max(0.5, min(0.95, 0.62 + (adx - 20.0) * 0.01))


SIZE_SCALE_CFG = load_size_scale_config()

# Symbol specs
# Round-trip commission+fees per contract by symbol (NQ only, env COMMISSION_RT_NQ).
# MNQ/MES keep the legacy flat $2 + 1 tick (unchanged accounting).
COMMISSION_RT_NQ = float(os.getenv('COMMISSION_RT_NQ', '4.50'))


def round_trip_cost(symbol: str, size: int, point_value: float, tick_size: float) -> float:
    """Costs subtracted from a closed trade's P&L.

    NQ: (commission RT + 1 tick slippage) per contract.
    Others (MNQ/MES): legacy flat $2 commission + 1 tick slippage.
    """
    tick_usd = float(tick_size) * float(point_value)
    if str(symbol or '').upper() == 'NQ':
        return (COMMISSION_RT_NQ + tick_usd) * max(1, int(size or 1))
    return 2.00 + tick_usd


SYMBOL_SPECS = {
    'MES': {'point_value': 5.0, 'tick_size': 0.25},
    'MNQ': {'point_value': 2.0, 'tick_size': 0.25},
    'NQ':  {'point_value': 20.0, 'tick_size': 0.25},
}


@dataclass
class Position:
    order_id: str
    symbol: str
    direction: str
    entry_price: float
    size: int
    sl: float
    tp: float
    entry_time: datetime
    trailing_enabled: bool = True
    profit_lock_level: int = 0  # 0=none, 1=+$20 after +$100, 2=+$50 after +$150
    tp_cleared: bool = False  # True after price past TP: broker TP cancelled, trail-only
    original_tp: float = 0.0  # TP price at clear-time; trail SL floored here
    ensemble_signal: str = 'UNKNOWN'
    model_signals: Optional[Dict] = None
    regime: str = 'unknown'
    exit_sleeve: str = ''  # trend | breakout | mean_reversion (REVAMP path)
    initial_risk: float = 0.0  # R in points (entry -> first stop)
    mfe_pts: float = 0.0  # max favorable excursion in points
    peak_pts: float = 0.0  # best open profit (pts) seen AFTER entry (usd lock)
    fixed_lock_done: bool = False  # per-trade +$PROFIT_LOCK_AMOUNT lock applied (2026-10-22)
    confidence: float = 0.0  # entry confidence (a re-entry inherits its parent's)
    reentry_gen: int = 0  # 0 = normal entry, 1 = first re-entry, ...
    reentry_parent: str = ''  # parent order id when this trade is a re-entry
    addon: str = ''  # 'onl' | 'e1' backtested add-on (fixed SL/TP, no locks/trail, 2h time exit)
    addon_exit_at: Optional[datetime] = None  # UTC time limit for an add-on trade
    flex3_stream: str = ''  # v2 | addon | 6pm | ogtrend (EVAL PASSER / FUNDED 12K)
    flex3_exit_at: Optional[Any] = None  # tz-aware time exit (ONL/E1 +2h, 6 PM 09:26 ET)


class LiveRithmicTrader:
    """Live trading with Enhanced Breakout Strategy via Rithmic."""
    
    def __init__(self, symbol: str = 'MNQ', symbols: Optional[List[str]] = None, paper_mode: bool = False, skip_confirm: bool = False, paper_orders: bool = False):
        self.skip_confirm = skip_confirm
        self.symbols = symbols or [symbol]
        self.symbol = self.symbols[0]
        self.paper_mode = paper_mode
        self.paper_orders = paper_orders  # Live Rithmic data + simulated orders
        
        for sym in self.symbols:
            if sym not in SYMBOL_SPECS:
                raise ValueError(f"Unsupported symbol: {sym}")
        
        # Strategy params
        self.lookback = PARAMS['lookback']
        self.atr_mult = PARAMS['atr_mult']
        self.tp_mult = PARAMS['tp_mult']
        self.ema_len = PARAMS['ema_len']
        
        # Session filters from risk overrides / per-symbol config
        self.skip_friday = RISK_OVERRIDES.get('skip_friday', False)
        self.bad_hours = RISK_OVERRIDES.get('bad_hours_utc', {'MNQ': [0, 4, 21]})
        
        # Risk management
        self.contracts = RISK_SETTINGS['contracts']
        self.daily_loss_limit = RISK_SETTINGS['daily_loss_limit']
        self.max_trades_per_day = RISK_SETTINGS['max_trades_per_day']
        self.cooldown_bars = RISK_SETTINGS['cooldown_bars']
        self.max_positions = RISK_SETTINGS.get('max_positions', 1)
        
        # State - support multiple positions
        self.positions: Dict[str, Position] = {}  # order_id -> Position
        self.cooldown = 0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_date = None
        self.trades: List[Dict] = []
        self._warmup_log_state: Dict[str, tuple] = {}
        self.symbol_priority = [s.strip() for s in os.getenv('SYMBOL_PRIORITY', 'MNQ').split(',') if s.strip()]
        self.loss_cooldown_until: Dict[str, datetime] = {}
        self._symbol_consec_losses: Dict[str, int] = {sym: 0 for sym in self.symbols}
        self._win_streak: int = 0  # consecutive winning round-trips (day)
        self._size_cfg = dict(SIZE_SCALE_CFG)
        self._consec_loss_pause_until: Dict[str, datetime] = {}
        self._last_orphan_sweep = 0.0
        self._unconfirmed_exit_logged: set = set()
        self._ghost_flat_kept_logged: set = set()
        self._reentry_arms: Dict[str, Any] = {}  # parent order_id -> _reentry.ReentryArm
        # Session limits (day -$500 / overnight -$300 / day +$2,000): block NEW entries only.
        self._session_ledger = None
        self._last_session_block: Optional[str] = None
        if _session_limits.enabled(REVAMP_EXITS_ON):
            self._session_ledger = _session_limits.SessionLedger(
                os.getenv('SESSION_PNL_FILE', _session_limits.DEFAULT_FILE))
        
        # Broker connector
        self.broker: Optional[RithmicConnector] = None
        
        # Full AI ensemble system
        self.ensemble: Optional[EnsembleTrader] = None
        self.technical = TechnicalAnalyzer()
        
        # Log file
        mode = "paper" if paper_mode else "live"
        symbols_tag = "_".join(self.symbols)
        self.log_file = f'logs/{mode}_rithmic_{symbols_tag}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    def _spec(self, symbol: str) -> Dict[str, float]:
        return SYMBOL_SPECS[symbol]

    def _is_trading_session_allowed(self, symbol: str) -> bool:
        """Block known low-edge sessions (Friday, bad UTC hours).

        When overnight/Globex hunting is on (OVERNIGHT_TRADING / SESSION_MODE=extended),
        skip the thin-hour UTC blacklist so entries can fire across the Globex window.
        """
        now = datetime.now(timezone.utc)
        pair_cfg = SCALPING_PAIRS.get(symbol, {})
        skip_friday = pair_cfg.get('skip_friday', self.skip_friday)
        bad_hours = pair_cfg.get('bad_hours_utc', self.bad_hours.get(symbol, []))
        if REVAMP_EXITS_ON and _revamp_exits.overnight_entries_enabled():
            bad_hours = []

        if skip_friday and now.weekday() == 4:
            bot_logger.info(f"🚫 Friday filter — skipping {symbol}")
            return False
        if now.hour in bad_hours:
            bot_logger.info(f"🚫 Bad hour {now.hour} UTC — skipping {symbol}")
            return False
        return True

    def _round_to_tick(self, price: float, tick_size: float) -> float:
        return round(round(price / tick_size) * tick_size, 8)

    def _unrealized_pnl_usd(self, pos: Position, price: float) -> float:
        if price <= 0:
            return 0.0
        pv = float(self._spec(pos.symbol)['point_value'])
        qty = max(1, int(pos.size))
        if pos.direction == 'long':
            return (price - pos.entry_price) * pv * qty
        return (pos.entry_price - price) * pv * qty

    def _lock_sl_price_for_usd(self, pos: Position, lock_usd: float) -> float:
        spec = self._spec(pos.symbol)
        pv = float(spec['point_value'])
        tick = float(spec['tick_size'])
        qty = max(1, int(pos.size))
        pts = float(lock_usd) / (pv * qty)
        if pos.direction == 'long':
            raw = pos.entry_price + pts
        else:
            raw = pos.entry_price - pts
        return self._round_to_tick(raw, tick)

    def _is_tighter_sl(self, pos: Position, new_sl: float, old_sl: float) -> bool:
        if pos.direction == 'long':
            return new_sl > old_sl + 1e-9
        return new_sl < old_sl - 1e-9

    @staticmethod
    def _broker_can_update_stop(info: Optional[Dict[str, Any]]) -> bool:
        """True when SL can be tightened: native bracket modify OR protective cancel/replace.

        Lucid sim often rests protective SL as STOP-LIMIT (not native template-330).
        That path sets supports_stop_modify=False but still updates via update_stop_loss.
        """
        if not info:
            return False
        if info.get('can_update_stop'):
            return True
        if info.get('supports_stop_modify'):
            return True
        if info.get('native_stop_attached'):
            return True
        if info.get('protective_sl_order_id'):
            return True
        if str(info.get('bracket_mode') or '') == 'protective_fallback':
            return True
        return False

    def _apply_stop_update(self, pos: Position, new_sl: float) -> bool:
        """Tighten SL locally and on broker via update_stop_loss / modify_position."""
        if not self._is_tighter_sl(pos, new_sl, pos.sl):
            return False
        if self.paper_mode or self.paper_orders:
            _o = pos.sl; pos.sl = new_sl
            F3EV.emit('sl_move', order_id=pos.order_id, old_sl=_o, sl=new_sl, tp=pos.tp)
            return True
        if not self.broker:
            return False
        ok = False
        if hasattr(self.broker, 'update_stop_loss'):
            side = 'BUY' if pos.direction == 'long' else 'SELL'
            ok = bool(
                self.broker.update_stop_loss(
                    ticket=pos.order_id,
                    symbol=pos.symbol,
                    side=side,
                    size=pos.size,
                    new_sl=new_sl,
                    take_profit=pos.tp,
                )
            )
        if not ok and not hasattr(self.broker, 'update_stop_loss') and hasattr(self.broker, 'modify_position'):
            ok = bool(self.broker.modify_position(ticket=pos.order_id, sl=new_sl, tp=pos.tp))
        if ok:
            _o = pos.sl; pos.sl = new_sl
            F3EV.emit('sl_move', order_id=pos.order_id, old_sl=_o, sl=new_sl, tp=pos.tp)
            # Do NOT clear trailing_enabled here: protective STOP/STOP-LIMIT legs
            # report supports_stop_modify=False but remain trailable via cancel/replace.
            return True
        return False

    def _cancel_position_take_profit(self, pos: 'Position', *, reason: str = 'past_tp_trail') -> bool:
        """Cancel broker TP only; never touch SL. Paper/local always succeeds."""
        if self.paper_mode or self.paper_orders or not self.broker:
            return True
        if hasattr(self.broker, 'cancel_take_profit'):
            try:
                return bool(
                    self.broker.cancel_take_profit(
                        ticket=pos.order_id,
                        symbol=pos.symbol,
                        reason=reason,
                    )
                )
            except Exception as e:
                bot_logger.error(
                    f"cancel_take_profit failed {pos.symbol} {pos.order_id}: {e}"
                )
                return False
        # Fallback: cancel known protective TP tag via entry bracket helper is unsafe
        # (cancels SL too). Refuse rather than drop the stop.
        bot_logger.warning(
            f"Broker lacks cancel_take_profit; cannot clear TP for {pos.order_id}"
        )
        return False

    def update_past_tp_to_trail(self, symbol: str, df: pd.DataFrame):
        """If price trades beyond TP, cancel TP and trail; floor SL at original TP.

        Winners may run past the original target. Local check_position_exit will not
        emit TAKE_PROFIT once tp is cleared. Trail must never loosen SL back through
        the old TP profit level (long: SL >= original_tp; short: SL <= original_tp).
        """
        if not self.positions or df is None or len(df) == 0:
            return
        row = df.iloc[-1]
        bar_high = float(row['high'])
        bar_low = float(row['low'])
        close = float(row.get('close', 0) or 0)

        for order_id, pos in list(self.positions.items()):
            if pos.symbol != symbol or getattr(pos, 'addon', ''):  # add-on: fixed SL/TP
                continue
            if getattr(pos, 'tp_cleared', False):
                continue
            tp = float(pos.tp or 0)
            if tp <= 0:
                continue

            if pos.direction == 'long':
                fav = max(bar_high, close) if close > 0 else bar_high
                past = fav >= tp
            else:
                fav = min(bar_low, close) if close > 0 else bar_low
                past = fav <= tp
            if not past:
                continue

            if not self._cancel_position_take_profit(pos, reason='past_tp_trail'):
                bot_logger.error(
                    f"Past-TP: broker TP cancel failed for {symbol} {order_id}; "
                    f"keeping TP working to avoid unprotected state"
                )
                print(f"   ⚠️ {symbol} past TP but cancel failed — TP left working")
                continue

            old_tp = tp
            pos.original_tp = old_tp
            pos.tp = 0.0
            pos.tp_cleared = True
            pos.trailing_enabled = True

            # Lock at least the original TP profit immediately when tighter than current SL.
            if self._is_tighter_sl(pos, old_tp, pos.sl):
                old_sl = pos.sl
                if self._apply_stop_update(pos, old_tp):
                    print(
                        f"   🎯 {symbol} past TP {old_tp:.2f}: TP cancelled, "
                        f"SL floored {old_sl:.2f} → {pos.sl:.2f}, trailing ON"
                    )
                else:
                    print(
                        f"   🎯 {symbol} past TP {old_tp:.2f}: TP cancelled, "
                        f"trailing ON (SL floor {old_tp:.2f} pending)"
                    )
            else:
                print(
                    f"   🎯 {symbol} past TP {old_tp:.2f}: TP cancelled, "
                    f"trailing ON (SL already ≥ TP floor)"
                )
            bot_logger.info(
                f"Past-TP trail {symbol} {order_id}: cancelled TP @ {old_tp:.2f}, "
                f"trailing_enabled=True, SL floor={old_tp:.2f}"
            )

    def update_fixed_profit_lock(self, symbol: str, df: pd.DataFrame):
        """2026-10-22: once open profit >= PROFIT_LOCK_TRIGGER $, SL locks +PROFIT_LOCK_AMOUNT $.

        Once per trade, tighten-only, TP kept. Uses _apply_stop_update (same broker
        path as the USD lock / trail). Runs for bot and adopted trades; add-ons skip.
        """
        if PROFIT_LOCK_TRIGGER <= 0 or PROFIT_LOCK_AMOUNT <= 0:
            return
        if not self.positions or df is None or len(df) == 0:
            return
        try:
            q = self.get_price(symbol) or {}
            live = float(q.get('last') or q.get('bid') or q.get('ask') or 0)
        except Exception:
            live = 0.0
        if live <= 0:
            return
        row = df.iloc[-1]
        spec = self._spec(symbol)
        tick = float(spec['tick_size'])
        pv = float(spec['point_value'])
        for order_id, pos in list(self.positions.items()):
            if pos.symbol != symbol or getattr(pos, 'addon', '') or getattr(pos, 'flex3_stream', ''):  # Flex3: own locks
                continue
            if getattr(pos, 'fixed_lock_done', False):
                continue
            live_pts = (live - pos.entry_price) if pos.direction == 'long' else (pos.entry_price - live)
            peak = max(float(getattr(pos, 'peak_pts', 0) or 0), live_pts)
            if self._bar_after_entry(row, pos.entry_time):
                fav_px = float(row['high']) if pos.direction == 'long' else float(row['low'])
                fav = (fav_px - pos.entry_price) if pos.direction == 'long' else (pos.entry_price - fav_px)
                peak = max(peak, fav)
            pos.peak_pts = max(float(getattr(pos, 'peak_pts', 0) or 0), peak, 0.0)
            status, new_sl = fixed_profit_lock_stop(
                pos.direction, pos.entry_price, pos.size, pv, pos.peak_pts, live, tick,
                float(pos.sl or 0))
            if status == 'tighter':
                pos.fixed_lock_done = True
                bot_logger.info(
                    f"PROFIT-LOCK {symbol} {order_id}: hit +${PROFIT_LOCK_TRIGGER:.0f}, SL {pos.sl:.2f} "
                    f"already locks >= +${PROFIT_LOCK_AMOUNT:.0f}; left alone")
                continue
            if status != 'move':
                continue
            old_sl = pos.sl
            if not self._apply_stop_update(pos, new_sl):
                bot_logger.warning(
                    f"PROFIT-LOCK {symbol} {order_id}: stop move to {new_sl:.2f} failed; will retry")
                continue
            pos.fixed_lock_done = True
            peak_usd = pos.peak_pts * pv * max(1, int(pos.size or 1))
            locked = abs(pos.sl - pos.entry_price) * pv * max(1, int(pos.size or 1))
            print(f"   🔐 {symbol} PROFIT-LOCK: up +${peak_usd:.0f} -> SL {old_sl:.2f} -> {pos.sl:.2f} (locks +${locked:.0f}), TP {pos.tp:.2f} kept")
            bot_logger.info(
                f"PROFIT-LOCK fired {symbol} {order_id} {pos.direction} x{pos.size}: peak=${peak_usd:.2f} "
                f"live={live:.2f} SL {old_sl:.2f} -> {pos.sl:.2f} locks ${locked:.2f} TP={pos.tp:.2f}")

    def update_dollar_profit_locks(self, symbol: str, df: pd.DataFrame):
        """R-based profit locks (name kept for call site). See r_lock_stop()."""
        if not PROFIT_LOCK_ENABLED or not self.positions or df is None or len(df) == 0:
            return
        row = df.iloc[-1]
        bar_high = float(row['high'])
        bar_low = float(row['low'])
        live = 0.0
        try:
            q = self.get_price(symbol) or {}
            live = float(q.get('last') or q.get('bid') or q.get('ask') or 0)
        except Exception:
            live = 0.0
        if live <= 0:
            return  # live price required
        tick = float(self._spec(symbol)['tick_size'])
        if LOCK_MODE == 'usd':
            self._update_usd_locks(symbol, row, live, tick)
            return
        for order_id, pos in list(self.positions.items()):
            if pos.symbol != symbol or getattr(pos, 'addon', ''):  # add-on: fixed SL/TP
                continue
            if float(getattr(pos, 'initial_risk', 0) or 0) <= 0:
                r = (pos.entry_price - pos.sl) if pos.direction == 'long' else (pos.sl - pos.entry_price)
                if r > 0:
                    pos.initial_risk = r
            risk = float(getattr(pos, 'initial_risk', 0) or 0)
            if risk <= 0:
                continue
            fav_px = bar_high if pos.direction == 'long' else bar_low
            fav = (fav_px - pos.entry_price) if pos.direction == 'long' else (pos.entry_price - fav_px)
            pos.mfe_pts = max(float(pos.mfe_pts or 0), fav)
            new_sl = r_lock_stop(pos.direction, pos.entry_price, risk, pos.mfe_pts,
                                 live, tick, float(pos.sl or 0))
            if new_sl is None:
                continue
            old_sl = pos.sl
            if not self._apply_stop_update(pos, new_sl):
                continue
            level = 2 if pos.mfe_pts / risk >= LOCK_R2 else 1
            pos.profit_lock_level = max(int(pos.profit_lock_level or 0), level)
            print(f"   🔐 {symbol} R-lock L{level} (MFE {pos.mfe_pts / risk:.2f}R): SL {old_sl:.2f} → {pos.sl:.2f}")
            bot_logger.info(
                f"R-lock L{level} {symbol} {order_id}: MFE={pos.mfe_pts:.2f}pts R={risk:.2f} "
                f"live={live:.2f} SL {old_sl:.2f} -> {pos.sl:.2f}"
            )

    @staticmethod
    def _bar_after_entry(row, entry_time) -> bool:
        """True only when this bar started after the fill (its high/low are ours)."""
        try:
            raw = row.get('datetime', None) if hasattr(row, 'get') else None
            if raw is None:
                raw = row.get('time', None) if hasattr(row, 'get') else None
            if raw is None or entry_time is None:
                return False
            bt = pd.Timestamp(raw)
            et = pd.Timestamp(entry_time)
            if bt.tzinfo is None:
                bt = bt.tz_localize('UTC')
            if et.tzinfo is None:
                et = et.tz_localize('UTC')
            return bt > et
        except Exception:
            return False

    def _update_usd_locks(self, symbol: str, row, live: float, tick: float) -> None:
        pv = float(self._spec(symbol)['point_value'])
        for order_id, pos in list(self.positions.items()):
            if pos.symbol != symbol or getattr(pos, 'addon', ''):  # add-on: fixed SL/TP
                continue
            live_pts = (live - pos.entry_price) if pos.direction == 'long' else (pos.entry_price - live)
            peak = max(float(getattr(pos, 'peak_pts', 0) or 0), live_pts)
            if self._bar_after_entry(row, pos.entry_time):
                fav_px = float(row['high']) if pos.direction == 'long' else float(row['low'])
                fav = (fav_px - pos.entry_price) if pos.direction == 'long' else (pos.entry_price - fav_px)
                peak = max(peak, fav)
            pos.peak_pts = max(0.0, peak)
            pos.mfe_pts = max(float(pos.mfe_pts or 0), pos.peak_pts)
            _lk = F3.lock_params(pos.flex3_stream, pos.size) if (FLEX3_ON and pos.flex3_stream) else {}
            new_sl = usd_lock_stop(pos.direction, pos.entry_price, pos.size, pv,
                                   pos.peak_pts, live, tick, float(pos.sl or 0), **_lk)
            if new_sl is None:
                continue
            old_sl = pos.sl
            if not self._apply_stop_update(pos, new_sl):
                continue
            peak_usd = pos.peak_pts * pv * max(1, int(pos.size or 1))
            level = 2 if peak_usd >= LOCK_TRAIL_START_USD else 1
            pos.profit_lock_level = max(int(pos.profit_lock_level or 0), level)
            locked_usd = abs(pos.sl - pos.entry_price) * pv * max(1, int(pos.size or 1))
            print(f"   $-lock L{level} {symbol}: peak +${peak_usd:.0f} -> SL {old_sl:.2f} -> {pos.sl:.2f} (locks +${locked_usd:.0f})")
            bot_logger.info(
                f"USD-lock L{level} {symbol} {order_id}: peak=${peak_usd:.2f} size={pos.size} "
                f"live={live:.2f} SL {old_sl:.2f} -> {pos.sl:.2f} locks ${locked_usd:.2f}"
            )

    def update_trailing_stops(self, symbol: str, df: pd.DataFrame):
        """Move SL before TP is reached to protect unrealized gains."""
        if not self.positions or df is None or len(df) == 0:
            return

        row = df.iloc[-1]
        bar_high = float(row['high'])
        bar_low = float(row['low'])
        atr = float(row.get('atr', np.nan))
        if not np.isfinite(atr) or atr <= 0:
            return

        tick = self._spec(symbol)['tick_size']
        min_move = max(1, TRAIL_SETTINGS['min_step_ticks']) * tick
        activation_frac = max(0.05, min(0.95, TRAIL_SETTINGS['activation_tp_fraction']))
        activation_r = max(0.05, TRAIL_SETTINGS.get('activation_r', 0.30))
        trail_dist = max(tick, atr * max(0.05, TRAIL_SETTINGS['atr_mult']))

        for order_id, pos in list(self.positions.items()):
            if pos.symbol != symbol or getattr(pos, 'addon', ''):  # add-on: fixed SL/TP
                continue
            if not pos.trailing_enabled:
                continue
            # USD lock mode owns stop movement before TP (no early ATR / 0.25R trail).
            # The ATR trail only runs after price went past TP (tp_cleared).
            if LOCK_MODE == 'usd' and not getattr(pos, 'tp_cleared', False):
                continue

            tp_dist = abs(pos.tp - pos.entry_price) if pos.tp and pos.tp > 0 else 0.0
            risk_dist = max(tick, abs(pos.entry_price - pos.sl))
            # Past-TP trail: already beyond target — keep trailing without fixed TP.
            # REVAMP trend sleeve: no fixed TP — activate trail on small R move (research-like).
            if getattr(pos, 'tp_cleared', False):
                activation_move = 0.0
            elif tp_dist <= 0:
                if not REVAMP_EXITS_ON:
                    continue
                activation_move = risk_dist * max(0.05, activation_r)
            else:
                activation_move = min(tp_dist * activation_frac, risk_dist * activation_r)

            if pos.direction == 'long':
                favorable_move = bar_high - pos.entry_price
                if favorable_move < activation_move:
                    continue

                breakeven_sl = pos.entry_price + tick
                trail_sl = bar_high - trail_dist
                candidate_sl = self._round_to_tick(max(breakeven_sl, trail_sl), tick)

                # Keep stop on the valid side of market and only ratchet upward.
                if candidate_sl >= bar_low:
                    candidate_sl = self._round_to_tick(bar_low - tick, tick)
                if candidate_sl <= pos.sl or (candidate_sl - pos.sl) < min_move:
                    continue
            else:
                favorable_move = pos.entry_price - bar_low
                if favorable_move < activation_move:
                    continue

                breakeven_sl = pos.entry_price - tick
                trail_sl = bar_low + trail_dist
                candidate_sl = self._round_to_tick(min(breakeven_sl, trail_sl), tick)

                # Keep stop on the valid side of market and only ratchet downward.
                if candidate_sl <= bar_high:
                    candidate_sl = self._round_to_tick(bar_high + tick, tick)
                if candidate_sl >= pos.sl or (pos.sl - candidate_sl) < min_move:
                    continue

            # After past-TP: never pull SL back through the original TP profit level.
            orig_tp = float(getattr(pos, 'original_tp', 0) or 0)
            if getattr(pos, 'tp_cleared', False) and orig_tp > 0:
                if pos.direction == 'long':
                    candidate_sl = max(candidate_sl, orig_tp)
                else:
                    candidate_sl = min(candidate_sl, orig_tp)
                # Re-check ratchet / min step after flooring.
                if pos.direction == 'long':
                    if candidate_sl <= pos.sl or (candidate_sl - pos.sl) < min_move:
                        continue
                else:
                    if candidate_sl >= pos.sl or (pos.sl - candidate_sl) < min_move:
                        continue

            old_sl = pos.sl
            if self.paper_mode or self.paper_orders:
                pos.sl = candidate_sl
                print(f"   🔒 {symbol} trail: SL {old_sl:.2f} → {candidate_sl:.2f}")
                continue

            # Prefer update_stop_loss (native modify OR protective STOP/STOP-LIMIT replace).
            if self._apply_stop_update(pos, candidate_sl):
                print(f"   🔒 {symbol} trail: SL {old_sl:.2f} → {pos.sl:.2f}")
                bot_logger.info(
                    f"Trailing stop updated {symbol} {order_id}: {old_sl:.2f} -> {pos.sl:.2f}"
                )
                continue

            if self.broker and hasattr(self.broker, 'get_order_info'):
                order_info = self.broker.get_order_info(order_id) or {}
                if order_info and not self._broker_can_update_stop(order_info):
                    pos.trailing_enabled = False
                    bot_logger.error(
                        f"Trailing disabled for {symbol} {order_id}: no broker stop available to trail"
                    )
                    print(f"   ⚠️ {symbol} trail disabled for {order_id}: no broker stop to update")
        
    def connect(self) -> bool:
        """Initialize Rithmic connection (skipped in paper mode)."""
        try:
            # Paper mode: skip Rithmic, use Yahoo Finance for data
            if self.paper_mode and not self.paper_orders:
                print("📝 PAPER MODE - Skipping Rithmic connection")
                print("   Using Yahoo Finance for market data")
                self.broker = None
                
                # Initialize full AI ensemble without broker
                print("\n🧠 Initializing AI Ensemble (ML + Advanced Strategies)...")
                self.ensemble = EnsembleTrader(broker=None)
                print("✅ AI Ensemble ready!")
                return True
            
            # Live mode: connect to Rithmic
            os.environ['RITHMIC_DISABLE_YAHOO_FALLBACK'] = 'true'
            self.broker = RithmicConnector()
            self.broker.initialize()
            
            if not self.broker.connected:
                print("❌ Rithmic not connected - check credentials in .env")
                print(f"   RITHMIC_USER_ID: {os.getenv('RITHMIC_USER_ID', 'NOT SET')}")
                print(f"   RITHMIC_SYSTEM: {os.getenv('RITHMIC_SYSTEM', 'NOT SET')}")
                if self.broker._last_connect_error:
                    print(f"   Error: {self.broker._last_connect_error}")
                return False
            
            # Get account info
            acct = self.broker.get_account_info()
            if FLEX3_ON:
                self._flex3_account = str(acct.get('account_id') or '')
                if not F3.account_ok(self._flex3_account):
                    print(f"❌ FLEX3: account {self._flex3_account!r} not allowed ({F3.__name__}) — refusing to trade")
                    bot_logger.error(f"FLEX3 account lock: {self._flex3_account!r} refused")
                    try:
                        self.broker.shutdown()
                    except Exception:
                        pass
                    self.broker = None
                    return False
                print(f"   FLEX3: account lock OK ({self._flex3_account})")
            print(f"✅ Rithmic Connected!")
            print(f"   System: {acct.get('system', 'Unknown')}")
            print(f"   Balance: ${acct.get('balance', 0):,.2f}")
            print(f"   Equity: ${acct.get('equity', 0):,.2f}")
            print(f"   Mode: ⚠️  LIVE - Real money at risk!")
            
            # Initialize full AI ensemble
            print("\n🧠 Initializing AI Ensemble (ML + Advanced Strategies)...")
            self.ensemble = EnsembleTrader(broker=self.broker)
            print("✅ AI Ensemble ready!")
            
            return True
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def get_candles(self, symbol: str, count: int = 100) -> Optional[pd.DataFrame]:
        """Fetch candles from Rithmic only in live mode."""
        try:
            # Paper mode or no broker: use Yahoo Finance
            if self.broker is None:
                return self._get_yahoo_candles(count)
            
            df = self.broker.get_candles(symbol, timeframe_minutes=5, num_candles=count)
            min_bars = self.lookback + self.ema_len + 15  # 75 bars needed
            if REVAMP_EXITS_ON:
                min_bars = max(min_bars, 220)  # loop needs 220 for revamp
            min_bars = min(min_bars, int(count))
            if df is None or len(df) < min_bars:
                rith_bars = 0 if df is None else len(df)
                local_df = self._get_local_history_candles(symbol, count)
                merged = self._merge_candles(local_df, df, count)
                # 2026-10-11: keep every bar seen this run so the count only
                # grows each loop (new 5m bars always count toward warm-up).
                accum = self.__dict__.setdefault('_warm_bar_accum', {})
                prev = accum.get(symbol)
                # Only extend fresh data; never hand back old bars on a failed fetch.
                if prev is not None and merged is not None and len(merged) > 0:
                    merged = stitch_history(prev, merged, count, timeframe_minutes=5)
                if merged is not None and len(merged) > 0:
                    accum[symbol] = merged.copy()
                merged_bars = 0 if merged is None else len(merged)
                # Avoid repeating the same warm-up message every loop.
                state_key = (rith_bars, merged_bars)
                if self._warmup_log_state.get(symbol) != state_key:
                    self._warmup_log_state[symbol] = state_key
                    print(
                        f"⚠️  {symbol} Rithmic returned {rith_bars} bars, need {min_bars} - "
                        f"using local warm history ({merged_bars} bars total)"
                    )
                return merged
            return df
        except Exception as e:
            print(f"❌ Error getting candles for {symbol}: {e}")
            return None

    def _get_local_history_candles(self, symbol: str, count: int = 100) -> Optional[pd.DataFrame]:
        """Load local 5m history CSV as indicator warm-start (no Yahoo usage)."""
        try:
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
            path = os.path.join(data_dir, f'{symbol}_5m.csv')
            if not os.path.exists(path):
                return None

            df = pd.read_csv(path)
            df.columns = [str(c).lower() for c in df.columns]
            if 'datetime' not in df.columns and 'date' in df.columns:
                df = df.rename(columns={'date': 'datetime'})
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
                df = df.dropna(subset=['datetime'])

            required = {'open', 'high', 'low', 'close'}
            if not required.issubset(set(df.columns)):
                return None

            keep_cols = [c for c in ['datetime', 'open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            # 2026-10-11: age in Globex trading time (weekend / 5-6 PM halt don't count).
            fresh = filter_warm_history_market(df[keep_cols], max_age_hours=6)
            if fresh is None or len(fresh) == 0:
                warned = getattr(self, "_stale_warm_warned", None)
                if warned is None:
                    self._stale_warm_warned = set()
                    warned = self._stale_warm_warned
                if symbol not in warned:
                    warned.add(symbol)
                    bot_logger.warning(
                        f"Local warm history for {symbol} is older than 6h of trading time — not trading off it"
                    )
                return None
            return fresh.tail(count).reset_index(drop=True)
        except Exception as e:
            bot_logger.warning(f"Local history warm-start failed for {symbol}: {e}")
            return None

    def _merge_candles(
        self,
        history_df: Optional[pd.DataFrame],
        live_df: Optional[pd.DataFrame],
        count: int,
    ) -> Optional[pd.DataFrame]:
        """Merge local history with live bars and keep latest rows.

        2026-10-11: history only goes in front of the first live bar, and only
        when there is no trading-time gap (stitch_history).
        """
        try:
            return stitch_history(history_df, live_df, count, timeframe_minutes=5)
        except Exception as _stitch_err:
            bot_logger.warning(f"Warm history stitch failed, plain merge: {_stitch_err}")
        if history_df is None and live_df is None:
            return None
        if history_df is None:
            return live_df.tail(count).reset_index(drop=True)
        if live_df is None:
            return history_df.tail(count).reset_index(drop=True)

        merged = pd.concat([history_df, live_df], ignore_index=True)
        if 'datetime' in merged.columns:
            merged['datetime'] = pd.to_datetime(merged['datetime'], utc=True, errors='coerce')
            merged = merged.dropna(subset=['datetime'])
            merged = merged.drop_duplicates(subset=['datetime'], keep='last')
            merged = merged.sort_values('datetime')
        else:
            merged = merged.drop_duplicates(keep='last')
        return merged.tail(count).reset_index(drop=True)
    
    def _get_yahoo_candles(self, count: int = 100, symbol: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Fetch candles from Yahoo Finance as fallback."""
        try:
            import yfinance as yf
            
            # Map futures symbols to Yahoo tickers
            ticker_map = {
                'MES': 'ES=F',  # E-mini S&P 500 futures
                'MNQ': 'NQ=F',  # E-mini Nasdaq futures
                'NQ':  'NQ=F',  # Nasdaq 100 futures
            }
            target = symbol or self.symbol
            ticker = ticker_map.get(target, f'{target}=F')
            
            data = yf.download(ticker, period='5d', interval='5m', progress=False)
            if data.empty:
                return None
            
            df = data.reset_index()
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            
            if 'datetime' not in df.columns and 'date' in df.columns:
                df = df.rename(columns={'date': 'datetime'})
            
            return df.tail(count).reset_index(drop=True)
        except Exception as e:
            bot_logger.warning(f"Yahoo Finance fallback failed: {e}")
            return None
    
    def get_price(self, symbol: Optional[str] = None) -> Optional[Dict]:
        """Get current bid/ask."""
        try:
            if self.broker is None:
                # Paper mode: get last close from Yahoo
                df = self._get_yahoo_candles(5, symbol=symbol)
                if df is not None and len(df) > 0:
                    price = float(df['close'].iloc[-1])
                    return {'bid': price - 0.25, 'ask': price + 0.25, 'last': price}
                return None
            return self.broker.get_latest_price(symbol or self.symbol)
        except Exception as e:
            print(f"❌ Error getting price: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy indicators."""
        df = df.copy()
        df['high_N'] = df['high'].rolling(self.lookback).max().shift(1)
        df['low_N'] = df['low'].rolling(self.lookback).min().shift(1)
        df['ema'] = df['close'].ewm(span=self.ema_len, adjust=False).mean()
        
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        return df
    
    def check_daily_limits(self) -> bool:
        """Check if daily limits allow trading."""
        today = datetime.now().date()
        if self.current_date != today:
            self.current_date = today
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.loss_cooldown_until.clear()
            self._symbol_consec_losses = {sym: 0 for sym in self.symbols}
            self._win_streak = 0
            self._consec_loss_pause_until.clear()
            self.__dict__.setdefault('_reentry_arms', {}).clear()
            if self.ensemble:
                self.ensemble.learner.decay_loss_patterns()
            print(f"\n📅 New trading day: {today}")
        
        # Session-limit mode (REVAMP): the calendar-day loss check is replaced by
        # _session_limit_block(), which only blocks new entries (loop keeps managing trades).
        if getattr(self, '_session_ledger', None) is None and self.daily_pnl <= -self.daily_loss_limit:
            print(f"🛑 Daily loss limit hit: ${self.daily_pnl:.2f}")
            return False
        if self.max_trades_per_day > 0 and self.daily_trades >= self.max_trades_per_day:
            print(f"🛑 Max trades hit: {self.daily_trades}/{self.max_trades_per_day}")
            return False
        return True
    
    def check_entry_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Check for entry signal using full AI Ensemble (or REVAMP research ensemble)."""
        if self._effective_open_count() >= self.max_positions or self.cooldown > 0:
            return None

        if self._symbol_in_loss_cooldown(symbol) or self._symbol_in_consec_loss_pause(symbol):
            return None

        if self._session_limit_block():
            return None

        if not self._is_trading_session_allowed(symbol):
            return None

        # OG Tokyo-open block (OG_TOKYO_BLOCK_MIN, default off; never under EVAL PASSER/FLEX3). New entries only.
        if not FLEX3_ON:
            try:
                from src.strategy.og_tokyo_block import tokyo_window_end
                _tk_end = tokyo_window_end()
            except Exception as _tk_e:
                _tk_end = None
                bot_logger.warning(f"OG Tokyo block check failed (ignored): {_tk_e}")
            if _tk_end is not None:
                _tk_msg = f"Tokyo open window: skipping entry until {_tk_end.strftime('%H:%M')} ET"
                if getattr(self, '_tk_last_msg', None) != _tk_msg:
                    print(f"   {_tk_msg}")
                    bot_logger.info(_tk_msg)
                    self._tk_last_msg = _tk_msg
                return None

        # REVAMP path: 5m ADX router (trend / MR / breakout) with research winner exits.
        if REVAMP_EXITS_ON:
            return self._check_revamp_research_signal(symbol, df)
        
        # Use full EnsembleTrader for signal
        if self.ensemble is None:
            return self._check_simple_breakout(symbol, df)
        
        try:
            # Add indicators via technical analyzer
            df_enriched = self.technical.calculate_indicators(df)
            
            # Get ensemble signal (includes sweep gate, ML, advanced strategies)
            signal_result = self.ensemble.get_trading_signal(df_enriched, symbol)
            
            # Check if should trade
            if not self.ensemble.should_trade(signal_result):
                return None
            
            signal = signal_result['signal']
            confidence = signal_result['confidence']
            
            if signal not in ('BUY', 'SELL'):
                return None
            
            row = df.iloc[-1]
            price = row['close']
            atr = df_enriched['atr'].iloc[-1] if 'atr' in df_enriched.columns else row.get('atr', 10.0)
            
            if pd.isna(atr) or atr < 2.0:
                atr = 10.0  # Fallback ATR for futures
            
            # Use dynamic SL/TP if available
            if hasattr(self.ensemble, 'get_dynamic_sl_tp'):
                direction = 'BUY' if signal == 'BUY' else 'SELL'
                sltp = self.ensemble.get_dynamic_sl_tp(
                    df_enriched,
                    direction,
                    price,
                    symbol=symbol,
                    sr_levels=signal_result.get('sr_levels'),
                    sweep_wick=signal_result.get('sweep_wick'),
                    timeframe='5m',
                )
                sl = sltp['sl_price']
                tp = sltp['tp_price']
                sltp_method = sltp.get('method', 'unknown') if isinstance(sltp, dict) else 'unknown'
            else:
                sl_dist = atr * self.atr_mult
                if signal == 'BUY':
                    sl = price - sl_dist
                    tp = price + (sl_dist * self.tp_mult)
                else:
                    sl = price + sl_dist
                    tp = price - (sl_dist * self.tp_mult)
                sltp_method = 'atr_fallback'
            
            direction = 'long' if signal == 'BUY' else 'short'
            
            bot_logger.info(f"🎯 ENSEMBLE SIGNAL: {symbol} {signal} @ {price:.2f} (conf={confidence:.1%})")
            bot_logger.info(f"   SL: {sl:.2f} | TP: {tp:.2f}")
            bot_logger.info(f"   SL/TP method: {sltp_method}")
            
            return {
                'symbol': symbol,
                'direction': direction,
                'entry': price,
                'sl': sl,
                'tp': tp,
                'atr': atr,
                'confidence': confidence,
                'ensemble_signal': signal,
                'model_signals': signal_result.get('models', {}),
                'regime': signal_result.get('regime', ''),
            }
            
        except Exception as e:
            bot_logger.warning(f"Ensemble signal error: {e}, falling back to breakout")
            return self._check_simple_breakout(symbol, df)
    

    def _check_revamp_research_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Research ensemble entries + revamp exits (trail 4ATR / breakout 2R / MR bb_far)."""
        try:
            # Need deeper 5m history for EMA200 + ADX router
            if df is None or len(df) < 220:
                deep = self.get_candles(symbol=symbol, count=REVAMP_CANDLE_COUNT)
                if deep is not None and (df is None or len(deep) > len(df)):
                    df = deep
            if df is None or len(df) < 220:
                bot_logger.info(
                    f"REVAMP {symbol}: need >=220 5m bars, have {0 if df is None else len(df)}"
                )
                return None

            hit = classify_last_sleeve(df)
            if not hit:
                return None

            tick = self._spec(symbol)["tick_size"]
            sl = self._round_to_tick(float(hit["sl"]), tick)
            tp_raw = hit.get("tp")
            if tp_raw is None:
                tp = 0.0  # trail-only (Rithmic accepts take_profit<=0)
            else:
                tp = self._round_to_tick(float(tp_raw), tick)

            sleeve = str(hit.get("sleeve") or "")
            _ovn_why = revamp_overnight_block(sleeve)
            if _ovn_why:
                self._ovn_sleeve_skip = (sleeve, time.time())
                bot_logger.info(f"REVAMP {symbol} {hit.get('ensemble_signal')} {sleeve} skipped: {_ovn_why}")
                return None
            trailing = bool(hit.get("trailing", True))
            if sleeve in ("breakout", "mean_reversion"):
                trailing = False
            elif sleeve == "trend":
                # 2026-10-08: trend gets a fixed TP (REVAMP_TREND_TP_R, default 1.5R) -> no trail,
                # broker TP LIMIT placed like breakouts. tp<=0 (REVAMP_TREND_TP_R=0) -> old trail-only.
                trailing = tp <= 0

            tp_label = f"ATR trail x{TRAIL_SETTINGS['atr_mult']}" if tp <= 0 else f"{tp:.2f}"
            bot_logger.info(
                f"REVAMP SIGNAL: {symbol} {hit['ensemble_signal']} sleeve={sleeve} "
                f"@ {hit['entry']:.2f} SL={sl:.2f} TP={tp_label} "
                f"ADX={hit.get('adx', float('nan')):.1f}"
            )
            print(
                f"   REVAMP {sleeve}: {hit['ensemble_signal']} @ {hit['entry']:.2f} | "
                f"SL {sl:.2f} | TP {tp_label}"
            )

            return {
                "symbol": symbol,
                "direction": hit["direction"],
                "entry": float(hit["entry"]),
                "sl": sl,
                "tp": tp,
                "atr": float(hit.get("atr") or 10.0),
                "confidence": _revamp_confidence(hit, sleeve),
                "ensemble_signal": hit["ensemble_signal"],
                "model_signals": {"revamp_sleeve": sleeve, "adx": hit.get("adx")},
                "regime": hit.get("regime", ""),
                "trailing_enabled": trailing,
                "exit_sleeve": sleeve,
            }
        except Exception as e:
            bot_logger.warning(f"REVAMP signal error for {symbol}: {e}")
            return None

    # ── DISPLAY-ONLY revamp readout (never changes state or orders) ─────────

    def _revamp_session_block_reason(self, symbol: str) -> Optional[str]:
        """Same rule as _is_trading_session_allowed, but silent and read-only."""
        now = datetime.now(timezone.utc)
        pair_cfg = SCALPING_PAIRS.get(symbol, {})
        skip_friday = pair_cfg.get('skip_friday', self.skip_friday)
        bad_hours = pair_cfg.get('bad_hours_utc', self.bad_hours.get(symbol, []))
        if _revamp_exits.overnight_entries_enabled():
            bad_hours = []
        if skip_friday and now.weekday() == 4:
            return "Friday filter"
        if now.hour in bad_hours:
            return f"bad hour filter ({now.hour}:00 UTC)"
        return None

    def _revamp_gate_state(self, symbol: str, open_count: int) -> Dict[str, Any]:
        """Read-only snapshot of the entry gates (does not clear expired timers)."""
        now = datetime.now(timezone.utc)

        def _mins_left(until: Optional[datetime]) -> int:
            if until is None or now >= until:
                return 0
            return int((until - now).total_seconds() // 60) + 1

        return {
            'daily_pnl': self.daily_pnl,
            'daily_loss_limit': (float('inf') if getattr(self, '_session_ledger', None) is not None
                                 else self.daily_loss_limit),
            'daily_trades': self.daily_trades,
            'max_trades': self.max_trades_per_day,
            'open_count': open_count,
            'max_positions': self.max_positions,
            'cooldown': self.cooldown,
            'loss_cooldown_min': _mins_left(self.loss_cooldown_until.get(symbol)),
            'consec_pause_min': _mins_left(self._consec_loss_pause_until.get(symbol)),
            'consec_losses': self._symbol_consec_losses.get(symbol, 0),
            'session_block': (getattr(self, '_last_session_block', None)
                              or self._revamp_session_block_reason(symbol)),
        }

    def _print_revamp_reasoning(
        self,
        symbol: str,
        df: pd.DataFrame,
        open_count: int,
        checked: bool,
        signal: Optional[Dict],
    ) -> None:
        """Print bias / regime / sleeves / decision. Errors never reach trading."""
        try:
            gates = self._revamp_gate_state(symbol, open_count)
            for line in _revamp_display.render_revamp_block(
                df, gates=gates, signal=signal, checked=checked, level=REVAMP_VERBOSE_LEVEL,
            ):
                print(line)
        except Exception as e:
            if not getattr(self, '_revamp_display_err_shown', False):
                self._revamp_display_err_shown = True
                print(f"   (revamp readout unavailable: {e})")
            try:
                bot_logger.debug(f"REVAMP display error for {symbol}: {e}")
            except Exception:
                pass

    def _check_simple_breakout(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Fallback simple breakout strategy."""
        row = df.iloc[-1]
        price = row['close']
        high = row['high']
        low = row['low']
        atr = row.get('atr', 10.0)
        high_n = row.get('high_N', row['high'])
        low_n = row.get('low_N', row['low'])
        ema = row.get('ema', price)
        
        tick_size = self._spec(symbol)['tick_size']
        if pd.isna(atr) or atr < 2.0:
            return None
        
        # Long breakout
        if high > high_n and price > high_n and price > ema:
            entry = high_n + tick_size
            sl_dist = atr * self.atr_mult
            sl = entry - sl_dist
            tp = entry + (sl_dist * self.tp_mult)
            return {'symbol': symbol, 'direction': 'long', 'entry': entry, 'sl': sl, 'tp': tp, 'atr': atr, 'confidence': 0.5}
        
        # Short breakout
        elif low < low_n and price < low_n and price < ema:
            entry = low_n - tick_size
            sl_dist = atr * self.atr_mult
            sl = entry + sl_dist
            tp = entry - (sl_dist * self.tp_mult)
            return {'symbol': symbol, 'direction': 'short', 'entry': entry, 'sl': sl, 'tp': tp, 'atr': atr, 'confidence': 0.5}
        
        return None
    
    def _symbol_in_loss_cooldown(self, symbol: str) -> bool:
        until = self.loss_cooldown_until.get(symbol)
        if until is None:
            return False
        now = datetime.now(timezone.utc)
        if now < until:
            remaining = int((until - now).total_seconds() // 60) + 1
            bot_logger.info(f"⏳ {symbol}: loss cooldown — {remaining} min left")
            return True
        del self.loss_cooldown_until[symbol]
        return False

    def _symbol_in_consec_loss_pause(self, symbol: str) -> bool:
        until = self._consec_loss_pause_until.get(symbol)
        if until is None:
            return False
        now = datetime.now(timezone.utc)
        if now < until:
            remaining = int((until - now).total_seconds() // 60) + 1
            bot_logger.info(
                f"🛑 {symbol}: consecutive-loss pause — {remaining} min left "
                f"({self._symbol_consec_losses.get(symbol, 0)} in a row)"
            )
            return True
        del self._consec_loss_pause_until[symbol]
        self._symbol_consec_losses[symbol] = 0
        return False

    def _update_loss_protection(self, symbol: str, pnl: float) -> None:
        if pnl < 0:
            self._win_streak = 0
            count = self._symbol_consec_losses.get(symbol, 0) + 1
            self._symbol_consec_losses[symbol] = count
            self.loss_cooldown_until[symbol] = (
                datetime.now(timezone.utc) + timedelta(minutes=LOSS_COOLDOWN_MINUTES)
            )
            if count >= ENSEMBLE_MAX_CONSEC_LOSSES:
                self._consec_loss_pause_until[symbol] = (
                    datetime.now(timezone.utc) + timedelta(minutes=ENSEMBLE_CONSEC_LOSS_PAUSE_MIN)
                )
                print(
                    f"   🛑 {symbol}: {count} consecutive losses — "
                    f"pause {ENSEMBLE_CONSEC_LOSS_PAUSE_MIN} min"
                )
        elif pnl > 0:
            self._symbol_consec_losses[symbol] = 0
            self._win_streak = int(getattr(self, '_win_streak', 0) or 0) + 1

    def _feed_learner_on_exit(
        self,
        pos: Position,
        pnl: float,
        exit_type: str,
        exit_price: float,
    ) -> None:
        if not self.ensemble:
            return
        trade_signal = (pos.ensemble_signal or '').upper()
        if trade_signal not in ('BUY', 'SELL'):
            trade_signal = 'BUY' if pos.direction == 'long' else 'SELL'
        self.ensemble.record_trade_result({
            'pair': pos.symbol,
            'signal': trade_signal,
            'profit_loss': pnl,
            'entry_price': pos.entry_price,
            'exit_price': exit_price,
            'exit_type': exit_type,
            # Learner expects {model: {'signal': ...}}; revamp stores plain values
            # (e.g. revamp_sleeve='breakout'), which made it crash on str.get.
            'model_signals': {
                k: v for k, v in (pos.model_signals if isinstance(pos.model_signals, dict) else {}).items()
                if isinstance(v, dict)
            },
            'regime': pos.regime or getattr(self.ensemble.learner, 'current_regime', 'unknown'),
        })

    def _contracts_for_entry(self, signal: Dict) -> Tuple[int, float, str]:
        """Gated confidence sizing: base=1 until day green + win streak, max 4."""
        cfg = getattr(self, '_size_cfg', None) or SIZE_SCALE_CFG
        try:
            confidence = float(signal.get('confidence', 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if signal.get('addon') and int(signal.get('fixed_contracts', 0) or 0) > 0:
            return int(signal['fixed_contracts']), confidence, f"addon_{signal['addon']}_fixed{int(signal['fixed_contracts'])}"
        contracts, reason = compute_sized_contracts(
            confidence,
            daily_pnl=float(getattr(self, 'daily_pnl', 0.0) or 0.0),
            win_streak=int(getattr(self, '_win_streak', 0) or 0),
            base=int(cfg.get('base', 1)),
            max_contracts=int(cfg.get('max_contracts', 4)),
            win_streak_min=int(cfg.get('win_streak_min', 0)),
            require_day_green=bool(cfg.get('require_day_green', False)),
            conf_2=float(cfg.get('conf_2', 0.72)),
            conf_3=float(cfg.get('conf_3', 0.78)),
            conf_4=float(cfg.get('conf_4', 0.85)),
        )
        contracts = max(1, min(int(contracts), int(cfg.get('max_contracts', 4))))
        # 2026-10-09: REVAMP is fixed 1 contract per trade in code too, so a launch without the
        # bat's MAX_CONTRACTS=1 (e.g. .env MAX_CONTRACTS=4 + conf sizing) can't size up.
        # REVAMP_MAX_CONTRACTS (default 1; 0 = no extra clamp).
        if REVAMP_EXITS_ON:
            _rv_max = _env_int('REVAMP_MAX_CONTRACTS', 1)
            if _rv_max > 0 and contracts > _rv_max:
                contracts = _rv_max
                reason = f"{reason} revamp_fixed{_rv_max}"
        return contracts, confidence, reason

    # ── flex3 hooks ─────────────────────────────────────────────────────────
    def _flex3_init(self) -> None:
        self._flex3 = F3.Flex3Guard.load(FLEX3_STATE_FILE)
        self._flex3_account = getattr(self, '_flex3_account', '') or (
            F3.ACCOUNT_ALLOWED if (self.paper_mode or self.paper_orders) else '')
        print(f"   FLEX3 ON: eval_pnl=${self._flex3.eval_pnl:.0f} hwm=${self._flex3.eval_hwm:.0f} "
              f"best_day=${self._flex3.best_day:.0f} loss_stop=${F3.DAILY_LOSS_STOP:.0f} cap=${F3.DAY_PROFIT_CAP:.0f}")

    def _flex3_allow(self, signal: Dict):
        if not hasattr(self, '_flex3'):
            self._flex3_init()
        now = datetime.now(F3.ET)
        self._flex3.new_day(F3.trading_day(now))
        stream = F3.classify_stream(signal)
        q = None
        if stream == 'og':                                            # FUNDED 12K: OG main signal must be A+ (causal features)
            try:
                d1 = self._flex3_bars()
                q = F3.og_quality(d1, pd.Timestamp(datetime.now(timezone.utc)), float(signal['entry']), signal['direction'])
            except Exception as e:
                bot_logger.warning(f"FUNDED og_quality failed: {e}")
        ok, why = self._flex3.allow_entry(stream, now.time(), getattr(self, '_flex3_account', ''), quality=q)
        return ok, why, stream

    def _flex3_on_close(self, pnl: float) -> None:
        if not hasattr(self, '_flex3'):
            self._flex3_init()
        self._flex3.on_close(float(pnl), F3.trading_day(datetime.now(F3.ET)))
        self._flex3.save(FLEX3_STATE_FILE)
        print(f"   FLEX3: day ${self._flex3.day_pnl:.0f} eval ${self._flex3.eval_pnl:.0f} "
              f"best_day ${self._flex3.best_day:.0f} passed={self._flex3.passed()}")

    def _flex3_save_planner(self) -> None:
        try:
            p = FLEX3_STATE_FILE.replace('.json', '_planner.json')
            with open(p + '.tmp', 'w') as fh:
                json.dump(self._flex3_planner.state(), fh)
            os.replace(p + '.tmp', p)
        except Exception as e:
            bot_logger.warning(f"FLEX3 planner state save failed: {e}")

    def _flex3_bars(self) -> Optional[pd.DataFrame]:
        """CLOSED 1m bars (UTC index = bar open): FLEX3_1M_CSV history (NEAT export) + fresh broker bars."""
        frames = []
        try:
            mt = os.path.getmtime(FLEX3_1M_CSV)
            if getattr(self, '_flex3_csv_mtime', None) != mt:
                h = pd.read_csv(FLEX3_1M_CSV)
                h['datetime'] = pd.to_datetime(h['datetime'], utc=True, errors='coerce')
                self._flex3_csv = h.dropna(subset=['datetime']); self._flex3_csv_mtime = mt
            frames.append(self._flex3_csv)
        except Exception as e:
            bot_logger.warning(f"FLEX3 1m history {FLEX3_1M_CSV}: {e}")
        if self._live_broker():
            try:
                x = self.broker.get_candles(self.symbols[0], timeframe_minutes=1, num_candles=300)
                if x is not None and len(x) > 1:
                    x = x.iloc[:-1].copy(); x['datetime'] = pd.to_datetime(x['datetime'], utc=True); frames.append(x)
            except Exception as e:
                bot_logger.warning(f"FLEX3 1m broker bars: {e}")
        if not frames:
            return None
        d = pd.concat(frames)[['datetime', 'open', 'high', 'low', 'close', 'volume']].drop_duplicates('datetime', keep='last')
        d = d.set_index('datetime').sort_index()
        if FLEX3_BAR_LABEL == 'close':
            d.index = d.index - pd.Timedelta(minutes=1)
        now = pd.Timestamp(datetime.now(timezone.utc))
        return d[d.index <= now.floor('1min') - pd.Timedelta(minutes=1)]

    def _flex3_history_ok(self, d1: Optional[pd.DataFrame], now) -> Tuple[bool, str]:
        if d1 is None or len(d1) == 0:
            return False, 'no 1m bars'
        span = (now - d1.index[0]).days                    # calendar days covered (weekends count)
        if span < F3S.HIST_DAYS - 3:
            return False, f'only {span} days of 1m history (need ~{F3S.HIST_DAYS})'
        _et = now.tz_convert(F3.ET)
        _reopen = _et.hour == 18 and _et.minute < 5            # 18:00-18:04 ET: last bar is from before the daily/weekend halt
        if now - d1.index[-1] > pd.Timedelta(minutes=3) and not F3.in_flat_window(_et.time()) and not (
                _reopen and now - d1.index[-1] < pd.Timedelta(days=5)):   # 5d: holiday weekends (Good Friday, Thu->Mon)
            return False, f'1m bars stale (last {d1.index[-1]})'
        return True, 'ok'

    def _flex3_open_pnl(self) -> float:
        tot = 0.0
        for pos in self.positions.values():
            q = self.get_price(pos.symbol) or {}
            px = float(q.get('bid' if pos.direction == 'long' else 'ask') or q.get('last') or 0)
            if px > 0:
                pv = self._spec(pos.symbol)['point_value']
                tot += ((px - pos.entry_price) if pos.direction == 'long' else (pos.entry_price - px)) * pv * pos.size
                tot -= F3.COST_PER_CONTRACT * pos.size
        return tot

    def flex3_close_all(self, reason: str, only=None) -> int:
        n = 0
        for oid, pos in list(self.positions.items()):
            if only is not None and oid not in only:
                continue
            px = 0.0
            if self._live_broker() and hasattr(self.broker, 'flatten_symbol'):
                try:
                    res = self.broker.flatten_symbol(pos.symbol) or {}
                    if not res.get('flat'):
                        bot_logger.error(f"FLEX3 {reason}: {pos.symbol} not confirmed flat: {res}")
                        print(f"   ❌ FLEX3 {reason}: {pos.symbol} NOT confirmed flat — check broker NOW")
                        continue
                    px = float(res.get('exit_price') or res.get('avg_fill_price') or 0)
                except Exception as e:
                    bot_logger.error(f"FLEX3 {reason} flatten {pos.symbol} failed: {e}")
                    continue
            if px <= 0:
                q = self.get_price(pos.symbol) or {}
                px = float(q.get('bid' if pos.direction == 'long' else 'ask') or q.get('price') or q.get('last') or 0)
            if px <= 0:
                bot_logger.error(f"FLEX3 {reason} {oid}: no exit price — leaving for broker sync")
                continue
            print(f"⏰ FLEX3 {reason}: {oid} {pos.symbol} @ {px:.2f}")
            self.process_exit(oid, f'FLEX3_{reason.upper()}', px)
            n += 1
        return n

    def flex3_cycle(self) -> int:
        """Every loop: 16:44 flat, time exits, day-cap flatten, then new EVAL PASSER entries."""
        if not hasattr(self, '_flex3'):
            self._flex3_init()
        if not hasattr(self, '_flex3_planner'):
            self._flex3_planner = F3S.Flex3Planner(size=F3.SIZE_MNQ, **F3.PLANNER_KW)
            try:
                with open(FLEX3_STATE_FILE.replace('.json', '_planner.json')) as fh:
                    self._flex3_planner.load(json.load(fh))
            except Exception:
                pass
        now = pd.Timestamp(datetime.now(timezone.utc))
        self._flex3.new_day(F3.trading_day(now.tz_convert(F3.ET).to_pydatetime()))
        n = self.flex3_flat_check()
        due = [oid for oid, p in self.positions.items() if p.flex3_exit_at is not None and now >= pd.Timestamp(p.flex3_exit_at)]
        if due:
            n += self.flex3_close_all('time_exit', only=set(due))
        if self.positions and self._flex3.cap_reached(self._flex3_open_pnl()):
            print(f"   FLEX3: day P&L reached cap ${F3.DAY_PROFIT_CAP:.0f} (realized+open) — flattening, done for the day")
            n += self.flex3_close_all('day_cap')
            self._flex3.save(FLEX3_STATE_FILE)
        d1 = self._flex3_bars()
        try:
            self._flex3_restore_sixpm()
        except Exception as e:
            bot_logger.warning(f"FLEX3 6pm restore failed: {e}")
        if d1 is not None and len(d1):
            self._flex3_og_trail(d1)
        ok, why = self._flex3_history_ok(d1, now)
        if not ok:
            if getattr(self, '_flex3_hist_msg', '') != why:
                print(f"   FLEX3: entries paused — {why}"); self._flex3_hist_msg = why
            return n
        self._flex3_hist_msg = ''
        _open = [(p.flex3_stream, 1 if p.direction == 'long' else -1) for p in self.positions.values()]
        _its = self._flex3_planner.intents(d1, now, open_pos=_open)
        for it in _its:
            print(f"🎯 FLEX3 {it['flex3_stream']} {it['source']} {it['direction']} sl={it['sl_pts']:.2f}pts")
            self.place_order(dict(it))
        if FLEX3_STATUS and getattr(self._flex3_planner, '_Tc', None) is not None:
            try:   # read-only readout; never affects orders
                from src.strategy import flex3_status as F3ST
                _ls = F3ST.lines(self._flex3_planner, self._flex3, d1, self._flex3_planner._Tc, now, getattr(self, '_flex3_account', ''),
                                 [(p.flex3_stream, 1 if p.direction == 'long' else -1) for p in self.positions.values()], _its)
                print("\n".join(_ls))
                with open(os.getenv('FLEX3_STATUS_LOG', 'logs/flex3_status.log'), 'a', encoding='utf-8') as fh:
                    fh.write("\n".join(_ls) + "\n")
            except Exception as e:
                print(f"   FLEX3 status readout error: {e}")
        self._flex3_save_planner()
        return n

    def _flex3_record_sixpm(self, pos) -> None:
        """Remember the open 6 PM long's trail state in the planner state file so a restart can re-attach it."""
        if getattr(pos, 'flex3_stream', '') != '6pm' or not hasattr(self, '_flex3_planner'):
            return
        self._flex3_planner.six_open = dict(
            entry=float(pos.entry_price), size=int(pos.size or 1), entry_time=pd.Timestamp(pos.entry_time).isoformat(),
            exit_at=(pd.Timestamp(pos.flex3_exit_at).isoformat() if pos.flex3_exit_at is not None else None),
            trail_atr=getattr(pos, 'flex3_trail_atr', None), trail_start=getattr(pos, 'flex3_trail_start', None))

    def _flex3_restore_sixpm(self) -> None:
        """After a restart the broker position is adopted without its stream: re-attach the saved 6 PM long state
        (stream, 09:26 exit, trail distance, entry time -> high water rebuilt from 1m bars since entry)."""
        rec = getattr(getattr(self, '_flex3_planner', None), 'six_open', None)
        if not rec:
            return
        now = pd.Timestamp(datetime.now(timezone.utc))
        if rec.get('exit_at') and now >= pd.Timestamp(rec['exit_at']) + pd.Timedelta(minutes=30):
            bot_logger.info(f"FLEX3 6pm: saved trail state expired ({rec.get('exit_at')}) - dropped")
            self._flex3_planner.six_open = None; self._flex3_save_planner(); return
        if any(p.flex3_stream == '6pm' for p in self.positions.values()):
            return
        for pos in self.positions.values():
            if pos.flex3_stream or pos.direction != 'long' or pos.symbol != 'MNQ':
                continue
            if abs(float(pos.entry_price or 0) - float(rec['entry'])) > 3.0:
                continue
            pos.flex3_stream = '6pm'; pos.trailing_enabled = False
            pos.flex3_exit_at = pd.Timestamp(rec['exit_at']) if rec.get('exit_at') else None
            pos.flex3_trail_atr = rec.get('trail_atr'); pos.flex3_trail_start = rec.get('trail_start')
            try:
                pos.entry_time = pd.Timestamp(rec['entry_time']).to_pydatetime()
            except Exception:
                pass
            msg = (f"FLEX3 6pm: re-attached saved state to adopted {pos.order_id} @ {pos.entry_price:.2f} "
                   f"(trail {rec.get('trail_atr')} pts after +{rec.get('trail_start')}, exit {rec.get('exit_at')}; high water rebuilt from 1m bars)")
            print(f"   {msg}"); bot_logger.warning(msg)
            return

    def _flex3_og_trail(self, d1) -> None:
        """EVAL PASSER OG trend: trail SL to best price -/+ 4*ATR14(5m, at entry) over CLOSED 1m bars since entry (research revamp exit).
        6 PM long (SIXPM_TRAIL): same chandelier with dist = 3*ATR15 at entry, armed once high since entry >= entry + 100 pts."""
        for pos in list(self.positions.values()):
            k = getattr(pos, 'flex3_trail_atr', None)
            if pos.flex3_stream not in ('ogtrend', '6pm') or not k:
                continue
            t0 = pd.Timestamp(pos.entry_time)
            t0 = (t0.tz_localize('UTC') if t0.tzinfo is None else t0.tz_convert('UTC')).floor('1min')
            w = d1[d1.index >= t0]
            if not len(w):
                continue
            if pos.direction == 'long':
                best = max(float(pos.entry_price), float(w['high'].max())); new_sl = best - float(k)
                if pos.flex3_stream == '6pm' and best - float(pos.entry_price) < float(getattr(pos, 'flex3_trail_start', 0) or 0):
                    continue                                                     # not armed yet (needs +100 pts)
                _step = float(os.getenv('SIXPM_TRAIL_STEP_PTS', '2'))           # fewer broker cancel/replaces
                if pos.sl and self._round_to_tick(new_sl, self._spec(pos.symbol)['tick_size']) < float(pos.sl) + _step - 1e-9:
                    continue
            else:
                best = min(float(pos.entry_price), float(w['low'].min())); new_sl = best + float(k)
            new_sl = self._round_to_tick(new_sl, self._spec(pos.symbol)['tick_size'])
            if self._is_tighter_sl(pos, new_sl, pos.sl) and self._apply_stop_update(pos, new_sl):
                bot_logger.info(f"FLEX3 {pos.flex3_stream} trail {pos.order_id}: SL -> {pos.sl:.2f} (best {best:.2f})")
                if pos.flex3_stream == '6pm':
                    print(f"   6PM trail {pos.order_id}: high {best:.2f} -> SL {pos.sl:.2f} ({float(k):.1f} pts below)")

    def flex3_flat_check(self) -> int:
        """16:44-18:00 ET: market-flatten every open bot position (Lucid flat by 16:45)."""
        now = datetime.now(F3.ET)
        if not (FLEX3_ON and self.positions and F3.in_flat_window(now.time())):
            return 0
        n = 0
        for oid, pos in list(self.positions.items()):
            px = 0.0
            if self._live_broker() and hasattr(self.broker, 'flatten_symbol'):
                try:
                    res = self.broker.flatten_symbol(pos.symbol) or {}
                    if not res.get('flat'):
                        bot_logger.error(f"FLEX3 flatten {pos.symbol} not confirmed flat: {res}")
                        print(f"   ❌ FLEX3 flatten {pos.symbol} NOT confirmed flat — check broker NOW")
                        continue
                    px = float(res.get('exit_price') or res.get('avg_fill_price') or 0)
                except Exception as e:
                    bot_logger.error(f"FLEX3 flatten {pos.symbol} failed: {e}")
                    continue
            if px <= 0:
                q = self.get_price(pos.symbol) or {}
                px = float(q.get('bid' if pos.direction == 'long' else 'ask') or q.get('price') or q.get('last') or 0)
            if px <= 0:
                bot_logger.error(f"FLEX3 flatten {oid}: no exit price — leaving for broker sync")
                continue
            print(f"⏰ FLEX3 16:44 flat: {oid} {pos.symbol} @ {px:.2f}")
            self.process_exit(oid, 'FLEX3_FLAT', px)
            n += 1
        return n

    def place_order(self, signal: Dict) -> bool:
        """Place order with bracket (SL+TP)."""
        symbol = signal['symbol']
        tick_size = self._spec(symbol)['tick_size']
        direction = signal['direction']
        if FLEX3_ON and signal.get('sl_pts') is not None and signal.get('entry') is None:
            q = self.get_price(symbol) or {}
            px = float(q.get('ask' if direction == 'long' else 'bid') or q.get('last') or signal.get('ref') or 0)
            if px <= 0:
                print("   FLEX3: no live price for entry — skipped")
                return False
            dd_ = 1 if direction == 'long' else -1
            tp_pts = signal.get('tp_pts') or F3.SIX_PM_FAR_TP_PTS     # 6 PM long has no TP in research: far TP
            signal = dict(signal, entry=px, sl=self._round_to_tick(px - dd_ * float(signal['sl_pts']), tick_size),
                          tp=self._round_to_tick(px + dd_ * float(tp_pts), tick_size))
        entry = signal['entry']
        sl = signal['sl']
        tp = signal['tp']
        contracts, confidence, size_reason = self._contracts_for_entry(signal)
        flex3_stream = ''
        if FLEX3_ON:
            # PC add-ons / re-entries are separate strategies: never trade them under EVAL PASSER / FUNDED 12K;
            # (the OG main stream is already off under EVAL PASSER; A+ gated under FUNDED 12K)
            _foreign = ('addon' if signal.get('addon') else 're-entry' if int(signal.get('reentry_gen', 0) or 0) > 0 else '')
            if _foreign:
                print(f"   FLEX3 blocked {_foreign} entry (not part of {F3.__name__})")
                bot_logger.info(f"FLEX3 blocked {_foreign} {symbol} signal: {signal.get('source') or signal.get('addon') or ''}")
                return False
            ok, why, flex3_stream = self._flex3_allow(signal)
            if not ok:
                print(f"   FLEX3 blocked {flex3_stream} entry: {why}")
                bot_logger.info(f"FLEX3 blocked {flex3_stream} {symbol}: {why}")
                return False
            contracts = self._flex3.size(flex3_stream)
            size_reason = f"flex3 {flex3_stream} fixed"
        _conf_lbl = f"conf {conf_pct(confidence)}" + (
            " re-entry" if int(signal.get('reentry_gen', 0) or 0) > 0 else "")
        size_log = format_size_log(contracts, confidence, size_reason)
        bot_logger.info(size_log)
        print(f"   {size_log}")

        # Per-trade risk cap: never send an entry whose stop risk x contracts > cap.
        _cap_usd = float(getattr(self, 'max_risk_per_trade_usd', MAX_RISK_PER_TRADE_USD) or 0)
        _allowed, _rpc = cap_contracts_by_risk(
            entry, sl, self._spec(symbol).get('point_value', 0), contracts, _cap_usd
        )
        if _allowed <= 0:
            _msg = (
                f"SKIP entry {direction} {symbol}: stop risk ${_rpc:.2f}/contract x {contracts} "
                f"> MAX_RISK_PER_TRADE_USD ${_cap_usd:.0f} (entry {entry:.2f} SL {sl:.2f})"
            )
            bot_logger.warning(_msg)
            print(f"   ⛔ {_msg}")
            return False
        if _allowed < contracts:
            _msg = (
                f"Risk cap: {symbol} contracts {contracts} -> {_allowed} "
                f"(${_rpc:.2f}/contract, cap ${_cap_usd:.0f})"
            )
            bot_logger.warning(_msg)
            print(f"   ⚖️ {_msg}")
            contracts = _allowed
            size_log = f"{size_log} risk_cap->{contracts}"
        
        if self.paper_mode or self.paper_orders:
            order_id = f"paper_{int(time.time())}"
            tag = "PAPER-ORDERS" if self.paper_orders else "PAPER"
            side = "LONG" if direction == 'long' else "SHORT"
            print(f"📝 [{tag}] ENTRY {side} {contracts} {symbol} @ {entry:.2f} {_conf_lbl}")
            if signal.get('addon'):
                print(f"   {_addons.entry_line(signal, contracts, entry)}")
            print(f"   SL: {sl:.2f} | TP: {tp:.2f}")
            
            trail_on = bool(signal.get('trailing_enabled', True))
            pos = Position(
                order_id=order_id,
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                size=contracts,
                sl=sl,
                tp=tp,
                entry_time=datetime.now(timezone.utc),
                trailing_enabled=trail_on,
                ensemble_signal=str(signal.get('ensemble_signal', '')).upper() or (
                    'BUY' if direction == 'long' else 'SELL'
                ),
                model_signals=signal.get('model_signals'),
                regime=str(signal.get('regime', 'unknown')),
                exit_sleeve=str(signal.get('exit_sleeve', '') or ''),
                confidence=float(confidence or 0.0),
                reentry_gen=int(signal.get('reentry_gen', 0) or 0),
                reentry_parent=str(signal.get('reentry_parent', '') or ''),
                addon=str(signal.get('addon', '') or ''),
                addon_exit_at=signal.get('addon_exit_at'),
            )
            pos.flex3_stream = flex3_stream
            pos.flex3_exit_at = signal.get('flex3_exit_at')
            pos.flex3_trail_atr = signal.get('trail_atr')
            pos.flex3_trail_start = signal.get('trail_start_pts')
            self.positions[order_id] = pos
            F3EV.emit('entry', order_id=order_id, symbol=symbol, direction=direction, size=contracts, entry=pos.entry_price,
                      sl=pos.sl, tp=pos.tp, stream=flex3_stream, source=signal.get('source', ''), td=signal.get('td', ''))
            if FLEX3_ON and hasattr(self, '_flex3'):
                self._flex3.on_entry(flex3_stream); self._flex3.save(FLEX3_STATE_FILE)
            if FLEX3_ON and hasattr(self, '_flex3_planner') and signal.get('td'):
                self._flex3_planner.on_entry(signal)
                self._flex3_record_sixpm(pos)
                self._flex3_save_planner()
            self._write_board_status(note=f"{len(self.positions)}/{self.max_positions} after fill")
            print(f"   Active positions: {len(self.positions)}/{self.max_positions}")
            if REVAMP_EXITS_ON:
                print(f"   REVAMP sleeve={pos.exit_sleeve or '?'} trail={'ON' if trail_on else 'OFF (fixed TP)'}")
            return True
        
        # Real order
        order_type = 'buy' if direction == 'long' else 'sell'
        result = self.broker.place_order(
            symbol=symbol,
            order_type=order_type,
            size=contracts,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp
        )
        
        if result:
            order_id = result.get('ticket', str(int(time.time())))
            want_trail = bool(signal.get('trailing_enabled', True))
            # Lucid protective STOP-LIMIT sets supports_stop_modify=False but is still trailable.
            stop_updatable = self._broker_can_update_stop(result)
            pos = Position(
                order_id=order_id,
                symbol=symbol,
                direction=direction,
                entry_price=result.get('entry_price', entry),
                size=contracts,
                sl=sl,
                tp=tp,
                entry_time=datetime.now(timezone.utc),
                trailing_enabled=want_trail and stop_updatable,
                ensemble_signal=str(signal.get('ensemble_signal', '')).upper() or (
                    'BUY' if direction == 'long' else 'SELL'
                ),
                model_signals=signal.get('model_signals'),
                regime=str(signal.get('regime', 'unknown')),
                exit_sleeve=str(signal.get('exit_sleeve', '') or ''),
                confidence=float(confidence or 0.0),
                reentry_gen=int(signal.get('reentry_gen', 0) or 0),
                reentry_parent=str(signal.get('reentry_parent', '') or ''),
                addon=str(signal.get('addon', '') or ''),
                addon_exit_at=signal.get('addon_exit_at'),
            )
            pos.flex3_stream = flex3_stream
            pos.flex3_exit_at = signal.get('flex3_exit_at')
            pos.flex3_trail_atr = signal.get('trail_atr')
            pos.flex3_trail_start = signal.get('trail_start_pts')
            self.positions[order_id] = pos
            F3EV.emit('entry', order_id=order_id, symbol=symbol, direction=direction, size=contracts, entry=pos.entry_price,
                      sl=pos.sl, tp=pos.tp, stream=flex3_stream, source=signal.get('source', ''), td=signal.get('td', ''))
            if FLEX3_ON and hasattr(self, '_flex3'):
                self._flex3.on_entry(flex3_stream); self._flex3.save(FLEX3_STATE_FILE)
            if FLEX3_ON and hasattr(self, '_flex3_planner') and signal.get('td'):
                self._flex3_planner.on_entry(signal)
                self._flex3_record_sixpm(pos)
                self._flex3_save_planner()
            self._write_board_status(note=f"{len(self.positions)}/{self.max_positions} after fill")
            print(f"ENTRY {direction.upper()} {contracts} {symbol} @ {pos.entry_price:.2f} {_conf_lbl}")
            if signal.get('addon'):
                print(f"   {_addons.entry_line(signal, contracts, float(pos.entry_price))}")
                bot_logger.info(f"ADDON ENTRY {signal['addon']}: {_addons.entry_line(signal, contracts, float(pos.entry_price))}")
            print(f"   SL: {sl:.2f} | TP: {tp:.2f}")
            print(f"   Active positions: {len(self.positions)}/{self.max_positions}")
            if REVAMP_EXITS_ON:
                print(f"   REVAMP sleeve={pos.exit_sleeve or '?'} trail={'ON' if pos.trailing_enabled else 'OFF (fixed TP)'}")
            if want_trail and not stop_updatable:
                print(f"   WARNING: Broker stop missing on entry - trailing disabled for {order_id}")
                bot_logger.error(
                    f"Order {order_id} for {symbol} filled without broker stop support; trailing is disabled"
                )
            trades_logger.info(f"ENTRY {direction} {symbol} @ {entry:.2f} SL={sl:.2f} TP={tp:.2f} {_conf_lbl} {size_log}")
            return True
        else:
            print(f"❌ Order failed!")
            return False
    
    # ── Backtested add-on setups (src/core/addon_setups.py, docs/ADDON_SPECS_20261010.md) ──

    def _addon_frame(self, symbol: str) -> Optional[pd.DataFrame]:
        """Deep 5m frame for the add-ons (overnight low needs bars back to 6 PM ET).

        Live: its own Rithmic history request with end_time set, so the 30 s candle cache
        (which may hold the Ensemble's 100-bar frame) is bypassed. Returns closed bars,
        indexed by bar START (ET), with add-on features.
        """
        n = _addons.candle_count()
        df = None
        now = datetime.now(timezone.utc)
        try:
            if self.broker is not None and hasattr(self.broker, 'get_candles'):
                df = self.broker.get_candles(symbol, timeframe_minutes=5, num_candles=n, end_time=now)
            if df is None or len(df) < _addons.WARMUP_BARS:
                alt = self.get_candles(symbol=symbol, count=n)
                if alt is not None and (df is None or len(alt) > len(df)):
                    df = alt
        except Exception as e:
            bot_logger.warning(f"ADDON {symbol}: candle fetch failed: {e}")
        bars = _addons.closed_bars(df, now=now) if df is not None else None
        if bars is None or len(bars) == 0:
            return None
        return _addons.add_features(bars)

    def check_addon_signals(self, symbol: str, main_signal: Optional[Dict] = None,
                            now: Optional[datetime] = None) -> Tuple[Optional[Dict], List[str]]:
        """Evaluate the add-ons on the newest closed bar. Returns (signal or None, status lines).

        Fires only when the bot has NO open trade (backtest: flat only), the session limits
        allow new entries, and the setup's own window/rules are met. Not blocked by the
        bad-hour / Friday filter (own windows). Errors never reach trading.
        """
        lines: List[str] = []
        try:
            kinds = [k for k in _addons.in_display_window(now) if k in ('onl', 'e1')]
            if not kinds:
                return None, lines
            frame = self._addon_frame(symbol)
            fired = self.__dict__.setdefault('_addon_fired', set())
            flat = (not self.positions) and self._effective_open_count() == 0
            block = self._session_limit_block()
            sig_out = None
            for kind in kinds:
                ev = _addons.evaluate(frame, kind, now=now)
                line = ev.status
                if ev.signal:
                    key = (kind, str(ev.bar_start))
                    if key in fired:
                        line += " - already handled"
                    elif not flat:
                        fired.add(key)
                        line += " - SKIPPED: bot already has an open trade (add-on is flat-only)"
                        bot_logger.info(f"ADDON {kind} skipped (not flat) bar {ev.bar_start}")
                    elif block:
                        fired.add(key)
                        line += f" - SKIPPED: {block}"
                    elif main_signal:
                        fired.add(key)
                        line += " - SKIPPED: main bot signal this bar takes the slot"
                    elif sig_out is None:
                        fired.add(key)
                        sig_out = _addons.make_signal(ev, symbol, float(self._spec(symbol)['tick_size']))
                        line += f" -> LONG 1 {symbol}"
                        bot_logger.info(f"ADDON SIGNAL {kind} {symbol} bar {ev.bar_start} close {ev.close:.2f} "
                                        f"ATR {ev.atr:.2f} SL {sig_out['sl']:.2f} TP {sig_out['tp']:.2f}")
                lines.append(line)
            return sig_out, lines
        except Exception as e:
            bot_logger.warning(f"ADDON check error for {symbol}: {e!r}")
            return None, lines

    def check_addon_time_exits(self, symbol: str, df: Optional[pd.DataFrame] = None,
                               now: Optional[datetime] = None) -> int:
        """Close add-on trades at their 2h time limit (backtest: 24 x 5m bars after the signal bar).

        Live: market-flatten only when the add-on is the ONLY thing open on the symbol
        (one tracked trade, broker net = its size); otherwise keep its SL/TP working and say so.
        """
        now = now or datetime.now(timezone.utc)
        done = 0
        for oid, pos in list(self.positions.items()):
            if pos.symbol != symbol or not getattr(pos, 'addon', '') or not getattr(pos, 'addon_exit_at', None):
                continue
            ex = pos.addon_exit_at
            if ex.tzinfo is None:
                ex = ex.replace(tzinfo=timezone.utc)
            if now < ex:
                continue
            px = 0.0
            try:
                q = self.get_price(symbol) or {}
                px = float(q.get('bid') or q.get('last') or 0)
            except Exception:
                px = 0.0
            if px <= 0 and df is not None and len(df):
                px = float(df.iloc[-1]['close'])
            if not self._live_broker():
                self.process_exit(oid, 'TIME_LIMIT', px or pos.entry_price)
                done += 1
                continue
            others = [p for k, p in self.positions.items() if k != oid and p.symbol == symbol]
            net = self._broker_net_qty(symbol)
            if others or net is None or int(net) != int(pos.size):
                warned = self.__dict__.setdefault('_addon_time_warned', set())
                if oid not in warned:
                    warned.add(oid)
                    msg = (f"{oid} add-on 2h time limit reached, but other trades/contracts are open on {symbol} "
                           f"(net {net}) - keeping its SL/TP working instead of flattening")
                    print(f"   ⏱️ {msg}")
                    bot_logger.warning(msg)
                continue
            res = {}
            try:
                res = self.broker.flatten_symbol(symbol) or {}
            except Exception as e:
                bot_logger.warning(f"ADDON time exit flatten failed for {oid}: {e!r}")
            if res.get('flat'):
                print(f"   ⏱️ {oid} add-on 2h time limit: closed at market")
                self.process_exit(oid, 'TIME_LIMIT', px or pos.entry_price)
                done += 1
            else:
                bot_logger.warning(f"ADDON time exit for {oid}: flatten not confirmed ({res}) - SL/TP stay, retry next bar")
        return done

    def check_position_exit(self, symbol: str, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Check if any positions hit SL/TP (for paper mode mainly). Returns list of (order_id, exit_type)."""
        exits = []
        if not self.positions:
            return exits
        
        row = df.iloc[-1]
        
        for order_id, pos in list(self.positions.items()):
            if pos.symbol != symbol:
                continue
            d = pos.direction
            
            if d == 'long':
                if row['low'] <= pos.sl:
                    exits.append((order_id, 'STOP_LOSS'))
                elif (
                    not getattr(pos, 'tp_cleared', False)
                    and pos.tp and pos.tp > 0
                    and row['high'] >= pos.tp
                ):
                    exits.append((order_id, 'TAKE_PROFIT'))
            else:
                if row['high'] >= pos.sl:
                    exits.append((order_id, 'STOP_LOSS'))
                elif (
                    not getattr(pos, 'tp_cleared', False)
                    and pos.tp and pos.tp > 0
                    and row['low'] <= pos.tp
                ):
                    exits.append((order_id, 'TAKE_PROFIT'))
        
        return exits
    
    def process_exit(self, order_id: str, exit_type: str, exit_price: float):
        """Process position exit.

        Idempotent per order id: P&L is booked at most once, and the position is
        removed from tracking BEFORE learner/loss bookkeeping runs, so an error in
        that bookkeeping can never make the next sync book the same exit again.
        """
        if order_id not in self.positions:
            return
        done = self.__dict__.setdefault('_exited_order_keys', set())
        _p = self.positions[order_id]
        key = (order_id, str(_p.entry_time), float(_p.entry_price or 0), int(_p.size or 0))
        if key in done:
            # Same trade already booked: just drop it from tracking, no P&L again.
            self.positions.pop(order_id, None)
            bot_logger.warning(f"Duplicate exit for {order_id} ignored (P&L already recorded)")
            return
        
        pos = self.positions.pop(order_id)
        done.add(key)
        spec = self._spec(pos.symbol)
        d = pos.direction
        if d == 'long':
            raw_pnl = (exit_price - pos.entry_price) * spec['point_value'] * pos.size
        else:
            raw_pnl = (pos.entry_price - exit_price) * spec['point_value'] * pos.size
        
        # Subtract costs (MNQ/MES: $2 + 1 tick flat; NQ: per contract, see round_trip_cost)
        pnl = raw_pnl - round_trip_cost(pos.symbol, pos.size, spec['point_value'], spec['tick_size'])
        if FLEX3_ON:
            pnl = raw_pnl - F3.COST_PER_CONTRACT * pos.size   # same as backtest: $4/contract RT ($12 at 3 MNQ)
        
        # Record trade
        trade = {
            'symbol': pos.symbol,
            'direction': d,
            'entry_time': str(pos.entry_time),
            'exit_time': str(datetime.now(timezone.utc)),
            'entry_price': pos.entry_price,
            'exit_price': exit_price,
            'exit_type': exit_type,
            'pnl': pnl
        }
        self.trades.append(trade)
        F3EV.emit('exit', order_id=order_id, symbol=pos.symbol, direction=d, size=pos.size, entry=pos.entry_price,
                  exit=exit_price, reason=exit_type, pnl=pnl, stream=getattr(pos, 'flex3_stream', ''))
        
        self.daily_pnl += pnl
        self.daily_trades += 1
        if FLEX3_ON:
            self._flex3_on_close(pnl)
            if pos.flex3_stream and hasattr(self, '_flex3_planner'):
                self._flex3_planner.on_close(pos.flex3_stream, pnl, pos.size, pd.Timestamp(pos.entry_time))
                self._flex3_save_planner()
        _led = getattr(self, '_session_ledger', None)
        if _led is not None:
            try:  # counted in the session the trade was OPENED in
                _led.book(pos.entry_time, pnl, order_id)
            except Exception as e:  # bookkeeping must never break an exit
                bot_logger.warning(f"Session ledger update failed for {order_id}: {e!r}")
        
        emoji = "🟢" if pnl > 0 else "🔴"
        print(f"{emoji} EXIT {exit_type}: {d.upper()} @ {exit_price:.2f}")
        print(f"   PnL: ${pnl:.2f} | Daily: ${self.daily_pnl:.2f}")
        trades_logger.info(f"EXIT {exit_type} {d} {pos.symbol} @ {exit_price:.2f} PnL=${pnl:.2f}")

        self.cooldown = COOLDOWN_BARS_AFTER_WIN if pnl > 0 else COOLDOWN_BARS_AFTER_LOSS

        try:
            self._update_loss_protection(pos.symbol, pnl)
        except Exception as e:  # bookkeeping must never break an exit
            bot_logger.warning(f"Loss-protection update failed for {order_id}: {e!r}")
        try:
            self._feed_learner_on_exit(pos, pnl, exit_type, exit_price)
        except Exception as e:  # learner must never break an exit
            bot_logger.warning(f"Learner update failed for {order_id}: {e!r}")
        
        try:
            self._maybe_arm_reentry(pos, exit_type, pnl, exit_price)
        except Exception as e:  # re-entry bookkeeping must never break an exit
            bot_logger.warning(f"Re-entry arm check failed for {order_id}: {e}")

        try:
            self._cancel_orders_for_closed_position(order_id, pos.symbol, f"exit_{exit_type.lower()}")
        except Exception as e:
            bot_logger.warning(f"Post-exit order cleanup failed for {order_id}: {e!r}")
        print(f"   Remaining positions: {len(self.positions)}/{self.max_positions}")

    # ── Re-entry after a profitable lock/trail stop exit (src/core/reentry.py) ──
    # Bot-side trigger + MARKET entry: no resting broker stop order, so the orphan
    # sweep never sees a re-entry order until it is a normal tracked trade.

    def _maybe_arm_reentry(self, pos: 'Position', exit_type: str, pnl: float, exit_price: float) -> bool:
        if FLEX3_ON or getattr(pos, 'addon', ''):
            return False  # add-on / Flex3 trades never arm a re-entry (not in the backtest)
        ok, why = _reentry.should_arm(pos, exit_type, pnl, exit_price)
        if not ok:
            if _reentry.REENTRY_ENABLED and str(exit_type).upper() == 'STOP_LOSS' and pnl > 0:
                bot_logger.info(f"Re-entry not armed for {pos.order_id}: {why}")
            return False
        now = datetime.now(timezone.utc)
        if not _reentry.in_entry_window_et(now):
            print("   🔁 Re-entry not armed: outside 09:30-15:55 ET")
            return False
        tick = float(self._spec(pos.symbol)['tick_size'])
        df = None
        try:
            df = self.get_candles(symbol=pos.symbol, count=60)
        except Exception:
            df = None
        peak = _reentry.parent_peak_price(pos, df, exit_time=now)
        arm = _reentry.make_arm(pos, peak, tick, now, float(getattr(pos, 'confidence', 0.0) or 0.0))
        self.__dict__.setdefault('_reentry_arms', {})[arm.parent_id] = arm
        print(f"   🔁 {arm.status_text(now)} (after {pos.order_id} stop exit +${pnl:.2f}, peak {peak:.2f})")
        bot_logger.info(
            f"REENTRY ARMED {pos.symbol} {arm.side} trigger={arm.trigger:.2f} peak={peak:.2f} "
            f"parent={pos.order_id} until={arm.expires_at.isoformat()} conf={arm.confidence:.2f} gen={arm.gen}"
        )
        return True

    def reentry_status_lines(self, symbol: str) -> List[str]:
        """'Re-entry armed: BUY above 31100.25 for 55 min' lines for the status print."""
        if not _reentry.REENTRY_ENABLED:
            return []
        now = datetime.now(timezone.utc)
        return [
            a.status_text(now)
            for a in (self.__dict__.get('_reentry_arms') or {}).values()
            if a.symbol == symbol and now < a.expires_at
        ]

    def _session_limit_block(self) -> Optional[str]:
        """Reason new entries are blocked by the session loss limits / daily profit target, else None."""
        led = getattr(self, '_session_ledger', None)
        if led is None:
            return None
        try:
            why = led.block_reason(datetime.now(timezone.utc))
        except Exception as e:  # a ledger problem must never stop trade management
            bot_logger.warning(f"Session-limit check failed: {e!r}")
            return None
        if why != getattr(self, '_last_session_block', None):
            self._last_session_block = why
            if why:
                print(f"   🛑 New trades paused: {why} (open trades keep their SL/TP)")
                bot_logger.warning(f"SESSION LIMIT: {why}")
            else:
                bot_logger.info("SESSION LIMIT cleared (new session / trading day)")
        return why

    def _day_total_text(self, now: Optional[datetime] = None) -> str:
        """Status-line P&L. Session limits on: trading-day total from the saved ledger
        (survives restarts, overnight + day session). Off: old in-memory since-start value."""
        led = getattr(self, '_session_ledger', None)
        if led is not None:
            try:
                _, td = _session_limits.session_of(now or datetime.now(timezone.utc))
                return f"Day total: ${led.pnl(td):.2f}"
            except Exception:
                pass
        return f"Daily: ${self.daily_pnl:.2f}"

    def _overnight_sleeve_status_line(self, now: Optional[datetime] = None) -> Optional[str]:
        """REVAMP only: plain overnight line, e.g. 'Overnight: trend trades only (breakout skipped)'."""
        if not REVAMP_EXITS_ON or revamp_overnight_sleeves() == 'all':
            return None
        if _session_limits.session_of(now or datetime.now(timezone.utc))[0] != 'overnight':
            return None
        skip = getattr(self, '_ovn_sleeve_skip', None)
        if skip and time.time() - float(skip[1]) <= 90:
            return f"Overnight: trend trades only ({_OVN_SLEEVE_LABEL.get(skip[0], skip[0])} skipped)"
        return "Overnight: trend trades only (breakout and mean-reversion skipped until 9:30 AM ET)"

    def _session_status_line(self) -> Optional[str]:
        led = getattr(self, '_session_ledger', None)
        if led is None:
            return None
        try:
            return led.status_line(datetime.now(timezone.utc))
        except Exception:
            return None

    def _extra_mode(self) -> str:
        """same_dir_profit (revamp default) or conf (old EXTRA_TRADE_MIN_CONF rule)."""
        return _extra_rule.extra_trade_mode(REVAMP_EXITS_ON)

    def _extra_point_value(self, symbol: Optional[str] = None) -> float:
        sym = symbol or (self.symbols[0] if getattr(self, 'symbols', None) else 'MNQ')
        try:
            return float(self._spec(sym).get('point_value', 2.0) or 2.0)
        except Exception:
            return 2.0

    def _extra_trade_block(self, direction: str, price: float, confidence: Any,
                           symbol: Optional[str] = None) -> Optional[str]:
        """None if a 2nd/3rd trade may open now, else the reason (plain words)."""
        open_n = self._effective_open_count()
        if open_n < 1:
            return None
        if self._extra_mode() == _extra_rule.SAME_DIR_PROFIT:
            return _extra_rule.check_same_dir_profit(
                list(self.positions.values()), open_n, direction, float(price or 0.0),
                self._extra_point_value(symbol))
        min_extra = float((getattr(self, '_size_cfg', None) or SIZE_SCALE_CFG).get('extra_trade_min_conf', 0.72))
        if float(confidence or 0.0) < min_extra:
            return f"conf {conf_pct(confidence)}, needs {conf_pct(min_extra)}"
        return None

    def _extra_trade_status(self, checked: bool, signal: Optional[Dict],
                            price: Optional[float] = None) -> Optional[str]:
        """Status for the next 2nd/3rd trade while 1..max-1 trades are open.

        same_dir_profit: '2nd trade: needs trade 1 in profit + same direction (now: down $40)'
                         '2nd trade: allowed, LONG setups only (now: up $25)'
        conf (old):      '2nd trade: conf 66%, needs 70%'
        """
        min_extra = float((getattr(self, '_size_cfg', None) or SIZE_SCALE_CFG).get('extra_trade_min_conf', 0.72))
        note = ""
        if not checked and self.cooldown > 0:
            note = f"cooldown {self.cooldown} bars"
        if self._extra_mode() == _extra_rule.SAME_DIR_PROFIT:
            return _extra_rule.status_line(
                list(self.positions.values()), self._effective_open_count(), self.max_positions,
                price, self._extra_point_value(), signal, note)
        cand = float(signal.get('confidence', 0.0) or 0.0) if signal else None
        return extra_trade_line(self._effective_open_count(), self.max_positions, min_extra, cand, note)

    def _reentry_block_reason(self, arm, price: Optional[float] = None) -> Optional[str]:
        """Same risk gates as a normal entry, minus the post-exit cooldown."""
        if getattr(self, '_session_ledger', None) is None and self.daily_pnl <= -self.daily_loss_limit:
            return f"daily loss cap (${self.daily_pnl:.2f})"
        _slb = self._session_limit_block()
        if _slb:
            return _slb
        if self.max_trades_per_day > 0 and self.daily_trades >= self.max_trades_per_day:
            return f"max trades/day ({self.daily_trades})"
        open_n = self._effective_open_count()
        if open_n >= self.max_positions:
            return f"max concurrent trades ({open_n}/{self.max_positions})"
        if self._symbol_in_consec_loss_pause(arm.symbol):
            return "consecutive-loss pause"
        if not self._is_trading_session_allowed(arm.symbol):
            return "session filter"
        if open_n >= 1 and self._extra_mode() == _extra_rule.SAME_DIR_PROFIT:
            px = float(price or 0.0) or float(getattr(arm, 'trigger', 0.0) or 0.0)
            why = _extra_rule.check_same_dir_profit(
                list(self.positions.values()), open_n, arm.direction, px, self._extra_point_value(arm.symbol))
            return f"extra trade: {why} ({open_n} open)" if why else None
        min_extra = float((getattr(self, '_size_cfg', None) or SIZE_SCALE_CFG).get('extra_trade_min_conf', 0.72))
        if open_n >= 1 and float(arm.confidence or 0.0) < min_extra:
            return f"extra trade needs conf {conf_pct(min_extra)}, parent had {conf_pct(arm.confidence)} ({open_n} open)"
        return None

    def check_reentry_triggers(self) -> int:
        """Called every loop (~5-10 s). Fires a market re-entry when price breaks the parent's peak."""
        arms = self.__dict__.get('_reentry_arms') or {}
        if not _reentry.REENTRY_ENABLED or not arms:
            return 0
        placed = 0
        now = datetime.now(timezone.utc)
        for pid, arm in list(arms.items()):
            try:
                if now >= arm.expires_at:
                    arms.pop(pid, None)
                    print(f"   🔁 Re-entry expired: {arm.side} {arm.symbol} {arm.trigger:.2f} not reached "
                          f"within {_reentry.REENTRY_WINDOW_BARS} bars")
                    bot_logger.info(f"REENTRY EXPIRED parent={pid} trigger={arm.trigger:.2f}")
                    continue
                if not _reentry.in_entry_window_et(now):
                    arms.pop(pid, None)
                    print(f"   🔁 Re-entry dropped: {arm.side} {arm.symbol} {arm.trigger:.2f} (outside 09:30-15:55 ET)")
                    bot_logger.info(f"REENTRY DROPPED parent={pid} outside entry window")
                    continue
                q = self.get_price(arm.symbol) or {}
                last = float(q.get('last') or 0) or 0.0
                if last <= 0:
                    last = float((q.get('ask') if arm.direction == 'long' else q.get('bid')) or 0) or 0.0
                if not _reentry.triggered(arm, last):
                    continue
                arms.pop(pid, None)  # the first break consumes the arm (as in the backtest)
                block = self._reentry_block_reason(arm, last)
                if block:
                    print(f"   🔁 Re-entry {arm.side} {arm.symbol}: {last:.2f} broke {arm.trigger:.2f} but skipped - {block}")
                    bot_logger.info(f"REENTRY SKIPPED parent={pid} price={last:.2f} reason={block}")
                    continue
                spec = self._spec(arm.symbol)
                tick = float(spec['tick_size'])
                atr = float('nan')
                done = None
                df = self.get_candles(symbol=arm.symbol, count=60)
                if df is not None and len(df):
                    df = self.calculate_indicators(df)
                    done = _reentry.completed_bars(df, now)
                    if done is not None and len(done) and 'atr' in done.columns:
                        atr = float(done['atr'].iloc[-1])
                cap = float(getattr(self, 'max_risk_per_trade_usd', MAX_RISK_PER_TRADE_USD) or 0)
                lv = _reentry.compute_levels(arm.direction, last, done, atr, tick, float(spec['point_value']), cap)
                if not lv:
                    print(f"   🔁 Re-entry {arm.side} {arm.symbol} skipped: could not build a stop (no bars/ATR)")
                    bot_logger.warning(f"REENTRY SKIPPED parent={pid}: no stop levels")
                    continue
                sl, tp, dist = lv
                signal = {
                    'symbol': arm.symbol,
                    'direction': arm.direction,
                    'entry': float(last),
                    'sl': sl,
                    'tp': tp,
                    'atr': atr if atr == atr else 0.0,
                    'confidence': float(arm.confidence or 0.0),
                    'ensemble_signal': arm.ensemble_signal,
                    'model_signals': {'revamp_sleeve': arm.sleeve, 'reentry_of': pid},
                    'regime': arm.regime,
                    'trailing_enabled': False,
                    'exit_sleeve': arm.sleeve,
                    'reentry_gen': int(arm.gen),
                    'reentry_parent': pid,
                }
                print(f"🔁 RE-ENTRY {arm.side} {arm.symbol}: {last:.2f} broke {arm.trigger:.2f} (parent {pid}) | "
                      f"SL {sl:.2f} ({dist:.2f} pts) | TP {tp:.2f} ({_reentry.REENTRY_TARGET_R:g}R)")
                bot_logger.info(f"REENTRY FIRE parent={pid} {arm.direction} {arm.symbol} @~{last:.2f} SL={sl:.2f} TP={tp:.2f}")
                if self.place_order(signal):
                    placed += 1
                    trades_logger.info(
                        f"REENTRY {arm.direction} {arm.symbol} parent={pid} trigger={arm.trigger:.2f} "
                        f"SL={sl:.2f} TP={tp:.2f}"
                    )
            except Exception as e:
                bot_logger.warning(f"Re-entry check failed for {pid}: {e}")
        return placed

    # ── Working-order hygiene (no orphan SL/TP lines on the chart) ──────────

    def _live_broker(self) -> bool:
        return bool(self.broker) and not (self.paper_mode or self.paper_orders)

    def _live_entry_ids(self, symbol: str, exclude: Optional[set] = None) -> set:
        skip = set(exclude or ())
        return {
            oid for oid, p in self.positions.items()
            if p.symbol == symbol and oid not in skip
        }

    def _cancel_orders_for_closed_position(self, order_id: str, symbol: str, reason: str) -> int:
        """Cancel every working order of a closed trade, then sweep symbol orphans.

        Other open trades on the same symbol (max concurrent 2) keep their legs.
        """
        if not self._live_broker():
            return 0
        total = 0
        try:
            if hasattr(self.broker, 'release_closed_entry'):
                total += int(self.broker.release_closed_entry(order_id, symbol, reason=reason) or 0)
            elif hasattr(self.broker, 'cancel_entry_bracket_legs'):
                total += int(self.broker.cancel_entry_bracket_legs(order_id, symbol, reason=reason) or 0)
            if hasattr(self.broker, 'sweep_orphan_working_orders'):
                res = self.broker.sweep_orphan_working_orders(
                    symbol, self._live_entry_ids(symbol, exclude={order_id}), reason=reason,
                ) or {}
                total += int(res.get('cancelled') or 0)
            self._last_orphan_sweep = time.time()
        except Exception as e:
            bot_logger.warning(f"Working-order cleanup failed for {order_id} {symbol} ({reason}): {e}")
        if total:
            bot_logger.info(f"🧹 {symbol} {order_id}: cancelled {total} working order(s) after close ({reason})")
            print(f"   🧹 Cancelled {total} leftover working order(s) for {order_id}")
        return total

    def sweep_orphan_orders(self, force: bool = False, reason: str = 'sync_cycle') -> int:
        """Cancel working bot orders on the chart that no tracked open trade owns."""
        if not self._live_broker() or not hasattr(self.broker, 'sweep_orphan_working_orders'):
            return 0
        now = time.time()
        if not force and now - self._last_orphan_sweep < ORPHAN_SWEEP_SEC:
            return 0
        self._last_orphan_sweep = now
        total = 0
        for sym in self.symbols:
            try:
                res = self.broker.sweep_orphan_working_orders(
                    sym, self._live_entry_ids(sym), reason=reason,
                ) or {}
                total += int(res.get('cancelled') or 0)
            except Exception as e:
                bot_logger.warning(f"Orphan sweep failed for {sym}: {e}")
        if total:
            print(f"🧹 Orphan sweep: cancelled {total} working order(s) not tied to an open trade")
        return total

    def cleanup_orders_on_shutdown(self) -> None:
        """On exit: never leave orphan working orders behind.

        Default keeps SL/TP of trades that are still open (a naked position is
        worse than a bracket line). SHUTDOWN_FLATTEN=1 flattens bot positions
        first, then everything bot-owned is cancelled.
        """
        if not self._live_broker():
            return
        try:
            if not getattr(self.broker, 'connected', True):
                return
        except Exception:
            pass
        if SHUTDOWN_FLATTEN and hasattr(self.broker, 'flatten_symbol'):
            for sym in self.symbols:
                try:
                    res = self.broker.flatten_symbol(sym) or {}
                    print(f"⏹️  Shutdown flatten {sym}: flat={res.get('flat')} cancelled={res.get('cancelled_bot', 0)}")
                    if res.get('flat'):
                        for oid in list(self._live_entry_ids(sym)):
                            pos = self.positions.pop(oid, None)
                            if pos and hasattr(self.broker, 'release_closed_entry'):
                                self.broker.release_closed_entry(oid, sym, reason='shutdown_flatten')
                except Exception as e:
                    bot_logger.warning(f"Shutdown flatten failed for {sym}: {e}")
        n = self.sweep_orphan_orders(force=True, reason='shutdown')
        time.sleep(2)  # second pass: connector requires a stable size check twice
        n += self.sweep_orphan_orders(force=True, reason='shutdown')
        still_open = len(self.positions)
        print(f"⏹️  Shutdown order cleanup: cancelled {n} orphan working order(s); "
              f"{still_open} open trade(s) keep their SL/TP")

    def _confirmed_live_exit(self, order_id: str, exit_type: str) -> Optional[Tuple[str, float]]:
        """Bar touched SL/TP: only close locally once the broker confirms the leg fill.

        Dropping the trade on a bar touch alone (old behaviour) lost track of a
        still-open position and left its brackets unmanaged.
        """
        pos = self.positions.get(order_id)
        if not pos:
            return None
        fill = None
        if hasattr(self.broker, 'confirm_bracket_exit_fill'):
            try:
                fill = self.broker.confirm_bracket_exit_fill(
                    order_id, pos.symbol, pos.sl, pos.tp, entry_price=pos.entry_price,
                )
            except Exception as e:
                bot_logger.warning(f"confirm_bracket_exit_fill failed for {order_id}: {e}")
                fill = None
        if not fill or not fill.get('confirmed'):
            if order_id not in self._unconfirmed_exit_logged:
                self._unconfirmed_exit_logged.add(order_id)
                print(f"   ⏳ {order_id}: bar hit {exit_type} but broker fill not confirmed — keeping SL/TP working")
            return None
        leg = str(fill.get('leg') or '').upper()
        et = 'TAKE_PROFIT' if leg == 'TP' else 'STOP_LOSS' if leg == 'SL' else exit_type
        px = float(fill.get('exit_price') or 0) or (pos.tp if et == 'TAKE_PROFIT' else pos.sl)
        return et, px
    
    def _broker_open_tickets(self) -> Dict[str, Dict[str, Any]]:
        """Map ticket -> position dict from connector (includes entry-grace fills)."""
        tickets: Dict[str, Dict[str, Any]] = {}
        if not self.broker:
            return tickets
        for sym in self.symbols:
            try:
                for pos in (self.broker.get_open_positions(symbol=sym) or []):
                    ticket = str(pos.get('ticket') or '').strip()
                    if ticket:
                        tickets[ticket] = pos
            except Exception as e:
                bot_logger.warning(f"get_open_positions({sym}) failed: {e}")
        return tickets

    def _broker_net_qty(self, symbol: str) -> Optional[int]:
        """Absolute broker net contracts for symbol, or None if unknown."""
        if not self.broker:
            return None
        try:
            # Prefer live PnL plant snapshot when connector exposes it.
            if hasattr(self.broker, '_positions'):
                with self.broker._state_lock:  # type: ignore[attr-defined]
                    raw = self.broker._positions.get(symbol)  # type: ignore[attr-defined]
                if raw is not None:
                    return abs(int(raw.get('size', 0) or 0))
            positions = self.broker.get_open_positions(symbol=symbol) or []
            # If only grace placeholders (broker_net_known False), treat as unknown
            known = [p for p in positions if p.get('broker_net_known')]
            if known:
                return sum(abs(int(p.get('size', 1) or 1)) for p in known)
            if positions:
                return None  # grace-only: do not claim flat
            # Empty after refresh: could be truly flat OR PnL plant lag.
            return 0
        except Exception as e:
            bot_logger.warning(f"broker net qty check failed for {symbol}: {e}")
            return None

    def _effective_open_count(self) -> int:
        """Open-count for entry gating: never under-count vs broker when known."""
        local = len(self.positions)
        if self.paper_mode or self.paper_orders or not self.broker:
            return local
        broker_tickets = self._broker_open_tickets()
        broker_count = len(broker_tickets)
        # Also fold in absolute net contracts when larger (aggregate positions).
        net_total = 0
        net_known = False
        for sym in self.symbols:
            net = self._broker_net_qty(sym)
            if net is None:
                continue
            net_known = True
            net_total += net
        candidates = [local, broker_count]
        if net_known:
            # Multi-contract trades: net contracts != trade count. Only count
            # contracts beyond what our tracked trades hold as extra (untracked) slots.
            tracked_contracts = sum(max(1, int(getattr(p, 'size', 1) or 1)) for p in self.positions.values())
            if net_total > tracked_contracts:
                candidates.append(local + (net_total - tracked_contracts))
        return max(candidates)

    def _write_board_status(self, note: str = '') -> None:
        """Persist open-slot board so UI/status readers see fills immediately."""
        try:
            os.makedirs('data', exist_ok=True)
            open_count = self._effective_open_count()
            payload = {
                'bot_running': True,
                'paper_mode': bool(self.paper_mode or self.paper_orders),
                'session': 'live_rithmic',
                'symbols': list(self.symbols),
                'open_trades': open_count,
                'max_concurrent_trades': self.max_positions,
                'long_open': sum(1 for p in self.positions.values() if p.direction == 'long'),
                'short_open': sum(1 for p in self.positions.values() if p.direction == 'short'),
                'positions': [
                    {
                        'order_id': oid,
                        'symbol': p.symbol,
                        'direction': p.direction,
                        'entry': p.entry_price,
                        'sl': p.sl,
                        'tp': p.tp,
                        'entry_time': p.entry_time.isoformat() if p.entry_time else '',
                    }
                    for oid, p in self.positions.items()
                ],
                'note': note or f"{open_count}/{self.max_positions} open",
                'updated_at': datetime.now(timezone.utc).isoformat(),
            }
            with open(os.path.join('data', 'bot_status.json'), 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            bot_logger.debug(f"board status write skipped: {e}")

    def _close_local_position_broker_sync(self, order_id: str, reason: str) -> None:
        pos = self.positions.get(order_id)
        if not pos:
            return
        bot_logger.warning(
            f"🔄 Position sync: {order_id} closed by broker ({reason}) — exit price UNKNOWN; "
            f"recorded as BROKER_SYNC_UNKNOWN (not fed to learner as a loss)"
        )
        self.trades.append({
            'entry_time': pos.entry_time.isoformat() if pos.entry_time else '',
            'exit_time': datetime.now(timezone.utc).isoformat(),
            'symbol': pos.symbol,
            'direction': pos.direction.upper(),
            'entry': pos.entry_price,
            'exit': 0,
            'pnl': 0,
            'exit_type': 'BROKER_SYNC_UNKNOWN',
            'pnl_unknown': True,
            'sl': pos.sl,
            'tp': pos.tp,
        })
        # Unknown exit != $0 LOSS: do not teach the learner a fake loss.
        self._ghost_flat_kept_logged.discard(order_id)
        if FLEX3_ON:
            self._flex3.on_unknown_exit()
            self._flex3.save(FLEX3_STATE_FILE)
            print("   FLEX3: unknown-pnl exit -> new entries blocked until flex3_state.json reconciled")
        del self.positions[order_id]
        print(f"🔄 Position {order_id} removed ({reason})")
        self._cancel_orders_for_closed_position(order_id, pos.symbol, 'broker_sync_flat')

    def _planned_sl_from_trades_log(self, symbol: str, direction: str, avg: float) -> float:
        """Original SL of the last logged ENTRY that matches this broker position."""
        try:
            path = os.path.join('logs', 'trades.log')
            with open(path, encoding='utf-8', errors='replace') as f:
                tail = f.readlines()[-400:]
        except Exception:
            return 0.0
        pat = re.compile(
            rf"ENTRY {direction} {re.escape(symbol)} @ ([0-9.]+) SL=([0-9.]+)"
        )
        for line in reversed(tail):
            m = pat.search(line)
            if not m:
                continue
            entry, sl = float(m.group(1)), float(m.group(2))
            if avg > 0 and abs(entry - avg) <= 25.0:
                return sl
        return 0.0

    def _planned_tp_from_trades_log(self, symbol: str, direction: str, avg: float) -> float:
        """Planned TP of a logged ENTRY that IS this position (entry within a few points)."""
        try:
            path = os.path.join('logs', 'trades.log')
            with open(path, encoding='utf-8', errors='replace') as f:
                tail = f.readlines()[-400:]
        except Exception:
            return 0.0
        pat = re.compile(
            rf"ENTRY {direction} {re.escape(symbol)} @ ([0-9.]+) SL=([0-9.]+) TP=([0-9.]+)"
        )
        for line in reversed(tail):
            m = pat.search(line)
            if not m:
                continue
            entry, tp = float(m.group(1)), float(m.group(3))
            if avg > 0 and abs(entry - avg) <= ADOPT_PLANNED_TP_MATCH_PTS:
                return tp
        return 0.0

    @staticmethod
    def _adopt_tp_r() -> float:
        """R multiple for an adopted trade's TP when its setup is unknown."""
        raw = os.getenv('ADOPT_TP_R', '').strip()
        if raw:
            try:
                return float(raw)
            except ValueError:
                pass
        try:
            r = _revamp_exits.revamp_trend_target_r()
        except Exception:
            r = None
        return float(r) if r else 1.5

    def _adopt_untracked_exposure(self) -> int:
        """Broker net > tracked: rebuild a tracked trade (with a stop) instead of only warning."""
        if not ADOPT_UNTRACKED or not self._live_broker():
            return 0
        if not hasattr(self.broker, 'adopt_untracked_position'):
            return 0
        seen = self.__dict__.setdefault('_untracked_seen_at', {})
        adopted = 0
        now = time.time()
        for sym in self.symbols:
            # 2026-10-11b ADOPT-SAFE: adopted SL/TP follow the real position (resize /
            # cancel on manual close, partial close or flip) before anything else.
            self._reconcile_adopted(sym)
            net = self._broker_net_qty(sym)
            try:
                _raw = (self.broker._positions or {}).get(sym) or {}  # type: ignore[attr-defined]
                _signed = int(_raw.get('size', 0) or 0)
            except Exception:
                _signed = 0
            _net_dir = 'long' if _signed > 0 else ('short' if _signed < 0 else '')
            # Only same-side trades count against broker net (a bot long does not
            # "cover" a manual short).
            tracked = sum(
                max(1, int(getattr(p, 'size', 1) or 1))
                for p in self.positions.values()
                if p.symbol == sym and (not _net_dir or p.direction == _net_dir)
            )
            if _net_dir and any(
                p.symbol == sym and p.direction != _net_dir
                and not str(p.order_id).startswith('bot_adopt_')
                for p in self.positions.values()
            ):
                seen.pop(sym, None)
                continue  # bot trade on the other side: ambiguous, never adopt
            if net is None or net <= tracked:
                seen.pop(sym, None)
                continue
            first = seen.get(sym)
            if first is None:
                seen[sym] = now
                bot_logger.warning(
                    f"Untracked {sym} size: broker net={net} > tracked={tracked} - "
                    f"adopting after {ADOPT_CONFIRM_SEC:.0f}s confirm"
                )
                continue
            if now - first < ADOPT_CONFIRM_SEC:
                continue
            atr = float((self.__dict__.get('_last_atr') or {}).get(sym) or 0)
            stop_pts = atr * ADOPT_STOP_ATR_MULT if atr > 0 else ADOPT_DEFAULT_STOP_PTS
            direction = 'long'
            try:
                raw = (self.broker._positions or {}).get(sym) or {}  # type: ignore[attr-defined]
                direction = 'long' if int(raw.get('size', 1) or 1) > 0 else 'short'
                avg = float(raw.get('avg_price') or 0)
            except Exception:
                avg = 0.0
            planned = self._planned_sl_from_trades_log(sym, direction, avg)
            planned_tp = self._planned_tp_from_trades_log(sym, direction, avg)
            res = self.broker.adopt_untracked_position(
                sym, default_stop_pts=stop_pts, planned_sl=planned,
                planned_tp=planned_tp, tp_r=self._adopt_tp_r(),
                max_risk_usd=float(getattr(self, 'max_risk_per_trade_usd', MAX_RISK_PER_TRADE_USD) or 0),
            )
            seen.pop(sym, None)
            if res and res.get('merged'):
                pos = self.positions.get(str(res.get('ticket')))
                if pos is not None:
                    pos.size = int(res.get('size') or pos.size)
                    if float(res.get('open_price') or 0) > 0:
                        pos.entry_price = float(res['open_price'])
                print(
                    f"🛡️ ADOPT-SAFE: manual {sym} size now x{res.get('size')} - merged into "
                    f"{res.get('ticket')} (SL/TP resized, no extra orders stacked)"
                )
                continue
            if res:
                adopted += 1
                print(
                    f"🛡️ ADOPTED untracked {sym} {res.get('type')} x{res.get('size')} @ "
                    f"{float(res.get('open_price') or 0):.2f} as {res.get('ticket')} | "
                    f"SL {float(res.get('sl') or 0):.2f} "
                    f"({'existing' if res.get('sl_existing') else 'placed' if res.get('sl_placed') else 'FAILED - set one by hand'}) | "
                    f"TP {float(res.get('tp') or 0):.2f} "
                    f"({'existing' if res.get('tp_existing') else 'placed' if res.get('tp_placed') else 'NONE - set one by hand'})"
                )
        return adopted

    def _reconcile_adopted(self, sym: str) -> None:
        """2026-10-11b: keep adopted SL/TP == real manual size; drop adopted trades whose
        position was closed or flipped at the broker (their orders are cancelled first)."""
        if not self.broker or not hasattr(self.broker, 'reconcile_adopted_protection'):
            return
        if not any(p.symbol == sym and str(p.order_id).startswith('bot_adopt_')
                   for p in self.positions.values()):
            return
        last = self.__dict__.setdefault('_adopt_reconcile_ts', {})
        now = time.time()
        if now - float(last.get(sym) or 0) < 3.0:
            return
        last[sym] = now
        try:
            res = self.broker.reconcile_adopted_protection(sym) or {}
        except Exception as e:
            bot_logger.warning(f"reconcile_adopted_protection({sym}) failed: {e}")
            return
        for oid, size in (res.get('sizes') or {}).items():
            pos = self.positions.get(oid)
            if pos is not None and int(size or 0) > 0 and int(pos.size or 0) != int(size):
                bot_logger.warning(f"ADOPT-SAFE {oid}: tracked size {pos.size} -> {size}")
                pos.size = int(size)
        for oid in res.get('stale') or []:
            if oid not in self.positions:
                continue
            fill = self._confirmed_fill_for(oid)
            if fill.get('confirmed') and float(fill.get('exit_price') or 0) > 0:
                continue  # normal exit path records the real P&L
            self._close_local_position_broker_sync(
                oid, 'ADOPT-SAFE: manual position closed/flipped at broker - adopted SL/TP cancelled'
            )

    def _adopt_untracked_broker_positions(self, broker_tickets: Dict[str, Dict[str, Any]]) -> int:
        """Record broker-open tickets the bot was not tracking so the board stays honest."""
        adopted = 0
        for ticket, raw in broker_tickets.items():
            if ticket in self.positions:
                continue
            if self._effective_open_count() >= self.max_positions and ticket not in self.positions:
                # Still adopt into tracking so we do not open MORE; count includes this after adopt.
                pass
            symbol = str(raw.get('symbol') or raw.get('pair') or self.symbol)
            side = str(raw.get('type', 'BUY')).upper()
            direction = 'long' if side in ('BUY', 'LONG', '0') else 'short'
            entry = float(raw.get('open_price') or raw.get('entry_price') or 0) or 0.0
            # After restart, allow trail if a broker stop (native or protective STOP-LIMIT) exists.
            adopt_info = {}
            if self.broker and hasattr(self.broker, 'get_order_info'):
                adopt_info = self.broker.get_order_info(ticket) or {}
            trail_ok = self._broker_can_update_stop(adopt_info) or (
                float(raw.get('sl') or 0) > 0 and bool(adopt_info.get('protective_sl_order_id'))
            )
            pos = Position(
                order_id=ticket,
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                size=int(raw.get('size', 1) or 1),
                sl=float(raw.get('sl') or 0) or 0.0,
                tp=float(raw.get('tp') or 0) or 0.0,
                entry_time=datetime.now(timezone.utc),
                trailing_enabled=bool(trail_ok),
                ensemble_signal='BUY' if direction == 'long' else 'SELL',
                model_signals=None,
                regime='broker_adopt',
            )
            self.positions[ticket] = pos
            adopted += 1
            bot_logger.warning(
                f"Adopted untracked broker position {ticket} {direction} {symbol} @ {entry}"
            )
            print(f"⚠️ Adopted broker position into tracking: {ticket} {direction} {symbol} conf n/a")
        return adopted

    def _sync_live_brackets(self) -> int:
        """Follow SL/TP the user moved / re-placed on Lucid (broker list_orders = truth).

        Updates local pos.sl / pos.tp so bar-touch exit checks, board status and
        fill confirmation use the prices that are actually working.
        """
        if not self._live_broker() or not hasattr(self.broker, 'sync_tracked_brackets'):
            return 0
        changed = 0
        for sym in self.symbols:
            if not any(p.symbol == sym for p in self.positions.values()):
                continue
            try:
                live = self.broker.sync_tracked_brackets(sym) or {}
            except Exception as e:
                bot_logger.warning(f"sync_tracked_brackets({sym}) failed: {e}")
                continue
            for oid, rec in live.items():
                pos = self.positions.get(oid)
                if not pos or not isinstance(rec, dict):
                    continue
                if (
                    rec.get('sl_working') is False
                    and float(pos.sl or 0) > 0
                    and hasattr(self.broker, 'repair_missing_stop')
                ):
                    try:
                        if self.broker.repair_missing_stop(
                            ticket=oid,
                            symbol=pos.symbol,
                            side='BUY' if pos.direction == 'long' else 'SELL',
                            size=pos.size,
                            stop_loss=float(pos.sl),
                        ):
                            print(f"   🛡️ {oid} SL re-placed (was missing)")
                    except Exception as e:
                        bot_logger.error(f"SL repair {oid} failed: {e}")
                for leg, attr in (('sl', 'sl'), ('tp', 'tp')):
                    if not rec.get(f'{leg}_working'):
                        continue
                    try:
                        px = float(rec.get(leg) or 0)
                    except (TypeError, ValueError):
                        px = 0.0
                    old = float(getattr(pos, attr) or 0)
                    if px > 0 and abs(px - old) > 1e-9:
                        setattr(pos, attr, px)
                        changed += 1
                        how = 'rebound' if leg in (rec.get('rebound') or []) else 'adjusted'
                        print(f"   ✏️ {oid} {leg.upper()} {how} on broker: {old:.2f} → {px:.2f} (still tracked)")
                        bot_logger.info(f"{oid} {leg.upper()} {how} on broker {old:.2f} -> {px:.2f}")
                        self._unconfirmed_exit_logged.discard(oid)
        return changed

    def _confirmed_fill_for(self, order_id: str) -> Dict[str, Any]:
        try:
            with self.broker._order_lock:  # type: ignore[union-attr]
                return dict(
                    (getattr(self.broker, '_confirmed_exit_fills', {}) or {}).get(order_id) or {}
                )
        except Exception:
            try:
                return dict((getattr(self.broker, '_confirmed_exit_fills', {}) or {}).get(order_id) or {})
            except Exception:
                return {}

    def sync_broker_position(self):
        """Sync with actual broker position (live mode).

        Lucid/Rithmic list_positions often lags or returns flat right after a
        fill while brackets are still working. Never drop a local fill solely
        because its ticket is missing from an empty broker snapshot.
        """
        if self.paper_mode or self.paper_orders or not self.broker:
            self._write_board_status()
            return

        try:
            if self._sync_live_brackets():
                changed_prices = True
            else:
                changed_prices = False
            if self._adopt_untracked_exposure():
                changed_prices = True
            broker_tickets = self._broker_open_tickets()
            changed = changed_prices

            for order_id, pos in list(self.positions.items()):
                # Broker-confirmed SL/TP fill (incl. user-adjusted legs): close now
                # and free the slot — do not wait for a bar to touch stale levels.
                fill = self._confirmed_fill_for(order_id)
                if fill.get('confirmed') and float(fill.get('exit_price') or 0) > 0 and str(
                    fill.get('leg') or ''
                ).upper() in ('TP', 'SL'):
                    leg = str(fill.get('leg')).upper()
                    self.process_exit(
                        order_id, 'TAKE_PROFIT' if leg == 'TP' else 'STOP_LOSS',
                        float(fill['exit_price']),
                    )
                    changed = True
                    continue

                # Keep fills inside connector entry-grace / ghost-flat grace.
                in_grace = False
                if hasattr(self.broker, 'entry_protection_grace_active'):
                    try:
                        in_grace = bool(self.broker.entry_protection_grace_active(order_id))
                    except Exception:
                        in_grace = False
                age_sec = 0.0
                if pos.entry_time:
                    et = pos.entry_time
                    if et.tzinfo is None:
                        et = et.replace(tzinfo=timezone.utc)
                    age_sec = (datetime.now(timezone.utc) - et).total_seconds()
                if in_grace or age_sec < GHOST_FLAT_GRACE_SEC:
                    continue

                if order_id in broker_tickets:
                    continue

                # Connector still tracking this entry (fill + brackets) → keep local,
                # even if Lucid list_positions is empty/lagging.
                still_in_connector = False
                try:
                    if hasattr(self.broker, "_orders") and hasattr(self.broker, "_order_lock"):
                        with self.broker._order_lock:  # type: ignore[attr-defined]
                            still_in_connector = order_id in self.broker._orders  # type: ignore[attr-defined]
                    elif hasattr(self.broker, "get_order_info"):
                        still_in_connector = self.broker.get_order_info(order_id) is not None
                except Exception:
                    still_in_connector = False
                if still_in_connector:
                    continue

                net = self._broker_net_qty(pos.symbol)
                if net is None:
                    # Unknown — do not wipe local tracking.
                    continue
                if net > 0:
                    # Broker still has exposure; ticket IDs may not match aggregates.
                    # With 2 concurrent trades the OTHER one can keep the symbol
                    # open: if the remaining size is fully owned by trades the
                    # connector still tracks, this (dropped) one is closed.
                    others_qty = 0
                    for oid2, p2 in self.positions.items():
                        if oid2 == order_id or p2.symbol != pos.symbol:
                            continue
                        try:
                            with self.broker._order_lock:  # type: ignore[attr-defined]
                                tracked2 = oid2 in self.broker._orders  # type: ignore[attr-defined]
                        except Exception:
                            tracked2 = True
                        if tracked2:
                            others_qty += abs(int(p2.size or 1))
                    if net > others_qty:
                        continue

                # Broker flat, connector dropped the order, and past grace → drop local.
                confirmed = {}
                try:
                    confirmed = dict(
                        (getattr(self.broker, '_confirmed_exit_fills', {}) or {}).get(order_id) or {}
                    )
                except Exception:
                    confirmed = {}
                if not (confirmed.get('confirmed') and float(confirmed.get('exit_price') or 0) > 0):
                    # Try to recover the REAL exit (SL/TP fill) before giving up.
                    if hasattr(self.broker, 'recover_exit_fill'):
                        try:
                            confirmed = dict(self.broker.recover_exit_fill(order_id, pos.symbol) or {})
                        except Exception as e:
                            bot_logger.warning(f"recover_exit_fill({order_id}) failed: {e}")
                if confirmed.get('confirmed') and float(confirmed.get('exit_price') or 0) > 0 and str(
                    confirmed.get('leg') or ''
                ).upper() in ('TP', 'SL'):
                    leg = str(confirmed.get('leg') or '').upper()
                    et = 'TAKE_PROFIT' if leg == 'TP' else 'STOP_LOSS'
                    self.process_exit(order_id, et, float(confirmed['exit_price']))
                else:
                    # Never wipe a trade whose SL/TP is still working (or unknown).
                    live = None
                    if hasattr(self.broker, 'entry_protection_live'):
                        try:
                            live = self.broker.entry_protection_live(order_id, pos.symbol)
                        except Exception:
                            live = None
                    else:
                        live = False
                    if live is not False:
                        if order_id not in self._ghost_flat_kept_logged:
                            self._ghost_flat_kept_logged.add(order_id)
                            bot_logger.warning(
                                f"Keeping {order_id} tracked — broker reads flat but SL/TP "
                                f"still working/unknown (live={live}); not wiping"
                            )
                        continue
                    self._close_local_position_broker_sync(order_id, 'broker flat after grace')
                changed = True

            adopted = self._adopt_untracked_broker_positions(broker_tickets)
            if adopted:
                changed = True

            # Hard cap safety: if somehow over max, block is via _effective_open_count.
            open_count = self._effective_open_count()
            if open_count > self.max_positions:
                bot_logger.warning(
                    f"Open count {open_count} exceeds max {self.max_positions} — "
                    f"entry gate will block until flat"
                )

            self._write_board_status(
                note=f"{open_count}/{self.max_positions} open (sync)"
            )
            if changed:
                print(f"🔄 Position sync: {len(self.positions)}/{self.max_positions} active "
                      f"(effective {open_count})")

        except Exception as e:
            bot_logger.warning(f"Position sync error: {e}")

    def save_log(self):
        """Save trade log."""
        os.makedirs('logs', exist_ok=True)
        
        stats = {'trades': 0}
        if self.trades:
            wins = [t for t in self.trades if t['pnl'] > 0]
            losses = [t for t in self.trades if t['pnl'] <= 0]
            gross_profit = sum(t['pnl'] for t in wins) if wins else 0
            gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
            
            stats = {
                'trades': len(self.trades),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': len(wins) / len(self.trades) * 100 if self.trades else 0,
                'total_pnl': sum(t['pnl'] for t in self.trades),
                'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0
            }
        
        with open(self.log_file, 'w') as f:
            json.dump({
                'symbols': self.symbols,
                'paper_mode': self.paper_mode,
                'strategy': 'enhanced_breakout',
                'params': PARAMS,
                'stats': stats,
                'trades': self.trades
            }, f, indent=2, default=str)
        
        print(f"📁 Log saved: {self.log_file}")
    
    def run(self):
        """Main trading loop."""
        print("\n" + "=" * 70)
        if REVAMP_EXITS_ON:
            print("  LIVE TRADING - Rithmic + REVAMP research ensemble exits")
        else:
            print("  LIVE TRADING - Rithmic + Full AI Ensemble")
        print("=" * 70)
        print(f"Symbols:    {', '.join(self.symbols)}")
        cfg = getattr(self, "_size_cfg", {}) or {}
        print(
            f"Contracts:  base={self.contracts} max={cfg.get('max_contracts', 4)} "
            f"(conf bands {cfg.get('conf_2')}/{cfg.get('conf_3')}/{cfg.get('conf_4')}; "
            f"gate day_green+streak>={cfg.get('win_streak_min', 2)})"
        )
        if self.paper_orders:
            mode_str = '📝 PAPER-ORDERS (live data, simulated fills)'
        elif self.paper_mode:
            mode_str = 'PAPER'
        else:
            mode_str = '⚠️ LIVE'
        print(f"Mode:       {mode_str}")
        if REVAMP_EXITS_ON:
            print(f"Strategy:   Research 5m ADX ensemble (trend/MR/breakout)")
            print(f"REVAMP:     {_revamp_exits.describe()}")
            print(f"Session:    {_revamp_exits.session_hours_label()}")
            print(f"Trail ATR:  {TRAIL_SETTINGS['atr_mult']} (live default without revamp bat: 0.6)")
            print(f"Extra trades: {_extra_rule.describe(self._extra_mode(), float((getattr(self, '_size_cfg', None) or SIZE_SCALE_CFG).get('extra_trade_min_conf', 0.72)))}")
            _rv_mode = {1: 'compact (REVAMP_VERBOSE=2 detailed, 0 off)', 2: 'detailed (REVAMP_VERBOSE=1 compact, 0 off)'}
            print(f"Readout:    {('ON ' + _rv_mode.get(REVAMP_VERBOSE_LEVEL, '')) if REVAMP_VERBOSE else 'OFF (set REVAMP_VERBOSE=1)'}")
        else:
            print(f"Strategy:   Sweep-Gate + ML (IntelligentTrader, AdvancedStrategies)")
            print(f"Features:   Dynamic SL/TP, Trailing Stops, 8 Confirmation Models")
            print(f"Trail ATR:  {TRAIL_SETTINGS['atr_mult']}")
        for _sym in self.symbols:
            _sp = self._spec(_sym)
            print(f"Contract:   {_sym} ${_sp['point_value']:.0f}/pt, tick {_sp['tick_size']} = "
                  f"${_sp['tick_size'] * _sp['point_value']:.2f}")
        if PROFIT_LOCK_ENABLED and LOCK_MODE == 'usd':
            print(
                f"ProfitLock: $-lock whole trade: +${LOCK_ACTIVATE_USD:.0f} -> lock +${LOCK_KEEP_USD:.0f}; "
                f"from +${LOCK_TRAIL_START_USD:.0f} keep {LOCK_TRAIL_KEEP_PCT:.0%} of peak"
            )
        elif PROFIT_LOCK_ENABLED and LOCK_MODE == 'r':
            print(
                f"ProfitLock: R-lock +{LOCK_R1}R -> +{LOCK_R1_KEEP}R; +{LOCK_R2}R -> "
                f"max(+{LOCK_R2_KEEP}R, {TRAIL_KEEP_PCT:.0%} of MFE)"
            )
        elif PROFIT_LOCK_ENABLED:
            print(
                f"ProfitLock: +${PROFIT_LOCK_TRIGGER_1_USD:.0f}→+${PROFIT_LOCK_LEVEL_1_USD:.0f} SL; "
                f"+${PROFIT_LOCK_TRIGGER_2_USD:.0f}→+${PROFIT_LOCK_LEVEL_2_USD:.0f} SL"
            )
        if getattr(self, '_session_ledger', None) is not None:
            _dl, _ol, _pt = _session_limits.limits()
            print(f"Risk:       Day session (9:30-4 ET) loss limit ${_dl:,.0f} | overnight loss limit ${_ol:,.0f} | "
                  f"daily profit target ${_pt:,.0f} | Max concurrent {self.max_positions} (no daily trade cap)")
            print(f"            (calendar-day ${self.daily_loss_limit:,.0f} limit replaced; limits only block new entries)")
        else:
            print(f"Risk:       Max ${self.daily_loss_limit} daily loss | Max concurrent {self.max_positions}")
        _mr = float(getattr(self, 'max_risk_per_trade_usd', MAX_RISK_PER_TRADE_USD) or 0)
        print(f"            Max stop risk per trade: {('$' + format(_mr, '.0f')) if _mr > 0 else 'OFF'} (MAX_RISK_PER_TRADE_USD)")
        if _reentry.REENTRY_ENABLED:
            print(f"Re-entry:   ON - after a profitable stop exit, re-enter on a 1-tick break of the peak within "
                  f"{_reentry.REENTRY_WINDOW_BARS} x 5m bars (max {_reentry.REENTRY_MAX}/trade, 09:30-15:55 ET, "
                  f"SL max(6-bar swing, {_reentry.REENTRY_ATR_MULT:g} ATR) capped by max risk, TP {_reentry.REENTRY_TARGET_R:g}R)")
        else:
            print("Re-entry:   OFF (REENTRY_ENABLED=0)")
        if 'NQ' in self.symbols:
            print("            NOTE: daily loss only blocks NEW entries; one NQ stop ($20/pt) can lose more than that.")
        print("=" * 70)
        
        if self.paper_orders:
            print("\n📝 PAPER-ORDERS MODE - Connected to live Rithmic data, orders are simulated.")
        elif not self.paper_mode and not self.skip_confirm:
            print("\n⚠️  WARNING: LIVE MODE - Real money at risk!")
            print("Press Ctrl+C within 10 seconds to cancel...")
            try:
                time.sleep(10)
            except KeyboardInterrupt:
                print("\nCancelled.")
                return
        elif not self.paper_mode:
            print("\n⚠️  LIVE MODE - Starting immediately (--yes flag)")
        
        print("\n🚀 Starting trading loop...\n")
        
        last_bar_time: Dict[str, Any] = {}
        
        try:
            while True:
                # Check daily limits
                if not self.check_daily_limits():
                    print(f"⏸️  Limits reached - waiting...")
                    time.sleep(300)
                    continue
                
                # Sync position with broker
                self.sync_broker_position()
                # Cancel leftover sell/buy limits + stops not tied to an open trade.
                self.sweep_orphan_orders()
                # Re-entry after a profitable stop exit (bot-side market entry, REENTRY_ENABLED=1).
                if not FLEX3_ON:   # re-entries are not part of EVAL PASSER / FUNDED 12K
                    self.check_reentry_triggers()
                
                any_new_bar = False
                cycle_candidates: List[Dict[str, Any]] = []

                for symbol in self.symbols:
                    # Read-only NEAT sidecar: append MNQ 1m bars to CSV (no orders).
                    try:
                        from src.utils.neat_bar_export import maybe_export_1m
                        maybe_export_1m(self.broker, symbol)
                    except Exception as _neat_exp_err:
                        # Never break trading for export failures.
                        pass
                    df = self.get_candles(symbol=symbol, count=(REVAMP_CANDLE_COUNT if REVAMP_EXITS_ON else 100))
                    _min_bars = 220 if REVAMP_EXITS_ON else (self.lookback + self.ema_len + 15)
                    if df is None or len(df) < _min_bars:
                        print(f"⚠️  {symbol} insufficient data, waiting...")
                        continue

                    df = self.calculate_indicators(df)
                    try:
                        _atr_now = float(df.iloc[-1].get('atr', 0) or 0)
                        if np.isfinite(_atr_now) and _atr_now > 0:
                            self.__dict__.setdefault('_last_atr', {})[symbol] = _atr_now
                    except Exception:
                        pass

                    if 'datetime' in df.columns:
                        current_bar = df.iloc[-1]['datetime']
                    elif 'time' in df.columns:
                        current_bar = df.iloc[-1]['time']
                    else:
                        current_bar = datetime.now(timezone.utc)

                    if last_bar_time.get(symbol) == current_bar:
                        continue
                    last_bar_time[symbol] = current_bar
                    any_new_bar = True
                    try:
                        if str(symbol).upper().startswith("MNQ"):
                            from src.utils.shadow_bar_export import export_closed_bars
                            export_closed_bars(df, symbol)
                    except Exception:
                        pass

                    if self.positions:
                        # Past TP → cancel TP + trail (before local TAKE_PROFIT check).
                        self.update_past_tp_to_trail(symbol, df)
                        exits = self.check_position_exit(symbol, df)
                        for order_id, exit_type in exits:
                            pos = self.positions.get(order_id)
                            if not pos:
                                continue
                            if self._live_broker():
                                live_exit = self._confirmed_live_exit(order_id, exit_type)
                                if not live_exit:
                                    continue
                                exit_type, exit_price = live_exit
                            else:
                                exit_price = pos.tp if exit_type == 'TAKE_PROFIT' else pos.sl
                            self.process_exit(order_id, exit_type, exit_price)

                        # Dollar profit locks first, then ATR trail (both only tighten).
                        self.update_dollar_profit_locks(symbol, df)
                        # 2026-10-22 per-trade +$400 -> +$50 lock (after USD lock: no-op if already tighter)
                        self.update_fixed_profit_lock(symbol, df)
                        self.update_trailing_stops(symbol, df)
                        # Add-on trades: 2h time limit (backtest label A).
                        self.check_addon_time_exits(symbol, df)

                    _rv_open_n = self._effective_open_count()
                    _rv_checked = False
                    _rv_signal = None
                    if _rv_open_n < self.max_positions and self.cooldown == 0 and not (FLEX3_ON and not F3.OG_MAIN_ON):   # OG main OFF under EVAL PASSER; A+ only under FUNDED 12K
                        _rv_checked = True
                        signal = self.check_entry_signal(symbol, df)
                        _rv_signal = signal
                        if signal:
                            cycle_candidates.append(signal)
                    # Backtested add-ons (ONL reclaim / 6 PM long): flat-only, own windows.
                    _addon_sig, _addon_lines = (None, []) if FLEX3_ON else self.check_addon_signals(symbol, main_signal=_rv_signal)   # Flex3 has its own ONL/E1 + 6 PM
                    if _addon_sig:
                        cycle_candidates.append(_addon_sig)

                    row = df.iloc[-1]
                    pos_syms = [p.symbol for p in self.positions.values()]
                    pos_str = f"{len(self.positions)}/{self.max_positions} {pos_syms}" if self.positions else "None"
                    cooldown_str = f"(CD:{self.cooldown})" if self.cooldown > 0 else ""
                    print(f"[{format_ts_et(current_bar)}] {symbol} {row['close']:.2f} | Pos: {pos_str} {cooldown_str} | {self._day_total_text()}")
                    for _ti, _tp in enumerate([p for p in self.positions.values() if p.symbol == symbol], 1):
                        print(f"   {trade_status_line(_ti, _tp)}")
                    _xl = self._extra_trade_status(_rv_checked, _rv_signal, float(row['close']))
                    if _xl:
                        print(f"   {_xl}")
                    _ovl = self._overnight_sleeve_status_line()
                    if _ovl:
                        print(f"   {_ovl}")
                    _sll = self._session_status_line()
                    if _sll:
                        print(f"   {_sll}")
                    if OG_STATUS and not FLEX3_ON:
                        try:   # read-only readout; never affects orders
                            from src.strategy import og_status as OGST
                            OGST.install()
                            _ogl = OGST.lines(self, symbol, df, float(row['close']))
                            print("\n".join(_ogl))
                            with open(os.getenv('OG_STATUS_LOG', 'logs/og_status.log'), 'a', encoding='utf-8') as fh:
                                fh.write("\n".join(_ogl) + "\n")
                        except Exception as e:
                            print(f"   OG status readout error: {e}")
                    for _al in _addon_lines:
                        print(f"   ➕ {_al}")
                    for _rl in self.reentry_status_lines(symbol):
                        print(f"   🔁 {_rl}")
                    if REVAMP_EXITS_ON and REVAMP_VERBOSE:
                        self._print_revamp_reasoning(symbol, df, _rv_open_n, _rv_checked, _rv_signal)

                if FLEX3_ON:
                    self.flex3_cycle()

                if not any_new_bar:
                    time.sleep(5)
                    continue

                if self.cooldown > 0:
                    self.cooldown -= 1

                if self._effective_open_count() < self.max_positions and cycle_candidates:
                    priority_rank = {sym: idx for idx, sym in enumerate(self.symbol_priority)}
                    best = max(
                        cycle_candidates,
                        key=lambda s: (
                            float(s.get('confidence', 0.0)),
                            -priority_rank.get(s.get('symbol', ''), 999),
                        ),
                    )
                    if len(cycle_candidates) > 1:
                        print(
                            f"🏆 Best setup this cycle: {best['symbol']} "
                            f"({best.get('confidence', 0.0):.1%} confidence, priority={self.symbol_priority})"
                        )
                    _open_n = self._effective_open_count()
                    # 2nd/3rd trade gate: EXTRA_TRADE_MODE (same_dir_profit | conf). Price = signal bar close,
                    # as in the box backtest (second_trade2.py).
                    _xblock = self._extra_trade_block(
                        str(best.get('direction', '')), float(best.get('entry', 0.0) or 0.0),
                        best.get('confidence', 0.0), best.get('symbol'))
                    if _open_n >= 1 and _xblock:
                        print(f"   {ordinal(_open_n + 1)} trade skipped: {_xblock} ({_open_n} open)")
                        bot_logger.info(f"EXTRA TRADE SKIPPED {best.get('symbol')} {best.get('direction')}: {_xblock} ({_open_n} open)")
                    elif self.place_order(best):
                        spec = self._spec(best['symbol'])
                        print(f"📊 {best['symbol']} ATR: {best['atr']:.2f}")
                        risk = abs(best['entry'] - best['sl']) * spec['point_value']
                        print(f"   Risk: ${risk:.2f} per contract")

                time.sleep(10)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Trading stopped by user")
        finally:
            self.save_log()
            if self.broker:
                self.cleanup_orders_on_shutdown()
                self.broker.shutdown()
            
            print("\n" + "=" * 70)
            print("  SESSION SUMMARY")
            print("=" * 70)
            print(f"Trades:    {len(self.trades)}")
            print(f"Daily PnL: ${self.daily_pnl:.2f}")
            if self.trades:
                wins = len([t for t in self.trades if t['pnl'] > 0])
                print(f"Win rate:  {wins}/{len(self.trades)}")


def main():
    import argparse
    
    env_symbols = [s.strip() for s in os.getenv('TRADING_SYMBOLS', 'MNQ').split(',') if s.strip()] or ['MNQ']
    default_symbol = env_symbols[0]
    trading_mode = os.getenv('TRADING_MODE', 'paper').lower()

    parser = argparse.ArgumentParser(description='Live Breakout Trading via Rithmic')
    parser.add_argument('--symbol', default=default_symbol, choices=['MES', 'MNQ', 'NQ'], help='Single symbol mode')
    parser.add_argument('--symbols', nargs='+', choices=['MES', 'MNQ', 'NQ'], help='Multi-symbol mode, e.g. --symbols MNQ')
    parser.add_argument('--paper', action='store_true', help='Paper trading mode (Yahoo data + simulated orders)')
    parser.add_argument('--paper-orders', action='store_true', dest='paper_orders', help='Live Rithmic data + simulated orders')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip 10-second confirmation')
    
    args = parser.parse_args()

    if args.symbols:
        symbols = args.symbols
    elif os.getenv('TRADING_SYMBOLS'):
        symbols = env_symbols
    else:
        symbols = [args.symbol]

    paper_mode = args.paper or trading_mode in ('paper', 'backtest')
    trader = LiveRithmicTrader(
        symbol=symbols[0],
        symbols=symbols,
        paper_mode=paper_mode,
        skip_confirm=args.yes,
        paper_orders=args.paper_orders,
    )
    
    if not trader.connect():
        print("\n❌ Could not connect to Rithmic")
        print("\nCheck your .env file has:")
        print("  RITHMIC_USER_ID=your_user")
        print("  RITHMIC_PASSWORD=your_password")
        print("  RITHMIC_SYSTEM=LucidTrading")
        sys.exit(1)
    
    trader.run()


if __name__ == "__main__":
    main()
