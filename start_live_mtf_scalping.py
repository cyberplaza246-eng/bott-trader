#!/usr/bin/env python
"""
LIVE TRADING - Multi-Timeframe Scalping Strategy via Rithmic

Strategy: 5M Trend Filter + 1M Entry Signals
Symbols: MES, MNQ, NQ, MGC (Micro Gold)
- Up to 4 concurrent MNQ positions (each with SL+TP); other symbols use MAX_POSITIONS
- MES: TP 80 ticks below resistance
- Others: Standard 1.5:1 R:R

Requirements:
1. Rithmic credentials in .env
2. Active Tradesea/Lucid account with trading permissions

Usage:
    python start_live_mtf_scalping.py              # Trade all symbols
    python start_live_mtf_scalping.py --symbols MES MNQ  # Specific symbols
    python start_live_mtf_scalping.py --paper      # Paper mode
    python start_live_mtf_scalping.py --live       # Live mode (overrides TRADING_MODE and --paper)
"""

import os
import sys
import time
import json
import traceback
import argparse
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

import pytz

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.utils.email_notify import send_email, notify_trade_placed
from src.broker.rithmic_connector import RithmicConnector, resample_subminute_to_1m
from src.ai.mnq_context import compute_vwap
from src.ai.llm_advisor import LLMTradeAdvisor
from src.ai.news_bias import NewsBiasAdvisor, fetch_headlines_for_symbol, headline_providers_label
from src.ai.policy_scorer import PolicyScorer
from src.ai.mnq_smart_filters import MNQSmartFilters
from src.ai.entry_quality import (
    DEFAULT_ENTRY_QUALITY,
    check_long_entry_quality,
    check_short_entry_quality,
    entry_quality_enabled,
    parse_entry_quality,
)
from src.ai.order_flow import TickFlowTracker
from src.utils.logger import bot_logger, trades_logger
from src.utils.bias_15m import (
    compute_15m_trend,
    resolve_15m_bias_buffer_pts,
    resolve_15m_bias_mode,
    trend_rule_label,
)
from src.utils.trading_session import (
    SESSION_EXTENDED,
    SESSION_RTH,
    format_session_windows,
    is_in_session_windows_et,
    is_session_open_et,
    load_scalp_sessions_from_env,
    resolve_session_mode_from_env,
    seconds_until_scalp_window_et,
    seconds_until_session_open_et,
    session_mode_label,
)
from src.utils.flow_counter_trend import (
    evaluate_counter_trend,
    flow_blocks_long,
    flow_blocks_short,
    flow_confirms_long_direction,
    flow_contradicts_5m_trend,
    resolve_flow_counter_config,
    resolve_flow_entry_guard,
)
from src.strategy.entry_diagnostics import (
    GateEvalContext,
    evaluate_global_gates,
    evaluate_long_gates,
    evaluate_short_gates,
)
from src.strategy.scalp_momentum_live import (
    ScalpSymbolState,
    check_scalp_entry,
    evaluate_scalp_gates,
)
from src.ai.deepseek_trade_learner import DeepSeekTradeLearner
from src.ai.adaptive_learner import AdaptiveLearner
from config.strategy_config import ADAPTIVE_SKIP_ENABLED, adaptive_skip_enabled
from src.strategy.scalp_hybrid import (
    ScalpHybridState,
    add_30s_body_stats,
    check_hybrid_entry,
    evaluate_hybrid_gates,
    format_30s_trigger_log,
    hybrid_block_summary,
    hybrid_trigger_no_signal_reason,
    _trigger_eval,
)
from src.strategy.scalp_brackets import compute_scalp_bracket_pts, format_scalp_bracket_log
from src.strategy.mnq_profit_defaults import (
    apply_profit_env_defaults,
    profit_mode_applied_summary,
)
from src.strategy.fvs1 import (
    FVS1Config,
    FVS1RiskState,
    FVS1State,
    check_fvs1_entry,
    evaluate_fvs1_gates,
)


def _scalp_bracket_for_atr(
    atr_1m: Optional[float] = None,
    sl_mult: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Smart scalp SL/TP from env base + optional 1M ATR bounds."""
    sl_pts, tp_pts, atr_used, rr = compute_scalp_bracket_pts(
        atr_1m if atr_1m is not None else 0.0,
        sl_pts=SCALP_SL_PTS,
        tp_pts=SCALP_TP_PTS,
        sl_min=SCALP_SL_MIN,
        sl_max=SCALP_SL_MAX,
        tp_min=SCALP_TP_MIN,
        tp_max=SCALP_TP_MAX,
        min_rr=SCALP_MIN_RR,
        use_atr_bounds=SCALP_USE_ATR_BOUNDS,
    )
    if sl_mult > 1.0:
        sl_pts = min(sl_pts * sl_mult, SCALP_SL_MAX)
        tp_pts = max(tp_pts, sl_pts * SCALP_MIN_RR)
        tp_pts = min(tp_pts, SCALP_TP_MAX)
        rr = tp_pts / sl_pts if sl_pts > 0 else rr
    return sl_pts, tp_pts, atr_used, rr


def _scalp_chase_flow_cfg() -> Dict:
    """Flow-confirmed momentum: relax or skip chase filter."""
    return {
        "chase_flow_relax": SCALP_CHASE_FLOW_RELAX,
        "chase_skip_on_flow": SCALP_CHASE_SKIP_ON_FLOW,
        "chase_ema_atr_flow": SCALP_CHASE_EMA_ATR_FLOW,
        "chase_body_mult_flow": SCALP_CHASE_BODY_MULT_FLOW,
        "chase_flow_buy_pct_long": SCALP_CHASE_FLOW_BUY_PCT_LONG,
        "chase_flow_buy_pct_short": SCALP_CHASE_FLOW_BUY_PCT_SHORT,
        "chase_flow_delta_min": SCALP_CHASE_FLOW_DELTA_MIN,
    }


def _scalp_trigger_flow_cfg() -> Dict:
    """Flow-confirmed 30s trigger: relax green/red bar requirement."""
    return {
        "trigger_relax_flow": SCALP_TRIGGER_RELAX_FLOW,
        "trigger_relax_buy_pct_long": SCALP_TRIGGER_RELAX_BUY_PCT_LONG,
        "trigger_relax_buy_pct_short": SCALP_TRIGGER_RELAX_BUY_PCT_SHORT,
        "trigger_relax_delta_min": SCALP_TRIGGER_RELAX_DELTA_MIN,
    }


def _scalp_rsi_gate_cfg() -> Dict:
    """RSI confirmation gate for hybrid entries (blocks selling dips / buying tops)."""
    return {
        "rsi_gate_enabled": SCALP_RSI_GATE,
        "rsi_min_short": SCALP_RSI_MIN_SHORT,
        "rsi_max_long": SCALP_RSI_MAX_LONG,
        "rsi_relax_adx": SCALP_RSI_RELAX_ADX,
        "rsi_relax_min_short": SCALP_RSI_RELAX_MIN_SHORT,
        "rsi_relax_max_long": SCALP_RSI_RELAX_MAX_LONG,
    }


def _scalp_hybrid_params(atr_1m: Optional[float] = None) -> Dict:
    """Shared hybrid scalp kwargs from env (live + diagnostics)."""
    sl_pts, tp_pts, _, _ = _scalp_bracket_for_atr(atr_1m)
    return {
        "pullback_enabled": PULLBACK_ENABLED,
        "continuation_enabled": CONTINUATION_ENABLED,
        "adx_min_pullback": SCALP_ADX_MIN,
        "adx_min_continuation": SCALP_CONTINUATION_ADX_MIN,
        "pullback_atr": SCALP_PULLBACK_ATR,
        "setup_bars": SCALP_SETUP_BARS,
        "setup_window_sec": SCALP_SETUP_WINDOW_SEC,
        "trigger_bar_seconds": TRIGGER_BAR_SECONDS,
        "sl_pts": sl_pts,
        "tp_pts": tp_pts,
        "trend_mode": SCALP_TREND_MODE,
        "continuation_volume_strict": SCALP_CONTINUATION_VOLUME,
        "cont_volume_min_ratio": SCALP_CONT_VOLUME_MIN_RATIO,
        "chase_body_mult": SCALP_CHASE_BODY_MULT,
        "chase_ema_atr": SCALP_CHASE_EMA_ATR,
        "chase_flow_cfg": _scalp_chase_flow_cfg(),
        "trigger_flow_cfg": _scalp_trigger_flow_cfg(),
        "rsi_gate_cfg": _scalp_rsi_gate_cfg(),
        "momentum_burst_enabled": SCALP_MOMENTUM_BURST,
        "momentum_burst_adx": SCALP_MOMENTUM_BURST_ADX,
        "aggressive_mode": SCALP_AGGRESSIVE,
        "flow_burst_delta_min": SCALP_FLOW_BURST_DELTA_MIN,
        "micro_break_pts": SCALP_TRIGGER_MICRO_BREAK_PTS,
        "flow_trigger_delta_min": SCALP_FLOW_TRIGGER_DELTA_MIN,
        "flow_trigger_buy_pct_long": SCALP_FLOW_TRIGGER_BUY_PCT_LONG,
        "flow_trigger_buy_pct_short": SCALP_FLOW_TRIGGER_BUY_PCT_SHORT,
        "flow_strong_delta_min": SCALP_FLOW_STRONG_DELTA_MIN,
        "flow_strong_buy_pct_long": SCALP_FLOW_STRONG_BUY_PCT_LONG,
        "flow_strong_buy_pct_short": SCALP_FLOW_STRONG_BUY_PCT_SHORT,
    }


def _scalp_bracket_log_for_atr(atr_1m: Optional[float] = None) -> str:
    sl_pts, tp_pts, atr_used, rr = _scalp_bracket_for_atr(atr_1m)
    return format_scalp_bracket_log(sl_pts, tp_pts, atr_used, rr)

# ── Trading Hours (env: OVERNIGHT_TRADING=true or SESSION_MODE=extended) ──
SESSION_MODE = resolve_session_mode_from_env()
# Intraday liquid windows for hybrid. Empty = full outer session (SCALP_SESSIONS=off).
_SCALP_MODE_HINT = (
    os.getenv("SCALP_MODE", "").strip().lower()
    or os.getenv("STRATEGY_MODE", "").strip().lower()
)
SCALP_SESSIONS = load_scalp_sessions_from_env(
    use_defaults=_SCALP_MODE_HINT in (
        "hybrid", "pullback", "continuation", "scalp_hybrid",
    ),
)


def is_market_open_et(now=None):
    """Return True if current ET time is inside the configured session window."""
    return is_session_open_et(now, SESSION_MODE)


def is_scalp_window_open_et(now=None) -> bool:
    """Outer session open AND (if configured) inside SCALP_SESSIONS."""
    if not is_session_open_et(now, SESSION_MODE):
        return False
    ok, _ = is_in_session_windows_et(now, SCALP_SESSIONS)
    return ok


def print_profit_mode_banner() -> None:
    """Startup banner so live operator sees the locked profit recipe."""
    if STRATEGY_MODE != "scalp_hybrid":
        return
    pb = "ON" if PULLBACK_ENABLED else "off"
    ct = "ON" if CONTINUATION_ENABLED else "off"
    be = (
        f"BE@{SCALP_BREAKEVEN_PCT:.0%}+trail"
        if SCALP_BREAKEVEN_ENABLED else "BE off"
    )
    sess = format_session_windows(SCALP_SESSIONS)
    print()
    print("=" * 70)
    print("  PROFIT MODE: hybrid ultra_fast (continuation-heavy + burst)")
    print(
        f"  SL/TP base {SCALP_SL_PTS:.0f}/{SCALP_TP_PTS:.0f}pt | "
        f"max_hold={MAX_HOLD_SECONDS}s | ADX≥{SCALP_ADX_MIN}/"
        f"cont≥{SCALP_CONTINUATION_ADX_MIN} | aggressive="
        f"{'ON' if SCALP_AGGRESSIVE else 'off'}"
    )
    print(
        f"  pullback={pb} continuation={ct} | {be} | "
        f"max_pos={MAX_POSITIONS_MNQ} | loss_cd={LOSS_COOLDOWN_MINUTES}m"
    )
    print(f"  outer={SESSION_MODE.upper()} | SCALP_SESSIONS: {sess}")
    print(
        "  Backtest target: MNQ ultra_fast ~PF 2.2+ (fees in) — "
        "sim ≠ live; trade RTH only"
    )
    print("=" * 70)
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy Parameters (Validated via backtest)
# ══════════════════════════════════════════════════════════════════════════════

# 5M Trend Filter
TREND_EMA_FAST = 50
TREND_EMA_SLOW = 200
ADX_THRESHOLD = 17
ADX_PERIOD = 14
DI_TOLERANCE = float(os.getenv("DI_TOLERANCE", "3"))  # DI+/DI- gap allowed vs opposite side
DI_FLOW_TOLERANCE = float(os.getenv("DI_FLOW_TOLERANCE", "8"))  # relaxed when flow confirms direction
USE_FLOW_DI_OVERRIDE = os.getenv("USE_FLOW_DI_OVERRIDE", "true").lower() == "true"
USE_FLOW_ADX_RELAX = os.getenv("USE_FLOW_ADX_RELAX", "true").lower() == "true"
FLOW_ADX_RELAX = int(os.getenv("FLOW_ADX_RELAX", "1"))
DI_COUNTER_TREND = 20.0        # Allow counter-trend when DI dominates by 20+ pts (config override)
COUNTER_ADX = 25               # Min ADX for standard counter-trend (config override)
COUNTER_TREND_SHORTS = True
COUNTER_TREND_LONGS = True
FLOW_COUNTER_CFG: Dict = {}
FLOW_ENTRY_GUARD: Dict = {}

# 1M Entry Conditions
ENTRY_EMA_FAST = 9
ENTRY_EMA_SLOW = 21
RSI_PERIOD = 14
RSI_LONG_MIN, RSI_LONG_MAX = 35, 60
RSI_SHORT_MIN, RSI_SHORT_MAX = 40, 65
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
VOLUME_RATIO_THRESHOLD = 0.4   # Looser = more trades (backtest-validated)
BB_PERIOD, BB_STD = 20, 2
BB_EXTREME_LOW, BB_EXTREME_HIGH = 0.05, 0.95
CANDLE_CONFIRMATION = os.getenv("CANDLE_CONFIRMATION", "false").lower() == "true"

# Risk Settings (defaults; overridden by data/mnq_profit_config.json when present)
ATR_PERIOD = 14
ATR_MULT = 1.2             # SL = 1.2 × ATR
TP_MULT = 1.3              # TP = 1.3 × SL — dual-symbol backtest winner
TP_BUFFER_ATR_MULT = float(os.getenv("TP_BUFFER_ATR_MULT", "0.5"))  # TP sits below resistance / above support
RESISTANCE_LOOKBACK = 20   # 5M bars for swing high/low
MIN_5M_BARS_BASELINE = 50  # EMA50/200 + ADX warmup for baseline MTF
MIN_5M_BARS_SCALP_IDEAL = int(os.getenv("MIN_5M_BARS_SCALP_IDEAL", "20"))
MIN_5M_BARS_SCALP_FLOOR = int(os.getenv("MIN_5M_BARS_SCALP_FLOOR", "15"))
CANDLE_HISTORY_HOURS = float(os.getenv("CANDLE_HISTORY_HOURS", "8"))
CANDLE_1M_COUNT = int(os.getenv("CANDLE_1M_COUNT", "300"))
_scalp_fast_raw = os.getenv("SCALP_FAST_MODE", "").strip().lower()
SCALP_FAST_MODE: Optional[bool] = (
    True if _scalp_fast_raw in ("true", "1", "yes")
    else False if _scalp_fast_raw in ("false", "0", "no")
    else None  # auto: on for scalp modes
)
FAST_SCAN_LOG = os.getenv("FAST_SCAN_LOG", "false").lower() == "true"
SHOW_NEWS = os.getenv("SHOW_NEWS", os.getenv("NEWS_CONSOLE", "true")).lower() == "true"
MIN_RR_AFTER_CAP = 1.0     # Match backtest (no min-R:R skip); closer TP needs lower floor
MAX_PULLBACK_ATR = 1.5     # Max distance from EMA for pullback (matches backtest)
PULLBACK_ATR_MULT = float(os.getenv("PULLBACK_ATR_MULT", "1.5"))  # widen zone when 5M ADX >= 40
STRONG_PULLBACK_ADX = 40

# Risk Management
DAILY_LOSS_LIMIT = 300.0           # Stop trading after $300 daily loss
MAX_LOSS_PER_TRADE = 250.0         # Force close if unrealized loss > $250
MAX_TRADES_PER_DAY = 20            # Daily entry cap (all symbols)
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "1"))  # Non-MNQ concurrent cap
MAX_POSITIONS_MNQ = 4              # MNQ concurrent open positions (each qty=1 + SL/TP)
CONTRACTS = 1
ORPHAN_ORDER_THRESHOLD = int(os.getenv("ORPHAN_ORDER_THRESHOLD", "10"))
ORPHAN_SWEEP_EVERY_N_SCANS = int(os.getenv("ORPHAN_SWEEP_EVERY_N_SCANS", "5"))
BROKER_POSITION_BLOCK = os.getenv("BROKER_POSITION_BLOCK", "true").lower() == "true"

# Volatility Filter — max 1M bar range (high-low pts); NQ default 60, others 45
VOLATILITY_FILTER_POINTS = 45
_DEFAULT_MAX_1M_BAR_PTS = {'NQ': 60, 'MNQ': 45, 'MES': 45, 'MGC': 45}


def _parse_max_1m_bar_pts():
    """Parse MAX_1M_BAR_PTS: '60' (global) or 'NQ:60,MNQ:50' (per symbol)."""
    raw = os.getenv("MAX_1M_BAR_PTS", "").strip()
    global_default = None
    overrides = {}
    if not raw:
        return global_default, overrides
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            sym, val = part.split(":", 1)
            overrides[sym.strip().upper()] = float(val.strip())
        else:
            global_default = float(part)
    return global_default, overrides


_MAX_1M_BAR_GLOBAL, _MAX_1M_BAR_OVERRIDES = _parse_max_1m_bar_pts()


def max_1m_bar_pts_for(symbol: str) -> float:
    sym = symbol.upper()
    if sym in _MAX_1M_BAR_OVERRIDES:
        return _MAX_1M_BAR_OVERRIDES[sym]
    if _MAX_1M_BAR_GLOBAL is not None:
        return _MAX_1M_BAR_GLOBAL
    return _DEFAULT_MAX_1M_BAR_PTS.get(sym, VOLATILITY_FILTER_POINTS)

# Symbol specs — MNQ and NQ share index price; NQ is 10× $/point → tighter SL pt cap for same $ risk
SYMBOL_SPECS = {
    'MES': {'point_value': 5.0, 'tick_size': 0.25, 'max_loss_per_trade': 250, 'daily_loss_limit': 300},
    'MNQ': {'point_value': 2.0, 'tick_size': 0.25, 'max_loss_per_trade': 250, 'daily_loss_limit': 300},
    'NQ': {'point_value': 20.0, 'tick_size': 0.25, 'max_loss_per_trade': 250, 'daily_loss_limit': 300},
    'MGC': {'point_value': 10.0, 'tick_size': 0.10, 'max_loss_per_trade': 250, 'daily_loss_limit': 300},
}
NASDQ_SYMBOLS = frozenset({'MNQ', 'NQ'})

# Default symbols to trade (Micros only - safer position sizing)
# Using single symbol to reduce Rithmic API load and avoid lock timeouts
DEFAULT_SYMBOLS = ['MNQ']

SMART_FILTERS_ENABLED = os.getenv("SMART_FILTERS_ENABLED", "false").lower() == "true"
VERBOSE_SKIP_REASONS = os.getenv("VERBOSE_SKIP_REASONS", "true").lower() == "true"
if FAST_SCAN_LOG:
    VERBOSE_SKIP_REASONS = False
COMPACT_NEWS = os.getenv("COMPACT_NEWS", "true").lower() == "true"
NEWS_CONSOLE = os.getenv("NEWS_CONSOLE", "true").lower() == "true"
POLICY_CONSOLE = os.getenv("POLICY_CONSOLE", "false").lower() == "true"
NEWS_HEADLINE_COUNT = max(0, int(os.getenv("NEWS_HEADLINE_COUNT", "2")))

USE_ORDER_FLOW = os.getenv("USE_ORDER_FLOW", "false").lower() == "true"
ORDER_FLOW_MODE = os.getenv("ORDER_FLOW_MODE", "advisory").lower()
ORDER_FLOW_WINDOW_SEC = int(os.getenv("ORDER_FLOW_WINDOW_SEC", "60"))

# Trade outcome learner (hybrid scalp) — local pattern blocks default ON (no API key)
USE_LOCAL_PATTERN_LEARNER = os.getenv("USE_LOCAL_PATTERN_LEARNER", "true").lower() == "true"
USE_DEEPSEEK_LEARNER = os.getenv("USE_DEEPSEEK_LEARNER", "false").lower() == "true"
DEEPSEEK_LEARN_EVERY_N = int(os.getenv("DEEPSEEK_LEARN_EVERY_N", "5"))

# Adaptive learner — records closes, skips bad hours/regimes/patterns (data/adaptive_learning.json)
USE_ADAPTIVE_LEARNER = os.getenv("USE_ADAPTIVE_LEARNER", "true").lower() == "true"

# Per-symbol consecutive loss halt (MTF path; FVS-1 has its own session halt)
# Default 0 = disabled — tighten entries via learners instead of pausing trading
MTF_MAX_CONSEC_LOSSES = int(os.getenv("MTF_MAX_CONSEC_LOSSES", "0"))
MTF_CONSEC_LOSS_PAUSE_MIN = int(os.getenv("MTF_CONSEC_LOSS_PAUSE_MINUTES", "30"))

# Sub-minute trigger bars (Rithmic SECOND_BAR history + tick fallback)
USE_30S_BARS = os.getenv("USE_30S_BARS", "false").lower() == "true"
_trigger_sec_raw = (
    os.getenv("TRIGGER_BAR_SECONDS")
    or os.getenv("TRIGGER_TIMEFRAME_SEC")
    or "30"
)
TRIGGER_BAR_SECONDS = max(1, int(_trigger_sec_raw))
SCALP_30S_FALLBACK_1M = os.getenv("SCALP_30S_FALLBACK_1M", "false").lower() == "true"

# Exit / cooldown — prevent re-entry after losses and overshoot past SL between scans
LOSS_COOLDOWN_MINUTES = int(os.getenv("LOSS_COOLDOWN_MINUTES", "5"))
DAILY_HALF_STOP_ENABLED = os.getenv("DAILY_HALF_STOP", "true").lower() == "true"
DAILY_HALF_STOP_PCT = float(os.getenv("DAILY_HALF_STOP_PCT", "0.5"))
SCAN_SLEEP_OPEN_SEC = int(os.getenv("SCAN_SLEEP_OPEN_SEC", "15"))
SCAN_SLEEP_IDLE_SEC = int(os.getenv("SCAN_SLEEP_IDLE_SEC", "60"))
MAX_HOLD_SECONDS = int(os.getenv("MAX_HOLD_SECONDS", "0"))  # 0 = disabled; force market exit when exceeded
# Breakeven / trail — move SL when trade reaches fraction of entry→TP (never loosen SL)
SCALP_BREAKEVEN_ENABLED = os.getenv("SCALP_BREAKEVEN_ENABLED", "true").lower() == "true"
SCALP_BREAKEVEN_PCT = float(os.getenv("SCALP_BREAKEVEN_PCT", "0.4"))
SCALP_BREAKEVEN_OFFSET_PTS = float(os.getenv("SCALP_BREAKEVEN_OFFSET_PTS", "0.25"))
SCALP_SECURE_PROFIT_PCT = float(os.getenv("SCALP_SECURE_PROFIT_PCT", "0.75"))
SCALP_TRAIL_AFTER_BE = os.getenv("SCALP_TRAIL_AFTER_BE", "true").lower() == "true"
SCALP_TRAIL_MFE_LOCK_PCT = float(os.getenv("SCALP_TRAIL_MFE_LOCK_PCT", "0.25"))
SCALP_SECURE_LOCK_PCT = float(os.getenv("SCALP_SECURE_LOCK_PCT", "0.5"))
SCALP_MAX_HOLD_TIGHTEN_PCT = float(os.getenv("SCALP_MAX_HOLD_TIGHTEN_PCT", "0.85"))
EXIT_CANDLE_COUNT = 100  # min bars for position exit high/low checks (never use 5)
PAPER_RITHMIC_BRACKETS = os.getenv("PAPER_RITHMIC_BRACKETS", "false").lower() == "true"
BROKER_PROTECTION_MAX_RETRIES = int(os.getenv("BROKER_PROTECTION_MAX_RETRIES", "5"))
SCAN_PROTECTION_REPAIR_RETRIES = int(os.getenv("SCAN_PROTECTION_REPAIR_RETRIES", "3"))
BROKER_FLAT_SYNC_SEC = float(os.getenv("BROKER_FLAT_SYNC_SEC", "30"))
BROKER_FLAT_SYNC_SCANS = int(os.getenv("BROKER_FLAT_SYNC_SCANS", "2"))

# Adaptive strategy (from data/mnq_profit_config.json backtest winner)
STRATEGY_MODE = "baseline"
USE_SCALP_MOMENTUM = os.getenv("USE_SCALP_MOMENTUM", "false").lower() == "true"
FULL_TRADE_DIAGNOSTICS = os.getenv("FULL_TRADE_DIAGNOSTICS", "true").lower() == "true"
if FAST_SCAN_LOG:
    FULL_TRADE_DIAGNOSTICS = False
DI_RELAX_STRENGTH = float(os.getenv("DI_RELAX_STRENGTH", "20"))
FLOW_RELAX_STRENGTH = float(os.getenv("FLOW_RELAX_STRENGTH", "20"))
FLOW_RELAX_LONG_BUY_PCT = float(os.getenv("FLOW_RELAX_LONG_BUY_PCT", "0.45"))
SCALP_MODE = os.getenv("SCALP_MODE", "").strip().lower()
SCALP_SL_PTS = float(os.getenv("SCALP_SL_PTS", "8"))
SCALP_TP_PTS = float(os.getenv("SCALP_TP_PTS", "14"))
SCALP_SL_MIN = float(os.getenv("SCALP_SL_MIN", "6"))
SCALP_SL_MAX = float(os.getenv("SCALP_SL_MAX", "12"))
SCALP_TP_MIN = float(os.getenv("SCALP_TP_MIN", "10"))
SCALP_TP_MAX = float(os.getenv("SCALP_TP_MAX", "20"))
SCALP_MIN_RR = float(os.getenv("SCALP_MIN_RR", "1.4"))
SCALP_USE_ATR_BOUNDS = os.getenv("SCALP_USE_ATR_BOUNDS", "true").lower() == "true"
SCALP_AGGRESSIVE = os.getenv("SCALP_AGGRESSIVE", "false").lower() == "true"
SCALP_ADX_MIN = int(os.getenv("SCALP_ADX_MIN", "15" if SCALP_AGGRESSIVE else "20"))
SCALP_CONTINUATION_ADX_MIN = int(
    os.getenv("SCALP_CONTINUATION_ADX_MIN", "20" if SCALP_AGGRESSIVE else "25")
)
SCALP_PULLBACK_ATR = float(os.getenv("SCALP_PULLBACK_ATR", "1.0" if SCALP_AGGRESSIVE else "0.5"))
SCALP_SETUP_BARS = int(os.getenv("SCALP_SETUP_BARS", "3"))
SCALP_SETUP_WINDOW_SEC = int(os.getenv("SCALP_SETUP_WINDOW_SEC", "0"))
SCALP_TREND_MODE = os.getenv(
    "SCALP_TREND_MODE", "vwap" if SCALP_AGGRESSIVE else "both",
).strip().lower()
SCALP_CONTINUATION_VOLUME = os.getenv(
    "SCALP_CONTINUATION_VOLUME", "false" if SCALP_AGGRESSIVE else "true",
).lower() == "true"
SCALP_CONT_VOLUME_MIN_RATIO = float(os.getenv("SCALP_CONT_VOLUME_MIN_RATIO", "0.85"))
SCALP_CHASE_BODY_MULT = float(os.getenv("SCALP_CHASE_BODY_MULT", "1.5"))
SCALP_CHASE_EMA_ATR = float(os.getenv("SCALP_CHASE_EMA_ATR", "1.25" if SCALP_AGGRESSIVE else "1.0"))
SCALP_CHASE_EMA_ATR_FLOW = float(os.getenv("SCALP_CHASE_EMA_ATR_FLOW", "2.5" if SCALP_AGGRESSIVE else "2.0"))
SCALP_CHASE_BODY_MULT_FLOW = float(os.getenv("SCALP_CHASE_BODY_MULT_FLOW", "2.0"))
SCALP_CHASE_FLOW_RELAX = os.getenv("SCALP_CHASE_FLOW_RELAX", "true").lower() == "true"
SCALP_CHASE_SKIP_ON_FLOW = os.getenv("SCALP_CHASE_SKIP_ON_FLOW", "false").lower() == "true"
SCALP_CHASE_FLOW_BUY_PCT_LONG = float(os.getenv("SCALP_CHASE_FLOW_BUY_PCT_LONG", "0.60"))
SCALP_CHASE_FLOW_BUY_PCT_SHORT = float(os.getenv("SCALP_CHASE_FLOW_BUY_PCT_SHORT", "0.40"))
SCALP_CHASE_FLOW_DELTA_MIN = float(os.getenv("SCALP_CHASE_FLOW_DELTA_MIN", "0"))
SCALP_MOMENTUM_BURST = os.getenv("SCALP_MOMENTUM_BURST", "true" if SCALP_AGGRESSIVE else "false").lower() == "true"
SCALP_MOMENTUM_BURST_ADX = int(os.getenv("SCALP_MOMENTUM_BURST_ADX", "15" if SCALP_AGGRESSIVE else "25"))
SCALP_FLOW_BURST_DELTA_MIN = float(os.getenv("SCALP_FLOW_BURST_DELTA_MIN", "50"))
SCALP_TRIGGER_MICRO_BREAK_PTS = float(os.getenv("SCALP_TRIGGER_MICRO_BREAK_PTS", "0.25"))
SCALP_FLOW_TRIGGER_DELTA_MIN = float(os.getenv("SCALP_FLOW_TRIGGER_DELTA_MIN", "30"))
SCALP_FLOW_TRIGGER_BUY_PCT_LONG = float(os.getenv("SCALP_FLOW_TRIGGER_BUY_PCT_LONG", "0.55"))
SCALP_FLOW_TRIGGER_BUY_PCT_SHORT = float(os.getenv("SCALP_FLOW_TRIGGER_BUY_PCT_SHORT", "0.45"))
SCALP_FLOW_STRONG_DELTA_MIN = float(os.getenv("SCALP_FLOW_STRONG_DELTA_MIN", "100"))
SCALP_FLOW_STRONG_BUY_PCT_LONG = float(os.getenv("SCALP_FLOW_STRONG_BUY_PCT_LONG", "0.60"))
SCALP_FLOW_STRONG_BUY_PCT_SHORT = float(os.getenv("SCALP_FLOW_STRONG_BUY_PCT_SHORT", "0.40"))
SCALP_TRIGGER_RELAX_FLOW = os.getenv(
    "SCALP_TRIGGER_RELAX_FLOW", "true" if SCALP_AGGRESSIVE else "false",
).lower() == "true"
SCALP_TRIGGER_RELAX_BUY_PCT_LONG = float(os.getenv("SCALP_TRIGGER_RELAX_BUY_PCT_LONG", "0.60"))
SCALP_TRIGGER_RELAX_BUY_PCT_SHORT = float(os.getenv("SCALP_TRIGGER_RELAX_BUY_PCT_SHORT", "0.40"))
SCALP_TRIGGER_RELAX_DELTA_MIN = float(os.getenv("SCALP_TRIGGER_RELAX_DELTA_MIN", "0"))
SCALP_RSI_GATE = os.getenv("SCALP_RSI_GATE", "true").lower() == "true"
SCALP_RSI_MIN_SHORT = float(os.getenv("SCALP_RSI_MIN_SHORT", "40"))
SCALP_RSI_MAX_LONG = float(os.getenv("SCALP_RSI_MAX_LONG", "65"))
SCALP_RSI_RELAX_ADX = float(os.getenv("SCALP_RSI_RELAX_ADX", "25"))
SCALP_RSI_RELAX_MIN_SHORT = float(os.getenv("SCALP_RSI_RELAX_MIN_SHORT", "35"))
SCALP_RSI_RELAX_MAX_LONG = float(os.getenv("SCALP_RSI_RELAX_MAX_LONG", "70"))
MIN_SECONDS_BETWEEN_ENTRIES = int(os.getenv("MIN_SECONDS_BETWEEN_ENTRIES", "0"))
PULLBACK_ENABLED = os.getenv("PULLBACK_ENABLED", "true").lower() == "true"
CONTINUATION_ENABLED = os.getenv("CONTINUATION_ENABLED", "true").lower() == "true"
FVS1_CFG: Optional[FVS1Config] = None
USE_15M_BIAS = False
USE_15M_ENTRY_GATE = False  # when false: 15M is direction-only (computed/printed, does not block entries)
BEAR_ADAPTIVE = False
BULL_ADAPTIVE = False
STRONG_TREND_ADX = 30
VWAP_ADX_MIN = 30
SOFT_15M_BIAS = False
BIAS_15M_MODE = "ema_cross"
BIAS_15M_BUFFER_PTS = 0.0
VWAP_REQUIRED = True
ENTRY_QUALITY: Dict = dict(DEFAULT_ENTRY_QUALITY)
STRONG_TREND_SKIP_MACD = False
STRONG_TREND_MIN_RR = 0.6
STRONG_TREND_RELAX_ADX = 40
BULL_RSI_LO = 35
BULL_RSI_HI = 70


def max_trades_per_day_for(symbol: str) -> int:
    """Per-symbol daily entry cap (same global limit for all symbols)."""
    return MAX_TRADES_PER_DAY


def max_positions_for(symbol: str) -> int:
    """Max concurrent open positions for a symbol (MNQ=4 by default)."""
    if symbol.upper() == "MNQ":
        return MAX_POSITIONS_MNQ
    return MAX_POSITIONS


def effective_adx_threshold(
    ctx_5m: Dict,
    flow_snap: Optional[Dict],
    direction: str,
) -> int:
    """ADX gate — relax by FLOW_ADX_RELAX when flow confirms trend direction."""
    threshold = ADX_THRESHOLD
    if not USE_FLOW_ADX_RELAX or not flow_snap or not USE_ORDER_FLOW:
        return threshold
    trend = ctx_5m.get("trend")
    if direction == "long" and trend == "bullish":
        if flow_confirms_long_direction(flow_snap, FLOW_ENTRY_GUARD):
            return max(0, threshold - FLOW_ADX_RELAX)
    elif direction == "short" and trend == "bearish":
        blocked, _ = flow_blocks_short(flow_snap, FLOW_ENTRY_GUARD)
        if not blocked:
            return max(0, threshold - FLOW_ADX_RELAX)
    return threshold


def apply_scalp_hybrid_config(cfg: Optional[Dict]) -> None:
    """Apply scalp_hybrid block from mnq_profit_config.json (env vars win if set)."""
    global MAX_HOLD_SECONDS, SCALP_SL_PTS, SCALP_TP_PTS, SCALP_SL_MIN, SCALP_SL_MAX
    global SCALP_TP_MIN, SCALP_TP_MAX, SCALP_MIN_RR, SCALP_AGGRESSIVE, SCALP_ADX_MIN
    global SCALP_CONTINUATION_ADX_MIN, SCALP_PULLBACK_ATR, SCALP_SETUP_BARS
    global SCALP_SETUP_WINDOW_SEC, SCALP_TREND_MODE, SCALP_CONTINUATION_VOLUME
    global SCALP_CHASE_BODY_MULT, SCALP_CHASE_EMA_ATR, SCALP_MOMENTUM_BURST
    global SCALP_MOMENTUM_BURST_ADX, SCALP_BREAKEVEN_PCT, SCALP_TRIGGER_RELAX_FLOW
    global SCALP_RSI_GATE, SCALP_RSI_MIN_SHORT, SCALP_RSI_MAX_LONG
    global SCALP_RSI_RELAX_ADX, SCALP_RSI_RELAX_MIN_SHORT, SCALP_RSI_RELAX_MAX_LONG
    if not cfg:
        return
    sh = cfg.get("scalp_hybrid") or {}
    if not sh.get("enabled", bool(sh)):
        return

    def _env_set(name: str) -> bool:
        return bool(os.getenv(name, "").strip())

    mapping = (
        ("SCALP_SL_PTS", "sl_pts", float, "SCALP_SL_PTS"),
        ("SCALP_TP_PTS", "tp_pts", float, "SCALP_TP_PTS"),
        ("SCALP_SL_MIN", "sl_min", float, "SCALP_SL_MIN"),
        ("SCALP_SL_MAX", "sl_max", float, "SCALP_SL_MAX"),
        ("SCALP_TP_MIN", "tp_min", float, "SCALP_TP_MIN"),
        ("SCALP_TP_MAX", "tp_max", float, "SCALP_TP_MAX"),
        ("SCALP_MIN_RR", "min_rr", float, "SCALP_MIN_RR"),
        ("MAX_HOLD_SECONDS", "max_hold_sec", int, "MAX_HOLD_SECONDS"),
        ("SCALP_ADX_MIN", "adx_min", int, "SCALP_ADX_MIN"),
        ("SCALP_CONTINUATION_ADX_MIN", "continuation_adx_min", int, "SCALP_CONTINUATION_ADX_MIN"),
        ("SCALP_PULLBACK_ATR", "pullback_atr", float, "SCALP_PULLBACK_ATR"),
        ("SCALP_SETUP_BARS", "setup_bars", int, "SCALP_SETUP_BARS"),
        ("SCALP_SETUP_WINDOW_SEC", "setup_window_sec", int, "SCALP_SETUP_WINDOW_SEC"),
        ("SCALP_CHASE_BODY_MULT", "chase_body_mult", float, "SCALP_CHASE_BODY_MULT"),
        ("SCALP_CHASE_EMA_ATR", "chase_ema_atr", float, "SCALP_CHASE_EMA_ATR"),
        ("SCALP_MOMENTUM_BURST_ADX", "momentum_burst_adx", int, "SCALP_MOMENTUM_BURST_ADX"),
        ("SCALP_BREAKEVEN_PCT", "breakeven_pct", float, "SCALP_BREAKEVEN_PCT"),
        ("SCALP_RSI_MIN_SHORT", "rsi_min_short", float, "SCALP_RSI_MIN_SHORT"),
        ("SCALP_RSI_MAX_LONG", "rsi_max_long", float, "SCALP_RSI_MAX_LONG"),
        ("SCALP_RSI_RELAX_ADX", "rsi_relax_adx", float, "SCALP_RSI_RELAX_ADX"),
        ("SCALP_RSI_RELAX_MIN_SHORT", "rsi_relax_min_short", float, "SCALP_RSI_RELAX_MIN_SHORT"),
        ("SCALP_RSI_RELAX_MAX_LONG", "rsi_relax_max_long", float, "SCALP_RSI_RELAX_MAX_LONG"),
    )
    g = globals()
    for env_name, key, cast, global_name in mapping:
        if key in sh and not _env_set(env_name):
            g[global_name] = cast(sh[key])
    if "trend_mode" in sh and not _env_set("SCALP_TREND_MODE"):
        SCALP_TREND_MODE = str(sh["trend_mode"]).strip().lower()
    if "continuation_volume" in sh and not _env_set("SCALP_CONTINUATION_VOLUME"):
        SCALP_CONTINUATION_VOLUME = bool(sh["continuation_volume"])
    if "momentum_burst" in sh and not _env_set("SCALP_MOMENTUM_BURST"):
        SCALP_MOMENTUM_BURST = bool(sh["momentum_burst"])
    if "rsi_gate" in sh and not _env_set("SCALP_RSI_GATE"):
        SCALP_RSI_GATE = bool(sh["rsi_gate"])
    if "scalp_aggressive" in sh and not _env_set("SCALP_AGGRESSIVE"):
        SCALP_AGGRESSIVE = bool(sh["scalp_aggressive"])
        if SCALP_AGGRESSIVE and not _env_set("SCALP_TRIGGER_RELAX_FLOW"):
            SCALP_TRIGGER_RELAX_FLOW = True


def _reload_profit_mode_globals() -> None:
    """Re-read knobs that may have been filled by apply_profit_env_defaults."""
    global SCALP_MODE, SESSION_MODE, SCALP_SESSIONS
    global SCAN_SLEEP_OPEN_SEC, SCAN_SLEEP_IDLE_SEC
    global USE_ORDER_FLOW, ORDER_FLOW_MODE
    global MTF_MAX_CONSEC_LOSSES, MTF_CONSEC_LOSS_PAUSE_MIN
    global LOSS_COOLDOWN_MINUTES, SCALP_AGGRESSIVE, SCALP_FAST_MODE
    global MAX_HOLD_SECONDS, USE_30S_BARS
    SCALP_MODE = os.getenv("SCALP_MODE", "").strip().lower()
    SESSION_MODE = resolve_session_mode_from_env()
    SCALP_SESSIONS = load_scalp_sessions_from_env(
        use_defaults=SCALP_MODE in (
            "hybrid", "pullback", "continuation", "scalp_hybrid",
        ),
    )
    SCAN_SLEEP_OPEN_SEC = int(os.getenv("SCAN_SLEEP_OPEN_SEC", str(SCAN_SLEEP_OPEN_SEC)))
    SCAN_SLEEP_IDLE_SEC = int(os.getenv("SCAN_SLEEP_IDLE_SEC", str(SCAN_SLEEP_IDLE_SEC)))
    USE_ORDER_FLOW = os.getenv("USE_ORDER_FLOW", "false").lower() == "true"
    ORDER_FLOW_MODE = os.getenv("ORDER_FLOW_MODE", ORDER_FLOW_MODE).lower()
    MTF_MAX_CONSEC_LOSSES = int(os.getenv("MTF_MAX_CONSEC_LOSSES", str(MTF_MAX_CONSEC_LOSSES)))
    MTF_CONSEC_LOSS_PAUSE_MIN = int(
        os.getenv("MTF_CONSEC_LOSS_PAUSE_MINUTES", str(MTF_CONSEC_LOSS_PAUSE_MIN))
    )
    LOSS_COOLDOWN_MINUTES = int(os.getenv("LOSS_COOLDOWN_MINUTES", str(LOSS_COOLDOWN_MINUTES)))
    SCALP_AGGRESSIVE = os.getenv("SCALP_AGGRESSIVE", "true" if SCALP_AGGRESSIVE else "false").lower() == "true"
    _fast = os.getenv("SCALP_FAST_MODE", "").strip().lower()
    if _fast in ("true", "1", "yes"):
        SCALP_FAST_MODE = True
    elif _fast in ("false", "0", "no"):
        SCALP_FAST_MODE = False
    if os.getenv("MAX_HOLD_SECONDS", "").strip():
        MAX_HOLD_SECONDS = int(os.getenv("MAX_HOLD_SECONDS", "0"))
    USE_30S_BARS = os.getenv("USE_30S_BARS", "true" if USE_30S_BARS else "false").lower() == "true"


def load_profit_config() -> None:
    """Apply validated MNQ params from data/mnq_profit_config.json."""
    global ATR_MULT, TP_MULT, ADX_THRESHOLD, MAX_TRADES_PER_DAY
    global MAX_POSITIONS, MAX_POSITIONS_MNQ
    global VOLUME_RATIO_THRESHOLD, TP_BUFFER_ATR_MULT
    global STRATEGY_MODE, USE_15M_BIAS, USE_15M_ENTRY_GATE, BEAR_ADAPTIVE, BULL_ADAPTIVE, STRONG_TREND_ADX
    global VWAP_ADX_MIN, SOFT_15M_BIAS, BIAS_15M_MODE, BIAS_15M_BUFFER_PTS
    global VWAP_REQUIRED, MIN_RR_AFTER_CAP, ENTRY_QUALITY
    global STRONG_TREND_SKIP_MACD, STRONG_TREND_MIN_RR, STRONG_TREND_RELAX_ADX
    global BULL_RSI_LO, BULL_RSI_HI, PULLBACK_ATR_MULT
    global DI_TOLERANCE, DI_COUNTER_TREND, COUNTER_ADX, COUNTER_TREND_SHORTS, COUNTER_TREND_LONGS
    global FLOW_COUNTER_CFG, FLOW_ENTRY_GUARD
    path = os.path.join(os.path.dirname(__file__), "data", "mnq_profit_config.json")
    cfg_name = "custom"
    cfg_wr = 0.0
    cfg: Optional[Dict] = None
    if not os.path.isfile(path):
        pass
    else:
        try:
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
            applied = apply_profit_env_defaults(cfg, os.environ)
            if applied:
                _reload_profit_mode_globals()
                print(profit_mode_applied_summary(applied))
            cfg_name = cfg.get("name", "custom")
            cfg_wr = float(cfg.get("wr_mnq", cfg.get("wr", 0)))
            ATR_MULT = float(cfg.get("atr", ATR_MULT))
            TP_MULT = float(cfg.get("tp", TP_MULT))
            ADX_THRESHOLD = int(cfg.get("adx", ADX_THRESHOLD))
            MAX_TRADES_PER_DAY = int(cfg.get("max_tr", MAX_TRADES_PER_DAY))
            if "vol" in cfg:
                VOLUME_RATIO_THRESHOLD = float(cfg["vol"])
            if "tp_buffer" in cfg:
                TP_BUFFER_ATR_MULT = float(cfg["tp_buffer"])
            if cfg.get("atr") is None and "atr_mult" in cfg:
                ATR_MULT = float(cfg["atr_mult"])
            if cfg.get("tp") is None and "tp_mult" in cfg:
                TP_MULT = float(cfg["tp_mult"])
            if "max_trades" in cfg and "max_tr" not in cfg:
                MAX_TRADES_PER_DAY = int(cfg["max_trades"])
            if "max_positions_mnq" in cfg:
                MAX_POSITIONS_MNQ = int(cfg["max_positions_mnq"])
            if "max_positions" in cfg:
                MAX_POSITIONS = int(cfg["max_positions"])
            STRATEGY_MODE = cfg.get("strategy_mode", STRATEGY_MODE)
            USE_15M_BIAS = bool(cfg.get("use_15m_bias", USE_15M_BIAS))
            USE_15M_ENTRY_GATE = bool(cfg.get("15m_entry_gate", USE_15M_ENTRY_GATE))
            BEAR_ADAPTIVE = bool(cfg.get("bear_adaptive", BEAR_ADAPTIVE))
            if "pullback_atr_mult" in cfg:
                PULLBACK_ATR_MULT = float(cfg["pullback_atr_mult"])
            BULL_ADAPTIVE = bool(cfg.get("bull_adaptive", BULL_ADAPTIVE))
            STRONG_TREND_ADX = int(cfg.get("strong_trend_adx", STRONG_TREND_ADX))
            VWAP_ADX_MIN = int(cfg.get("vwap_adx_min", cfg.get("strong_trend_adx", VWAP_ADX_MIN)))
            SOFT_15M_BIAS = bool(cfg.get("soft_15m_bias", SOFT_15M_BIAS))
            BIAS_15M_MODE = resolve_15m_bias_mode(cfg)
            BIAS_15M_BUFFER_PTS = resolve_15m_bias_buffer_pts(cfg)
            VWAP_REQUIRED = bool(cfg.get("vwap_required", VWAP_REQUIRED))
            if "min_rr" in cfg:
                MIN_RR_AFTER_CAP = float(cfg["min_rr"])
            STRONG_TREND_SKIP_MACD = bool(cfg.get("strong_trend_skip_macd", STRONG_TREND_SKIP_MACD))
            STRONG_TREND_MIN_RR = float(cfg.get("strong_trend_min_rr", STRONG_TREND_MIN_RR))
            STRONG_TREND_RELAX_ADX = int(
                cfg.get("strong_trend_relax_adx", cfg.get("strong_trend_adx", STRONG_TREND_RELAX_ADX))
            )
            BULL_RSI_LO = int(cfg.get("bull_rsi_lo", BULL_RSI_LO))
            BULL_RSI_HI = int(cfg.get("bull_rsi_hi", BULL_RSI_HI))
            ENTRY_QUALITY = parse_entry_quality(cfg)
            if "di_tol" in cfg:
                DI_TOLERANCE = float(cfg["di_tol"])
            DI_COUNTER_TREND = float(cfg.get("di_counter", DI_COUNTER_TREND))
            COUNTER_ADX = int(cfg.get("counter_adx", COUNTER_ADX))
            COUNTER_TREND_SHORTS = bool(cfg.get("counter_trend_shorts", COUNTER_TREND_SHORTS))
            COUNTER_TREND_LONGS = bool(cfg.get("counter_trend_longs", COUNTER_TREND_LONGS))
            FLOW_COUNTER_CFG = resolve_flow_counter_config(cfg)
            FLOW_ENTRY_GUARD = resolve_flow_entry_guard(cfg)
        except Exception as e:
            print(f"Could not load profit config: {e}")
    apply_scalp_hybrid_config(cfg)
    apply_symbol_risk_overrides(cfg)
    if not FLOW_COUNTER_CFG:
        FLOW_COUNTER_CFG = resolve_flow_counter_config({})
    if not FLOW_ENTRY_GUARD:
        FLOW_ENTRY_GUARD = resolve_flow_entry_guard({})
    env_15m = os.getenv("USE_15M_BIAS", "").strip().lower()
    if env_15m in ("true", "1", "yes", "on"):
        USE_15M_BIAS = True
    elif env_15m in ("false", "0", "no", "off"):
        USE_15M_BIAS = False
    env_15m_gate = os.getenv("USE_15M_ENTRY_GATE", "").strip().lower()
    if env_15m_gate in ("true", "1", "yes", "on"):
        USE_15M_ENTRY_GATE = True
    elif env_15m_gate in ("false", "0", "no", "off"):
        USE_15M_ENTRY_GATE = False
    env_pb_mult = os.getenv("PULLBACK_ATR_MULT", "").strip()
    if env_pb_mult:
        PULLBACK_ATR_MULT = float(env_pb_mult)
    env_bias_mode = os.getenv("15M_BIAS_MODE", "").strip()
    if env_bias_mode:
        BIAS_15M_MODE = resolve_15m_bias_mode({"15m_bias_mode": env_bias_mode})
    env_bias_buf = os.getenv("15M_BIAS_BUFFER_PTS", "").strip()
    if env_bias_buf:
        BIAS_15M_BUFFER_PTS = float(env_bias_buf)
    env_max_tr = os.getenv("MAX_TRADES_PER_DAY", "").strip()
    if env_max_tr:
        MAX_TRADES_PER_DAY = int(env_max_tr)
    env_max_pos = os.getenv("MAX_POSITIONS", "").strip()
    if env_max_pos:
        MAX_POSITIONS = int(env_max_pos)
    env_mnq_max_pos = os.getenv("MNQ_MAX_POSITIONS", "").strip()
    if env_mnq_max_pos:
        MAX_POSITIONS_MNQ = int(env_mnq_max_pos)
    env_adx = os.getenv("ADX_THRESHOLD", "").strip()
    if env_adx:
        ADX_THRESHOLD = int(env_adx)
    _apply_strategy_mode_env()
    eq_on = entry_quality_enabled(ENTRY_QUALITY)
    print(
        f"Profit config: ATRx{ATR_MULT} TPx{TP_MULT} ADX>={ADX_THRESHOLD} "
        f"vol>={VOLUME_RATIO_THRESHOLD} TP buffer={TP_BUFFER_ATR_MULT}xATR "
        f"max_tr={MAX_TRADES_PER_DAY} MNQ_pos={MAX_POSITIONS_MNQ} "
        f"mode={STRATEGY_MODE} 15M={'on' if USE_15M_BIAS else 'off'} "
        f"15m_gate={'on' if USE_15M_ENTRY_GATE else 'off (direction only)'} "
        f"15m_rule={BIAS_15M_MODE} soft15={'on' if SOFT_15M_BIAS else 'off'} "
        f"strong_adx={STRONG_TREND_ADX} vwap@{VWAP_ADX_MIN} "
        f"entry_quality={'on' if eq_on else 'off'} "
        f"di_relax@{DI_RELAX_STRENGTH} flow_relax@{FLOW_RELAX_STRENGTH} "
        f"diag={'on' if FULL_TRADE_DIAGNOSTICS else 'off'} "
        f"({cfg_name}, backtest WR={cfg_wr:.0f}%)"
    )
    if STRATEGY_MODE == "scalp_b":
        atr_note = (
            f"ATR bounds {SCALP_SL_MIN}-{SCALP_SL_MAX}pt SL / "
            f"{SCALP_TP_MIN}-{SCALP_TP_MAX}pt TP R:R>={SCALP_MIN_RR}"
            if SCALP_USE_ATR_BOUNDS else
            f"fixed {SCALP_SL_MIN}-{SCALP_SL_MAX}pt SL / {SCALP_TP_MIN}-{SCALP_TP_MAX}pt TP"
        )
        print(
            f"Scalp B: base SL={SCALP_SL_PTS}pt TP={SCALP_TP_PTS}pt ({atr_note}) "
            f"ADX>={SCALP_ADX_MIN} pullback={SCALP_PULLBACK_ATR}xATR "
            f"setup={SCALP_SETUP_BARS}bars max_hold={MAX_HOLD_SECONDS}s "
            f"30s_bars={'on' if USE_30S_BARS else 'off'}"
        )
    if STRATEGY_MODE == "scalp_hybrid":
        vol_mode = "strict" if SCALP_CONTINUATION_VOLUME else f"relaxed≥{SCALP_CONT_VOLUME_MIN_RATIO:.0%}"
        burst = f" burst={'on' if SCALP_MOMENTUM_BURST else 'off'}"
        agg = f" aggressive={'ON' if SCALP_AGGRESSIVE else 'off'}"
        atr_note = (
            f"ATR bounds {SCALP_SL_MIN}-{SCALP_SL_MAX}pt SL / "
            f"{SCALP_TP_MIN}-{SCALP_TP_MAX}pt TP R:R>={SCALP_MIN_RR}"
            if SCALP_USE_ATR_BOUNDS else
            f"fixed {SCALP_SL_MIN}-{SCALP_SL_MAX}pt SL / {SCALP_TP_MIN}-{SCALP_TP_MAX}pt TP"
        )
        rsi_note = (
            f"RSI gate ON short≥{SCALP_RSI_MIN_SHORT:.0f} long≤{SCALP_RSI_MAX_LONG:.0f}"
            if SCALP_RSI_GATE else "RSI gate off"
        )
        print(
            f"Scalp HYBRID: base SL={SCALP_SL_PTS}pt TP={SCALP_TP_PTS}pt ({atr_note}) "
            f"trend={SCALP_TREND_MODE} ADX≥{SCALP_ADX_MIN} pullback={SCALP_PULLBACK_ATR}xATR "
            f"vol={vol_mode} chase={SCALP_CHASE_BODY_MULT}x/{SCALP_CHASE_EMA_ATR}ATR | "
            f"scan {SCAN_SLEEP_OPEN_SEC}s/{SCAN_SLEEP_IDLE_SEC}s "
            f"entry_gap={MIN_SECONDS_BETWEEN_ENTRIES}s{burst}{agg} | "
            f"max_hold={MAX_HOLD_SECONDS}s max_pos={MAX_POSITIONS_MNQ} | {rsi_note}"
        )
    if STRATEGY_MODE == "fvs1" and FVS1_CFG is not None:
        sess = ", ".join(f"{w.label} {w.start}-{w.end}" for w in FVS1_CFG.sessions)
        print(
            f"FVS-1: log_only={'on' if FVS1_CFG.log_only else 'OFF (live entries)'} "
            f"ADX>={FVS1_CFG.adx_min} max_hold={FVS1_CFG.max_hold_seconds}s "
            f"fee=${FVS1_CFG.round_trip_fee:.2f} | sessions: {sess}"
        )


def _apply_strategy_mode_env() -> None:
    """Env overrides for STRATEGY_MODE=scalp_b / scalp_hybrid / fvs1 / USE_SCALP_MOMENTUM."""
    global STRATEGY_MODE, USE_SCALP_MOMENTUM, ADX_THRESHOLD, USE_15M_ENTRY_GATE
    global ENTRY_QUALITY, MAX_HOLD_SECONDS, USE_30S_BARS, USE_15M_BIAS
    global PULLBACK_ENABLED, CONTINUATION_ENABLED, MAX_POSITIONS, MAX_POSITIONS_MNQ, SCALP_MODE
    global FVS1_CFG
    env_mode = os.getenv("STRATEGY_MODE", "").strip().lower()
    env_scalp_mode = SCALP_MODE or env_mode
    use_fvs1 = env_scalp_mode in ("fvs1", "fvs1_log") or env_mode == "fvs1"
    use_hybrid = env_scalp_mode in ("hybrid", "pullback", "continuation", "scalp_hybrid")
    use_scalp_b = USE_SCALP_MOMENTUM or env_mode == "scalp_b"
    if use_fvs1:
        FVS1_CFG = FVS1Config.from_env()
        if env_scalp_mode == "fvs1_log":
            FVS1_CFG.log_only = True
        STRATEGY_MODE = "fvs1"
        ADX_THRESHOLD = FVS1_CFG.adx_min
        USE_15M_ENTRY_GATE = False
        USE_15M_BIAS = False
        ENTRY_QUALITY = parse_entry_quality({"entry_quality": {"enabled": False}})
        if not os.getenv("MAX_HOLD_SECONDS", "").strip():
            MAX_HOLD_SECONDS = FVS1_CFG.max_hold_seconds
        if os.getenv("USE_30S_BARS", "").strip() == "":
            USE_30S_BARS = True
        if not os.getenv("MAX_POSITIONS", "").strip():
            MAX_POSITIONS = 2
        if not os.getenv("MNQ_MAX_POSITIONS", "").strip():
            MAX_POSITIONS_MNQ = 2
        return
    if env_mode == "scalp_b":
        USE_SCALP_MOMENTUM = True
    if env_scalp_mode == "pullback":
        PULLBACK_ENABLED = True
        CONTINUATION_ENABLED = False
    elif env_scalp_mode == "continuation":
        PULLBACK_ENABLED = False
        CONTINUATION_ENABLED = True
    elif env_scalp_mode in ("hybrid", "scalp_hybrid"):
        if os.getenv("PULLBACK_ENABLED", "").strip() == "":
            PULLBACK_ENABLED = True
        if os.getenv("CONTINUATION_ENABLED", "").strip() == "":
            CONTINUATION_ENABLED = True
    if use_hybrid:
        STRATEGY_MODE = "scalp_hybrid"
        ADX_THRESHOLD = SCALP_ADX_MIN
        USE_15M_ENTRY_GATE = False
        USE_15M_BIAS = os.getenv("USE_15M_BIAS", "false").lower() == "true"
        ENTRY_QUALITY = parse_entry_quality({"entry_quality": {"enabled": False}})
        if not os.getenv("MAX_HOLD_SECONDS", "").strip():
            MAX_HOLD_SECONDS = 30
        if os.getenv("USE_30S_BARS", "").strip() == "":
            USE_30S_BARS = True
        if not os.getenv("MAX_POSITIONS", "").strip():
            MAX_POSITIONS = 2
        if not os.getenv("MNQ_MAX_POSITIONS", "").strip():
            MAX_POSITIONS_MNQ = 2
        return
    if not use_scalp_b:
        return
    STRATEGY_MODE = "scalp_b"
    ADX_THRESHOLD = SCALP_ADX_MIN
    USE_15M_ENTRY_GATE = False
    USE_15M_BIAS = os.getenv("USE_15M_BIAS", "false").lower() == "true"
    ENTRY_QUALITY = parse_entry_quality({"entry_quality": {"enabled": False}})
    if not os.getenv("MAX_HOLD_SECONDS", "").strip():
        MAX_HOLD_SECONDS = 90
    if os.getenv("USE_30S_BARS", "").strip() == "":
        USE_30S_BARS = True


def scalp_fast_mode_active() -> bool:
    """Quick-scalp mode: 1M trend + lower 5M warmup (default on for scalp strategies)."""
    if SCALP_FAST_MODE is not None:
        return SCALP_FAST_MODE
    return STRATEGY_MODE in ("scalp_b", "scalp_hybrid", "fvs1")


def adaptive_rsi_long(ctx_5m: Dict, is_counter_trend: bool = False) -> Tuple[float, float]:
    if is_counter_trend:
        return 30, 70
    lo, hi = RSI_LONG_MIN, RSI_LONG_MAX
    if BULL_ADAPTIVE and ctx_5m.get("trend") == "bullish" and ctx_5m.get("adx", 0) >= STRONG_TREND_ADX:
        lo, hi = BULL_RSI_LO, BULL_RSI_HI
    return lo, hi


def adaptive_rsi_short(ctx_5m: Dict, is_counter_trend: bool) -> Tuple[float, float]:
    if is_counter_trend:
        return 30, 70
    lo, hi = RSI_SHORT_MIN, RSI_SHORT_MAX
    if BEAR_ADAPTIVE and ctx_5m.get("trend") == "bearish" and ctx_5m.get("adx", 0) >= STRONG_TREND_ADX:
        return 25, 65
    return lo, hi


def adaptive_pullback_atr(ctx_5m: Dict, direction: str, is_counter_trend: bool) -> float:
    if is_counter_trend:
        base = 2.5
    elif direction == "short" and BEAR_ADAPTIVE:
        if ctx_5m.get("trend") == "bearish" and ctx_5m.get("adx", 0) >= STRONG_TREND_ADX:
            base = 2.0
        else:
            base = MAX_PULLBACK_ATR
    elif direction == "long" and BULL_ADAPTIVE:
        if ctx_5m.get("trend") == "bullish" and ctx_5m.get("adx", 0) >= STRONG_TREND_ADX:
            base = 2.0
        else:
            base = MAX_PULLBACK_ATR
    else:
        base = MAX_PULLBACK_ATR
    if (
        not is_counter_trend
        and ctx_5m.get("adx", 0) >= STRONG_PULLBACK_ADX
        and PULLBACK_ATR_MULT > 1.0
    ):
        base *= PULLBACK_ATR_MULT
    return base


def adaptive_vwap_ok(price: float, vwap: float, direction: str, ctx_5m: Dict) -> bool:
    if not VWAP_REQUIRED:
        return True
    if pd.isna(vwap):
        return True
    if BEAR_ADAPTIVE and direction == "short":
        if ctx_5m.get("trend") == "bearish" and ctx_5m.get("adx", 0) >= VWAP_ADX_MIN:
            return price < vwap
    if BULL_ADAPTIVE and direction == "long":
        if ctx_5m.get("trend") == "bullish" and ctx_5m.get("adx", 0) >= VWAP_ADX_MIN:
            return price > vwap
    return True


def format_15m_bias_line(symbol: str, ctx_15m: Dict) -> str:
    """One-line 15M bias reason for scan output."""
    trend = ctx_15m.get("trend") or "unknown"
    close = ctx_15m.get("close")
    ema50 = ctx_15m.get("ema_50")
    ema200 = ctx_15m.get("ema_200")
    mode = ctx_15m.get("bias_mode", BIAS_15M_MODE)
    if close is None or ema50 is None or ema200 is None:
        return f"   {symbol}: 15M bias = {trend} (insufficient 15M history)"
    rule = trend_rule_label(trend, close, ema50, ema200, mode)
    bar_ts = ctx_15m.get("bar_time")
    ts_suffix = f" @ {bar_ts}" if bar_ts else ""
    gate_note = "" if USE_15M_ENTRY_GATE else " (direction only — not gating entries)"
    return (
        f"   {symbol}: 15M bias = {trend}{gate_note} [{mode}] — {rule}; "
        f"close {close:.2f} (vs EMA50 {close - ema50:+.1f}, EMA200 {close - ema200:+.1f}){ts_suffix}"
    )


def bias_15m_allows(direction: str, ctx_15m: Dict) -> bool:
    if not USE_15M_ENTRY_GATE or not USE_15M_BIAS:
        return True
    trend = ctx_15m.get("trend")
    want = "bullish" if direction == "long" else "bearish"
    if SOFT_15M_BIAS:
        if trend in (None, "neutral"):
            return True
        return trend == want
    return trend == want


def strong_trend_relaxed(direction: str, ctx_5m: Dict, ctx_15m: Dict) -> bool:
    """5M ADX >= threshold and 5M+15M hard-aligned — unlock MACD skip / lower min_rr."""
    if not STRONG_TREND_SKIP_MACD:
        return False
    if ctx_5m.get("adx", 0) < STRONG_TREND_RELAX_ADX:
        return False
    want = "bullish" if direction == "long" else "bearish"
    if ctx_5m.get("trend") != want:
        return False
    return ctx_15m.get("trend") == want


def entry_min_rr(direction: str, ctx_5m: Dict, ctx_15m: Dict) -> float:
    if strong_trend_relaxed(direction, ctx_5m, ctx_15m):
        return STRONG_TREND_MIN_RR
    return MIN_RR_AFTER_CAP


def round_bracket_prices(
    symbol: str, direction: str, sl: float, tp: float,
) -> Tuple[float, float]:
    """Round SL/TP to valid exchange ticks (MNQ/NQ tick_size=0.25)."""
    import math
    tick = SYMBOL_SPECS[symbol]["tick_size"]
    is_long = direction == "long"

    def _round_price(price: float, mode: str) -> float:
        if tick <= 0 or price <= 0:
            return price
        ticks = price / tick
        if mode == "up":
            rounded_ticks = math.ceil(ticks - 1e-9)
        elif mode == "down":
            rounded_ticks = math.floor(ticks + 1e-9)
        else:
            rounded_ticks = round(ticks)
        return round(rounded_ticks * tick, 6)

    sl_mode = "down" if is_long else "up"
    tp_mode = "up" if is_long else "down"
    return _round_price(sl, sl_mode), _round_price(tp, tp_mode)


def cap_sl_distance(symbol: str, sl_distance: float, contracts: int = CONTRACTS) -> float:
    """Cap SL so dollar risk never exceeds max_loss_per_trade (NQ: ~12.5 pts vs MNQ ~125 pts)."""
    max_pts = max_sl_points(symbol, contracts)
    return min(sl_distance, max_pts)


def spec_limit(symbol: str, key: str, default: float) -> float:
    return float(SYMBOL_SPECS.get(symbol, {}).get(key, default))


def max_sl_points(symbol: str, contracts: int = CONTRACTS) -> float:
    """Max SL distance in index points for capped dollar risk (1 NQ ct = 10× MNQ $/pt)."""
    pv = SYMBOL_SPECS[symbol]["point_value"]
    max_loss = spec_limit(symbol, "max_loss_per_trade", MAX_LOSS_PER_TRADE)
    return max_loss / (pv * contracts)


def apply_symbol_risk_overrides(cfg: Optional[Dict] = None) -> None:
    """Apply per-symbol $ risk from mnq_profit_config symbol_risk and {SYMBOL}_MAX_LOSS env vars."""
    overrides: Dict[str, Dict] = {}
    if cfg and isinstance(cfg.get("symbol_risk"), dict):
        for sym, vals in cfg["symbol_risk"].items():
            if isinstance(vals, dict):
                overrides[sym.upper()] = dict(vals)
    for sym in SYMBOL_SPECS:
        env_loss = os.getenv(f"{sym}_MAX_LOSS_PER_TRADE", "").strip()
        if env_loss:
            overrides.setdefault(sym, {})["max_loss_per_trade"] = float(env_loss)
        env_daily = os.getenv(f"{sym}_DAILY_LOSS_LIMIT", "").strip()
        if env_daily:
            overrides.setdefault(sym, {})["daily_loss_limit"] = float(env_daily)
    for sym, vals in overrides.items():
        if sym not in SYMBOL_SPECS:
            continue
        if "max_loss_per_trade" in vals:
            SYMBOL_SPECS[sym]["max_loss_per_trade"] = float(vals["max_loss_per_trade"])
        if "daily_loss_limit" in vals:
            SYMBOL_SPECS[sym]["daily_loss_limit"] = float(vals["daily_loss_limit"])


def symbol_risk_line(symbol: str, contracts: int = CONTRACTS) -> str:
    """One-line sizing summary for startup logs."""
    spec = SYMBOL_SPECS[symbol]
    pv = spec["point_value"]
    tick_usd = pv * spec["tick_size"]
    max_loss = spec_limit(symbol, "max_loss_per_trade", MAX_LOSS_PER_TRADE)
    daily = spec_limit(symbol, "daily_loss_limit", DAILY_LOSS_LIMIT)
    sl_cap = max_sl_points(symbol, contracts)
    return (
        f"{symbol}: ${pv:.0f}/pt (${tick_usd:.2f}/tick), "
        f"qty={contracts}, max ${max_loss:.0f}/trade (~{sl_cap:.1f} pt SL), "
        f"daily ${daily:.0f}"
    )


def calc_trade_dollars(signal: Dict, contracts: int = CONTRACTS) -> Dict[str, float]:
    """Risk and reward in dollars for a bracket trade."""
    pv = SYMBOL_SPECS[signal["symbol"]]["point_value"]
    entry, sl, tp = signal["entry"], signal["sl"], signal["tp"]
    if signal["direction"] == "long":
        risk_pts = entry - sl
        reward_pts = tp - entry
    else:
        risk_pts = sl - entry
        reward_pts = entry - tp
    risk_usd = abs(risk_pts) * pv * contracts
    reward_usd = abs(reward_pts) * pv * contracts
    rr = reward_usd / risk_usd if risk_usd > 0 else 0.0
    return {
        "risk_usd": risk_usd,
        "reward_usd": reward_usd,
        "rr": rr,
        "risk_pts": abs(risk_pts),
        "reward_pts": abs(reward_pts),
    }


def print_trade_money(signal: Dict, prefix: str = "   ") -> None:
    """Plain-English dollars at risk and profit target."""
    m = calc_trade_dollars(signal)
    sym = signal.get("symbol", "")
    suffix = ""
    if sym in SYMBOL_SPECS:
        suffix = f"  ({m['risk_pts']:.1f} pt SL, ${SYMBOL_SPECS[sym]['point_value']:.0f}/pt)"
    print(
        f"{prefix}💵 You could LOSE: ${m['risk_usd']:.0f}  |  "
        f"You could MAKE: ${m['reward_usd']:.0f}{suffix}"
    )


def bracket_risk_over_limit(
    symbol: str, direction: str, entry: float, sl: float, tp: float,
) -> Optional[str]:
    """Return error message if rounded bracket exceeds max_loss_per_trade."""
    max_loss = spec_limit(symbol, "max_loss_per_trade", MAX_LOSS_PER_TRADE)
    m = calc_trade_dollars({
        "symbol": symbol, "direction": direction, "entry": entry, "sl": sl, "tp": tp,
    })
    if m["risk_usd"] > max_loss + 1:
        return f"Rounded SL too wide (${m['risk_usd']:.0f} risk, max ${max_loss:.0f})"
    return None


load_profit_config()
print(f"Trading session: {SESSION_MODE.upper()} — {session_mode_label(SESSION_MODE)}")
if STRATEGY_MODE == "scalp_hybrid":
    print(f"Scalp windows: {format_session_windows(SCALP_SESSIONS)}")
print_profit_mode_banner()


def _cached_df(
    cached: Optional[pd.DataFrame],
    fallback: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return cached OHLCV frame without ambiguous DataFrame truthiness (`df or other`)."""
    if cached is not None and not cached.empty:
        return cached
    if fallback is not None and not fallback.empty:
        return fallback
    return pd.DataFrame()


def daily_loss_limit_for(symbols: List[str]) -> float:
    return max(spec_limit(s, 'daily_loss_limit', DAILY_LOSS_LIMIT) for s in symbols)


def prompt_symbol_choice() -> List[str]:
    """Interactive MNQ / NQ / both selection at startup."""
    print("\n" + "=" * 50)
    print("  Nasdaq contract — same strategy, different size")
    print("=" * 50)
    print("  1) MNQ  Micro  (~$0.50/tick, $2/pt)  — max ~125 pt SL @ $250 risk")
    print("  2) NQ   Full   (~$5.00/tick, $20/pt) — same $250 cap, ~12.5 pt SL max")
    print("  3) Both — scan MNQ + NQ, max 1 open position at a time")
    print("=" * 50)
    while True:
        choice = input("Choice [1]: ").strip().lower() or "1"
        if choice in ("1", "mnq"):
            return ["MNQ"]
        if choice in ("2", "nq"):
            return ["NQ"]
        if choice in ("3", "both", "mnq+nq", "mnq,nq"):
            return ["MNQ", "NQ"]
        print("  Please enter 1, 2, or 3")


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
    initial_sl: float = 0.0
    broker_bracket: bool = False  # True when Rithmic native SL/TP attached at entry
    breakeven_hit: bool = False
    secure_tightened: bool = False
    max_favorable_price: float = 0.0  # long: highest seen; short: lowest seen (MFE trail)
    entry_meta: Optional[Dict] = None  # hybrid journal fields (mode, adx, flow, trigger)


# ══════════════════════════════════════════════════════════════════════════════
#  Indicator Calculations
# ══════════════════════════════════════════════════════════════════════════════

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate ADX, DI+, DI-"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()
    
    return adx, plus_di, minus_di


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger(series: pd.Series, period: int = 20, std_dev: int = 2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    pctb = (series - lower) / (upper - lower + 1e-10)
    return upper, lower, pctb


def resample_1m_to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Build 5M OHLCV from 1M bars when Rithmic 5M history is short (e.g. Globex open)."""
    if df_1m is None or df_1m.empty or "datetime" not in df_1m.columns:
        return pd.DataFrame()
    d = df_1m.copy()
    d["datetime"] = pd.to_datetime(d["datetime"], utc=True)
    d = d.set_index("datetime").sort_index()
    df_5m = d.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna().reset_index()
    return df_5m


def merge_5m_candles(rithmic: pd.DataFrame, resampled: pd.DataFrame) -> pd.DataFrame:
    """Combine 5M series; Rithmic bars win on duplicate timestamps."""
    if rithmic is None or rithmic.empty:
        return resampled.copy() if resampled is not None else pd.DataFrame()
    if resampled is None or resampled.empty:
        return rithmic.copy()
    a = resampled.copy()
    b = rithmic.copy()
    for df in (a, b):
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    merged = pd.concat([a, b], ignore_index=True)
    merged = merged.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    return merged.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Live Trader Class
# ══════════════════════════════════════════════════════════════════════════════

class LiveMTFScalper:
    """Live trading with Multi-Timeframe Scalping Strategy via Rithmic."""
    
    def __init__(self, symbols: List[str] = None, paper_mode: bool = False, skip_confirm: bool = False):
        self.skip_confirm = skip_confirm
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.paper_mode = paper_mode
        
        # Validate symbols
        for sym in self.symbols:
            if sym not in SYMBOL_SPECS:
                raise ValueError(f"Unsupported symbol: {sym}")
        
        # State - track multiple positions per symbol (order_id -> Position)
        self.positions: Dict[str, Dict[str, Position]] = {}
        self.daily_pnl = 0.0
        self.daily_trades_by_symbol: Dict[str, int] = {sym: 0 for sym in self.symbols}
        self.current_date = None
        self.trades: List[Dict] = []
        self.loss_cooldown_until: Dict[str, datetime] = {}  # symbol -> UTC datetime
        self._symbol_consec_losses: Dict[str, int] = {sym: 0 for sym in self.symbols}
        self._consec_loss_pause_until: Dict[str, datetime] = {}
        
        # 5M context cache per symbol
        self.context_cache: Dict[str, Dict] = {}
        
        # Broker connector
        self.broker: Optional[RithmicConnector] = None
        self.llm_advisor = LLMTradeAdvisor()
        self.trade_learner = DeepSeekTradeLearner(learn_every_n=DEEPSEEK_LEARN_EVERY_N)
        self.adaptive_learner = AdaptiveLearner() if USE_ADAPTIVE_LEARNER else None
        self.news_bias = NewsBiasAdvisor()
        self.policy_scorer = PolicyScorer()
        self.smart_filters = MNQSmartFilters()
        self._news_bias_cache: Dict[str, Dict] = {}
        self._policy_cache: Dict[str, Dict] = {}
        self._df_1m_cache: Optional[pd.DataFrame] = None
        self._df_5m_cache: Optional[pd.DataFrame] = None
        self._df_15m_cache: Optional[pd.DataFrame] = None
        self._df_30s_cache: Dict[str, pd.DataFrame] = {}
        self._candle_fetch_warned: set = set()  # (symbol, tf) once per session
        self._last_entry_1m_bar: Dict[str, pd.Timestamp] = {}
        self._flow_regime_streak: Dict[str, int] = {}  # consecutive scans flow vs 5M EMA
        self._broker_flat_since: Dict[str, datetime] = {}
        self._broker_flat_scans: Dict[str, int] = {}
        self._place_order_lock = threading.Lock()
        self._pending_entries: Dict[str, int] = {}
        self._broker_untracked_block: Dict[str, bool] = {}
        self._scalp_state: Dict[str, ScalpSymbolState] = {}
        self._scalp_hybrid_state: Dict[str, ScalpHybridState] = {}
        self._fvs1_state: Dict[str, FVS1State] = {}
        self._fvs1_risk = FVS1RiskState()
        self._last_entry_30s_bar: Dict[str, pd.Timestamp] = {}
        self._last_30s_trigger_log_bar: Dict[str, pd.Timestamp] = {}
        self._hybrid_retry_30s_bar: Dict[str, pd.Timestamp] = {}
        self._last_entry_time: Dict[str, datetime] = {}

        # Log file
        mode = "paper" if paper_mode else "live"
        self.log_file = f'logs/{mode}_mtf_multi_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    def _adaptive_sl_mult(self, symbol: str) -> float:
        """Widen SL when learner sees repeated losses — keeps trading, reduces stop-outs."""
        if not self.adaptive_learner:
            return 1.0
        mult = float(self.adaptive_learner.sl_multiplier_by_pair.get(symbol, 1.0) or 1.0)
        if self.adaptive_learner.in_drawdown_protection:
            mult = max(mult, 1.1)
        if self._symbol_consec_losses.get(symbol, 0) >= 2:
            mult = max(mult, 1.05)
        return max(1.0, min(mult, 1.25))

    def _hybrid_params_with_learner(self, atr_1m: Optional[float] = None, symbol: str = "") -> Dict:
        """Hybrid kwargs with optional ADX floor and SL widen from learners."""
        sl_mult = self._adaptive_sl_mult(symbol) if symbol else 1.0
        sl_pts, tp_pts, _, _ = _scalp_bracket_for_atr(atr_1m, sl_mult=sl_mult)
        params = _scalp_hybrid_params(atr_1m)
        params["sl_pts"] = sl_pts
        params["tp_pts"] = tp_pts
        if STRATEGY_MODE == "scalp_hybrid" and self.trade_learner.blocking_active:
            pb, cont = self.trade_learner.get_adx_floor(
                params["adx_min_pullback"], params["adx_min_continuation"],
            )
            params["adx_min_pullback"] = pb
            params["adx_min_continuation"] = cont
        return params

    def _build_hybrid_entry_meta(
        self,
        signal: Dict,
        ctx_5m: Dict,
        flow_snap: Optional[Dict],
        row_30s: pd.Series,
        prev_30s: pd.Series,
    ) -> Dict:
        entry = float(signal["entry"])
        sl, tp = float(signal["sl"]), float(signal["tp"])
        direction = signal["direction"]
        if direction == "long":
            sl_pts, tp_pts = entry - sl, tp - entry
        else:
            sl_pts, tp_pts = sl - entry, entry - tp
        dir_int = 1 if direction == "long" else -1
        mode = str(signal.get("scalp_mode", "hybrid"))
        fired, trigger_reason = _trigger_eval(
            float(row_30s["open"]),
            float(row_30s["close"]),
            float(prev_30s["high"]),
            float(prev_30s["low"]),
            float(prev_30s["close"]),
            dir_int,
            h=float(row_30s["high"]),
            l=float(row_30s["low"]),
            aggressive=SCALP_AGGRESSIVE,
            setup_mode=mode,
            flow_snap=flow_snap,
            micro_break_pts=SCALP_TRIGGER_MICRO_BREAK_PTS,
            flow_trigger_delta_min=SCALP_FLOW_TRIGGER_DELTA_MIN,
            flow_trigger_buy_pct_long=SCALP_FLOW_TRIGGER_BUY_PCT_LONG,
            flow_trigger_buy_pct_short=SCALP_FLOW_TRIGGER_BUY_PCT_SHORT,
            flow_strong_delta_min=SCALP_FLOW_STRONG_DELTA_MIN,
            flow_strong_buy_pct_long=SCALP_FLOW_STRONG_BUY_PCT_LONG,
            flow_strong_buy_pct_short=SCALP_FLOW_STRONG_BUY_PCT_SHORT,
            trigger_flow_cfg=_scalp_trigger_flow_cfg(),
        )
        return {
            "entry_mode": mode,
            "adx": float(ctx_5m.get("adx", 0) or 0),
            "flow_delta": float(flow_snap.get("delta", 0) or 0) if flow_snap else 0.0,
            "buy_pct": float(flow_snap.get("buy_pct", 0.5) or 0.5) if flow_snap else 0.5,
            "trigger_reason": trigger_reason if fired else "",
            "sl_pts": round(sl_pts, 2),
            "tp_pts": round(tp_pts, 2),
            "hour": datetime.now(timezone.utc).hour,
        }

    def _deepseek_blocks_hybrid_entry(
        self,
        signal: Dict,
        ctx_5m: Dict,
        flow_snap: Optional[Dict],
        entry_meta: Dict,
    ) -> Optional[str]:
        if STRATEGY_MODE != "scalp_hybrid" or not self.trade_learner.blocking_active:
            return None
        ctx = {
            "direction": signal.get("direction"),
            "entry_mode": entry_meta.get("entry_mode"),
            "adx": entry_meta.get("adx", ctx_5m.get("adx")),
            "flow_delta": entry_meta.get("flow_delta"),
            "buy_pct": entry_meta.get("buy_pct"),
            "hour": entry_meta.get("hour"),
        }
        blocked, reason = self.trade_learner.check_entry_block(ctx)
        if blocked:
            print(f"   [BLOCK] AI: {reason}")
            return reason
        mode = str(entry_meta.get("entry_mode", ""))
        if self.trade_learner.is_mode_boosted(mode):
            boosts = self.trade_learner.advice.get("boost_modes") or []
            if boosts:
                bot_logger.info(f"[LEARN] boost mode {mode} (recent winners)")
        return None

    @staticmethod
    def _normalize_exit_reason(
        reason: str, pnl: float, position: Position, exit_price: float,
    ) -> str:
        r = (reason or "").upper()
        if position.breakeven_hit and pnl >= 0 and "SL" in r:
            return "BE"
        if r in ("TP", "TAKE_PROFIT"):
            return "TP"
        if r in ("SL", "MAX_LOSS"):
            return "SL"
        if r == "MAX_HOLD":
            return "TIME"
        if r in ("BROKER_BRACKET", "BROKER_SYNC"):
            if abs(exit_price - position.tp) <= abs(exit_price - position.sl):
                return "TP" if pnl > 0 else ("BE" if pnl == 0 else "SL")
            return "TP" if pnl > 0 else "SL"
        if pnl > 0:
            return "TP"
        if pnl < 0:
            return "SL"
        return reason

    def _record_hybrid_trade_close(
        self,
        position: Position,
        reason: str,
        exit_price: float,
        pnl: float,
    ) -> None:
        if STRATEGY_MODE != "scalp_hybrid" or not position.entry_meta:
            return
        meta = position.entry_meta
        exit_time = datetime.now(timezone.utc)
        hold_sec = (exit_time - position.entry_time).total_seconds()
        record = {
            "symbol": position.symbol,
            "direction": position.direction,
            "entry_mode": meta.get("entry_mode"),
            "adx": meta.get("adx"),
            "flow_delta": meta.get("flow_delta"),
            "buy_pct": meta.get("buy_pct"),
            "trigger_reason": meta.get("trigger_reason"),
            "sl_pts": meta.get("sl_pts"),
            "tp_pts": meta.get("tp_pts"),
            "hold_seconds": round(hold_sec, 1),
            "pnl": round(pnl, 2),
            "exit_reason": self._normalize_exit_reason(reason, pnl, position, exit_price),
            "hour": meta.get("hour", exit_time.hour),
            "entry_time": position.entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
        }
        self.trade_learner.record_trade(record)
        bot_logger.info(
            f"Journal: {position.symbol} {position.direction} {record['entry_mode']} "
            f"→ {record['exit_reason']} ${pnl:+.2f} ({hold_sec:.0f}s)"
        )

    def _record_adaptive_trade_close(
        self,
        position: Position,
        reason: str,
        exit_price: float,
        pnl: float,
    ) -> None:
        if not self.adaptive_learner:
            return
        exit_type = self._normalize_exit_reason(reason, pnl, position, exit_price)
        signal = "BUY" if position.direction == "long" else "SELL"
        meta = position.entry_meta or {}
        model_signals = {}
        if meta.get("entry_mode"):
            model_signals["hybrid_mode"] = {"signal": signal, "mode": meta.get("entry_mode")}
        self.adaptive_learner.record_trade({
            "pair": position.symbol,
            "signal": signal,
            "profit_loss": pnl,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "exit_type": exit_type,
            "model_signals": model_signals,
            "regime": self.adaptive_learner.current_regime,
        })

    def _update_consec_loss_state(self, symbol: str, pnl: float) -> None:
        if pnl < 0:
            count = self._symbol_consec_losses.get(symbol, 0) + 1
            self._symbol_consec_losses[symbol] = count
            if MTF_MAX_CONSEC_LOSSES > 0 and count >= MTF_MAX_CONSEC_LOSSES:
                until = datetime.now(timezone.utc) + timedelta(minutes=MTF_CONSEC_LOSS_PAUSE_MIN)
                self._consec_loss_pause_until[symbol] = until
                print(
                    f"   🛑 {symbol}: {count} consecutive losses — "
                    f"pause new entries {MTF_CONSEC_LOSS_PAUSE_MIN} min"
                )
        elif pnl > 0:
            self._symbol_consec_losses[symbol] = 0

    def _symbol_in_consec_loss_pause(self, symbol: str, *, quiet: bool = False) -> bool:
        until = self._consec_loss_pause_until.get(symbol)
        if until is None:
            return False
        now = datetime.now(timezone.utc)
        if now < until:
            if not quiet and VERBOSE_SKIP_REASONS:
                remaining = int((until - now).total_seconds() // 60) + 1
                print(
                    f"   ⏳ {symbol}: consecutive-loss pause — {remaining} min left "
                    f"({self._symbol_consec_losses.get(symbol, 0)} losses in a row)"
                )
            return True
        del self._consec_loss_pause_until[symbol]
        self._symbol_consec_losses[symbol] = 0
        return False

    def _adaptive_blocks_entry(self, symbol: str, *, verbose: bool = False) -> Optional[str]:
        if not self.adaptive_learner:
            return None
        learner = self.adaptive_learner
        if not adaptive_skip_enabled():
            if verbose:
                hour = datetime.now(timezone.utc).hour
                skip, reason = learner.should_skip_trade(
                    symbol, learner.current_regime, hour,
                )
                if skip:
                    print(f"      ℹ️ Adaptive advisory (skip OFF): {reason}")
                elif learner.should_skip_loss_pattern(symbol):
                    print(f"      ℹ️ Adaptive advisory (skip OFF): loss pattern on {symbol}")
            elif (
                learner.in_drawdown_protection
                and learner.consecutive_losses >= 3
            ):
                pass  # drawdown widens SL only — never blocks when skip disabled
            return None
        hour = datetime.now(timezone.utc).hour
        skip, reason = learner.should_skip_trade(symbol, learner.current_regime, hour)
        if skip:
            msg = f"Adaptive skip: {reason}"
            if verbose:
                print(f"      ❌ {msg}")
            return msg
        if learner.should_skip_loss_pattern(symbol):
            msg = f"Adaptive loss-pattern block on {symbol}"
            if verbose:
                print(f"      ❌ {msg}")
            return msg
        if (
            learner.in_drawdown_protection
            and learner.consecutive_losses >= 3
            and verbose
        ):
            print(
                f"      ⚠️ Drawdown protection — wider SL / higher ADX "
                f"({learner.current_drawdown_pct:.1f}% DD, "
                f"{learner.consecutive_losses} consec losses); still trading"
            )
        return None

    def _positions_for(self, symbol: str) -> List[Position]:
        return list(self.positions.get(symbol, {}).values())

    def _position_count(self, symbol: str, *, active_only: bool = True) -> int:
        bucket = self.positions.get(symbol, {})
        if not active_only:
            return len(bucket)
        return sum(
            1 for oid in bucket
            if not self._is_ghost_position(symbol, oid)
        )

    def _concurrent_exposure(self, symbol: str) -> Tuple[int, Dict[str, int]]:
        """Exposure for capacity: local tracked + list_positions net + working entries."""
        local = self._position_count(symbol)
        pending = self._pending_entries.get(symbol, 0)
        list_pos_net = 0
        broker_net = 0
        tag_inferred = 0
        open_entries = 0
        working = 0
        exposure_source = "flat"
        broker_ok = False
        orphan_cancelled = 0

        if not self.paper_mode and self.broker and self.broker.connected:
            signed = self.broker.get_symbol_list_positions_net(symbol)
            if signed is not None:
                list_pos_net = int(signed)
                broker_net = abs(list_pos_net)
                broker_ok = True
                if broker_net > 0:
                    exposure_source = "list_positions"
                elif list_pos_net == 0 and local == 0 and pending == 0:
                    if not self.broker.symbol_has_preserved_bot_orders(symbol):
                        orphan_cancelled = self.broker.cancel_all_bot_orders(symbol)
                        if orphan_cancelled:
                            bot_logger.info(
                                f"{symbol}: flat scan cancelled {orphan_cancelled} "
                                f"working bot_* order(s)"
                            )
            working = self.broker.count_working_entry_orders(symbol)

        if broker_ok:
            if broker_net > 0:
                total = max(local + pending, broker_net + working)
            else:
                # Flat: orphan SL/TP must not block new entries.
                total = local + pending + working
        else:
            total = local + pending + working

        untracked_broker = max(0, broker_net - local - pending) if broker_net > 0 else 0
        max_pos = max_positions_for(symbol)
        needs_reconcile = (
            not self.paper_mode
            and self.broker
            and self.broker.connected
            and (
                total >= max_pos
                or local > 0
                or untracked_broker > 0
                or orphan_cancelled > 0
                or (broker_ok and broker_net == 0 and self.broker.using_simulator_route)
            )
        )
        if needs_reconcile:
            report = self.broker.reconcile_symbol_exposure(symbol)
            broker_net = int(report.get("broker_net", broker_net))
            list_pos_net = int(report.get("list_positions_net", list_pos_net))
            open_entries = int(report.get("open_entries", 0))
            tag_inferred = int(report.get("tag_inferred_entries", 0))
            working = int(report.get("working_entries", working))
            exposure_source = str(report.get("exposure_source", exposure_source))
            untracked_broker = max(0, broker_net - local - pending) if broker_net > 0 else 0
            if (
                self.broker.using_simulator_route
                and broker_net > 0
                and (local > 0 or open_entries > 0 or tag_inferred > 0)
            ):
                tag_based = max(local + pending, open_entries + working, tag_inferred + working)
                if tag_based > 0 and broker_net > tag_based:
                    broker_net = tag_based
                untracked_broker = max(0, broker_net - local - pending) if broker_net > 0 else 0
            if broker_net > 0:
                total = max(local + pending, broker_net + working)
            elif self.broker.using_simulator_route and tag_inferred > 0:
                total = max(local + pending, tag_inferred + working)
            else:
                total = local + pending + working
            if (
                total > max_pos
                and broker_net == 0
                and tag_inferred == 0
                and working == 0
                and local > 0
            ):
                cleared = self._purge_stale_local_positions(symbol)
                if cleared:
                    local = self._position_count(symbol)
                    untracked_broker = max(0, broker_net - local - pending) if broker_net > 0 else 0
                    if broker_net > 0:
                        total = max(local + pending, broker_net + working)
                    else:
                        total = local + pending + working

        if (
            not self.paper_mode
            and self.broker
            and self.broker.connected
            and broker_ok
            and list_pos_net == 0
            and (local > 0 or tag_inferred > 0 or working > 0)
        ):
            ghosts = self._purge_stale_local_positions(symbol)
            if ghosts:
                local = self._position_count(symbol)
                if broker_net > 0:
                    total = max(local + pending, broker_net + working)
                else:
                    total = local + pending + working

        return total, {
            "local": local,
            "pending": pending,
            "broker_net": broker_net,
            "list_positions_net": list_pos_net,
            "open_entries": open_entries,
            "tag_inferred": tag_inferred,
            "working": working,
            "untracked_broker": untracked_broker,
            "exposure_source": exposure_source,
            "orphan_cancelled": orphan_cancelled,
        }

    def _log_position_capacity_block(self, symbol: str) -> None:
        max_pos = max_positions_for(symbol)
        total, detail = self._concurrent_exposure(symbol)
        untracked = int(detail.get("untracked_broker", 0))
        if BROKER_POSITION_BLOCK and untracked > 0:
            print(
                f"❌ {symbol} untracked broker exposure "
                f"({detail['broker_net']} contracts, local={detail['local']}) — "
                f"flatten on Rithmic before new entries"
            )
        else:
            print(f"❌ {symbol} max concurrent positions ({total}/{max_pos}) — skip entry")
        bot_logger.info(
            f"{symbol} entry blocked at {total}/{max_pos} "
            f"(local={detail['local']} pending={detail['pending']} "
            f"broker={detail['broker_net']} source={detail.get('exposure_source')} "
            f"untracked={untracked} tag_inferred={detail.get('tag_inferred', 0)} "
            f"working={detail['working']})"
        )

    def _has_position_capacity(self, symbol: str) -> bool:
        total, detail = self._concurrent_exposure(symbol)
        untracked = int(detail.get("untracked_broker", 0))
        if BROKER_POSITION_BLOCK and untracked > 0:
            return False
        return total < max_positions_for(symbol)

    def _enforce_position_capacity(self, symbol: str) -> bool:
        """Hard gate before place_order — broker net qty + local + working entries."""
        if self._has_position_capacity(symbol):
            return True
        self._log_position_capacity_block(symbol)
        return False

    def _flat_key(self, symbol: str, order_id: str) -> str:
        return f"{symbol}:{order_id}"

    def _add_position(self, position: Position) -> None:
        self.positions.setdefault(position.symbol, {})[str(position.order_id)] = position

    def _iter_tracked_positions(self):
        for symbol, bucket in self.positions.items():
            for order_id, position in bucket.items():
                yield symbol, order_id, position

    def connect(self) -> bool:
        """Initialize Rithmic connection (Yahoo fallback in paper mode without credentials)."""
        try:
            self.broker = RithmicConnector(live_mode=not self.paper_mode)
            self.broker._symbols_to_watch = list(self.symbols)
            if not self.paper_mode:
                self.broker._disable_yahoo_fallback = True
            self.broker.initialize()

            if not self.broker.connected:
                if self.paper_mode:
                    print("📝 PAPER MODE — no Rithmic credentials in .env")
                    print("   Using Yahoo Finance for market data (~15–20 min delay)")
                    print("   Orders are simulated locally (no broker connection)")
                    return True
                print("❌ Rithmic not connected - check credentials in .env")
                print(f"   RITHMIC_USER_ID: {os.getenv('RITHMIC_USER_ID') or 'NOT SET'}")
                print(f"   RITHMIC_SYSTEM: {os.getenv('RITHMIC_SYSTEM', 'NOT SET')}")
                if self.broker and self.broker._last_connect_error:
                    print(f"   Error: {self.broker._last_connect_error}")
                return False

            acct = self.broker.get_account_info()
            print(f"✅ Rithmic Connected!")
            print(f"   System: {acct.get('system', 'Unknown')}")
            print(f"   Account ID: {acct.get('account_id') or 'UNKNOWN'}")
            routes = acct.get("trade_routes") or {}
            if routes:
                print(f"   Trade routes: {routes}")
            else:
                print(f"   Trade route: (not resolved yet)")
            print(f"   Balance: ${acct.get('balance', 0):,.2f}")

            if self.paper_mode:
                print(f"   Mode: PAPER - Simulated orders (Rithmic data feed)")
            else:
                print(f"   Mode: ⚠️  LIVE - Real money!")
                if acct.get("simulator_route"):
                    allow_sim = (
                        os.getenv("RITHMIC_ALLOW_SIMULATOR", "false").lower() == "true"
                    )
                    acct_id = str(acct.get("account_id") or "")
                    if allow_sim and "PRO004" in acct_id.upper():
                        print(
                            "\n   ════════════════════════════════════════════════════════\n"
                            "   PRO004 SIM MODE — orders via Rithmic simulator until "
                            "Lucid enables live route\n"
                            "   ════════════════════════════════════════════════════════\n"
                        )
                    else:
                        print(
                            f"\n   🚨🚨🚨 ORDERS GOING TO SIMULATOR NOT LIVE 🚨🚨🚨\n"
                            f"   Set RITHMIC_TRADE_ROUTE in .env to your Lucid LIVE route "
                            f"(not 'simulator').\n"
                            f"   Optional: RITHMIC_ACCOUNT_ID for your funded account.\n"
                        )
                self._repair_open_positions_without_brackets()
                self._cleanup_orphan_orders_on_startup()
                self._reconcile_positions_on_startup()

            return True
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def get_candles_seconds(
        self,
        symbol: str,
        period_seconds: int = TRIGGER_BAR_SECONDS,
        count: int = 120,
        min_bars: int = 2,
        df_1m: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch sub-minute bars from Rithmic (SECOND_BAR, tick, or 1M-derived fallback)."""
        if not self.broker:
            return None
        try:
            df = self.broker.get_candles_seconds(
                symbol,
                period_seconds=period_seconds,
                num_candles=count,
                df_1m_fallback=df_1m,
            )
            if df is not None and len(df) >= min_bars:
                return df
        except Exception as e:
            print(f"❌ Error getting {symbol} {period_seconds}s candles: {e}")
        return None

    def _resolve_30s_trigger_rows(
        self,
        symbol: str,
        row_1m: pd.Series,
        df_1m: pd.DataFrame,
    ) -> Tuple[Optional[pd.Series], Optional[pd.Series], str]:
        """Return (row_30s, prev_30s, source_label) for hybrid/scalp trigger logic."""
        df_30s = self._df_30s_cache.get(symbol)
        if df_30s is not None and len(df_30s) >= 2:
            return df_30s.iloc[-1], df_30s.iloc[-2], "30s"
        if SCALP_30S_FALLBACK_1M and df_1m is not None and len(df_1m) >= 2:
            return row_1m, df_1m.iloc[-2], "1M"
        return None, None, "none"

    def _warm_30s_bars(self) -> None:
        """Load sub-minute trigger bars at startup when USE_30S_BARS is enabled."""
        if not USE_30S_BARS or not self.broker:
            return
        print(f"\n📊 Loading {TRIGGER_BAR_SECONDS}s trigger bars...")
        max_attempts = 1 if self.paper_mode else 3
        for sym in self.symbols:
            df_1m = self.get_candles(
                sym, 1, count=120, min_bars=2, min_bars_floor=1,
            )
            df = None
            for attempt in range(max_attempts):
                df = self.get_candles_seconds(sym, count=120, min_bars=2, df_1m=df_1m)
                if df is not None and len(df) >= 2:
                    break
                src = getattr(self.broker, "second_bar_source", "unknown")
                if attempt < max_attempts - 1:
                    print(
                        f"   {sym}: {0 if df is None else len(df)} bars "
                        f"(source={src}) — retry {attempt + 1}/{max_attempts}"
                    )
                    time.sleep(2 + attempt)
            n = 0 if df is None else len(df)
            src = getattr(self.broker, "second_bar_source", "unknown")
            print(f"   {sym}: {n} bars (source={src})")
            if df is not None and n >= 2:
                self._df_30s_cache[sym] = self.add_30s_indicators(df)
                for _, row in df.tail(5).iterrows():
                    print(
                        f"      {row['datetime']}  "
                        f"O={row['open']:.2f} H={row['high']:.2f} "
                        f"L={row['low']:.2f} C={row['close']:.2f} "
                        f"V={int(row['volume'])}"
                    )
            elif src == "ticks":
                print(
                    "      tick fallback active — history fills as LAST_TRADE "
                    "ticks stream in (Lucid may not expose SECOND_BAR history)"
                )
            elif src == "1m_derived":
                print(
                    f"      using 1M-derived {TRIGGER_BAR_SECONDS}s fallback "
                    f"(SECOND_BAR unavailable at session open)"
                )
            elif SCALP_30S_FALLBACK_1M:
                print(
                    "      30s bars still empty — hybrid will use 1M trigger "
                    "fallback (SCALP_30S_FALLBACK_1M=true)"
                )
            else:
                print(
                    "      ⚠️ 0 trigger bars — entries blocked until SECOND_BAR or "
                    "ticks populate (or set SCALP_30S_FALLBACK_1M=true)"
                )
        print()

    def _warn_candle_fetch_once(self, symbol: str, timeframe_minutes: int, msg: str) -> None:
        key = (symbol, timeframe_minutes)
        if key in self._candle_fetch_warned:
            return
        self._candle_fetch_warned.add(key)
        print(msg)

    def _derive_1m_from_30s(
        self,
        symbol: str,
        count: int,
        df_30s: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        """Aggregate cached or fresh 30s bars into 1M when Rithmic MINUTE_BAR fails."""
        if df_30s is None:
            df_30s = self._df_30s_cache.get(symbol)
        if df_30s is None or len(df_30s) < 4:
            df_30s = self.get_candles_seconds(
                symbol, count=max(count * 2, 120), min_bars=4,
            )
        if df_30s is None or len(df_30s) < 4:
            return None
        derived = resample_subminute_to_1m(df_30s, TRIGGER_BAR_SECONDS)
        if derived is None or len(derived) < 1:
            return None
        out = derived.tail(count).reset_index(drop=True)
        self._warn_candle_fetch_once(
            symbol, 1,
            f"⚡ {symbol}: 1M from {len(df_30s)}×{TRIGGER_BAR_SECONDS}s bars "
            f"→ {len(out)}×1M (Rithmic MINUTE_BAR unavailable)",
        )
        return out

    def get_candles(
        self,
        symbol: str,
        timeframe_minutes: int,
        count: int = 200,
        min_bars: int = 50,
        min_bars_floor: Optional[int] = None,
        deep_history: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Fetch candles from Rithmic for a specific symbol."""
        max_attempts = 3 if not self.paper_mode else 1
        last_len = 0
        last_df: Optional[pd.DataFrame] = None
        for attempt in range(max_attempts):
            try:
                if (
                    deep_history
                    and not self.paper_mode
                    and hasattr(self.broker, "get_candles_deep")
                ):
                    df = self.broker.get_candles_deep(
                        symbol,
                        timeframe_minutes=timeframe_minutes,
                        num_candles=count,
                        lookback_hours=CANDLE_HISTORY_HOURS,
                    )
                else:
                    df = self.broker.get_candles(
                        symbol, timeframe_minutes=timeframe_minutes, num_candles=count
                    )
                last_len = 0 if df is None else len(df)
                if df is not None and last_len > 0:
                    last_df = df
                if df is not None and last_len >= min_bars:
                    return df
                if not self.paper_mode:
                    self._warn_candle_fetch_once(
                        symbol, timeframe_minutes,
                        f"⚠️ Rithmic returned {last_len} bars for {symbol} "
                        f"({timeframe_minutes}m, need>={min_bars}) — "
                        f"retry {attempt + 1}/{max_attempts}",
                    )
                    time.sleep(1 + attempt)
            except Exception as e:
                print(f"❌ Error getting {symbol} candles: {e}")
                if not self.paper_mode and attempt < max_attempts - 1:
                    time.sleep(1 + attempt)
                    continue
                return None
        floor = min_bars_floor if min_bars_floor is not None else min_bars
        if last_df is not None and last_len >= floor:
            if not self.paper_mode and last_len < min_bars:
                self._warn_candle_fetch_once(
                    symbol, timeframe_minutes,
                    f"⚠️ {symbol}: using {last_len} Rithmic {timeframe_minutes}m bars "
                    f"(ideal>={min_bars}, floor>={floor}) — session-open mode",
                )
            return last_df
        if timeframe_minutes == 1 and not self.paper_mode:
            derived = self._derive_1m_from_30s(symbol, count)
            if derived is not None:
                dlen = len(derived)
                if dlen >= min_bars:
                    return derived
                if dlen >= floor:
                    self._warn_candle_fetch_once(
                        symbol, 1,
                        f"⚠️ {symbol}: using {dlen} 1M bars from 30s aggregation "
                        f"(ideal>={min_bars}, floor>={floor}) — session-open mode",
                    )
                    return derived
        if not self.paper_mode:
            self._warn_candle_fetch_once(
                symbol, timeframe_minutes,
                f"❌ LIVE: insufficient Rithmic candles for {symbol} "
                f"({timeframe_minutes}m, got {last_len}, need>={min_bars}) — "
                f"NO Yahoo fallback",
            )
        return None

    def _resolve_5m_candles(
        self, symbol: str, df_1m: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        """Fetch 5M bars; resample from 1M or accept partial history at Globex open."""
        scalp = STRATEGY_MODE in ("scalp_b", "scalp_hybrid", "fvs1")
        ideal = MIN_5M_BARS_SCALP_IDEAL if scalp else MIN_5M_BARS_BASELINE
        floor = MIN_5M_BARS_SCALP_FLOOR if scalp else MIN_5M_BARS_BASELINE
        count = RESISTANCE_LOOKBACK + 50 if not scalp else max(ideal + 10, 40)

        df_5m = self.get_candles(
            symbol, timeframe_minutes=5, count=count,
            min_bars=ideal, min_bars_floor=floor if scalp else ideal,
        )

        if df_5m is not None and scalp and len(df_5m) < ideal:
            if df_1m is not None and len(df_1m) >= ideal * 5:
                rs = resample_1m_to_5m(df_1m)
                if len(rs) > 0:
                    n_before = len(df_5m)
                    df_5m = merge_5m_candles(df_5m, rs)
                    if len(df_5m) > n_before:
                        print(
                            f"⚠️ {symbol}: enriched 5M {n_before}→{len(df_5m)} bars "
                            f"(Rithmic + 1M resample from {len(df_1m)}×1M, ideal>={ideal})"
                        )
            elif df_1m is not None and len(df_1m) >= floor * 5:
                rs = resample_1m_to_5m(df_1m)
                if len(rs) > 0:
                    n_before = len(df_5m)
                    df_5m = merge_5m_candles(df_5m, rs)
                    if len(df_5m) > n_before:
                        print(
                            f"⚠️ {symbol}: enriched 5M {n_before}→{len(df_5m)} bars "
                            f"(Rithmic + 1M resample, ideal>={ideal})"
                        )

        if df_5m is not None and len(df_5m) >= (floor if scalp else ideal):
            return df_5m

        if df_1m is not None and len(df_1m) >= floor * 5:
            rs = resample_1m_to_5m(df_1m)
            if len(rs) >= floor:
                print(
                    f"⚠️ {symbol}: Rithmic 5M unavailable — "
                    f"using {len(rs)} bars resampled from 1M (ideal>={ideal})"
                )
                return rs

        if scalp:
            df_5m = self.get_candles(
                symbol, timeframe_minutes=5, count=count,
                min_bars=floor, min_bars_floor=floor,
            )
            if df_5m is not None and df_1m is not None and len(df_1m) >= floor * 5:
                rs = resample_1m_to_5m(df_1m)
                if len(rs) > 0:
                    n_before = len(df_5m)
                    df_5m = merge_5m_candles(df_5m, rs)
                    if len(df_5m) > n_before:
                        print(
                            f"⚠️ {symbol}: enriched 5M {n_before}→{len(df_5m)} bars "
                            f"(Rithmic + 1M resample, ideal>={ideal})"
                        )
        return df_5m
    
    def add_1m_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 1M entry indicators."""
        df = df.copy()
        
        # EMAs
        df['ema_9'] = calculate_ema(df['close'], ENTRY_EMA_FAST)
        df['ema_21'] = calculate_ema(df['close'], ENTRY_EMA_SLOW)
        if STRATEGY_MODE in ("scalp_b", "scalp_hybrid", "fvs1") or scalp_fast_mode_active():
            df['ema_20'] = calculate_ema(df['close'], 20)
        
        # ATR
        df['atr'] = calculate_atr(df, ATR_PERIOD)
        
        # RSI
        df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
        df['rsi_prev'] = df['rsi'].shift(1)
        
        # MACD
        _, _, df['macd_hist'] = calculate_macd(df['close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        df['macd_hist_prev'] = df['macd_hist'].shift(1)
        
        # Volume ratio
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
        
        # Bollinger %B
        _, _, df['bb_pctb'] = calculate_bollinger(df['close'], BB_PERIOD, BB_STD)
        df['vwap'] = compute_vwap(df)
        if scalp_fast_mode_active():
            df['adx'], df['di_plus'], df['di_minus'] = calculate_adx(df, ADX_PERIOD)
        
        return df
    
    def add_5m_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 5M trend indicators."""
        df = df.copy()
        
        # Trend EMAs
        df['ema_50'] = calculate_ema(df['close'], TREND_EMA_FAST)
        df['ema_200'] = calculate_ema(df['close'], TREND_EMA_SLOW)
        
        # ADX
        df['adx'], df['di_plus'], df['di_minus'] = calculate_adx(df, ADX_PERIOD)
        
        # ATR
        df['atr'] = calculate_atr(df, ATR_PERIOD)
        if STRATEGY_MODE in ("scalp_b", "scalp_hybrid", "fvs1"):
            df['ema_20'] = calculate_ema(df['close'], 20)
            df['vwap'] = compute_vwap(df)

        return df

    def add_30s_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 30s body stats for chase protection."""
        return add_30s_body_stats(df)
    
    def get_1m_trend_context(self, df_1m: pd.DataFrame) -> Dict:
        """Fast-scalp trend from 1M VWAP + EMA20 + ADX (no 5M wait)."""
        min_bars = max(MIN_5M_BARS_SCALP_FLOOR, 20)
        if df_1m is None or len(df_1m) < min_bars:
            return {
                'trend': None, 'adx': 0, 'adx_rising': False,
                'di_plus': 0, 'di_minus': 0, 'resistance': 0, 'support': 0,
            }
        recent = df_1m.tail(max(RESISTANCE_LOOKBACK, min_bars))
        row = recent.iloc[-1]
        row_prev = recent.iloc[-2] if len(recent) >= 2 else row
        close = float(row['close'])
        vwap = float(row.get('vwap', float('nan')))
        ema20 = float(row.get('ema_20', float('nan')))
        if pd.isna(vwap):
            trend = None
        elif close > vwap and (pd.isna(ema20) or close > ema20):
            trend = 'bullish'
        elif close < vwap and (pd.isna(ema20) or close < ema20):
            trend = 'bearish'
        else:
            trend = 'neutral'
        adx_rising = float(row.get('adx', 0)) > float(row_prev.get('adx', 0))
        return {
            'trend': trend,
            'adx': float(row.get('adx', 0)),
            'adx_rising': adx_rising,
            'di_plus': float(row.get('di_plus', 0)),
            'di_minus': float(row.get('di_minus', 0)),
            'atr': float(row.get('atr', 0)),
            'resistance': float(recent['high'].max()),
            'support': float(recent['low'].min()),
        }

    def _use_1m_trend_row(
        self, df_1m: pd.DataFrame, df_5m: Optional[pd.DataFrame],
    ) -> bool:
        if not scalp_fast_mode_active():
            return False
        if SCALP_AGGRESSIVE:
            return True
        bars_5m = len(df_5m) if df_5m is not None else 0
        return bars_5m < MIN_5M_BARS_SCALP_IDEAL

    def _resolve_trend_rows(
        self,
        df_1m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame],
    ) -> Tuple[pd.Series, Optional[pd.Series], bool]:
        """Return (row_trend, prev_trend, from_1m) for hybrid/FVS1 gate rows."""
        if self._use_1m_trend_row(df_1m, df_5m):
            row = df_1m.iloc[-1]
            prev = df_1m.iloc[-2] if len(df_1m) >= 2 else None
            return row, prev, True
        if df_5m is not None and len(df_5m):
            row = df_5m.iloc[-1]
            prev = df_5m.iloc[-2] if len(df_5m) >= 2 else None
            return row, prev, False
        if self._df_5m_cache is not None and len(self._df_5m_cache):
            row = self._df_5m_cache.iloc[-1]
            prev = self._df_5m_cache.iloc[-2] if len(self._df_5m_cache) >= 2 else None
            return row, prev, False
        row = df_1m.iloc[-1]
        return row, None, True

    def get_5m_context(self, df_5m: pd.DataFrame) -> Dict:
        """Get current 5M trend context."""
        if df_5m is None or len(df_5m) < RESISTANCE_LOOKBACK:
            return {'trend': None, 'adx': 0, 'adx_rising': False, 'di_plus': 0, 'di_minus': 0, 'resistance': 0, 'support': 0}
        
        recent = df_5m.tail(RESISTANCE_LOOKBACK)
        row = recent.iloc[-1]
        row_prev = recent.iloc[-2] if len(recent) >= 2 else row
        
        # Trend direction
        if row['ema_50'] > row['ema_200']:
            trend = 'bullish'
        elif row['ema_50'] < row['ema_200']:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        # ADX direction (rising = trend strengthening, falling = trend weakening)
        adx_rising = row['adx'] > row_prev['adx']
        
        # Swing high/low for resistance/support
        resistance = recent['high'].max()
        support = recent['low'].min()
        
        return {
            'trend': trend,
            'adx': row['adx'],
            'adx_rising': adx_rising,
            'di_plus': row['di_plus'],
            'di_minus': row['di_minus'],
            'atr': row['atr'],
            'resistance': resistance,
            'support': support,
        }
    
    def get_15m_context(self, df_15m: Optional[pd.DataFrame]) -> Dict:
        """Get 15M higher-timeframe bias on last 15M bar (mode from BIAS_15M_MODE)."""
        if df_15m is None or len(df_15m) < 50:
            return {'trend': None, 'adx': 0}
        row = df_15m.iloc[-1]
        close = float(row['close'])
        ema50 = float(row['ema_50'])
        ema200 = float(row['ema_200'])
        trend = compute_15m_trend(close, ema50, ema200, BIAS_15M_MODE, BIAS_15M_BUFFER_PTS)
        bar_time = row.get('datetime')
        if bar_time is not None and not pd.isna(bar_time):
            try:
                bar_time = pd.Timestamp(bar_time).tz_convert('US/Eastern').strftime('%H:%M ET')
            except Exception:
                bar_time = str(bar_time)
        else:
            bar_time = None
        return {
            'trend': trend,
            'adx': row.get('adx', 0),
            'close': close,
            'ema_50': ema50,
            'ema_200': ema200,
            'bar_time': bar_time,
            'bias_mode': BIAS_15M_MODE,
        }
    
    def _daily_trades_for(self, symbol: str) -> int:
        return self.daily_trades_by_symbol.get(symbol, 0)

    def _increment_daily_trades(self, symbol: str) -> None:
        self.daily_trades_by_symbol[symbol] = self._daily_trades_for(symbol) + 1
        self._last_entry_time[symbol] = datetime.now(timezone.utc)

    def _symbol_trade_limit_reached(self, symbol: str) -> bool:
        return self._daily_trades_for(symbol) >= max_trades_per_day_for(symbol)

    def _sweep_orphan_bot_orders(self) -> None:
        """Periodic cleanup: cancel all working bot_* orders when broker is flat."""
        if self.paper_mode or not self.broker or not self.broker.connected:
            return
        for symbol in self.symbols:
            try:
                signed = self.broker.get_symbol_list_positions_net(symbol)
                if signed is None or signed != 0:
                    continue
                local = self._position_count(symbol)
                pending = self._pending_entries.get(symbol, 0)
                if local > 0 or pending > 0:
                    continue
                cancelled = self.broker.cancel_all_bot_orders(symbol)
                if cancelled:
                    bot_logger.info(
                        f"{symbol}: orphan sweep cancelled {cancelled} working bot_* order(s)"
                    )
                    self.broker.reconcile_symbol_exposure(symbol)
                    self._purge_stale_local_positions(symbol)
            except Exception as e:
                bot_logger.warning(f"{symbol} orphan sweep failed: {e}")

    def _cleanup_orphan_orders_on_startup(self) -> None:
        """Cancel stale protective orders when broker order count exceeds threshold."""
        if not self.broker or self.paper_mode or ORPHAN_ORDER_THRESHOLD <= 0:
            return
        try:
            result = self.broker.purge_orphan_protective_orders(
                self.symbols, threshold=ORPHAN_ORDER_THRESHOLD,
            )
            for sym, info in (result or {}).items():
                cancelled = info.get("cancelled", 0)
                before = info.get("order_count", 0)
                if cancelled:
                    print(
                        f"🧹 {sym}: cancelled {cancelled} orphan working order(s) "
                        f"(had {before}, threshold {ORPHAN_ORDER_THRESHOLD})"
                    )
        except Exception as e:
            print(f"⚠️  Orphan order cleanup failed: {e}")

    def _force_clear_stale_position(self, symbol: str, order_id: str) -> bool:
        """Remove a ghost local position when broker is flat."""
        bucket = self.positions.get(symbol)
        if not bucket or order_id not in bucket:
            return False
        position = bucket[order_id]
        flat_key = self._flat_key(symbol, order_id)
        quote = self.broker.get_latest_price(symbol) if self.broker else None
        exit_price = (
            self._resolve_current_price(quote)
            if quote
            else position.entry_price
        )
        if self.broker:
            self.broker.acknowledge_flat_position(order_id, symbol)
        self._reset_broker_flat_tracking(flat_key)
        self.close_position(symbol, order_id, "BROKER_SYNC", exit_price)
        return True

    def _purge_stale_local_positions(self, symbol: str) -> int:
        """Drop local positions when broker reconcile shows flat exposure."""
        if self.paper_mode or not self.broker:
            return 0
        report = self.broker.reconcile_symbol_exposure(symbol)
        open_ids = set(report.get("open_order_ids") or [])
        broker_net = int(report.get("broker_net", 0))
        open_entries = int(report.get("open_entries", 0))
        working = int(report.get("working_entries", 0))
        cleared = 0

        if broker_net == 0 and open_entries == 0 and working == 0:
            for order_id in list(self.positions.get(symbol, {}).keys()):
                if self._force_clear_stale_position(symbol, order_id):
                    cleared += 1
            return cleared

        for order_id in list(self.positions.get(symbol, {}).keys()):
            if order_id not in open_ids:
                if self._force_clear_stale_position(symbol, order_id):
                    cleared += 1
        return cleared

    def _reconcile_positions_on_startup(self) -> None:
        """On connect: align local position dict with list_positions + list_orders."""
        if self.paper_mode or not self.broker or not self.broker.connected:
            return
        print("🔄 Reconciling local positions with Rithmic...")
        for symbol in self.symbols:
            try:
                report = self._reconcile_symbol_startup(symbol)
                stale = int(report.get("stale_cleared", 0))
                broker_net = int(report.get("broker_net", 0))
                open_entries = int(report.get("open_entries", 0))
                working = int(report.get("working_entries", 0))
                legs_cancelled = int(report.get("legs_cancelled", 0))
                local_before = len(self.positions.get(symbol, {}))
                ghosts = self._purge_stale_local_positions(symbol)
                local_after = len(self.positions.get(symbol, {}))
                if stale or ghosts or local_before or legs_cancelled:
                    print(
                        f"   {symbol}: broker_net={broker_net} open_entries={open_entries} "
                        f"working={working} | local {local_before}→{local_after} "
                        f"(stale_orders={stale} legs_cancelled={legs_cancelled} "
                        f"ghosts_cleared={ghosts})"
                    )
                elif broker_net > 0 and local_after == 0:
                    src = str(report.get("exposure_source", "list_positions"))
                    self._broker_untracked_block[symbol] = True
                    print(
                        f"   ⚠️  {symbol}: Rithmic list_positions net={broker_net} "
                        f"({src}) but bot tracks 0 — entries blocked until flattened"
                    )
                    print(
                        f"      Flatten on Rithmic UI or: "
                        f"python scripts/rithmic_flatten_symbol.py {symbol}"
                    )
                elif int(report.get("tag_inferred_entries", 0)) > 0 and local_after == 0:
                    tag_n = int(report.get("tag_inferred_entries", 0))
                    print(
                        f"   ⚠️  {symbol}: {tag_n} stale bot_* fills inferred "
                        f"(list_positions flat) — run: "
                        f"python scripts/rithmic_flatten_symbol.py {symbol}"
                    )
            except Exception as e:
                print(f"   ⚠️  {symbol} reconcile failed: {e}")
                bot_logger.warning(f"{symbol} startup reconcile failed: {e}")

    def _reconcile_symbol_startup(self, symbol: str) -> Dict:
        """Reconcile once; if flat but tag-inferred exposure remains, purge orphans and retry."""
        report = self.broker.reconcile_symbol_exposure(symbol)
        cancelled = self.broker.cancel_all_bot_orders(symbol)
        if cancelled:
            bot_logger.info(
                f"{symbol} startup: cancelled {cancelled} orphan stop(s) before re-reconcile"
            )
            report = self.broker.reconcile_symbol_exposure(symbol)
        broker_net = int(report.get("broker_net", 0))
        tag_n = int(report.get("tag_inferred_entries", 0))
        if broker_net == 0 and tag_n > 0 and ORPHAN_ORDER_THRESHOLD > 0:
            purge = self.broker.purge_orphan_protective_orders(
                [symbol], threshold=1,
            )
            cancelled = int((purge.get(symbol) or {}).get("cancelled", 0))
            if cancelled:
                bot_logger.info(
                    f"{symbol} startup: purged {cancelled} orphan order(s) "
                    f"before re-reconcile (tag_inferred={tag_n})"
                )
            report = self.broker.reconcile_symbol_exposure(symbol)
        return report

    def _repair_open_positions_without_brackets(self) -> None:
        """Attach broker SL/TP to any live position missing native brackets."""
        if not self.broker or self.paper_mode:
            return
        try:
            self.broker._refresh_positions()
            with self.broker._state_lock:
                open_syms = {
                    sym for sym, pos in self.broker._positions.items()
                    if pos.get("size", 0) != 0
                }
            for sym in open_syms:
                tracked = self.positions.get(sym, {})
                if tracked:
                    for order_id, pos in tracked.items():
                        self._ensure_broker_brackets(sym, pos)
                    continue
                # Bot restarted but Rithmic still has exposure — cannot infer SL/TP
                print(
                    f"⚠️  LIVE: {sym} open on Rithmic but not tracked by bot — "
                    f"place SL/TP manually or restart bot before next entry"
                )
        except Exception as e:
            print(f"⚠️  Bracket repair check failed: {e}")

    def _repair_all_open_positions(self, *, entry_verify: bool = False) -> None:
        """Ensure every tracked live position has broker SL/TP attached."""
        if self.paper_mode or not self.broker or not self.broker.connected:
            return
        for sym, bucket in list(self.positions.items()):
            for order_id, pos in list(bucket.items()):
                self._ensure_broker_brackets(sym, pos, entry_verify=entry_verify)

    def _ensure_broker_brackets(
        self, symbol: str, position: Position, *, entry_verify: bool = False,
    ) -> bool:
        """Ensure SL/TP exist on Rithmic; returns True when broker-protected."""
        if self.paper_mode or not self.broker or not self.broker.connected:
            return position.broker_bracket
        side = "BUY" if position.direction == "long" else "SELL"
        max_attempts = (
            BROKER_PROTECTION_MAX_RETRIES if entry_verify else SCAN_PROTECTION_REPAIR_RETRIES
        )

        pos_open = self.broker._broker_symbol_has_position(symbol)
        if pos_open is False:
            return position.broker_bracket
        if pos_open is None:
            bot_logger.warning(
                f"{symbol}: broker position unknown — attempting protective repair anyway"
            )

        sl_ok, tp_ok = self.broker.query_broker_protection(
            position.order_id,
            stop_loss=position.sl,
            take_profit=position.tp,
        )
        if not (sl_ok and tp_ok):
            missing = []
            if position.sl > 0 and not sl_ok:
                missing.append("SL")
            if position.tp > 0 and not tp_ok:
                missing.append("TP")
            alert = (
                f"⚠️ UNPROTECTED — re-submitting {'/'.join(missing) or 'SL/TP'} "
                f"for {symbol} (order {position.order_id})"
            )
            print(f"   {alert}")
            bot_logger.warning(alert)

        sl_ok, tp_ok = self.broker.aggressive_repair_protection(
            ticket=position.order_id,
            symbol=symbol,
            side=side,
            size=position.size,
            stop_loss=position.sl,
            take_profit=position.tp,
            max_attempts=max_attempts,
        )
        if sl_ok and tp_ok:
            position.broker_bracket = True
            print(
                f"   ✅ VERIFIED: SL @ {position.sl:.2f} | TP @ {position.tp:.2f} on Rithmic"
            )
        elif sl_ok:
            position.broker_bracket = True
            msg = (
                f"⚠️ {symbol}: SL verified @ {position.sl:.2f} but TP still MISSING "
                f"— stop-protected; retrying TP each scan"
            )
            print(f"   {msg}")
            bot_logger.warning(msg)
        else:
            msg = (
                f"🚨 CRITICAL {symbol}: broker protection FAILED — "
                f"SL={'yes' if sl_ok else 'MISSING'} TP={'yes' if tp_ok else 'MISSING'} — "
                f"place manually on Rithmic NOW"
            )
            print(f"   {msg}")
            bot_logger.error(msg)
            send_email(
                f"UNPROTECTED POSITION: {symbol}",
                f"{msg}\n\nOrder: {position.order_id}\n"
                f"Direction: {position.direction}\nEntry: {position.entry_price:.2f}\n"
                f"Expected SL: {position.sl:.2f}\nExpected TP: {position.tp:.2f}",
            )
        return sl_ok

    def _reset_broker_flat_tracking(self, flat_key: str) -> None:
        self._broker_flat_since.pop(flat_key, None)
        self._broker_flat_scans.pop(flat_key, None)

    def _broker_flat_duration_sec(self, flat_key: str) -> float:
        flat_since = self._broker_flat_since.get(flat_key)
        if not flat_since:
            return 0.0
        return (datetime.now(timezone.utc) - flat_since).total_seconds()

    def _should_force_sync_ghost(self, flat_key: str) -> bool:
        if flat_key not in self._broker_flat_since:
            return False
        return (
            self._broker_flat_duration_sec(flat_key) >= BROKER_FLAT_SYNC_SEC
            or self._broker_flat_scans.get(flat_key, 0) >= BROKER_FLAT_SYNC_SCANS
        )

    def _is_ghost_position(self, symbol: str, order_id: str) -> bool:
        """Broker flat while bot still tracks an open bracket position."""
        bucket = self.positions.get(symbol)
        if not bucket or order_id not in bucket or self.paper_mode or not self.broker:
            return False
        position = bucket[order_id]
        if not position.broker_bracket:
            return False
        pos_open = self.broker._broker_symbol_has_position(symbol)
        if pos_open is not False:
            return False
        flat_key = self._flat_key(symbol, order_id)
        return self._should_force_sync_ghost(flat_key)

    def _sync_broker_closed_position(self, symbol: str, order_id: str) -> bool:
        """Detect position closed by broker bracket; update bot state."""
        bucket = self.positions.get(symbol)
        if not bucket or order_id not in bucket or not self.broker:
            return False
        position = bucket[order_id]
        if not position.broker_bracket:
            return False
        if self.broker.entry_protection_grace_active(order_id):
            return False

        flat_key = self._flat_key(symbol, order_id)
        fill_info = self.broker.confirm_bracket_exit_fill(
            order_id,
            symbol,
            position.sl,
            position.tp,
            entry_price=position.entry_price,
        )
        if fill_info and fill_info.get("confirmed"):
            exit_price = float(fill_info.get("exit_price") or 0)
            if exit_price <= 0:
                if fill_info.get("leg") == "TP":
                    exit_price = position.tp
                elif fill_info.get("leg") == "SL":
                    exit_price = position.sl
                else:
                    quote = self.broker.get_latest_price(symbol)
                    exit_price = (
                        self._resolve_current_price(quote) if quote else position.entry_price
                    )
            leg = fill_info.get("leg", "?")
            route = fill_info.get("trade_route", "")
            acct_id = fill_info.get("account_id", "")
            fill_src = "broker fill price" if fill_info.get("fill_from_broker") else "inferred price"
            print(
                f"   ✅ Broker-confirmed bracket exit: {leg} @ {exit_price:.2f} "
                f"({fill_src}, order {order_id}, account={acct_id} route={route})"
            )
            self.broker.log_execution_diagnostics(
                symbol, order_id=order_id, context="bracket_exit",
            )
            self.broker.acknowledge_flat_position(order_id, symbol)
            self._reset_broker_flat_tracking(flat_key)
            self.close_position(symbol, order_id, "BROKER_BRACKET", exit_price)
            return True

        pos_open = self.broker._broker_symbol_has_position(symbol)
        if pos_open is None:
            bot_logger.info(
                f"Keeping {symbol} order {order_id} — broker exposure unknown this scan"
            )
            return False
        if pos_open:
            self._reset_broker_flat_tracking(flat_key)
            return False

        now = datetime.now(timezone.utc)
        if flat_key not in self._broker_flat_since:
            self._broker_flat_since[flat_key] = now
        self._broker_flat_scans[flat_key] = self._broker_flat_scans.get(flat_key, 0) + 1
        flat_duration = self._broker_flat_duration_sec(flat_key)
        flat_scans = self._broker_flat_scans[flat_key]
        force_sync = self._should_force_sync_ghost(flat_key)

        fill_info = self.broker.confirm_bracket_exit_fill(
            order_id,
            symbol,
            position.sl,
            position.tp,
            force_sync=force_sync,
            flat_duration_sec=flat_duration,
            entry_price=position.entry_price,
        )
        if not fill_info or not fill_info.get("confirmed"):
            if force_sync:
                bot_logger.warning(
                    f"{symbol}: broker flat {flat_duration:.0f}s ({flat_scans} scans) — "
                    f"force-sync still pending (order {order_id})"
                )
            else:
                sim_note = ""
                if getattr(self.broker, "using_simulator_route", False):
                    sim_note = (
                        " — list_positions often empty on simulator; "
                        "using entry-fill + SL/TP state"
                    )
                bot_logger.info(
                    f"{symbol}: broker flat {flat_duration:.0f}s — waiting for bracket fill "
                    f"({BROKER_FLAT_SYNC_SEC:.0f}s / {flat_scans}/{BROKER_FLAT_SYNC_SCANS} scans, "
                    f"order {order_id}){sim_note}"
                )
            return False

        exit_price = float(fill_info.get("exit_price") or 0)
        if exit_price <= 0:
            if fill_info.get("leg") == "TP":
                exit_price = position.tp
            elif fill_info.get("leg") == "SL":
                exit_price = position.sl
            else:
                quote = self.broker.get_latest_price(symbol)
                exit_price = (
                    self._resolve_current_price(quote) if quote else position.entry_price
                )
        leg = fill_info.get("leg", "?")
        sync_reason = fill_info.get("sync_reason", "BROKER_BRACKET")
        close_reason = sync_reason if fill_info.get("inferred") else "BROKER_BRACKET"
        route = fill_info.get("trade_route", "")
        acct_id = fill_info.get("account_id", "")
        fill_src = "broker fill price" if fill_info.get("fill_from_broker") else "inferred price"
        if fill_info.get("inferred"):
            detail = fill_info.get("detail") or fill_src
            print(
                f"   🔄 Ghost position cleared: {close_reason} ({leg}) @ {exit_price:.2f} "
                f"— {detail} (flat {flat_duration:.0f}s, order {order_id}, "
                f"account={acct_id} route={route})"
            )
        else:
            print(
                f"   ✅ Broker-confirmed bracket exit: {leg} @ {exit_price:.2f} "
                f"({fill_src}, order {order_id}, account={acct_id} route={route})"
            )
        self.broker.log_execution_diagnostics(
            symbol, order_id=order_id, context="bracket_exit",
        )
        self.broker.acknowledge_flat_position(order_id, symbol)
        self._reset_broker_flat_tracking(flat_key)
        self.close_position(symbol, order_id, close_reason, exit_price)
        return True

    def _sync_all_broker_closed_positions(self, symbol: str) -> bool:
        """Sync any broker-closed positions for symbol; returns True if any closed."""
        closed = False
        for order_id in list(self.positions.get(symbol, {}).keys()):
            if self._sync_broker_closed_position(symbol, order_id):
                closed = True
        return closed

    def check_daily_limits(self) -> bool:
        """Check if daily limits allow trading."""
        today = datetime.now().date()
        if self.current_date != today:
            self.current_date = today
            self.daily_pnl = 0.0
            self.daily_trades_by_symbol = {sym: 0 for sym in self.symbols}
            self.loss_cooldown_until.clear()
            self._symbol_consec_losses = {sym: 0 for sym in self.symbols}
            self._consec_loss_pause_until.clear()
            if self.adaptive_learner:
                self.adaptive_learner.decay_loss_patterns()
            print(f"\n📅 New trading day: {today}")

        daily_limit = daily_loss_limit_for(self.symbols)
        if self.daily_pnl <= -daily_limit:
            print(f"🛑 Daily loss limit hit: ${self.daily_pnl:.2f}")
            return False
        if DAILY_HALF_STOP_ENABLED and self.daily_pnl <= -(daily_limit * DAILY_HALF_STOP_PCT):
            half_stop = daily_limit * DAILY_HALF_STOP_PCT
            print(
                f"🛑 Daily half-stop ({DAILY_HALF_STOP_PCT:.0%} of ${daily_limit:.0f}): "
                f"${self.daily_pnl:.2f} — no new trades today"
            )
            return False
        return True

    def _symbol_in_loss_cooldown(self, symbol: str, *, quiet: bool = False) -> bool:
        """Block new entries on symbol for LOSS_COOLDOWN_MINUTES after SL/MAX_LOSS/broker stop-out."""
        until = self.loss_cooldown_until.get(symbol)
        if until is None:
            return False
        now = datetime.now(timezone.utc)
        if now < until:
            remaining = int((until - now).total_seconds() // 60) + 1
            if not quiet and VERBOSE_SKIP_REASONS:
                print(f"   ⏳ {symbol}: loss cooldown — {remaining} min left (no re-entry)")
            return True
        del self.loss_cooldown_until[symbol]
        return False

    def _loss_cooldown_minutes_left(self, symbol: str) -> int:
        until = self.loss_cooldown_until.get(symbol)
        if until is None:
            return 0
        now = datetime.now(timezone.utc)
        if now >= until:
            return 0
        return int((until - now).total_seconds() // 60) + 1

    def _update_flow_regime_streak(self, symbol: str, ctx_5m: Dict, flow_snap: Optional[Dict]) -> int:
        """Count consecutive scans where flow contradicts 5M EMA trend."""
        trend = ctx_5m.get("trend")
        if trend not in ("bullish", "bearish"):
            self._flow_regime_streak[symbol] = 0
            return 0
        if flow_contradicts_5m_trend(trend, flow_snap, self._flow_counter_config()):
            self._flow_regime_streak[symbol] = self._flow_regime_streak.get(symbol, 0) + 1
        else:
            self._flow_regime_streak[symbol] = 0
        return self._flow_regime_streak.get(symbol, 0)

    def _flow_counter_config(self) -> Dict:
        fc = dict(FLOW_COUNTER_CFG)
        fc["di_counter"] = DI_COUNTER_TREND
        fc["counter_adx"] = COUNTER_ADX
        fc["counter_trend_shorts"] = COUNTER_TREND_SHORTS
        fc["counter_trend_longs"] = COUNTER_TREND_LONGS
        return fc

    def _build_gate_eval_context(self, symbol: str) -> GateEvalContext:
        fc = self._flow_counter_config()
        return GateEvalContext(
            adx_threshold=ADX_THRESHOLD,
            di_tolerance=DI_TOLERANCE,
            di_flow_tolerance=DI_FLOW_TOLERANCE,
            di_relax_strength=DI_RELAX_STRENGTH,
            flow_relax_strength=FLOW_RELAX_STRENGTH,
            flow_relax_long_buy_pct=FLOW_RELAX_LONG_BUY_PCT,
            use_order_flow=USE_ORDER_FLOW,
            use_flow_di_override=USE_FLOW_DI_OVERRIDE,
            use_flow_adx_relax=USE_FLOW_ADX_RELAX,
            flow_adx_relax=FLOW_ADX_RELAX,
            flow_entry_guard=FLOW_ENTRY_GUARD,
            flow_counter_cfg=fc,
            use_15m_bias=USE_15M_BIAS,
            use_15m_entry_gate=USE_15M_ENTRY_GATE,
            candle_confirmation=CANDLE_CONFIRMATION,
            volume_ratio_threshold=VOLUME_RATIO_THRESHOLD,
            bb_extreme_low=BB_EXTREME_LOW,
            bb_extreme_high=BB_EXTREME_HIGH,
            vwap_required=VWAP_REQUIRED,
            entry_quality=ENTRY_QUALITY,
            session_mode=SESSION_MODE,
            max_1m_bar_pts=max_1m_bar_pts_for(symbol),
            strategy_mode=STRATEGY_MODE,
            effective_adx_fn=effective_adx_threshold,
            adaptive_rsi_long_fn=adaptive_rsi_long,
            adaptive_rsi_short_fn=adaptive_rsi_short,
            adaptive_pullback_atr_fn=adaptive_pullback_atr,
            adaptive_vwap_ok_fn=adaptive_vwap_ok,
            bias_15m_allows_fn=bias_15m_allows,
            strong_trend_relaxed_fn=strong_trend_relaxed,
        )

    def _log_30s_trigger_eval(
        self,
        symbol: str,
        df_1m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame],
        flow_snap: Optional[Dict],
    ) -> None:
        """Log 30s OHLC + trigger result on each new 30s bar (hybrid scalp)."""
        if STRATEGY_MODE != "scalp_hybrid":
            return
        df_30s = self._df_30s_cache.get(symbol)
        if df_30s is None or len(df_30s) < 2:
            return
        row = df_1m.iloc[-1]
        row_30s = df_30s.iloc[-1]
        prev_30s = df_30s.iloc[-2]
        row_5m, _, _ = self._resolve_trend_rows(df_1m, df_5m)
        use_1m_trend = self._use_1m_trend_row(df_1m, df_5m)
        ctx = self.get_1m_trend_context(df_1m) if use_1m_trend else self.get_5m_context(df_5m)
        trend = ctx.get("trend") or "none"
        direction = 1 if trend == "bullish" else -1 if trend == "bearish" else 0
        if direction == 0:
            return
        st = self._scalp_hybrid_state.get(symbol, ScalpHybridState())
        phase, mode, _ = (
            (st.long_phase, st.long_mode, None)
            if direction == 1
            else (st.short_phase, st.short_mode, None)
        )
        setup_mode = mode if phase == 1 else "burst" if SCALP_AGGRESSIVE and SCALP_MOMENTUM_BURST else ""
        trig_params = _scalp_hybrid_params(float(row.get("atr", 0) or 0))
        print(
            f"   ⚡ {symbol}: {format_30s_trigger_log(
                row_30s, prev_30s, direction,
                aggressive=SCALP_AGGRESSIVE,
                setup_mode=setup_mode,
                flow_snap=flow_snap,
                micro_break_pts=trig_params.get('micro_break_pts'),
                flow_trigger_delta_min=trig_params.get('flow_trigger_delta_min'),
                flow_trigger_buy_pct_long=trig_params.get('flow_trigger_buy_pct_long'),
                flow_trigger_buy_pct_short=trig_params.get('flow_trigger_buy_pct_short'),
                flow_strong_delta_min=trig_params.get('flow_strong_delta_min'),
                flow_strong_buy_pct_long=trig_params.get('flow_strong_buy_pct_long'),
                flow_strong_buy_pct_short=trig_params.get('flow_strong_buy_pct_short'),
                trigger_flow_cfg=trig_params.get('trigger_flow_cfg'),
            )}"
        )

    def _explain_entry_blocked(
        self,
        symbol: str,
        row,
        flow_streak: Optional[int] = None,
    ) -> Optional[str]:
        """First global gate blocking new entries (hybrid live logs)."""
        if self._symbol_in_loss_cooldown(symbol, quiet=True):
            mins = self._loss_cooldown_minutes_left(symbol)
            return f"loss cooldown ({mins} min left)"
        if self._symbol_in_consec_loss_pause(symbol, quiet=True):
            return "consecutive-loss pause"
        adaptive_block = self._adaptive_blocks_entry(symbol, verbose=False)
        if adaptive_block:
            return adaptive_block
        if MIN_SECONDS_BETWEEN_ENTRIES > 0:
            last = self._last_entry_time.get(symbol)
            if last is not None:
                elapsed = (datetime.now(timezone.utc) - last).total_seconds()
                if elapsed < MIN_SECONDS_BETWEEN_ENTRIES:
                    wait = int(MIN_SECONDS_BETWEEN_ENTRIES - elapsed)
                    return f"entry spacing ({wait}s left)"
        open_count, _ = self._concurrent_exposure(symbol)
        max_pos = max_positions_for(symbol)
        if open_count >= max_pos:
            return f"max positions ({open_count}/{max_pos})"
        atr = row.get("atr")
        if pd.isna(atr) or atr <= 0:
            return "warming up (no ATR yet)"
        candle_range = float(row["high"]) - float(row["low"])
        max_bar_pts = max_1m_bar_pts_for(symbol)
        if candle_range > max_bar_pts:
            return (
                f"1M bar too wide ({candle_range:.0f} pts, max {max_bar_pts:.0f})"
            )
        return None

    def _hybrid_trigger_status(
        self,
        symbol: str,
        df_1m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame],
        flow_snap: Optional[Dict],
    ) -> Tuple[bool, int, str]:
        """Return (trigger_fired, direction, reason) for the current 30s bar."""
        df_30s = self._df_30s_cache.get(symbol)
        if df_30s is None or len(df_30s) < 2:
            return False, 0, ""
        row = df_1m.iloc[-1]
        row_30s = df_30s.iloc[-1]
        prev_30s = df_30s.iloc[-2]
        use_1m_trend = self._use_1m_trend_row(df_1m, df_5m)
        ctx = self.get_1m_trend_context(df_1m) if use_1m_trend else self.get_5m_context(df_5m)
        trend = ctx.get("trend") or "none"
        direction = 1 if trend == "bullish" else -1 if trend == "bearish" else 0
        if direction == 0:
            return False, 0, "no trend"
        st = self._scalp_hybrid_state.get(symbol, ScalpHybridState())
        phase, mode, _ = (
            (st.long_phase, st.long_mode, None)
            if direction == 1
            else (st.short_phase, st.short_mode, None)
        )
        setup_mode = mode if phase == 1 else "burst" if SCALP_AGGRESSIVE and SCALP_MOMENTUM_BURST else ""
        trig_params = _scalp_hybrid_params(float(row.get("atr", 0) or 0))
        fired, reason = _trigger_eval(
            float(row_30s["open"]),
            float(row_30s["close"]),
            float(prev_30s["high"]),
            float(prev_30s["low"]),
            float(prev_30s["close"]),
            direction,
            h=float(row_30s["high"]),
            l=float(row_30s["low"]),
            aggressive=SCALP_AGGRESSIVE,
            setup_mode=setup_mode,
            flow_snap=flow_snap,
            micro_break_pts=trig_params.get("micro_break_pts"),
            flow_trigger_delta_min=trig_params.get("flow_trigger_delta_min"),
            flow_trigger_buy_pct_long=trig_params.get("flow_trigger_buy_pct_long"),
            flow_trigger_buy_pct_short=trig_params.get("flow_trigger_buy_pct_short"),
            flow_strong_delta_min=trig_params.get("flow_strong_delta_min"),
            flow_strong_buy_pct_long=trig_params.get("flow_strong_buy_pct_long"),
            flow_strong_buy_pct_short=trig_params.get("flow_strong_buy_pct_short"),
            trigger_flow_cfg=trig_params.get("trigger_flow_cfg"),
        )
        return fired, direction, reason

    def _should_run_30s_entry_check(
        self,
        symbol: str,
        row_30s,
    ) -> Tuple[bool, bool]:
        """(run_entry_check, is_new_30s_bar). Retries same bar after trigger miss."""
        ts = self._latest_bar_ts(row_30s)
        is_new = self._is_new_30s_bar(symbol, row_30s)
        if is_new:
            return True, True
        if ts is not None and self._hybrid_retry_30s_bar.get(symbol) == ts:
            return True, False
        return False, False

    def _is_new_30s_trigger_log_bar(self, symbol: str, row) -> bool:
        ts = self._latest_bar_ts(row)
        if ts is None:
            return True
        prev = self._last_30s_trigger_log_bar.get(symbol)
        if prev is None:
            return True
        return prev != ts

    def _mark_30s_trigger_logged(self, symbol: str, row) -> None:
        ts = self._latest_bar_ts(row)
        if ts is not None:
            self._last_30s_trigger_log_bar[symbol] = ts

    def _log_hybrid_entry_miss(
        self,
        symbol: str,
        df_1m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame],
        ctx_5m: Dict,
        flow_snap: Optional[Dict],
        *,
        trigger_reason: str = "",
        flow_streak: Optional[int] = None,
    ) -> None:
        """Log why trigger=yes did not produce an order (always on in hybrid mode)."""
        if STRATEGY_MODE != "scalp_hybrid":
            return
        row = df_1m.iloc[-1]
        gate = self._explain_entry_blocked(symbol, row, flow_streak=flow_streak)
        if gate:
            print(
                f"   ⚠️ {symbol}: trigger yes ({trigger_reason}) but no entry — "
                f"gate: {gate}"
            )
            return
        row_30s, prev_30s, _ = self._resolve_30s_trigger_rows(symbol, row, df_1m)
        if row_30s is None or prev_30s is None:
            print(f"   ⚠️ {symbol}: trigger yes but no entry — no 30s bars")
            return
        row_5m, _, _ = self._resolve_trend_rows(df_1m, df_5m)
        use_1m_trend = self._use_1m_trend_row(df_1m, df_5m)
        ctx = self.get_1m_trend_context(df_1m) if use_1m_trend else ctx_5m
        trend = ctx.get("trend") or "none"
        direction = 1 if trend == "bullish" else -1 if trend == "bearish" else 0
        if direction == 0:
            print(f"   ⚠️ {symbol}: trigger yes but no entry — no trend direction")
            return
        hybrid_params = self._hybrid_params_with_learner(float(row.get("atr", 0) or 0), symbol)
        reason = hybrid_trigger_no_signal_reason(
            row,
            row_5m,
            row_30s,
            prev_30s,
            direction=direction,
            aggressive_mode=bool(hybrid_params.get("aggressive_mode")),
            adx_min_pullback=hybrid_params.get("adx_min_pullback", SCALP_ADX_MIN),
            trend_mode=hybrid_params.get("trend_mode", SCALP_TREND_MODE),
            chase_body_mult=hybrid_params.get("chase_body_mult", SCALP_CHASE_BODY_MULT),
            chase_ema_atr=hybrid_params.get("chase_ema_atr", SCALP_CHASE_EMA_ATR),
            flow_snap=flow_snap,
            micro_break_pts=hybrid_params.get("micro_break_pts", SCALP_TRIGGER_MICRO_BREAK_PTS),
            flow_trigger_delta_min=hybrid_params.get(
                "flow_trigger_delta_min", SCALP_FLOW_TRIGGER_DELTA_MIN,
            ),
            flow_trigger_buy_pct_long=hybrid_params.get(
                "flow_trigger_buy_pct_long", SCALP_FLOW_TRIGGER_BUY_PCT_LONG,
            ),
            flow_trigger_buy_pct_short=hybrid_params.get(
                "flow_trigger_buy_pct_short", SCALP_FLOW_TRIGGER_BUY_PCT_SHORT,
            ),
            flow_strong_delta_min=hybrid_params.get(
                "flow_strong_delta_min", SCALP_FLOW_STRONG_DELTA_MIN,
            ),
            flow_strong_buy_pct_long=hybrid_params.get(
                "flow_strong_buy_pct_long", SCALP_FLOW_STRONG_BUY_PCT_LONG,
            ),
            flow_strong_buy_pct_short=hybrid_params.get(
                "flow_strong_buy_pct_short", SCALP_FLOW_STRONG_BUY_PCT_SHORT,
            ),
            chase_flow_cfg=hybrid_params.get("chase_flow_cfg"),
            trigger_flow_cfg=hybrid_params.get("trigger_flow_cfg"),
        )
        print(
            f"   ⚠️ {symbol}: trigger yes ({trigger_reason}) but no entry — "
            f"{reason or 'unknown gate'}"
        )

    def _print_flat_scan_line(
        self,
        symbol: str,
        df_1m: pd.DataFrame,
        ctx_5m: Dict,
        *,
        df_5m: Optional[pd.DataFrame] = None,
        flow_snap: Optional[Dict] = None,
        new_trigger_bar: bool = True,
        open_count: int = 0,
        max_pos: int = 0,
    ) -> None:
        """One-line ⚡ summary when flat — always on in fast/aggressive scalp modes."""
        use_1m_trend = self._use_1m_trend_row(df_1m, df_5m)
        ctx = self.get_1m_trend_context(df_1m) if use_1m_trend else ctx_5m
        trend = ctx.get("trend") or "none"
        adx = float(ctx.get("adx", 0) or 0)
        bars_1m = len(df_1m) if df_1m is not None else 0
        bars_5m = len(df_5m) if df_5m is not None else 0
        tf_label = "1M" if use_1m_trend else "5M"
        retry_ts = self._hybrid_retry_30s_bar.get(symbol)
        row_30s_for_retry = None
        if symbol in self._df_30s_cache and len(self._df_30s_cache[symbol]) >= 1:
            row_30s_for_retry = self._df_30s_cache[symbol].iloc[-1]
        retry_pending = (
            row_30s_for_retry is not None
            and retry_ts is not None
            and self._latest_bar_ts(row_30s_for_retry) == retry_ts
        )
        if not new_trigger_bar and retry_pending:
            block = "retry entry (trigger pending)"
        elif not new_trigger_bar:
            block = "waiting new 30s bar"
        else:
            block = "evaluating entry"
        if STRATEGY_MODE == "scalp_hybrid" and (new_trigger_bar or retry_pending):
            row = df_1m.iloc[-1]
            gate = self._explain_entry_blocked(symbol, row)
            if gate:
                block = f"block: {gate}"
            else:
                row_5m, prev_5m, _ = self._resolve_trend_rows(df_1m, df_5m)
                row_30s, prev_30s, _ = self._resolve_30s_trigger_rows(symbol, row, df_1m)
                st = self._scalp_hybrid_state.get(symbol, ScalpHybridState())
                block = hybrid_block_summary(
                    row, row_5m, prev_5m, row_30s, prev_30s, st,
                    flow_snap=flow_snap,
                    **self._hybrid_params_with_learner(
                        float(row.get("atr", 0) or 0), symbol,
                    ),
                )
        flow_note = ""
        if flow_snap is not None:
            delta = float(flow_snap.get("delta", 0) or 0)
            buy_pct = float(flow_snap.get("buy_pct", 0.5) or 0.5)
            flow_note = f" | flow Δ{delta:+.0f} buy%={buy_pct:.0%}"
        print(
            f"   ⚡ {symbol}: {tf_label} trend={trend} adx={adx:.0f} | "
            f"1M={bars_1m} 5M={bars_5m} | Open {open_count}/{max_pos} | "
            f"{block}{flow_note}"
        )

    def _print_entry_gate_diagnostics(
        self,
        symbol: str,
        df_1m: pd.DataFrame,
        ctx_5m: Dict,
        ctx_15m: Dict,
        flow_snap: Optional[Dict],
        *,
        new_1m_bar: bool,
        flow_streak: Optional[int] = None,
        df_5m: Optional[pd.DataFrame] = None,
    ) -> None:
        """Print scannable PASS/BLOCK gate lines every scan (not gated on position capacity)."""
        open_count, _ = self._concurrent_exposure(symbol)
        max_pos = max_positions_for(symbol)
        flat = open_count == 0
        row = df_1m.iloc[-1]
        if flat and (
            FAST_SCAN_LOG
            or scalp_fast_mode_active()
            or SCALP_AGGRESSIVE
            or STRATEGY_MODE in ("scalp_b", "scalp_hybrid", "fvs1")
        ):
            has_30s = (
                symbol in self._df_30s_cache
                and len(self._df_30s_cache[symbol]) >= 2
            )
            new_trigger = new_1m_bar
            if USE_30S_BARS and has_30s and STRATEGY_MODE in ("scalp_b", "scalp_hybrid", "fvs1"):
                new_trigger = self._is_new_30s_bar(
                    symbol, self._df_30s_cache[symbol].iloc[-1],
                )
            self._print_flat_scan_line(
                symbol, df_1m, ctx_5m, df_5m=df_5m, flow_snap=flow_snap,
                new_trigger_bar=new_trigger, open_count=open_count, max_pos=max_pos,
            )

        if not FULL_TRADE_DIAGNOSTICS and not VERBOSE_SKIP_REASONS:
            return

        prev_1m = df_1m.iloc[-2] if len(df_1m) >= 2 else None
        if flow_streak is None:
            flow_streak = self._update_flow_regime_streak(symbol, ctx_5m, flow_snap)
        open_count, detail = self._concurrent_exposure(symbol)
        max_pos = max_positions_for(symbol)
        untracked = int(detail.get("untracked_broker", 0))
        daily_limit = daily_loss_limit_for(self.symbols)
        daily_ok = self.daily_pnl > -daily_limit
        half_stop = (
            DAILY_HALF_STOP_ENABLED
            and self.daily_pnl <= -(daily_limit * DAILY_HALF_STOP_PCT)
        )
        atr = row.get("atr")
        atr_ok = not pd.isna(atr) and float(atr) > 0
        candle_range = float(row["high"]) - float(row["low"])
        max_bar_pts = max_1m_bar_pts_for(symbol)
        vol_ok = candle_range <= max_bar_pts

        scalp_mode = STRATEGY_MODE in ("scalp_b", "scalp_hybrid", "fvs1")
        bars_5m = len(df_5m) if df_5m is not None else 0
        bars_1m = len(df_1m) if df_1m is not None else 0
        ideal_5m = MIN_5M_BARS_SCALP_IDEAL if scalp_mode else MIN_5M_BARS_BASELINE
        use_1m_trend = scalp_mode and self._use_1m_trend_row(df_1m, df_5m)
        skip_5m_warming = use_1m_trend or (
            scalp_fast_mode_active() and bars_1m >= MIN_5M_BARS_SCALP_FLOOR
        )
        warming_5m = scalp_mode and not skip_5m_warming and bars_5m < ideal_5m

        if not (FULL_TRADE_DIAGNOSTICS or VERBOSE_SKIP_REASONS):
            return

        print(f"   📋 Entry gates ({symbol}) — mode={STRATEGY_MODE}:")
        for line in evaluate_global_gates(
            symbol,
            row,
            daily_limit_ok=daily_ok and not half_stop,
            daily_pnl=self.daily_pnl,
            daily_limit=daily_limit,
            trade_limit_reached=self._symbol_trade_limit_reached(symbol),
            trade_count=self._daily_trades_for(symbol),
            trade_limit=max_trades_per_day_for(symbol),
            loss_cooldown_min=self._loss_cooldown_minutes_left(symbol),
            open_count=open_count,
            max_positions=max_pos,
            untracked_broker=untracked,
            broker_block=BROKER_POSITION_BLOCK,
            atr_ok=atr_ok,
            volatility_ok=vol_ok,
            candle_range=candle_range,
            max_bar_pts=max_bar_pts,
            new_1m_bar=new_1m_bar,
            daily_half_stop=half_stop,
            bars_5m=bars_5m if scalp_mode else None,
            bars_5m_ideal=ideal_5m if scalp_mode else None,
            bars_5m_floor=MIN_5M_BARS_SCALP_FLOOR if scalp_mode else None,
            skip_5m_warming=skip_5m_warming,
            use_1m_trend=use_1m_trend,
            bars_1m=bars_1m,
        ):
            print(line)

        if warming_5m:
            return

        row_5m, prev_5m, from_1m = self._resolve_trend_rows(df_1m, df_5m)
        if from_1m and use_1m_trend:
            print(f"   ⚡ fast mode — hybrid gates use 1M trend row (not 5M)")

        if STRATEGY_MODE == "scalp_hybrid":
            df_30s = self._df_30s_cache.get(symbol)
            row_30s, prev_30s, _trig_src = self._resolve_30s_trigger_rows(
                symbol, row, df_1m,
            )
            st = self._scalp_hybrid_state.get(symbol, ScalpHybridState())
            long_short_lines, _ = evaluate_hybrid_gates(
                row, row_5m, prev_5m, row_30s, prev_30s, st,
                **_scalp_hybrid_params(float(row.get("atr", 0) or 0)),
            )
            for line in long_short_lines:
                print(line)
            return

        if STRATEGY_MODE == "fvs1" and FVS1_CFG is not None:
            row_30s, prev_30s, _ = self._resolve_30s_trigger_rows(symbol, row, df_1m)
            st = self._fvs1_state.get(symbol, FVS1State())
            try:
                now_et = pd.Timestamp(row.get("datetime", datetime.now(timezone.utc))).tz_convert("US/Eastern")
            except Exception:
                now_et = datetime.now(pytz.timezone("US/Eastern"))
            flow_snap = None
            if USE_ORDER_FLOW and self.broker:
                flow_snap = self.broker.get_tick_flow(symbol)
            df_30s = self._df_30s_cache.get(symbol)
            long_short_lines, _ = evaluate_fvs1_gates(
                df_1m, row_5m, row_30s, prev_30s, st, FVS1_CFG,
                df_30s=df_30s, now_et=now_et, flow_snap=flow_snap, risk=self._fvs1_risk,
            )
            for line in long_short_lines:
                print(line)
            return

        if STRATEGY_MODE == "scalp_b":
            df_30s = self._df_30s_cache.get(symbol)
            row_30s = df_30s.iloc[-1] if df_30s is not None and len(df_30s) else None
            prev_30s = df_30s.iloc[-2] if df_30s is not None and len(df_30s) >= 2 else None
            st = self._scalp_state.get(symbol, ScalpSymbolState())
            sl_pts, tp_pts, _, _ = _scalp_bracket_for_atr(float(row.get("atr", 0) or 0))
            long_short_lines, _ = evaluate_scalp_gates(
                row, row_5m, prev_1m, st,
                row_30s=row_30s,
                prev_30s=prev_30s,
                adx_min=SCALP_ADX_MIN,
                pullback_atr=SCALP_PULLBACK_ATR,
                setup_bars=SCALP_SETUP_BARS,
                sl_pts=sl_pts,
                tp_pts=tp_pts,
            )
            for line in long_short_lines:
                print(line)
            return

        gate_cfg = self._build_gate_eval_context(symbol)
        for line in evaluate_long_gates(
            row, ctx_5m, ctx_15m, flow_snap, flow_streak, gate_cfg, df_1m=df_1m,
        ):
            print(line)
        for line in evaluate_short_gates(
            row, ctx_5m, ctx_15m, flow_snap, flow_streak, gate_cfg, df_1m=df_1m,
        ):
            print(line)
    
    def check_long_entry(
        self,
        row_1m: pd.Series,
        ctx_5m: Dict,
        ctx_15m: Dict,
        verbose: bool = False,
        df_1m: Optional[pd.DataFrame] = None,
        flow_snap: Optional[Dict] = None,
        flow_streak: int = 0,
    ) -> bool:
        """Check if long entry conditions are met."""
        if verbose:
            print(f"      — Looking to BUY —")
        fc = self._flow_counter_config()
        is_counter_trend, ct_reason, ct_near = evaluate_counter_trend(
            "long", ctx_5m, flow_snap, fc, flow_streak=flow_streak,
        )
        if not is_counter_trend and not bias_15m_allows("long", ctx_15m):
            if verbose:
                print(f"      ❌ 15M chart not aligned for buys (15M trend is {ctx_15m.get('trend')})")
            return False
        # 5M Trend Filter
        if ctx_5m['trend'] != 'bullish' and not is_counter_trend:
            if verbose:
                detail = ct_near or f"5M trend is {ctx_5m['trend']}, need up"
                print(f"      ❌ Big picture is not pointing up ({detail})")
            return False
        if is_counter_trend:
            if verbose:
                print(f"      ⚡ Counter-trend LONG allowed — {ct_reason}")
        adx_min = effective_adx_threshold(ctx_5m, flow_snap, "long")
        if ctx_5m['adx'] < adx_min:
            if verbose:
                relax = f" ({adx_min} w/ flow)" if adx_min < ADX_THRESHOLD else ""
                print(
                    f"      ❌ Trend not strong enough yet "
                    f"(strength {ctx_5m['adx']:.0f}, need {ADX_THRESHOLD}+{relax})"
                )
            return False
        if not is_counter_trend and USE_ORDER_FLOW and flow_snap:
            blocked, flow_reason = flow_blocks_long(flow_snap, FLOW_ENTRY_GUARD)
            adx_val = float(ctx_5m.get("adx", 0))
            if blocked and adx_val >= FLOW_RELAX_STRENGTH:
                buy_pct = float(flow_snap.get("buy_pct", 0.5))
                if buy_pct >= FLOW_RELAX_LONG_BUY_PCT:
                    blocked = False
                    if verbose:
                        print(
                            f"      ⚡ Flow guard relaxed — strength {adx_val:.0f}≥{FLOW_RELAX_STRENGTH:.0f}, "
                            f"buy% {buy_pct:.0%}≥{FLOW_RELAX_LONG_BUY_PCT:.0%}"
                        )
            if blocked:
                if verbose:
                    print(f"      ❌ {flow_reason}")
                return False

        di_tol = DI_TOLERANCE
        flow_confirms = (
            not is_counter_trend
            and USE_ORDER_FLOW
            and USE_FLOW_DI_OVERRIDE
            and flow_snap
            and ctx_5m['trend'] == 'bullish'
            and bias_15m_allows("long", ctx_15m)
            and flow_confirms_long_direction(flow_snap, FLOW_ENTRY_GUARD)
            and TickFlowTracker.confirms_direction("long", flow_snap)
        )
        if flow_confirms:
            di_tol = DI_FLOW_TOLERANCE
            if verbose:
                print(f"      ⚡ Flow confirms buy — DI tolerance relaxed ({di_tol:.0f} pts)")
        adx_val = float(ctx_5m.get("adx", 0))
        if not is_counter_trend and adx_val >= DI_RELAX_STRENGTH:
            if verbose:
                print(
                    f"      ⚡ DI check skipped — strength {adx_val:.0f}≥{DI_RELAX_STRENGTH:.0f} "
                    f"(DI+ {ctx_5m['di_plus']:.0f} vs DI− {ctx_5m['di_minus']:.0f})"
                )
        elif not is_counter_trend and ctx_5m['di_plus'] < (ctx_5m['di_minus'] - di_tol):
            di_gap = ctx_5m['di_plus'] - ctx_5m['di_minus']
            if verbose:
                print(
                    f"      ❌ Buyers not in control on 5M chart "
                    f"(DI+ {ctx_5m['di_plus']:.0f} vs DI− {ctx_5m['di_minus']:.0f}, gap {di_gap:+.0f}; "
                    f"need DI+ ≥ DI−−{di_tol:.0f})"
                )
            return False
        
        # 1M Entry Conditions
        price = row_1m['close']
        ema_9 = row_1m['ema_9']
        ema_21 = row_1m['ema_21']
        rsi = row_1m['rsi']
        macd_hist = row_1m['macd_hist']
        macd_hist_prev = row_1m['macd_hist_prev']
        volume_ratio = row_1m['volume_ratio']
        bb_pctb = row_1m['bb_pctb']
        atr = row_1m['atr']
        vwap = row_1m.get('vwap', float('nan'))
        candle_open = row_1m['open']
        rsi_lo, rsi_hi = adaptive_rsi_long(ctx_5m, is_counter_trend)
        pullback = atr * adaptive_pullback_atr(ctx_5m, "long", is_counter_trend)
        
        if not adaptive_vwap_ok(price, vwap, "long", ctx_5m):
            if verbose: print(f"      ❌ Strong uptrend — price must stay above VWAP to buy")
            return False
        
        # Optional bullish candle confirmation (off by default — backtest allows any close)
        if CANDLE_CONFIRMATION and price <= candle_open:
            if verbose: print(f"      ❌ Last 1M candle was red — waiting for a green candle to buy")
            return False
        ema9_tolerance = atr * 0.1
        if price < (ema_9 - ema9_tolerance):
            if verbose: print(f"      ❌ Price too far below short-term average — not ready to buy")
            return False
        if not is_counter_trend and abs(price - ema_21) > pullback:
            if verbose: print(f"      ❌ Price not close enough to the pullback zone")
            return False
        if pd.isna(rsi) or not (rsi_lo <= rsi <= rsi_hi):
            if verbose: print(f"      ❌ RSI not in the buy zone ({rsi:.0f}, want {rsi_lo:.0f}-{rsi_hi:.0f})")
            return False
        if pd.isna(macd_hist) or pd.isna(macd_hist_prev):
            if verbose:
                print(f"      ❌ Momentum indicator not ready yet (MACD warming up)")
            return False
        relax = strong_trend_relaxed("long", ctx_5m, ctx_15m) or is_counter_trend
        if not relax and macd_hist <= macd_hist_prev:
            if verbose: print(f"      ❌ Momentum not turning up yet (MACD still flat or falling)")
            return False
        if pd.isna(volume_ratio) or volume_ratio < VOLUME_RATIO_THRESHOLD:
            if verbose: print(f"      ❌ Not enough volume on this candle")
            return False
        if pd.isna(bb_pctb) or bb_pctb <= BB_EXTREME_LOW or bb_pctb >= BB_EXTREME_HIGH:
            if verbose: print(f"      ❌ Price at an extreme — risk of reversal")
            return False

        ts = row_1m.get("datetime")
        ok, reason = check_long_entry_quality(
            row_1m, ctx_5m, ENTRY_QUALITY,
            df_1m=df_1m, timestamp=ts, session_mode=SESSION_MODE,
            is_counter_trend=is_counter_trend,
        )
        if not ok:
            if verbose:
                print(f"      ❌ {reason}")
            return False
        
        return True
    
    def check_short_entry(
        self,
        row_1m: pd.Series,
        ctx_5m: Dict,
        ctx_15m: Dict,
        verbose: bool = False,
        df_1m: Optional[pd.DataFrame] = None,
        flow_snap: Optional[Dict] = None,
        flow_streak: int = 0,
    ) -> bool:
        """Check if short entry conditions are met."""
        if verbose:
            print(f"      — Looking to SELL —")
        fc = self._flow_counter_config()
        is_counter_trend, ct_reason, ct_near = evaluate_counter_trend(
            "short", ctx_5m, flow_snap, fc, flow_streak=flow_streak,
        )

        if ctx_5m['trend'] != 'bearish' and not is_counter_trend:
            if verbose:
                detail = ct_near or f"5M trend is {ctx_5m['trend']}"
                print(f"      ❌ Big picture is not pointing down ({detail})")
            return False
        if not is_counter_trend and not bias_15m_allows("short", ctx_15m):
            if verbose:
                print(f"      ❌ 15M chart not aligned for sells (15M trend is {ctx_15m.get('trend')})")
            return False
        if is_counter_trend:
            if verbose:
                print(f"      ⚡ Counter-trend SHORT allowed — {ct_reason}")
        adx_min = effective_adx_threshold(ctx_5m, flow_snap, "short")
        if ctx_5m['adx'] < adx_min:
            if verbose:
                relax = f" ({adx_min} w/ flow)" if adx_min < ADX_THRESHOLD else ""
                print(
                    f"      ❌ Trend not strong enough yet "
                    f"(strength {ctx_5m['adx']:.0f}, need {ADX_THRESHOLD}+{relax})"
                )
            return False
        if not is_counter_trend and USE_ORDER_FLOW and flow_snap:
            blocked, flow_reason = flow_blocks_short(flow_snap, FLOW_ENTRY_GUARD)
            if blocked:
                if verbose:
                    print(f"      ❌ {flow_reason}")
                return False

        di_tol = DI_TOLERANCE
        flow_confirms = (
            not is_counter_trend
            and USE_ORDER_FLOW
            and USE_FLOW_DI_OVERRIDE
            and flow_snap
            and ctx_5m['trend'] == 'bearish'
            and bias_15m_allows("short", ctx_15m)
            and TickFlowTracker.confirms_direction("short", flow_snap)
        )
        if flow_confirms:
            di_tol = DI_FLOW_TOLERANCE
            if verbose:
                print(f"      ⚡ Flow confirms sell — DI tolerance relaxed ({di_tol:.0f} pts)")
        if not is_counter_trend and ctx_5m['di_minus'] < (ctx_5m['di_plus'] - di_tol):
            di_gap = ctx_5m['di_minus'] - ctx_5m['di_plus']
            if verbose:
                print(
                    f"      ❌ Sellers not in control on 5M chart "
                    f"(DI− {ctx_5m['di_minus']:.0f} vs DI+ {ctx_5m['di_plus']:.0f}, gap {di_gap:+.0f}; "
                    f"need DI− ≥ DI+−{di_tol:.0f})"
                )
            return False
        
        # 1M Entry Conditions
        price = row_1m['close']
        ema_9 = row_1m['ema_9']
        ema_21 = row_1m['ema_21']
        rsi = row_1m['rsi']
        macd_hist = row_1m['macd_hist']
        macd_hist_prev = row_1m['macd_hist_prev']
        volume_ratio = row_1m['volume_ratio']
        bb_pctb = row_1m['bb_pctb']
        atr = row_1m['atr']
        vwap = row_1m.get('vwap', float('nan'))
        candle_open = row_1m['open']
        rsi_lo, rsi_hi = adaptive_rsi_short(ctx_5m, is_counter_trend)
        pullback = atr * adaptive_pullback_atr(ctx_5m, "short", is_counter_trend)
        
        if not adaptive_vwap_ok(price, vwap, "short", ctx_5m):
            if verbose: print(f"      ❌ Strong downtrend — price must stay below VWAP to sell")
            return False
        
        # Optional bearish candle confirmation (off by default)
        if CANDLE_CONFIRMATION and price >= candle_open:
            if verbose: print(f"      ❌ Last 1M candle was green — waiting for a red candle to sell")
            return False
        ema9_tolerance = atr * 0.1
        if price > (ema_9 + ema9_tolerance):
            if verbose: print(f"      ❌ Price too far above short-term average — not ready to sell")
            return False
        if not is_counter_trend and abs(price - ema_21) > pullback:
            if verbose: print(f"      ❌ Price not close enough to the pullback zone")
            return False
        if pd.isna(rsi) or not (rsi_lo <= rsi <= rsi_hi):
            if verbose: print(f"      ❌ RSI not in the sell zone ({rsi:.0f}, want {rsi_lo:.0f}-{rsi_hi:.0f})")
            return False
        if pd.isna(macd_hist) or pd.isna(macd_hist_prev):
            if verbose:
                print(f"      ❌ Momentum indicator not ready yet (MACD warming up)")
            return False
        relax = strong_trend_relaxed("short", ctx_5m, ctx_15m) or is_counter_trend
        if not relax and macd_hist >= macd_hist_prev:
            if verbose: print(f"      ❌ Momentum not turning down yet (MACD still flat or rising)")
            return False
        if pd.isna(volume_ratio) or volume_ratio < VOLUME_RATIO_THRESHOLD:
            if verbose: print(f"      ❌ Not enough volume on this candle")
            return False
        if pd.isna(bb_pctb) or bb_pctb <= BB_EXTREME_LOW or bb_pctb >= BB_EXTREME_HIGH:
            if verbose: print(f"      ❌ Price at an extreme — risk of reversal")
            return False

        ts = row_1m.get("datetime")
        ok, reason = check_short_entry_quality(
            row_1m, ctx_5m, ENTRY_QUALITY,
            is_counter_trend=is_counter_trend,
            df_1m=df_1m, timestamp=ts, session_mode=SESSION_MODE,
        )
        if not ok:
            if verbose:
                print(f"      ❌ {reason}")
            return False
        
        return True
    
    def check_entry_signal(
        self,
        symbol: str,
        df_1m: pd.DataFrame,
        ctx_5m: Dict,
        ctx_15m: Optional[Dict] = None,
        verbose: Optional[bool] = None,
        flow_streak: Optional[int] = None,
    ) -> Optional[Dict]:
        """Check for entry signal on latest 1M bar for a symbol."""
        if ctx_15m is None:
            ctx_15m = {'trend': None, 'adx': 0}
        if not self.paper_mode and self.broker:
            self._sync_all_broker_closed_positions(symbol)

        if verbose is None:
            verbose = VERBOSE_SKIP_REASONS

        row = df_1m.iloc[-1]
        entry_blocked = False

        if self._symbol_trade_limit_reached(symbol):
            limit = max_trades_per_day_for(symbol)
            count = self._daily_trades_for(symbol)
            msg = f"Max trades/day hit for {symbol}: {count}/{limit}"
            if verbose:
                print(f"      ❌ {msg} — no new entries today")
            else:
                print(f"   🛑 {symbol}: {msg}")
            return None

        if self._symbol_in_loss_cooldown(symbol, quiet=verbose):
            entry_blocked = True

        if self._symbol_in_consec_loss_pause(symbol, quiet=verbose):
            entry_blocked = True

        adaptive_block = self._adaptive_blocks_entry(symbol, verbose=verbose)
        if adaptive_block:
            entry_blocked = True

        if MIN_SECONDS_BETWEEN_ENTRIES > 0:
            last = self._last_entry_time.get(symbol)
            if last is not None:
                elapsed = (datetime.now(timezone.utc) - last).total_seconds()
                if elapsed < MIN_SECONDS_BETWEEN_ENTRIES:
                    if verbose:
                        wait = int(MIN_SECONDS_BETWEEN_ENTRIES - elapsed)
                        print(f"      ⏳ Entry spacing — wait {wait}s ({MIN_SECONDS_BETWEEN_ENTRIES}s min gap)")
                    entry_blocked = True

        open_count, _ = self._concurrent_exposure(symbol)
        max_pos = max_positions_for(symbol)
        if open_count >= max_pos:
            if not verbose:
                print(f"   ⏸️ {symbol}: max positions ({open_count}/{max_pos}) open — no new entries")
            entry_blocked = True

        atr = row['atr']
        if pd.isna(atr) or atr <= 0:
            if not verbose:
                print(f"   ❌ {symbol}: Still warming up — not enough price history yet")
            entry_blocked = True
            atr = 0.0

        candle_range = row['high'] - row['low']
        max_bar_pts = max_1m_bar_pts_for(symbol)
        if candle_range > max_bar_pts:
            if not verbose:
                print(
                    f"   ❌ {symbol}: This 1-minute bar moved too much "
                    f"({candle_range:.0f} pts, max {max_bar_pts:.0f}) — skipping for safety"
                )
            entry_blocked = True

        flow_snap = None
        if USE_ORDER_FLOW and self.broker:
            flow_snap = self.broker.get_tick_flow(symbol)
        if flow_streak is None:
            flow_streak = self._update_flow_regime_streak(symbol, ctx_5m, flow_snap)

        if STRATEGY_MODE == "scalp_hybrid":
            if entry_blocked:
                return None
            df_30s = self._df_30s_cache.get(symbol)
            row_30s, prev_30s, trig_src = self._resolve_30s_trigger_rows(
                symbol, row, df_1m,
            )
            if row_30s is None or prev_30s is None:
                return None
            row_5m, prev_5m, from_1m = self._resolve_trend_rows(df_1m, self._df_5m_cache)
            st = self._scalp_hybrid_state.get(symbol, ScalpHybridState())
            bracket_log = _scalp_bracket_log_for_atr(atr)
            hybrid_params = self._hybrid_params_with_learner(atr, symbol)
            signal, st = check_hybrid_entry(
                symbol, row, row_5m, prev_5m, row_30s, prev_30s, st,
                flow_snap=flow_snap,
                **hybrid_params,
            )
            self._scalp_hybrid_state[symbol] = st
            if not signal:
                return None
            entry_meta = self._build_hybrid_entry_meta(
                signal, ctx_5m, flow_snap, row_30s, prev_30s,
            )
            if self._deepseek_blocks_hybrid_entry(signal, ctx_5m, flow_snap, entry_meta):
                return None
            signal["entry_meta"] = entry_meta
            signal["scalp_bracket_log"] = bracket_log
            if verbose and trig_src == "1M" and not scalp_fast_mode_active():
                print(f"      ⚡ {symbol}: 1M trigger fallback (30s bars unavailable)")
            sl, tp = round_bracket_prices(symbol, signal["direction"], signal["sl"], signal["tp"])
            signal["sl"] = sl
            signal["tp"] = tp
            over = bracket_risk_over_limit(symbol, signal["direction"], signal["entry"], sl, tp)
            if over:
                print(f"   ❌ {symbol}: {over} — skipping")
                return None
            signal.update(calc_trade_dollars({
                "symbol": symbol,
                "direction": signal["direction"],
                "entry": signal["entry"],
                "sl": sl,
                "tp": tp,
            }))
            print(f"   {bracket_log}")
            if verbose:
                mode = signal.get("scalp_mode", "hybrid")
                print(
                    f"      ✅ Scalp {mode.upper()} {signal['direction'].upper()} "
                    f"— SL {sl:.2f} TP {tp:.2f}"
                )
            return signal

        if STRATEGY_MODE == "fvs1" and FVS1_CFG is not None:
            if entry_blocked:
                return None
            if FVS1_CFG.log_only:
                return None
            df_30s = self._df_30s_cache.get(symbol)
            row_30s, prev_30s, trig_src = self._resolve_30s_trigger_rows(
                symbol, row, df_1m,
            )
            if row_30s is None or prev_30s is None:
                return None
            row_5m, prev_5m, _from_1m = self._resolve_trend_rows(df_1m, self._df_5m_cache)
            try:
                now_et = pd.Timestamp(row.get("datetime", datetime.now(timezone.utc))).tz_convert("US/Eastern")
            except Exception:
                now_et = datetime.now(pytz.timezone("US/Eastern"))
            st = self._fvs1_state.get(symbol, FVS1State())
            signal, st = check_fvs1_entry(
                symbol, df_1m, row_5m, row_30s, prev_30s, st, FVS1_CFG,
                df_30s=df_30s, now_et=now_et, flow_snap=flow_snap, risk=self._fvs1_risk,
            )
            self._fvs1_state[symbol] = st
            if not signal:
                return None
            if verbose and trig_src == "1M" and not scalp_fast_mode_active():
                print(f"      ⚡ {symbol}: 1M trigger fallback (30s bars unavailable)")
            sl, tp = round_bracket_prices(symbol, signal["direction"], signal["sl"], signal["tp"])
            signal["sl"] = sl
            signal["tp"] = tp
            over = bracket_risk_over_limit(symbol, signal["direction"], signal["entry"], sl, tp)
            if over:
                print(f"   ❌ {symbol}: {over} — skipping")
                return None
            signal.update(calc_trade_dollars({
                "symbol": symbol,
                "direction": signal["direction"],
                "entry": signal["entry"],
                "sl": sl,
                "tp": tp,
            }))
            if verbose:
                print(
                    f"      ✅ FVS-1 {signal['direction'].upper()} "
                    f"— SL {sl:.2f} TP {tp:.2f} (POC {signal.get('poc', 0):.2f})"
                )
            return signal

        if STRATEGY_MODE == "scalp_b":
            if entry_blocked:
                return None
            prev_1m = df_1m.iloc[-2] if len(df_1m) >= 2 else None
            row_5m, _prev_5m, _from_1m = self._resolve_trend_rows(df_1m, self._df_5m_cache)
            df_30s = self._df_30s_cache.get(symbol)
            row_30s = df_30s.iloc[-1] if df_30s is not None and len(df_30s) else None
            prev_30s = df_30s.iloc[-2] if df_30s is not None and len(df_30s) >= 2 else None
            if USE_30S_BARS and (row_30s is None or prev_30s is None):
                return None
            st = self._scalp_state.get(symbol, ScalpSymbolState())
            sl_pts, tp_pts, atr_used, rr = _scalp_bracket_for_atr(atr)
            bracket_log = format_scalp_bracket_log(sl_pts, tp_pts, atr_used, rr)
            signal, st = check_scalp_entry(
                symbol, row, row_5m, prev_1m, st,
                row_30s=row_30s if USE_30S_BARS else None,
                prev_30s=prev_30s if USE_30S_BARS else None,
                adx_min=SCALP_ADX_MIN,
                pullback_atr=SCALP_PULLBACK_ATR,
                setup_bars=SCALP_SETUP_BARS,
                sl_pts=sl_pts,
                tp_pts=tp_pts,
            )
            self._scalp_state[symbol] = st
            if not signal:
                return None
            signal["scalp_bracket_log"] = bracket_log
            sl, tp = round_bracket_prices(symbol, signal["direction"], signal["sl"], signal["tp"])
            signal["sl"] = sl
            signal["tp"] = tp
            over = bracket_risk_over_limit(symbol, signal["direction"], signal["entry"], sl, tp)
            if over:
                print(f"   ❌ {symbol}: {over} — skipping")
                return None
            signal.update(calc_trade_dollars({
                "symbol": symbol,
                "direction": signal["direction"],
                "entry": signal["entry"],
                "sl": sl,
                "tp": tp,
            }))
            print(f"   {bracket_log}")
            if verbose:
                print(f"      ✅ Scalp B {signal['direction'].upper()} signal — SL {sl:.2f} TP {tp:.2f}")
            return signal

        long_ok = self.check_long_entry(
            row, ctx_5m, ctx_15m, verbose=verbose and not FULL_TRADE_DIAGNOSTICS, df_1m=df_1m,
            flow_snap=flow_snap, flow_streak=flow_streak,
        )
        short_ok = self.check_short_entry(
            row, ctx_5m, ctx_15m, verbose=verbose and not FULL_TRADE_DIAGNOSTICS, df_1m=df_1m,
            flow_snap=flow_snap, flow_streak=flow_streak,
        )

        if entry_blocked:
            return None

        sl_distance = atr * ATR_MULT
        sl_distance = cap_sl_distance(symbol, sl_distance)
        tp_distance = sl_distance * TP_MULT
        entry_price = row['close']
        max_loss = spec_limit(symbol, "max_loss_per_trade", MAX_LOSS_PER_TRADE)
        risk_at_sl = sl_distance * SYMBOL_SPECS[symbol]["point_value"] * CONTRACTS

        if long_ok:
            tp_rr = entry_price + tp_distance
            tp_buffer = atr * TP_BUFFER_ATR_MULT
            tp_at_structure = ctx_5m['resistance'] - tp_buffer  # below swing high
            tp_final = min(tp_rr, tp_at_structure) if tp_at_structure > entry_price else tp_rr
            capped = tp_final < tp_rr - 0.01

            actual_tp_distance = tp_final - entry_price
            actual_rr = actual_tp_distance / sl_distance if sl_distance > 0 else 0
            min_rr = entry_min_rr("long", ctx_5m, ctx_15m)
            if actual_rr < min_rr:
                print(f"   ❌ {symbol}: Profit target too close — not enough reward vs risk for a long")
                return None
            if risk_at_sl > max_loss + 1:
                print(f"   ❌ {symbol}: Stop too wide (${risk_at_sl:.0f} risk, max ${max_loss:.0f}) — skipping")
                return None

            if USE_ORDER_FLOW and self.broker:
                flow_ev = self.broker.evaluate_order_flow(
                    "long", symbol, mode=ORDER_FLOW_MODE,
                )
                if verbose and flow_ev.get("advisory_note"):
                    print(f"      📊 {flow_ev['advisory_note']}")
                if ORDER_FLOW_MODE == "block" and not flow_ev.get("allowed", True):
                    if verbose:
                        print(f"      ❌ Order flow: {flow_ev.get('reason', 'blocks LONG')}")
                    return None

            sl_raw = entry_price - sl_distance
            sl, tp = round_bracket_prices(symbol, "long", sl_raw, tp_final)
            over = bracket_risk_over_limit(symbol, "long", entry_price, sl, tp)
            if over:
                print(f"   ❌ {symbol}: {over} — skipping")
                return None
            return {
                'symbol': symbol,
                'direction': 'long',
                'entry': entry_price,
                'sl': sl,
                'tp': tp,
                'atr': atr,
                'resistance': ctx_5m['resistance'],
                'structure_capped': capped,
                **calc_trade_dollars({
                    'symbol': symbol, 'direction': 'long',
                    'entry': entry_price,
                    'sl': sl,
                    'tp': tp,
                }),
            }
        
        if short_ok:
            tp_rr = entry_price - tp_distance
            tp_buffer = atr * TP_BUFFER_ATR_MULT
            tp_at_structure = ctx_5m['support'] + tp_buffer  # above swing low
            tp_final = max(tp_rr, tp_at_structure) if tp_at_structure < entry_price else tp_rr
            capped = tp_final > tp_rr + 0.01

            actual_tp_distance = entry_price - tp_final
            actual_rr = actual_tp_distance / sl_distance if sl_distance > 0 else 0
            min_rr = entry_min_rr("short", ctx_5m, ctx_15m)
            if actual_rr < min_rr:
                print(f"   ❌ {symbol}: Profit target too close — not enough reward vs risk for a short")
                return None
            if risk_at_sl > max_loss + 1:
                print(f"   ❌ {symbol}: Stop too wide (${risk_at_sl:.0f} risk, max ${max_loss:.0f}) — skipping")
                return None

            if USE_ORDER_FLOW and self.broker:
                flow_ev = self.broker.evaluate_order_flow(
                    "short", symbol, mode=ORDER_FLOW_MODE,
                )
                if verbose and flow_ev.get("advisory_note"):
                    print(f"      📊 {flow_ev['advisory_note']}")
                if ORDER_FLOW_MODE == "block" and not flow_ev.get("allowed", True):
                    if verbose:
                        print(f"      ❌ Order flow: {flow_ev.get('reason', 'blocks SHORT')}")
                    return None

            sl_raw = entry_price + sl_distance
            sl, tp = round_bracket_prices(symbol, "short", sl_raw, tp_final)
            over = bracket_risk_over_limit(symbol, "short", entry_price, sl, tp)
            if over:
                print(f"   ❌ {symbol}: {over} — skipping")
                return None
            return {
                'symbol': symbol,
                'direction': 'short',
                'entry': entry_price,
                'sl': sl,
                'tp': tp,
                'atr': atr,
                'support': ctx_5m['support'],
                'structure_capped': capped,
                **calc_trade_dollars({
                    'symbol': symbol, 'direction': 'short',
                    'entry': entry_price,
                    'sl': sl,
                    'tp': tp,
                }),
            }
        
        return None
    
    def place_order(self, signal: Dict) -> bool:
        """Place order with bracket (SL+TP)."""
        symbol = signal['symbol']
        direction = signal['direction']
        entry = signal['entry']
        sl, tp = round_bracket_prices(symbol, direction, signal['sl'], signal['tp'])
        signal['sl'] = sl
        signal['tp'] = tp

        with self._place_order_lock:
            if not self._enforce_position_capacity(symbol):
                return False
            self._pending_entries[symbol] = self._pending_entries.get(symbol, 0) + 1
            try:
                return self._place_order_locked(signal, symbol, direction, entry, sl, tp)
            finally:
                pending = self._pending_entries.get(symbol, 0) - 1
                if pending <= 0:
                    self._pending_entries.pop(symbol, None)
                else:
                    self._pending_entries[symbol] = pending

    def _place_order_locked(
        self,
        signal: Dict,
        symbol: str,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
    ) -> bool:
        """Inner place_order — caller holds _place_order_lock and pending slot."""
        
        spec = SYMBOL_SPECS[symbol]
        point_value = spec['point_value']
        
        if self.paper_mode:
            order_id = f"paper_{symbol}_{int(time.time())}"
            broker_bracket = False
            if PAPER_RITHMIC_BRACKETS and self.broker and self.broker.connected:
                side = 'BUY' if direction == 'long' else 'SELL'
                try:
                    result = self.broker.place_order(
                        symbol=symbol,
                        order_type=side,
                        size=CONTRACTS,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit=tp,
                    )
                    if result and result.get('ticket'):
                        order_id = result['ticket']
                        broker_bracket = bool(
                            result.get('native_stop_attached') or result.get('native_target_attached')
                        )
                        print(f"📝 [PAPER+Rithmic] Bracket order {order_id} (SL/TP on broker)")
                except Exception as e:
                    print(f"   ⚠️ Paper bracket failed — simulating SL/TP locally: {e}")

            print(f"📝 [PAPER] {direction.upper()} {symbol} @ {entry:.2f}")
            print(f"   ⚠️  NOT SENT TO BROKER — use start_nq_live.bat (or add --live)")
            print(f"   Stop loss: {sl:.2f}  |  Take profit: {tp:.2f}")
            if not broker_bracket:
                print(f"   Bracket: simulated via bar high/low each scan")
            print_trade_money(signal)
            
            self._add_position(Position(
                order_id=order_id,
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                size=CONTRACTS,
                sl=sl,
                tp=tp,
                entry_time=datetime.now(timezone.utc),
                initial_sl=sl,
                broker_bracket=broker_bracket,
                entry_meta=signal.get("entry_meta"),
            ))
            self._increment_daily_trades(symbol)
            notify_trade_placed(
                symbol=symbol,
                direction=direction,
                entry=entry,
                sl=sl,
                tp=tp,
                ticket_id=str(order_id),
                mode='paper',
                protection='broker bracket' if broker_bracket else 'simulated locally',
            )
            return True
        
        # Live order
        try:
            side = 'BUY' if direction == 'long' else 'SELL'
            result = self.broker.place_order(
                symbol=symbol,
                order_type=side,
                size=CONTRACTS,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp
            )
            
            if result and result.get('ticket'):
                ticket = result['ticket']
                bracket_mode = result.get('bracket_mode', 'unknown')
                exec_info = self.broker.get_account_info()
                route = self.broker.get_trade_route("CME")
                acct_id = exec_info.get("account_id") or "UNKNOWN"
                rith_sym, exchange = self.broker._resolve_symbol(symbol)
                print(f"\n{'='*60}")
                print(f"  🔴 LIVE ORDER PLACED — Ticket: {ticket}")
                print(f"{'='*60}")
                print(
                    f"   account_id={acct_id}  trade_route={route or 'AUTO'}  "
                    f"exchange={exchange}  symbol={symbol}  rithmic={rith_sym}"
                )
                if self.broker.using_simulator_route:
                    print(
                        f"   🚨 SIMULATOR ROUTE — fills are sim-only until Lucid enables live route"
                    )
                    print(
                        f"   list_positions may stay 0; bot tracks via entry fill + SL/TP"
                    )
                print(f"   {direction.upper()} {symbol} @ {entry:.2f}")
                print(f"   SL: {sl:.2f} | TP: {tp:.2f}")
                print_trade_money(signal)
                print(f"   Broker brackets: {bracket_mode} "
                      f"(SL={'yes' if result.get('native_stop_attached') else 'no'}, "
                      f"TP={'yes' if result.get('native_target_attached') else 'no'})")
                if bracket_mode == "protective_fallback":
                    if direction == "short":
                        print(
                            "   📊 Rithmic chart: BUY STOP = SL, BUY LIMIT = TP "
                            "(correct for SHORT — Lucid may not label them 'SL/TP')"
                        )
                    else:
                        print(
                            "   📊 Rithmic chart: SELL STOP = SL, SELL LIMIT = TP "
                            "(correct for LONG)"
                        )
                print(f"   ⏳ Verifying broker SL + TP on Rithmic...")

                sl_ok, tp_ok, already_closed = self.broker.verify_and_ensure_protection(
                    ticket=ticket,
                    symbol=symbol,
                    side=side,
                    size=CONTRACTS,
                    stop_loss=sl,
                    take_profit=tp,
                    max_attempts=BROKER_PROTECTION_MAX_RETRIES,
                )

                if already_closed:
                    fill_info = self.broker.confirm_bracket_exit_fill(
                        ticket, symbol, sl, tp,
                    )
                    if not fill_info or not fill_info.get("confirmed"):
                        bot_logger.error(
                            f"{symbol} {ticket}: verify reported closed but fill unconfirmed — "
                            f"not recording trade"
                        )
                        return False
                    exit_price = float(fill_info.get("exit_price") or 0)
                    leg = fill_info.get("leg", "?")
                    fill_entry = float(result.get("entry_price") or entry)
                    print(
                        f"   ⚡ Bracket completed during entry: {leg} @ {exit_price:.2f} "
                        f"(account={acct_id} route={route})"
                    )
                    self._add_position(Position(
                        order_id=ticket,
                        symbol=symbol,
                        direction=direction,
                        entry_price=fill_entry,
                        size=CONTRACTS,
                        sl=sl,
                        tp=tp,
                        entry_time=datetime.now(timezone.utc),
                        initial_sl=sl,
                        broker_bracket=True,
                        entry_meta=signal.get("entry_meta"),
                    ))
                    self._increment_daily_trades(symbol)
                    self.close_position(symbol, ticket, "BROKER_BRACKET", exit_price)
                    return True

                if not sl_ok:
                    if self.broker.entry_protection_grace_active(ticket):
                        msg = (
                            f"⚠️ {symbol} entry {ticket}: SL/TP not visible yet during "
                            f"entry grace — keeping position, repair will continue each scan"
                        )
                        print(f"\n   {msg}")
                        bot_logger.warning(msg)
                        protection = (
                            f"PENDING — SL/TP settling on broker "
                            f"(grace {int(os.getenv('RITHMIC_ENTRY_POSITION_GRACE_SEC', '30'))}s)"
                        )
                    else:
                        msg = (
                            f"🚨 CRITICAL: {symbol} entry {ticket} has NO broker stop-loss — "
                            f"SL=MISSING TP={'yes' if tp_ok else 'MISSING'} — "
                            f"flattening position"
                        )
                        print(f"\n   {msg}")
                        bot_logger.error(msg)
                        send_email(
                            f"EMERGENCY: Unprotected {symbol} entry flattened",
                            f"{msg}\n\nEntry: {entry:.2f}\nExpected SL: {sl:.2f}\nExpected TP: {tp:.2f}",
                        )
                        self.broker.close_position(symbol=symbol, ticket=ticket)
                        return False
                elif not tp_ok:
                    msg = (
                        f"⚠️ PARTIAL PROTECTION: {symbol} entry {ticket} — "
                        f"SL verified @ {sl:.2f}, TP still MISSING after retries — "
                        f"keeping position (stop-protected); TP retry continues each scan"
                    )
                    print(f"\n   {msg}")
                    bot_logger.warning(msg)
                    protection = f"PARTIAL — SL @ {sl:.2f}, TP missing (retrying)"
                else:
                    print(
                        f"   ✅ VERIFIED: SL @ {sl:.2f} | TP @ {tp:.2f} on Rithmic"
                    )
                    protection = f"VERIFIED — SL @ {sl:.2f}, TP @ {tp:.2f} ({bracket_mode})"

                self._add_position(Position(
                    order_id=ticket,
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry,
                    size=CONTRACTS,
                    sl=sl,
                    tp=tp,
                    entry_time=datetime.now(timezone.utc),
                    initial_sl=sl,
                    broker_bracket=sl_ok,
                    entry_meta=signal.get("entry_meta"),
                ))
                self._increment_daily_trades(symbol)
                notify_trade_placed(
                    symbol=symbol,
                    direction=direction,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    ticket_id=str(ticket),
                    mode='live',
                    protection=protection,
                )
                return True
            else:
                print(f"❌ Order failed: {result}")
                return False
        except Exception as e:
            print(f"❌ Order error: {e}")
            return False
    
    def test_order(self):
        """Test broker connectivity and order capability."""
        print("\n" + "="*60)
        print("🧪 ORDER SYSTEM TEST")
        print("="*60)
        
        test_symbol = self.symbols[0] if self.symbols else 'MES'
        
        # Step 1: Connect
        print(f"\n[1/3] 📡 Connecting to broker...")
        try:
            if not self.connect():
                print(f"❌ Failed to connect to broker")
                return False
            print(f"✅ Connected!")
            time.sleep(3)
            
            # Step 2: Get quote
            print(f"\n[2/3] 📊 Getting {test_symbol} quote...")
            quote = self.broker.get_latest_price(test_symbol)
            
            if not quote:
                df = self.broker.get_candles(test_symbol, timeframe_minutes=1, num_candles=5)
                if df is not None and len(df) > 0:
                    last = float(df.iloc[-1]['close'])
                    quote = {'bid': last, 'ask': last, 'last': last}
            
            if not quote:
                print(f"❌ Failed to get quote for {test_symbol}")
                return False
            
            bid = quote.get('bid', 0)
            ask = quote.get('ask', 0)
            last = quote.get('last', 0)
            current_price = last if last > 0 else (bid + ask) / 2 if bid and ask else 0
            
            if current_price == 0:
                print(f"❌ No valid price for {test_symbol}")
                return False
                
            print(f"✅ Quote received: Bid={bid:.2f} Ask={ask:.2f} Last={last:.2f}")
            
            # Step 3: Test order (with immediate close)
            print(f"\n[3/3] 📝 Testing order placement...")
            print(f"   This will place a REAL order and immediately close it!")
            
            confirm = input("   Continue? (type 'yes'): ")
            if confirm.lower() != 'yes':
                print("   Test skipped.")
                return True
            
            # Place order at market with tight brackets
            spec = SYMBOL_SPECS.get(test_symbol, SYMBOL_SPECS['MES'])
            sl = current_price - (5 * spec['tick_size'])  # 5 ticks SL
            tp = current_price + (3 * spec['tick_size'])  # 3 ticks TP
            
            result = self.broker.place_order(
                symbol=test_symbol,
                order_type='BUY',
                size=1,
                entry_price=current_price,
                stop_loss=sl,
                take_profit=tp
            )
            
            if result and result.get('ticket'):
                order_id = result['ticket']
                print(f"✅ Order placed! Ticket: {order_id}")
                
                # Immediately close
                time.sleep(0.3)
                print(f"🗑️ Closing position...")
                close_result = self.broker.close_position(symbol=test_symbol)
                
                if close_result:
                    print(f"✅ Position closed!")
                    print("\n" + "="*60)
                    print("✅ ORDER SYSTEM VERIFIED - WORKING!")
                    print("="*60 + "\n")
                    return True
                else:
                    print(f"⚠️ Close failed - CHECK BROKER MANUALLY!")
                    print(f"   You may have an open position on {test_symbol}")
                    return True  # Order worked
            else:
                print(f"❌ Order failed: {result}")
                return False
                
        except Exception as e:
            print(f"❌ Test error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if self.broker:
                self.broker.shutdown()
    
    def _position_hold_seconds(self, position: Position) -> float:
        return (datetime.now(timezone.utc) - position.entry_time).total_seconds()

    def _profit_pct_to_tp(self, position: Position, price: float) -> float:
        """Fraction of entry→TP distance achieved (0..1+)."""
        if price <= 0:
            return 0.0
        if position.direction == "long":
            tp_dist = position.tp - position.entry_price
            if tp_dist <= 0:
                return 0.0
            return (price - position.entry_price) / tp_dist
        tp_dist = position.entry_price - position.tp
        if tp_dist <= 0:
            return 0.0
        return (position.entry_price - price) / tp_dist

    def _breakeven_sl_price(self, position: Position) -> float:
        off = SCALP_BREAKEVEN_OFFSET_PTS
        if position.direction == "long":
            return position.entry_price + off
        return position.entry_price - off

    def _secure_sl_price(self, position: Position) -> float:
        """Lock a fraction of the TP distance (secure profit before stall / max-hold)."""
        lock = max(0.05, min(0.95, SCALP_SECURE_LOCK_PCT))
        if position.direction == "long":
            tp_dist = position.tp - position.entry_price
            return position.entry_price + tp_dist * lock
        tp_dist = position.entry_price - position.tp
        return position.entry_price - tp_dist * lock

    def _trail_sl_from_mfe(self, position: Position) -> float:
        """After breakeven, trail SL to lock a fraction of max favorable excursion."""
        lock = max(0.05, min(0.95, SCALP_TRAIL_MFE_LOCK_PCT))
        mf = position.max_favorable_price or position.entry_price
        if position.direction == "long":
            mfe = mf - position.entry_price
            if mfe <= 0:
                return position.sl
            return position.entry_price + mfe * lock
        mfe = position.entry_price - mf
        if mfe <= 0:
            return position.sl
        return position.entry_price - mfe * lock

    @staticmethod
    def _is_tighter_sl(position: Position, new_sl: float, old_sl: float) -> bool:
        if position.direction == "long":
            return new_sl > old_sl + 1e-9
        return new_sl < old_sl - 1e-9

    def _update_mfe(
        self,
        position: Position,
        bar_high: float,
        bar_low: float,
        current_price: float,
    ) -> None:
        if position.direction == "long":
            base = position.max_favorable_price or position.entry_price
            position.max_favorable_price = max(base, bar_high, current_price)
        else:
            base = position.max_favorable_price or position.entry_price
            position.max_favorable_price = min(base, bar_low, current_price)

    def _approaching_max_hold(self, position: Position) -> bool:
        if MAX_HOLD_SECONDS <= 0:
            return False
        hold = self._position_hold_seconds(position)
        threshold = MAX_HOLD_SECONDS * max(0.5, min(1.0, SCALP_MAX_HOLD_TIGHTEN_PCT))
        return hold >= threshold

    def _apply_stop_update(
        self,
        symbol: str,
        position: Position,
        new_sl: float,
    ) -> bool:
        """Tighten SL locally and on broker (native modify or protective cancel/replace)."""
        old_sl = position.sl
        if not self._is_tighter_sl(position, new_sl, old_sl):
            return False

        new_sl, _ = round_bracket_prices(symbol, position.direction, new_sl, position.tp)
        if not self._is_tighter_sl(position, new_sl, old_sl):
            return False

        if self.paper_mode and not position.broker_bracket:
            position.sl = new_sl
            return True

        if not self.broker or not self.broker.connected:
            return False

        side = "BUY" if position.direction == "long" else "SELL"
        ok = self.broker.update_stop_loss(
            ticket=position.order_id,
            symbol=symbol,
            side=side,
            size=position.size,
            new_sl=new_sl,
            take_profit=position.tp,
        )
        if ok:
            position.sl = new_sl
        return ok

    def _update_breakeven_trail(
        self,
        symbol: str,
        position: Position,
        current_price: float,
        bar_high: float,
        bar_low: float,
    ) -> None:
        """Move SL to breakeven / trail / secure-profit when thresholds are hit."""
        if not SCALP_BREAKEVEN_ENABLED or current_price <= 0:
            return

        self._update_mfe(position, bar_high, bar_low, current_price)
        fav_price = bar_high if position.direction == "long" else bar_low
        pct_fav = self._profit_pct_to_tp(position, fav_price)

        candidates: List[float] = []

        if not position.breakeven_hit and pct_fav >= SCALP_BREAKEVEN_PCT:
            candidates.append(self._breakeven_sl_price(position))

        if (position.breakeven_hit or pct_fav >= SCALP_BREAKEVEN_PCT) and SCALP_TRAIL_AFTER_BE:
            candidates.append(self._trail_sl_from_mfe(position))

        secure_trigger = pct_fav >= SCALP_SECURE_PROFIT_PCT
        if not secure_trigger and self._approaching_max_hold(position):
            secure_trigger = self._profit_pct_to_tp(position, current_price) > 0.05
        if secure_trigger and not position.secure_tightened:
            candidates.append(self._secure_sl_price(position))

        if not candidates:
            return

        if position.direction == "long":
            new_sl = max(candidates)
        else:
            new_sl = min(candidates)

        if not self._is_tighter_sl(position, new_sl, position.sl):
            return

        was_be = not position.breakeven_hit and pct_fav >= SCALP_BREAKEVEN_PCT
        was_secure = secure_trigger and not position.secure_tightened
        old_sl = position.sl
        if not self._apply_stop_update(symbol, position, new_sl):
            return

        if was_be:
            position.breakeven_hit = True
        if was_secure:
            position.secure_tightened = True

        pct_log = self._profit_pct_to_tp(position, current_price) * 100.0
        if was_secure:
            label = "secure"
        elif was_be:
            label = "breakeven"
        else:
            label = "trail"

        msg = (
            f"🔒 {symbol} {label} SL @ {position.sl:.2f} "
            f"(was {old_sl:.2f}, profit {pct_log:.0f}% to TP)"
        )
        print(f"   {msg}")
        bot_logger.info(msg)

    def _try_max_hold_exit(
        self,
        symbol: str,
        order_id: str,
        position: Position,
        current_price: float,
    ) -> bool:
        """Force market exit when MAX_HOLD_SECONDS exceeded (before SL/TP)."""
        if MAX_HOLD_SECONDS <= 0 or current_price <= 0:
            return False
        hold_sec = self._position_hold_seconds(position)
        if hold_sec < MAX_HOLD_SECONDS:
            return False
        print(
            f"   ⏱️ {symbol}: MAX HOLD {hold_sec:.0f}s >= {MAX_HOLD_SECONDS}s "
            f"— force market exit @ {current_price:.2f}"
        )
        needs_broker_close = (
            self.broker
            and self.broker.connected
            and (
                not self.paper_mode
                or position.broker_bracket
            )
        )
        if needs_broker_close:
            self.broker.close_position(symbol=symbol, ticket=order_id)
            time.sleep(2)
        self.close_position(symbol, order_id, "MAX_HOLD", current_price)
        return True

    def check_exit(
        self,
        position: Position,
        current_price: float,
        bar_high: float,
        bar_low: float,
    ) -> Optional[Tuple[str, float]]:
        """Check if position should close; use bar high/low so SL/TP fill at bracket prices."""
        symbol = position.symbol
        if current_price <= 0 or bar_high <= 0 or bar_low <= 0 or bar_high < bar_low:
            return None

        if MAX_HOLD_SECONDS > 0 and self._position_hold_seconds(position) >= MAX_HOLD_SECONDS:
            return ('MAX_HOLD', current_price)

        max_deviation = abs(position.entry_price) * 0.1
        if abs(current_price - position.entry_price) > max_deviation:
            print(f"   ⚠️ Ignoring suspicious price {current_price:.2f} (entry was {position.entry_price:.2f})")
            return None

        spec = SYMBOL_SPECS[symbol]
        point_value = spec['point_value']
        max_loss = spec_limit(symbol, 'max_loss_per_trade', MAX_LOSS_PER_TRADE)

        if position.direction == 'long':
            # SL before TP on same bar (conservative)
            if bar_low <= position.sl:
                return ('SL', position.sl)
            if bar_high >= position.tp:
                return ('TP', position.tp)
            worst_price = bar_low
            unrealized_pnl = (worst_price - position.entry_price) * point_value * position.size
        else:
            if bar_high >= position.sl:
                return ('SL', position.sl)
            if bar_low <= position.tp:
                return ('TP', position.tp)
            worst_price = bar_high
            unrealized_pnl = (position.entry_price - worst_price) * point_value * position.size

        if unrealized_pnl < -max_loss:
            # Gap/slippage past bracket — cap exit at worst-case max-loss price
            if position.direction == 'long':
                exit_price = position.entry_price - (max_loss / (point_value * position.size))
            else:
                exit_price = position.entry_price + (max_loss / (point_value * position.size))
            print(
                f"   🛑 {symbol}: MAX LOSS (intrabar) ${unrealized_pnl:.2f} < -${max_loss:.0f} "
                f"→ exit @ {exit_price:.2f}"
            )
            return ('MAX_LOSS', exit_price)

        return None

    def _latest_bar_ts(self, row) -> Optional[pd.Timestamp]:
        ts = row.get("datetime")
        if ts is None or pd.isna(ts):
            return None
        return pd.Timestamp(ts)

    def _latest_1m_bar_ts(self, row) -> Optional[pd.Timestamp]:
        return self._latest_bar_ts(row)

    def _is_new_1m_bar(self, symbol: str, row) -> bool:
        ts = self._latest_1m_bar_ts(row)
        if ts is None:
            return True
        prev = self._last_entry_1m_bar.get(symbol)
        if prev is None:
            return True
        return prev != ts

    def _mark_entry_1m_bar_evaluated(self, symbol: str, row) -> None:
        ts = self._latest_1m_bar_ts(row)
        if ts is not None:
            self._last_entry_1m_bar[symbol] = ts

    def _is_new_30s_bar(self, symbol: str, row) -> bool:
        ts = self._latest_bar_ts(row)
        if ts is None:
            return True
        prev = self._last_entry_30s_bar.get(symbol)
        if prev is None:
            return True
        return prev != ts

    def _mark_entry_30s_bar_evaluated(self, symbol: str, row) -> None:
        ts = self._latest_bar_ts(row)
        if ts is not None:
            self._last_entry_30s_bar[symbol] = ts

    def _resolve_current_price(self, price_data: Dict) -> float:
        bid = price_data.get('bid', 0)
        ask = price_data.get('ask', 0)
        last = price_data.get('last', 0)
        if last > 0:
            return last
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return 0.0

    def _try_close_position(
        self,
        symbol: str,
        order_id: str,
        position: Position,
        bar_high: float,
        bar_low: float,
        current_price: float,
    ) -> bool:
        """Evaluate exit and close position; returns True if closed."""
        if not self.paper_mode and position.broker_bracket:
            return False

        exit_result = self.check_exit(position, current_price, bar_high, bar_low)
        if not exit_result:
            return False

        reason, exit_price = exit_result
        needs_broker_close = (
            self.broker
            and self.broker.connected
            and (
                (self.paper_mode and position.broker_bracket)
                or (not self.paper_mode and not position.broker_bracket)
            )
        )
        if needs_broker_close:
            self.broker.close_position(symbol=symbol, ticket=order_id)
            time.sleep(2)
        self.close_position(symbol, order_id, reason, exit_price)
        return True

    def close_position(self, symbol: str, order_id: str, reason: str, exit_price: float):
        """Close a tracked position by order id."""
        bucket = self.positions.get(symbol)
        if not bucket or order_id not in bucket:
            return

        position = bucket[order_id]
        spec = SYMBOL_SPECS[symbol]
        point_value = spec['point_value']
        
        # Calculate P&L with real fill price
        if position.direction == 'long':
            pnl = (exit_price - position.entry_price) * point_value * position.size
        else:
            pnl = (position.entry_price - exit_price) * point_value * position.size
        self.daily_pnl += pnl
        emoji = "✅" if pnl > 0 else "❌"
        print(f"{emoji} Closed {symbol} {position.direction.upper()} @ {exit_price:.2f} ({reason})")
        print(f"   P&L: ${pnl:+.2f} | Daily: ${self.daily_pnl:+.2f}")
        
        # Send email notification for closed trade
        send_email(
            f"Closed: {symbol} ({reason})",
            f"Trade Closed!\n\nSymbol: {symbol}\nDirection: {position.direction.upper()}\nEntry: {position.entry_price:.2f}\nExit: {exit_price:.2f}\nReason: {reason}\nP&L: ${pnl:+.2f}\nDaily P&L: ${self.daily_pnl:+.2f}\nTime: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        
        # Log trade
        trade = {
            'symbol': symbol,
            'order_id': order_id,
            'direction': position.direction,
            'entry': position.entry_price,
            'exit': exit_price,
            'sl': position.sl,
            'tp': position.tp,
            'pnl': pnl,
            'reason': reason,
            'entry_time': position.entry_time.isoformat(),
            'exit_time': datetime.now(timezone.utc).isoformat(),
        }
        self.trades.append(trade)
        self._record_hybrid_trade_close(position, reason, exit_price, pnl)
        self._record_adaptive_trade_close(position, reason, exit_price, pnl)
        self._update_consec_loss_state(symbol, pnl)
        # Save to log
        with open(self.log_file, 'w') as f:
            json.dump(self.trades, f, indent=2)

        if pnl < 0 and reason not in ("BE",):
            self.loss_cooldown_until[symbol] = (
                datetime.now(timezone.utc) + timedelta(minutes=LOSS_COOLDOWN_MINUTES)
            )
            print(f"   ⏳ {symbol}: {LOSS_COOLDOWN_MINUTES} min loss cooldown before re-entry")
        if STRATEGY_MODE == "fvs1" and FVS1_CFG is not None:
            net_pnl = pnl - FVS1_CFG.round_trip_fee
            self._fvs1_risk.record_trade(net_pnl, FVS1_CFG)
            e = self._fvs1_risk.expectancy()
            print(
                f"   FVS-1 risk: E={e:.2f} over {len(self._fvs1_risk.recent_pnls)} trades "
                f"(consec_loss={self._fvs1_risk.consecutive_losses})"
            )

        if not self.paper_mode and self.broker and self.broker.connected:
            legs = self.broker.cancel_entry_bracket_legs(
                order_id, symbol, reason=f"close_{reason}",
            )
            orphans = self.broker.cancel_all_bot_orders(symbol)
            if legs or orphans:
                bot_logger.info(
                    f"{symbol} close {order_id}: cancelled bracket legs={legs} "
                    f"orphan_orders={orphans} ({reason})"
                )

        del bucket[order_id]
        if not bucket:
            del self.positions[symbol]
        return

    def flatten_for_session(self, reason: str) -> None:
        """Flatten broker + local MNQ (and other) positions when a session/window ends."""
        if not self.broker or not getattr(self.broker, "connected", False):
            return
        for symbol in list(self.symbols):
            local = self.positions.get(symbol) or {}
            broker_net = 0
            if not local:
                try:
                    broker_net = abs(int(self.broker.get_symbol_list_positions_net(symbol) or 0))
                except Exception:
                    broker_net = 0
                if broker_net <= 0:
                    continue
            print(f"📤 {reason}: flattening {symbol}")
            try:
                result = self.broker.flatten_symbol(symbol)
                print(f"   flatten {symbol}: {result}")
            except Exception as e:
                print(f"   flatten {symbol} failed: {e}")
            px = 0.0
            try:
                quote = self.broker.get_latest_price(symbol)
                if quote:
                    px = self._resolve_current_price(quote)
            except Exception:
                px = 0.0
            self._sync_all_broker_closed_positions(symbol)
            for order_id, pos in list((self.positions.get(symbol) or {}).items()):
                self.close_position(symbol, order_id, reason, px or pos.entry_price)

    def run(self, duration_minutes: int = 0):
        """Main trading loop - scans all symbols."""
        print(f"\n{'='*70}")
        print(f"  🔪 SAFE SCALPING BOT - MICROS")
        print(f"{'='*70}")
        if self.paper_mode:
            print(f"  📝 PAPER MODE — orders NOT sent to Rithmic")
            print(f"     For LIVE: start_nq_live.bat  or  --live")
        else:
            print(f"  🔴 LIVE MODE — REAL orders will be sent to Rithmic")
        print(f"{'='*70}")
        print_profit_mode_banner()
        print(f"Strategy: 15M+5M Trend + 1M Entry ({STRATEGY_MODE})")
        if USE_15M_BIAS:
            gate_label = "gates entries" if USE_15M_ENTRY_GATE else "direction only — not gating entries"
            print(
                f"15M bias: ON ({gate_label}) | "
                f"Bear adaptive: {'ON' if BEAR_ADAPTIVE else 'off'} | "
                f"Bull adaptive: {'ON' if BULL_ADAPTIVE else 'off'}"
            )
        print(f"Symbols: {', '.join(self.symbols)}")
        for sym in self.symbols:
            print(f"  Contract sizing: {symbol_risk_line(sym)}")
        if len(self.symbols) > 1 and NASDQ_SYMBOLS.issuperset(self.symbols):
            print(
                f"Multi-symbol: scans both; MNQ up to {MAX_POSITIONS_MNQ} concurrent, "
                f"NQ up to {MAX_POSITIONS}; 1 NQ contract = 10× MNQ $/point"
            )
        print(f"Session mode: {SESSION_MODE.upper()} — {session_mode_label(SESSION_MODE)}")
        if STRATEGY_MODE == "scalp_hybrid":
            print(f"Scalp windows: {format_session_windows(SCALP_SESSIONS)}")
        print(f"Max Position: {MAX_POSITIONS_MNQ} (MNQ concurrent)")
        if any(s != "MNQ" for s in self.symbols):
            others = [s for s in self.symbols if s != "MNQ"]
            print(f"Max Position (other): {MAX_POSITIONS} ({', '.join(others)})")
        print(f"R:R Ratio: 1:{TP_MULT:.1f} (SL={ATR_MULT:.1f}×ATR, TP capped {TP_BUFFER_ATR_MULT:.1f}×ATR before structure)")
        print(f"Daily Loss Limit: ${daily_loss_limit_for(self.symbols):.0f} (max across active symbols)")
        print(f"Volatility Filter: {', '.join(f'{s}≤{max_1m_bar_pts_for(s):.0f}pts' for s in self.symbols)}")
        print(
            f"DI tolerance: {DI_TOLERANCE:.0f} pts"
            + (f" → {DI_FLOW_TOLERANCE:.0f} pts when flow confirms" if USE_FLOW_DI_OVERRIDE and USE_ORDER_FLOW else "")
        )
        print(f"Max Trades/Day: {MAX_TRADES_PER_DAY} (all symbols)")
        print(f"Loss cooldown after any loss: {LOSS_COOLDOWN_MINUTES} min per symbol")
        if MTF_MAX_CONSEC_LOSSES > 0:
            print(
                f"Consecutive-loss halt: {MTF_MAX_CONSEC_LOSSES} losses → "
                f"{MTF_CONSEC_LOSS_PAUSE_MIN} min pause per symbol"
            )
        else:
            print("Consecutive-loss halt: OFF (set MTF_MAX_CONSEC_LOSSES>0 to enable)")
        if USE_ADAPTIVE_LEARNER:
            skip_on = adaptive_skip_enabled()
            print(
                f"Adaptive learner: ON (skip={'BLOCKING' if skip_on else 'OFF/advisory'}) — "
                f"updates data/adaptive_learning.json on each close"
            )
            if skip_on:
                print(
                    "  ⚠️ ADAPTIVE_SKIP_ENABLED=true — regime/hour/pair skips BLOCK entries. "
                    "Set ADAPTIVE_SKIP_ENABLED=false for profitability path."
                )
        else:
            print("Adaptive learner: OFF (set USE_ADAPTIVE_LEARNER=true)")
        if DAILY_HALF_STOP_ENABLED:
            half = daily_loss_limit_for(self.symbols) * DAILY_HALF_STOP_PCT
            print(f"Daily half-stop: pause new entries at -${half:.0f} ({DAILY_HALF_STOP_PCT:.0%} of daily limit)")
        print(f"Scan interval: {SCAN_SLEEP_OPEN_SEC}s with position / {SCAN_SLEEP_IDLE_SEC}s idle")
        if MAX_HOLD_SECONDS > 0:
            print(
                f"Max hold: {MAX_HOLD_SECONDS}s — force market exit when exceeded "
                f"(enforce ~±{SCAN_SLEEP_OPEN_SEC + 3}s due to scan cadence)"
            )
        if SCALP_BREAKEVEN_ENABLED:
            print(
                f"Breakeven SL: ON @ {SCALP_BREAKEVEN_PCT:.0%} to TP "
                f"(offset {SCALP_BREAKEVEN_OFFSET_PTS} pt"
                f"{', trail after BE' if SCALP_TRAIL_AFTER_BE else ''}"
                f", secure @ {SCALP_SECURE_PROFIT_PCT:.0%} to TP)"
            )
        else:
            print("Breakeven SL: OFF (SCALP_BREAKEVEN_ENABLED=false)")
        if STRATEGY_MODE == "scalp_hybrid":
            print(
                "Entry signals: evaluated on each new 30s bar close "
                f"(hybrid; trend={SCALP_TREND_MODE}, gap={MIN_SECONDS_BETWEEN_ENTRIES}s)"
            )
        elif STRATEGY_MODE == "fvs1":
            log_label = "log-only" if (FVS1_CFG and FVS1_CFG.log_only) else "live entries"
            print(f"Entry signals: evaluated on each new 30s bar close (FVS-1; {log_label})")
        elif STRATEGY_MODE == "scalp_b" and USE_30S_BARS:
            print("Entry signals: evaluated on each new 30s bar close (scalp_b)")
        else:
            print("Entry signals: evaluated once per new 1M bar close (position mgmt every scan)")
        print(f"Smart Filters: {'ON' if SMART_FILTERS_ENABLED else 'OFF (backtest: filters hurt PF)'}")
        print(f"LLM Advisor: {'ON' if self.llm_advisor.enabled else 'OFF'}")
        if STRATEGY_MODE == "scalp_hybrid":
            if self.trade_learner.blocking_active:
                if USE_DEEPSEEK_LEARNER and self.trade_learner.api_key:
                    learner_label = f"ON (DeepSeek every {DEEPSEEK_LEARN_EVERY_N} closes + local blocks)"
                elif USE_DEEPSEEK_LEARNER:
                    learner_label = "ON (local blocks — set DEEPSEEK_API_KEY for AI)"
                else:
                    learner_label = "ON (local pattern blocks — no API key)"
            else:
                learner_label = "journal-only (set USE_LOCAL_PATTERN_LEARNER=true)"
            print(
                f"Trade Learner: {learner_label} | "
                f"{self.trade_learner.get_status_line()}"
            )
        if self.news_bias.enabled:
            print(
                f"News Bias (DeepSeek): ON — mode={self.news_bias.mode} "
                f"(headlines={headline_providers_label(self.news_bias.newsapi_key, self.news_bias.tiingo_key, self.news_bias.use_tiingo_news)})"
            )
        else:
            print("News Bias: OFF (set USE_LLM_NEWS=true + DEEPSEEK_API_KEY)")
        if self.policy_scorer.enabled:
            print(
                f"Policy Scorer: ON — mode={self.policy_scorer.mode} "
                f"(min |score|>={self.policy_scorer.min_score}, "
                f"conf>={self.policy_scorer.min_confidence:.0%})"
            )
        else:
            print("Policy Scorer: OFF (set USE_POLICY_SCORER=true)")
        if USE_ORDER_FLOW:
            print(
                f"Order Flow: ON — mode={ORDER_FLOW_MODE} "
                f"(window={ORDER_FLOW_WINDOW_SEC}s; "
                f"long: delta>0 or buy>55% | short: delta<0 or buy<45%)"
            )
        else:
            print("Order Flow: OFF (set USE_ORDER_FLOW=true)")
        if USE_30S_BARS:
            print(
                f"30s trigger bars: ON — {TRIGGER_BAR_SECONDS}s via Rithmic SECOND_BAR "
                f"(tick aggregator fallback if Lucid blocks sub-minute history)"
            )
        else:
            print("30s trigger bars: OFF (set USE_30S_BARS=true)")
        if FLOW_COUNTER_CFG.get("enabled"):
            print(
                f"Flow counter-trend: ON — DI≥{FLOW_COUNTER_CFG['di_margin']:.0f} "
                f"ADX≥{FLOW_COUNTER_CFG['adx_min']} when flow confirms "
                f"(std DI≥{DI_COUNTER_TREND:.0f} ADX≥{COUNTER_ADX}; "
                f"shorts={'on' if COUNTER_TREND_SHORTS else 'off'} "
                f"longs={'on' if COUNTER_TREND_LONGS else 'off'}; "
                f"regime scans={FLOW_COUNTER_CFG.get('regime_scans', 3)})"
            )
        else:
            print(
                f"Flow counter-trend: OFF — std counter DI≥{DI_COUNTER_TREND:.0f} ADX≥{COUNTER_ADX} only"
            )
        print(f"Skip reasons: {'ON' if VERBOSE_SKIP_REASONS else 'OFF (set VERBOSE_SKIP_REASONS=true)'}")
        if scalp_fast_mode_active():
            print(
                f"Fast scalp: ON — 1M trend when 5M<{MIN_5M_BARS_SCALP_IDEAL} bars | "
                f"floor={MIN_5M_BARS_SCALP_FLOOR} | 1M fetch={CANDLE_1M_COUNT} "
                f"({CANDLE_HISTORY_HOURS}h lookback)"
            )
        print(
            f"Scan log: {'FAST (one-line)' if FAST_SCAN_LOG else 'verbose'} | "
            f"News: {'ON' if SHOW_NEWS else 'OFF (SHOW_NEWS=false)'}"
        )
        print(
            f"News console: {'ON' if NEWS_CONSOLE else 'OFF (file log only)'} | "
            f"{'compact' if COMPACT_NEWS else 'verbose'} | headlines={NEWS_HEADLINE_COUNT}"
        )
        print(
            f"Policy console: {'ON' if POLICY_CONSOLE else 'OFF (file log only)'}"
        )
        print(f"{'='*70}\n")
        
        if not self.connect():
            return
        
        if not self.paper_mode and not self.skip_confirm:
            confirm = input("\n⚠️  LIVE TRADING MODE - Type 'YES' to confirm: ")
            if confirm != 'YES':
                print("Aborted.")
                return
        
        start_time = datetime.now()
        end_time = (
            None if duration_minutes <= 0
            else start_time + timedelta(minutes=duration_minutes)
        )
        
        print(f"\n🚀 Starting trading loop...")
        if end_time is None:
            print("   Will run until stopped (Ctrl+C) — unattended MNQ")
        else:
            print(f"   Will run until {end_time.strftime('%H:%M:%S')}")
        print(f"   Waiting 10s for Rithmic connection to stabilize...\n")
        time.sleep(10)  # Let Rithmic connection stabilize
        self._warm_30s_bars()
        if STRATEGY_MODE == "scalp_hybrid":
            self.trade_learner.on_session_start()
        
        loop_count = 0
        while end_time is None or datetime.now() < end_time:
            try:
                loop_count += 1

                # ── Trading Hours Check ──
                if not is_market_open_et():
                    now_et = datetime.now(pytz.timezone('US/Eastern'))
                    closed_label = "Globex halt" if SESSION_MODE == SESSION_EXTENDED else "RTH closed"
                    if self.positions:
                        self.flatten_for_session("SESSION_END")
                    print(f"⏸️  {closed_label} ({SESSION_MODE}) — {now_et.strftime('%a %H:%M ET')}")
                    sleep_sec = seconds_until_session_open_et(now_et, SESSION_MODE)
                    sleep_sec = min(sleep_sec, 3600) if SESSION_MODE == SESSION_RTH else sleep_sec
                    time.sleep(sleep_sec)
                    continue

                # ── Scalp liquid windows (skip Globex dead zones / lunch chop) ──
                if STRATEGY_MODE == "scalp_hybrid" and SCALP_SESSIONS:
                    now_et = datetime.now(pytz.timezone("US/Eastern"))
                    in_win, win_label = is_in_session_windows_et(now_et, SCALP_SESSIONS)
                    if not in_win:
                        if self.positions:
                            self.flatten_for_session("WINDOW_END")
                        sleep_sec = seconds_until_scalp_window_et(
                            now_et, SCALP_SESSIONS, SESSION_MODE,
                        )
                        sleep_sec = min(max(60.0, sleep_sec), 1800)
                        print(
                            f"⏸️  Outside SCALP_SESSIONS ({win_label}) — "
                            f"{now_et.strftime('%a %H:%M ET')} | sleep {int(sleep_sec)}s"
                        )
                        time.sleep(sleep_sec)
                        continue
                
                # Check daily limits
                if not self.check_daily_limits():
                    if self.positions:
                        self.flatten_for_session("DAILY_LIMIT")
                    print("⏸️  Daily limits reached - waiting for next day")
                    time.sleep(300)
                    continue

                # Repair missing SL/TP before slow per-symbol candle fetches
                if self.positions and not self.paper_mode:
                    self._repair_all_open_positions()

                trade_counts = ", ".join(
                    f"{s}:{self._daily_trades_for(s)}/{max_trades_per_day_for(s)}"
                    for s in self.symbols
                )
                scan_parts = [f"🔍 Scan #{loop_count}", f"Trades today: {trade_counts}"]
                if "MNQ" in self.symbols:
                    mnq_total, mnq_detail = self._concurrent_exposure("MNQ")
                    mnq_max = max_positions_for("MNQ")
                    mnq_local = int(mnq_detail.get("local", 0))
                    untracked = int(mnq_detail.get("untracked_broker", 0))
                    tag_inf = int(mnq_detail.get("tag_inferred", 0))
                    working = int(mnq_detail.get("working", 0))
                    if untracked > 0:
                        scan_parts.append(
                            f"MNQ untracked broker: {untracked} "
                            f"(broker_net={mnq_detail['broker_net']} open {mnq_total}/{mnq_max}, "
                            f"local {mnq_local}) — entries blocked"
                        )
                    elif mnq_total >= mnq_max:
                        scan_parts.append(
                            f"MNQ at capacity: {mnq_total}/{mnq_max} "
                            f"(local={mnq_local} inferred={tag_inf} working={working}) "
                            f"— entries blocked"
                        )
                    elif mnq_total > mnq_local:
                        scan_parts.append(
                            f"Open MNQ: {mnq_total}/{mnq_max} "
                            f"(local={mnq_local} inferred={tag_inf} working={working})"
                        )
                    else:
                        scan_parts.append(
                            f"Open MNQ: {mnq_total}/{mnq_max}"
                        )

                if (
                    ORPHAN_SWEEP_EVERY_N_SCANS > 0
                    and loop_count % ORPHAN_SWEEP_EVERY_N_SCANS == 0
                ):
                    self._sweep_orphan_bot_orders()
                print(" | ".join(scan_parts))

                macro_symbols = [
                    s for s in self.symbols
                    if s in NASDQ_SYMBOLS or s in ("MES", "MGC")
                ]
                if macro_symbols and SHOW_NEWS and (self.news_bias.enabled or self.policy_scorer.enabled):
                    if self.news_bias.enabled:
                        newsapi_key = self.news_bias.newsapi_key
                        tiingo_key = self.news_bias.tiingo_key
                        use_tiingo = self.news_bias.use_tiingo_news
                        cooldown = self.news_bias.newsapi_cooldown_min
                    else:
                        newsapi_key = self.policy_scorer.newsapi_key
                        tiingo_key = self.policy_scorer.tiingo_key
                        use_tiingo = self.policy_scorer.use_tiingo_news
                        cooldown = self.policy_scorer.newsapi_cooldown_min
                    for nb_sym in macro_symbols:
                        headlines = fetch_headlines_for_symbol(
                            nb_sym,
                            newsapi_key,
                            cooldown_min=cooldown,
                            tiingo_key=tiingo_key,
                            use_tiingo=use_tiingo,
                        )
                        if self.news_bias.enabled:
                            bias = self.news_bias.get_bias(nb_sym, headlines=headlines)
                            self._news_bias_cache[nb_sym] = bias
                            news_line = self.news_bias.format_scan_line(
                                nb_sym, bias, headlines=headlines, compact=COMPACT_NEWS,
                            )
                            if news_line:
                                print(news_line)
                        if self.policy_scorer.enabled and nb_sym in NASDQ_SYMBOLS:
                            policy = self.policy_scorer.get_score(nb_sym, headlines=headlines)
                            self._policy_cache[nb_sym] = policy
                            policy_line = self.policy_scorer.format_scan_line(
                                nb_sym, policy, compact=COMPACT_NEWS,
                            )
                            if policy_line:
                                print(policy_line)
                
                # Process each symbol
                for symbol in self.symbols:
                    # Longer delay between symbols to prevent Rithmic lock timeout
                    time.sleep(8)
                    
                    # Fetch data for this symbol
                    scalp_mode = STRATEGY_MODE in ("scalp_b", "scalp_hybrid", "fvs1")
                    min_1m_ideal = 50 if not scalp_fast_mode_active() else max(30, MIN_5M_BARS_SCALP_FLOOR + 5)
                    min_1m_floor = MIN_5M_BARS_SCALP_FLOOR if scalp_mode else 50
                    min_1m_count = CANDLE_1M_COUNT if scalp_mode else 100
                    if (
                        scalp_fast_mode_active()
                        and USE_30S_BARS
                        and STRATEGY_MODE in ("scalp_b", "scalp_hybrid", "fvs1")
                    ):
                        df_30s_pre = self.get_candles_seconds(
                            symbol, count=120, min_bars=2,
                        )
                        if df_30s_pre is not None and len(df_30s_pre) >= 2:
                            self._df_30s_cache[symbol] = self.add_30s_indicators(df_30s_pre)
                    df_1m = self.get_candles(
                        symbol, timeframe_minutes=1, count=min_1m_count,
                        min_bars=min_1m_ideal, min_bars_floor=min_1m_floor,
                        deep_history=scalp_mode and not self.paper_mode,
                    )
                    time.sleep(5)  # Longer pause between requests
                    df_5m = self._resolve_5m_candles(symbol, df_1m)

                    stale_fallback = False
                    if df_1m is None and self._df_1m_cache is not None and not self._df_1m_cache.empty:
                        df_1m = self._df_1m_cache.copy()
                        stale_fallback = True
                    if df_5m is None and self._df_5m_cache is not None and not self._df_5m_cache.empty:
                        df_5m = self._df_5m_cache.copy()
                        stale_fallback = True

                    if df_1m is None:
                        df_1m = self._derive_1m_from_30s(symbol, min_1m_count)
                    if df_1m is None:
                        if VERBOSE_SKIP_REASONS or not FAST_SCAN_LOG:
                            print(f"   ⚠️ {symbol}: 1M candle fetch failed — skip scan this loop")
                        continue
                    if df_5m is None and not scalp_fast_mode_active():
                        if VERBOSE_SKIP_REASONS or not FAST_SCAN_LOG:
                            print(f"   ⚠️ {symbol}: 5M candle fetch failed — skip scan this loop")
                        continue
                    if df_5m is None and scalp_fast_mode_active():
                        rs = resample_1m_to_5m(df_1m)
                        df_5m = rs if rs is not None and len(rs) else pd.DataFrame(columns=[
                            "datetime", "open", "high", "low", "close", "volume",
                        ])
                    if stale_fallback and VERBOSE_SKIP_REASONS:
                        print(f"   ⚠️ {symbol}: Rithmic stall — using stale candle cache this scan")
                    
                    # Add indicators
                    df_1m = self.add_1m_indicators(df_1m)
                    if df_5m is not None and len(df_5m) > 0:
                        df_5m = self.add_5m_indicators(df_5m)
                    self._df_1m_cache = df_1m
                    self._df_5m_cache = df_5m if df_5m is not None and len(df_5m) > 0 else None
                    self._df_15m_cache = None
                    if USE_15M_BIAS:
                        time.sleep(5)
                        df_15m = self.get_candles(symbol, timeframe_minutes=15, count=250)
                        if df_15m is not None and len(df_15m) >= 50:
                            self._df_15m_cache = self.add_5m_indicators(df_15m)
                    
                    # Trend context — 1M in fast mode when 5M is short
                    use_1m_trend = self._use_1m_trend_row(df_1m, df_5m)
                    ctx_5m = self.get_1m_trend_context(df_1m) if use_1m_trend else self.get_5m_context(df_5m)
                    ctx_15m = self.get_15m_context(self._df_15m_cache)
                    
                    trend = ctx_5m.get('trend', 'none')
                    adx = ctx_5m.get('adx', 0)
                    di_p = ctx_5m.get('di_plus', 0)
                    di_m = ctx_5m.get('di_minus', 0)
                    flow_snap = None
                    if USE_ORDER_FLOW and self.broker:
                        flow_snap = self.broker.get_tick_flow(symbol)
                    flow_streak = self._update_flow_regime_streak(symbol, ctx_5m, flow_snap)
                    adx_need = ADX_THRESHOLD
                    if trend == "bullish":
                        adx_need = effective_adx_threshold(ctx_5m, flow_snap, "long")
                    elif trend == "bearish":
                        adx_need = effective_adx_threshold(ctx_5m, flow_snap, "short")
                    adx_note = (
                        f"{adx_need}+ w/ flow" if adx_need < ADX_THRESHOLD else f"{ADX_THRESHOLD}+"
                    )
                    tf_label = "1M" if use_1m_trend else "5M"
                    if not FAST_SCAN_LOG or VERBOSE_SKIP_REASONS:
                        print(
                            f"   {symbol}: {tf_label} trend = {trend} | "
                            f"Strength = {adx:.0f} (need {adx_note}) | "
                            f"Buyers vs sellers = {di_p:.0f} vs {di_m:.0f}"
                        )
                        if use_1m_trend:
                            print(
                                f"   ⚡ {symbol}: fast mode — "
                                f"1M={len(df_1m)} bars, 5M={len(df_5m)} bars (trend on 1M)"
                            )
                    if USE_15M_BIAS and (not FAST_SCAN_LOG or VERBOSE_SKIP_REASONS):
                        print(format_15m_bias_line(symbol, ctx_15m))

                    if USE_ORDER_FLOW and self.broker and flow_snap is not None:
                        if not FAST_SCAN_LOG or VERBOSE_SKIP_REASONS:
                            print(f"   {TickFlowTracker.format_scan_line(flow_snap)}")

                    row_1m = df_1m.iloc[-1]
                    bar_high = float(row_1m['high'])
                    bar_low = float(row_1m['low'])

                    if not self.paper_mode and symbol in self.positions:
                        if self._sync_all_broker_closed_positions(symbol):
                            row_1m = df_1m.iloc[-1]
                            bar_high = float(row_1m['high'])
                            bar_low = float(row_1m['low'])

                    if USE_30S_BARS and STRATEGY_MODE in ("scalp_b", "scalp_hybrid", "fvs1"):
                        df_30s = self.get_candles_seconds(
                            symbol, count=120, min_bars=2, df_1m=df_1m,
                        )
                        if df_30s is not None and len(df_30s) >= 2:
                            self._df_30s_cache[symbol] = self.add_30s_indicators(df_30s)

                    new_1m_bar = self._is_new_1m_bar(symbol, row_1m)
                    self._print_entry_gate_diagnostics(
                        symbol, df_1m, ctx_5m, ctx_15m, flow_snap,
                        new_1m_bar=new_1m_bar, flow_streak=flow_streak, df_5m=df_5m,
                    )

                    # Entry signal when flat capacity + daily limits + new trigger bar
                    has_30s_cache = (
                        symbol in self._df_30s_cache
                        and len(self._df_30s_cache[symbol]) >= 2
                    )
                    use_30s_entry = (
                        USE_30S_BARS
                        and STRATEGY_MODE in ("scalp_b", "scalp_hybrid", "fvs1")
                        and has_30s_cache
                    )
                    use_1m_trigger_fallback = (
                        USE_30S_BARS
                        and STRATEGY_MODE == "scalp_hybrid"
                        and SCALP_30S_FALLBACK_1M
                        and not has_30s_cache
                        and (not scalp_fast_mode_active() or SCALP_AGGRESSIVE)
                    )
                    new_trigger_bar = new_1m_bar
                    run_entry_check = new_trigger_bar
                    row_30s_entry = None
                    if use_30s_entry or (scalp_fast_mode_active() and has_30s_cache):
                        row_30s_entry = self._df_30s_cache[symbol].iloc[-1]
                        if STRATEGY_MODE == "scalp_hybrid":
                            run_entry_check, is_new_30s = self._should_run_30s_entry_check(
                                symbol, row_30s_entry,
                            )
                            new_trigger_bar = is_new_30s
                        else:
                            new_trigger_bar = self._is_new_30s_bar(symbol, row_30s_entry)
                            run_entry_check = new_trigger_bar
                    elif USE_30S_BARS and STRATEGY_MODE in ("scalp_b", "scalp_hybrid", "fvs1"):
                        if not use_1m_trigger_fallback:
                            new_trigger_bar = False
                            run_entry_check = False

                    if self._has_position_capacity(symbol) and self.check_daily_limits():
                        if not run_entry_check:
                            open_count, _ = self._concurrent_exposure(symbol)
                            if open_count == 0 and (
                                FAST_SCAN_LOG or scalp_fast_mode_active() or SCALP_AGGRESSIVE
                            ):
                                src = f"{TRIGGER_BAR_SECONDS}s" if use_30s_entry or has_30s_cache else "1M"
                                print(f"   ⚡ {symbol}: same {src} bar — entry check skipped this scan")
                        else:
                            trigger_fired = False
                            trigger_reason = ""
                            if use_30s_entry and STRATEGY_MODE == "scalp_hybrid":
                                if self._is_new_30s_trigger_log_bar(symbol, row_30s_entry):
                                    self._log_30s_trigger_eval(
                                        symbol, df_1m, df_5m, flow_snap,
                                    )
                                    self._mark_30s_trigger_logged(symbol, row_30s_entry)
                                trigger_fired, _, trigger_reason = self._hybrid_trigger_status(
                                    symbol, df_1m, df_5m, flow_snap,
                                )
                            order_placed = False
                            signal = None
                            try:
                                signal = self.check_entry_signal(
                                    symbol, df_1m, ctx_5m, ctx_15m,
                                    verbose=VERBOSE_SKIP_REASONS, flow_streak=flow_streak,
                                )
                                if (
                                    not signal
                                    and self.policy_scorer.enabled
                                    and symbol in NASDQ_SYMBOLS
                                    and VERBOSE_SKIP_REASONS
                                    and POLICY_CONSOLE
                                ):
                                    pc = self._policy_cache.get(symbol) or {}
                                    if pc.get("score") is not None:
                                        conf = float(pc.get("confidence", 0))
                                        if conf > 1.0:
                                            conf /= 100.0
                                        cats = pc.get("categories") or []
                                        cat_str = cats[0] if cats else "—"
                                        print(
                                            f"      🏛️ {symbol}: {pc.get('score', 0):+d} [{cat_str}] "
                                            f"{conf:.0%}"
                                        )
                                if signal:
                                    row_1m = df_1m.iloc[-1]
                                    dt = row_1m.get("datetime", datetime.now(timezone.utc))
                                    size_pct = 100
                                    llm: Dict = {}
                                    if SMART_FILTERS_ENABLED:
                                        smart = self.smart_filters.evaluate(
                                            signal["direction"], dt, row_1m, ctx_5m,
                                            _cached_df(self._df_1m_cache, df_1m),
                                            _cached_df(self._df_5m_cache, df_5m),
                                            _cached_df(self._df_15m_cache),
                                        )
                                        if not smart["allowed"]:
                                            print(
                                                f"🧠 Smart skip {symbol}: "
                                                f"{', '.join(smart['block_reasons'])} "
                                                f"(score={smart['setup_score']})"
                                            )
                                            continue
                                        size_pct = smart.get("position_size_pct") or 100
                                    if self.news_bias.enabled and symbol in NASDQ_SYMBOLS:
                                        nb = self.news_bias.evaluate_direction(
                                            signal["direction"],
                                            symbol,
                                            bias=self._news_bias_cache.get(symbol),
                                        )
                                        if nb.get("advisory") and not COMPACT_NEWS:
                                            print(f"   {nb['advisory']}")
                                        if not nb.get("allowed", True):
                                            print(
                                                f"📰 News skip: {signal['direction'].upper()} {symbol} — "
                                                f"{nb.get('bias', 'neutral')} {nb.get('confidence', 0):.0%}"
                                            )
                                            continue
                                        boost = nb.get("size_adjust_pct") or 0
                                        if boost and size_pct < 100:
                                            size_pct = min(100, size_pct + boost)
                                    if self.policy_scorer.enabled and symbol in NASDQ_SYMBOLS:
                                        pc = self._policy_cache.get(symbol) or {}
                                        vol_ok = float(row_1m.get("volume_ratio", 0) or 0) >= VOLUME_RATIO_THRESHOLD
                                        pe = self.policy_scorer.evaluate_entry(
                                            signal["direction"],
                                            score=int(pc.get("score", 0)),
                                            confidence=float(pc.get("confidence", 0)),
                                            ctx_5m=ctx_5m,
                                            ctx_15m=ctx_15m,
                                            volume_confirmed=vol_ok,
                                        )
                                        if pe.get("advisory_note") and POLICY_CONSOLE:
                                            print(f"   🏛️ {pe['advisory_note']}")
                                        if not pe.get("allowed", True):
                                            print(
                                                f"🏛️ Policy skip: {signal['direction'].upper()} {symbol} — "
                                                f"{pe.get('reason', 'blocked')}"
                                            )
                                            continue
                                        pboost = pe.get("size_adjust_pct") or 0
                                        if pboost and size_pct < 100:
                                            size_pct = min(100, size_pct + pboost)
                                    if self.llm_advisor.enabled:
                                        llm = self.llm_advisor.evaluate_trade(
                                            signal, ctx_5m, row_1m=row_1m,
                                            df_1m=_cached_df(self._df_1m_cache, df_1m),
                                            df_5m=_cached_df(self._df_5m_cache, df_5m),
                                            df_15m=_cached_df(self._df_15m_cache),
                                            dt=dt,
                                        )
                                        if not llm.get("allowed", True):
                                            print(
                                                f"🤖 LLM skip: {signal['direction'].upper()} {symbol} — "
                                                f"{llm.get('reason', 'no reason')}"
                                            )
                                            continue
                                        size_pct = llm.get("position_size_pct") or size_pct
                                    if USE_ORDER_FLOW and self.broker and ORDER_FLOW_MODE == "block":
                                        flow_ev = self.broker.evaluate_order_flow(
                                            signal["direction"], symbol, mode=ORDER_FLOW_MODE,
                                        )
                                        if not flow_ev.get("allowed", True):
                                            print(
                                                f"📊 Flow skip: {signal['direction'].upper()} {symbol} — "
                                                f"{flow_ev.get('reason', 'flow conflict')}"
                                            )
                                            continue
                                    print(f"🎯 Signal: {signal['direction'].upper()} {symbol} @ {signal['entry']:.2f}")
                                    print(f"   Stop loss: {signal['sl']:.2f}  |  Take profit: {signal['tp']:.2f}", end="")
                                    if signal.get('structure_capped'):
                                        if signal['direction'] == 'long':
                                            print(f" (target below ceiling {signal.get('resistance', 0):.2f})")
                                        else:
                                            print(f" (target above floor {signal.get('support', 0):.2f})")
                                    else:
                                        print()
                                    print_trade_money(signal)
                                    if not self._has_position_capacity(symbol):
                                        self._log_position_capacity_block(symbol)
                                    else:
                                        self.place_order(signal)
                                        order_placed = True
                                    time.sleep(2)  # Delay after order
                            finally:
                                if use_30s_entry and row_30s_entry is not None:
                                    if STRATEGY_MODE == "scalp_hybrid":
                                        bar_ts = self._latest_bar_ts(row_30s_entry)
                                        if order_placed:
                                            self._mark_entry_30s_bar_evaluated(
                                                symbol, row_30s_entry,
                                            )
                                            self._hybrid_retry_30s_bar.pop(symbol, None)
                                        elif trigger_fired:
                                            if not signal:
                                                self._log_hybrid_entry_miss(
                                                    symbol, df_1m, df_5m, ctx_5m, flow_snap,
                                                    trigger_reason=trigger_reason,
                                                    flow_streak=flow_streak,
                                                )
                                            if bar_ts is not None:
                                                self._hybrid_retry_30s_bar[symbol] = bar_ts
                                        else:
                                            self._mark_entry_30s_bar_evaluated(
                                                symbol, row_30s_entry,
                                            )
                                            self._hybrid_retry_30s_bar.pop(symbol, None)
                                    else:
                                        self._mark_entry_30s_bar_evaluated(
                                            symbol, row_30s_entry,
                                        )
                                elif not use_30s_entry:
                                    self._mark_entry_1m_bar_evaluated(symbol, row_1m)

                    # Exit checks need a live quote (intrabar SL/TP vs mark)
                    for order_id, pos in list(self.positions.get(symbol, {}).items()):
                        if not self.paper_mode:
                            if self._sync_broker_closed_position(symbol, order_id):
                                continue
                            self._ensure_broker_brackets(symbol, pos)
                        price_data = self.broker.get_latest_price(symbol)
                        time.sleep(1)  # Delay after price request
                        if not price_data:
                            if VERBOSE_SKIP_REASONS:
                                print(f"   ⚠️ {symbol}: no live quote — exit check skipped this scan")
                            continue
                        current_price = self._resolve_current_price(price_data)
                        if current_price <= 0:
                            if VERBOSE_SKIP_REASONS:
                                print(f"   ⚠️ {symbol}: invalid live quote — exit check skipped")
                            continue
                        if self._try_max_hold_exit(symbol, order_id, pos, current_price):
                            continue
                        self._update_breakeven_trail(
                            symbol, pos, current_price, bar_high, bar_low,
                        )
                        self._try_close_position(
                            symbol, order_id, pos, bar_high, bar_low, current_price,
                        )
                
                # Status update every 10 loops
                if loop_count % 10 == 0:
                    pos_parts = []
                    for sym, bucket in self.positions.items():
                        pos_parts.append(f"{sym}:{len(bucket)}")
                    pos_str = ", ".join(pos_parts) if pos_parts else "None"
                    print(f"⏰ {datetime.now().strftime('%H:%M:%S')} | Positions: {pos_str} | Daily P&L: ${self.daily_pnl:+.2f}")

                # Faster scans + second exit pass while a position is open
                if self.positions:
                    time.sleep(SCAN_SLEEP_OPEN_SEC)
                    if not self.paper_mode:
                        self._repair_all_open_positions()
                    for sym, bucket in list(self.positions.items()):
                        for order_id, pos in list(bucket.items()):
                            time.sleep(3)
                            df_exit = self.get_candles(
                                sym, timeframe_minutes=1,
                                count=EXIT_CANDLE_COUNT, min_bars=10,
                            )
                            quote = self.broker.get_latest_price(sym)
                            if df_exit is None or not quote:
                                if not self.paper_mode:
                                    print(f"❌ LIVE exit check skipped for {sym} — no Rithmic candles/quote")
                                continue
                            row = df_exit.iloc[-1]
                            px = self._resolve_current_price(quote)
                            if px > 0:
                                if self._try_max_hold_exit(sym, order_id, pos, px):
                                    continue
                                self._update_breakeven_trail(
                                    sym, pos, px,
                                    float(row['high']), float(row['low']),
                                )
                                self._try_close_position(
                                    sym, order_id, pos,
                                    float(row['high']), float(row['low']), px,
                                )
                else:
                    time.sleep(SCAN_SLEEP_IDLE_SEC)
                
            except KeyboardInterrupt:
                print("\n\n⛔ Interrupted by user")
                if self.positions:
                    self.flatten_for_session("SHUTDOWN")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                tb = traceback.format_exc()
                print(tb)
                bot_logger.error(f"Scan loop error: {e}\n{tb}")
                time.sleep(30)
        
        # Final summary
        print(f"\n{'='*70}")
        print(f"  SESSION SUMMARY")
        print(f"{'='*70}")
        print(f"Symbols traded: {', '.join(self.symbols)}")
        print(f"Total Trades: {len(self.trades)}")
        total_pnl = sum(t['pnl'] for t in self.trades)
        wins = len([t for t in self.trades if t['pnl'] > 0])
        losses = len([t for t in self.trades if t['pnl'] <= 0])
        print(f"Wins: {wins} | Losses: {losses}")
        print(f"Total P&L: ${total_pnl:+.2f}")
        print(f"Log saved to: {self.log_file}")


def main():
    env_symbols = [s.strip() for s in os.getenv('TRADING_SYMBOLS', 'MNQ').split(',') if s.strip()]
    trading_mode = os.getenv('TRADING_MODE', 'paper').lower()
    default_symbols = env_symbols or ['MNQ']

    parser = argparse.ArgumentParser(description='Multi-Timeframe Scalping Bot - Multi-Symbol')
    parser.add_argument('--symbols', type=str, nargs='+', default=None,
                        choices=['MES', 'MNQ', 'NQ', 'MGC'],
                        help='Symbols to trade (skips prompt if set)')
    parser.add_argument('--prompt', action='store_true',
                        help='Ask MNQ / NQ / both at startup')
    parser.add_argument('--paper', action='store_true', help='Paper trading mode')
    parser.add_argument('--live', action='store_true',
                        help='Live trading mode (overrides TRADING_MODE and --paper)')
    parser.add_argument('--confirm', action='store_true',
                        help='Require typing YES before live trading starts')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip live confirmation (default when --live)')
    parser.add_argument(
        '--duration', type=int, default=0,
        help='Duration in minutes (0 = run until stopped)',
    )
    parser.add_argument('--test', action='store_true', help='Place test order and cancel immediately')
    args = parser.parse_args()

    if args.symbols:
        symbols = args.symbols
    elif args.prompt:
        symbols = prompt_symbol_choice()
    else:
        symbols = default_symbols

    # Never trade MNQ and NQ simultaneously — max 1 position handles this; keep scan order stable
    if NASDQ_SYMBOLS.issuperset(symbols) and len(symbols) > 1:
        symbols = [s for s in ('MNQ', 'NQ') if s in symbols]

    for sym in symbols:
        print(f"  Contract sizing: {symbol_risk_line(sym)}")

    if args.live:
        paper_mode = False
        if args.paper:
            print("⚠️  Both --live and --paper passed — --live wins (real orders)")
    else:
        paper_mode = args.paper or trading_mode in ('paper', 'backtest')

    mode_label = "PAPER" if paper_mode else "LIVE"
    print(f"\n>>> Trading mode: {mode_label} <<<")
    if paper_mode and not args.paper:
        print(f"    (TRADING_MODE={trading_mode!r} in .env — pass --live to override)\n")
    elif paper_mode:
        print(f"    (--paper flag set)\n")
    else:
        print(f"    (--live flag set)\n")
    
    skip_confirm = args.yes or (not paper_mode and not args.confirm)

    trader = LiveMTFScalper(
        symbols=symbols,
        paper_mode=paper_mode,
        skip_confirm=skip_confirm,
    )
    
    if args.test:
        trader.test_order()
    else:
        trader.run(duration_minutes=args.duration)


if __name__ == '__main__':
    main()
