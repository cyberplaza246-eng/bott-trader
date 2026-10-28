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
from src.core.ensemble_trader import EnsembleTrader
from src.ai.technical_analyzer import TechnicalAnalyzer
from src.utils.logger import bot_logger, trades_logger
from config.strategy_config import SCALPING_PAIRS


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

# Dollar profit locks (MNQ $2/pt: +$100 uPnL = 50 pts → SL locks +$15 = 7.5 pts)
PROFIT_LOCK_ENABLED = os.getenv('PROFIT_LOCK_ENABLED', 'true').strip().lower() in (
    '1', 'true', 'yes', 'on',
)
PROFIT_LOCK_TRIGGER_1_USD = float(os.getenv('PROFIT_LOCK_TRIGGER_1_USD', '100'))
PROFIT_LOCK_LEVEL_1_USD = float(os.getenv('PROFIT_LOCK_LEVEL_1_USD', '15'))
PROFIT_LOCK_TRIGGER_2_USD = float(os.getenv('PROFIT_LOCK_TRIGGER_2_USD', '150'))
PROFIT_LOCK_LEVEL_2_USD = float(os.getenv('PROFIT_LOCK_LEVEL_2_USD', '50'))

# ---------------------------------------------------------------------------
# REVAMP exits (overnight research winner). Default OFF = original live path
# (AI ensemble + TRAIL_ATR_MULT=0.6). Enable only via start_mnq_live_revamp.bat
# or env REVAMP_EXITS=1. Does not change start_mnq_live.bat / ema15_eod.
#   trail ATR mult 4.0 | breakout fixed 2R (no trail) | MR target bb_far
# ---------------------------------------------------------------------------
from src.research_ensemble import revamp_exits as _revamp_exits
from src.research_ensemble.live_signals import classify_last_sleeve
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
_HARD_MAX_CONCURRENT = 2
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
    'max_trades_per_day': 20,    # Conservative daily cap
    'cooldown_bars': 5,          # Bars between trades
    'max_positions': ENV_MAX_CONCURRENT_TRADES,
}

# Symbol specs
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
    profit_lock_level: int = 0  # 0=none, 1=+$15 after +$100, 2=+$50 after +$150
    ensemble_signal: str = 'UNKNOWN'
    model_signals: Optional[Dict] = None
    regime: str = 'unknown'
    exit_sleeve: str = ''  # trend | breakout | mean_reversion (REVAMP path)


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
        self._consec_loss_pause_until: Dict[str, datetime] = {}
        self._last_orphan_sweep = 0.0
        self._unconfirmed_exit_logged: set = set()
        self._ghost_flat_kept_logged: set = set()
        
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
        """Block known low-edge sessions (Friday, bad UTC hours)."""
        now = datetime.now(timezone.utc)
        pair_cfg = SCALPING_PAIRS.get(symbol, {})
        skip_friday = pair_cfg.get('skip_friday', self.skip_friday)
        bad_hours = pair_cfg.get('bad_hours_utc', self.bad_hours.get(symbol, []))

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
            pos.sl = new_sl
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
        if not ok and hasattr(self.broker, 'modify_position'):
            ok = bool(self.broker.modify_position(ticket=pos.order_id, sl=new_sl, tp=pos.tp))
        if ok:
            pos.sl = new_sl
            # Do NOT clear trailing_enabled here: protective STOP/STOP-LIMIT legs
            # report supports_stop_modify=False but remain trailable via cancel/replace.
            return True
        return False

    def update_dollar_profit_locks(self, symbol: str, df: pd.DataFrame):
        """+$100 uPnL → lock +$15; +$150 → ratchet to +$50 lock (env-configurable)."""
        if not PROFIT_LOCK_ENABLED or not self.positions or df is None or len(df) == 0:
            return
        row = df.iloc[-1]
        bar_high = float(row['high'])
        bar_low = float(row['low'])
        close = float(row.get('close', 0) or 0)

        for order_id, pos in list(self.positions.items()):
            if pos.symbol != symbol:
                continue
            if pos.direction == 'long':
                fav = max(bar_high, close) if close > 0 else bar_high
            else:
                fav = min(bar_low, close) if close > 0 else bar_low
            if fav <= 0:
                continue
            upnl = self._unrealized_pnl_usd(pos, fav)
            if upnl >= PROFIT_LOCK_TRIGGER_2_USD:
                target_level, lock_usd = 2, PROFIT_LOCK_LEVEL_2_USD
            elif upnl >= PROFIT_LOCK_TRIGGER_1_USD:
                target_level, lock_usd = 1, PROFIT_LOCK_LEVEL_1_USD
            else:
                continue
            new_sl = self._lock_sl_price_for_usd(pos, lock_usd)
            if int(getattr(pos, 'profit_lock_level', 0) or 0) >= target_level:
                if not self._is_tighter_sl(pos, new_sl, pos.sl):
                    continue
            if not self._is_tighter_sl(pos, new_sl, pos.sl):
                pos.profit_lock_level = max(int(getattr(pos, 'profit_lock_level', 0) or 0), target_level)
                continue
            old_sl = pos.sl
            if not self._apply_stop_update(pos, new_sl):
                continue
            pos.profit_lock_level = target_level
            print(
                f"   🔐 {symbol} L{target_level} lock ${lock_usd:.0f}: "
                f"uPnL~${upnl:.0f} SL {old_sl:.2f} → {pos.sl:.2f}"
            )
            bot_logger.info(
                f"Profit lock L{target_level} ${lock_usd:.0f} {symbol} {order_id}: "
                f"uPnL~${upnl:.2f} SL {old_sl:.2f} -> {pos.sl:.2f}"
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
            if pos.symbol != symbol:
                continue
            if not pos.trailing_enabled:
                continue

            tp_dist = abs(pos.tp - pos.entry_price) if pos.tp and pos.tp > 0 else 0.0
            risk_dist = max(tick, abs(pos.entry_price - pos.sl))
            # REVAMP trend sleeve: no fixed TP — activate trail on small R move (research-like).
            if tp_dist <= 0:
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
            if df is None or len(df) < min_bars:
                rith_bars = 0 if df is None else len(df)
                local_df = self._get_local_history_candles(symbol, count)
                merged = self._merge_candles(local_df, df, count)
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
            fresh = filter_warm_history(df[keep_cols], max_age_hours=6)
            if fresh is None or len(fresh) == 0:
                warned = getattr(self, "_stale_warm_warned", None)
                if warned is None:
                    self._stale_warm_warned = set()
                    warned = self._stale_warm_warned
                if symbol not in warned:
                    warned.add(symbol)
                    bot_logger.warning(
                        f"Local warm history for {symbol} is older than 6h — not trading off it"
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
        """Merge local history with live bars and keep latest rows."""
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
            self._consec_loss_pause_until.clear()
            if self.ensemble:
                self.ensemble.learner.decay_loss_patterns()
            print(f"\n📅 New trading day: {today}")
        
        if self.daily_pnl <= -self.daily_loss_limit:
            print(f"🛑 Daily loss limit hit: ${self.daily_pnl:.2f}")
            return False
        if self.daily_trades >= self.max_trades_per_day:
            print(f"🛑 Max trades hit: {self.daily_trades}/{self.max_trades_per_day}")
            return False
        return True
    
    def check_entry_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Check for entry signal using full AI Ensemble (or REVAMP research ensemble)."""
        if self._effective_open_count() >= self.max_positions or self.cooldown > 0:
            return None

        if self._symbol_in_loss_cooldown(symbol) or self._symbol_in_consec_loss_pause(symbol):
            return None

        if not self._is_trading_session_allowed(symbol):
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
            trailing = bool(hit.get("trailing", True))
            if sleeve in ("breakout", "mean_reversion"):
                trailing = False
            elif sleeve == "trend":
                trailing = True

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
                "confidence": 0.80,
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
            'daily_loss_limit': self.daily_loss_limit,
            'daily_trades': self.daily_trades,
            'max_trades': self.max_trades_per_day,
            'open_count': open_count,
            'max_positions': self.max_positions,
            'cooldown': self.cooldown,
            'loss_cooldown_min': _mins_left(self.loss_cooldown_until.get(symbol)),
            'consec_pause_min': _mins_left(self._consec_loss_pause_until.get(symbol)),
            'consec_losses': self._symbol_consec_losses.get(symbol, 0),
            'session_block': self._revamp_session_block_reason(symbol),
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
            'model_signals': pos.model_signals or {},
            'regime': pos.regime or getattr(self.ensemble.learner, 'current_regime', 'unknown'),
        })

    def place_order(self, signal: Dict) -> bool:
        """Place order with bracket (SL+TP)."""
        symbol = signal['symbol']
        tick_size = self._spec(symbol)['tick_size']
        direction = signal['direction']
        entry = signal['entry']
        sl = signal['sl']
        tp = signal['tp']
        
        if self.paper_mode or self.paper_orders:
            order_id = f"paper_{int(time.time())}"
            tag = "PAPER-ORDERS" if self.paper_orders else "PAPER"
            side = "LONG" if direction == 'long' else "SHORT"
            print(f"📝 [{tag}] {side} {symbol} @ {entry:.2f}")
            print(f"   SL: {sl:.2f} | TP: {tp:.2f}")
            
            trail_on = bool(signal.get('trailing_enabled', True))
            pos = Position(
                order_id=order_id,
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                size=self.contracts,
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
            )
            self.positions[order_id] = pos
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
            size=self.contracts,
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
                size=self.contracts,
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
            )
            self.positions[order_id] = pos
            self._write_board_status(note=f"{len(self.positions)}/{self.max_positions} after fill")
            print(f"ORDER FILLED: {direction.upper()} {symbol}")
            print(f"   Entry: {pos.entry_price:.2f}")
            print(f"   SL: {sl:.2f} | TP: {tp:.2f}")
            print(f"   Active positions: {len(self.positions)}/{self.max_positions}")
            if REVAMP_EXITS_ON:
                print(f"   REVAMP sleeve={pos.exit_sleeve or '?'} trail={'ON' if pos.trailing_enabled else 'OFF (fixed TP)'}")
            if want_trail and not stop_updatable:
                print(f"   WARNING: Broker stop missing on entry - trailing disabled for {order_id}")
                bot_logger.error(
                    f"Order {order_id} for {symbol} filled without broker stop support; trailing is disabled"
                )
            trades_logger.info(f"ENTRY {direction} {symbol} @ {entry:.2f} SL={sl:.2f} TP={tp:.2f}")
            return True
        else:
            print(f"❌ Order failed!")
            return False
    
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
                elif pos.tp and pos.tp > 0 and row['high'] >= pos.tp:
                    exits.append((order_id, 'TAKE_PROFIT'))
            else:
                if row['high'] >= pos.sl:
                    exits.append((order_id, 'STOP_LOSS'))
                elif pos.tp and pos.tp > 0 and row['low'] <= pos.tp:
                    exits.append((order_id, 'TAKE_PROFIT'))
        
        return exits
    
    def process_exit(self, order_id: str, exit_type: str, exit_price: float):
        """Process position exit."""
        if order_id not in self.positions:
            return
        
        pos = self.positions[order_id]
        spec = self._spec(pos.symbol)
        d = pos.direction
        if d == 'long':
            raw_pnl = (exit_price - pos.entry_price) * spec['point_value'] * pos.size
        else:
            raw_pnl = (pos.entry_price - exit_price) * spec['point_value'] * pos.size
        
        # Subtract costs ($2 commission + 1 tick slippage)
        commission = 2.00
        slippage = spec['tick_size'] * spec['point_value']
        pnl = raw_pnl - commission - slippage
        
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
        
        self.daily_pnl += pnl
        self.daily_trades += 1
        
        emoji = "🟢" if pnl > 0 else "🔴"
        print(f"{emoji} EXIT {exit_type}: {d.upper()} @ {exit_price:.2f}")
        print(f"   PnL: ${pnl:.2f} | Daily: ${self.daily_pnl:.2f}")
        trades_logger.info(f"EXIT {exit_type} {d} {pos.symbol} @ {exit_price:.2f} PnL=${pnl:.2f}")

        self._feed_learner_on_exit(pos, pnl, exit_type, exit_price)
        self._update_loss_protection(pos.symbol, pnl)
        
        del self.positions[order_id]
        self._cancel_orders_for_closed_position(order_id, pos.symbol, f"exit_{exit_type.lower()}")
        print(f"   Remaining positions: {len(self.positions)}/{self.max_positions}")
        self.cooldown = self.cooldown_bars

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
            candidates.append(net_total)
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
        del self.positions[order_id]
        print(f"🔄 Position {order_id} removed ({reason})")
        self._cancel_orders_for_closed_position(order_id, pos.symbol, 'broker_sync_flat')

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
            print(f"⚠️ Adopted broker position into tracking: {ticket} {direction} {symbol}")
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
        print(f"Contracts:  {self.contracts}")
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
            print(f"Trail ATR:  {TRAIL_SETTINGS['atr_mult']} (live default without revamp bat: 0.6)")
            _rv_mode = {1: 'compact (REVAMP_VERBOSE=2 detailed, 0 off)', 2: 'detailed (REVAMP_VERBOSE=1 compact, 0 off)'}
            print(f"Readout:    {('ON ' + _rv_mode.get(REVAMP_VERBOSE_LEVEL, '')) if REVAMP_VERBOSE else 'OFF (set REVAMP_VERBOSE=1)'}")
        else:
            print(f"Strategy:   Sweep-Gate + ML (IntelligentTrader, AdvancedStrategies)")
            print(f"Features:   Dynamic SL/TP, Trailing Stops, 8 Confirmation Models")
            print(f"Trail ATR:  {TRAIL_SETTINGS['atr_mult']}")
        if PROFIT_LOCK_ENABLED:
            print(
                f"ProfitLock: +${PROFIT_LOCK_TRIGGER_1_USD:.0f}→+${PROFIT_LOCK_LEVEL_1_USD:.0f} SL; "
                f"+${PROFIT_LOCK_TRIGGER_2_USD:.0f}→+${PROFIT_LOCK_LEVEL_2_USD:.0f} SL"
            )
        print(f"Risk:       Max ${self.daily_loss_limit} daily loss | Max concurrent {self.max_positions}")
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

                    if 'datetime' in df.columns:
                        current_bar = df.iloc[-1]['datetime']
                    elif 'time' in df.columns:
                        current_bar = df.iloc[-1]['time']
                    else:
                        current_bar = datetime.now()

                    if last_bar_time.get(symbol) == current_bar:
                        continue
                    last_bar_time[symbol] = current_bar
                    any_new_bar = True

                    if self.positions:
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
                        self.update_trailing_stops(symbol, df)

                    _rv_open_n = self._effective_open_count()
                    _rv_checked = False
                    _rv_signal = None
                    if _rv_open_n < self.max_positions and self.cooldown == 0:
                        _rv_checked = True
                        signal = self.check_entry_signal(symbol, df)
                        _rv_signal = signal
                        if signal:
                            cycle_candidates.append(signal)

                    row = df.iloc[-1]
                    pos_syms = [p.symbol for p in self.positions.values()]
                    pos_str = f"{len(self.positions)}/{self.max_positions} {pos_syms}" if self.positions else "None"
                    cooldown_str = f"(CD:{self.cooldown})" if self.cooldown > 0 else ""
                    print(f"[{current_bar}] {symbol} {row['close']:.2f} | Pos: {pos_str} {cooldown_str} | Daily: ${self.daily_pnl:.2f}")
                    if REVAMP_EXITS_ON and REVAMP_VERBOSE:
                        self._print_revamp_reasoning(symbol, df, _rv_open_n, _rv_checked, _rv_signal)

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
                    if self.place_order(best):
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
