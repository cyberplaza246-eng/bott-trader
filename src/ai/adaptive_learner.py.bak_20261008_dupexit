"""
Adaptive Learning System — God Tier v2

Learns from past trades to improve future decisions:
  - Tracks which model combinations produce winning trades PER PAIR and PER SESSION
  - Bayesian weight updates with exponential recency weighting
  - Market regime detection (trending / ranging / volatile)
  - Drawdown protection: auto-tighten after losing streaks, widen after wins
  - Session-aware performance tracking (avoid bad hours)
  - Model correlation tracking: which model combos win together
  - Rolling performance windows with decay
"""
import os
import json
import math
import threading
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from src.utils.logger import bot_logger

LEARNING_DB_PATH = 'data/adaptive_learning.json'


class AdaptiveLearner:
    """
    God-tier adaptive learning system.

    Key features:
      - Bayesian model weight optimization with recency decay
      - Per-pair, per-session, per-regime tracking
      - Market regime detection (trending/ranging/volatile)
      - Drawdown circuit breaker
      - Win-streak momentum scaling
      - Model combo synergy tracking
      - Rolling Sharpe ratio per model
    """

    REGIME_TRENDING = 'trending'
    REGIME_RANGING = 'ranging'
    REGIME_VOLATILE = 'volatile'

    _save_lock = threading.Lock()  # Prevent concurrent file writes from multiple threads

    def _get_base_confidence_threshold(self) -> float:
        from config.strategy_config import ENSEMBLE_CONFIDENCE_THRESHOLD

        try:
            if os.path.exists(self.RISK_OVERRIDES_PATH):
                with open(self.RISK_OVERRIDES_PATH, 'r') as f:
                    config = json.load(f)
                override = config.get('entry_confidence_threshold')
                if override is not None:
                    return float(override)
        except Exception:
            pass

        return ENSEMBLE_CONFIDENCE_THRESHOLD

    def _has_explicit_threshold_override(self) -> bool:
        try:
            if os.path.exists(self.RISK_OVERRIDES_PATH):
                with open(self.RISK_OVERRIDES_PATH, 'r') as f:
                    config = json.load(f)
                return config.get('entry_confidence_threshold') is not None
        except Exception:
            return False

        return False

    def __init__(self, initial_weights: dict = None):
        self.model_weights = initial_weights or {
            'scalping': 0.28,
            'technical': 0.18,
            'volume': 0.14,
            'ema_crossover': 0.12,
            'candlestick': 0.10,
            'multi_tf': 0.08,
            'support_resistance': 0.04,
            'lstm': 0.03,
            'sentiment': 0.03,
        }

        self.trade_history = []
        self.model_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.pair_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0})
        self.session_stats = defaultdict(lambda: {'wins': 0, 'losses': 0})

        # Per-pair per-model accuracy
        self.pair_model_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})

        # Per-session per-pair stats
        self.session_pair_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0})

        # Model combo synergy
        self.model_combos = defaultdict(lambda: {'wins': 0, 'total': 0})

        # Rolling model PnL (for Sharpe-like scoring)
        self.model_pnl_history = defaultdict(list)

        # Regime tracking
        self.current_regime = self.REGIME_RANGING
        self.regime_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0})

        # Hourly performance (24 buckets)
        self.hourly_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0})

        self.RISK_OVERRIDES_PATH = 'data/risk_overrides.json'
        self.confidence_threshold = self._get_base_confidence_threshold()

        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.consecutive_wins = 0

        # Drawdown tracking
        self.peak_balance = 0.0
        self.current_drawdown_pct = 0.0
        self.in_drawdown_protection = False

        # Loss pattern tracking — learn WHY trades fail
        self.loss_patterns = defaultdict(lambda: {'count': 0, 'total_loss': 0.0})
        # Recent trade tracker for cooldown logic
        self.recent_trades_window = []  # last N trades for fast reaction

        # ── ATR-Centric Adaptive Features ────────────────────────────
        # 1. Session-pair ATR profiling: avg ATR per session per pair
        self.session_atr_profiles = defaultdict(lambda: {'atr_values': [], 'avg_atr': 0.0})

        # 2. Spread-cost tracking: actual spreads seen per pair
        self.spread_tracking = defaultdict(lambda: {'spreads': [], 'avg_spread': 0.0})

        # 3. SL effectiveness auto-tuning
        self.recent_sl_values = defaultdict(list)   # per-pair list of SL distances
        self.sl_multiplier_by_pair = {}             # pair → ATR multiplier (0.6 to 1.2)

        # 4. Time-exit optimization
        self.time_exit_candles = 8   # default: exit after 8 candles
        self.time_exit_outcomes = []  # list of {candles_held, is_win, pnl}

        # 5. ATR regime state (expanding / contracting / neutral)
        self.atr_regime = 'neutral'

        # 6. Exit type tracking for auto-learning (SL_HIT, TP_HIT, TIME_EXIT, MANUAL)
        self.exit_type_tracking = defaultdict(lambda: {'sl_hit': 0, 'tp_hit': 0, 'time_exit': 0, 'manual': 0, 'total': 0})
        self.auto_learning_trade_count = 0  # Trades since last auto-learning update

        self.decay_factor = 0.95

        self._load()

    @staticmethod
    def _normalize_pair(pair: str) -> str:
        raw = str(pair or 'UNKNOWN').upper().strip().replace(' ', '')
        if raw == 'UNKNOWN':
            return raw
        compact = raw.replace('/', '')
        if len(compact) == 6 and compact.isalpha():
            return f"{compact[:3]}/{compact[3:]}"
        return raw

    @staticmethod
    def _get_session(hour: int = None) -> str:
        if hour is None:
            hour = datetime.utcnow().hour
        if 0 <= hour < 8:
            return 'asian'
        elif 8 <= hour < 13:
            return 'london'
        elif 13 <= hour < 17:
            return 'ny_overlap'
        elif 17 <= hour < 22:
            return 'new_york'
        else:
            return 'off_hours'

    # ------------------------------------------------------------------
    # Market Regime Detection
    # ------------------------------------------------------------------
    def detect_regime(self, df) -> str:
        """Detect market regime using ATR-centric logic.

        Enhanced: uses ATR trend (expanding/contracting) + ADX for classification.
        Also updates self.atr_regime for other components to query.
        """
        if df is None or len(df) < 20:
            return self.REGIME_RANGING
        try:
            latest = df.iloc[-1]
            adx = float(latest.get('adx', 0) or 0)
            atr = float(latest.get('atr', 0) or 0)
            close = float(latest['close'])
            atr_pct = (atr / close * 100) if close > 0 else 0

            # ATR trend: check consecutive rises/falls
            atr_rise_streak = int(latest.get('atr_rise_streak', 0) or 0)
            atr_fall_streak = int(latest.get('atr_fall_streak', 0) or 0)

            # ATR regime classification
            if atr_rise_streak >= 5:
                self.atr_regime = 'expanding'
            elif atr_fall_streak >= 5:
                self.atr_regime = 'contracting'
            else:
                self.atr_regime = 'neutral'

            ema200 = float(latest.get('ema_200', close) or close)
            trend_deviation = abs(close - ema200) / close * 100 if close > 0 else 0

            if adx > 22 and trend_deviation > 0.15:
                regime = self.REGIME_TRENDING
            elif atr_pct > 0.12 or self.atr_regime == 'expanding':
                regime = self.REGIME_VOLATILE
            else:
                regime = self.REGIME_RANGING

            self.current_regime = regime
            return regime
        except Exception:
            return self.REGIME_RANGING

    # ------------------------------------------------------------------
    # Trade Recording
    # ------------------------------------------------------------------
    def record_trade(self, trade_result: dict):
        pnl = trade_result.get('profit_loss', 0)
        is_win = pnl > 0
        is_breakeven = (pnl == 0)  # $0.00 = breakeven, not a loss
        pair = self._normalize_pair(trade_result.get('pair', 'UNKNOWN'))
        signal = trade_result.get('signal', 'UNKNOWN')
        model_signals = trade_result.get('model_signals', {})
        regime = trade_result.get('regime', self.current_regime)
        exit_type = trade_result.get('exit_type', 'UNKNOWN')  # SL_HIT, TP_HIT, TIME_EXIT, MANUAL

        hour = datetime.utcnow().hour
        session = self._get_session(hour)

        # Track exit type for auto-learning
        self._track_exit_type(pair, exit_type)

        # Core stats — breakeven trades are neutral (don't count as win or loss)
        if is_win:
            self.pair_stats[pair]['wins'] += 1
            self.consecutive_losses = 0
            self.consecutive_wins += 1
        elif is_breakeven:
            # Breakeven: reset streaks — the position didn't lose money
            self.consecutive_losses = max(0, self.consecutive_losses - 1)
            self.consecutive_wins = 0
        else:
            self.pair_stats[pair]['losses'] += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)

        self.pair_stats[pair]['total_pnl'] += pnl

        # Session stats (skip breakeven)
        if not is_breakeven:
            if is_win:
                self.session_stats[session]['wins'] += 1
            else:
                self.session_stats[session]['losses'] += 1

        # Session + Pair stats (skip breakeven)
        sp_key = f"{session}:{pair}"
        if not is_breakeven:
            if is_win:
                self.session_pair_stats[sp_key]['wins'] += 1
            else:
                self.session_pair_stats[sp_key]['losses'] += 1
        self.session_pair_stats[sp_key]['total_pnl'] += pnl

        # Hourly stats (skip breakeven)
        h_key = str(hour)
        if not is_breakeven:
            if is_win:
                self.hourly_stats[h_key]['wins'] += 1
            else:
                self.hourly_stats[h_key]['losses'] += 1
        self.hourly_stats[h_key]['total_pnl'] += pnl

        # Regime stats (skip breakeven)
        if not is_breakeven:
            if is_win:
                self.regime_stats[regime]['wins'] += 1
            else:
                self.regime_stats[regime]['losses'] += 1
        self.regime_stats[regime]['total_pnl'] += pnl

        # Per-model accuracy
        agreeing_models = set()
        for model_name, model_signal in model_signals.items():
            if model_name not in self.model_accuracy:
                self.model_accuracy[model_name] = {'correct': 0, 'total': 0}

            model_dir = model_signal.get('signal')
            model_agreed = model_dir == signal
            model_opposed = model_dir in ('BUY', 'SELL') and model_dir != signal

            # Fixed accuracy: only count directional agreement/disagreement
            # HOLD/SKIP = neutral (not counted either way)
            if model_dir in ('BUY', 'SELL'):
                self.model_accuracy[model_name]['total'] += 1
                if (model_agreed and is_win) or (model_opposed and not is_win):
                    self.model_accuracy[model_name]['correct'] += 1

            if model_agreed:
                agreeing_models.add(model_name)

            # Per-pair per-model accuracy (fixed: only directional signals)
            pm_key = f"{pair}:{model_name}"
            if model_dir in ('BUY', 'SELL'):
                self.pair_model_accuracy[pm_key]['total'] += 1
                if (model_agreed and is_win) or (model_opposed and not is_win):
                    self.pair_model_accuracy[pm_key]['correct'] += 1

            # Rolling PnL per model
            if model_agreed:
                self.model_pnl_history[model_name].append(pnl)
            if len(self.model_pnl_history[model_name]) > 100:
                self.model_pnl_history[model_name] = self.model_pnl_history[model_name][-100:]

        # Model combo synergy
        if len(agreeing_models) >= 2:
            combo_key = ":".join(sorted(agreeing_models))
            self.model_combos[combo_key]['total'] += 1
            if is_win:
                self.model_combos[combo_key]['wins'] += 1

        # Drawdown tracking
        cumulative_pnl = sum(t.get('profit_loss', 0) for t in self.trade_history) + pnl
        if cumulative_pnl > self.peak_balance:
            self.peak_balance = cumulative_pnl
        if self.peak_balance > 0:
            self.current_drawdown_pct = (self.peak_balance - cumulative_pnl) / self.peak_balance * 100
        else:
            self.current_drawdown_pct = 0

        # Activate drawdown protection at 3% (was 5% — too late for small accounts)
        if self.current_drawdown_pct > 3:
            if not self.in_drawdown_protection:
                bot_logger.warning(f"🛡️ Drawdown protection ACTIVATED: {self.current_drawdown_pct:.1f}% drawdown")
            self.in_drawdown_protection = True
        elif self.current_drawdown_pct < 1.5:
            if self.in_drawdown_protection:
                bot_logger.info(f"✅ Drawdown protection DEACTIVATED: recovered to {self.current_drawdown_pct:.1f}%")
            self.in_drawdown_protection = False

        # Store trade
        trade_record = {
            **trade_result,
            'pair': pair,
            'timestamp': datetime.now().isoformat(),
            'session': session,
            'hour': hour,
            'regime': regime,
            'is_win': is_win,
        }
        for key in ('entry_time', 'exit_time'):
            if key in trade_record and hasattr(trade_record[key], 'isoformat'):
                trade_record[key] = trade_record[key].isoformat()
        self.trade_history.append(trade_record)

        # Update recent trades window (last 10 trades for fast reaction)
        self.recent_trades_window.append({
            'pair': pair, 'session': session, 'hour': hour,
            'regime': regime, 'is_win': is_win, 'pnl': pnl,
            'timestamp': datetime.now().isoformat(),
        })
        if len(self.recent_trades_window) > 10:
            self.recent_trades_window = self.recent_trades_window[-10:]

        # ── Loss pattern analysis ───────────────────────────────────
        if not is_win and not is_breakeven:
            self._record_loss_pattern(pair, session, hour, regime, pnl, model_signals, signal)

        # Adapt
        self._adapt_weights()
        self._adapt_confidence_threshold()
        
        # Auto-learn and persist risk parameters 
        self._auto_learn_risk_params()
        
        self._save()

        bot_logger.info(
            f"📚 Trade recorded: {pair} {signal} → {'WIN' if is_win else 'LOSS'} "
            f"(${pnl:+.2f}) | Regime: {regime} | Session: {session} | "
            f"Streak: {'W' if is_win else 'L'}{self.consecutive_wins if is_win else self.consecutive_losses}"
        )

    # ------------------------------------------------------------------
    # Weight Adaptation (Bayesian with Recency + Sharpe)
    # ------------------------------------------------------------------
    def _adapt_weights(self):
        min_trades = 5
        total_trades = sum(m['total'] for m in self.model_accuracy.values())
        if total_trades < min_trades:
            return

        scores = {}
        for model_name in self.model_weights:
            stats = self.model_accuracy.get(model_name, {'correct': 0, 'total': 0})
            if stats['total'] == 0:
                scores[model_name] = 0.5
                continue

            accuracy = stats['correct'] / stats['total']

            # Sharpe-like bonus
            pnl_history = self.model_pnl_history.get(model_name, [])
            if len(pnl_history) >= 5:
                pnl_arr = np.array(pnl_history[-30:])
                mean_pnl = np.mean(pnl_arr)
                std_pnl = np.std(pnl_arr) + 1e-6
                sharpe = mean_pnl / std_pnl
                sharpe_bonus = np.clip(sharpe * 0.05, -0.1, 0.2)
            else:
                sharpe_bonus = 0.0

            scores[model_name] = max(0.05, accuracy + sharpe_bonus)

        if not scores:
            return

        total_score = sum(scores.values())
        if total_score > 0:
            new_weights = {m: s / total_score for m, s in scores.items()}
            blend = min(0.15, 0.05 + (total_trades / 1000))  # Slower blend to prevent whipsawing

            for model in self.model_weights:
                if model in new_weights:
                    self.model_weights[model] = (
                        (1 - blend) * self.model_weights[model] +
                        blend * new_weights[model]
                    )

            # Minimum weight floor (3%)
            for model in self.model_weights:
                self.model_weights[model] = max(0.03, self.model_weights[model])

            weight_sum = sum(self.model_weights.values())
            self.model_weights = {k: v / weight_sum for k, v in self.model_weights.items()}

            bot_logger.info(
                f"📊 Adapted weights (blend={blend:.0%}): " +
                " | ".join(f"{k}: {v:.2f}" for k, v in
                           sorted(self.model_weights.items(), key=lambda x: -x[1]))
            )

    def _adapt_confidence_threshold(self):
        base = self._get_base_confidence_threshold()

        if self._has_explicit_threshold_override():
            self.confidence_threshold = base
            return

        ceiling = base + 0.10  # Max threshold = base + 10% (was +25%)

        # ── Recent-loss reaction (gentle) ────────────────────────────
        recent_5 = self.recent_trades_window[-5:] if len(self.recent_trades_window) >= 3 else []
        recent_loss_count = sum(1 for t in recent_5 if not t.get('is_win', True))
        recent_loss_ratio = recent_loss_count / len(recent_5) if recent_5 else 0

        if self.in_drawdown_protection:
            self.confidence_threshold = min(ceiling, base + 0.05)
            bot_logger.info(f"🛡️ Drawdown protection: threshold at {self.confidence_threshold:.2f}")
            return

        if self.consecutive_losses >= 5:
            bump = min(0.08, self.consecutive_losses * 0.01)
            self.confidence_threshold = min(ceiling, base + bump)
            bot_logger.info(f"⚠️  Threshold → {self.confidence_threshold:.2f} after {self.consecutive_losses} consecutive losses")
        elif self.consecutive_losses >= 3:
            bump = self.consecutive_losses * 0.01
            self.confidence_threshold = min(ceiling, base + bump)
            bot_logger.info(f"⚠️  Threshold → {self.confidence_threshold:.2f} after {self.consecutive_losses} consecutive losses")
        elif self.consecutive_wins >= 3:
            ease = min(0.06, self.consecutive_wins * 0.01)
            self.confidence_threshold = max(base, self.confidence_threshold - ease)
            bot_logger.info(f"✅ Threshold → {self.confidence_threshold:.2f} (winning streak of {self.consecutive_wins})")
        else:
            if self.confidence_threshold > base:
                self.confidence_threshold = max(base, self.confidence_threshold - 0.005)

    # ------------------------------------------------------------------
    # Loss Pattern Analysis
    # ------------------------------------------------------------------
    def _record_loss_pattern(self, pair, session, hour, regime, pnl, model_signals, signal):
        """Categorise and record loss patterns so the bot avoids repeating mistakes."""
        # Pattern keys: pair+session, pair+regime, pair+hour, model combos
        patterns = [
            f"pair:{pair}",
            f"session:{session}",
            f"pair_session:{pair}:{session}",
            f"pair_regime:{pair}:{regime}",
            f"hour:{hour}",
            f"signal:{signal}",
        ]
        # Which models agreed with the losing signal?
        for model_name, ms in model_signals.items():
            if ms.get('signal') == signal:
                patterns.append(f"model_loss:{model_name}")

        for pattern in patterns:
            self.loss_patterns[pattern]['count'] += 1
            self.loss_patterns[pattern]['total_loss'] += abs(pnl)

        # Log the top recurring loss patterns
        top_patterns = sorted(
            self.loss_patterns.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:5]
        if top_patterns and top_patterns[0][1]['count'] >= 3:
            bot_logger.warning(
                "🔍 Top loss patterns: " +
                " | ".join(f"{k}: {v['count']}x (${v['total_loss']:.2f})" for k, v in top_patterns)
            )

    def should_skip_loss_pattern(self, pair: str, session: str = None, regime: str = None) -> bool:
        """Check if current conditions match a known heavy-loss pattern."""
        pair = self._normalize_pair(pair)
        if session is None:
            session = self._get_session()
        if regime is None:
            regime = self.current_regime

        # Check pair+session pattern (with decay — counts < 5 after daily decay)
        ps_key = f"pair_session:{pair}:{session}"
        ps_stats = self.loss_patterns.get(ps_key, {'count': 0})
        if ps_stats['count'] >= 7:  # Raised from 5 to 7 (accounts for decay)
            bot_logger.info(f"🚫 Loss pattern block: {pair} in {session} has {ps_stats['count']} losses")
            return True

        # Check pair+regime pattern (with decay)
        pr_key = f"pair_regime:{pair}:{regime}"
        pr_stats = self.loss_patterns.get(pr_key, {'count': 0})
        if pr_stats['count'] >= 7:  # Raised from 5 to 7
            bot_logger.info(f"🚫 Loss pattern block: {pair} in {regime} regime has {pr_stats['count']} losses")
            return True

        return False

    def decay_loss_patterns(self, factor: float = 0.85):
        """Apply daily decay to loss pattern counts so old losses fade.
        Call once per day (e.g. from bot.py daily_reset or save cycle).
        """
        keys_to_remove = []
        for key in list(self.loss_patterns.keys()):
            self.loss_patterns[key]['count'] = int(self.loss_patterns[key]['count'] * factor)
            self.loss_patterns[key]['total_loss'] *= factor
            if self.loss_patterns[key]['count'] <= 0:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.loss_patterns[key]
        if keys_to_remove:
            bot_logger.info(f"♻️ Loss pattern decay: removed {len(keys_to_remove)} stale patterns")

    def get_recent_win_rate(self, n: int = 10) -> float:
        """Win rate over last n real trades (excludes breakeven/pnl=0)."""
        recent = self.recent_trades_window[-n:]
        if not recent:
            return 0.5
        # Only count trades with actual P/L (skip breakeven $0 trades)
        real_trades = [t for t in recent if t.get('pnl', 0) != 0]
        if not real_trades:
            return 0.5  # No real data yet — assume neutral
        wins = sum(1 for t in real_trades if t.get('is_win', False))
        return wins / len(real_trades)

    # ------------------------------------------------------------------
    # ATR-Centric Adaptive Methods
    # ------------------------------------------------------------------

    def record_session_atr(self, pair: str, session: str, atr_value: float):
        """Track ATR values per session per pair for profiling.

        Called each cycle to build a rolling profile of typical
        ATR levels during each session for each pair.
        """
        pair = self._normalize_pair(pair)
        key = f"{session}:{pair}"
        profile = self.session_atr_profiles[key]
        profile['atr_values'].append(atr_value)
        # Keep last 200 readings
        if len(profile['atr_values']) > 200:
            profile['atr_values'] = profile['atr_values'][-200:]
        if profile['atr_values']:
            profile['avg_atr'] = sum(profile['atr_values']) / len(profile['atr_values'])

    def get_session_atr_threshold(self, pair: str, session: str = None) -> float:
        """Get learned ATR threshold for this pair in this session.

        Returns 0 if insufficient data (use static config default).
        """
        pair = self._normalize_pair(pair)
        if session is None:
            session = self._get_session()
        key = f"{session}:{pair}"
        profile = self.session_atr_profiles.get(key, {})
        if len(profile.get('atr_values', [])) < 20:
            return 0.0  # Not enough data
        # Threshold = 50th percentile of session ATR values
        return float(np.percentile(profile['atr_values'], 50))

    def record_spread(self, pair: str, spread: float):
        """Track actual spreads seen for cost analysis."""
        pair = self._normalize_pair(pair)
        tracker = self.spread_tracking[pair]
        tracker['spreads'].append(spread)
        if len(tracker['spreads']) > 200:
            tracker['spreads'] = tracker['spreads'][-200:]
        if tracker['spreads']:
            tracker['avg_spread'] = sum(tracker['spreads']) / len(tracker['spreads'])

    def get_avg_spread(self, pair: str) -> float:
        """Get average observed spread for a pair."""
        pair = self._normalize_pair(pair)
        tracker = self.spread_tracking.get(pair, {})
        return tracker.get('avg_spread', 0.0)

    def record_sl_outcome(self, pair: str, sl_distance: float, is_win: bool):
        """Record SL distance and outcome for auto-tuning.

        Over time, this learns whether 0.8×ATR is too tight or too wide.
        """
        pair = self._normalize_pair(pair)
        self.recent_sl_values[pair].append(sl_distance)
        if len(self.recent_sl_values[pair]) > 100:
            self.recent_sl_values[pair] = self.recent_sl_values[pair][-100:]

        # Track win/loss per SL bucket for multiplier tuning
        # This is a simple approach: adjust multiplier toward winning SL distances
        if len(self.recent_sl_values[pair]) >= 20:
            # Get recent trade outcomes paired with SL values
            # Use a simple heuristic: if win rate low and SL tight, widen; if high and SL wide, tighten
            recent_trades = [t for t in self.recent_trades_window[-20:] if self._normalize_pair(t.get('pair', '')) == pair]
            if len(recent_trades) >= 10:
                wins = sum(1 for t in recent_trades if t.get('is_win', False))
                wr = wins / len(recent_trades)
                current_mult = self.sl_multiplier_by_pair.get(pair, 0.8)

                if wr < 0.35:
                    # Too many losses — widen SL slightly
                    new_mult = min(1.2, current_mult + 0.05)
                    self.sl_multiplier_by_pair[pair] = new_mult
                    bot_logger.info(
                        f"📐 SL auto-tune {pair}: widened to {new_mult:.2f}×ATR "
                        f"(win rate {wr:.0%} too low)"
                    )
                elif wr > 0.55:
                    # Winning well — tighten SL slightly for better R/R
                    new_mult = max(0.6, current_mult - 0.05)
                    self.sl_multiplier_by_pair[pair] = new_mult
                    bot_logger.info(
                        f"📐 SL auto-tune {pair}: tightened to {new_mult:.2f}×ATR "
                        f"(win rate {wr:.0%} healthy)"
                    )

    def get_sl_multiplier(self, pair: str) -> float:
        """Get the auto-tuned SL multiplier for a pair (0.6 to 1.2×ATR)."""
        pair = self._normalize_pair(pair)
        return self.sl_multiplier_by_pair.get(pair, 0.8)

    def get_recent_sl_values(self, pair: str) -> list:
        """Get recent SL distance values for median check."""
        pair = self._normalize_pair(pair)
        return list(self.recent_sl_values.get(pair, []))

    # ------------------------------------------------------------------
    # Exit Type Tracking & Auto-Learning
    # ------------------------------------------------------------------
    def _track_exit_type(self, pair: str, exit_type: str):
        """Track exit type (SL_HIT, TP_HIT, TIME_EXIT, MANUAL) per pair."""
        pair = self._normalize_pair(pair)
        exit_type = exit_type.upper() if exit_type else 'UNKNOWN'
        
        self.exit_type_tracking[pair]['total'] += 1
        
        if 'STOP' in exit_type or 'SL' in exit_type:
            self.exit_type_tracking[pair]['sl_hit'] += 1
        elif 'TAKE' in exit_type or 'TP' in exit_type or 'PROFIT' in exit_type:
            self.exit_type_tracking[pair]['tp_hit'] += 1
        elif 'TIME' in exit_type:
            self.exit_type_tracking[pair]['time_exit'] += 1
        else:
            self.exit_type_tracking[pair]['manual'] += 1
        
        self.auto_learning_trade_count += 1

    def _auto_learn_risk_params(self):
        """Auto-adjust risk parameters based on exit type patterns.
        
        Triggered every N trades (configurable in risk_overrides.json).
        - If SL hit rate > 60%: widen SL multiplier
        - If TP hit rate > 70%: consider widening TP for more profit
        """
        import json
        import os
        
        # Load auto-learning config from risk_overrides
        try:
            if os.path.exists(self.RISK_OVERRIDES_PATH):
                with open(self.RISK_OVERRIDES_PATH, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
        except Exception:
            config = {}
        
        auto_learn = config.get('auto_learning', {})
        if not auto_learn.get('enabled', False):
            return
        
        interval = auto_learn.get('adjustment_interval_trades', 30)
        sl_threshold = auto_learn.get('sl_hit_rate_threshold', 0.60)
        tp_threshold = auto_learn.get('tp_hit_rate_threshold', 0.70)
        max_sl_adj = auto_learn.get('max_sl_adjustment', 0.3)
        max_tp_adj = auto_learn.get('max_tp_adjustment', 0.3)
        
        if self.auto_learning_trade_count < interval:
            return
        
        # Reset counter
        self.auto_learning_trade_count = 0
        
        changes_made = []
        
        # Analyze each pair's exit type distribution
        for pair, stats in self.exit_type_tracking.items():
            total = stats.get('total', 0)
            if total < 10:
                continue
            
            sl_hits = stats.get('sl_hit', 0)
            tp_hits = stats.get('tp_hit', 0)
            
            sl_rate = sl_hits / total
            tp_rate = tp_hits / total
            
            # Get current pair multiplier
            pair_mult = config.get('sl_multiplier_by_pair', {}).get(pair, 1.0)
            
            # SL hit rate too high -> widen SL
            if sl_rate > sl_threshold:
                new_mult = min(pair_mult + 0.1, pair_mult + max_sl_adj, 2.0)
                if 'sl_multiplier_by_pair' not in config:
                    config['sl_multiplier_by_pair'] = {}
                config['sl_multiplier_by_pair'][pair] = round(new_mult, 2)
                changes_made.append(f"SL_{pair}={new_mult:.2f} (SL hit rate {sl_rate:.0%})")
                bot_logger.warning(
                    f"📐 AUTO-LEARN: {pair} SL hit rate {sl_rate:.0%} > {sl_threshold:.0%} — "
                    f"widening SL multiplier to {new_mult:.2f}×ATR"
                )
            
            # TP hit rate very high -> consider widening TP for more profit
            if tp_rate > tp_threshold:
                current_tp_ratio = config.get('tp_base_ratio', 1.2)
                new_tp_ratio = min(current_tp_ratio + 0.05, current_tp_ratio + max_tp_adj, 2.0)
                config['tp_base_ratio'] = round(new_tp_ratio, 2)
                changes_made.append(f"TP_BASE={new_tp_ratio:.2f} (TP hit rate {tp_rate:.0%})")
                bot_logger.info(
                    f"📐 AUTO-LEARN: {pair} TP hit rate {tp_rate:.0%} > {tp_threshold:.0%} — "
                    f"widening TP ratio to {new_tp_ratio:.2f}"
                )
        
        # Persist changes if any
        if changes_made:
            config['updated_at_utc'] = datetime.utcnow().isoformat()
            config['auto_learning_adjustments'] = changes_made
            
            try:
                with open(self.RISK_OVERRIDES_PATH, 'w') as f:
                    json.dump(config, f, indent=2)
                bot_logger.info(
                    f"💾 AUTO-LEARN: Saved {len(changes_made)} adjustments to {self.RISK_OVERRIDES_PATH}"
                )
            except Exception as e:
                bot_logger.error(f"Failed to save auto-learning adjustments: {e}")

    def get_exit_type_stats(self, pair: str = None) -> dict:
        """Get exit type statistics for analysis."""
        if pair:
            pair = self._normalize_pair(pair)
            return dict(self.exit_type_tracking.get(pair, {}))
        return {k: dict(v) for k, v in self.exit_type_tracking.items()}

    def record_time_exit(self, candles_held: int, is_win: bool, pnl: float):
        """Record how long a trade was held vs outcome for time-exit tuning.

        Used to learn optimal max hold duration (4-8 candles).
        """
        self.time_exit_outcomes.append({
            'candles_held': candles_held,
            'is_win': is_win,
            'pnl': pnl,
        })
        if len(self.time_exit_outcomes) > 200:
            self.time_exit_outcomes = self.time_exit_outcomes[-200:]

        # Auto-tune time exit after enough data
        if len(self.time_exit_outcomes) >= 30:
            self._tune_time_exit()

    def _tune_time_exit(self):
        """Auto-tune the time-exit candle threshold.

        Analyzes win rates by hold duration buckets and sets
        the exit threshold where win rate drops below 35%.
        """
        # Group outcomes by candle buckets: 1-3, 4-6, 7-8, 9+
        buckets = {
            '1-3': [], '4-6': [], '7-8': [], '9+': [],
        }
        for outcome in self.time_exit_outcomes:
            c = outcome['candles_held']
            if c <= 3:
                buckets['1-3'].append(outcome)
            elif c <= 6:
                buckets['4-6'].append(outcome)
            elif c <= 8:
                buckets['7-8'].append(outcome)
            else:
                buckets['9+'].append(outcome)

        # Find optimal cutoff
        best_cutoff = 8
        for bucket_name, max_candles in [('1-3', 3), ('4-6', 6), ('7-8', 8), ('9+', 12)]:
            trades = buckets.get(bucket_name, [])
            if len(trades) >= 5:
                wr = sum(1 for t in trades if t['is_win']) / len(trades)
                if wr < 0.35:
                    best_cutoff = max_candles
                    break

        # Update if changed
        if best_cutoff != self.time_exit_candles:
            old = self.time_exit_candles
            self.time_exit_candles = max(4, min(8, best_cutoff))
            if old != self.time_exit_candles:
                bot_logger.info(
                    f"⏱️ Time-exit auto-tuned: {old} → {self.time_exit_candles} candles"
                )

    def get_time_exit_candles(self) -> int:
        """Get the current auto-tuned max hold duration in candles."""
        return self.time_exit_candles

    def get_atr_regime(self) -> str:
        """Get the current ATR regime (expanding/contracting/neutral)."""
        return self.atr_regime

    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------
    def get_adjusted_weights(self, pair: str = None, session: str = None) -> dict:
        weights = dict(self.model_weights)

        if pair:
            pair = self._normalize_pair(pair)
            pair_trades = sum(
                v['total'] for k, v in self.pair_model_accuracy.items()
                if k.startswith(f"{pair}:")
            )
            if pair_trades >= 15:
                pair_adjustments = {}
                for model in weights:
                    pm_key = f"{pair}:{model}"
                    stats = self.pair_model_accuracy.get(pm_key, {'correct': 0, 'total': 0})
                    if stats['total'] >= 3:
                        pair_acc = stats['correct'] / stats['total']
                        pair_adjustments[model] = pair_acc

                if pair_adjustments:
                    adj_sum = sum(pair_adjustments.values())
                    if adj_sum > 0:
                        for model, acc in pair_adjustments.items():
                            normalized_acc = acc / adj_sum * len(pair_adjustments)
                            weights[model] = weights[model] * 0.8 + (normalized_acc / len(weights)) * 0.2

        w_sum = sum(weights.values())
        if w_sum > 0:
            weights = {k: v / w_sum for k, v in weights.items()}
        return weights

    def get_adjusted_threshold(self) -> float:
        # Backtest mode: use the raw threshold without floor
        if getattr(self, 'backtest_mode', False):
            return self.confidence_threshold
        # Live mode: floor at 0.60 — only high-quality sweep signals should enter.
        # Liquidity-sweep signals arrive at 0.70+, so this filters out marginal setups.
        return max(self.confidence_threshold, 0.60)

    def get_pair_win_rate(self, pair: str) -> float:
        pair = self._normalize_pair(pair)
        stats = self.pair_stats.get(pair, {'wins': 0, 'losses': 0})
        total = stats['wins'] + stats['losses']
        return stats['wins'] / total if total > 0 else 0.0

    def get_session_pair_win_rate(self, session: str, pair: str) -> float:
        pair = self._normalize_pair(pair)
        sp_key = f"{session}:{pair}"
        stats = self.session_pair_stats.get(sp_key, {'wins': 0, 'losses': 0})
        total = stats['wins'] + stats['losses']
        return stats['wins'] / total if total > 0 else 0.5

    def get_hour_win_rate(self, hour: int) -> float:
        stats = self.hourly_stats.get(str(hour), {'wins': 0, 'losses': 0})
        total = stats['wins'] + stats['losses']
        return stats['wins'] / total if total > 0 else 0.5

    def get_regime_win_rate(self, regime: str) -> float:
        stats = self.regime_stats.get(regime, {'wins': 0, 'losses': 0})
        total = stats['wins'] + stats['losses']
        return stats['wins'] / total if total > 0 else 0.5

    def get_best_model_combo(self, top_n: int = 3) -> list:
        combos = []
        for combo_key, stats in self.model_combos.items():
            if stats['total'] >= 3:
                wr = stats['wins'] / stats['total']
                combos.append({'models': combo_key.split(':'), 'win_rate': wr, 'total': stats['total']})
        combos.sort(key=lambda x: x['win_rate'], reverse=True)
        return combos[:top_n]

    def get_regime_confidence_modifier(self, regime: str = None) -> float:
        if regime is None:
            regime = self.current_regime
        stats = self.regime_stats.get(regime, {'wins': 0, 'losses': 0})
        total = stats['wins'] + stats['losses']
        if total < 10:
            return 1.0  # Not enough data — stay neutral
        win_rate = stats['wins'] / total
        # Floor at 0.80 to prevent crushing confidence when data is mostly losses
        return max(0.80, 0.6 + (win_rate * 0.8))

    def should_skip_pair(self, pair: str) -> bool:
        pair = self._normalize_pair(pair)
        stats = self.pair_stats.get(pair, {'wins': 0, 'losses': 0})
        total = stats['wins'] + stats['losses']
        if total < 10:
            return False
        win_rate = stats['wins'] / total
        if win_rate < 0.30:
            bot_logger.info(f"📉 Skip recommendation: {pair} win rate {win_rate:.0%} ({stats['wins']}W / {stats['losses']}L)")
            return True
        return False

    def should_skip_session(self, pair: str, session: str = None) -> bool:
        pair = self._normalize_pair(pair)
        if session is None:
            session = self._get_session()
        sp_key = f"{session}:{pair}"
        stats = self.session_pair_stats.get(sp_key, {'wins': 0, 'losses': 0})
        total = stats['wins'] + stats['losses']
        if total < 25:
            return False
        win_rate = stats['wins'] / total
        if win_rate < 0.25:
            bot_logger.info(f"📉 Session skip: {pair} in {session} has {win_rate:.0%} win rate")
            return True
        return False

    def should_skip_hour(self, hour: int = None) -> bool:
        if hour is None:
            hour = datetime.utcnow().hour
        stats = self.hourly_stats.get(str(hour), {'wins': 0, 'losses': 0})
        total = stats['wins'] + stats['losses']
        if total < 8:
            return False
        win_rate = stats['wins'] / total
        return win_rate < 0.20

    def _normalize_pair_stats(self):
        merged = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0})
        for pair, stats in dict(self.pair_stats).items():
            key = self._normalize_pair(pair)
            merged[key]['wins'] += stats.get('wins', 0)
            merged[key]['losses'] += stats.get('losses', 0)
            merged[key]['total_pnl'] += stats.get('total_pnl', 0.0)
        self.pair_stats = merged

    def get_performance_summary(self) -> dict:
        total_trades = len(self.trade_history)
        total_wins = sum(1 for t in self.trade_history if t.get('is_win', False))
        total_pnl = sum(t.get('profit_loss', 0) for t in self.trade_history)
        recent = self.trade_history[-20:]
        recent_wins = sum(1 for t in recent if t.get('is_win', False))
        recent_wr = recent_wins / len(recent) if recent else 0
        model_scores = {}
        for model, stats in self.model_accuracy.items():
            if stats['total'] > 0:
                model_scores[model] = stats['correct'] / stats['total']

        return {
            'total_trades': total_trades,
            'total_wins': total_wins,
            'total_losses': total_trades - total_wins,
            'win_rate': total_wins / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'recent_win_rate': recent_wr,
            'model_weights': dict(self.model_weights),
            'model_accuracy': dict(self.model_accuracy),
            'model_scores': model_scores,
            'confidence_threshold': self.confidence_threshold,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'current_regime': self.current_regime,
            'in_drawdown_protection': self.in_drawdown_protection,
            'drawdown_pct': self.current_drawdown_pct,
            'pair_stats': dict(self.pair_stats),
            'session_stats': dict(self.session_stats),
            'best_combos': self.get_best_model_combo(3),
        }

    # ------------------------------------------------------------------
    # Regime-Adaptive Indicator Tuning
    # ------------------------------------------------------------------
    def get_regime_adjustments(self) -> dict:
        """Get recommended indicator adjustments based on regime performance.
        
        Returns dict with suggested parameter changes for each regime.
        These can be applied by the sweep/technical analyzers.
        """
        adjustments = {}
        
        for regime, stats in self.regime_stats.items():
            wins = stats.get('wins', 0)
            losses = stats.get('losses', 0)
            total = wins + losses
            
            if total < 10:
                continue  # Not enough data
            
            win_rate = wins / total
            
            # Regime-specific adjustments
            if regime == 'volatile':
                if win_rate < 0.30:
                    # Volatile losing → widen stops, tighten entry
                    adjustments[regime] = {
                        'confidence_boost': 0.10,  # Require higher confidence
                        'sl_multiplier': 1.3,     # Wider stops
                        'skip_regime': win_rate < 0.20,  # Skip entirely if very bad
                    }
                else:
                    adjustments[regime] = {
                        'confidence_boost': 0.05,
                        'sl_multiplier': 1.2,
                        'skip_regime': False,
                    }
                    
            elif regime == 'trending':
                if win_rate > 0.45:
                    # Trending doing well → lower threshold, tighter stops
                    adjustments[regime] = {
                        'confidence_boost': -0.05,  # More aggressive
                        'sl_multiplier': 0.9,
                        'skip_regime': False,
                    }
                else:
                    adjustments[regime] = {
                        'confidence_boost': 0.0,
                        'sl_multiplier': 1.0,
                        'skip_regime': False,
                    }
                    
            elif regime == 'ranging':
                if win_rate < 0.35:
                    adjustments[regime] = {
                        'confidence_boost': 0.08,
                        'sl_multiplier': 1.1,
                        'skip_regime': win_rate < 0.25,
                    }
                else:
                    adjustments[regime] = {
                        'confidence_boost': 0.0,
                        'sl_multiplier': 1.0,
                        'skip_regime': False,
                    }
        
        return adjustments
    
    def get_hour_filter(self) -> set:
        """Get hours to skip based on poor performance.
        
        Returns set of UTC hours where win rate is below 30% 
        with at least 10 trades.
        """
        skip_hours = set()
        
        for hour_str, stats in self.hourly_stats.items():
            wins = stats.get('wins', 0)
            losses = stats.get('losses', 0)
            total = wins + losses
            
            if total >= 10:
                win_rate = wins / total
                if win_rate < 0.30:
                    skip_hours.add(int(hour_str))
        
        return skip_hours
    
    def get_session_confidence_modifier(self, session: str) -> float:
        """Get confidence modifier for a session based on performance.
        
        Returns value between -0.15 and +0.10 to adjust confidence.
        """
        stats = self.session_stats.get(session, {})
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        total = wins + losses
        
        if total < 15:
            return 0.0  # Not enough data
        
        win_rate = wins / total
        
        if win_rate >= 0.50:
            return 0.05  # Good session, slight boost
        elif win_rate >= 0.40:
            return 0.0   # Neutral
        elif win_rate >= 0.30:
            return -0.05  # Below average, slight penalty
        else:
            return -0.10  # Poor session, significant penalty
    
    def should_skip_trade(self, pair: str, regime: str, hour: int = None) -> tuple:
        """Check if a trade should be skipped based on learned patterns.
        
        Returns (should_skip: bool, reason: str)
        """
        pair = self._normalize_pair(pair)
        
        # Check regime skip
        adjustments = self.get_regime_adjustments()
        if regime in adjustments and adjustments[regime].get('skip_regime', False):
            return True, f"regime {regime} has <20% win rate"
        
        # Check hour skip
        if hour is not None:
            skip_hours = self.get_hour_filter()
            if hour in skip_hours:
                return True, f"hour {hour} has <30% win rate"
        
        # Check pair-specific loss patterns
        pair_pattern = self.loss_patterns.get(f"pair:{pair}", {})
        if pair_pattern.get('count', 0) >= 20:
            # Too many losses on this pair
            total_pair_trades = self.pair_stats.get(pair, {}).get('wins', 0) + \
                               self.pair_stats.get(pair, {}).get('losses', 0)
            if total_pair_trades > 0:
                loss_ratio = pair_pattern['count'] / total_pair_trades
                if loss_ratio > 0.70:
                    return True, f"pair {pair} has {loss_ratio:.0%} loss ratio"
        
        return False, ""

    # ------------------------------------------------------------------
    # Persistence (with backup rotation)
    # ------------------------------------------------------------------
    MAX_BACKUPS = 5  # Keep up to 5 rotating backups

    def _save(self):
        with self._save_lock:
            self._save_unlocked()

    def _save_unlocked(self):
        os.makedirs(os.path.dirname(LEARNING_DB_PATH), exist_ok=True)

        # Rotate backups before writing
        self._rotate_backups()

        data = {
            'model_weights': self.model_weights,
            'model_accuracy': dict(self.model_accuracy),
            'pair_stats': dict(self.pair_stats),
            'session_stats': dict(self.session_stats),
            'pair_model_accuracy': dict(self.pair_model_accuracy),
            'session_pair_stats': dict(self.session_pair_stats),
            'model_combos': dict(self.model_combos),
            'model_pnl_history': {k: v[-100:] for k, v in self.model_pnl_history.items()},
            'regime_stats': dict(self.regime_stats),
            'hourly_stats': dict(self.hourly_stats),
            'confidence_threshold': self.confidence_threshold,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'peak_balance': self.peak_balance,
            'current_drawdown_pct': self.current_drawdown_pct,
            'in_drawdown_protection': self.in_drawdown_protection,
            'current_regime': self.current_regime,
            'trade_history': self.trade_history[-500:],
            'loss_patterns': dict(self.loss_patterns),
            'recent_trades_window': self.recent_trades_window[-10:],
            # ATR-centric adaptive data
            'session_atr_profiles': {k: {'atr_values': v['atr_values'][-100:], 'avg_atr': v['avg_atr']}
                                     for k, v in self.session_atr_profiles.items()},
            'spread_tracking': {k: {'spreads': v['spreads'][-100:], 'avg_spread': v['avg_spread']}
                                for k, v in self.spread_tracking.items()},
            'recent_sl_values': {k: v[-100:] for k, v in self.recent_sl_values.items()},
            'sl_multiplier_by_pair': self.sl_multiplier_by_pair,
            'time_exit_candles': self.time_exit_candles,
            'time_exit_outcomes': self.time_exit_outcomes[-100:],
            'atr_regime': self.atr_regime,
            # Exit type tracking for auto-learning
            'exit_type_tracking': {k: dict(v) for k, v in self.exit_type_tracking.items()},
            'auto_learning_trade_count': self.auto_learning_trade_count,
            'last_updated': datetime.now().isoformat(),
        }
        with open(LEARNING_DB_PATH, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _rotate_backups(self):
        """Rotate backup files: .bak.1 (newest) through .bak.N (oldest)."""
        try:
            if not os.path.exists(LEARNING_DB_PATH):
                return

            # Shift existing backups: .bak.4 → .bak.5, .bak.3 → .bak.4, etc.
            for i in range(self.MAX_BACKUPS, 1, -1):
                src = f"{LEARNING_DB_PATH}.bak.{i-1}"
                dst = f"{LEARNING_DB_PATH}.bak.{i}"
                if os.path.exists(src):
                    os.replace(src, dst)

            # Copy current file to .bak.1
            import shutil
            shutil.copy2(LEARNING_DB_PATH, f"{LEARNING_DB_PATH}.bak.1")

        except Exception as e:
            bot_logger.warning(f"Backup rotation failed: {e}")

    def restore_from_backup(self, backup_num=1):
        """Restore adaptive state from a specific backup number.

        Args:
            backup_num: Which backup to restore (1=newest, N=oldest).
        """
        backup_path = f"{LEARNING_DB_PATH}.bak.{backup_num}"
        if not os.path.exists(backup_path):
            bot_logger.error(f"Backup {backup_path} not found")
            return False

        try:
            import shutil
            shutil.copy2(backup_path, LEARNING_DB_PATH)
            self._load()
            bot_logger.info(f"✅ Restored adaptive state from backup #{backup_num}")
            return True
        except Exception as e:
            bot_logger.error(f"Failed to restore from backup: {e}")
            return False

    def _load(self):
        if not os.path.exists(LEARNING_DB_PATH):
            return
        try:
            with open(LEARNING_DB_PATH, 'r') as f:
                data = json.load(f)

            base_threshold = self._get_base_confidence_threshold()

            self.model_weights = data.get('model_weights', self.model_weights)
            saved_threshold = data.get('confidence_threshold', base_threshold)
            self.confidence_threshold = max(base_threshold, min(saved_threshold, base_threshold + 0.10))
            self.consecutive_losses = data.get('consecutive_losses', 0)
            self.consecutive_wins = data.get('consecutive_wins', 0)
            self.max_consecutive_losses = data.get('max_consecutive_losses', 0)
            self.trade_history = data.get('trade_history', [])
            self.peak_balance = data.get('peak_balance', 0.0)
            self.current_drawdown_pct = data.get('current_drawdown_pct', 0.0)
            self.in_drawdown_protection = data.get('in_drawdown_protection', False)
            self.current_regime = data.get('current_regime', self.REGIME_RANGING)

            for k, v in data.get('model_accuracy', {}).items():
                self.model_accuracy[k] = v
            for k, v in data.get('pair_stats', {}).items():
                self.pair_stats[k] = v
            for k, v in data.get('session_stats', {}).items():
                self.session_stats[k] = v
            for k, v in data.get('pair_model_accuracy', {}).items():
                self.pair_model_accuracy[k] = v
            for k, v in data.get('session_pair_stats', {}).items():
                self.session_pair_stats[k] = v
            for k, v in data.get('model_combos', {}).items():
                self.model_combos[k] = v
            for k, v in data.get('model_pnl_history', {}).items():
                self.model_pnl_history[k] = v
            for k, v in data.get('regime_stats', {}).items():
                self.regime_stats[k] = v
            for k, v in data.get('hourly_stats', {}).items():
                self.hourly_stats[k] = v
            for k, v in data.get('loss_patterns', {}).items():
                self.loss_patterns[k] = v
            self.recent_trades_window = data.get('recent_trades_window', [])

            # ATR-centric adaptive data
            for k, v in data.get('session_atr_profiles', {}).items():
                self.session_atr_profiles[k] = v
            for k, v in data.get('spread_tracking', {}).items():
                self.spread_tracking[k] = v
            for k, v in data.get('recent_sl_values', {}).items():
                self.recent_sl_values[k] = v
            self.sl_multiplier_by_pair = data.get('sl_multiplier_by_pair', {})
            self.time_exit_candles = data.get('time_exit_candles', 8)
            self.time_exit_outcomes = data.get('time_exit_outcomes', [])
            self.atr_regime = data.get('atr_regime', 'neutral')
            
            # Exit type tracking for auto-learning
            for k, v in data.get('exit_type_tracking', {}).items():
                self.exit_type_tracking[k] = v
            self.auto_learning_trade_count = data.get('auto_learning_trade_count', 0)

            self._normalize_pair_stats()
            for t in self.trade_history:
                if 'pair' in t:
                    t['pair'] = self._normalize_pair(t.get('pair'))

            self._adapt_confidence_threshold()

            bot_logger.info(
                f"📚 Loaded learning data: {len(self.trade_history)} past trades | "
                f"threshold={self.confidence_threshold:.2f} | "
                f"regime={self.current_regime} | "
                f"drawdown_protection={'ON' if self.in_drawdown_protection else 'OFF'}"
            )
        except Exception as e:
            bot_logger.warning(f"Could not load learning data: {e}")
