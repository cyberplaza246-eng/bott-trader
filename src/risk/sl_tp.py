"""
Unified SL/TP Calculation Module

Single source of truth for stop-loss and take-profit placement.

SL Strategy (priority order):
  1. Sweep wick + ATR buffer (when sweep detected)
  2. Nearest swing structure + ATR buffer
  3. ATR fallback (1.2×ATR from entry)
  Floor: max(1.0×ATR, 2×spread)
  Hard cap: 5 pips (1M) / 10 pips (5M)

TP Strategy:
  1. 85% of distance to nearest S/R level
  2. Must give ≥ 1.2R
  3. No valid S/R → skip trade (return None)
"""
import numpy as np
from src.utils.logger import bot_logger


# ── Pair Configuration ──────────────────────────────────────────────
PAIR_CONFIG = {
    'EUR/USD': {
        'spread_sim': 0.00006,    # 0.6 pips (ECN TradersWay)
        'pip_size': 0.0001,
    },
    'GBP/USD': {
        'spread_sim': 0.00010,    # 1.0 pip (ECN TradersWay)
        'pip_size': 0.0001,
    },
    'USD/JPY': {
        'spread_sim': 0.008,      # 0.8 pips (ECN TradersWay)
        'pip_size': 0.01,
    },
}

# ── Constants ───────────────────────────────────────────────────────
SL_ATR_BUFFER = 0.50          # Buffer beyond structure level (50% ATR)
SL_ATR_FALLBACK = 1.2         # Fallback: 1.2×ATR when no structure
SL_MIN_ATR = 1.0              # Floor: at least 1.0×ATR
SL_MIN_SPREAD_MULT = 2        # Floor: at least 2×spread
SL_MAX_PIPS_1M = 5.0          # Hard cap: 5 pips for 1M trades
SL_MAX_PIPS_5M = 10.0         # Hard cap: 10 pips for 5M trades

SR_TP_FRACTION = 0.90         # TP at 90% of distance to S/R (avoids reversal before TP)
MIN_RR = 1.2                  # Minimum reward-to-risk ratio
MAX_RR = 2.0                  # Maximum R:R (scalps don't need 3R+)
TP_MAX_PIPS_1M = 8.0          # Hard cap: 8 pips TP for 1M scalps
TP_MAX_PIPS_5M = 15.0         # Hard cap: 15 pips TP for 5M scalps

STRUCTURE_LOOKBACK = 30       # Bars to scan for swing structure


def calculate_sl_tp(df, direction, pair, timeframe,
                    sr_levels=None, sweep_wick=None):
    """Calculate stop-loss and take-profit for a trade.

    Args:
        df: DataFrame with OHLCV + indicators (needs 'atr', 'volume')
        direction: 'BUY' or 'SELL'
        pair: Currency pair e.g. 'EUR/USD'
        timeframe: '1m' or '5m'
        sr_levels: dict with 'resistance_levels' and 'support_levels' lists
        sweep_wick: float price of the sweep candle wick (or None)

    Returns:
        dict with {stop_loss, take_profit, sl_pips, tp_pips, rr_ratio,
                   sl_reason, tp_reason} or None if no valid setup
    """
    config = PAIR_CONFIG.get(pair, PAIR_CONFIG['EUR/USD'])
    pip_size = config['pip_size']
    spread = config['spread_sim']

    latest = df.iloc[-1]
    entry_price = float(latest['close'])
    atr = float(latest.get('atr', 0) or 0)

    if atr <= 0:
        bot_logger.warning(f"SL/TP skip: ATR={atr} for {pair}")
        return None

    # ════════════════════════════════════════════════════════════════
    #  STOP LOSS
    # ════════════════════════════════════════════════════════════════
    sl_distance, sl_reason = _calculate_sl(
        df, direction, entry_price, atr, pip_size, spread,
        timeframe, sweep_wick
    )

    if sl_distance is None:
        return None

    if direction == 'BUY':
        stop_loss = round(entry_price - sl_distance, 5)
    else:
        stop_loss = round(entry_price + sl_distance, 5)

    # ════════════════════════════════════════════════════════════════
    #  TAKE PROFIT
    # ════════════════════════════════════════════════════════════════
    tp_result = _calculate_tp(
        direction, entry_price, sl_distance, pip_size, sr_levels
    )

    if tp_result is None:
        bot_logger.info(f"🚫 {pair} no valid S/R target for TP — trade skipped")
        return None

    tp_distance, tp_reason = tp_result

    # ── TP Cap: max R:R ─────────────────────────────────────────
    max_rr_dist = sl_distance * MAX_RR
    if tp_distance > max_rr_dist:
        tp_distance = max_rr_dist
        tp_reason += f" → capped to {MAX_RR:.0f}R"

    # ── TP Cap: hard pip limit ──────────────────────────────────
    tp_max_pips = TP_MAX_PIPS_1M if timeframe == '1m' else TP_MAX_PIPS_5M
    tp_max_dist = tp_max_pips * pip_size
    if tp_distance > tp_max_dist:
        tp_distance = tp_max_dist
        tp_reason += f" → capped {tp_max_pips:.0f}p ({timeframe})"

    if direction == 'BUY':
        take_profit = round(entry_price + tp_distance, 5)
    else:
        take_profit = round(entry_price - tp_distance, 5)

    rr_ratio = tp_distance / sl_distance
    sl_pips = sl_distance / pip_size
    tp_pips = tp_distance / pip_size

    bot_logger.info(
        f"📍 SL: {sl_pips:.1f}p — {sl_reason}"
    )
    bot_logger.info(
        f"🎯 TP: {tp_pips:.1f}p ({rr_ratio:.1f}R) — {tp_reason}"
    )

    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'entry_price': entry_price,
        'sl_distance': sl_distance,
        'tp_distance': tp_distance,
        'sl_pips': sl_pips,
        'tp_pips': tp_pips,
        'rr_ratio': rr_ratio,
        'sl_reason': sl_reason,
        'tp_reason': tp_reason,
    }


# ════════════════════════════════════════════════════════════════════
#  SL INTERNALS
# ════════════════════════════════════════════════════════════════════

def _calculate_sl(df, direction, entry_price, atr, pip_size, spread,
                  timeframe, sweep_wick):
    """Determine SL distance and reason.

    Priority: sweep wick → swing structure → ATR fallback
    Then apply floor + hard pip cap.

    Returns:
        (sl_distance, reason) or (None, None)
    """
    atr_buffer = atr * SL_ATR_BUFFER

    # ── Priority 1: Sweep wick ──────────────────────────────────────
    if sweep_wick is not None:
        if direction == 'BUY':
            raw_dist = entry_price - (sweep_wick - atr_buffer)
        else:
            raw_dist = (sweep_wick + atr_buffer) - entry_price

        if raw_dist > 0:
            sl_distance = raw_dist
            reason = f"sweep wick {sweep_wick:.5f} + {SL_ATR_BUFFER}×ATR buffer"
        else:
            # Wick on wrong side — fall through to structure
            sweep_wick = None

    # ── Priority 2: Swing structure ─────────────────────────────────
    if sweep_wick is None:
        struct = _find_structure_sl(df, direction, entry_price, atr)
        if struct:
            sl_distance = struct['distance']
            reason = struct['reason']
        else:
            # ── Priority 3: ATR fallback ────────────────────────────
            sl_distance = atr * SL_ATR_FALLBACK
            reason = f"ATR fallback ({SL_ATR_FALLBACK}×ATR)"

    # ── Floor ───────────────────────────────────────────────────────
    floor = max(atr * SL_MIN_ATR, spread * SL_MIN_SPREAD_MULT)
    if sl_distance < floor:
        sl_distance = floor
        reason += f" → floored to {sl_distance/pip_size:.1f}p"

    # ── Hard pip cap ────────────────────────────────────────────────
    max_pips = SL_MAX_PIPS_1M if timeframe == '1m' else SL_MAX_PIPS_5M
    max_dist = max_pips * pip_size
    if sl_distance > max_dist:
        old_pips = sl_distance / pip_size
        sl_distance = max_dist
        reason = f"capped {old_pips:.1f}p → {max_pips:.0f}p ({timeframe} limit)"

    return sl_distance, reason


def _find_structure_sl(df, direction, entry_price, atr):
    """Find the best swing-structure SL level.

    Scans recent candles for swing lows (BUY) or swing highs (SELL),
    scores by significance (volume, touch count, distance), and places
    SL beyond with an ATR buffer.

    Returns:
        dict {distance, level, reason} or None
    """
    if len(df) < STRUCTURE_LOOKBACK:
        return None

    lookback = df.iloc[-STRUCTURE_LOOKBACK:]
    atr_buffer = atr * SL_ATR_BUFFER

    vol_avg = lookback['volume'].rolling(window=5, min_periods=1).mean()

    if direction == 'BUY':
        candidates = _score_swing_lows(lookback, entry_price, atr, vol_avg)
    else:
        candidates = _score_swing_highs(lookback, entry_price, atr, vol_avg)

    if not candidates:
        return None

    # Best = highest significance score
    candidates.sort(key=lambda c: c['score'], reverse=True)
    best = candidates[0]
    level = best['price']

    if direction == 'BUY':
        sl_level = level - atr_buffer
        sl_distance = entry_price - sl_level
    else:
        sl_level = level + atr_buffer
        sl_distance = sl_level - entry_price

    if sl_distance <= 0:
        return None

    touches = best['touches']
    touch_txt = f" ({touches} touches)" if touches > 0 else ""

    return {
        'distance': sl_distance,
        'level': sl_level,
        'reason': f"swing {'low' if direction == 'BUY' else 'high'} "
                  f"{level:.5f}{touch_txt} + {SL_ATR_BUFFER}×ATR",
    }


def _score_swing_lows(data, entry_price, atr, vol_avg):
    """Detect and score swing lows within the lookback window."""
    results = []
    for i in range(2, len(data) - 1):
        lo = float(data.iloc[i]['low'])
        if lo < data.iloc[i - 1]['low'] and lo < data.iloc[i + 1]['low']:
            dist = entry_price - lo
            if dist <= 0 or dist > atr * 3.0:
                continue
            score, touches = _significance(data, i, 'low', lo, atr, vol_avg)
            results.append({'price': lo, 'distance': dist,
                            'score': score, 'touches': touches})
    return results


def _score_swing_highs(data, entry_price, atr, vol_avg):
    """Detect and score swing highs within the lookback window."""
    results = []
    for i in range(2, len(data) - 1):
        hi = float(data.iloc[i]['high'])
        if hi > data.iloc[i - 1]['high'] and hi > data.iloc[i + 1]['high']:
            dist = hi - entry_price
            if dist <= 0 or dist > atr * 3.0:
                continue
            score, touches = _significance(data, i, 'high', hi, atr, vol_avg)
            results.append({'price': hi, 'distance': dist,
                            'score': score, 'touches': touches})
    return results


def _significance(data, idx, col, price, atr, vol_avg):
    """Score a swing point by volume, touch count, and distance."""
    row = data.iloc[idx]
    vol = float(row.get('volume', 1.0))
    avg = float(vol_avg.iloc[idx]) if idx < len(vol_avg) else 1.0

    score = 1.0

    # Volume weight
    ratio = vol / max(avg, 0.001)
    if ratio > 1.2:
        score += 0.3
    elif ratio > 1.0:
        score += 0.1

    # Touch count (retests within ±10% ATR)
    tol = atr * 0.1
    touches = 0
    for j in range(max(0, idx - 15), min(len(data), idx + 5)):
        if j != idx:
            val = float(data.iloc[j][col])
            if abs(val - price) <= tol:
                touches += 1
    if touches >= 2:
        score += 0.4
    elif touches == 1:
        score += 0.2

    # Prefer closer levels (distance decay)
    entry = float(data.iloc[-1]['close'])
    dist = abs(entry - price)
    score *= 1.0 / (1.0 + dist / (atr * 0.5))

    return score, touches


# ════════════════════════════════════════════════════════════════════
#  TP INTERNALS
# ════════════════════════════════════════════════════════════════════

def _calculate_tp(direction, entry_price, sl_distance, pip_size, sr_levels):
    """Find TP at 85% of distance to nearest S/R level.

    Returns:
        (tp_distance, reason) or None
    """
    if not sr_levels:
        return None

    if direction == 'BUY':
        levels = sr_levels.get('resistance_levels', [])
        above = sorted([r for r in levels if r > entry_price])
        if not above:
            return None
        nearest = above[0]
        full_dist = nearest - entry_price
    else:
        levels = sr_levels.get('support_levels', [])
        below = sorted([s for s in levels if s < entry_price], reverse=True)
        if not below:
            return None
        nearest = below[0]
        full_dist = entry_price - nearest

    if full_dist <= 0:
        return None

    tp_dist = full_dist * SR_TP_FRACTION
    rr = tp_dist / sl_distance if sl_distance > 0 else 0

    if rr < MIN_RR:
        bot_logger.info(
            f"⚠️ S/R level {nearest:.5f} gives {rr:.1f}R < {MIN_RR}R minimum"
        )
        # Try next level if available
        if direction == 'BUY':
            further = [r for r in above if r > nearest]
        else:
            further = [s for s in below if s < nearest]

        for level in further:
            if direction == 'BUY':
                fd = level - entry_price
            else:
                fd = entry_price - level
            td = fd * SR_TP_FRACTION
            rr2 = td / sl_distance if sl_distance > 0 else 0
            if rr2 >= MIN_RR:
                tp_dist = td
                nearest = level
                rr = rr2
                break
        else:
            return None

    reason = (
        f"85% to {'resistance' if direction == 'BUY' else 'support'} "
        f"{nearest:.5f} ({full_dist/pip_size:.1f}p away)"
    )
    return tp_dist, reason
