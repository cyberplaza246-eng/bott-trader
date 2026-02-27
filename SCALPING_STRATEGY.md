# 5-Minute Scalping Bot — Complete Documentation

## Overview
A high-frequency scalping strategy for GBPUSD and EURUSD on 5-minute timeframes. Uses pullback entries in trending markets with strict micro risk management (6-12 pips stop loss, 1-2R take profit).

**Target Markets:** GBP/USD, EUR/USD  
**Timeframe:** 5 minutes  
**Session:** Prefers London session (08:00-22:00 UTC)  
**Trade Type:** Directional scalp trades, 5-75 minute hold times

---

## Strategy Components

### 1. Indicators

#### RSI (9-Period)
- **Purpose:** Primary momentum filter for entry timing
- **Levels:**
  - 70 = Overbought
  - 50 = Midline (primary trend filter)
  - 30 = Oversold
- **Usage:**
  - **Buy:** RSI 50-55 (above midline, room to run)
  - **Sell:** RSI 45-50 (below midline, room to fall)
  - **Extreme:** RSI 70/30 = momentum continuation, NOT reversal (in a trend)

#### EMA 20 (Entry Zone)
- **Purpose:** Defines the pullback entry zone
- **Behavior:** Price pulls back to/near EMA 20 within a trend
- **Entry Logic:** Trade enters when price closes beyond EMA 20 with confirmation

#### EMA 50 (Medium Trend)
- **Purpose:** Confirms trend direction relative to EMA 200
- **Buy Signal:** EMA 50 > EMA 200 (bullish alignment)
- **Sell Signal:** EMA 50 < EMA 200 (bearish alignment)

#### EMA 200 (Macro Bias)
- **Purpose:** Overall bearish/bullish bias filter
- **Buy Only:** If price > EMA 200 (bullish bias)
- **Sell Only:** If price < EMA 200 (bearish bias)
- **Avoids:** Counter-trend trades against macro trend

---

## Entry Rules

### Buy Setup
1. **Trend Requirement:**
   - Price > EMA 200
   - EMA 50 > EMA 200
   - EMA 200 sloping up (uptrend confirmed)

2. **Pullback Confirmation:**
   - Price pulls back from recent high, approaching EMA 20
   - Price still above EMA 20 (no full mean reversion)
   - EMA 20 acts as support/platform

3. **RSI Confirmation:**
   - RSI falls to 50-55 range during pullback
   - RSI NOT below 45 (too weak)
   - Divergence preferred: price makes higher low, RSI makes lower low

4. **Candle Confirmation:**
   - Bullish candle closes above EMA 20
   - Previous candle showed rejection of EMA 20 downward
   - Current candle momentum is upward

### Sell Setup
1. **Trend Requirement:**
   - Price < EMA 200
   - EMA 50 < EMA 200
   - EMA 200 sloping down (downtrend confirmed)

2. **Pullback Confirmation:**
   - Price pulls back from recent low, approaching EMA 20
   - Price still below EMA 20 (no full mean reversion)
   - EMA 20 acts as resistance/platform

3. **RSI Confirmation:**
   - RSI rises to 45-50 range during pullback
   - RSI NOT above 55 (too strong)
   - Divergence preferred: price makes lower high, RSI makes higher high

4. **Candle Confirmation:**
   - Bearish candle closes below EMA 20
   - Previous candle showed rejection of EMA 20 upward
   - Current candle momentum is downward

---

## Risk Management

### Stop Loss & Take Profit (Per Pair)

#### GBP/USD (Faster Moving)
- **Stop Loss:** 8-12 pips from entry
  - Exact placement varies by pullback strength
  - Further pullback = wider SL (12p), shallow pullback = tighter SL (8p)
- **Take Profit:** 1.5R to 2R
  - First TP at 1.5× SL distance (strong scalp)
  - Second TP at 2× SL distance (extended scalp)
  - Scale out 50/50 on each TP level

#### EUR/USD (Slower Moving)
- **Stop Loss:** 6-10 pips from entry
  - Tighter stops due to lower volatility
  - SL placement priority: under recent swing
- **Take Profit:** 1R to 1.5R
  - First TP at 1× SL distance (quick exits)
  - Second TP at 1.5× SL distance
  - EUR moves more deliberately than GBP

### Position Sizing
- Based on account balance via `RiskManager`
- **Risk per trade:** 1-2% of account (tier dependent)
- **Max concurrent scalps:** 2-3 trades
- **Minimum lot:** 0.01 (10k units)
- **Maximum hold:** 15 minutes (75 candles)

---

## Trading Filters

### 1. Session Filter
- **Preferred:** London session (08:00-22:00 UTC)
- **Acceptable:** Asian + London overlap (good liquidity)
- **Avoid:** New York close + Asia dead zone (22:00-00:00 UTC) — wide spreads

### 2. Trend Confirmation
- **Only trade pullbacks in established trends**
- Trend = EMA 50/200 alignment + price above/below 200
- **Minimum strength:** EMA spread must be >10 pips to confirm trend

### 3. Spread Filter
- **GBP/USD:** Max 3.5 pips spread
- **EUR/USD:** Max 3.0 pips spread
- Skip if spread > max (poor execution)

### 4. News Avoidance
- Skip entries 30 minutes before/after high-impact economic events
- Focus on medium+ currency pairs (check daily events)
- Examples: ECB/Fed rate decisions, unemployment, inflation

### 5. Cooldown
- Minimum **2 minutes between signals** on same pair
- Prevents overtrading/whipsaws
- If setup triggers again in cooldown → skip

### 6. Max Trade Duration
- Force exit after max hold time:
  - **GBP/USD:** 15 minutes max
  - **EUR/USD:** 20 minutes max
- Close at market if still in trade

---

## Signal Output Format

```python
{
    'signal': 'BUY' | 'SELL' | 'SKIP',
    'confidence': 0.0-1.0,
    'setup': 'buy_setup' | 'sell_setup' | 'none',
    'entry_price': float,
    'stop_loss': float,
    'take_profit': float,
    'risk_reward': {
        'risk_pips': float,
        'reward_pips_1': float,
        'reward_pips_2': float,
    },
    'reasons': ['reason1', 'reason2', ...],
    'trend': {
        'direction': 'BUY' | 'SELL',
        'strength': 0.0-1.0,
        'price_position': 'above_200' | 'below_200',
    }
}
```

---

## Implementation Files

### Core Modules

#### `src/ai/scalping_analyzer.py` — Main Strategy Logic
- **Class:** `ScalpingAnalyzer`
- **Methods:**
  - `calculate_indicators(df)` — Compute RSI(9), EMA 20/50/200
  - `detect_trend(df)` — Identify trend direction/strength
  - `detect_pullback(df, trend)` — Detect pullback into entry zone
  - `detect_buy_setup(df, pair)` — Check all conditions for long entry
  - `detect_sell_setup(df, pair)` — Check all conditions for short entry
  - `calculate_risk_reward(entry, direction, pair)` — Compute SL/TP
  - `get_signal(df, pair)` — Generate final trading signal

#### `src/core/scalping_trader.py` — Trade Execution & Management
- **Class:** `ScalpingTrader`
- **Methods:**
  - `analyze_pair(df, pair)` — Get signal with cooldown
  - `validate_trade_conditions(signal, pair)` — Pre-trade checks
  - `calculate_lot_size(pair, sl_pips)` — Position sizing
  - `execute_trade(signal, pair)` — Open position via broker
  - `check_active_trades()` — Monitor holds, force-close on timeout
  - `process_candle(df_gbp, df_eur)` — Main processing loop
  - `get_summary()` — Current status report

### Scripts

#### `scripts/run_scalping_bot.py` — Live/Paper Trading Bot
```bash
# Run in paper trading mode (default)
python scripts/run_scalping_bot.py paper

# Run in live trading mode
python scripts/run_scalping_bot.py live
```

**What it does:**
- Connects to MT5 broker
- Fetches 5-min candles every 5 minutes
- Runs analyzer on both GBP/USD and EUR/USD
- Executes trades automatically
- Monitors positions for closure conditions
- Logs all activity

#### `scripts/backtest_scalping.py` — Historical Backtesting
```bash
python scripts/backtest_scalping.py
```

**What it does:**
- Loads historical CSV data from `data/` directory
- Simulates strategy on past data
- Generates:
  - Win rate, profit factor
  - Average profit/loss per trade
  - Trade-by-trade analysis
  - Equity curve
- Validates strategy before live trading

---

## Usage Guide

### 1. Installation
```bash
# Ensure dependencies installed
pip install -r requirements.txt

# Optional: TensorFlow (for LSTM in main ensemble)
pip install tensorflow>=2.13.0
```

### 2. Configuration
Edit `config/strategy_config.py`:
```python
TRADING_MODE = 'paper'  # or 'live' after validation
INITIAL_BALANCE = 1000
TRADING_MODE = 'paper'
PAIRS = ['EUR/USD', 'GBP/USD']  # Include scalping pairs
```

### 3. Backtest First
```bash
# Validate strategy on historical data
python scripts/backtest_scalping.py

# Look for:
# - Win rate > 50%
# - Profit factor > 1.5
# - Reasonable avg pips per trade
```

### 4. Paper Trade
```bash
# Test in paper mode (no real money)
python scripts/run_scalping_bot.py paper

# Monitor output for:
# - Correct entries at EMA pullbacks
# - Proper risk/reward
# - Natural exit on TP/SL
# Run for 1-2 weeks to gather stats
```

### 5. Live Trade (After Validation)
```bash
# Only after successful paper trading
python scripts/run_scalping_bot.py live
```

---

## Key Advantages

1. **High Win Rate** — Pullback entries have good risk/reward
2. **Short Duration** — 5-75 min holds = quick capital turnover
3. **Trend-Aligned** — EMA filter prevents counter-trend trades
4. **Pair-Specific Tuning** — GBP vs EUR settings optimized separately
5. **Multiple Confirmation** — RSI + EMA + candle = low false signals

---

## Key Risks

1. **Overtrading** — Multiple opportunities per day. Use cooldown & max hold.
2. **News Events** — Check economic calendar. Skip 30min before/after.
3. **Spread Widening** — During low liquidity. Monitor & skip trades.
4. **Whipsaws** — EMA can act as double-edged sword. Rely on RSI+candle.
5. **Micro Position Size** — Can be hard to scale. Start on micro account.

---

## Optimization Tips

### If Win Rate < 50%
- Increase RSI thresholds (more confirmation needed)
- Use tighter SL (prioritize fewer large losses)
- Require stronger EMA spread (20p min between EMA50/200)

### If Average Profit Too Low
- Reduce SL distance (wider TP = more margin)
- Only trade during peak liquidity hours
- Use EMA divergence as additional filter

### If Too Few Signals
- Loosen pullback detection threshold
- Use smaller EMA periods (EMA 15 + EMA 40 + EMA 100)
- Accept weaker trend (EMA spread > 5p instead of 10p)

### If Positions Holding Too Long
- Reduce max hold time (10 min instead of 15)
- Use earlier partial profit-taking (0.75R instead of 1R)
- Tighten RSI range for entry (50-52 instead of 50-55)

---

## Troubleshooting

### No signals generated
- Check if historical data is 200+ candles
- Verify EMA calculations (should see values in "details")
- Confirm trend requirements (price vs EMA 200, EMA 50 vs EMA 200)

### Too many false signals
- Increase confidence threshold (change from 0.70 to 0.75+)
- Require divergence (check RSI divergence code)
- Use additional filter (e.g., volume breakout)

### Positions holding beyond max time
- Verify `MAX_HOLD_MINUTES` config
- Check if broker is responding to close commands
- Monitor logs for force-close attempts

### Large losses on entry
- Check SL calculation (should be 6-12 pips)
- Verify spread didn't widen at entry
- Consider skipping during low-liquidity periods

---

## Performance Expectations

Based on the strategy design:
- **Win Rate:** 55-65% (pullback entries are high-probability)
- **Avg Profit/Loss:** 1:1.5 to 1:2 risk/reward
- **Trades Per Week:** 5-15 (depending on market conditions)
- **Annualized Return:** 20-40% on micro account (high leverage from position sizing)

*Note: Actual results depend on market conditions, execution quality, and risk management discipline.*

---

## Integration with Main Bot

The scalping strategy can run **standalone** or integrated into the main ensemble trader:

### Standalone (Recommended for Learning)
```bash
python scripts/run_scalping_bot.py paper
```
- Runs only scalping strategy
- Easier to test & debug
- Direct MT5 connection

### Integrated (Advanced)
Edit `src/bot.py` to add scalping as a parallel strategy:
```python
from src.core.scalping_trader import ScalpingTrader

class TradingBot:
    def __init__(self):
        self.ensemble = EnsembleTrader(...)  # Existing
        self.scalper = ScalpingTrader(...)   # New
    
    def process_signals(self, df_gbp, df_eur):
        # Get ensemble signal
        ensemble_signal = self.ensemble.get_trading_signal(df_gbp, 'GBP/USD')
        
        # Get scalping signal (as additional overlay)
        scalp_signal = self.scalper.analyze_pair(df_gbp, 'GBP/USD')
        
        # Use both for trading decisions
        combined_signal = self.combine_signals(ensemble_signal, scalp_signal)
```

---

## Questions & Support

- Check logs in `logs/` directory for detailed execution info
- Verify your data format matches expected OHLCV
- Validate broker connection: `test_mt5.py`
- Review backtest results before any live trading
