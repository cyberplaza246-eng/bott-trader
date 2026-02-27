# Scalping Bot Architecture

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    5-Minute Scalping Bot                        │
│                   GBPUSD & EURUSD Pairs                         │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                    Data Flow                                         │
└──────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │   MT5 Broker     │
    │ (Live Account)   │
    └────────┬─────────┘
             │ Fetch 5-min candles
             │ (every 5 minutes)
             ▼
    ┌──────────────────────────────────────────┐
    │  Historical Data                         │
    │  (OHLCV Candles)                         │
    │  GBP/USD 5-min, EUR/USD 5-min           │
    │  Minimum: 200+ candles                   │
    └────────┬──────────────────────────────────┘
             │
             │
             ▼
    ╔═══════════════════════════════════════════╗
    ║   src/ai/scalping_analyzer.py             ║
    ║                                           ║
    ║  • Calculate RSI(9)                       ║
    ║  • Calculate EMA 20/50/200                ║
    ║  ├─ Detect Trend (BUY/SELL/NONE)         ║
    ║  ├─ Detect Pullback                      ║
    ║  ├─ Validate Buy Setup                   ║
    ║  ├─ Validate Sell Setup                  ║
    ║  ├─ Calculate Risk/Reward                ║
    ║  └─ Generate Signal                      ║
    ║                                           ║
    ║  Output: {'signal': 'BUY'|'SELL'|'SKIP',║
    ║           'confidence': 0.0-1.0,         ║
    ║           'entry': price,                ║
    ║           'stop_loss': price,            ║
    ║           'take_profit': price}          ║
    ╚═════════┬═════════════════════════════════╝
              │
              │ Signal with confidence score
              │
              ▼
    ╔═══════════════════════════════════════════╗
    ║   src/core/scalping_trader.py             ║
    ║                                           ║
    ║  • Apply signal cooldown (2 min)          ║
    ║  • Validate trade conditions              ║
    ║  • Calculate lot size                     ║
    ║  • Execute trade (BUY/SELL)               ║
    ║  • Monitor active positions               ║
    ║  └─ Check TP/SL/timeout exits             ║
    ║                                           ║
    ║  Uses: RiskManager                        ║
    ║  Integrates: MT5Connector                 ║
    ║  Logs: All trades                         ║
    ╚═════════┬═════════════════════════════════╝
              │
              │ Trade signal
              │
              ▼
    ┌──────────────────────────────────────────┐
    │   MT5 Broker Connection                  │
    │   (via MT5Connector)                     │
    │                                          │
    │   • Open Position                        │
    │   • Monitor Price                        │
    │   • Execute TP / SL                      │
    │   • Close Position                       │
    │   • Report Profit/Loss                   │
    │                                          │
    │   Result: Closed Trade                   │
    └────────────────────────────────────────┘
```

## Processing Loop (Every 5 Minutes)

```
START
  │
  ├─→ [1] Fetch latest 200 candles (5-min)
  │        GBP/USD and EUR/USD
  │        Source: MT5 Broker
  │
  ├─→ [2] Analyze GBP/USD
  │   ├─→ Calculate RSI(9), EMA 20/50/200
  │   ├─→ Detect trend (BUY/SELL/NONE)
  │   ├─→ Check pullback into EMA 20
  │   ├─→ Validate RSI, candle confirmation
  │   ├─→ Score confidence (0-1.0)
  │   └─→ Generate signal + R:R
  │
  ├─→ [3] Analyze EUR/USD
  │   (same as GBP/USD)
  │
  ├─→ [4] Check Signal Cooldown
  │   ├─→ Skip if signal fired < 2 min ago
  │   └─→ Otherwise, proceed
  │
  ├─→ [5] Execute Valid Signals
  │   ├─→ If confidence ≥ 0.70:
  │   ├─→ Calculate lot size (risk management)
  │   ├─→ Send BUY/SELL order to broker
  │   ├─→ Register with position tracker
  │   └─→ Log trade entry
  │
  ├─→ [6] Monitor Active Positions
  │   ├─→ Check TP/SL (auto-close)
  │   ├─→ Check max hold time:
  │   │   GBP: 15 min max
  │   │   EUR: 20 min max
  │   ├─→ Force-close if exceeded
  │   └─→ Log trade exit + P&L
  │
  ├─→ [7] Update Account Balance
  │   ├─→ Fetch current balance from broker
  │   ├─→ Update risk manager tier
  │   └─→ Adjust position size for next trades
  │
  └─→ [8] Wait 5 minutes
       Then repeat from START


    Next Update: +5 minutes
    Daily Status: Printed hourly
    Weekly Stats: Every Monday
```

## Signal Generation (Confidence Scoring)

```
┌─────────────────────────────────────────────────────────────┐
│          Signal Scoring System (0.0 - 1.0)                  │
└─────────────────────────────────────────────────────────────┘

BUY SIGNAL COMPOSITION:
┌─────────────────────────────────────┐
│ ① Trend Alignment (25%)             │
│   ✓ Price > EMA 200           +0.25 │
│   ✓ EMA 50 > EMA 200                │
│   └─ Strength varies 0-1.0           │
│                                     │
│ ② Pullback Detected (20%)           │
│   ✓ Price near EMA 20         +0.20 │
│   ├─ Distance 3-15 pips             │
│   ├─ Within recent swing             │
│   └─ Pullback strength scoring       │
│                                     │
│ ③ RSI Confirmation (30%)            │
│   ✓ RSI 50-55 range           +0.30 │
│   ✓ Rising or stable (>45)          │
│   ├─ Perfect: 50-52 (+0.30)         │
│   ├─ Good: 48-55 (+0.15)            │
│   └─ Weak: <45 (+0.00, skip)        │
│                                     │
│ ④ Candle Confirmation (25%)         │
│   ✓ Bull candle > EMA20       +0.25 │
│   ├─ Strong close (>0.5 range)      │
│   ├─ Rejection of EMA (prev candle) │
│   └─ Color + momentum                │
│                                     │
│ ⑤ RSI Divergence (Bonus +10%)       │
│   ✓ Price higher, RSI lower   +0.10 │
│   └─ Bullish divergence detected    │
└─────────────────────────────────────┘

TOTAL SCORE = Sum of above (max 1.0)

✓ Score ≥ 0.70 → EXECUTE
✗ Score < 0.70 → SKIP


EXAMPLES:
─────────────────────────────────────
Price > 200, EMA50 > 200: +0.25  Score: 0.25
+ Pullback to EMA20:      +0.20  Score: 0.45
+ RSI 52 (good range):    +0.30  Score: 0.75 ✓ EXECUTE
+ Strong bull candle:     +0.25  Score: 1.00
```

## Entry & Exit Logic

```
╔════════════════════════════════════════════════╗
║              BUY SETUP                         ║
╠════════════════════════════════════════════════╣
║ ENTRY CONDITIONS:                              ║
║ 1. Trend: Price > EMA200, EMA50 > EMA200       ║
║ 2. Pullback: Price near EMA20 from above       ║
║ 3. RSI: Falls to 50-55 (above midline)         ║
║ 4. Candle: Bull close > EMA20                  ║
║ 5. Confidence ≥ 0.70                           ║
║                                                ║
║ ENTRY METHOD: Market Buy @ current price       ║
║                                                ║
║ POSITION MANAGEMENT:                           ║
║ ├─ SL: 8-12 pips below entry (GBP)             ║
║ ├─ SL: 6-10 pips below entry (EUR)             ║
║ ├─ TP1: 1.5R below entry (GBP)                 ║
║ ├─ TP2: 2.0R below entry (GBP)                 ║
║ ├─ Max Hold: 15 min (GBP), 20 min (EUR)        ║
║ └─ Scale Out: 50% at TP1, 50% at TP2           ║
║                                                ║
║ EXIT SCENARIOS:                                ║
║ A) Price hits TP → Close at TP (profit)        ║
║ B) Price hits SL → Close at SL (loss)          ║
║ C) Time exceeds max → Close at market          ║
║ D) Force exit rules trigger → Close at market  ║
╚════════════════════════════════════════════════╝

╔════════════════════════════════════════════════╗
║              SELL SETUP                        ║
╠════════════════════════════════════════════════╣
║ ENTRY CONDITIONS:                              ║
║ 1. Trend: Price < EMA200, EMA50 < EMA200       ║
║ 2. Pullback: Price near EMA20 from below       ║
║ 3. RSI: Rises to 45-50 (below midline)         ║
║ 4. Candle: Bear close < EMA20                  ║
║ 5. Confidence ≥ 0.70                           ║
║                                                ║
║ ENTRY METHOD: Market Sell @ current price      ║
║                                                ║
║ POSITION MANAGEMENT:                           ║
║ ├─ SL: 8-12 pips above entry (GBP)             ║
║ ├─ SL: 6-10 pips above entry (EUR)             ║
║ ├─ TP1: 1.5R above entry (GBP)                 ║
║ ├─ TP2: 2.0R above entry (GBP)                 ║
║ ├─ Max Hold: 15 min (GBP), 20 min (EUR)        ║
║ └─ Scale Out: 50% at TP1, 50% at TP2           ║
║                                                ║
║ EXIT SCENARIOS:                                ║
║ A) Price hits TP → Close at TP (profit)        ║
║ B) Price hits SL → Close at SL (loss)          ║
║ C) Time exceeds max → Close at market          ║
║ D) Force exit rules trigger → Close at market  ║
╚════════════════════════════════════════════════╝
```

## Risk Management Flow

```
┌──────────────────────────────────────────────────────────┐
│  Position Sizing & Risk Management                       │
└──────────────────────────────────────────────────────────┘

Input: New trade signal
       Entry Price, Direction (BUY/SELL)
       Pair (GBP/USD or EUR/USD)

        ↓
    ┌─────────────────────────────────────────┐
    │ 1. GET ACCOUNT INFORMATION              │
    │    - Current balance ($)                 │
    │    - Account tier (micro/mini/standard)  │
    │    - Max concurrent trades allowed       │
    │    - Max lot size per tier               │
    └─────────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────────┐
    │ 2. CALCULATE STOP LOSS DISTANCE          │
    │    GBP/USD: 8-12 pips target            │
    │    EUR/USD: 6-10 pips target            │
    │    (Pair-specific from config)           │
    └─────────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────────┐
    │ 3. CALCULATE RISK AMOUNT ($)            │
    │    Risk = Balance × Risk% × Tier        │
    │    Example: $1000 × 1% = $10/trade      │
    └─────────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────────┐
    │ 4. CALCULATE LOT SIZE                   │
    │    Lot = Risk / (SL_pips × pip_value)   │
    │                                          │
    │    Example:                              │
    │    Lot = $10 / (10p × $10/pip)          │
    │    Lot = $10 / $100 = 0.10              │
    └─────────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────────┐
    │ 5. APPLY CONSTRAINTS                    │
    │    ├─ Min: 0.01 lot (micro)             │
    │    ├─ Max: 0.05 lot (per tier)          │
    │    ├─ Max concurrent: 2-3 trades        │
    │    └─ Daily loss limit: 10% balance     │
    └─────────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────────┐
    │ 6. EXECUTE TRADE                        │
    │    ├─ Open position:                    │
    │    │  Volume: 0.10 lots                 │
    │    │  Type: BUY or SELL                 │
    │    │  Entry: current price              │
    │    │  SL: entry ± SL pips               │
    │    │  TP: entry ± TP pips               │
    │    │  Comment: "Scalp BUY"              │
    │    │                                     │
    │    └─ Register in tracking:             │
    │       ticket, direction, entry_time,   │
    │       max_hold_time, R:R ratio          │
    └─────────────────────────────────────────┘
        ↓
    Output: Position opened with all protections
```

## Integration Points

```
┌────────────────────────────────────────────────────────────┐
│         Integration with Existing Bot Components           │
└────────────────────────────────────────────────────────────┘

src/ai/scalping_analyzer.py
    ├─ Standalone: Can be used independently
    └─ Integrated: Can feed signals to ensemble

src/core/scalping_trader.py
    ├─ MT5Connector: Executes trades via broker
    ├─ RiskManager: Calculates position size, account tier
    ├─ TradeLogger: Logs all trades for analysis
    └─ BackgroundScheduler: Updates every 5 minutes

scripts/run_scalping_bot.py
    ├─ Paper Mode: Simulates trades (no real money)
    ├─ Live Mode: Real trading via MT5
    └─ Logging: Detailed trade logs to files

scripts/backtest_scalping.py
    ├─ Historical Data: CSV format OHLCV
    ├─ Simulation: Runs analyzer on past candles
    └─ Analysis: Win rate, profit factor, stats
```

## File Locations

```
/workspaces/Ai-bot/
│
├── src/
│   ├── ai/
│   │   └── scalping_analyzer.py        ← Core strategy
│   ├── core/
│   │   └── scalping_trader.py          ← Trade execution
│   ├── broker/
│   │   └── mt5_connector.py            ← Broker connection
│   ├── risk/
│   │   └── position_manager.py         ← Risk calculation
│   └── utils/
│       └── logger.py                   ← Logging
│
├── scripts/
│   ├── run_scalping_bot.py             ← Main bot (LIVE/PAPER)
│   └── backtest_scalping.py            ← Backtest engine
│
├── config/
│   ├── strategy_config.py              ← Main config
│   └── scalping_config_template.py     ← Scalping config
│
├── data/
│   ├── EUR_USD_1h.csv                  ← Historical data
│   └── GBP_USD_1h.csv
│
├── logs/
│   └── trades.log                      ← Trade log
│
├── SCALPING_README.md                  ← This file
├── SCALPING_QUICKSTART.md              ← Quick start
├── SCALPING_STRATEGY.md                ← Full strategy
└── SCALPING_BOT_SUMMARY.md             ← Summary
```

---

## Quick Command Reference

```bash
# Setup
cd /workspaces/Ai-bot
source .venv/bin/activate
pip install -r requirements.txt

# Validate on historical data
python scripts/backtest_scalping.py

# Paper trading (safe test)
python scripts/run_scalping_bot.py paper

# Live trading (real money - caution!)
python scripts/run_scalping_bot.py live

# Monitor logs
tail -f logs/trades.log

# View active trades
python -c "from src.core.scalping_trader import ScalpingTrader; \
           print(ScalpingTrader().get_summary())"
```

---

## Expected Performance Timeline

```
Day 1-2:      Backtest (verify strategy works)
Week 1-2:     Paper trade (learn & adjust)
Week 3-4:     Paper trade continued (validate stats)
Month 2+      Live trading (after validation)


Milestones:
├─ Backtest win rate > 50%
├─ Paper trading matches backtest (+/- 5%)
├─ 2+ weeks paper trading is profitable
├─ All risk controls verified
└─ Ready for live trading!
```

---

This is a complete, production-ready scalping bot! 🚀
