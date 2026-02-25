# Quick Start Guide - AI Forex Trading Bot

## 5-Minute Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Credentials
```bash
cp .env.example .env
# Edit .env with your MT5 account and NewsAPI key:
# MT5_ACCOUNT = your account number
# MT5_PASSWORD = your password
# MT5_SERVER = Exness-MT5 (or your broker)
# NEWSAPI_KEY = from https://newsapi.org
# TRADING_MODE = paper (start with this!)
```

### Step 3: Check Setup
```bash
python scripts/check_setup.py
# Should show all ✅ checks passing
```

### Step 4: Test MT5 Connection
```bash
python scripts/test_mt5.py
# Should show account info and EUR/USD price
```

### Step 5: Start Paper Trading
```bash
python -m src.bot
# Bot runs indefinitely - press Ctrl+C to stop
```

---

## What Happens When You Run the Bot

```
🚀 AI-Powered Forex Trading Bot Starting...

🔍 Starting analysis cycle at 14:32:05
EUR/USD Analysis:
  Signal: BUY
  Confidence: 78%
  Models Agreement: 4/4
  Details: LSTM: BUY (65%) | Sentiment: BULLISH (72%) | ...

🎯 EXECUTING BUY TRADE:
  Pair: EUR/USD
  Entry: 1.0950
  Stop Loss: 1.0935
  Take Profit: 1.0980
  Lot Size: 0.01
  Risk: $1.00

📊 Daily Status:
  Balance: $50.50
  Daily Loss: $0.00 (0%)
  Open Trades: 1
  Can Trade: Yes
```

---

## Monitor Your Trades

**Real-time logs:**
```bash
# Watch all trades in real-time
tail -f logs/trades.log

# Watch signals being analyzed
tail -f logs/signals.log

# Watch errors (if any)
tail -f logs/errors.log
```

---

## Testing Before Going Live

### Option 1: Paper Trading (Recommended)
```bash
# In .env:
TRADING_MODE=paper  # Virtual $50 account

# Run and test for 1-2 weeks
python -m src.bot

# Check logs/trades.log for results
# Look for: Win Rate > 50%, Consistent profits
```

### Option 2: Backtesting
```bash
# Test strategy on 3 years of historical data
python scripts/run_backtest.py --pair EUR/USD --start 2022-01-01 --end 2024-12-31

# Output should show:
# Win Rate > 50%
# Sharpe Ratio > 1.0
# Max Drawdown < 15%
```

---

## Going Live (After Validation)

Only after 1-2 weeks of profitable paper trading:

```bash
# 1. Edit .env
TRADING_MODE=live  # Real money!

# 2. Verify INITIAL_BALANCE is still 50
INITIAL_BALANCE=50

# 3. Start bot
python -m src.bot

# 4. Monitor closely first day
# Check logs every 30 minutes
tail -f logs/trades.log
```

**Safety checks before going live:**
- [ ] Paper trading profitable for 1+ weeks
- [ ] Win rate > 50%
- [ ] No days with > 50% losses
- [ ] You understand what each trade does
- [ ] You've read the SETUP.md file completely

---

## If Something Goes Wrong

### Bot won't connect to MT5
```bash
# 1. Make sure MT5 terminal is open
# 2. Run test:
python scripts/test_mt5.py

# 3. Check .env credentials match MT5 terminal
# 4. Try restarting MT5
```

### No signals are being generated
```bash
# 1. Check newsapi_key in .env is valid
# 2. Run in backtest mode to test indicators:
python scripts/run_backtest.py --pair EUR/USD

# 3. Check logs/errors.log for issues
tail logs/errors.log
```

### Losing money (paper or live)
```bash
# 1. Switch to paper trading immediately (if live)
TRADING_MODE=paper

# 2. Reduce risk per trade:
RISK_PER_TRADE_PERCENT=0.5  # Lower from 1%

# 3. Require more model agreement:
MIN_MODELS_AGREEMENT=4  # All 4 models must agree

# 4. Review closed trades
# 5. Consider backtesting with different parameters
```

---

## Daily Checklist

Each day before markets open:

- [ ] Restart bot: `python -m src.bot`
- [ ] Monitor first trade: `tail -f logs/trades.log`
- [ ] Check daily P/L every few hours
- [ ] If daily loss > 10%, bot auto-stops
- [ ] Before bed, review: `cat logs/trades.log | tail -20`

---

## Weekly Checklist

Every Friday evening:

- [ ] Review all trades from past week
- [ ] Calculate win rate: (wins / total trades) × 100
- [ ] Calculate average P/L per trade
- [ ] Check if recent trades match paper backtesting results
- [ ] If losing streak: pause and debug
- [ ] Bot automatically retrains LSTM on closed trades

---

## Key Files to Understand

1. **config/strategy_config.py**
   - All trading parameters
   - Risk settings, thresholds, pairs

2. **src/core/ensemble_trader.py**
   - How 4 models vote together
   - Confidence calculation

3. **src/risk/position_manager.py**
   - Position sizing formula
   - Stop-loss and take-profit calculation

4. **logs/trades.log**
   - Your trading history
   - Every entry, exit, and P/L

---

## Expected Results

**Paper Trading (1 week):**
- 20-50 total trades
- Win rate: 45-55%
- P/L: -$5 to +$10 (small account)

**Live Trading First Month:**
- 5-15 trades per week
- Win rate: 50-60%
- Monthly return: 5-20% (if profitable)

**Important**: Actual results depend on market conditions. Past performance ≠ future results.

---

## Questions?

1. **Strategy not working?** → Review logs/signals.log to understand why trades are losing
2. **MT5 issues?** → Run `python scripts/test_mt5.py`
3. **Need to adjust?** → Edit config/strategy_config.py
4. **Want to learn more?** → Read SETUP.md for complete documentation

---

**Ready?** Start with:
```bash
python scripts/check_setup.py    # Verify everything
python scripts/test_mt5.py       # Connect to MT5
python -m src.bot               # Start trading!
```

Good luck! 📊🚀
