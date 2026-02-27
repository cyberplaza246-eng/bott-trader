# Scalping Bot — Quick Start Guide

Get your 5-minute GBPUSD/EURUSD scalping strategy running in 5 minutes.

## What You'll Get

✅ Fully automated 5-minute scalping on GBPUSD & EURUSD  
✅ RSI(9) + EMA 20/50/200 based entries  
✅ Trend-aligned pullback trades  
✅ Micro risk management (6-12 pips SL, 1-2R TP)  
✅ London session preference (08:00-22:00 UTC)  
✅ Paper trading mode for safe testing  

---

## Step 1: Activate Your Python Environment

```bash
cd /workspaces/Ai-bot
source .venv/bin/activate
```

---

## Step 2: Install Dependencies (if not already done)

```bash
pip install -r requirements.txt
```

---

## Step 3: Test with Paper Trading (RECOMMENDED)

### Option A: Backtest on Historical Data First
```bash
python scripts/backtest_scalping.py
```

This will:
- Load historical CSV files from `data/` directory
- Simulate all trades on past data
- Show you win rate, profit factor, average pips
- Help you validate strategy before live trading

**Expected Output:**
```
Results for EUR/USD
─────────────────────────────────────────
Total trades:        42
Winning trades:      28 (66.7%)
Losing trades:       14
Total P&L:           $284.50
Avg P&L per trade:   $6.77
Profit factor:       2.1
```

### Option B: Live Paper Trading
```bash
python scripts/run_scalping_bot.py paper
```

This will:
- Connect to your MT5 account (paper trading mode)
- Monitor GBP/USD and EUR/USD on 5-minute candles
- Automatically execute trades when setups appear
- Print real-time updates to console
- Log all activity to `logs/`

**What You'll See:**
```
✓ London session active (good liquidity)
✓ BUY trend detected (Price > EMA200, EMA50 > EMA200)
✓ Pullback detected at 15.2 pips from entry zone
✓ RSI 52.3 in ideal range (50-55)
✓ Previous candle shows bullish EMA20 rejection

Buy Entry: SL 10p, TP 15p (1.5R)

✅ Scalp trade opened: EUR/USD BUY @ 1.09523
   SL=1.09423 TP=1.09673 (10p risk)
```

**Run it for 1-2 weeks** to collect performance data before going live.

---

## Step 4: Go Live (After Successful Paper Trading)

Once you're confident in the strategy:

```bash
python scripts/run_scalping_bot.py live
```

⚠️ **Only do this after:**
- ✅ Backtesting shows >50% win rate
- ✅ Paper trading for 1-2 weeks is profitable
- ✅ You understand the strategy (read SCALPING_STRATEGY.md)

---

## Configuration

### Basic Setup (Already Done)
Your bot is pre-configured. No changes needed to start.

### Customize Parameters

If you want to adjust risk or entry/exit rules, edit `src/ai/scalping_analyzer.py`:

```python
PAIR_CONFIG = {
    'GBP/USD': {
        'min_sl_pips': 8,
        'max_sl_pips': 12,
        'tp_ratio': [1.5, 2.0],  # ← Change 1.5R/2R here
        'min_rsi_for_buy': 50,
        'max_rsi_for_buy': 55,   # ← Change RSI range here
    },
    'EUR/USD': {
        'min_sl_pips': 6,
        'max_sl_pips': 10,
        'tp_ratio': [1.0, 1.5],   # ← EUR is more conservative
    }
}
```

Or adjust in `src/core/scalping_trader.py`:

```python
MAX_HOLD_MINUTES = {
    'GBP/USD': 15,  # ← Max hold time before force close
    'EUR/USD': 20,
}
```

---

## Monitor Your Trades

### View Live Logs
```bash
# While bot is running in another terminal:
tail -f logs/trades.log
```

### Check Statistics
```python
# In your trading terminal:
python
>>> from src.core.scalping_trader import ScalpingTrader
>>> scalper = ScalpingTrader()
>>> summary = scalper.get_summary()
>>> print(f"Active trades: {summary['active_scalp_trades']}")
```

### Analyze Performance
Backtest results are printed to console. Save important stats:

```
Capture from backtest output:
- Win rate: XX%
- Profit factor: X.X
- Total P&L: $XXX
- Avg pips per trade: X.X
```

---

## Expected Performance

On a **$1000 starting account**:

- **Conservative:** $50-100/week (5-10% return)
- **Normal:** $100-200/week (10-20% return)
- **Aggressive:** $200+/week (20%+ return)

*Performance depends on:*
- Market volatility
- Liquidity (London session preferred)
- Number of setups that appear
- Your spreads (ECN broker = tighter spreads = better returns)

---

## Troubleshooting

### "No signals generated"
- Check your data: need 200+ candles minimum
- Verify trend exists: is price above/below EMA 200?
- Check EMA alignment: is EMA 50 above/below EMA 200?

### "Too many losses"
- Lower your confidence threshold (currently 0.70)
- Require RSI divergence confirmation
- Skip trades during low liquidity (outside London session)

### "Positions hold too long"
- Max hold time is set correctly (15min GBP, 20min EUR)
- Verify bot is actually closing positions
- Check broker connectivity

### "Lots are too small / too large"
- Edit `calculate_lot_size()` in `scalping_trader.py`
- Adjust based on your account balance
- Current range: 0.01 to 0.05 lots

---

## Key Files

| File | Purpose |
|------|---------|
| `src/ai/scalping_analyzer.py` | Core strategy logic (indicators, setups) |
| `src/core/scalping_trader.py` | Trade execution & position management |
| `scripts/run_scalping_bot.py` | Main bot entry point (live/paper) |
| `scripts/backtest_scalping.py` | Historical performance validation |
| `SCALPING_STRATEGY.md` | Full strategy documentation |

---

## Next Steps

1. **Backtest:** `python scripts/backtest_scalping.py`
2. **Paper Trade:** `python scripts/run_scalping_bot.py paper` (1-2 weeks)
3. **Analyze:** Review logs and win rates
4. **Optimize:** Adjust parameters if needed
5. **Go Live:** `python scripts/run_scalping_bot.py live` (only after validation)

---

## Common Questions

**Q: Can I run both scalping and ensemble bot at same time?**  
A: No, not recommended. They may take conflicting trades. Run scalping standalone.

**Q: What broker do I need?**  
A: MT5-based broker (Exness, MetaTrader, etc). ECN accounts with tight spreads recommended.

**Q: Do I need to monitor it?**  
A: Not required once running, but check logs daily. Max trade duration prevents runaway losses.

**Q: Is this legal/profitable?**  
A: Scalping is legal on most brokers. Profitability depends on your discipline and market conditions.

**Q: How much money do I need?**  
A: Minimum $100 for paper trading. For live scalping, $500-1000 recommended.

---

## Support & Debugging

If bot crashes or doesn't work:

1. Check logs: `cat logs/trades.log`
2. Verify MT5 connection: `python scripts/test_mt5.py`
3. Verify data exists: `ls data/`
4. Check Python environment: `python --version`
5. Reinstall deps: `pip install --upgrade -r requirements.txt`

---

## Ready? Let's Go!

```bash
# Test backtest first:
python scripts/backtest_scalping.py

# Then paper trade:
python scripts/run_scalping_bot.py paper

# Check logs in another terminal:
tail -f logs/trades.log
```

Good luck! 🔪📈
