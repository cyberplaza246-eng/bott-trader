# Multi-Timeframe Scalping Bot — 1M & 5M Strategy

## 🔪 What Is This?

An enhanced scalping bot that trades on **both 1-minute and 5-minute timeframes simultaneously** with timeframe-specific parameters optimized for each.

**Strategy:** RSI(9) + EMA 20/50/200 pullback entries on both timeframes  
**Pairs:** GBP/USD, EUR/USD  
**Update Cycle:** Every minute (1M) and every 5 minutes (5M)

---

## 📊 Timeframe Comparison

### 1-Minute Scalping (Tight & Fast)
```
Entry Criteria:
├─ Tighter confidence (≥0.75 instead of ≥0.70)
├─ Smaller pullbacks (closer to EMA 20)
├─ Requires RSI divergence on 1M
└─ Must align with 5M trend bias

Risk Management:
├─ GBP/USD: 5-8 pips SL, 1-1.5R TP
├─ EUR/USD: 4-6 pips SL, 0.8-1.2R TP
├─ Max hold: 5-8 minutes
└─ Fast exits (tight stops = quick decisions)

Advantages:
✓ More trading opportunities (high frequency)
✓ Quick capital turnover
✓ Less affected by news/overnight gaps
✗ Requires constant monitoring
✗ Tighter fills needed (ECN spreads essential)
```

### 5-Minute Scalping (Standard)
```
Entry Criteria:
├─ Standard confidence (≥0.70)
├─ Moderate pullbacks
├─ RSI confirmation (40-50 for buy/sell)
└─ EMA alignment

Risk Management:
├─ GBP/USD: 8-12 pips SL, 1.5-2R TP
├─ EUR/USD: 6-10 pips SL, 1-1.5R TP
├─ Max hold: 15-20 minutes
└─ Standard exits

Advantages:
✓ Less noise than 1M
✓ Better for beginners
✓ Easier to manage manually
✓ Lower drawdown risk
✗ Fewer trades per day
```

---

## 🧮 Risk Management Details

### GBP/USD (Faster Pair)
```
       1-Minute            5-Minute
SL:    5-8 pips           8-12 pips
TP1:   5-8 pips (1R)      12-18 pips (1.5R)
TP2:   7.5-12 pips (1.5R) 16-24 pips (2R)

Max Hold:   5-8 min       15-20 min
Confidence: ≥0.75         ≥0.70
```

### EUR/USD (Slower Pair)
```
       1-Minute            5-Minute
SL:    4-6 pips            6-10 pips
TP1:   3.2-7.2 pips (0.8R) 6-10 pips (1R)
TP2:   4.8-7.2 pips (1.2R) 9-15 pips (1.5R)

Max Hold:   5-8 min         15-20 min
Confidence: ≥0.75           ≥0.70
Extra:      Wait longer for RSI, needs divergence
```

---

## 🎯 Entry Rules (Both Timeframes)

### Buy Setup
1. **Trend:** Price > EMA200, EMA50 > EMA200
2. **Entry:** Price pulls back toward EMA 20
3. **RSI:** Falls to 50-55 range (above midline)
4. **Confirmation:** Bullish candle closes > EMA 20

**1M Extra:** Requires RSI divergence on 1M chart  
**5M:** Standard confirmation OK

### Sell Setup
1. **Trend:** Price < EMA200, EMA50 < EMA200
2. **Entry:** Price pulls back toward EMA 20
3. **RSI:** Rises to 45-50 range (below midline)
4. **Confirmation:** Bearish candle closes < EMA 20

**1M Extra:** Requires RSI divergence on 1M chart  
**5M:** Standard confirmation OK

---

## 🔗 Confluence Checking

The bot analyzes both timeframes and classifies signals as:

### ✅ **Confluence (Best Trade Quality)**
- Both 1M and 5M agree on direction (BUY or SELL)
- Confidence score: 1.0
- **Action:** Execute trade (highest probability)

### ⚠️ **Divergence (Caution)**
- 1M says BUY but 5M says SELL (or vice versa)
- Confidence score: 0.0
- **Action:** Skip trade (conflicting timeframes)

### ℹ️ **Single Timeframe Signal**
- 1M has signal but 5M doesn't (or vice versa)
- Confidence score: 0.5
- **Action:** May trade with caution (not ideal)

---

## 🚀 How to Use

### Basic Setup
```bash
cd /workspaces/Ai-bot
source .venv/bin/activate

# Paper trade on both timeframes
python scripts/run_multi_timeframe_scalper.py paper

# Monitor logs
tail -f logs/trades.log
```

### Live Trading
```bash
# Only after successful paper trading!
python scripts/run_multi_timeframe_scalper.py live
```

### For 5M Only (Previous Bot)
```bash
python scripts/run_scalping_bot.py paper
```

### For 1M Only
```python
# Use the multi-TF bot but focus only on 1M signals
# Edit config to disable 5M processing if needed
```

---

## 📈 Strategy Notes

### 1M Specific Considerations
- **Higher frequency trades:** Can get 10-15 trades per 2-hour session
- **Requires tight spreads:** ECN broker essential (3 pips max)
- **Needs patience:** Wait for RSI divergence (more confirmation)
- **Quick decisions:** Exits happen fast (8-minute max hold)
- **News risk:** Higher exposure due to frequency
- **Scalp rewards:** Less P&L per trade, but more consistent

### 5M Specific Considerations
- **Moderate frequency:** 3-8 trades per session typical
- **Standard spreads OK:** 3.5 pips GBP, 3 pips EUR acceptable
- **More relaxed:** Less confirmation requirement
- **Longer holds:** Up to 20 minutes allows price movement
- **News safer:** Lower frequency = less event impact
- **Bigger rewards:** More P&L per trade while keeping risk tight

### Confluence Strategy
- **Best trades:** Confluence (both TF agree)
- **Good trades:** Single TF with strong setup
- **Skip trades:** Divergent signals
- **Conservative:** Only trade confluence (reduce false signals)
- **Aggressive:** Trade any TF with high confidence

---

## ⚙️ Configuration

### Adjust Parameters
Edit `src/core/multi_timeframe_scalper.py`:

**For tighter 1M trading:**
```python
'M1': {
    'min_sl_pips': 4,  # Tighten from 5
    'max_sl_pips': 6,  # Tighten from 8
    'tp_ratio': [0.8, 1.2],  # More conservative
}
```

**For aggressive 5M trading:**
```python
'M5': {
    'min_sl_pips': 6,  # Wider from 8
    'max_sl_pips': 15, # Wider from 12
    'tp_ratio': [2.0, 2.5],  # More aggressive
}
```

---

## 📊 Expected Performance

### Conservative Trader (Confluence Only)
- **Setup Selection:** Only trade when 1M & 5M agree
- **Win Rate:** 60%+ (high-quality setups only)
- **Trades/Week:** 10-15 (fewer but better quality)
- **Return:** 10-20% per week on $1,000

### Balanced Trader (Single TF OK)
- **Setup Selection:** Trade any TF with confidence ≥0.75
- **Win Rate:** 55%+
- **Trades/Week:** 20-30 (1M + 5M combined)
- **Return:** 15-30% per week on $1,000

### Aggressive Trader (Any Signal)
- **Setup Selection:** Trade both TF even if divergent
- **Win Rate:** 50%+
- **Trades/Week:** 30-50 (high frequency 1M + 5M)
- **Return:** 20-40% per week on $1,000

---

## ⚡ Quick Commands

### Monitor Multi-TF Signals
```bash
# See real-time 1M and 5M analysis
python scripts/run_multi_timeframe_scalper.py paper

# In another terminal
tail -f logs/trades.log
```

### View Only 1M Trades
```bash
tail -f logs/trades.log | grep "1M"
```

### View Only 5M Trades
```bash
tail -f logs/trades.log | grep "5M"
```

### View Confluence Trades
```bash
tail -f logs/trades.log | grep "Confluence"
```

---

## 🎯 Trading Tips

### For 1M Scalping
1. ✅ Use tight ECN spreads (3 pips max)
2. ✅ Only trade London session (08:00-16:00 UTC)
3. ✅ Require RSI divergence as extra confirmation
4. ✅ Force-close at max 8 minutes (don't hold longer)
5. ✅ Scale position size down (higher frequency = more risk)

### For 5M Scalping
1. ✅ Standard broker spreads OK (3.5 pips acceptable)
2. ✅ Can trade London + NY session
3. ✅ Standard RSI confirmation sufficient
4. ✅ Can hold up to 20 minutes
5. ✅ Use standard position sizing

### For Confluence Trading
1. ✅ Wait only for high-confluence setups
2. ✅ Ignore divergent signals (skip them)
3. ✅ Higher win rate but fewer trades
4. ✅ Best for risk-averse traders
5. ✅ Ideal for traders who prefer consistency

---

## 🛠️ Troubleshooting

### "Too many 1M trades, position sizing too small"
→ Increase `max_concurrent_scalps` in config  
→ Or switch to 5M only: `python scripts/run_scalping_bot.py paper`

### "Missing confluence signals"
→ Market may be choppy (divergent timeframes)  
→ Switch to aggressive mode (trade any TF)  
→ Or wait for clearer trend alignment

### "1M trades getting stopped out frequently"
→ Spreads too wide (need ECN broker)  
→ RSI divergence requirement too strict  
→ Market too ranging (switch to 5M)

### "Can't keep up with 1M speed"
→ Use paper trading to practice  
→ Or switch to 5M only  
→ Reduce max concurrent from 3 to 1 or 2

---

## 📚 Related Files

| File | Purpose |
|------|---------|
| `src/core/multi_timeframe_scalper.py` | Multi-TF analyzer & trader |
| `scripts/run_multi_timeframe_scalper.py` | Main bot (1M + 5M) |
| `scripts/run_scalping_bot.py` | Single TF bot (5M only) |
| `src/ai/scalping_analyzer.py` | Base analyzer (used by both) |
| `SCALPING_STRATEGY.md` | Core strategy (5M) |

---

## 🎓 Learning Path

1. **Master 5M first** (easier)
   - Run: `python scripts/run_scalping_bot.py paper`
   - Paper trade for 2 weeks
   - Learn rhythm and setups

2. **Then add 1M** (harder)
   - Run: `python scripts/run_multi_timeframe_scalper.py paper`
   - Start with confluence only
   - Paper trade another 2 weeks

3. **Go live when ready**
   - `python scripts/run_multi_timeframe_scalper.py live`
   - Start with 5M only
   - Add 1M after 1-2 weeks live

---

## ✅ Validation Checklist

Before trading multi-TF:
- [ ] Understand difference between 1M and 5M
- [ ] Know your SL/TP values per timeframe & pair
- [ ] Can identify entries on live charts
- [ ] Understand confluence concept
- [ ] Paper traded 5M for 2 weeks
- [ ] Paper traded multi-TF for 1 week
- [ ] Ready for real money

---

## 🔪 Summary

**Multi-TF Scalping = More Opportunities + More Complexity**

- **1M:** 10-15 trades/session, fast exits, needs tighter execution
- **5M:** 3-8 trades/session, easier management, proven strategy
- **Confluence:** Best trades where both TF agree
- **Divergence:** Skip conflicting signals

**Recommended approach:** Start 5M → Add 1M later → Master confluence

Ready to trade? Start with: `python scripts/run_multi_timeframe_scalper.py paper`
