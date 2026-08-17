# AI-Powered Forex Trading Bot 🤖📈

A sophisticated **4-model ensemble AI trading bot** that automatically trades forex using LSTM predictions, sentiment analysis, technical indicators, and volume analysis. Designed for accuracy and profitability even on small accounts.

## Features

✅ **4-Model Ensemble AI**
- LSTM Neural Network for price prediction
- Real-time sentiment analysis (news + social media)
- Technical indicators (RSI, MACD, Bollinger Bands, ATR)
- Volume analysis for signal confirmation

✅ **Automatic Trading**
- Auto-execute trades based on high-confidence signals
- Automatic position sizing based on risk management
- Stop-loss and take-profit calculation
- Real-time trade monitoring

✅ **Risk Management**
- Position sizing (2% risk per trade)
- Drawdown limits (max 10% daily loss)
- Correlated pair hedging
- Maximum concurrent position limits

✅ **Multiple Trading Modes**
- **Paper Trading**: Test signals with virtual money before going live
- **Live Trading**: Real money execution via MT5
- **Backtesting**: Validate strategy on historical data

✅ **Learning System**
- Weekly model retraining on closed trades
- Adaptive parameters based on P/L history
- Continuous improvement as you trade

## Quick Start

### 1. Prerequisites
- Python 3.8+
- MetaTrader 5 account (Exness, OANDA, or similar)
- NewsAPI key (free at https://newsapi.org)
- ~$50+ for micro lot trading

### 2. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Ai-bot.git
cd Ai-bot

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your MT5 credentials and API keys
```

### 3. Configuration

Edit `.env` with:
```
MT5_ACCOUNT=12345678
MT5_PASSWORD=your_password
MT5_SERVER=Exness-MT5
NEWSAPI_KEY=your_newsapi_key
TRADING_MODE=paper  # Start with 'paper' trading first
INITIAL_BALANCE=50
```

### 4. Start Paper Trading (Recommended First)

```bash
# Run in paper trading mode (virtual money, 24 hours)
python -m src.bot

# Monitor signals in logs/signals.log and logs/trades.log
```

Expected output:
```
🔍 Starting analysis cycle at 14:32:05
EUR/USD Analysis:
  Signal: BUY
  Confidence: 78%
  Models Agreement: 4/4
  Details: LSTM: BUY (65%) | Sentiment: BULLISH (72%) | Technical: BUY (75%) | Volume: BUY (60%)

🎯 EXECUTING BUY TRADE:
  Pair: EUR/USD
  Entry: 1.0950
  Stop Loss: 1.0935
  Take Profit: 1.0980
  Lot Size: 0.01
  Risk: $1.00
```

### 5. Run Backtest

```bash
# Test strategy on historical data (3 years)
python scripts/run_backtest.py --pair EUR/USD --start 2022-01-01 --end 2024-12-31

# Results:
# Total Trades: 128
# Win Rate: 58%
# Sharpe Ratio: 1.8
# Max Drawdown: -12%
```

### 6. Go Live (After Validation)

Once you're confident in paper trading results:

1. Monitor paper trades for 1-2 weeks
2. Verify win rate > 50% and profits consistent
3. Change `.env`:
   ```
   TRADING_MODE=live
   INITIAL_BALANCE=50
   ```
4. Start bot with micro lots (0.01 lot size)
5. Monitor daily P/L in `logs/trades.log`

## 24/7 Deployment (Recommended)

For always-on operation when you are offline, run the bot as a `systemd` service
on a Linux VPS/server (not Codespaces).

```bash
# from project root
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# configure env
cp .env.example .env
# edit .env with MT5 relay + live settings

# install and start service
./scripts/install_systemd_service.sh

# inspect service
sudo systemctl status ai-bot
sudo journalctl -u ai-bot -f
```

Service files are generated from:

- `deploy/systemd/ai-bot.service.template`
- `scripts/install_systemd_service.sh`

## Architecture

### Core Components

```
src/
├── ai/                    # 4 AI models
│   ├── lstm_predictor.py      # Neural network for price direction
│   ├── sentiment_analyzer.py  # News + Twitter sentiment
│   ├── technical_analyzer.py  # RSI, MACD, Bollinger Bands
│   └── volume_analyzer.py     # Volume spike detection
├── broker/
│   └── mt5_connector.py       # MT5 API integration
├── core/
│   ├── ensemble_trader.py     # 4-model voting system
│   ├── paper_trading.py       # Virtual account for testing
│   └── bot.py                 # Main trading loop
├── risk/
│   └── position_manager.py    # Position sizing, stop-loss, take-profit
├── backtest/
│   └── backtest_engine.py     # Historical data testing
└── utils/
    └── logger.py              # Structured logging
```

### Trading Flow

```
1. Fetch OHLCV data (5-minute candles)
   ↓
2. Calculate Technical Indicators (RSI, MACD, Bollinger Bands, ATR)
   ↓
3. Run 4 AI Models in parallel:
   ├─ LSTM: Predict price direction
   ├─ Sentiment: Analyze news/Twitter for bullish/bearish signals
   ├─ Technical: Confirm with indicator patterns
   └─ Volume: Check for unusual volume spikes
   ↓
4. Ensemble Voting:
   - Need 3+ models agreeing (75%+ combined confidence)
   ↓
5. Risk Management Check:
   - Calculate position size (1% account risk)
   - Set stop-loss (1.5x ATR below entry)
   - Set take-profit (1:2 risk:reward ratio)
   - Check max concurrent trades (3) and daily loss limit (10%)
   ↓
6. Execute Trade (if all checks pass)
   ├─ Live mode: Real MT5 order
   └─ Paper mode: Virtual account simulation
   ↓
7. Monitor Position:
   - Update P/L with current price
   - Close if stop-loss or take-profit hit
   ↓
8. Log Trade Results
   - Record in trades.log for weekly retraining
```

## Configuration Reference

### Risk Parameters (`config/strategy_config.py`)
- `RISK_PER_TRADE_PERCENT`: 1% (max risk per trade = $0.50 for $50 account)
- `MAX_CONCURRENT_TRADES`: 3 (avoid overlapping positions)
- `DAILY_LOSS_LIMIT_PERCENT`: 10% (auto-pause at $5 loss)
- `STOP_LOSS_MULTIPLIER`: 1.5x ATR
- `TAKE_PROFIT_RATIO`: 1:2 risk:reward

### AI Thresholds
- `ENSEMBLE_CONFIDENCE_THRESHOLD`: 75% (need high confidence)
- `MIN_MODELS_AGREEMENT`: 3/4 models must agree

### Trading Pairs
- EUR/USD (most liquid, tightest spread)
- GBP/USD
- USD/JPY

## Logs & Monitoring

All trading activity is logged to `logs/`:

- **signals.log**: AI signal analysis and confidence scores
- **trades.log**: Entry/exit prices, P/L, reasons
- **trading_bot.log**: General bot status and errors
- **errors.log**: Error details for debugging

View live logs:
```bash
tail -f logs/trades.log
```

## Weekly Workflow

1. **Review**: Check P/L, win rate, biggest winners/losers
2. **Retrain**: Bot retrains LSTM on closed trades from past week
3. **Optimize**: Adjust thresholds if P/L is negative
4. **Deploy**: Updated models roll out automatically

## Troubleshooting

### MT5 Connection Failed
```
Check:
- MT5 terminal is running
- Account number and password are correct
- Internet connection is stable
```

### No Signals Generated
```
Check:
- NEWSAPI_KEY is valid
- Market hours (forex opens 17:00 UTC Sunday - 21:00 UTC Friday)
- Enough candle data (need 100+ candles for indicators)
```

### Losing Money
```
1. Switch to paper mode: less risk
2. Lower RISK_PER_TRADE_PERCENT to 0.5%
3. Increase MIN_MODELS_AGREEMENT to 4 (stricter filtering)
4. Review logs to understand why trades are losing
5. Backtest on different timeframes
```

## Performance Targets

**After 1 week of paper trading:**
- Win rate: > 50%
- Sharpe ratio: > 1.0
- Max drawdown: < 15%

**After 1 month of live trading ($50 account):**
- Consistent 5-10% monthly growth
- No catastrophic loss days (> 50%)

## Advanced Usage

### Use Different LSTM Model
```python
from src.ai.lstm_predictor import LSTMPredictor
lstm = LSTMPredictor(lookback_window=120)  # Use longer history
lstm.train(historical_prices, epochs=100)
```

### Adjust Model Weights
```python
# In src/core/ensemble_trader.py
self.model_weights = {
    'lstm': 0.40,        # More weight to ML
    'sentiment': 0.25,   # Less weight to news
    'technical': 0.20,
    'volume': 0.15
}
```

### Add More Currency Pairs
```python
# In config/strategy_config.py
PAIRS = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD']
```

## API Keys Needed

1. **NewsAPI** (free): https://newsapi.org
   - Get up to 100 requests/day free
   - Need for sentiment analysis

2. **MetaTrader 5 Account** (free): Any major broker
   - Exness: Good for small accounts (micro lots)
   - OANDA: Good for USA traders
   - IC Markets: Good for high liquidity

## Contributing

Suggestions for improvements:
- [ ] Add more AI models (SVM, Random Forest)
- [ ] Integrate crypto sentiment
- [ ] Multi-timeframe analysis
- [ ] Economic calendar awareness
- [ ] Machine learning hyperparameter optimization
- [ ] Discord/Telegram alerts

## License

MIT License - feel free to use and modify for your own trading

## Disclaimer

⚠️ **IMPORTANT**: 
- This bot automates trading - you can lose money quickly
- Start with paper trading (fake money) for at least 1 week
- Never risk money you can't afford to lose
- Past performance does not guarantee future results
- This is not financial advice - do your own research

## Support

- Open an issue on GitHub for bugs
- Check logs/ directory for detailed error messages
- Review strategy_config.py for tuning parameters
- Test on backtest_engine.py before going live

---

**Ready to trade?** Start with:
```bash
cp .env.example .env
# Edit .env with your credentials
python -m src.bot  # Paper trading mode
```

Good luck! 📊🚀
