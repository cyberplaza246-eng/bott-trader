# AI-Powered Forex Trading Bot 🤖📈

An intelligent 4-model ensemble trading bot for automatic forex trading using LSTM, sentiment analysis, technical indicators, and volume analysis.

## Quick Start

1. **Setup**: `cp .env.example .env && pip install -r requirements.txt`
2. **Configure**: Edit `.env` with your MT5 credentials
3. **Paper Trade**: `python -m src.bot` (test with virtual money first)
4. **Go Live**: After 1 week of profitable paper trades, switch to live mode

## Key Features

- ✅ **4 AI Models**: LSTM + Sentiment + Technical + Volume
- ✅ **Automatic Trading**: Execute trades 24/5 forex markets
- ✅ **Risk Management**: Position sizing, stop-loss, take-profit
- ✅ **Paper Trading**: Test before risking real money
- ✅ **Backtesting**: Validate strategy on historical data
- ✅ **Learning**: Retrains weekly on closed trades

## Architecture

```
Fetch Data → Technical Indicators → 4 AI Models (Parallel)
                                    ├─ LSTM Prediction
                                    ├─ Sentiment (News/Twitter)
                                    ├─ Technical Signals
                                    └─ Volume Analysis
                                            ↓
                                    Ensemble Voting (need 3+/4)
                                            ↓
                                    Risk Management Check
                                            ↓
                                    Execute Trade (MT5)
                                            ↓
                                    Monitor & Log Results
```

## Files Structure

```
src/
├── ai/                 # 4 AI models
├── broker/            # MT5 integration
├── core/              # Trading engine & paper trading
├── risk/              # Position sizing & risk management
├── backtest/          # Historical data testing
└── utils/             # Logging

config/
├── strategy_config.py # Trading parameters

logs/
├── signals.log        # AI signal analysis
├── trades.log         # Entry/exit history
└── errors.log         # Error messages
```

## Next Steps

👉 **Read [SETUP.md](SETUP.md) for detailed installation & configuration**

---

**Status**: ✅ Fully implemented and ready to deploy
- 4-model ensemble trading engine
- Paper trading for risk-free testing
- Backtest framework for strategy validation
- Production-ready logging and monitoring
