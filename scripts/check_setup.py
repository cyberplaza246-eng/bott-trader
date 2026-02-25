#!/usr/bin/env python
"""
Setup verification script
Run: python -m scripts.check_setup
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv


def check_setup():
    print("=" * 60)
    print("  AI Forex Trading Bot - Setup Check")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # 1. Python version
    v = sys.version_info
    print(f"\n1. Python version: {v.major}.{v.minor}.{v.micro}", end=" ")
    if v.major >= 3 and v.minor >= 9:
        print("✅")
    else:
        print("❌  Need 3.9+")
        errors.append("Python 3.9+ required")
    
    # 2. Core dependencies
    print("\n2. Dependencies:")
    deps = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'tensorflow': 'tensorflow',
        'sklearn': 'scikit-learn',
        'pandas_ta': 'pandas-ta',
        'flask': 'Flask',
        'flask_cors': 'Flask-CORS',
        'apscheduler': 'APScheduler',
        'dotenv': 'python-dotenv',
        'requests': 'requests',
    }
    
    for module, pip_name in deps.items():
        try:
            __import__(module)
            print(f"   {pip_name:25s} ✅")
        except ImportError:
            print(f"   {pip_name:25s} ❌  pip install {pip_name}")
            errors.append(f"Missing: {pip_name}")
    
    # Optional
    optional = {
        'yfinance': 'yfinance (for data download)',
        'MetaTrader5': 'MetaTrader5 (Windows only)',
    }
    for module, desc in optional.items():
        try:
            __import__(module)
            print(f"   {desc:25s} ✅")
        except ImportError:
            print(f"   {desc:25s} ⚠️  (optional)")
            warnings.append(f"Optional: {desc}")
    
    # 3. Bot modules
    print("\n3. Bot modules:")
    modules = [
        ('config.strategy_config', 'Strategy Config'),
        ('src.broker.mt5_connector', 'MT5 Connector'),
        ('src.ai.lstm_predictor', 'LSTM Predictor'),
        ('src.ai.sentiment_analyzer', 'Sentiment Analyzer'),
        ('src.ai.technical_analyzer', 'Technical Analyzer'),
        ('src.ai.volume_analyzer', 'Volume Analyzer'),
        ('src.ai.multi_timeframe', 'Multi-Timeframe'),
        ('src.ai.support_resistance', 'Support/Resistance'),
        ('src.ai.candlestick_patterns', 'Candlestick Patterns'),
        ('src.ai.adaptive_learner', 'Adaptive Learner'),
        ('src.core.ensemble_trader', 'Ensemble Trader (7 models)'),
        ('src.core.paper_trading', 'Paper Trading'),
        ('src.risk.position_manager', 'Risk Manager'),
        ('src.dashboard.app', 'Web Dashboard'),
        ('src.data.historical_downloader', 'Data Downloader'),
        ('src.backtest.backtest_engine', 'Backtest Engine'),
    ]
    
    for module_path, name in modules:
        try:
            __import__(module_path)
            print(f"   {name:30s} ✅")
        except Exception as e:
            print(f"   {name:30s} ❌  {str(e)[:50]}")
            errors.append(f"Module error: {name}")
    
    # 4. Environment
    print("\n4. Environment:")
    env_file = os.path.exists('.env')
    print(f"   .env file:        {'✅ Found' if env_file else '⚠️  Missing (copy .env.example → .env)'}")
    if not env_file:
        warnings.append("Missing .env file")
    
    load_dotenv()
    
    newsapi = os.getenv('NEWSAPI_KEY', '')
    if newsapi and newsapi != 'YOUR_NEWSAPI_KEY':
        print(f"   NewsAPI key:      ✅ Configured")
    else:
        print(f"   NewsAPI key:      ⚠️  Not set (sentiment will be disabled)")
        warnings.append("NewsAPI key not configured")
    
    mode = os.getenv('TRADING_MODE', 'paper')
    print(f"   Trading mode:     {mode.upper()}")
    
    # 5. LSTM model
    print("\n5. AI Model:")
    if os.path.exists('models/lstm_model.h5'):
        print(f"   LSTM model:       ✅ Trained")
    else:
        print(f"   LSTM model:       ⚠️  Not trained yet (run: python -m scripts.train_lstm)")
        warnings.append("LSTM model not trained")
    
    # 6. Adaptive learning data
    if os.path.exists('data/adaptive_learning.json'):
        import json
        with open('data/adaptive_learning.json') as f:
            data = json.load(f)
        trade_count = len(data.get('trade_history', []))
        print(f"   Learning data:    ✅ {trade_count} trades recorded")
    else:
        print(f"   Learning data:    ⚠️  No history yet (bot will start learning)")
    
    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"❌ {len(errors)} error(s) found:")
        for e in errors:
            print(f"   - {e}")
    else:
        print("✅ All checks passed!")
    
    if warnings:
        print(f"\n⚠️  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"   - {w}")
    
    print("=" * 60)
    return len(errors) == 0


if __name__ == '__main__':
    check_setup()
