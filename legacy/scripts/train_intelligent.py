#!/usr/bin/env python3
"""
Train Intelligent Trading Models

This script trains the ML models from the intelligent-trading-bot integration.
It can be run standalone or scheduled for periodic retraining.

Usage:
    python scripts/train_intelligent.py --pair MES --data data/MES_1m.csv
    python scripts/train_intelligent.py --all
"""
import argparse
import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime

from src.ai.intelligent_trading import (
    IntelligentTrader,
    FeatureGenerator,
    LabelGenerator,
    BacktestEngine
)
from src.utils.logger import bot_logger


def load_config(config_path: str = None) -> dict:
    """Load intelligent trading configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config' / 'intelligent_config.jsonc'
    
    if not Path(config_path).exists():
        bot_logger.warning(f"Config not found at {config_path}, using defaults")
        return {}
    
    with open(config_path) as f:
        content = f.read()
        # Remove JSONC comments
        import re
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        return json.loads(content)


def load_data(data_path: str) -> pd.DataFrame:
    """Load OHLCV data from CSV."""
    df = pd.read_csv(data_path, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(data_path, nrows=1).columns else [0])
    
    # Standardize column names
    col_map = {
        'time': 'timestamp',
        'date': 'timestamp',
        'datetime': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'vol': 'volume'
    }
    
    df.columns = [col_map.get(c.lower(), c.lower()) for c in df.columns]
    
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)
    
    # Ensure required columns
    required = ['open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Add volume if missing
    if 'volume' not in df.columns:
        df['volume'] = 1.0
    
    return df


def train_pair(pair: str, data_path: str, config: dict = None) -> dict:
    """Train intelligent models for a specific pair."""
    bot_logger.info(f"\n{'=' * 50}")
    bot_logger.info(f"Training IntelligentTrader for {pair}")
    bot_logger.info(f"Data: {data_path}")
    bot_logger.info(f"{'=' * 50}\n")
    
    # Load data
    df = load_data(data_path)
    bot_logger.info(f"Loaded {len(df)} rows of data")
    bot_logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Initialize trader
    trader = IntelligentTrader(config=config, auto_train=False)
    
    # Train
    start_time = datetime.now()
    result = trader.train(df, pair, force=True)
    train_time = (datetime.now() - start_time).total_seconds()
    
    bot_logger.info(f"\nTraining completed in {train_time:.1f}s")
    bot_logger.info(f"Status: {result.get('status')}")
    bot_logger.info(f"Samples used: {result.get('samples')}")
    
    if result.get('model_results'):
        for model_name, metrics in result['model_results'].items():
            if isinstance(metrics, dict):
                acc = metrics.get('accuracy', 'N/A')
                bot_logger.info(f"  {model_name}: accuracy={acc}")
    
    # Save models
    trader.save_models()
    
    # Run quick backtest
    bot_logger.info("\nRunning backtest...")
    backtest_result = trader.backtest_strategy(df, pair)
    
    perf = backtest_result.get('performance', {})
    if perf:
        bot_logger.info(f"Backtest Results:")
        bot_logger.info(f"  Total trades: {perf.get('total_trades', 0)}")
        bot_logger.info(f"  Win rate: {perf.get('win_rate', 0):.1%}")
        bot_logger.info(f"  Total P/L: {perf.get('profit_pct', 0):.2f}%")
        bot_logger.info(f"  Sharpe ratio: {perf.get('sharpe_ratio', 0):.2f}")
    
    return {
        'pair': pair,
        'training': result,
        'backtest': backtest_result,
        'train_time': train_time
    }


def train_all(data_dir: str = 'data', config: dict = None):
    """Train models for all available data files."""
    data_path = Path(data_dir)
    results = []
    
    # Find all 1m data files
    pattern = '*_1m.csv'
    data_files = list(data_path.glob(pattern))
    
    if not data_files:
        bot_logger.warning(f"No data files matching {pattern} found in {data_dir}")
        return results
    
    bot_logger.info(f"Found {len(data_files)} data files to train on")
    
    for data_file in data_files:
        # Extract pair from filename (e.g., MES_1m.csv -> MES)
        pair = data_file.stem.replace('_1m', '')
        
        try:
            result = train_pair(pair, str(data_file), config)
            results.append(result)
        except Exception as e:
            bot_logger.error(f"Failed to train {pair}: {e}")
            results.append({'pair': pair, 'error': str(e)})
    
    return results


def analyze_features(data_path: str):
    """Analyze feature importance from a data file."""
    df = load_data(data_path)
    
    bot_logger.info("Generating features...")
    fg = FeatureGenerator()
    features_df = fg.generate_features(df)
    
    bot_logger.info(f"Generated {len(features_df.columns)} features")
    bot_logger.info("\nTop features by variance:")
    
    # Calculate variance of each feature
    variance = features_df.var().sort_values(ascending=False)
    for feat, var in variance.head(20).items():
        bot_logger.info(f"  {feat}: {var:.6f}")
    
    return features_df


def main():
    parser = argparse.ArgumentParser(description='Train Intelligent Trading Models')
    parser.add_argument('--pair', type=str, help='Trading pair to train (e.g., MES, EUR_USD)')
    parser.add_argument('--data', type=str, help='Path to OHLCV data CSV')
    parser.add_argument('--all', action='store_true', help='Train all available pairs')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--analyze', action='store_true', help='Analyze features instead of training')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory with data files')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.analyze:
        if not args.data:
            print("Error: --data required for feature analysis")
            return 1
        analyze_features(args.data)
        return 0
    
    if args.all:
        results = train_all(args.data_dir, config)
        
        # Summary
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        for r in results:
            pair = r.get('pair')
            if r.get('error'):
                print(f"  {pair}: FAILED - {r['error']}")
            else:
                perf = r.get('backtest', {}).get('performance', {})
                wr = perf.get('win_rate', 0) * 100
                print(f"  {pair}: OK - Win Rate: {wr:.1f}%")
        return 0
    
    if args.pair and args.data:
        train_pair(args.pair, args.data, config)
        return 0
    
    # Default: show usage
    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
