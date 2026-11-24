"""
Intelligent Trading System — Advanced ML Integration

Ported and enhanced from intelligent-trading-bot by asavinov.
Provides:
  - Advanced feature generators (TA-Lib, statistics, rolling aggregations)
  - Multiple ML classifiers (Neural Network, Gradient Boosting, SVC)
  - Intelligent signal generation with threshold rules
  - Label generation for training (high/low predictions, top/bottom detection)
  - Backtesting and trade simulation
  - Model store for persistence
"""

from .feature_generator import FeatureGenerator
from .label_generator import LabelGenerator
from .ml_classifiers import NeuralNetClassifier, GradientBoostClassifier, SVCClassifier
from .signal_generator import SignalGenerator
from .backtesting import BacktestEngine
from .model_store import ModelStore
from .analyzer import IntelligentAnalyzer
from .intelligent_trader import IntelligentTrader

__all__ = [
    'FeatureGenerator',
    'LabelGenerator', 
    'NeuralNetClassifier',
    'GradientBoostClassifier',
    'SVCClassifier',
    'SignalGenerator',
    'BacktestEngine',
    'ModelStore',
    'IntelligentAnalyzer',
    'IntelligentTrader',
]
