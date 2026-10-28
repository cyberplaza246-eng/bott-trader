"""
Intelligent Analyzer Module — Intelligent Trading System

Main analyzer that orchestrates feature generation, ML prediction, and signal generation:
  - Manages state and data window
  - Coordinates feature generation
  - Runs ML predictions
  - Generates trading signals
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .feature_generator import FeatureGenerator
from .label_generator import LabelGenerator
from .ml_classifiers import NeuralNetClassifier, GradientBoostClassifier, SVCClassifier, EnsembleClassifier
from .signal_generator import SignalGenerator
from .model_store import ModelStore

from src.utils.logger import bot_logger


class IntelligentAnalyzer:
    """
    Intelligent analyzer for real-time trading signals.
    
    Combines:
      - Advanced feature generation
      - Multi-model ML predictions  
      - Intelligent signal generation
      - Adaptive learning from outcomes
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize components
        self.feature_gen = FeatureGenerator()
        self.label_gen = LabelGenerator()
        self.signal_gen = SignalGenerator(config.get('signal_config', {}))
        self.model_store = ModelStore()
        
        # Configuration
        self.train_features = self.config.get('train_features', [])
        self.labels = self.config.get('labels', ['high_10', 'low_10'])
        self.feature_sets = self.config.get('feature_sets', [])
        self.signal_sets = self.config.get('signal_sets', [])
        
        # State tracking
        self.df = None
        self.last_analysis_time = None
        self.models = {}
        self.predictions = {}
        
        # Default feature horizon (lookback requirement)
        self.features_horizon = self.config.get('features_horizon', 200)
        self.min_window_length = self.features_horizon
        
        # Initialize classifiers
        self._init_classifiers()
        
    def _init_classifiers(self):
        """Initialize ML classifiers based on configuration."""
        algorithms = self.config.get('algorithms', [])
        
        if not algorithms:
            # Default ensemble
            algorithms = [
                {'name': 'gb', 'algo': 'gb'},
                {'name': 'svc', 'algo': 'svc'},
            ]
            
        for algo_config in algorithms:
            name = algo_config.get('name', algo_config.get('algo'))
            algo_type = algo_config.get('algo', 'gb')
            params = algo_config.get('params', {})
            
            if algo_type == 'nn':
                self.models[name] = NeuralNetClassifier(params)
            elif algo_type == 'gb':
                self.models[name] = GradientBoostClassifier(params)
            elif algo_type == 'svc':
                self.models[name] = SVCClassifier(params)
            elif algo_type == 'ensemble':
                self.models[name] = EnsembleClassifier(params)
                
        bot_logger.info(f"Initialized {len(self.models)} classifiers: {list(self.models.keys())}")
        
    def analyze(
        self,
        df: pd.DataFrame,
        pair: str = None,
        timeframe: str = '1m'
    ) -> Dict[str, Any]:
        """
        Perform full analysis pipeline on data.
        
        Args:
            df: OHLCV DataFrame
            pair: Trading pair symbol
            timeframe: Data timeframe
            
        Returns:
            Analysis result with signals
        """
        if df is None or len(df) < self.min_window_length:
            return {
                'signal': 'SKIP',
                'confidence': 0.0,
                'reason': 'Insufficient data'
            }
            
        try:
            df = df.copy()
            
            # Step 1: Generate features
            feature_sets = self.feature_sets or self.feature_gen.get_default_feature_sets(timeframe)
            df, feature_cols = self.feature_gen.generate_features(df, feature_sets)
            
            # Update train_features if empty
            if not self.train_features:
                self.train_features = feature_cols[:20]  # Limit to top 20
                
            # Step 2: Get latest row with features
            last_row = df.iloc[-1]
            
            # Step 3: Run predictions from each model
            predictions = {}
            scores = {}
            
            for name, classifier in self.models.items():
                if not classifier.is_trained:
                    # Try to load pre-trained model
                    stored = self.model_store.get_model(f"{pair}_{name}" if pair else name)
                    if stored and stored.get('model'):
                        classifier.model = stored['model']
                        classifier.scaler = stored.get('scaler')
                        classifier.is_trained = True
                        
                if classifier.is_trained:
                    try:
                        # Prepare features
                        features_df = df[self.train_features].dropna().tail(1)
                        
                        if len(features_df) > 0:
                            proba = classifier.predict_proba(features_df)
                            score = proba.iloc[-1] if len(proba) > 0 else 0.5
                            
                            predictions[name] = 'BUY' if score > 0.6 else ('SELL' if score < 0.4 else 'HOLD')
                            scores[name] = score
                    except Exception as e:
                        bot_logger.debug(f"Prediction error for {name}: {e}")
                        
            # Step 4: Combine predictions
            if scores:
                avg_score = np.mean(list(scores.values()))
                
                # Convert to trade direction
                if avg_score > 0.55:
                    signal = 'BUY'
                    confidence = (avg_score - 0.5) * 2  # Scale to 0-1
                elif avg_score < 0.45:
                    signal = 'SELL'
                    confidence = (0.5 - avg_score) * 2
                else:
                    signal = 'HOLD'
                    confidence = 0.0
            else:
                # Fallback to technical analysis
                signal = 'HOLD'
                confidence = 0.0
                avg_score = 0.5
                
            # Step 5: Additional signal validation
            # Use price action confirmation
            price_change = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100
            
            # Confirm signal with recent momentum
            if signal == 'BUY' and price_change < -1:
                confidence *= 0.7  # Reduce confidence if price falling
            elif signal == 'SELL' and price_change > 1:
                confidence *= 0.7  # Reduce confidence if price rising
                
            result = {
                'signal': signal,
                'confidence': round(confidence, 4),
                'trade_score': round(avg_score - 0.5, 4),  # Center around 0
                'model_scores': scores,
                'model_predictions': predictions,
                'price': df['close'].iloc[-1],
                'price_change_5bar': round(price_change, 4),
                'timestamp': df.index[-1] if hasattr(df.index[-1], 'isoformat') else str(df.index[-1]),
                'features_used': len(self.train_features),
                'models_used': len(scores)
            }
            
            self.last_analysis_time = datetime.now()
            
            return result
            
        except Exception as e:
            bot_logger.error(f"Analysis error: {e}")
            return {
                'signal': 'SKIP',
                'confidence': 0.0,
                'reason': f'Analysis error: {str(e)}'
            }
            
    def train(
        self,
        df: pd.DataFrame,
        pair: str = None,
        label_horizon: int = 60
    ) -> Dict[str, Any]:
        """
        Train models on historical data.
        
        Args:
            df: Historical OHLCV DataFrame
            pair: Trading pair for model naming
            label_horizon: Future horizon for label generation
            
        Returns:
            Training results
        """
        if len(df) < self.min_window_length + label_horizon:
            return {'error': 'Insufficient data for training'}
            
        try:
            df = df.copy()
            
            # Step 1: Generate features
            feature_sets = self.feature_sets or self.feature_gen.get_default_feature_sets()
            df, feature_cols = self.feature_gen.generate_features(df, feature_sets)
            
            # Step 2: Generate labels
            label_sets = self.config.get('label_sets', [])
            if not label_sets:
                label_sets = self.label_gen.get_default_label_sets(horizon=label_horizon)
            df, label_cols = self.label_gen.generate_labels(df, label_sets, horizon=label_horizon)
            
            # Update config
            self.train_features = feature_cols[:30]  # Top 30 features
            self.labels = label_cols[:4]  # First 4 labels
            
            # Step 3: Prepare training data
            # Remove rows with NaN
            train_df = df[self.train_features + self.labels].dropna()
            
            # Remove last label_horizon rows (no valid labels)
            train_df = train_df.iloc[:-label_horizon]
            
            if len(train_df) < 100:
                return {'error': 'Insufficient valid training data'}
                
            X = train_df[self.train_features]
            
            # Step 4: Train each model for each label
            results = {}
            
            for label in self.labels:
                y = train_df[label]
                
                for name, classifier in self.models.items():
                    try:
                        metrics = classifier.train(X, y)
                        
                        # Save model
                        model_name = f"{pair}_{name}_{label}" if pair else f"{name}_{label}"
                        self.model_store.save_model(
                            model_name=model_name,
                            model=classifier.model,
                            model_type=name,
                            config=classifier.config,
                            metrics=metrics,
                            feature_names=self.train_features,
                            scaler=classifier.scaler
                        )
                        
                        results[f'{name}_{label}'] = metrics
                        
                    except Exception as e:
                        bot_logger.warning(f"Training error for {name}/{label}: {e}")
                        results[f'{name}_{label}'] = {'error': str(e)}
                        
            return {
                'status': 'success',
                'samples': len(train_df),
                'features': len(self.train_features),
                'labels': self.labels,
                'model_results': results
            }
            
        except Exception as e:
            bot_logger.error(f"Training error: {e}")
            return {'error': str(e)}
            
    def train_on_window(
        self,
        df: pd.DataFrame,
        pair: str = None,
        train_size: int = 5000,
        label_horizon: int = 60
    ) -> Dict[str, Any]:
        """
        Incremental training on a rolling window.
        
        Args:
            df: Full DataFrame
            pair: Trading pair
            train_size: Window size for training
            label_horizon: Future horizon for labels
            
        Returns:
            Training results
        """
        # Use most recent data
        if len(df) > train_size + label_horizon:
            df = df.tail(train_size + label_horizon)
            
        return self.train(df, pair, label_horizon)
        
    def get_signal_strength(self, result: Dict) -> str:
        """Convert analysis result to signal strength description."""
        confidence = result.get('confidence', 0)
        signal = result.get('signal', 'HOLD')
        
        if signal == 'BUY':
            if confidence >= 0.7:
                return 'STRONG_BUY'
            elif confidence >= 0.5:
                return 'BUY'
            else:
                return 'WEAK_BUY'
        elif signal == 'SELL':
            if confidence >= 0.7:
                return 'STRONG_SELL'
            elif confidence >= 0.5:
                return 'SELL'
            else:
                return 'WEAK_SELL'
        else:
            return 'NEUTRAL'
            
    def update_with_outcome(
        self,
        prediction: Dict,
        outcome: Dict
    ) -> None:
        """
        Update models based on trade outcome (for adaptive learning).
        
        Args:
            prediction: Original prediction dict
            outcome: Trade outcome dict with 'profit', 'was_correct', etc.
        """
        # This can be used for online learning / model adjustment
        # Implementation depends on specific requirements
        pass
        
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from all trained models."""
        importance = {}
        
        for name, classifier in self.models.items():
            if hasattr(classifier, 'get_feature_importance'):
                importance[name] = classifier.get_feature_importance()
            elif hasattr(classifier, 'get_coefficients'):
                importance[name] = classifier.get_coefficients()
                
        return importance
        
    def save_state(self, path: str = None) -> bool:
        """Save analyzer state to disk."""
        import json
        
        if path is None:
            path = 'intelligent_analyzer_state.json'
            
        state = {
            'train_features': self.train_features,
            'labels': self.labels,
            'config': self.config,
            'last_analysis': str(self.last_analysis_time) if self.last_analysis_time else None
        }
        
        try:
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
            return True
        except Exception as e:
            bot_logger.error(f"Error saving state: {e}")
            return False
            
    def load_state(self, path: str = None) -> bool:
        """Load analyzer state from disk."""
        import json
        
        if path is None:
            path = 'intelligent_analyzer_state.json'
            
        try:
            with open(path, 'r') as f:
                state = json.load(f)
                
            self.train_features = state.get('train_features', [])
            self.labels = state.get('labels', [])
            self.config.update(state.get('config', {}))
            
            return True
        except Exception as e:
            bot_logger.warning(f"Error loading state: {e}")
            return False
