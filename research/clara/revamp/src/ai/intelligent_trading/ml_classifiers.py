"""
ML Classifiers Module — Intelligent Trading System

Multiple machine learning classifiers for trading predictions:
  - Neural Network (TensorFlow/Keras based)
  - Gradient Boosting (LightGBM based)
  - Support Vector Classifier (scikit-learn)
  - Linear Classifier (Logistic Regression)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from abc import ABC, abstractmethod

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.utils.logger import bot_logger


class BaseClassifier(ABC):
    """Abstract base class for all classifiers."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_names = []
        self.metrics = {}
        
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """Train the model."""
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Get prediction probabilities."""
        pass
        
    def get_params(self) -> Dict:
        """Get model parameters."""
        return self.config.copy()
    
    def set_params(self, **params):
        """Set model parameters."""
        self.config.update(params)
        
    def _scale_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Scale features using StandardScaler."""
        if fit:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X)
        elif self.scaler is not None:
            return self.scaler.transform(X)
        else:
            return X.values
            
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict:
        """Compute classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_proba is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics['auc'] = 0.0
                
        return metrics


class NeuralNetClassifier(BaseClassifier):
    """
    Neural Network Classifier using TensorFlow/Keras.
    
    Features:
      - Configurable hidden layers
      - Dropout regularization
      - Early stopping
      - Learning rate scheduling
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.tf_available = self._check_tensorflow()
        
        # Default configuration
        self.default_config = {
            'layers': [64, 32],
            'activation': 'relu',
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 100,
            'early_stopping_patience': 10,
            'is_scale': True,
            'is_regression': False,
        }
        
        # Merge with provided config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
                
    def _check_tensorflow(self) -> bool:
        """Check if TensorFlow is available."""
        try:
            import tensorflow as tf
            return True
        except ImportError:
            bot_logger.warning("TensorFlow not available for NeuralNetClassifier")
            return False
            
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """Train the neural network."""
        if not self.tf_available:
            return {'error': 'TensorFlow not available'}
            
        import tensorflow as tf
        from tensorflow import keras
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        from keras.optimizers import Adam
        from keras.callbacks import EarlyStopping
        
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self._scale_features(X, fit=True)
        y_values = y.values
        
        # Build model
        n_features = X_scaled.shape[1]
        layers = self.config['layers']
        
        model = Sequential()
        model.add(Dense(layers[0], activation=self.config['activation'], input_dim=n_features))
        model.add(Dropout(self.config['dropout']))
        
        for layer_size in layers[1:]:
            model.add(Dense(layer_size, activation=self.config['activation']))
            model.add(Dropout(self.config['dropout']))
            
        if self.config['is_regression']:
            model.add(Dense(1))
            model.compile(
                loss='mean_squared_error',
                optimizer=Adam(learning_rate=self.config['learning_rate']),
                metrics=['mae']
            )
        else:
            model.add(Dense(1, activation='sigmoid'))
            model.compile(
                loss='binary_crossentropy',
                optimizer=Adam(learning_rate=self.config['learning_rate']),
                metrics=['accuracy']
            )
            
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='loss',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True
        )
        
        # Train
        history = model.fit(
            X_scaled, y_values,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )
        
        self.model = model
        self.is_trained = True
        
        # Compute metrics on training data
        y_proba = model.predict(X_scaled, verbose=0).flatten()
        y_pred = (y_proba >= 0.5).astype(int)
        
        self.metrics = self._compute_metrics(y_values, y_pred, y_proba)
        self.metrics['epochs_trained'] = len(history.history['loss'])
        
        bot_logger.info(f"NN Classifier trained: acc={self.metrics['accuracy']:.3f}, auc={self.metrics.get('auc', 0):.3f}")
        
        return self.metrics
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make binary predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        X_scaled = self._scale_features(X, fit=False)
        y_proba = self.model.predict(X_scaled, verbose=0).flatten()
        y_pred = (y_proba >= 0.5).astype(int)
        
        return pd.Series(y_pred, index=X.index)
        
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        X_scaled = self._scale_features(X, fit=False)
        y_proba = self.model.predict(X_scaled, verbose=0).flatten()
        
        return pd.Series(y_proba, index=X.index)


class GradientBoostClassifier(BaseClassifier):
    """
    Gradient Boosting Classifier using LightGBM.
    
    Features:
      - Fast training on large datasets
      - Handles categorical features
      - Built-in feature importance
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.lgbm_available = self._check_lightgbm()
        
        # Default configuration
        self.default_config = {
            'learning_rate': 0.05,
            'max_depth': 6,
            'num_leaves': 32,
            'num_boost_round': 200,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'is_scale': False,
            'objective': 'binary',
        }
        
        # Merge with provided config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
                
    def _check_lightgbm(self) -> bool:
        """Check if LightGBM is available."""
        try:
            import lightgbm
            return True
        except ImportError:
            bot_logger.warning("LightGBM not available for GradientBoostClassifier")
            return False
            
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """Train the gradient boosting model."""
        if not self.lgbm_available:
            return {'error': 'LightGBM not available'}
            
        import lightgbm as lgbm
        
        self.feature_names = X.columns.tolist()
        
        # Scale if requested
        if self.config['is_scale']:
            X_data = self._scale_features(X, fit=True)
        else:
            X_data = X.values
            
        y_values = y.values
        
        # LightGBM parameters
        lgbm_params = {
            'learning_rate': self.config['learning_rate'],
            'max_depth': self.config['max_depth'],
            'num_leaves': self.config['num_leaves'],
            'min_data_in_leaf': self.config['min_data_in_leaf'],
            'lambda_l1': self.config['lambda_l1'],
            'lambda_l2': self.config['lambda_l2'],
            'objective': self.config['objective'],
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'is_unbalance': True,
            'verbose': -1,
        }
        
        # Train
        train_data = lgbm.Dataset(X_data, y_values)
        
        self.model = lgbm.train(
            lgbm_params,
            train_data,
            num_boost_round=self.config['num_boost_round'],
            valid_sets=[train_data],
        )
        
        self.is_trained = True
        
        # Compute metrics
        y_proba = self.model.predict(X_data)
        y_pred = (y_proba >= 0.5).astype(int)
        
        self.metrics = self._compute_metrics(y_values, y_pred, y_proba)
        
        # Feature importance
        self.feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importance(importance_type='gain')
        ))
        
        bot_logger.info(f"GB Classifier trained: acc={self.metrics['accuracy']:.3f}, auc={self.metrics.get('auc', 0):.3f}")
        
        return self.metrics
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make binary predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        if self.config['is_scale']:
            X_data = self._scale_features(X, fit=False)
        else:
            X_data = X.values
            
        y_proba = self.model.predict(X_data)
        y_pred = (y_proba >= 0.5).astype(int)
        
        return pd.Series(y_pred, index=X.index)
        
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        if self.config['is_scale']:
            X_data = self._scale_features(X, fit=False)
        else:
            X_data = X.values
            
        y_proba = self.model.predict(X_data)
        
        return pd.Series(y_proba, index=X.index)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if hasattr(self, 'feature_importance'):
            return self.feature_importance
        return {}


class SVCClassifier(BaseClassifier):
    """
    Support Vector Classifier using scikit-learn.
    
    Features:
      - Effective for high-dimensional data
      - Probability calibration
      - Multiple kernel options
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Default configuration
        self.default_config = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'is_scale': True,
            'max_iter': 10000,
        }
        
        # Merge with provided config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
                
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """Train the SVC model."""
        from sklearn.svm import SVC
        
        self.feature_names = X.columns.tolist()
        
        # Scale features (almost always needed for SVC)
        if self.config['is_scale']:
            X_data = self._scale_features(X, fit=True)
        else:
            X_data = X.values
            
        y_values = y.values
        
        # Create and train model
        self.model = SVC(
            C=self.config['C'],
            kernel=self.config['kernel'],
            gamma=self.config['gamma'],
            probability=self.config['probability'],
            max_iter=self.config['max_iter'],
        )
        
        self.model.fit(X_data, y_values)
        self.is_trained = True
        
        # Compute metrics
        y_pred = self.model.predict(X_data)
        y_proba = self.model.predict_proba(X_data)[:, 1] if self.config['probability'] else None
        
        self.metrics = self._compute_metrics(y_values, y_pred, y_proba)
        
        bot_logger.info(f"SVC Classifier trained: acc={self.metrics['accuracy']:.3f}, auc={self.metrics.get('auc', 0):.3f}")
        
        return self.metrics
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make binary predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        if self.config['is_scale']:
            X_data = self._scale_features(X, fit=False)
        else:
            X_data = X.values
            
        y_pred = self.model.predict(X_data)
        
        return pd.Series(y_pred, index=X.index)
        
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        if not self.config['probability']:
            raise ValueError("Model not trained with probability=True")
            
        if self.config['is_scale']:
            X_data = self._scale_features(X, fit=False)
        else:
            X_data = X.values
            
        y_proba = self.model.predict_proba(X_data)[:, 1]
        
        return pd.Series(y_proba, index=X.index)


class LinearClassifier(BaseClassifier):
    """
    Logistic Regression Classifier.
    
    Features:
      - Fast training
      - Interpretable coefficients
      - L1/L2 regularization
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Default configuration
        self.default_config = {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'is_scale': True,
        }
        
        # Merge with provided config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
                
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """Train the logistic regression model."""
        from sklearn.linear_model import LogisticRegression
        
        self.feature_names = X.columns.tolist()
        
        # Scale features
        if self.config['is_scale']:
            X_data = self._scale_features(X, fit=True)
        else:
            X_data = X.values
            
        y_values = y.values
        
        # Create and train model
        self.model = LogisticRegression(
            C=self.config['C'],
            penalty=self.config['penalty'],
            solver=self.config['solver'],
            max_iter=self.config['max_iter'],
        )
        
        self.model.fit(X_data, y_values)
        self.is_trained = True
        
        # Compute metrics
        y_pred = self.model.predict(X_data)
        y_proba = self.model.predict_proba(X_data)[:, 1]
        
        self.metrics = self._compute_metrics(y_values, y_pred, y_proba)
        
        # Feature coefficients
        self.feature_coefficients = dict(zip(
            self.feature_names,
            self.model.coef_[0]
        ))
        
        bot_logger.info(f"Linear Classifier trained: acc={self.metrics['accuracy']:.3f}, auc={self.metrics.get('auc', 0):.3f}")
        
        return self.metrics
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make binary predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        if self.config['is_scale']:
            X_data = self._scale_features(X, fit=False)
        else:
            X_data = X.values
            
        y_pred = self.model.predict(X_data)
        
        return pd.Series(y_pred, index=X.index)
        
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        if self.config['is_scale']:
            X_data = self._scale_features(X, fit=False)
        else:
            X_data = X.values
            
        y_proba = self.model.predict_proba(X_data)[:, 1]
        
        return pd.Series(y_proba, index=X.index)
        
    def get_coefficients(self) -> Dict[str, float]:
        """Get feature coefficients."""
        if hasattr(self, 'feature_coefficients'):
            return self.feature_coefficients
        return {}


class EnsembleClassifier(BaseClassifier):
    """
    Ensemble of multiple classifiers with weighted voting.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Default configuration
        self.default_config = {
            'classifiers': ['nn', 'gb', 'svc'],
            'weights': [0.4, 0.4, 0.2],
            'voting': 'soft',  # 'hard' or 'soft'
        }
        
        # Merge with provided config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
                
        self.classifiers = {}
        self._init_classifiers()
        
    def _init_classifiers(self):
        """Initialize individual classifiers."""
        classifier_map = {
            'nn': NeuralNetClassifier,
            'gb': GradientBoostClassifier,
            'svc': SVCClassifier,
            'linear': LinearClassifier,
        }
        
        for clf_name in self.config['classifiers']:
            if clf_name in classifier_map:
                self.classifiers[clf_name] = classifier_map[clf_name]()
                
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """Train all classifiers."""
        self.feature_names = X.columns.tolist()
        
        all_metrics = {}
        
        for name, clf in self.classifiers.items():
            try:
                metrics = clf.train(X, y, **kwargs)
                all_metrics[name] = metrics
            except Exception as e:
                bot_logger.error(f"Error training {name}: {e}")
                
        self.is_trained = True
        self.metrics = all_metrics
        
        return all_metrics
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions using ensemble voting."""
        if not self.is_trained:
            raise ValueError("Ensemble not trained")
            
        predictions = []
        weights = self.config['weights']
        
        for i, (name, clf) in enumerate(self.classifiers.items()):
            if clf.is_trained:
                if self.config['voting'] == 'soft':
                    pred = clf.predict_proba(X)
                else:
                    pred = clf.predict(X)
                predictions.append((pred, weights[i] if i < len(weights) else 1.0))
                
        if not predictions:
            return pd.Series([0] * len(X), index=X.index)
            
        # Weighted average
        total_weight = sum(w for _, w in predictions)
        ensemble_pred = sum(p * w for p, w in predictions) / total_weight
        
        if self.config['voting'] == 'soft':
            final_pred = (ensemble_pred >= 0.5).astype(int)
        else:
            final_pred = (ensemble_pred >= 0.5).astype(int)
            
        return pd.Series(final_pred, index=X.index)
        
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Get ensemble probability predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble not trained")
            
        predictions = []
        weights = self.config['weights']
        
        for i, (name, clf) in enumerate(self.classifiers.items()):
            if clf.is_trained:
                try:
                    pred = clf.predict_proba(X)
                    predictions.append((pred, weights[i] if i < len(weights) else 1.0))
                except:
                    pass
                    
        if not predictions:
            return pd.Series([0.5] * len(X), index=X.index)
            
        total_weight = sum(w for _, w in predictions)
        ensemble_proba = sum(p * w for p, w in predictions) / total_weight
        
        return pd.Series(ensemble_proba, index=X.index)
