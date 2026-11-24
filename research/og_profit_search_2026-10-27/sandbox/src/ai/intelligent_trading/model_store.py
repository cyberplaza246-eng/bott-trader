"""
Model Store Module — Intelligent Trading System

Model persistence and management:
  - Save/load trained models
  - Version tracking
  - Model metadata management
"""
import os
import json
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.utils.logger import bot_logger


class ModelStore:
    """
    Model storage and management.
    
    Features:
      - Save and load trained models
      - Track model versions and metadata
      - Automatic model directory management
    """
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'intelligent')
            
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.metadata_file = self.base_path / 'model_registry.json'
        self._load_registry()
        
    def _load_registry(self):
        """Load the model registry from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.registry = json.load(f)
            except Exception as e:
                bot_logger.warning(f"Error loading model registry: {e}")
                self.registry = {'models': {}}
        else:
            self.registry = {'models': {}}
            
    def _save_registry(self):
        """Save the model registry to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.registry, f, indent=2, default=str)
        except Exception as e:
            bot_logger.error(f"Error saving model registry: {e}")
            
    def save_model(
        self,
        model_name: str,
        model,
        model_type: str,
        config: Dict = None,
        metrics: Dict = None,
        feature_names: List[str] = None,
        scaler = None
    ) -> str:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name for the model
            model: The trained model object
            model_type: Type of model ('nn', 'gb', 'svc', etc.)
            config: Model configuration
            metrics: Training metrics
            feature_names: List of feature names used
            scaler: Optional scaler object
            
        Returns:
            Path to saved model
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = self.base_path / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model based on type
        model_path = model_dir / f"model_{timestamp}.pkl"
        
        try:
            if model_type == 'nn':
                # For neural networks, use Keras save format
                try:
                    keras_path = model_dir / f"model_{timestamp}.keras"
                    model.save(keras_path)
                    model_path = keras_path
                except Exception:
                    # Fallback to pickle
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
            else:
                # Use pickle for other models
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                    
        except Exception as e:
            bot_logger.error(f"Error saving model {model_name}: {e}")
            return None
            
        # Save scaler if provided
        scaler_path = None
        if scaler is not None:
            scaler_path = model_dir / f"scaler_{timestamp}.pkl"
            try:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
            except Exception as e:
                bot_logger.warning(f"Error saving scaler: {e}")
                
        # Save metadata
        metadata = {
            'model_name': model_name,
            'model_type': model_type,
            'timestamp': timestamp,
            'model_path': str(model_path),
            'scaler_path': str(scaler_path) if scaler_path else None,
            'config': config or {},
            'metrics': metrics or {},
            'feature_names': feature_names or [],
        }
        
        metadata_path = model_dir / f"metadata_{timestamp}.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            bot_logger.warning(f"Error saving metadata: {e}")
            
        # Update registry
        if model_name not in self.registry['models']:
            self.registry['models'][model_name] = []
            
        self.registry['models'][model_name].append({
            'timestamp': timestamp,
            'model_path': str(model_path),
            'metadata_path': str(metadata_path),
            'metrics': metrics or {}
        })
        
        self._save_registry()
        
        bot_logger.info(f"Model {model_name} saved to {model_path}")
        return str(model_path)
        
    def load_model(
        self,
        model_name: str,
        version: str = 'latest',
        model_type: str = None
    ) -> tuple:
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name of the model to load
            version: Version timestamp or 'latest'
            model_type: Optional model type hint
            
        Returns:
            Tuple of (model, scaler, metadata)
        """
        if model_name not in self.registry['models']:
            bot_logger.warning(f"Model {model_name} not found in registry")
            return None, None, None
            
        versions = self.registry['models'][model_name]
        if not versions:
            return None, None, None
            
        # Get the specified version
        if version == 'latest':
            version_info = versions[-1]
        else:
            matching = [v for v in versions if v['timestamp'] == version]
            if not matching:
                bot_logger.warning(f"Version {version} not found for model {model_name}")
                return None, None, None
            version_info = matching[0]
            
        model_path = Path(version_info['model_path'])
        metadata_path = Path(version_info.get('metadata_path', ''))
        
        # Load metadata
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                bot_logger.warning(f"Error loading metadata: {e}")
                
        # Load model
        model = None
        try:
            if model_path.suffix == '.keras':
                # Load Keras model
                from tensorflow import keras
                model = keras.models.load_model(model_path)
            else:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
        except Exception as e:
            bot_logger.error(f"Error loading model: {e}")
            
        # Load scaler
        scaler = None
        scaler_path = metadata.get('scaler_path')
        if scaler_path and Path(scaler_path).exists():
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            except Exception as e:
                bot_logger.warning(f"Error loading scaler: {e}")
                
        return model, scaler, metadata
        
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a model from cache or load from disk."""
        if model_name in self.models:
            return self.models[model_name]
            
        model, scaler, metadata = self.load_model(model_name)
        if model is not None:
            self.models[model_name] = {
                'model': model,
                'scaler': scaler,
                'metadata': metadata
            }
            return self.models[model_name]
            
        return None
        
    def list_models(self) -> List[Dict]:
        """List all registered models."""
        models = []
        for model_name, versions in self.registry['models'].items():
            if versions:
                latest = versions[-1]
                models.append({
                    'name': model_name,
                    'versions': len(versions),
                    'latest_timestamp': latest['timestamp'],
                    'metrics': latest.get('metrics', {})
                })
        return models
        
    def get_model_versions(self, model_name: str) -> List[Dict]:
        """Get all versions of a model."""
        return self.registry['models'].get(model_name, [])
        
    def delete_model(
        self,
        model_name: str,
        version: str = None
    ) -> bool:
        """
        Delete a model from the store.
        
        Args:
            model_name: Name of the model
            version: Specific version to delete, or None to delete all
            
        Returns:
            True if successful
        """
        if model_name not in self.registry['models']:
            return False
            
        if version is None:
            # Delete all versions
            model_dir = self.base_path / model_name
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
            del self.registry['models'][model_name]
        else:
            # Delete specific version
            versions = self.registry['models'][model_name]
            matching = [v for v in versions if v['timestamp'] == version]
            
            for v in matching:
                # Delete files
                for path_key in ['model_path', 'metadata_path']:
                    path = v.get(path_key)
                    if path and Path(path).exists():
                        Path(path).unlink()
                        
                versions.remove(v)
                
        self._save_registry()
        
        if model_name in self.models:
            del self.models[model_name]
            
        return True
        
    def get_best_model(
        self,
        model_name: str,
        metric: str = 'auc'
    ) -> tuple:
        """
        Get the best performing version of a model.
        
        Args:
            model_name: Name of the model
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model, scaler, metadata)
        """
        if model_name not in self.registry['models']:
            return None, None, None
            
        versions = self.registry['models'][model_name]
        if not versions:
            return None, None, None
            
        # Find best by metric
        best_version = max(
            versions,
            key=lambda v: v.get('metrics', {}).get(metric, 0)
        )
        
        return self.load_model(model_name, version=best_version['timestamp'])
        
    def export_model(
        self,
        model_name: str,
        export_path: str,
        version: str = 'latest'
    ) -> bool:
        """
        Export a model to a specified path.
        
        Args:
            model_name: Name of the model to export
            export_path: Path to export to
            version: Version to export
            
        Returns:
            True if successful
        """
        model, scaler, metadata = self.load_model(model_name, version)
        
        if model is None:
            return False
            
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Export model
            model_file = export_dir / f"{model_name}_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
                
            # Export scaler
            if scaler is not None:
                scaler_file = export_dir / f"{model_name}_scaler.pkl"
                with open(scaler_file, 'wb') as f:
                    pickle.dump(scaler, f)
                    
            # Export metadata
            metadata_file = export_dir / f"{model_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
                
            return True
            
        except Exception as e:
            bot_logger.error(f"Error exporting model: {e}")
            return False
            
    def import_model(
        self,
        import_path: str,
        model_name: str = None
    ) -> bool:
        """
        Import a model from an external path.
        
        Args:
            import_path: Path to import from
            model_name: Optional name override
            
        Returns:
            True if successful
        """
        import_dir = Path(import_path)
        
        # Find model file
        model_files = list(import_dir.glob('*_model.pkl'))
        if not model_files:
            bot_logger.warning("No model file found in import path")
            return False
            
        model_file = model_files[0]
        
        # Derive model name
        if model_name is None:
            model_name = model_file.stem.replace('_model', '')
            
        try:
            # Load model
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
                
            # Load scaler if exists
            scaler = None
            scaler_file = import_dir / f"{model_name}_scaler.pkl"
            if scaler_file.exists():
                with open(scaler_file, 'rb') as f:
                    scaler = pickle.load(f)
                    
            # Load metadata if exists
            metadata = {}
            metadata_file = import_dir / f"{model_name}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
            # Save to store
            self.save_model(
                model_name=model_name,
                model=model,
                model_type=metadata.get('model_type', 'unknown'),
                config=metadata.get('config'),
                metrics=metadata.get('metrics'),
                feature_names=metadata.get('feature_names'),
                scaler=scaler
            )
            
            return True
            
        except Exception as e:
            bot_logger.error(f"Error importing model: {e}")
            return False
