import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from ..ml.model_interface import TrendModelInterface
from ..ml.adaptive_classifier import AdaptiveTrendClassifier
from ..ml.adaptive_regressor import AdaptiveTrendRegressor
from .config import AppConfig

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages multiple model versions and instances"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.models: Dict[str, TrendModelInterface] = {}
        self.model_versions: Dict[str, List[str]] = {}
        
        # Create model storage directory
        self.model_path = Path(config.storage_config.model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Create type-specific directories
        (self.model_path / "classifier").mkdir(exist_ok=True)
        (self.model_path / "regressor").mkdir(exist_ok=True)
        
        logger.info(f"ModelManager initialized with storage path: {self.model_path}")
        logger.info(f"Default model type: {config.model_settings.type}")

    def create_model(self, model_name: str, model_type: Optional[str] = None, version: Optional[str] = None) -> TrendModelInterface:
        """Create a new model instance"""
        if model_type is None:
            model_type = self.config.model_settings.type
            
        if version is None:
            version = self.config.storage_config.version_format.format(
                len(self.model_versions.get(model_name, [])) + 1
            )
        
        # Prepare model configuration
        model_config = {
            'n_trees': self.config.model_settings.n_trees,
            'drift_threshold': self.config.model_settings.drift_threshold,
            'memory_size': self.config.model_settings.memory_size,
            'max_features': self.config.model_settings.max_features,
            'update_frequency': self.config.model_settings.update_frequency,
            'embedding_dim': self.config.model_settings.embedding_dim,
            'max_clusters': self.config.model_settings.max_clusters,
            'time_decay_hours': self.config.model_settings.time_decay_hours,
            'output_range': self.config.model_settings.output_range,
            'model_version': version
        }
        
        model = TrendModelInterface(model_type, model_config)
        
        self.models[model_name] = model
        
        if model_name not in self.model_versions:
            self.model_versions[model_name] = []
        if version not in self.model_versions[model_name]:
            self.model_versions[model_name].append(version)
        
        logger.info(f"Created {model_type} model {model_name} version {version}")
        return model

    def load_model(self, model_name: str, model_type: Optional[str] = None, filepath: Optional[str] = None) -> bool:
        """Load a model from file"""
        if model_type is None:
            model_type = self.config.model_settings.type
            
        if filepath is None:
            # Use type-specific directory
            type_dir = "classifier" if model_type == "classification" else "regressor"
            filepath = self.model_path / type_dir / f"{model_name}_latest.pkl"
        
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Model file not found: {filepath}")
                return False
            
            # Create a new model instance
            model = self.create_model(model_name, model_type)
            
            # Load the saved state
            model.load(str(filepath))
            
            logger.info(f"Loaded {model_type} model {model_name} from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def save_model(self, model_name: str, filepath: Optional[str] = None) -> bool:
        """Save a model to file"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return False
        
        if filepath is None:
            filepath = self.model_path / f"{model_name}_latest.pkl"
        
        try:
            model = self.models[model_name]
            model.save_model(str(filepath))
            
            # Also save with timestamp for versioning
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            versioned_filepath = self.model_path / f"{model_name}_{timestamp}.pkl"
            model.save_model(str(versioned_filepath))
            
            logger.info(f"Saved model {model_name} to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
            return False

    def get_classifier(self, model_name: str) -> AdaptiveTrendClassifier:
        """Get classifier instance"""
        if model_name not in self.models:
            # Try to load existing model or create new one
            if not self.load_model(model_name):
                logger.info(f"Creating new model {model_name}")
                self.create_model(model_name)
        
        return self.models[model_name]

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if model is loaded"""
        return model_name in self.models

    def get_available_versions(self, model_name: str) -> List[str]:
        """Get available versions for a model"""
        return self.model_versions.get(model_name, [])

    def is_version_available(self, model_name: str, version: str) -> bool:
        """Check if specific version is available"""
        return version in self.get_available_versions(model_name)

    def load_model_version(self, model_name: str, version: str) -> bool:
        """Load specific model version"""
        filepath = self.model_path / f"{model_name}_{version}.pkl"
        return self.load_model(model_name, str(filepath))

    def is_version_loaded(self, model_name: str, version: str) -> bool:
        """Check if specific version is loaded"""
        if model_name not in self.models:
            return False
        return self.models[model_name].model_version == version

    def load_latest_model(self, model_name: str) -> bool:
        """Load the latest version of a model"""
        return self.load_model(model_name)

# Global model manager instance
model_manager: Optional[ModelManager] = None

def initialize_model_manager(config: AppConfig) -> ModelManager:
    """Initialize global model manager"""
    global model_manager
    model_manager = ModelManager(config)
    return model_manager

def get_model_manager() -> ModelManager:
    """Get global model manager instance"""
    if model_manager is None:
        raise RuntimeError("Model manager not initialized")
    return model_manager