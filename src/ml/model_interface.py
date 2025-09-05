import numpy as np
from typing import Union, Dict, List, Any, Optional
import logging

from .adaptive_classifier import AdaptiveTrendClassifier, PredictionResult
from .adaptive_regressor import AdaptiveTrendRegressor, RegressionResult

logger = logging.getLogger(__name__)

class TrendModelInterface:
    """Unified interface for both classification and regression trend models"""
    
    def __init__(self, model_type: str, config: Dict[str, Any]):
        self.model_type = model_type.lower()
        
        if self.model_type == "classification":
            self.model = AdaptiveTrendClassifier(
                n_trees=config.get('n_trees', 10),
                drift_threshold=config.get('drift_threshold', 0.01),
                memory_size=config.get('memory_size', 10000),
                max_features=config.get('max_features', 0.6),
                update_frequency=config.get('update_frequency', 1000),
                embedding_dim=config.get('embedding_dim', 512),
                max_clusters=config.get('max_clusters', 20),
                time_decay_hours=config.get('time_decay_hours', 24),
                model_version=config.get('model_version', 'v1')
            )
        elif self.model_type == "regression":
            self.model = AdaptiveTrendRegressor(
                n_trees=config.get('n_trees', 10),
                drift_threshold=config.get('drift_threshold', 0.01),
                memory_size=config.get('memory_size', 10000),
                max_features=config.get('max_features', 0.6),
                update_frequency=config.get('update_frequency', 1000),
                embedding_dim=config.get('embedding_dim', 512),
                max_clusters=config.get('max_clusters', 20),
                time_decay_hours=config.get('time_decay_hours', 24),
                output_range=config.get('output_range', [-1, 1]),
                model_version=config.get('model_version', 'v1')
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. Must be 'classification' or 'regression'")
            
        logger.info(f"TrendModelInterface initialized with model type: {self.model_type}")

    def predict(self, features: np.ndarray) -> Union[PredictionResult, RegressionResult]:
        """Unified prediction interface"""
        if self.model_type == "classification":
            return self.model.predict_trend(features)
        else:  # regression
            return self.model.predict_trend_score(features)

    def learn(self, features: np.ndarray, target: Union[str, float], 
              predicted_value: Optional[Union[str, float]] = None,
              timestamp: Optional[float] = None) -> bool:
        """Unified learning interface"""
        if self.model_type == "classification":
            return self.model.update_with_feedback(
                content_vector=features,
                actual_trend=target,
                predicted_trend=predicted_value,
                timestamp=timestamp
            )
        else:  # regression
            return self.model.update_with_feedback(
                content_vector=features,
                actual_score=target,
                predicted_score=predicted_value,
                timestamp=timestamp
            )

    def fit_initial(self, vectors: List[np.ndarray], targets: List[Union[str, float]], 
                   timestamps: List[float]) -> None:
        """Initial training interface"""
        if self.model_type == "classification":
            self.model.fit_initial(vectors, targets, timestamps)
        else:  # regression
            self.model.fit_initial(vectors, targets, timestamps)

    def save(self, path: str) -> None:
        """Save model to file"""
        self.model.save_model(path)

    def load(self, path: str) -> None:
        """Load model from file"""
        self.model.load_model(path)

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        return self.model.get_model_stats()

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted"""
        return self.model.is_fitted

    @property
    def model_version(self) -> str:
        """Get model version"""
        return self.model.model_version

    @property
    def embedding_dim(self) -> int:
        """Get expected embedding dimension"""
        return self.model.embedding_dim

    def get_model_type(self) -> str:
        """Get the model type"""
        return self.model_type