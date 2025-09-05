import numpy as np
import time
import joblib
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from river import ensemble, metrics, preprocessing, forest
from river.forest import ARFRegressor

from ..core.gpu_utils import get_gpu_manager
from .trend_memory import DynamicTrendMemory

import logging
logger = logging.getLogger(__name__)

@dataclass
class RegressionResult:
    predicted_score: float
    confidence: float
    method: str
    timestamp: datetime
    model_version: str

@dataclass
class RegressionStats:
    mae: float
    rmse: float
    r2: float
    prediction_count: int
    last_drift_detection: int
    memory_size: int
    is_fitted: bool
    score_distribution: Dict[str, float]
    last_update: datetime
    model_created: datetime

class AdaptiveTrendRegressor:
    """Adaptive trend regressor with continuous score output in range [-1, 1]"""

    def __init__(self,
                 n_trees: int = 10,
                 drift_threshold: float = 0.01,
                 memory_size: int = 10000,
                 max_features: float = 0.6,
                 update_frequency: int = 1000,
                 embedding_dim: int = 512,
                 max_clusters: int = 20,
                 time_decay_hours: int = 24,
                 output_range: List[float] = [-1, 1],
                 model_version: str = "v1"):

        # Core regressor
        self.regressor = ARFRegressor(
            n_models=n_trees,
            max_features=max_features,
            seed=42
        )

        # Performance trackers
        self.mae_metric = metrics.MAE()
        self.rmse_metric = metrics.RMSE() 
        self.r2_metric = metrics.R2()

        # Custom trend memory
        self.trend_memory = DynamicTrendMemory(
            max_clusters=max_clusters,
            memory_size=memory_size,
            time_decay_hours=time_decay_hours
        )

        # Feature preprocessing
        self.scaler = preprocessing.StandardScaler()

        # Configuration
        self.embedding_dim = embedding_dim
        self.update_frequency = update_frequency
        self.model_version = model_version
        self.drift_threshold = drift_threshold
        self.output_range = output_range

        # Model state
        self.is_fitted = False
        self.prediction_count = 0
        self.last_drift_detection = 0
        self.model_created = datetime.now()
        self.last_update = datetime.now()
        self.drift_count = 0

        # GPU manager
        try:
            self.gpu_manager = get_gpu_manager()
        except RuntimeError:
            logger.warning("GPU manager not initialized, using CPU only")
            self.gpu_manager = None

        logger.info(f"AdaptiveTrendRegressor initialized with version {model_version}")

    def _validate_input(self, content_vector: np.ndarray) -> None:
        """Validate input vector dimensions"""
        if content_vector.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"got {content_vector.shape[0]}"
            )

    def _prepare_features(self, content_vector: np.ndarray) -> Dict[str, float]:
        """Prepare feature vector for prediction using only embeddings"""
        self._validate_input(content_vector)
        features = {}

        # Add embedding features only (no velocity features)
        for i, val in enumerate(content_vector):
            features[f'embedding_{i}'] = float(val)

        # Add similarity features from memory (GPU accelerated)
        similar_trends = self.trend_memory.get_similar_trends(content_vector)
        if similar_trends:
            trend_counts = {'upward': 0, 'downward': 0, 'neutral': 0}
            total_similarity = 0

            for item in similar_trends:
                trend_counts[item['trend']] += item['similarity']
                total_similarity += item['similarity']

            if total_similarity > 0:
                for trend, count in trend_counts.items():
                    features[f'similar_{trend}_ratio'] = count / total_similarity

        # Add distance to trend centroids
        trend_distances = self.trend_memory.get_trend_distances(content_vector)
        for trend, distance in trend_distances.items():
            features[f'distance_to_{trend}'] = distance

        return features

    def _normalize_score(self, score: float) -> float:
        """Normalize score to output range using tanh"""
        return np.tanh(score)

    def fit_initial(self,
                   vectors: List[np.ndarray],
                   scores: List[float],
                   timestamps: List[float]) -> None:
        """Initial training with batch data"""
        
        logger.info(f"Starting initial training with {len(vectors)} samples")

        # Convert scores to categorical for memory (for backwards compatibility)
        trends = []
        for score in scores:
            if score > 0.3:
                trends.append("upward")
            elif score < -0.3:
                trends.append("downward")
            else:
                trends.append("neutral")

        # Add all data to memory first
        for i, (vector, label, ts) in enumerate(zip(vectors, trends, timestamps)):
            self.trend_memory.add_trend_sample(vector, label, ts)

        # Process each sample for training
        for i, (vector, score) in enumerate(zip(vectors, scores)):
            features = self._prepare_features(vector)
            
            # Learn from this sample
            self.regressor.learn_one(features, score)

        # Update trend boundaries
        self.trend_memory.update_trend_boundaries()
        self.is_fitted = True
        self.last_update = datetime.now()

        logger.info("Initial training completed")

    def predict_trend_score(self, content_vector: np.ndarray) -> RegressionResult:
        """Predict continuous trend score for new content"""

        try:
            self._validate_input(content_vector)
        except ValueError as e:
            logger.error(f"Input validation failed: {e}")
            raise

        if not self.is_fitted:
            # Bootstrap prediction based on memory
            similar_trends = self.trend_memory.get_similar_trends(content_vector)
            if similar_trends:
                most_similar = similar_trends[0]
                # Convert categorical trend to score approximation
                score_map = {'upward': 0.5, 'neutral': 0.0, 'downward': -0.5}
                score = score_map.get(most_similar['trend'], 0.0)
                return RegressionResult(
                    predicted_score=score,
                    confidence=most_similar['similarity'],
                    method='similarity_bootstrap',
                    timestamp=datetime.now(),
                    model_version=self.model_version
                )
            else:
                return RegressionResult(
                    predicted_score=0.0,
                    confidence=0.0,
                    method='default',
                    timestamp=datetime.now(),
                    model_version=self.model_version
                )

        # Prepare features
        features = self._prepare_features(content_vector)

        # Get prediction
        try:
            raw_score = self.regressor.predict_one(features)
            normalized_score = self._normalize_score(raw_score)
            
            # Calculate confidence based on score magnitude
            confidence = min(abs(normalized_score), 1.0)

            result = RegressionResult(
                predicted_score=normalized_score,
                confidence=confidence,
                method='regressor',
                timestamp=datetime.now(),
                model_version=self.model_version
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Prediction score: {normalized_score:.3f}, Confidence: {confidence:.3f}")

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fallback to similarity-based prediction
            similar_trends = self.trend_memory.get_similar_trends(content_vector)
            if similar_trends:
                most_similar = similar_trends[0]
                score_map = {'upward': 0.5, 'neutral': 0.0, 'downward': -0.5}
                score = score_map.get(most_similar['trend'], 0.0)
                return RegressionResult(
                    predicted_score=score,
                    confidence=most_similar['similarity'],
                    method='similarity_fallback',
                    timestamp=datetime.now(),
                    model_version=self.model_version
                )
            else:
                return RegressionResult(
                    predicted_score=0.0,
                    confidence=0.0,
                    method='default_fallback',
                    timestamp=datetime.now(),
                    model_version=self.model_version
                )

    def update_with_feedback(self,
                           content_vector: np.ndarray,
                           actual_score: float,
                           predicted_score: Optional[float] = None,
                           timestamp: Optional[float] = None) -> bool:
        """Update model with actual score outcome"""

        try:
            self._validate_input(content_vector)
        except ValueError as e:
            logger.error(f"Input validation failed during update: {e}")
            return False

        # Prepare features
        features = self._prepare_features(content_vector)

        # Update regressor
        if self.is_fitted:
            self.regressor.learn_one(features, actual_score)

        # Update metrics
        if predicted_score is not None:
            self.mae_metric.update(actual_score, predicted_score)
            self.rmse_metric.update(actual_score, predicted_score)
            self.r2_metric.update(actual_score, predicted_score)

        # Simple drift detection based on prediction errors
        drift_detected = False
        if predicted_score is not None:
            error = abs(predicted_score - actual_score)
            # Simple drift detection: if MAE is high
            if self.prediction_count % 100 == 0:
                recent_mae = float(self.mae_metric.get())
                if recent_mae > 0.5:  # If MAE is high
                    drift_detected = True
                    self.drift_count += 1
                    logger.warning(f"Concept drift detected at sample {self.prediction_count}")
                    self.last_drift_detection = self.prediction_count
                    # Update trend boundaries when drift is detected
                    self.trend_memory.update_trend_boundaries()

        # Add to memory (convert score to categorical for compatibility)
        ts = timestamp if timestamp else time.time()
        if actual_score > 0.3:
            trend_label = "upward"
        elif actual_score < -0.3:
            trend_label = "downward"
        else:
            trend_label = "neutral"
        
        self.trend_memory.add_trend_sample(content_vector, trend_label, ts)

        # Periodically update trend boundaries
        if self.prediction_count % self.update_frequency == 0:
            logger.info("Updating trend boundaries (periodic)")
            self.trend_memory.update_trend_boundaries()

        self.prediction_count += 1
        self.last_update = datetime.now()

        return drift_detected

    def get_model_stats(self) -> RegressionStats:
        """Get current model performance statistics"""

        # Calculate score distribution
        score_distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
        if self.trend_memory.trend_history:
            total_samples = len(self.trend_memory.trend_history)
            for sample in self.trend_memory.trend_history:
                trend = sample['trend']
                if trend == 'upward':
                    score_distribution['positive'] += 1
                elif trend == 'downward':
                    score_distribution['negative'] += 1
                else:
                    score_distribution['neutral'] += 1

            # Convert to percentages
            for category in score_distribution:
                score_distribution[category] /= total_samples

        return RegressionStats(
            mae=float(self.mae_metric.get()),
            rmse=float(self.rmse_metric.get()),
            r2=float(self.r2_metric.get()),
            prediction_count=self.prediction_count,
            last_drift_detection=self.last_drift_detection,
            memory_size=len(self.trend_memory.trend_history),
            is_fitted=self.is_fitted,
            score_distribution=score_distribution,
            last_update=self.last_update,
            model_created=self.model_created
        )

    def save_model(self, filepath: str) -> None:
        """Save model state with comprehensive metadata"""
        model_state = {
            'regressor': self.regressor,
            'mae_metric': self.mae_metric,
            'rmse_metric': self.rmse_metric,
            'r2_metric': self.r2_metric,
            'scaler': self.scaler,
            'trend_memory': {
                'history': list(self.trend_memory.trend_history),
                'centroids': self.trend_memory.trend_centroids,
                'max_clusters': self.trend_memory.max_clusters,
                'memory_size': self.trend_memory.memory_size,
                'time_decay_hours': self.trend_memory.time_decay_hours
            },
            'config': {
                'embedding_dim': self.embedding_dim,
                'update_frequency': self.update_frequency,
                'model_version': self.model_version,
                'output_range': self.output_range
            },
            'state': {
                'is_fitted': self.is_fitted,
                'prediction_count': self.prediction_count,
                'last_drift_detection': self.last_drift_detection,
                'model_created': self.model_created.isoformat(),
                'last_update': self.last_update.isoformat()
            }
        }

        joblib.dump(model_state, filepath)
        logger.info(f"Regression model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load model state with full restoration"""
        model_state = joblib.load(filepath)

        # Restore core components
        self.regressor = model_state['regressor']
        self.mae_metric = model_state['mae_metric']
        self.rmse_metric = model_state['rmse_metric']
        self.r2_metric = model_state['r2_metric']
        self.scaler = model_state['scaler']

        # Restore trend memory
        trend_memory_data = model_state['trend_memory']
        self.trend_memory = DynamicTrendMemory(
            max_clusters=trend_memory_data['max_clusters'],
            memory_size=trend_memory_data['memory_size'],
            time_decay_hours=trend_memory_data['time_decay_hours']
        )

        # Restore history
        for item in trend_memory_data['history']:
            self.trend_memory.trend_history.append(item)

        self.trend_memory.trend_centroids = trend_memory_data['centroids']

        # Restore configuration
        config = model_state['config']
        self.embedding_dim = config['embedding_dim']
        self.update_frequency = config['update_frequency']
        self.model_version = config['model_version']
        self.output_range = config.get('output_range', [-1, 1])

        # Restore state
        state = model_state['state']
        self.is_fitted = state['is_fitted']
        self.prediction_count = state['prediction_count']
        self.last_drift_detection = state['last_drift_detection']
        self.model_created = datetime.fromisoformat(state['model_created'])
        self.last_update = datetime.fromisoformat(state['last_update'])

        logger.info(f"Regression model loaded from {filepath}, version {self.model_version}")