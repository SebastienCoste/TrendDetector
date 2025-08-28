import numpy as np
import time
import joblib
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from river import ensemble, metrics, preprocessing, forest
from river.forest import ARFClassifier

from ..core.gpu_utils import get_gpu_manager
from .trend_memory import DynamicTrendMemory

import logging
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    predicted_trend: str
    confidence: float
    probabilities: Dict[str, float]
    method: str
    timestamp: datetime
    model_version: str

@dataclass
class ModelStats:
    accuracy: float
    prediction_count: int
    last_drift_detection: int
    memory_size: int
    is_fitted: bool
    trend_distribution: Dict[str, float]
    last_update: datetime
    model_created: datetime

class AdaptiveTrendClassifier:
    """Enhanced adaptive classifier with GPU acceleration and comprehensive logging"""

    def __init__(self,
                 n_trees: int = 10,
                 drift_threshold: float = 0.01,
                 memory_size: int = 10000,
                 max_features: float = 0.6,
                 update_frequency: int = 1000,
                 embedding_dim: int = 512,
                 max_clusters: int = 20,
                 time_decay_hours: int = 24,
                 model_version: str = "v1"):

        # Core classifier
        self.classifier = AdaptiveRandomForestClassifier(
            n_models=n_trees,
            max_features=max_features,
            random_state=42
        )

        # Performance tracker
        self.accuracy_metric = metrics.Accuracy()

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

        logger.info(f"AdaptiveTrendClassifier initialized with version {model_version}")

    def _validate_input(self, content_vector: np.ndarray) -> None:
        """Validate input vector dimensions"""
        if content_vector.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"got {content_vector.shape[0]}"
            )

    def _prepare_features(self,
                         content_vector: np.ndarray,
                         velocity_features: Optional[Dict[str, float]] = None,
                         contextual_features: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Prepare feature vector for prediction with GPU acceleration"""

        self._validate_input(content_vector)
        features = {}

        # Add embedding features
        for i, val in enumerate(content_vector):
            features[f'embedding_{i}'] = float(val)

        # Add velocity features
        if velocity_features:
            features.update(velocity_features)

        # Add contextual features
        if contextual_features:
            features.update(contextual_features)

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

    def fit_initial(self,
                   vectors: List[np.ndarray],
                   trends: List[str],
                   timestamps: List[float],
                   velocity_features_list: Optional[List[Dict[str, float]]] = None,
                   contextual_features_list: Optional[List[Dict[str, float]]] = None) -> None:
        """Initial training with batch data"""

        if velocity_features_list is None:
            velocity_features_list = [{}] * len(vectors)
        if contextual_features_list is None:
            contextual_features_list = [{}] * len(vectors)

        logger.info(f"Starting initial training with {len(vectors)} samples")

        # Add all data to memory first
        for i, (vector, label, ts) in enumerate(zip(vectors, trends, timestamps)):
            self.trend_memory.add_trend_sample(vector, label, ts)

        # Process each sample for training
        for i, (vector, label) in enumerate(zip(vectors, trends)):
            features = self._prepare_features(
                vector,
                velocity_features_list[i],
                contextual_features_list[i]
            )

            # Learn from this sample
            self.classifier.learn_one(features, label)

        # Update trend boundaries
        self.trend_memory.update_trend_boundaries()
        self.is_fitted = True
        self.last_update = datetime.now()

        logger.info("Initial training completed")

    def predict_trend(self,
                     content_vector: np.ndarray,
                     velocity_features: Optional[Dict[str, float]] = None,
                     contextual_features: Optional[Dict[str, float]] = None) -> PredictionResult:
        """Predict trend for new content"""

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
                return PredictionResult(
                    predicted_trend=most_similar['trend'],
                    confidence=most_similar['similarity'],
                    probabilities={most_similar['trend']: most_similar['similarity']},
                    method='similarity_bootstrap',
                    timestamp=datetime.now(),
                    model_version=self.model_version
                )
            else:
                return PredictionResult(
                    predicted_trend='neutral',
                    confidence=0.0,
                    probabilities={'neutral': 1.0},
                    method='default',
                    timestamp=datetime.now(),
                    model_version=self.model_version
                )

        # Prepare features
        features = self._prepare_features(content_vector, velocity_features, contextual_features)

        # Get prediction with probabilities
        try:
            prediction = self.classifier.predict_one(features)
            proba = self.classifier.predict_proba_one(features)

            confidence = max(proba.values()) if proba else 0.0

            result = PredictionResult(
                predicted_trend=prediction,
                confidence=confidence,
                probabilities=proba,
                method='classifier',
                timestamp=datetime.now(),
                model_version=self.model_version
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Prediction: {prediction}, Confidence: {confidence:.3f}")

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fallback to similarity-based prediction
            similar_trends = self.trend_memory.get_similar_trends(content_vector)
            if similar_trends:
                most_similar = similar_trends[0]
                return PredictionResult(
                    predicted_trend=most_similar['trend'],
                    confidence=most_similar['similarity'],
                    probabilities={most_similar['trend']: most_similar['similarity']},
                    method='similarity_fallback',
                    timestamp=datetime.now(),
                    model_version=self.model_version
                )
            else:
                return PredictionResult(
                    predicted_trend='neutral',
                    confidence=0.0,
                    probabilities={'neutral': 1.0},
                    method='default_fallback',
                    timestamp=datetime.now(),
                    model_version=self.model_version
                )

    def update_with_feedback(self,
                           content_vector: np.ndarray,
                           actual_trend: str,
                           predicted_trend: Optional[str] = None,
                           velocity_features: Optional[Dict[str, float]] = None,
                           contextual_features: Optional[Dict[str, float]] = None,
                           timestamp: Optional[float] = None) -> bool:
        """Update model with actual trend outcome"""

        try:
            self._validate_input(content_vector)
        except ValueError as e:
            logger.error(f"Input validation failed during update: {e}")
            return False

        # Prepare features
        features = self._prepare_features(content_vector, velocity_features, contextual_features)

        # Update classifier
        if self.is_fitted:
            self.classifier.learn_one(features, actual_trend)

        # Update accuracy metric
        if predicted_trend:
            self.accuracy_metric.update(actual_trend, predicted_trend)

        # Simple drift detection based on prediction errors
        drift_detected = False
        if predicted_trend:
            error = 1.0 if predicted_trend != actual_trend else 0.0
            # Simple drift detection: if error rate is high
            if error > 0 and self.prediction_count % 100 == 0:
                recent_accuracy = float(self.accuracy_metric.get())
                if recent_accuracy < 0.7:  # If accuracy drops below 70%
                    drift_detected = True
                    self.drift_count += 1
                    logger.warning(f"Concept drift detected at sample {self.prediction_count}")
                    self.last_drift_detection = self.prediction_count
                    # Update trend boundaries when drift is detected
                    self.trend_memory.update_trend_boundaries()

        # Add to memory
        ts = timestamp if timestamp else time.time()
        self.trend_memory.add_trend_sample(content_vector, actual_trend, ts)

        # Periodically update trend boundaries
        if self.prediction_count % self.update_frequency == 0:
            logger.info("Updating trend boundaries (periodic)")
            self.trend_memory.update_trend_boundaries()

        self.prediction_count += 1
        self.last_update = datetime.now()

        return drift_detected

    def get_model_stats(self) -> ModelStats:
        """Get current model performance statistics"""

        # Calculate trend distribution
        trend_distribution = {'upward': 0, 'downward': 0, 'neutral': 0}
        if self.trend_memory.trend_history:
            total_samples = len(self.trend_memory.trend_history)
            for sample in self.trend_memory.trend_history:
                trend_distribution[sample['trend']] += 1

            # Convert to percentages
            for trend in trend_distribution:
                trend_distribution[trend] /= total_samples

        return ModelStats(
            accuracy=float(self.accuracy_metric.get()),
            prediction_count=self.prediction_count,
            last_drift_detection=self.last_drift_detection,
            memory_size=len(self.trend_memory.trend_history),
            is_fitted=self.is_fitted,
            trend_distribution=trend_distribution,
            last_update=self.last_update,
            model_created=self.model_created
        )

    def save_model(self, filepath: str) -> None:
        """Save model state with comprehensive metadata"""
        model_state = {
            'classifier': self.classifier,
            'accuracy_metric': self.accuracy_metric,
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
                'model_version': self.model_version
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
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load model state with full restoration"""
        model_state = joblib.load(filepath)

        # Restore core components
        self.classifier = model_state['classifier']
        self.accuracy_metric = model_state['accuracy_metric']
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

        # Restore state
        state = model_state['state']
        self.is_fitted = state['is_fitted']
        self.prediction_count = state['prediction_count']
        self.last_drift_detection = state['last_drift_detection']
        self.model_created = datetime.fromisoformat(state['model_created'])
        self.last_update = datetime.fromisoformat(state['last_update'])

        logger.info(f"Model loaded from {filepath}, version {self.model_version}")