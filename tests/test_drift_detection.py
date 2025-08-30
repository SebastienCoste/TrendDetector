"""
Unit tests for concept drift detection functionality
"""
import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.ml.adaptive_classifier import AdaptiveTrendClassifier

class TestDriftDetection:
    """Test concept drift detection functionality"""

    def setup_method(self):
        self.classifier = AdaptiveTrendClassifier(
            n_trees=5,
            drift_threshold=0.01,
            memory_size=1000,
            embedding_dim=10,
            max_clusters=5,
            time_decay_hours=24
        )

    def test_drift_detection_with_consistent_data(self):
        # No drift if predictions match actuals
        np.random.seed(42)
        vectors = [np.random.rand(10) for _ in range(100)]
        trends = ['upward'] * 50 + ['downward'] * 50
        timestamps = [time.time() + i for i in range(100)]
        self.classifier.fit_initial(vectors, trends, timestamps)

        drift_detected = False
        for i in range(50):
            vector = np.random.rand(10) + 0.1
            predicted = self.classifier.predict_trend(vector)
            drift = self.classifier.update_with_feedback(
                vector,
                predicted.predicted_trend,
                predicted.predicted_trend
            )
            if drift:
                drift_detected = True
                break
        assert not drift_detected

    def test_drift_detection_with_changing_data(self):
        # Drift on consistent error
        np.random.seed(42)
        vectors = [np.random.rand(10) for _ in range(100)]
        trends = ['upward'] * 100
        timestamps = [time.time() + i for i in range(100)]
        self.classifier.fit_initial(vectors, trends, timestamps)

        np.random.seed(123)
        drift_detected = False
        for i in range(200):
            vector = np.random.rand(10) * 5 + 2
            predicted = self.classifier.predict_trend(vector)
            actual_trend = 'downward' if predicted.predicted_trend == 'upward' else 'upward'
            drift = self.classifier.update_with_feedback(
                vector,
                actual_trend,
                predicted.predicted_trend
            )
            if drift:
                drift_detected = True
                break
        assert drift_detected

    def test_model_stats_drift_information(self):
        stats = self.classifier.get_model_stats()
        assert hasattr(stats, 'last_drift_detection')
        assert hasattr(stats, 'accuracy')
        assert hasattr(stats, 'prediction_count')
        assert stats.last_drift_detection == 0

if __name__ == "__main__":
    pytest.main([__file__])
