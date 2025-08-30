"""
Performance tests for TrendDetector system
"""
import pytest
import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.ml.adaptive_classifier import AdaptiveTrendClassifier
from src.core.gpu_utils import GPUManager

class TestPerformanceBenchmarks:
    def setup_method(self):
        self.embedding_dim = 100
        self.classifier = AdaptiveTrendClassifier(
            n_trees=5,
            embedding_dim=self.embedding_dim,
            memory_size=1000
        )
    def test_prediction_throughput(self):
        vectors = [np.random.rand(self.embedding_dim) for _ in range(100)]
        trends = np.random.choice(['upward', 'downward', 'neutral'], 100).tolist()
        timestamps = [time.time() + i for i in range(100)]
        self.classifier.fit_initial(vectors, trends, timestamps)
        test_vectors = [np.random.rand(self.embedding_dim) for _ in range(200)]
        start_time = time.time()
        for vector in test_vectors:
            _ = self.classifier.predict_trend(vector)
        elapsed = time.time() - start_time
        throughput = len(test_vectors) / elapsed
        assert throughput > 10

if __name__ == "__main__":
    pytest.main([__file__])
