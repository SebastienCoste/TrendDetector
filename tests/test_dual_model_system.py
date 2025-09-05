#!/usr/bin/env python3
"""
Comprehensive tests for the dual-model system
"""

import pytest
import numpy as np
import json
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.config import AppConfig
from src.ml.model_interface import TrendModelInterface
from src.ml.adaptive_classifier import AdaptiveTrendClassifier
from src.ml.adaptive_regressor import AdaptiveTrendRegressor
from src.ml.evaluation import ModelEvaluator

class TestModelInterface:
    """Test the unified model interface"""
    
    def test_classification_interface(self):
        """Test classification model through interface"""
        config = {
            'n_trees': 5,
            'embedding_dim': 10,
            'memory_size': 100
        }
        
        interface = TrendModelInterface("classification", config)
        
        assert interface.model_type == "classification"
        assert interface.embedding_dim == 10
        assert isinstance(interface.model, AdaptiveTrendClassifier)
    
    def test_regression_interface(self):
        """Test regression model through interface"""
        config = {
            'n_trees': 5,
            'embedding_dim': 10,
            'memory_size': 100
        }
        
        interface = TrendModelInterface("regression", config)
        
        assert interface.model_type == "regression"
        assert interface.embedding_dim == 10
        assert isinstance(interface.model, AdaptiveTrendRegressor)
    
    def test_invalid_model_type(self):
        """Test error handling for invalid model type"""
        config = {'embedding_dim': 10}
        
        with pytest.raises(ValueError):
            TrendModelInterface("invalid_type", config)
    
    def test_classification_prediction(self):
        """Test classification prediction through interface"""
        config = {'embedding_dim': 10, 'n_trees': 3}
        interface = TrendModelInterface("classification", config)
        
        # Generate test data
        vectors = [np.random.randn(10) for _ in range(20)]
        trends = ['upward', 'downward', 'neutral'] * 7 + ['upward']
        timestamps = list(range(20))
        
        # Train model
        interface.fit_initial(vectors, trends, timestamps)
        
        # Make prediction
        test_vector = np.random.randn(10)
        result = interface.predict(test_vector)
        
        assert hasattr(result, 'predicted_trend')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'probabilities')
        assert result.predicted_trend in ['upward', 'downward', 'neutral']
    
    def test_regression_prediction(self):
        """Test regression prediction through interface"""
        config = {'embedding_dim': 10, 'n_trees': 3}
        interface = TrendModelInterface("regression", config)
        
        # Generate test data
        vectors = [np.random.randn(10) for _ in range(20)]
        scores = [np.random.uniform(-1, 1) for _ in range(20)]
        timestamps = list(range(20))
        
        # Train model
        interface.fit_initial(vectors, scores, timestamps)
        
        # Make prediction
        test_vector = np.random.randn(10)
        result = interface.predict(test_vector)
        
        assert hasattr(result, 'predicted_score')
        assert hasattr(result, 'confidence')
        assert -1 <= result.predicted_score <= 1

class TestAdaptiveRegressor:
    """Test the regression model specifically"""
    
    def test_regressor_initialization(self):
        """Test regressor initialization"""
        regressor = AdaptiveTrendRegressor(
            n_trees=5,
            embedding_dim=10
        )
        
        assert regressor.embedding_dim == 10
        assert not regressor.is_fitted
        assert regressor.prediction_count == 0
    
    def test_score_normalization(self):
        """Test score normalization"""
        regressor = AdaptiveTrendRegressor(embedding_dim=10)
        
        # Test normal values
        assert -1 <= regressor._normalize_score(0.5) <= 1
        assert -1 <= regressor._normalize_score(-0.5) <= 1
        
        # Test extreme values
        assert regressor._normalize_score(100) < 1
        assert regressor._normalize_score(-100) > -1
        
        # Test infinite/NaN values
        assert regressor._normalize_score(float('inf')) == 0.0
        assert regressor._normalize_score(float('-inf')) == 0.0
        assert regressor._normalize_score(float('nan')) == 0.0
    
    def test_regressor_training(self):
        """Test regressor training"""
        regressor = AdaptiveTrendRegressor(embedding_dim=5, n_trees=3)
        
        # Generate training data
        vectors = [np.random.randn(5) for _ in range(10)]
        scores = [np.random.uniform(-1, 1) for _ in range(10)]
        timestamps = list(range(10))
        
        regressor.fit_initial(vectors, scores, timestamps)
        
        assert regressor.is_fitted
        assert regressor.trend_memory.trend_history
    
    def test_regressor_stats(self):
        """Test regressor statistics"""
        regressor = AdaptiveTrendRegressor(embedding_dim=5, n_trees=3)
        
        # Generate training data
        vectors = [np.random.randn(5) for _ in range(10)]
        scores = [0.5, -0.5, 0.1, -0.1, 0.0] * 2
        timestamps = list(range(10))
        
        regressor.fit_initial(vectors, scores, timestamps)
        stats = regressor.get_model_stats()
        
        assert hasattr(stats, 'mae')
        assert hasattr(stats, 'rmse')
        assert hasattr(stats, 'r2')
        assert hasattr(stats, 'score_distribution')
        assert stats.is_fitted

class TestModelEvaluator:
    """Test the evaluation metrics system"""
    
    def test_classification_evaluator(self):
        """Test classification evaluation metrics"""
        evaluator = ModelEvaluator("classification", {
            'classification_metrics': ['accuracy', 'precision'],
            'primary_classification_metric': 'accuracy'
        })
        
        y_true = ['upward', 'downward', 'neutral', 'upward']
        y_pred = ['upward', 'neutral', 'neutral', 'upward']
        
        results = evaluator.evaluate(y_true, y_pred)
        
        assert 'accuracy' in results
        assert 'precision' in results
        assert 0 <= results['accuracy'] <= 1
        assert 0 <= results['precision'] <= 1
    
    def test_regression_evaluator(self):
        """Test regression evaluation metrics"""
        evaluator = ModelEvaluator("regression", {
            'regression_metrics': ['mae', 'rmse', 'r2'],
            'primary_regression_metric': 'mae'
        })
        
        y_true = [0.1, 0.5, -0.3, 0.8]
        y_pred = [0.2, 0.4, -0.2, 0.7]
        
        results = evaluator.evaluate(y_true, y_pred)
        
        assert 'mae' in results
        assert 'rmse' in results
        assert 'r2' in results
        assert results['mae'] >= 0
        assert results['rmse'] >= 0
    
    def test_primary_metric_selection(self):
        """Test primary metric value extraction"""
        evaluator = ModelEvaluator("regression", {
            'regression_metrics': ['mae', 'r2'],
            'primary_regression_metric': 'mae'
        })
        
        results = {'mae': 0.1, 'r2': 0.8}
        primary_value = evaluator.get_primary_metric_value(results)
        
        assert primary_value == 0.1
    
    def test_score_comparison(self):
        """Test score comparison logic"""
        # MAE evaluator (lower is better)
        mae_evaluator = ModelEvaluator("regression", {
            'primary_regression_metric': 'mae'
        })
        
        assert mae_evaluator.is_better_score(0.1, 0.2)  # Lower MAE is better
        assert not mae_evaluator.is_better_score(0.2, 0.1)
        
        # Accuracy evaluator (higher is better)
        acc_evaluator = ModelEvaluator("classification", {
            'primary_classification_metric': 'accuracy'
        })
        
        assert acc_evaluator.is_better_score(0.9, 0.8)  # Higher accuracy is better
        assert not acc_evaluator.is_better_score(0.8, 0.9)

class TestConfiguration:
    """Test configuration system for dual models"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = AppConfig()
        
        assert hasattr(config.model_settings, 'type')
        assert hasattr(config.evaluation_config, 'regression_metrics')
        assert hasattr(config.evaluation_config, 'classification_metrics')
        
        assert config.model_settings.type in ['classification', 'regression']
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = AppConfig()
        
        # Check evaluation config
        assert 'mae' in config.evaluation_config.regression_metrics
        assert 'accuracy' in config.evaluation_config.classification_metrics
        
        # Check model config
        assert isinstance(config.model_settings.output_range, list)
        assert len(config.model_settings.output_range) == 2

class TestDataGeneration:
    """Test synthetic data generation for both model types"""
    
    def test_classification_data_generation(self):
        """Test classification data generation"""
        from scripts.generate_synthetic_data import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator(seed=42)
        vectors, targets, timestamps, velocity_features = generator.generate_dataset(
            n_samples=10,
            time_span_days=1,
            embedding_dim=5,
            model_type="classification"
        )
        
        assert len(vectors) == 10
        assert len(targets) == 10
        assert all(isinstance(target, str) for target in targets)
        assert all(target in ['upward', 'downward', 'neutral'] for target in targets)
    
    def test_regression_data_generation(self):
        """Test regression data generation"""
        from scripts.generate_synthetic_data import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator(seed=42)
        vectors, targets, timestamps, velocity_features = generator.generate_dataset(
            n_samples=10,
            time_span_days=1,
            embedding_dim=5,
            model_type="regression"
        )
        
        assert len(vectors) == 10
        assert len(targets) == 10
        assert all(isinstance(target, (int, float)) for target in targets)
        assert all(-1 <= target <= 1 for target in targets)
    
    def test_score_category_conversion(self):
        """Test score to category conversion"""
        from scripts.generate_synthetic_data import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator()
        
        assert generator.score_to_category(0.5) == "upward"
        assert generator.score_to_category(-0.5) == "downward"
        assert generator.score_to_category(0.1) == "neutral"
        assert generator.score_to_category(-0.1) == "neutral"

if __name__ == "__main__":
    pytest.main([__file__])