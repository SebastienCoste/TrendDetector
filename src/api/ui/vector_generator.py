import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
import logging

from .models import VectorGenerationConfig, VectorGenerationResult, BasePattern, AlgorithmConfig

logger = logging.getLogger(__name__)

class VectorGenerator:
    """Generates embedding vectors based on configurable algorithms"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_vector(self, config: VectorGenerationConfig) -> VectorGenerationResult:
        """Generate embedding vector based on configuration"""
        
        # Use current time if not provided
        timestamp = config.timestamp or datetime.now()
        
        # Generate base vector using selected algorithm
        vector = self._generate_base_vector(
            config.algorithm_params,
            config.embedding_dim,
            config.trend_score,
            timestamp
        )
        
        # Apply trend correlation
        vector = self._apply_trend_correlation(
            vector,
            config.trend_score,
            config.algorithm_params.embedding_correlation
        )
        
        # Add noise
        if config.algorithm_params.noise_level > 0:
            noise = np.random.normal(0, config.algorithm_params.noise_level, config.embedding_dim)
            vector += noise
        
        # Calculate the actual expected trend from the generated vector
        expected_trend = self._calculate_trend_from_vector(vector, timestamp, config.trend_score)
        
        return VectorGenerationResult(
            vector=vector.tolist(),
            expected_trend=expected_trend,
            timestamp=timestamp,
            algorithm_used=config.algorithm_params.base_pattern.value
        )
    
    def _generate_base_vector(self, algorithm: AlgorithmConfig, embedding_dim: int, 
                             target_trend: float, timestamp: datetime) -> np.ndarray:
        """Generate base vector using selected algorithm"""
        
        if algorithm.base_pattern == BasePattern.LINEAR:
            return self._generate_linear_pattern(embedding_dim, target_trend)
        elif algorithm.base_pattern == BasePattern.SINUSOIDAL:
            return self._generate_sinusoidal_pattern(embedding_dim, target_trend, timestamp, algorithm)
        elif algorithm.base_pattern == BasePattern.EXPONENTIAL:
            return self._generate_exponential_pattern(embedding_dim, target_trend)
        elif algorithm.base_pattern == BasePattern.RANDOM_WALK:
            return self._generate_random_walk_pattern(embedding_dim, target_trend)
        else:
            # Default to random
            return np.random.randn(embedding_dim)
    
    def _generate_linear_pattern(self, embedding_dim: int, target_trend: float) -> np.ndarray:
        """Generate linear pattern correlated with target trend"""
        
        # Create base vector with linear relationship to target
        base_magnitude = abs(target_trend) * 2.0 + 1.0
        direction = 1.0 if target_trend >= 0 else -1.0
        
        # Generate vector with linear progression
        vector = np.linspace(-base_magnitude * direction, base_magnitude * direction, embedding_dim)
        
        # Add some randomness while preserving the linear trend
        random_component = np.random.randn(embedding_dim) * 0.3
        vector += random_component
        
        return vector
    
    def _generate_sinusoidal_pattern(self, embedding_dim: int, target_trend: float, 
                                   timestamp: datetime, algorithm: AlgorithmConfig) -> np.ndarray:
        """Generate sinusoidal pattern with temporal factors"""
        
        # Base frequency and amplitude based on target trend
        base_frequency = 2 * np.pi / embedding_dim
        amplitude = abs(target_trend) * 2.0 + 0.5
        phase_shift = target_trend * np.pi
        
        # Generate index array
        indices = np.arange(embedding_dim)
        
        # Base sinusoidal pattern
        vector = amplitude * np.sin(base_frequency * indices + phase_shift)
        
        # Apply temporal factors
        if algorithm.temporal_factors.hourly:
            hour_factor = np.sin(2 * np.pi * timestamp.hour / 24)
            vector *= (1.0 + 0.3 * hour_factor)
        
        if algorithm.temporal_factors.daily:
            # Day of month effect
            day_factor = np.sin(2 * np.pi * timestamp.day / 31)
            vector *= (1.0 + 0.2 * day_factor)
        
        if algorithm.temporal_factors.weekly:
            # Day of week effect
            week_factor = np.sin(2 * np.pi * timestamp.weekday() / 7)
            vector *= (1.0 + 0.1 * week_factor)
        
        return vector
    
    def _generate_exponential_pattern(self, embedding_dim: int, target_trend: float) -> np.ndarray:
        """Generate exponential pattern"""
        
        # Parameters based on target trend
        if target_trend >= 0:
            # Growing exponential for positive trends
            growth_rate = 0.01 + target_trend * 0.02
            vector = np.exp(growth_rate * np.arange(embedding_dim))
        else:
            # Decaying exponential for negative trends  
            decay_rate = 0.01 - target_trend * 0.02
            vector = np.exp(-decay_rate * np.arange(embedding_dim))
        
        # Normalize to prevent extreme values
        vector = (vector - np.mean(vector)) / np.std(vector)
        
        # Scale by target magnitude
        vector *= (abs(target_trend) * 2.0 + 0.5)
        
        return vector
    
    def _generate_random_walk_pattern(self, embedding_dim: int, target_trend: float) -> np.ndarray:
        """Generate random walk pattern biased toward target trend"""
        
        # Start at zero
        vector = np.zeros(embedding_dim)
        
        # Random walk with bias toward target
        bias = target_trend * 0.1  # Small bias per step
        
        for i in range(1, embedding_dim):
            step = np.random.normal(bias, 0.5)
            vector[i] = vector[i-1] + step
        
        return vector
    
    def _apply_trend_correlation(self, vector: np.ndarray, target_trend: float, 
                               correlation: float) -> np.ndarray:
        """Apply correlation between vector and target trend"""
        
        if correlation <= 0:
            return vector
        
        # Create a trend-correlated component
        trend_component = np.full_like(vector, target_trend)
        
        # Add some variation to the trend component
        variation = np.random.randn(len(vector)) * 0.3
        trend_component += variation
        
        # Blend with correlation factor
        blended_vector = (1 - correlation) * vector + correlation * trend_component
        
        return blended_vector
    
    def _calculate_trend_from_vector(self, vector: np.ndarray, timestamp: datetime, 
                                   target_trend: float) -> float:
        """Calculate expected trend from generated vector"""
        
        # Vector-based features
        vector_sum = np.sum(vector)
        vector_magnitude = np.linalg.norm(vector)
        vector_variance = np.var(vector)
        vector_mean = np.mean(vector)
        
        # Time-based features
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Calculate trend score
        trend_score = 0.0
        
        # Vector influence (weighted by statistics)
        if vector_sum > 0:
            trend_score += min(vector_sum / 100, 0.3)
        else:
            trend_score += max(vector_sum / 100, -0.3)
        
        # Magnitude influence
        if vector_magnitude > 20:
            trend_score += 0.2
        elif vector_magnitude < 10:
            trend_score -= 0.1
        
        # Variance influence (higher variance = more uncertainty)
        if vector_variance > 2.0:
            trend_score += 0.1
        
        # Mean influence
        trend_score += np.tanh(vector_mean) * 0.2
        
        # Time-based influences
        if 9 <= hour_of_day <= 11 or 19 <= hour_of_day <= 21:
            trend_score += 0.1
        elif 2 <= hour_of_day <= 6:
            trend_score -= 0.1
        
        # Weekend effect
        if day_of_week in [5, 6]:  # Saturday, Sunday
            trend_score += 0.05
        
        # Bias toward target (to make the generation more predictable)
        trend_score += target_trend * 0.3
        
        # Add small amount of noise
        trend_score += np.random.normal(0, 0.05)
        
        # Normalize to [-1, 1] range
        return float(np.tanh(trend_score))
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about available algorithms"""
        return {
            "base_patterns": [pattern.value for pattern in BasePattern],
            "temporal_factors": ["hourly", "daily", "weekly"],
            "parameter_ranges": {
                "noise_level": [0.0, 1.0],
                "velocity_influence": [0.0, 1.0],
                "embedding_correlation": [0.0, 1.0],
                "trend_score": [-1.0, 1.0]
            },
            "embedding_dimensions": [512]  # Currently fixed
        }