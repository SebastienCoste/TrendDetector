import numpy as np
from typing import Dict, Any, Optional

def validate_embedding_vector(embedding: np.ndarray, expected_dim: int = 512) -> bool:
    """Validate embedding vector format and dimensions"""
    if not isinstance(embedding, np.ndarray):
        raise ValueError("Embedding must be a numpy array")
    
    if embedding.ndim != 1:
        raise ValueError("Embedding must be 1-dimensional")
    
    if embedding.shape[0] != expected_dim:
        raise ValueError(f"Expected embedding dimension {expected_dim}, got {embedding.shape[0]}")
    
    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains non-finite values")
    
    return True

def validate_velocity_features(features: Dict[str, float]) -> bool:
    """Validate velocity features format"""
    required_features = [
        'download_velocity_1h',
        'download_velocity_24h', 
        'like_velocity_1h',
        'like_velocity_24h',
        'dislike_velocity_1h',
        'dislike_velocity_24h',
        'rating_velocity_1h',
        'rating_velocity_24h'
    ]
    
    for feature in required_features:
        if feature not in features:
            raise ValueError(f"Missing required feature: {feature}")
        
        if not isinstance(features[feature], (int, float, np.integer, np.floating)):
            raise ValueError(f"Feature {feature} must be numeric")
        
        if not np.isfinite(features[feature]):
            raise ValueError(f"Feature {feature} contains non-finite value")
    
    return True

def validate_trend_label(trend: str) -> bool:
    """Validate trend label"""
    valid_trends = ['upward', 'downward', 'neutral']
    
    if not isinstance(trend, str):
        raise ValueError("Trend must be a string")
    
    if trend not in valid_trends:
        raise ValueError(f"Invalid trend '{trend}'. Must be one of: {valid_trends}")
    
    return True