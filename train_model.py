#!/usr/bin/env python3
"""
Training script for the Trending Content Detection System
"""

import json
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import AppConfig
from src.core.model_manager import initialize_model_manager
from src.core.gpu_utils import initialize_gpu

def main():
    # Load configuration
    config = AppConfig()
    
    # Initialize GPU and model manager
    gpu_manager = initialize_gpu(config.gpu_config.dict())
    model_manager = initialize_model_manager(config)
    
    # Load synthetic data
    data_path = Path('test_data')
    if not data_path.exists():
        print("No test data found. Please run: python scripts/generate_synthetic_data.py")
        return
    
    vectors = np.load(data_path / 'vectors.npy')
    with open(data_path / 'data.json', 'r') as f:
        data = json.load(f)
    
    trends = data['trends']
    timestamps = data['timestamps']
    velocity_features_list = data['velocity_features']
    
    print(f"Loaded {len(vectors)} samples")
    print(f"Trend distribution: {np.unique(trends, return_counts=True)}")
    
    # Get or create classifier
    classifier = model_manager.get_classifier("trend_classifier")
    
    # Initial training
    print("Starting initial training...")
    classifier.fit_initial(
        vectors=[vectors[i] for i in range(len(vectors))],  # Convert to list of arrays
        trends=trends,
        timestamps=timestamps,
        velocity_features_list=velocity_features_list
    )
    
    print("Training completed!")
    
    # Check stats
    stats = classifier.get_model_stats()
    print(f"Is Fitted: {stats.is_fitted}")
    print(f"Memory Size: {stats.memory_size}")
    print(f"Trend Distribution: {stats.trend_distribution}")
    
    # Save model
    model_manager.save_model("trend_classifier")
    print("Model saved!")

if __name__ == "__main__":
    main()