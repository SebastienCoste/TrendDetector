#!/usr/bin/env python3
"""
Training script for the Trending Content Detection System
"""

import json

import argparse
import numpy as np
from pathlib import Path
import sys
import os
import logging
from src.core.logging import setup_logging
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import AppConfig
from src.core.model_manager import initialize_model_manager
from src.core.gpu_utils import initialize_gpu

def main(path: str,
         model_name: str,
         ):
    # Load configuration
    config_path = Path("config/config.yaml")
    if config_path.exists():
        config = AppConfig.from_yaml(str(config_path))
    else:
        config = AppConfig()
        config.to_yaml(str(config_path))
        logging.info(f"Created default config at {config_path}")

    setup_logging(config.logging_config)

    # Initialize GPU and model manager
    gpu_manager = initialize_gpu(config.gpu_config)
    logging.info(f"GPU enabled: {gpu_manager.is_gpu_enabled}")
    model_manager = initialize_model_manager(config)
    
    # Load synthetic data
    data_path = Path(path)
    if not data_path.exists():
        print("No training data found. Generate random data from: python scripts/generate_synthetic_data.py")
        return
    
    vectors = np.load(data_path / 'vectors.npy')
    with open(data_path / 'data.json', 'r') as f:
        data = json.load(f)
    
    trends = data['trends']
    timestamps = data['timestamps']
    velocity_features_list = data['velocity_features']

    logging.info(f"Loaded {len(vectors)} samples")
    logging.info(f"Trend distribution: {np.unique(trends, return_counts=True)}")
    
    # Get or create classifier
    classifier = model_manager.get_classifier(model_name)
    
    # Initial training
    logging.info("Starting initial training...")
    classifier.fit_initial(
        vectors=[vectors[i] for i in range(len(vectors))],  # Convert to list of arrays
        trends=trends,
        timestamps=timestamps,
        velocity_features_list=velocity_features_list
    )

    logging.info("Training completed!")
    
    # Check stats
    stats = classifier.get_model_stats()
    logging.info(f"Is Fitted: {stats.is_fitted}")
    logging.info(f"Memory Size: {stats.memory_size}")
    logging.info(f"Trend Distribution: {stats.trend_distribution}")
    
    # Save model
    model_manager.save_model(model_name)
    logging.info("Model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for trend detection")
    parser.add_argument("--data_path", type=str, default="test_data", help="initial data")
    parser.add_argument("--model_name", type=str, default="trend_classifier_poc", help="model name")
    args = parser.parse_args()

    main(
        args.data_path,
        args.model_name,
    )