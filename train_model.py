#!/usr/bin/env python3
"""
Training script for the Trending Content Detection System (Dual Model Support)
"""

import json
import numpy as np
from pathlib import Path
import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import AppConfig
from src.core.model_manager import initialize_model_manager
from src.core.gpu_utils import initialize_gpu

def main():
    parser = argparse.ArgumentParser(description="Train trending content detection model")
    parser.add_argument("--model-type", type=str, 
                        choices=["classification", "regression"],
                        help="Model type to train (overrides config)")
    parser.add_argument("--model-name", type=str, default="trend_model",
                        help="Name of the model")
    parser.add_argument("--data-path", type=str, default="test_data",
                        help="Path to training data")
    
    args = parser.parse_args()
    
    # Load configuration
    config = AppConfig()
    
    # Override model type if specified
    if args.model_type:
        config.model_settings.type = args.model_type
    
    # Initialize GPU and model manager
    gpu_manager = initialize_gpu(config.gpu_config.dict())
    model_manager = initialize_model_manager(config)
    
    # Load synthetic data
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"No test data found at {data_path}. Please run: python scripts/generate_synthetic_data.py")
        return
    
    vectors = np.load(data_path / 'vectors.npy')
    with open(data_path / 'data.json', 'r') as f:
        data = json.load(f)
    
    # Handle both old and new data formats
    data_model_type = data.get('model_type', 'classification')
    targets = data.get('targets', data.get('trends', []))
    timestamps = data['timestamps']
    
    print(f"Loaded {len(vectors)} samples")
    print(f"Data model type: {data_model_type}")
    print(f"Training model type: {config.model_settings.type}")
    
    if config.model_settings.type == "classification":
        if data_model_type == "regression":
            # Convert scores to categories
            print("Converting regression scores to classification labels...")
            categorical_targets = []
            for score in targets:
                if score > 0.3:
                    categorical_targets.append("upward")
                elif score < -0.3:
                    categorical_targets.append("downward")
                else:
                    categorical_targets.append("neutral")
            targets = categorical_targets
            
        print(f"Trend distribution: {np.unique(targets, return_counts=True)}")
    else:  # regression
        if data_model_type == "classification":
            # Convert categories to scores
            print("Converting classification labels to regression scores...")
            score_map = {"upward": 0.5, "neutral": 0.0, "downward": -0.5}
            targets = [score_map.get(trend, 0.0) for trend in targets]
            
        target_array = np.array(targets)
        print(f"Score statistics - Mean: {np.mean(target_array):.3f}, Std: {np.std(target_array):.3f}")
    
    # Get or create model
    model = model_manager.get_model(args.model_name, config.model_settings.type)
    
    # Initial training
    print(f"Starting initial training for {config.model_settings.type} model...")
    model.fit_initial(
        vectors=[vectors[i] for i in range(len(vectors))],  # Convert to list of arrays
        targets=targets,
        timestamps=timestamps
    )
    
    print("Training completed!")
    
    # Check stats
    stats = model.get_stats()
    print(f"Is Fitted: {stats.is_fitted}")
    print(f"Memory Size: {stats.memory_size}")
    
    if config.model_settings.type == "classification":
        print(f"Trend Distribution: {stats.trend_distribution}")
    else:
        print(f"MAE: {stats.mae:.4f}")
        print(f"RMSE: {stats.rmse:.4f}")
        print(f"R²: {stats.r2:.4f}")
    
    # Save model
    model_manager.save_model(args.model_name)
    print(f"{config.model_settings.type.capitalize()} model saved!")

if __name__ == "__main__":
    main()