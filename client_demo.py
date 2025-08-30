#!/usr/bin/env python3
"""
Simple test client for the Trending Content Detection System
"""
import argparse
from datetime import datetime, timedelta
import requests
import json
import numpy as np
from pathlib import Path

from scripts.generate_synthetic_data import SyntheticDataGenerator


class TrendClient:
    """Client for interacting with the trend detection API"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        """Check service health"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def get_model_metadata(self, model_name: str):
        """Get model metadata"""
        response = self.session.get(f"{self.base_url}/v2/models/{model_name}")
        return response.json()
    
    def predict_trend(self, embedding_vector: np.ndarray, velocity_features: list, model_name: str):
        """Make a trend prediction"""
        
        inputs = [
            {
                "name": "embedding_vector",
                "shape": [len(embedding_vector)],
                "datatype": "FP32",
                "data": embedding_vector.tolist()
            }
        ]
        
        if velocity_features:
            inputs.append({
                "name": "velocity_features",
                "shape": [len(velocity_features)],
                "datatype": "FP32", 
                "data": velocity_features
            })
        
        request_data = {"inputs": inputs}
        
        response = self.session.post(
            f"{self.base_url}/v2/models/{model_name}/infer",
            json=request_data
        )
        return response.json()
    
    def update_model(self, updates: list, model_name: str):
        """Update model with feedback"""
        request_data = {"updates": updates}
        
        response = self.session.post(
            f"{self.base_url}/v2/models/{model_name}/update",
            json=request_data
        )
        return response.json()
    
    def get_stats(self, model_name: str):
        """Get model statistics"""
        response = self.session.get(f"{self.base_url}/v2/models/{model_name}/stats")
        return response.json()

def demo(model_name, predictions):
    """Run a simple demo"""
    client = TrendClient()

    print("=== Trending Content Detection System Demo ===")
    
    # Check health
    print("\n1. Checking service health...")
    try:
        health = client.health_check()
        print(f"Health Status: {health}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Get model metadata
    print("\n2. Getting model metadata...")
    try:
        metadata = client.get_model_metadata(model_name)
        print(f"Model Platform: {metadata['platform']}")
        print(f"Available Versions: {metadata.get('versions', [])}")
    except Exception as e:
        print(f"Metadata request failed: {e}")
        return
    
    # Make predictions
    correct_predictions = 0
    print(f"\n3. Making {predictions} predictions...")
    
    # Generate some sample data
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    for i in range(predictions):
        # Random embedding vector
        # embedding = np.random.randn(512)
        embedding = rng.standard_normal(512)
        e_sum = np.sum(embedding)
        e_mean = np.mean(embedding)
        e_var = np.var(embedding)
        data_generator = SyntheticDataGenerator()
        inference_time = (datetime.now() - timedelta(days=np.random.uniform(0, 30))).timestamp()
        generated_mean = data_generator.generate_mean(inference_time, embedding)
        trend_from_data_generator = data_generator.calculate_trend_from_vector_and_time(
            embedding,
            inference_time,
            inference_time,
            0.3
        )
        generated_velocity = data_generator.generate_velocity_features(
            trend_from_data_generator,
            inference_time
        )

        velocity_features = [
            generated_velocity.get('download_velocity_1h'),
            generated_velocity.get('download_velocity_24h'),
            generated_velocity.get('like_velocity_1h'),
            generated_velocity.get('like_velocity_24h'),
            generated_velocity.get('dislike_velocity_1h'),
            generated_velocity.get('dislike_velocity_24h'),
            generated_velocity.get('rating_velocity_1h'),
            generated_velocity.get('rating_velocity_24h'),
        ]

        try:
            result = client.predict_trend(embedding, velocity_features, model_name)
            
            # Extract outputs
            if 'outputs' in result:
                predicted_trend = result['outputs'][0]['data'][0]
                confidence = result['outputs'][1]['data'][0]
            else:
                print(f"   Error: {result}")
                continue
            
            print(f"#{i+1} {'YES' if predicted_trend == trend_from_data_generator else 'NOP'}: {predicted_trend} (conf: {confidence:.3f})."
                  f" Truth: {trend_from_data_generator} (mean: {generated_mean:.3f}). E: sum: {e_sum:.3f}, mean: {e_mean:.3f}, var: {e_var:.3f}")
            if trend_from_data_generator == predicted_trend:
                correct_predictions += 1

            # Send feedback from time to time, to inform the model how it performs
            # In reality, it would send feedback everytime a trend is acknowledged
            if i%10 == 0:
                client.update_model(
                    updates=[{
                        'embedding_vector': embedding.tolist(),
                        'actual_trend': trend_from_data_generator,
                        'predicted_trend': predicted_trend,
                        'timestamp': inference_time,
                        'velocity_features': generated_velocity
                    }],
                    model_name=model_name,
                )
        except Exception as e:
            print(f"Prediction {i+1} failed: {e}")

    print(f"Predictions done. {correct_predictions}/{predictions} were corrects")
    # Get stats
    print("\n4. Getting model statistics...")
    try:
        stats = client.get_stats(model_name)
        model_stats = stats['statistics']['model_stats']
        print(f"Accuracy: {model_stats['accuracy']:.3f}")
        print(f"Prediction Count: {model_stats['prediction_count']}")
        print(f"Is Fitted: {model_stats['is_fitted']}")
        print(f"Trend Distribution: {model_stats['trend_distribution']}")
        
    except Exception as e:
        print(f"Stats request failed: {e}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for trend detection")
    parser.add_argument("--model_name", type=str, default="trend_classifier_poc", help="model name")
    parser.add_argument("--predictions", type=int, default=100, help="model name")
    args = parser.parse_args()

    demo(
        args.model_name,
        args.predictions,
    )
