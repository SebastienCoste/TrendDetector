#!/usr/bin/env python3
"""
Simple test client for the Trending Content Detection System
"""

import requests
import json
import numpy as np
from pathlib import Path

class TrendClient:
    """Client for interacting with the trend detection API"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        """Check service health"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def get_model_metadata(self, model_name: str = "trend_classifier"):
        """Get model metadata"""
        response = self.session.get(f"{self.base_url}/v2/models/{model_name}")
        return response.json()
    
    def predict_trend(self, embedding_vector: np.ndarray, velocity_features: list = None, model_name: str = "trend_classifier"):
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
    
    def update_model(self, updates: list, model_name: str = "trend_classifier"):
        """Update model with feedback"""
        request_data = {"updates": updates}
        
        response = self.session.post(
            f"{self.base_url}/v2/models/{model_name}/update",
            json=request_data
        )
        return response.json()
    
    def get_stats(self, model_name: str = "trend_classifier"):
        """Get model statistics"""
        response = self.session.get(f"{self.base_url}/v2/models/{model_name}/stats")
        return response.json()

def demo():
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
        metadata = client.get_model_metadata()
        print(f"Model Platform: {metadata['platform']}")
        print(f"Available Versions: {metadata.get('versions', [])}")
    except Exception as e:
        print(f"Metadata request failed: {e}")
        return
    
    # Make predictions
    print("\n3. Making sample predictions...")
    
    # Generate some sample data
    np.random.seed(42)
    
    for i in range(3):
        # Random embedding vector
        embedding = np.random.randn(512)
        
        # Sample velocity features 
        velocity_features = [
            float(np.random.exponential(10)),  # download_velocity_1h
            float(np.random.exponential(8)),   # download_velocity_24h
            float(np.random.exponential(5)),   # like_velocity_1h
            float(np.random.exponential(4)),   # like_velocity_24h
            float(np.random.exponential(2)),   # dislike_velocity_1h
            float(np.random.exponential(1)),   # dislike_velocity_24h
            float(np.random.normal(0, 0.1)),   # rating_velocity_1h
            float(np.random.normal(0, 0.08))   # rating_velocity_24h
        ]
        
        try:
            result = client.predict_trend(embedding, velocity_features)
            
            # Extract outputs
            predicted_trend = result['outputs'][0]['data'][0]
            confidence = result['outputs'][1]['data'][0]
            
            print(f"Prediction {i+1}: {predicted_trend} (confidence: {confidence:.3f})")
            
        except Exception as e:
            print(f"Prediction {i+1} failed: {e}")
    
    # Get stats
    print("\n4. Getting model statistics...")
    try:
        stats = client.get_stats()
        model_stats = stats['statistics']['model_stats']
        print(f"Accuracy: {model_stats['accuracy']:.3f}")
        print(f"Prediction Count: {model_stats['prediction_count']}")
        print(f"Is Fitted: {model_stats['is_fitted']}")
        print(f"Trend Distribution: {model_stats['trend_distribution']}")
        
    except Exception as e:
        print(f"Stats request failed: {e}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    demo()