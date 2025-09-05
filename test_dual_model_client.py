#!/usr/bin/env python3
"""
Dual-Model Test Client for the Trending Content Detection System
Tests both classification and regression models
"""

import requests
import json
import numpy as np
from pathlib import Path

class DualModelClient:
    """Client for testing both classification and regression models"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        """Check service health"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def get_model_metadata(self, model_name: str, model_type: str):
        """Get model metadata for specific type"""
        response = self.session.get(f"{self.base_url}/v2/models/{model_name}?model_type={model_type}")
        return response.json()
    
    def predict_classification(self, embedding_vector: np.ndarray, model_name: str = "trend_classifier"):
        """Make a classification prediction"""
        
        data = {
            "inputs": [{
                "name": "embedding_vector",
                "shape": [len(embedding_vector)],
                "datatype": "FP32",
                "data": embedding_vector.tolist()
            }],
            "parameters": {"model_type": "classification"}
        }
        
        response = self.session.post(
            f"{self.base_url}/v2/models/{model_name}/infer?model_type=classification",
            json=data
        )
        return response.json()
    
    def predict_regression(self, embedding_vector: np.ndarray, model_name: str = "trend_regressor"):
        """Make a regression prediction"""
        
        data = {
            "inputs": [{
                "name": "embedding_vector",
                "shape": [len(embedding_vector)],
                "datatype": "FP32",
                "data": embedding_vector.tolist()
            }],
            "parameters": {"model_type": "regression"}
        }
        
        response = self.session.post(
            f"{self.base_url}/v2/models/{model_name}/infer?model_type=regression",
            json=data
        )
        return response.json()
    
    def update_classification(self, updates: list, model_name: str = "trend_classifier"):
        """Update classification model with feedback"""
        request_data = {
            "updates": updates,
            "model_type": "classification"
        }
        
        response = self.session.post(
            f"{self.base_url}/v2/models/{model_name}/update",
            json=request_data
        )
        return response.json()
    
    def update_regression(self, updates: list, model_name: str = "trend_regressor"):
        """Update regression model with feedback"""
        request_data = {
            "updates": updates,
            "model_type": "regression"
        }
        
        response = self.session.post(
            f"{self.base_url}/v2/models/{model_name}/update",
            json=request_data
        )
        return response.json()

def demo():
    """Run comprehensive demo of both model types"""
    client = DualModelClient()
    
    print("=== Dual-Model Trending Content Detection System Demo ===\n")
    
    # Check health
    print("1. Checking service health...")
    try:
        health = client.health_check()
        print(f"   Health Status: {health['status']}")
    except Exception as e:
        print(f"   Health check failed: {e}")
        return
    
    # Test model metadata
    print("\n2. Getting model metadata...")
    try:
        # Classification metadata
        class_metadata = client.get_model_metadata("trend_classifier", "classification")
        print(f"   Classification Platform: {class_metadata['platform']}")
        print(f"   Classification Outputs: {[out['name'] for out in class_metadata['outputs']]}")
        
        # Regression metadata
        reg_metadata = client.get_model_metadata("trend_regressor", "regression")
        print(f"   Regression Platform: {reg_metadata['platform']}")
        print(f"   Regression Outputs: {[out['name'] for out in reg_metadata['outputs']]}")
        
    except Exception as e:
        print(f"   Metadata request failed: {e}")
        return
    
    # Test predictions
    print("\n3. Testing predictions on both models...")
    
    # Generate test embeddings
    np.random.seed(42)
    test_embeddings = [np.random.randn(512) for _ in range(3)]
    
    for i, embedding in enumerate(test_embeddings):
        print(f"\n   Test Case {i+1}:")
        
        try:
            # Classification prediction
            class_result = client.predict_classification(embedding)
            if 'outputs' in class_result:
                predicted_trend = class_result['outputs'][0]['data'][0]
                confidence = class_result['outputs'][1]['data'][0]
                probabilities = class_result['outputs'][2]['data']
                print(f"   Classification: {predicted_trend} (confidence: {confidence:.3f})")
                print(f"   Probabilities: [up:{probabilities[0]:.3f}, down:{probabilities[1]:.3f}, neutral:{probabilities[2]:.3f}]")
            else:
                print(f"   Classification Error: {class_result}")
            
            # Regression prediction
            reg_result = client.predict_regression(embedding)
            if 'outputs' in reg_result:
                predicted_score = reg_result['outputs'][0]['data'][0]
                confidence = reg_result['outputs'][1]['data'][0]
                print(f"   Regression: {predicted_score:.4f} (confidence: {confidence:.3f})")
            else:
                print(f"   Regression Error: {reg_result}")
                
        except Exception as e:
            print(f"   Prediction failed: {e}")
    
    # Test model updates
    print("\n4. Testing model updates...")
    
    try:
        # Classification update
        class_updates = [{
            'embedding_vector': np.random.randn(512).tolist(),
            'actual_trend': 'upward',
            'predicted_trend': 'neutral'
        }]
        
        class_update_result = client.update_classification(class_updates)
        print(f"   Classification Update: {class_update_result['processed_updates']} updates processed")
        print(f"   Drift Detected: {class_update_result['drift_detected']}")
        
        # Regression update
        reg_updates = [{
            'embedding_vector': np.random.randn(512).tolist(),
            'actual_score': 0.8,
            'predicted_score': 0.2
        }]
        
        reg_update_result = client.update_regression(reg_updates)
        print(f"   Regression Update: {reg_update_result['processed_updates']} updates processed")
        print(f"   Drift Detected: {reg_update_result['drift_detected']}")
        
    except Exception as e:
        print(f"   Update test failed: {e}")
    
    # Model comparison
    print("\n5. Model Type Comparison:")
    test_embedding = np.random.randn(512)
    
    try:
        class_result = client.predict_classification(test_embedding)
        reg_result = client.predict_regression(test_embedding)
        
        if 'outputs' in class_result and 'outputs' in reg_result:
            class_trend = class_result['outputs'][0]['data'][0]
            reg_score = reg_result['outputs'][0]['data'][0]
            
            print(f"   Same input vector:")
            print(f"   → Classification: {class_trend}")
            print(f"   → Regression: {reg_score:.4f}")
            
            # Convert score to category for comparison
            if reg_score > 0.3:
                reg_category = "upward"
            elif reg_score < -0.3:
                reg_category = "downward" 
            else:
                reg_category = "neutral"
            
            print(f"   → Regression as category: {reg_category}")
            print(f"   → Agreement: {'✓' if class_trend == reg_category else '✗'}")
        
    except Exception as e:
        print(f"   Comparison failed: {e}")
    
    print("\n=== Demo Complete ===")
    print("\nKey Features Demonstrated:")
    print("✓ Dual-model support (classification and regression)")
    print("✓ Model-specific API endpoints and responses")
    print("✓ Dynamic model type selection via parameters")
    print("✓ Embedding-only input (no velocity features)")
    print("✓ Model-specific training and updates")
    print("✓ KServe V2 compliance for both model types")

if __name__ == "__main__":
    demo()