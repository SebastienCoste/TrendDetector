#!/usr/bin/env python3
"""
TrendDetector UI Drift Test Demo
Simple script to demonstrate drift testing workflow
"""

import requests
import time
import json
from datetime import datetime

class TrendDetectorDemo:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/ui"
        
    def print_header(self, title):
        print(f"\n{'='*60}")
        print(f"🎯 {title}")
        print(f"{'='*60}")
    
    def print_step(self, step, description):
        print(f"\n{step}. {description}")
        print("-" * 40)
    
    def check_system_health(self):
        """Check system health and models"""
        self.print_step("1", "Checking System Health")
        
        try:
            # Check main health endpoint
            health_response = requests.get(f"{self.base_url}/health")
            if health_response.status_code == 200:
                health = health_response.json()
                print(f"   ✅ Service Status: {health.get('status', 'unknown').upper()}")
                print(f"   ✅ Model Loaded: {health.get('model_loaded', False)}")
                print(f"   ✅ GPU Enabled: {health.get('gpu_enabled', False)}")
                print(f"   ✅ Version: {health.get('version', 'unknown')}")
            
            # Check models
            models_response = requests.get(f"{self.api_base}/models")
            if models_response.status_code == 200:
                models = models_response.json()
                print(f"\n   📊 Found {len(models)} models:")
                for model in models:
                    status = "✅ LOADED" if model.get('is_loaded') else "❌ NOT LOADED"
                    print(f"      {status}: {model.get('model_name')} ({model.get('model_type')})")
                return True
            else:
                print(f"   ❌ Failed to get models: {models_response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Health check failed: {e}")
            return False
    
    def demonstrate_vector_generation(self):
        """Demonstrate vector generation"""
        self.print_step("2", "Demonstrating Vector Generation")
        
        try:
            # Get available algorithms
            alg_response = requests.get(f"{self.api_base}/vector/algorithms")
            if alg_response.status_code == 200:
                algorithms = alg_response.json()
                print(f"   📋 Available patterns: {algorithms.get('base_patterns', [])}")
            
            # Generate a sample vector
            config = {
                "trend_score": 0.7,
                "algorithm_params": {
                    "base_pattern": "sinusoidal",
                    "noise_level": 0.1,
                    "temporal_factors": {
                        "hourly": True,
                        "daily": True,
                        "weekly": False
                    },
                    "velocity_influence": 0.3,
                    "embedding_correlation": 0.7
                },
                "embedding_dim": 512
            }
            
            gen_response = requests.post(f"{self.api_base}/vector/generate", json=config)
            if gen_response.status_code == 200:
                result = gen_response.json()
                vector = result.get('vector', [])
                expected_trend = result.get('expected_trend')
                
                print(f"   ✅ Generated {len(vector)}-dimensional vector")
                print(f"   📈 Expected trend: {expected_trend:.4f}")
                print(f"   🔢 Vector stats: min={min(vector):.3f}, max={max(vector):.3f}")
                return True
            else:
                print(f"   ❌ Vector generation failed: {gen_response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Vector generation error: {e}")
            return False
    
    def run_drift_test(self):
        """Run a complete drift test"""
        self.print_step("3", "Running Concept Drift Test")
        
        try:
            # Configure test
            config = {
                "num_requests": 60,  # Shorter for demo
                "feedback_frequency": 10,
                "drift_point": 30,
                "model_type": "regression"
            }
            
            print(f"   ⚙️  Configuration:")
            print(f"      • Total requests: {config['num_requests']}")
            print(f"      • Drift point: {config['drift_point']}")
            print(f"      • Feedback frequency: {config['feedback_frequency']}")
            print(f"      • Model type: {config['model_type']}")
            
            # Start test
            start_response = requests.post(f"{self.api_base}/drift-test/start", json=config)
            if start_response.status_code != 200:
                print(f"   ❌ Failed to start test: {start_response.status_code}")
                return False
            
            test_data = start_response.json()
            test_id = test_data.get('test_id')
            print(f"\n   🚀 Test started: {test_id[:8]}...")
            
            # Monitor progress
            print(f"   📊 Monitoring progress (updates every 2 seconds):")
            
            last_progress = 0
            while True:
                time.sleep(2)
                
                # Get status
                status_response = requests.get(f"{self.api_base}/drift-test/{test_id}/status")
                if status_response.status_code != 200:
                    print(f"   ❌ Failed to get status: {status_response.status_code}")
                    break
                
                status = status_response.json()
                progress = status.get('progress', 0) * 100
                current = status.get('current_request', 0)
                total = status.get('total_requests', 0)
                test_status = status.get('status', 'unknown')
                
                # Show progress bar
                bar_length = 30
                filled_length = int(bar_length * progress / 100)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                
                if progress > last_progress:
                    print(f"      [{bar}] {progress:5.1f}% ({current}/{total}) - {test_status.upper()}")
                    last_progress = progress
                
                if test_status in ['completed', 'error']:
                    break
            
            # Get final results
            results_response = requests.get(f"{self.api_base}/drift-test/{test_id}/results")
            if results_response.status_code == 200:
                results = results_response.json()
                
                # Analyze results
                pre_drift = [r for r in results if not r.get('is_drift_period', False)]
                drift_period = [r for r in results if r.get('is_drift_period', False)]
                
                if pre_drift and drift_period:
                    pre_error = sum(r.get('absolute_error', 0) for r in pre_drift) / len(pre_drift)
                    drift_error = sum(r.get('absolute_error', 0) for r in drift_period) / len(drift_period)
                    error_change = ((drift_error - pre_error) / pre_error) * 100 if pre_error > 0 else 0
                    
                    print(f"\n   📈 RESULTS ANALYSIS:")
                    print(f"      • Pre-drift error: {pre_error:.4f}")
                    print(f"      • Drift period error: {drift_error:.4f}")
                    print(f"      • Error change: {error_change:+.1f}%")
                    print(f"      • Total samples: {len(results)}")
                    print(f"      • Feedback events: {sum(1 for r in results if r.get('feedback_provided', False))}")
                
                # Cleanup
                cleanup_response = requests.delete(f"{self.api_base}/drift-test/{test_id}")
                if cleanup_response.status_code == 200:
                    print(f"   🧹 Test {test_id[:8]}... cleaned up")
                
                return True
            else:
                print(f"   ❌ Failed to get results: {results_response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Drift test error: {e}")
            return False
    
    def demonstrate_prediction(self):
        """Demonstrate single prediction"""
        self.print_step("4", "Demonstrating Single Prediction")
        
        try:
            # Generate a test vector
            test_vector = [0.5 + 0.1 * i for i in range(512)]  # Simple upward trending vector
            
            # Make prediction
            pred_response = requests.post(f"{self.api_base}/predict", json={
                "vector": test_vector,
                "model_type": "regression"
            })
            
            if pred_response.status_code == 200:
                result = pred_response.json()
                predicted_value = result.get('predicted_value')
                confidence = result.get('confidence', 0)
                
                print(f"   ✅ Prediction successful!")
                print(f"   📊 Predicted trend score: {predicted_value:.4f}")
                print(f"   🎯 Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
                
                # Interpret result
                if predicted_value > 0.3:
                    interpretation = "STRONG UPWARD trend"
                elif predicted_value < -0.3:
                    interpretation = "STRONG DOWNWARD trend"
                else:
                    interpretation = "NEUTRAL trend"
                
                print(f"   💡 Interpretation: {interpretation}")
                return True
            else:
                print(f"   ❌ Prediction failed: {pred_response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Prediction error: {e}")
            return False
    
    def run_complete_demo(self):
        """Run the complete demonstration"""
        self.print_header("TrendDetector Complete Demo")
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌐 Backend URL: {self.base_url}")
        print(f"🖥️  Frontend URL: http://localhost:3000")
        
        success_count = 0
        total_steps = 4
        
        # Run all demo steps
        if self.check_system_health():
            success_count += 1
        
        if self.demonstrate_vector_generation():
            success_count += 1
        
        if self.demonstrate_prediction():
            success_count += 1
        
        if self.run_drift_test():
            success_count += 1
        
        # Final summary
        self.print_header("Demo Summary")
        print(f"✅ Completed steps: {success_count}/{total_steps}")
        
        if success_count == total_steps:
            print("🎉 ALL DEMO STEPS SUCCESSFUL!")
            print("\n📋 Next Steps:")
            print("   • Visit http://localhost:3000 to see the UI")
            print("   • Click on 'Drift Testing' tab for visual interface")
            print("   • Try different model types and configurations")
            print("   • Explore the real-time visualization charts")
        else:
            print("⚠️  Some demo steps failed. Check the logs above.")
        
        print(f"\n🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

def main():
    """Main demo execution"""
    print("🎯 TrendDetector Drift Test Demo")
    print("This demo will test all major features of the TrendDetector system")
    
    demo = TrendDetectorDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()