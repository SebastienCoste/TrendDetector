#!/usr/bin/env python3
"""
Backend API Testing for TrendDetector UI System
Tests all UI API endpoints and functionality
"""

import requests
import json
import time
import sys
from datetime import datetime
from typing import Dict, Any, List

class TrendDetectorAPITester:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        result = {
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"   Details: {details}")
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                if "status" in data and data["status"] == "healthy":
                    self.log_test("Health Check", True, f"Service healthy, model_loaded: {data.get('model_loaded', False)}")
                    return True
                else:
                    self.log_test("Health Check", False, f"Unexpected response: {data}")
                    return False
            else:
                self.log_test("Health Check", False, f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_test("Health Check", False, f"Exception: {str(e)}")
            return False
    
    def test_models_endpoint(self):
        """Test GET /api/ui/models endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/api/ui/models")
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    model_count = len(data)
                    model_types = [model.get("model_type") for model in data]
                    self.log_test("Models Info", True, f"Found {model_count} models: {model_types}")
                    return True
                else:
                    self.log_test("Models Info", False, f"Expected list, got: {type(data)}")
                    return False
            else:
                self.log_test("Models Info", False, f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_test("Models Info", False, f"Exception: {str(e)}")
            return False
    
    def test_vector_algorithms_endpoint(self):
        """Test GET /api/ui/vector/algorithms endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/api/ui/vector/algorithms")
            if response.status_code == 200:
                data = response.json()
                if "base_patterns" in data and "parameter_ranges" in data:
                    patterns = data["base_patterns"]
                    self.log_test("Vector Algorithms", True, f"Available patterns: {patterns}")
                    return True
                else:
                    self.log_test("Vector Algorithms", False, f"Missing expected fields in response: {data}")
                    return False
            else:
                self.log_test("Vector Algorithms", False, f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_test("Vector Algorithms", False, f"Exception: {str(e)}")
            return False
    
    def test_vector_generation(self):
        """Test POST /api/ui/vector/generate endpoint"""
        try:
            # Test configuration
            config = {
                "trend_score": 0.5,
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
            
            response = self.session.post(
                f"{self.base_url}/api/ui/vector/generate",
                json=config,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if "vector" in data and "expected_trend" in data:
                    vector_len = len(data["vector"])
                    expected_trend = data["expected_trend"]
                    algorithm = data.get("algorithm_used", "unknown")
                    self.log_test("Vector Generation", True, 
                                f"Generated vector length: {vector_len}, expected_trend: {expected_trend}, algorithm: {algorithm}")
                    return True
                else:
                    self.log_test("Vector Generation", False, f"Missing expected fields: {data}")
                    return False
            else:
                self.log_test("Vector Generation", False, f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_test("Vector Generation", False, f"Exception: {str(e)}")
            return False
    
    def test_drift_test_workflow(self):
        """Test complete drift test workflow"""
        try:
            # Step 1: Start drift test
            config = {
                "num_requests": 20,
                "feedback_frequency": 5,
                "drift_point": 10,
                "model_type": "regression"
            }
            
            response = self.session.post(
                f"{self.base_url}/api/ui/drift-test/start",
                json=config,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                self.log_test("Drift Test Start", False, f"HTTP {response.status_code}: {response.text}")
                return False
            
            start_data = response.json()
            test_id = start_data.get("test_id")
            
            if not test_id:
                self.log_test("Drift Test Start", False, "No test_id returned")
                return False
            
            self.log_test("Drift Test Start", True, f"Started test: {test_id}")
            
            # Step 2: Monitor test progress
            max_wait_time = 30  # seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                status_response = self.session.get(f"{self.base_url}/api/ui/drift-test/{test_id}/status")
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    test_status = status_data.get("status")
                    progress = status_data.get("progress", 0)
                    
                    print(f"   Test progress: {progress:.1%} - Status: {test_status}")
                    
                    if test_status == "completed":
                        self.log_test("Drift Test Progress", True, f"Test completed with progress: {progress:.1%}")
                        break
                    elif test_status == "error":
                        self.log_test("Drift Test Progress", False, f"Test failed with error")
                        return False
                    
                    time.sleep(2)  # Wait 2 seconds before checking again
                else:
                    self.log_test("Drift Test Progress", False, f"Status check failed: {status_response.status_code}")
                    return False
            else:
                self.log_test("Drift Test Progress", False, "Test did not complete within timeout")
                return False
            
            # Step 3: Get test results
            results_response = self.session.get(f"{self.base_url}/api/ui/drift-test/{test_id}/results")
            
            if results_response.status_code == 200:
                results_data = results_response.json()
                if isinstance(results_data, list) and len(results_data) > 0:
                    result_count = len(results_data)
                    drift_results = [r for r in results_data if r.get("is_drift_period", False)]
                    self.log_test("Drift Test Results", True, 
                                f"Retrieved {result_count} results, {len(drift_results)} in drift period")
                else:
                    self.log_test("Drift Test Results", False, "No results returned")
                    return False
            else:
                self.log_test("Drift Test Results", False, f"Results fetch failed: {results_response.status_code}")
                return False
            
            # Step 4: Test active tests listing
            active_response = self.session.get(f"{self.base_url}/api/ui/drift-test/active")
            if active_response.status_code == 200:
                active_tests = active_response.json()
                self.log_test("Active Tests List", True, f"Found {len(active_tests)} active tests")
            else:
                self.log_test("Active Tests List", False, f"HTTP {active_response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            self.log_test("Drift Test Workflow", False, f"Exception: {str(e)}")
            return False
    
    def test_prediction_endpoint(self):
        """Test POST /api/ui/predict endpoint"""
        try:
            # Generate a test vector
            test_vector = [0.1 * i for i in range(512)]  # Simple test vector
            
            # Test classification prediction - send vector directly as JSON array
            response = self.session.post(
                f"{self.base_url}/api/ui/predict?model_type=classification",
                json=test_vector,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if "predicted_value" in data and "confidence" in data:
                    predicted = data["predicted_value"]
                    confidence = data["confidence"]
                    self.log_test("Classification Prediction", True, 
                                f"Predicted: {predicted}, Confidence: {confidence:.3f}")
                else:
                    self.log_test("Classification Prediction", False, f"Missing fields in response: {data}")
                    return False
            else:
                self.log_test("Classification Prediction", False, f"HTTP {response.status_code}: {response.text}")
                return False
            
            # Test regression prediction
            reg_payload = {
                "vector": test_vector,
                "model_type": "regression"
            }
            
            response = self.session.post(
                f"{self.base_url}/api/ui/predict",
                json=reg_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if "predicted_value" in data and "confidence" in data:
                    predicted = data["predicted_value"]
                    confidence = data["confidence"]
                    self.log_test("Regression Prediction", True, 
                                f"Predicted: {predicted:.3f}, Confidence: {confidence:.3f}")
                    return True
                else:
                    self.log_test("Regression Prediction", False, f"Missing fields in response: {data}")
                    return False
            else:
                self.log_test("Regression Prediction", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Prediction Endpoint", False, f"Exception: {str(e)}")
            return False
    
    def test_error_scenarios(self):
        """Test error handling scenarios"""
        try:
            # Test invalid endpoint
            response = self.session.get(f"{self.base_url}/api/ui/invalid-endpoint")
            if response.status_code == 404:
                self.log_test("Invalid Endpoint Error", True, "Correctly returned 404")
            else:
                self.log_test("Invalid Endpoint Error", False, f"Expected 404, got {response.status_code}")
            
            # Test malformed request body
            response = self.session.post(
                f"{self.base_url}/api/ui/vector/generate",
                json={"invalid": "data"},
                headers={"Content-Type": "application/json"}
            )
            if response.status_code in [400, 422]:  # Bad request or validation error
                self.log_test("Malformed Request Error", True, f"Correctly returned {response.status_code}")
            else:
                self.log_test("Malformed Request Error", False, f"Expected 400/422, got {response.status_code}")
            
            # Test non-existent drift test
            response = self.session.get(f"{self.base_url}/api/ui/drift-test/non-existent-id/status")
            if response.status_code == 404:
                self.log_test("Non-existent Test Error", True, "Correctly returned 404")
            else:
                self.log_test("Non-existent Test Error", False, f"Expected 404, got {response.status_code}")
            
            return True
            
        except Exception as e:
            self.log_test("Error Scenarios", False, f"Exception: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all backend tests"""
        print("🚀 Starting TrendDetector UI API Backend Tests")
        print("=" * 60)
        
        # Test service health first
        if not self.test_health_endpoint():
            print("❌ Service health check failed - aborting tests")
            return False
        
        # Core API tests
        tests = [
            self.test_models_endpoint,
            self.test_vector_algorithms_endpoint,
            self.test_vector_generation,
            self.test_prediction_endpoint,
            self.test_drift_test_workflow,
            self.test_error_scenarios
        ]
        
        passed = 0
        total = len(tests)
        
        for test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                print(f"❌ Test {test_func.__name__} crashed: {e}")
        
        print("\n" + "=" * 60)
        print(f"📊 Test Summary: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All backend tests PASSED!")
            return True
        else:
            print(f"⚠️  {total - passed} tests FAILED")
            return False
    
    def get_test_summary(self):
        """Get detailed test summary"""
        passed = sum(1 for result in self.test_results if result["success"])
        total = len(self.test_results)
        
        summary = {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": passed / total if total > 0 else 0,
            "details": self.test_results
        }
        
        return summary

def main():
    """Main test execution"""
    tester = TrendDetectorAPITester()
    
    # Run all tests
    success = tester.run_all_tests()
    
    # Print detailed summary
    summary = tester.get_test_summary()
    print(f"\n📋 Detailed Summary:")
    print(f"   Success Rate: {summary['success_rate']:.1%}")
    print(f"   Total Tests: {summary['total_tests']}")
    print(f"   Passed: {summary['passed']}")
    print(f"   Failed: {summary['failed']}")
    
    # Save results to file
    with open("/app/backend_test_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())