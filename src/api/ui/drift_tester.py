import uuid
import asyncio
import numpy as np
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

from .models import DriftTestConfig, DriftTestResult, TestStatus, ModelType
from ...core.model_manager import ModelManager
from ...ml.model_interface import TrendModelInterface

logger = logging.getLogger(__name__)

class DriftTester:
    """Manages drift testing experiments for the UI"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.active_tests: Dict[str, TestStatus] = {}
        self.test_results: Dict[str, List[DriftTestResult]] = {}
        
    async def start_drift_test(self, config: DriftTestConfig) -> str:
        """Start a new drift test"""
        test_id = str(uuid.uuid4())
        
        # Initialize test status
        status = TestStatus(
            test_id=test_id,
            status="running",
            progress=0.0,
            current_request=0,
            total_requests=config.num_requests,
            metrics={}
        )
        
        self.active_tests[test_id] = status
        self.test_results[test_id] = []
        
        # Start test in background
        asyncio.create_task(self._run_drift_test(test_id, config))
        
        logger.info(f"Started drift test {test_id} with config: {config}")
        return test_id
    
    async def _run_drift_test(self, test_id: str, config: DriftTestConfig):
        """Execute the drift test"""
        try:
            # Get or create model
            model = self.model_manager.get_model("ui_test_model", config.model_type.value)
            
            # Prepare base data for consistent generation
            base_time = datetime.now().timestamp()
            results = []
            
            # Phase 1: Pre-drift period
            logger.info(f"Test {test_id}: Starting pre-drift phase")
            await self._run_test_phase(
                test_id, model, config, results, 
                0, config.drift_point, base_time, is_drift=False
            )
            
            # Phase 2: Introduce drift and continue
            logger.info(f"Test {test_id}: Introducing drift at request {config.drift_point}")
            await self._run_test_phase(
                test_id, model, config, results,
                config.drift_point, config.num_requests, base_time, is_drift=True
            )
            
            # Update final status
            self.active_tests[test_id].status = "completed"
            self.active_tests[test_id].progress = 1.0
            self.active_tests[test_id].metrics = self._calculate_final_metrics(results, config)
            
            logger.info(f"Test {test_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Test {test_id} failed: {e}")
            self.active_tests[test_id].status = "error"
            
    async def _run_test_phase(self, test_id: str, model: TrendModelInterface, 
                             config: DriftTestConfig, results: List[DriftTestResult],
                             start_idx: int, end_idx: int, base_time: float, is_drift: bool):
        """Run a phase of the drift test"""
        
        feedback_buffer = []
        
        for i in range(start_idx, end_idx):
            # Generate test data
            vector, expected_trend = self._generate_test_data(
                base_time + i * 3600, config.model_type, is_drift, config
            )
            
            # Make prediction
            prediction_result = model.predict(vector)
            
            if config.model_type == ModelType.CLASSIFICATION:
                predicted_trend = prediction_result.predicted_trend
                confidence = prediction_result.confidence
            else:  # regression
                predicted_trend = prediction_result.predicted_score
                confidence = prediction_result.confidence
            
            # Calculate error
            if config.model_type == ModelType.CLASSIFICATION:
                error = 0.0 if expected_trend == predicted_trend else 1.0
            else:
                error = abs(float(expected_trend) - float(predicted_trend))
            
            # Create result record
            result = DriftTestResult(
                request_id=i,
                timestamp=datetime.fromtimestamp(base_time + i * 3600),
                embedding_vector=vector.tolist(),
                expected_trend=expected_trend,
                predicted_trend=predicted_trend,
                confidence=confidence,
                absolute_error=error,
                is_drift_period=is_drift,
                feedback_provided=False
            )
            
            results.append(result)
            feedback_buffer.append((vector, expected_trend, predicted_trend, base_time + i * 3600))
            
            # Provide feedback periodically
            if len(feedback_buffer) >= config.feedback_frequency:
                logger.debug(f"Test {test_id}: Providing feedback for {len(feedback_buffer)} samples")
                
                drift_detected_any = False
                for fb_vector, fb_expected, fb_predicted, fb_timestamp in feedback_buffer:
                    drift_detected = model.learn(
                        features=fb_vector,
                        target=fb_expected,
                        predicted_value=fb_predicted,
                        timestamp=fb_timestamp
                    )
                    if drift_detected:
                        drift_detected_any = True
                
                # Mark recent records as having received feedback
                for j in range(len(feedback_buffer)):
                    results[-(j+1)].feedback_provided = True
                    results[-(j+1)].drift_detected = drift_detected_any
                
                feedback_buffer.clear()
            
            # Update progress
            self.active_tests[test_id].current_request = i + 1
            self.active_tests[test_id].progress = (i + 1) / config.num_requests
            
            # Store results for real-time access
            self.test_results[test_id] = results.copy()
            
            # Small delay to simulate real-time processing
            await asyncio.sleep(0.1)
    
    def _generate_test_data(self, timestamp: float, model_type: ModelType, 
                           is_drift: bool, config: DriftTestConfig) -> tuple:
        """Generate test embedding vector and expected trend"""
        
        # Generate embedding vector
        embedding_dim = 512
        vector = np.random.randn(embedding_dim)
        
        # Apply drift modifications if needed
        if is_drift:
            # Modify vector generation to simulate concept drift
            drift_factor = 1.5
            vector = vector * drift_factor
            noise_increase = np.random.normal(0, 0.3, embedding_dim)
            vector += noise_increase
        
        # Calculate expected trend based on vector and time
        expected_score = self._calculate_expected_trend(vector, timestamp)
        
        if model_type == ModelType.CLASSIFICATION:
            if expected_score > 0.3:
                expected_trend = "upward"
            elif expected_score < -0.3:
                expected_trend = "downward" 
            else:
                expected_trend = "neutral"
        else:
            expected_trend = expected_score
            
        return vector, expected_trend
    
    def _calculate_expected_trend(self, vector: np.ndarray, timestamp: float) -> float:
        """Calculate expected trend score based on vector and timestamp"""
        
        # Vector-based features
        vector_sum = np.sum(vector)
        vector_magnitude = np.linalg.norm(vector)
        vector_variance = np.var(vector)
        
        # Time-based features
        hour_of_day = (timestamp % (24 * 3600)) / 3600
        day_of_week = int(timestamp // (24 * 3600)) % 7
        
        # Calculate trend score
        trend_score = 0.0
        
        # Vector influence
        if vector_sum > 5.0:
            trend_score += 0.2
        elif vector_sum < -5.0:
            trend_score -= 0.2
            
        if vector_magnitude > 25.0:
            trend_score += 0.1
        elif vector_magnitude < 15.0:
            trend_score -= 0.1
            
        # Time influence
        if 9 <= hour_of_day <= 11 or 19 <= hour_of_day <= 21:
            trend_score += 0.15
        elif 2 <= hour_of_day <= 6:
            trend_score -= 0.15
            
        # Weekend effect
        if day_of_week in [5, 6]:
            trend_score += 0.1
            
        # Add noise
        trend_score += np.random.normal(0, 0.1)
        
        # Normalize to [-1, 1]
        return float(np.tanh(trend_score))
    
    def _calculate_final_metrics(self, results: List[DriftTestResult], 
                               config: DriftTestConfig) -> Dict[str, Any]:
        """Calculate final test metrics"""
        
        if not results:
            return {}
        
        # Separate pre-drift and drift periods
        pre_drift = [r for r in results if not r.is_drift_period]
        drift_period = [r for r in results if r.is_drift_period]
        
        metrics = {
            "total_requests": len(results),
            "pre_drift_requests": len(pre_drift),
            "drift_requests": len(drift_period),
            "feedback_events": sum(1 for r in results if r.feedback_provided),
            "drift_detections": sum(1 for r in results if r.drift_detected)
        }
        
        if pre_drift:
            pre_drift_errors = [r.absolute_error for r in pre_drift]
            metrics["pre_drift_error"] = {
                "mean": float(np.mean(pre_drift_errors)),
                "std": float(np.std(pre_drift_errors)),
                "max": float(np.max(pre_drift_errors))
            }
        
        if drift_period:
            drift_errors = [r.absolute_error for r in drift_period]
            metrics["drift_error"] = {
                "mean": float(np.mean(drift_errors)),
                "std": float(np.std(drift_errors)),
                "max": float(np.max(drift_errors))
            }
            
            # Calculate error change
            if pre_drift and drift_period:
                error_change = ((metrics["drift_error"]["mean"] - metrics["pre_drift_error"]["mean"]) / 
                              metrics["pre_drift_error"]["mean"]) * 100
                metrics["error_change_percent"] = error_change
        
        return metrics
    
    def get_test_status(self, test_id: str) -> Optional[TestStatus]:
        """Get current test status"""
        return self.active_tests.get(test_id)
    
    def get_test_results(self, test_id: str) -> Optional[List[DriftTestResult]]:
        """Get test results"""
        return self.test_results.get(test_id)
    
    def list_active_tests(self) -> List[str]:
        """List all active test IDs"""
        return list(self.active_tests.keys())
    
    def cleanup_test(self, test_id: str) -> bool:
        """Clean up test data"""
        if test_id in self.active_tests:
            del self.active_tests[test_id]
        if test_id in self.test_results:
            del self.test_results[test_id]
        return True