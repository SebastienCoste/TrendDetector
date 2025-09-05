from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from .models import (
    DriftTestConfig, DriftTestResult, TestStatus, ModelType,
    VectorGenerationConfig, VectorGenerationResult, ModelInfo,
    PredictionResult
)
from .drift_tester import DriftTester
from .vector_generator import VectorGenerator
from ...core.model_manager import ModelManager

logger = logging.getLogger(__name__)

# Global instances - will be initialized in main.py
drift_tester: Optional[DriftTester] = None
vector_generator: Optional[VectorGenerator] = None

def get_drift_tester() -> DriftTester:
    """Dependency to get drift tester instance"""
    if drift_tester is None:
        raise HTTPException(status_code=503, detail="Drift tester not initialized")
    return drift_tester

def get_vector_generator() -> VectorGenerator:
    """Dependency to get vector generator instance"""
    if vector_generator is None:
        raise HTTPException(status_code=503, detail="Vector generator not initialized")
    return vector_generator

# Create router
router = APIRouter(prefix="/ui", tags=["ui"])

# Model Information Endpoints
@router.get("/models", response_model=List[ModelInfo])
async def get_model_info():
    """Get information about available models"""
    try:
        from ...main import model_manager
        if not model_manager:
            raise HTTPException(status_code=503, detail="Model manager not available")
        
        models = []
        
        # Check classification model
        classifier_loaded = model_manager.is_model_loaded("trend_classifier")
        if classifier_loaded:
            model = model_manager.get_model("trend_classifier", "classification")
            models.append(ModelInfo(
                model_name="trend_classifier",
                model_type="classification",
                is_loaded=True,
                version="v1",
                last_updated=datetime.now() # TODO: Get actual timestamp
            ))
        
        # Check regression model  
        regressor_loaded = model_manager.is_model_loaded("trend_regressor")
        if regressor_loaded:
            model = model_manager.get_model("trend_regressor", "regression")
            models.append(ModelInfo(
                model_name="trend_regressor", 
                model_type="regression",
                is_loaded=True,
                version="v1",
                last_updated=datetime.now() # TODO: Get actual timestamp
            ))
        
        # Add default entries if no models loaded
        if not models:
            models = [
                ModelInfo(
                    model_name="trend_classifier",
                    model_type="classification", 
                    is_loaded=False,
                    version="v1"
                ),
                ModelInfo(
                    model_name="trend_regressor",
                    model_type="regression",
                    is_loaded=False, 
                    version="v1"
                )
            ]
        
        return models
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Drift Testing Endpoints
@router.post("/drift-test/start")
async def start_drift_test(
    config: DriftTestConfig,
    tester: DriftTester = Depends(get_drift_tester)
) -> Dict[str, str]:
    """Start a new drift test"""
    try:
        test_id = await tester.start_drift_test(config)
        return {"test_id": test_id, "status": "started"}
    except Exception as e:
        logger.error(f"Error starting drift test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/drift-test/{test_id}/status", response_model=TestStatus)
async def get_drift_test_status(
    test_id: str,
    tester: DriftTester = Depends(get_drift_tester)
):
    """Get drift test status"""
    try:
        status = tester.get_test_status(test_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Test not found")
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting test status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/drift-test/{test_id}/results", response_model=List[DriftTestResult])
async def get_drift_test_results(
    test_id: str,
    limit: Optional[int] = None,
    tester: DriftTester = Depends(get_drift_tester)
):
    """Get drift test results"""
    try:
        results = tester.get_test_results(test_id)
        if results is None:
            raise HTTPException(status_code=404, detail="Test results not found")
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            results = results[-limit:]
        
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting test results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/drift-test/active", response_model=List[str])
async def list_active_tests(tester: DriftTester = Depends(get_drift_tester)):
    """List active drift tests"""
    try:
        return tester.list_active_tests()
    except Exception as e:
        logger.error(f"Error listing active tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/drift-test/{test_id}")
async def cleanup_drift_test(
    test_id: str,  
    tester: DriftTester = Depends(get_drift_tester)
):
    """Clean up drift test data"""
    try:
        success = tester.cleanup_test(test_id)
        if not success:
            raise HTTPException(status_code=404, detail="Test not found")
        return {"message": "Test cleaned up successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Vector Generation Endpoints
@router.post("/vector/generate", response_model=VectorGenerationResult)
async def generate_vector(
    config: VectorGenerationConfig,
    generator: VectorGenerator = Depends(get_vector_generator)
):
    """Generate embedding vector based on configuration"""
    try:
        result = generator.generate_vector(config)
        return result
    except Exception as e:
        logger.error(f"Error generating vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/vector/algorithms")
async def get_algorithm_info(
    generator: VectorGenerator = Depends(get_vector_generator)
) -> Dict[str, Any]:
    """Get information about available vector generation algorithms"""
    try:
        return generator.get_algorithm_info()
    except Exception as e:
        logger.error(f"Error getting algorithm info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Single Prediction Endpoint for UI Testing
@router.post("/predict", response_model=PredictionResult)
async def make_prediction(
    vector: List[float],
    model_type: ModelType = ModelType.CLASSIFICATION
):
    """Make a single prediction for UI testing"""
    try:
        from ...main import model_manager
        if not model_manager:
            raise HTTPException(status_code=503, detail="Model manager not available")
        
        # Get the appropriate model
        model_name = "trend_classifier" if model_type == ModelType.CLASSIFICATION else "trend_regressor"
        model = model_manager.get_model(model_name, model_type.value)
        
        # Make prediction
        import numpy as np
        vector_array = np.array(vector)
        prediction_result = model.predict(vector_array)
        
        if model_type == ModelType.CLASSIFICATION:
            result = PredictionResult(
                predicted_value=prediction_result.predicted_trend,
                confidence=prediction_result.confidence,
                probabilities={
                    trend: float(prob) for trend, prob in zip(
                        ["upward", "downward", "neutral"],
                        prediction_result.probabilities if hasattr(prediction_result, 'probabilities') else [0.33, 0.33, 0.34]
                    )
                },
                model_type=model_type.value,
                timestamp=datetime.now()
            )
        else:
            result = PredictionResult(
                predicted_value=prediction_result.predicted_score,
                confidence=prediction_result.confidence,
                model_type=model_type.value,
                timestamp=datetime.now()
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def initialize_ui_services(model_manager: ModelManager):
    """Initialize UI services with model manager"""
    global drift_tester, vector_generator
    
    try:
        drift_tester = DriftTester(model_manager)
        vector_generator = VectorGenerator()
        logger.info("UI services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize UI services: {e}")
        raise