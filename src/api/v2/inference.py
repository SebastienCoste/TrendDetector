from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
import numpy as np
import logging

from .models import (
    InferenceRequest, InferenceResponse, InferTensor,
    ModelMetadata, ErrorResponse, DataType
)
from ...core.model_manager import get_model_manager, ModelManager
from ...utils.validators import validate_embedding_vector, validate_velocity_features

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/models/{model_name}")
async def get_model_metadata(model_name: str, model_type: Optional[str] = None) -> ModelMetadata:
    """Get model metadata (KServe V2 compliant)"""
    try:
        model_manager = get_model_manager()
        
        # Use provided model_type or default from config
        if model_type is None:
            model_type = model_manager.config.model_settings.type

        if not model_manager.is_model_loaded(model_name):
            # Try to load or create the model
            if not model_manager.load_model(model_name, model_type):
                model_manager.create_model(model_name, model_type)

        # Define inputs (only embedding vector as per spec)
        inputs = [
            {
                "name": "embedding_vector",
                "datatype": "FP32",
                "shape": [1, 512],
                "description": "Content embedding vector"
            }
        ]

        # Define outputs based on model type
        if model_type == "classification":
            outputs = [
                {
                    "name": "predicted_trend",
                    "datatype": "BYTES",
                    "shape": [1],
                    "description": "Predicted trend class"
                },
                {
                    "name": "confidence",
                    "datatype": "FP32",
                    "shape": [1],
                    "description": "Prediction confidence"
                },
                {
                    "name": "probabilities",
                    "datatype": "FP32",
                    "shape": [1, 3],
                    "description": "Class probabilities [upward, downward, neutral]"
                }
            ]
            platform = "River-AdaptiveRandomForest-Classifier"
        else:  # regression
            outputs = [
                {
                    "name": "trend_score",
                    "datatype": "FP32",
                    "shape": [1],
                    "description": "Continuous trend score [-1, 1]"
                },
                {
                    "name": "confidence",
                    "datatype": "FP32",
                    "shape": [1],
                    "description": "Prediction confidence"
                }
            ]
            platform = "River-AdaptiveRandomForest-Regressor"

        return ModelMetadata(
            name=model_name,
            versions=model_manager.get_available_versions(model_name),
            platform=platform,
            inputs=inputs,
            outputs=outputs
        )
    except Exception as e:
        logger.error(f"Error getting model metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_name}/versions/{version}")
async def get_model_version_metadata(model_name: str, version: str) -> ModelMetadata:
    """Get specific model version metadata"""
    try:
        model_manager = get_model_manager()

        if not model_manager.is_version_available(model_name, version):
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} version {version} not found"
            )

        metadata = await get_model_metadata(model_name)
        return metadata
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model version metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{model_name}/infer")
async def infer(
    model_name: str,
    request: InferenceRequest,
    model_type: Optional[str] = Query(None, description="Model type override"),
    model_manager: ModelManager = Depends(get_model_manager)
) -> InferenceResponse:
    """Perform inference (KServe V2 compliant)"""
    try:
        # Determine model type
        if model_type is None:
            model_type = request.get_model_type() or model_manager.config.model_settings.type

        # Validate model exists
        if not model_manager.is_model_loaded(model_name):
            if not model_manager.load_model(model_name, model_type):
                model_manager.create_model(model_name, model_type)

        # Extract embedding vector (only input as per spec)
        embedding_vector = None

        for input_tensor in request.inputs:
            if input_tensor.name == "embedding_vector":
                embedding_vector = np.array(input_tensor.data, dtype=np.float32)
                validate_embedding_vector(embedding_vector)
                break

        if embedding_vector is None:
            raise HTTPException(
                status_code=400,
                detail="embedding_vector input is required"
            )

        # Get prediction
        model = model_manager.get_model(model_name, model_type)
        result = model.predict(embedding_vector)

        # Create response based on model type
        if model_type == "classification":
            # Convert probabilities to ordered list [upward, downward, neutral]
            prob_order = ['upward', 'downward', 'neutral']
            prob_list = [result.probabilities.get(trend, 0.0) for trend in prob_order]

            outputs = [
                InferTensor(
                    name="predicted_trend",
                    shape=[1],
                    datatype=DataType.BYTES,
                    data=[result.predicted_trend]
                ),
                InferTensor(
                    name="confidence",
                    shape=[1],
                    datatype=DataType.FP32,
                    data=[result.confidence]
                ),
                InferTensor(
                    name="probabilities",
                    shape=[1, 3],
                    datatype=DataType.FP32,
                    data=prob_list
                )
            ]
            
            log_msg = f"Prediction: {result.predicted_trend}, Confidence: {result.confidence:.3f}"
        else:  # regression
            outputs = [
                InferTensor(
                    name="trend_score",
                    shape=[1],
                    datatype=DataType.FP32,
                    data=[result.predicted_score]
                ),
                InferTensor(
                    name="confidence",
                    shape=[1],
                    datatype=DataType.FP32,
                    data=[result.confidence]
                )
            ]
            
            log_msg = f"Prediction score: {result.predicted_score:.3f}, Confidence: {result.confidence:.3f}"

        response = InferenceResponse(
            model_name=model_name,
            model_version=result.model_version,
            outputs=outputs,
            parameters={
                "prediction_method": result.method,
                "model_type": model_type,
                "timestamp": result.timestamp.isoformat()
            }
        )

        # Log prediction
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"{log_msg}, Method: {result.method}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@router.post("/models/{model_name}/versions/{version}/infer")
async def infer_version(
    model_name: str,
    version: str,
    request: InferenceRequest,
    model_manager: ModelManager = Depends(get_model_manager)
) -> InferenceResponse:
    """Perform inference on specific model version"""
    try:
        # Load specific version if needed
        if not model_manager.is_version_loaded(model_name, version):
            model_manager.load_model_version(model_name, version)

        # Use same inference logic
        return await infer(model_name, request, model_manager)

    except Exception as e:
        logger.error(f"Version inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))