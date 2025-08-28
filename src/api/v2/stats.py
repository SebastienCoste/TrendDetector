from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

from .models import StatsResponse
from ...core.model_manager import get_model_manager, ModelManager

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/models/{model_name}/stats")
async def get_model_stats(
    model_name: str,
    model_manager: ModelManager = Depends(get_model_manager)
) -> StatsResponse:
    """Get model statistics"""
    try:
        # Ensure model exists
        if not model_manager.is_model_loaded(model_name):
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        classifier = model_manager.get_classifier(model_name)
        model_stats = classifier.get_model_stats()
        memory_stats = classifier.trend_memory.get_memory_stats()

        # Combine statistics
        statistics = {
            "model_stats": {
                "accuracy": model_stats.accuracy,
                "prediction_count": model_stats.prediction_count,
                "last_drift_detection": model_stats.last_drift_detection,
                "memory_size": model_stats.memory_size,
                "is_fitted": model_stats.is_fitted,
                "trend_distribution": model_stats.trend_distribution,
                "last_update": model_stats.last_update.isoformat(),
                "model_created": model_stats.model_created.isoformat()
            },
            "memory_stats": memory_stats,
            "gpu_info": None
        }

        # Add GPU information if available
        try:
            if classifier.gpu_manager and classifier.gpu_manager.is_gpu_enabled:
                gpu_info = classifier.gpu_manager.get_memory_info()
                statistics["gpu_info"] = gpu_info
        except Exception:
            pass

        return StatsResponse(
            model_name=model_name,
            model_version=classifier.model_version,
            statistics=statistics
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")