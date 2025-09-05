from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import numpy as np
import logging

from .models import UpdateRequest, UpdateResponse
from ...core.model_manager import get_model_manager, ModelManager
from ...utils.validators import validate_embedding_vector, validate_trend_label

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/models/{model_name}/update")
async def update_model(
    model_name: str,
    request: UpdateRequest,
    model_manager: ModelManager = Depends(get_model_manager)
) -> UpdateResponse:
    """Update model with feedback data"""
    try:
        # Determine model type
        model_type = request.model_type or model_manager.config.model_settings.type

        # Ensure model exists
        if not model_manager.is_model_loaded(model_name):
            if not model_manager.load_model(model_name, model_type):
                model_manager.create_model(model_name, model_type)

        model = model_manager.get_model(model_name, model_type)
        processed_updates = 0
        drift_detected_any = False

        for update in request.updates:
            try:
                # Extract update data
                embedding_vector = np.array(update['embedding_vector'], dtype=np.float32)
                timestamp = update.get('timestamp')

                # Validate embedding
                validate_embedding_vector(embedding_vector)

                # Handle different model types
                if model_type == "classification":
                    actual_trend = update.get('actual_trend')
                    predicted_trend = update.get('predicted_trend')
                    
                    if actual_trend is None:
                        logger.warning("Skipping update: actual_trend required for classification model")
                        continue
                    
                    validate_trend_label(actual_trend)
                    
                    # Update model
                    drift_detected = model.learn(
                        features=embedding_vector,
                        target=actual_trend,
                        predicted_value=predicted_trend,
                        timestamp=timestamp
                    )
                else:  # regression
                    actual_score = update.get('actual_score')
                    predicted_score = update.get('predicted_score')
                    
                    if actual_score is None:
                        logger.warning("Skipping update: actual_score required for regression model")
                        continue
                    
                    if not isinstance(actual_score, (int, float)):
                        logger.warning("Skipping update: actual_score must be numeric")
                        continue
                    
                    # Update model
                    drift_detected = model.learn(
                        features=embedding_vector,
                        target=float(actual_score),
                        predicted_value=predicted_score,
                        timestamp=timestamp
                    )

                if drift_detected:
                    drift_detected_any = True

                processed_updates += 1

            except Exception as e:
                logger.warning(f"Failed to process update: {e}")
                continue

        # Save model if auto_save is enabled
        if model_manager.config.storage_config.auto_save:
            model_manager.save_model(model_name)

        return UpdateResponse(
            processed_updates=processed_updates,
            drift_detected=drift_detected_any,
            model_version=model.model_version,
            message=f"Successfully processed {processed_updates} updates for {model_type} model"
        )

    except Exception as e:
        logger.error(f"Update error: {e}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")