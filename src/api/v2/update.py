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
    logger.info(f"Received update request for model {model_name}")
    try:
        # Ensure model exists
        if not model_manager.is_model_loaded(model_name):
            if not model_manager.load_model(model_name):
                model_manager.create_model(model_name)

        classifier = model_manager.get_classifier(model_name)
        processed_updates = 0
        drift_detected_any = False
        logger.info(f"Model {model_name} locked and loaded")

        for update in request.updates:
            try:
                # Extract update data
                embedding_vector = np.array(update['embedding_vector'], dtype=np.float32)
                actual_trend = update['actual_trend']
                predicted_trend = update.get('predicted_trend')
                velocity_features = update.get('velocity_features', {})
                timestamp = update.get('timestamp')

                # Validate inputs
                validate_embedding_vector(embedding_vector)
                validate_trend_label(actual_trend)
                validate_trend_label(predicted_trend)

                # Update model
                drift_detected = classifier.update_with_feedback(
                    content_vector=embedding_vector,
                    actual_trend=actual_trend,
                    predicted_trend=predicted_trend,
                    velocity_features=velocity_features,
                    timestamp=timestamp
                )

                if drift_detected:
                    drift_detected_any = True

                processed_updates += 1
                logger.info(f"Processed update #{processed_updates}")

            except Exception as e:
                logger.warning(f"Failed to process update: {e}")
                continue

        # Save model if auto_save is enabled and there's been a drift
        if drift_detected_any and model_manager.config.storage_config.auto_save:
            model_manager.save_model(model_name)
        logger.info(f"Successfully processed {processed_updates} updates")
        return UpdateResponse(
            processed_updates=processed_updates,
            drift_detected=drift_detected_any,
            model_version=classifier.model_version,
            message=f"Successfully processed {processed_updates} updates"
        )

    except Exception as e:
        logger.error(f"Update error: {e}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")