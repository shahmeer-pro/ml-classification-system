"""Inference endpoints for batch and manual predictions."""

from fastapi import APIRouter, Depends

from ml_app.api.dependencies import get_prediction_service
from ml_app.api.schemas import PredictRequest, PredictResponse
from ml_app.services.prediction_service import PredictionService

router = APIRouter(prefix="/api/v1/predict", tags=["Predictions"])


@router.post("", response_model=PredictResponse)
async def make_predictions(
    request: PredictRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictResponse:
    """Execute prediction on feature records using a previously fitted model artifact."""
    res = service.predict(
        artifact_id=request.artifact_id,
        input_data=request.records,
    )

    return PredictResponse(
        artifact_id=request.artifact_id,
        record_count=res.record_count,
        feature_count=res.feature_count,
        predicted_classes=res.predicted_classes,
        predicted_class_names=res.predicted_class_names,
        has_probabilities=res.has_probabilities,
        probabilities=res.probabilities,
        probability_class_labels=res.probability_class_labels,
        has_decision_scores=res.has_decision_scores,
        decision_scores=res.decision_scores,
    )
