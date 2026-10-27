"""FastAPI dependencies and service singletons."""

from ml_app.services.prediction_service import PredictionService, prediction_service
from ml_app.services.training_service import TrainingService, training_service


def get_training_service() -> TrainingService:
    """Dependency provider for TrainingService."""
    return training_service


def get_prediction_service() -> PredictionService:
    """Dependency provider for PredictionService."""
    return prediction_service
