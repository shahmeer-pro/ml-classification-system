"""Model registry and factory package."""

from ml_app.models.factory import create_classifier_pipeline
from ml_app.models.registry import MODEL_REGISTRY, ModelMetadata

__all__ = [
    "MODEL_REGISTRY",
    "ModelMetadata",
    "create_classifier_pipeline",
]
