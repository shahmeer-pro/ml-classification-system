"""Model discovery and metadata routes."""

from fastapi import APIRouter

from ml_app.api.schemas import ModelInfo, ParameterInfo
from ml_app.models.registry import MODEL_REGISTRY

router = APIRouter(prefix="/api/v1/models", tags=["Models"])


@router.get("", response_model=list[ModelInfo])
async def list_available_models() -> list[ModelInfo]:
    """Return metadata and hyperparameter contracts for all registered classifiers."""
    models: list[ModelInfo] = []
    for m in MODEL_REGISTRY.values():
        params = {
            p_name: ParameterInfo(
                name=p.name,
                param_type=p.param_type,
                default=p.default,
                options=p.options,
                min_value=p.min_value,
                max_value=p.max_value,
                description=p.description,
            )
            for p_name, p in m.parameters.items()
        }
        models.append(
            ModelInfo(
                model_id=m.model_id,
                display_name=m.display_name,
                supports_probabilities=m.supports_probabilities,
                supports_decision_scores=m.supports_decision_scores,
                requires_scaling=m.requires_scaling,
                description=m.description,
                parameters=params,
            )
        )
    return models
