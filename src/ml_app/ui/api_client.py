"""Client for interacting with FastAPI service, with standalone Python fallback."""

from typing import Any

import httpx
import pandas as pd

from ml_app.config import settings
from ml_app.logging_config import logger
from ml_app.models.registry import MODEL_REGISTRY
from ml_app.services.prediction_service import prediction_service
from ml_app.services.training_service import training_service


class ClassificationApiClient:
    """Hybrid client communicating with FastAPI when available, or service layer directly."""

    def __init__(self, base_url: str | None = None, timeout: float = 10.0) -> None:
        self.base_url = (base_url or settings.API_BASE_URL).rstrip("/")
        self.timeout = timeout
        self._is_backend_online: bool | None = None

    def check_health(self) -> dict[str, Any]:
        """Check if FastAPI backend is reachable using cached lookup."""
        from ml_app.ui.cache import check_api_health_cached

        res = check_api_health_cached(self.base_url)
        self._is_backend_online = res.get("status") == "healthy"
        return res

    @property
    def is_api_online(self) -> bool:
        """Return whether external FastAPI backend is accessible."""
        if self._is_backend_online is None:
            self.check_health()
        return bool(self._is_backend_online)

    def get_builtin_dataset_metadata(self) -> dict[str, Any]:
        """Retrieve built-in dataset metadata."""
        if self.is_api_online:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.get(f"{self.base_url}/api/v1/datasets/builtin")
                    if resp.status_code == 200:
                        return resp.json()
            except Exception as exc:
                logger.warning("API call failed, falling back to local service: %s", exc)

        from ml_app.ui.cache import get_cached_builtin_dataset

        _, meta = get_cached_builtin_dataset()
        return meta

    def get_available_models(self) -> list[dict[str, Any]]:
        """Retrieve model metadata from registry."""
        if self.is_api_online:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.get(f"{self.base_url}/api/v1/models")
                    if resp.status_code == 200:
                        return resp.json()
            except Exception as exc:
                logger.warning("API call failed, falling back to local registry: %s", exc)

        return [
            {
                "model_id": m.model_id,
                "display_name": m.display_name,
                "supports_probabilities": m.supports_probabilities,
                "supports_decision_scores": m.supports_decision_scores,
                "requires_scaling": m.requires_scaling,
                "description": m.description,
                "parameters": {
                    p_name: {
                        "name": p.name,
                        "param_type": p.param_type,
                        "default": p.default,
                        "options": p.options,
                        "min_value": p.min_value,
                        "max_value": p.max_value,
                        "description": p.description,
                    }
                    for p_name, p in m.parameters.items()
                },
            }
            for m in MODEL_REGISTRY.values()
        ]

    def validate_csv_bytes(
        self,
        file_bytes: bytes,
        filename: str = "upload.csv",
        target_column: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Validate uploaded CSV content with caching."""
        from ml_app.ui.cache import get_cached_validation_summary

        return get_cached_validation_summary(
            file_bytes=file_bytes, filename=filename, target_column=target_column
        )

    def train_model(
        self,
        df: pd.DataFrame,
        target_column: str,
        model_id: str,
        feature_columns: list[str] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        test_size: float = 0.20,
        random_state: int = 42,
        class_mapping: dict[Any, str] | None = None,
        positive_class: Any | None = None,
        positive_class_name: str | None = None,
    ) -> Any:
        """Train a classifier and return evaluation results."""
        return training_service.train_model(
            df=df,
            target_column=target_column,
            model_id=model_id,
            feature_columns=feature_columns,
            hyperparameters=hyperparameters,
            test_size=test_size,
            random_state=random_state,
            class_mapping=class_mapping,
            positive_class=positive_class,
            positive_class_name=positive_class_name,
        )

    def compare_models(
        self,
        df: pd.DataFrame,
        target_column: str,
        model_ids: list[str],
        feature_columns: list[str] | None = None,
        hyperparameters_by_model: dict[str, dict[str, Any]] | None = None,
        test_size: float = 0.20,
        random_state: int = 42,
        class_mapping: dict[Any, str] | None = None,
        positive_class: Any | None = None,
        positive_class_name: str | None = None,
    ) -> Any:
        """Compare multiple classifiers on identical data splits."""
        return training_service.compare_models(
            df=df,
            target_column=target_column,
            model_ids=model_ids,
            feature_columns=feature_columns,
            hyperparameters_by_model=hyperparameters_by_model,
            test_size=test_size,
            random_state=random_state,
            class_mapping=class_mapping,
            positive_class=positive_class,
            positive_class_name=positive_class_name,
        )

    def predict(
        self,
        artifact_id: str,
        input_data: pd.DataFrame | list[dict[str, Any]],
    ) -> Any:
        """Run predictions on new feature records."""
        return prediction_service.predict(
            artifact_id=artifact_id,
            input_data=input_data,
        )
