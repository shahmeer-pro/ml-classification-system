"""Training and comparison service with in-memory pipeline management."""

import datetime
import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from ml_app.evaluation.metrics import (
    ClassificationMetricsResult,
    calculate_classification_metrics,
)
from ml_app.exceptions import ModelNotFoundError, ModelTrainingError
from ml_app.logging_config import logger
from ml_app.models.factory import create_classifier_pipeline
from ml_app.models.registry import MODEL_REGISTRY, ModelMetadata
from ml_app.preprocessing.pipeline import prepare_and_split_data


@dataclass
class ModelArtifact:
    """In-memory fitted pipeline and associated training context."""

    artifact_id: str
    model_id: str
    display_name: str
    pipeline: Pipeline
    feature_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    target_column: str
    target_classes: list[Any]
    class_mapping: dict[Any, str] | None
    positive_class: Any | None
    positive_class_name: str | None
    hyperparameters: dict[str, Any]
    created_at: str
    metrics: ClassificationMetricsResult
    supports_probabilities: bool
    supports_decision_scores: bool


@dataclass
class TrainingResult:
    """Output summary of a training run."""

    artifact_id: str
    model_id: str
    display_name: str
    hyperparameters: dict[str, Any]
    metrics: ClassificationMetricsResult
    train_samples: int
    test_samples: int
    feature_count: int
    supports_probabilities: bool
    supports_decision_scores: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class ModelComparisonResult:
    """Outcome of training multiple models on the exact same data split."""

    train_samples: int
    test_samples: int
    feature_count: int
    split_random_state: int
    test_size: float
    results: list[TrainingResult]
    comparison_table: list[dict[str, Any]]


class TrainingService:
    """Orchestrates model training, evaluation, comparison, and in-memory registry."""

    def __init__(self) -> None:
        self._artifacts: dict[str, ModelArtifact] = {}

    def get_artifact(self, artifact_id: str) -> ModelArtifact:
        """Retrieve an in-memory trained model artifact by its unique ID."""
        if artifact_id not in self._artifacts:
            raise ModelNotFoundError(
                f"Model artifact '{artifact_id}' not found. In-memory models reset upon restart."
            )
        return self._artifacts[artifact_id]

    def list_artifacts(self) -> list[dict[str, Any]]:
        """List summary descriptors for all currently retained in-memory models."""
        return [
            {
                "artifact_id": art.artifact_id,
                "model_id": art.model_id,
                "display_name": art.display_name,
                "accuracy": art.metrics.accuracy,
                "f1_macro": art.metrics.f1_macro,
                "created_at": art.created_at,
            }
            for art in self._artifacts.values()
        ]

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
    ) -> TrainingResult:
        """Split data, fit an end-to-end pipeline, evaluate on isolated test set, and store artifact."""
        if model_id not in MODEL_REGISTRY:
            raise ModelTrainingError(
                f"Model '{model_id}' is not supported. Choose from: {list(MODEL_REGISTRY.keys())}"
            )

        meta: ModelMetadata = MODEL_REGISTRY[model_id]
        warnings: list[str] = []

        # 1. Split data (strictly before fitting transformers)
        split = prepare_and_split_data(
            df=df,
            target_column=target_column,
            feature_columns=feature_columns,
            test_size=test_size,
            random_state=random_state,
        )

        # 2. Build pipeline
        pipeline = create_classifier_pipeline(
            model_id=model_id,
            numeric_features=split.numeric_features,
            categorical_features=split.categorical_features,
            params=hyperparameters,
            random_state=random_state,
        )

        # 3. Fit pipeline on training data only
        try:
            pipeline.fit(split.X_train, split.y_train)
        except Exception as exc:
            logger.error("Model fitting failed for '%s': %s", model_id, exc)
            raise ModelTrainingError(f"Model training failed: {exc}") from exc

        # 4. Predict on test partition
        y_test_pred = pipeline.predict(split.X_test)

        # Extract probabilities or decision scores for diagnostic curves
        y_scores = None
        if meta.supports_probabilities and hasattr(pipeline, "predict_proba"):
            try:
                y_scores = pipeline.predict_proba(split.X_test)
            except Exception as exc:
                warnings.append(f"Probability estimation failed: {exc}")
        elif meta.supports_decision_scores and hasattr(pipeline, "decision_function"):
            try:
                y_scores = pipeline.decision_function(split.X_test)
            except Exception as exc:
                warnings.append(f"Decision function calculation failed: {exc}")

        # 5. Calculate metrics
        metrics = calculate_classification_metrics(
            y_true=split.y_test.to_numpy(),
            y_pred=y_test_pred,
            y_scores=y_scores,
            class_labels=split.target_classes,
            positive_class=positive_class,
            positive_class_name=positive_class_name,
        )

        # 6. Generate ID and save in-memory artifact
        artifact_id = str(uuid.uuid4())
        created_at = datetime.datetime.now(datetime.UTC).isoformat()

        artifact = ModelArtifact(
            artifact_id=artifact_id,
            model_id=model_id,
            display_name=meta.display_name,
            pipeline=pipeline,
            feature_names=split.feature_names,
            numeric_features=split.numeric_features,
            categorical_features=split.categorical_features,
            target_column=target_column,
            target_classes=split.target_classes,
            class_mapping=class_mapping,
            positive_class=positive_class,
            positive_class_name=positive_class_name,
            hyperparameters=hyperparameters or {},
            created_at=created_at,
            metrics=metrics,
            supports_probabilities=meta.supports_probabilities,
            supports_decision_scores=meta.supports_decision_scores,
        )

        self._artifacts[artifact_id] = artifact
        logger.info("Successfully trained and stored model artifact '%s'", artifact_id)

        return TrainingResult(
            artifact_id=artifact_id,
            model_id=model_id,
            display_name=meta.display_name,
            hyperparameters=hyperparameters or {},
            metrics=metrics,
            train_samples=len(split.X_train),
            test_samples=len(split.X_test),
            feature_count=len(split.feature_names),
            supports_probabilities=meta.supports_probabilities,
            supports_decision_scores=meta.supports_decision_scores,
            warnings=warnings + metrics.warnings,
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
    ) -> ModelComparisonResult:
        """Train multiple models on the EXACT same split and return comparative metrics."""
        if not model_ids:
            raise ModelTrainingError("At least one model must be selected for comparison.")

        # 1. Single split guarantees identical train/test partitions for every model
        split = prepare_and_split_data(
            df=df,
            target_column=target_column,
            feature_columns=feature_columns,
            test_size=test_size,
            random_state=random_state,
        )

        results: list[TrainingResult] = []
        comparison_table: list[dict[str, Any]] = []

        params_map = hyperparameters_by_model or {}

        for m_id in model_ids:
            if m_id not in MODEL_REGISTRY:
                continue

            meta = MODEL_REGISTRY[m_id]
            m_params = params_map.get(m_id, {})

            # Create fresh pipeline for each model
            pipeline = create_classifier_pipeline(
                model_id=m_id,
                numeric_features=split.numeric_features,
                categorical_features=split.categorical_features,
                params=m_params,
                random_state=random_state,
            )

            pipeline.fit(split.X_train, split.y_train)
            y_pred = pipeline.predict(split.X_test)

            y_scores = None
            if meta.supports_probabilities and hasattr(pipeline, "predict_proba"):
                try:
                    y_scores = pipeline.predict_proba(split.X_test)
                except Exception:
                    pass
            elif meta.supports_decision_scores and hasattr(pipeline, "decision_function"):
                try:
                    y_scores = pipeline.decision_function(split.X_test)
                except Exception:
                    pass

            metrics = calculate_classification_metrics(
                y_true=split.y_test.to_numpy(),
                y_pred=y_pred,
                y_scores=y_scores,
                class_labels=split.target_classes,
                positive_class=positive_class,
                positive_class_name=positive_class_name,
            )

            artifact_id = str(uuid.uuid4())
            created_at = datetime.datetime.now(datetime.UTC).isoformat()

            artifact = ModelArtifact(
                artifact_id=artifact_id,
                model_id=m_id,
                display_name=meta.display_name,
                pipeline=pipeline,
                feature_names=split.feature_names,
                numeric_features=split.numeric_features,
                categorical_features=split.categorical_features,
                target_column=target_column,
                target_classes=split.target_classes,
                class_mapping=class_mapping,
                positive_class=positive_class,
                positive_class_name=positive_class_name,
                hyperparameters=m_params,
                created_at=created_at,
                metrics=metrics,
                supports_probabilities=meta.supports_probabilities,
                supports_decision_scores=meta.supports_decision_scores,
            )
            self._artifacts[artifact_id] = artifact

            tr = TrainingResult(
                artifact_id=artifact_id,
                model_id=m_id,
                display_name=meta.display_name,
                hyperparameters=m_params,
                metrics=metrics,
                train_samples=len(split.X_train),
                test_samples=len(split.X_test),
                feature_count=len(split.feature_names),
                supports_probabilities=meta.supports_probabilities,
                supports_decision_scores=meta.supports_decision_scores,
                warnings=metrics.warnings,
            )
            results.append(tr)

            row: dict[str, Any] = {
                "Model": meta.display_name,
                "Accuracy": metrics.accuracy,
                "Macro Precision": metrics.precision_macro,
                "Macro Recall": metrics.recall_macro,
                "Macro F1": metrics.f1_macro,
                "Weighted F1": metrics.f1_weighted,
                "ROC AUC": metrics.roc_auc if metrics.roc_auc is not None else "N/A",
            }
            if metrics.positive_f1 is not None:
                lbl = metrics.positive_class_name or f"Class {metrics.positive_class}"
                row[f"{lbl} Precision"] = metrics.positive_precision
                row[f"{lbl} Recall"] = metrics.positive_recall
                row[f"{lbl} F1"] = metrics.positive_f1

            comparison_table.append(row)

        return ModelComparisonResult(
            train_samples=len(split.X_train),
            test_samples=len(split.X_test),
            feature_count=len(split.feature_names),
            split_random_state=random_state,
            test_size=test_size,
            results=results,
            comparison_table=comparison_table,
        )


training_service = TrainingService()
