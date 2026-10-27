"""Model training and comparison endpoints."""

import pandas as pd
from fastapi import APIRouter, Depends

from ml_app.api.dependencies import get_training_service
from ml_app.api.schemas import (
    CompareRequest,
    CompareResponse,
    CurveDataSchema,
    MetricsSchema,
    TrainRequest,
    TrainResponse,
)
from ml_app.data.loader import load_builtin_dataset
from ml_app.evaluation.metrics import ClassificationMetricsResult
from ml_app.exceptions import DataValidationError
from ml_app.services.training_service import TrainingService

router = APIRouter(prefix="/api/v1/train", tags=["Training"])


def _format_metrics(metrics: ClassificationMetricsResult) -> MetricsSchema:
    roc_schema = None
    if metrics.roc_curve:
        roc_schema = CurveDataSchema(
            x_values=metrics.roc_curve.x_values,
            y_values=metrics.roc_curve.y_values,
            thresholds=metrics.roc_curve.thresholds,
            score=metrics.roc_curve.score,
        )

    pr_schema = None
    if metrics.pr_curve:
        pr_schema = CurveDataSchema(
            x_values=metrics.pr_curve.x_values,
            y_values=metrics.pr_curve.y_values,
            thresholds=metrics.pr_curve.thresholds,
            score=metrics.pr_curve.score,
        )

    return MetricsSchema(
        accuracy=metrics.accuracy,
        precision_macro=metrics.precision_macro,
        recall_macro=metrics.recall_macro,
        f1_macro=metrics.f1_macro,
        precision_weighted=metrics.precision_weighted,
        recall_weighted=metrics.recall_weighted,
        f1_weighted=metrics.f1_weighted,
        confusion_matrix=metrics.confusion_matrix,
        labels=metrics.labels,
        per_class_metrics=metrics.per_class_metrics,
        support=metrics.support,
        classification_report_str=metrics.classification_report_str,
        is_binary=metrics.is_binary,
        positive_class=metrics.positive_class,
        positive_class_name=metrics.positive_class_name,
        positive_precision=metrics.positive_precision,
        positive_recall=metrics.positive_recall,
        positive_f1=metrics.positive_f1,
        roc_auc=metrics.roc_auc,
        roc_curve=roc_schema,
        pr_curve=pr_schema,
        warnings=metrics.warnings,
    )


@router.post("", response_model=TrainResponse)
async def train_model(
    request: TrainRequest,
    service: TrainingService = Depends(get_training_service),
) -> TrainResponse:
    """Train a classification model and return comprehensive evaluation metrics."""
    if request.use_builtin:
        df, meta = load_builtin_dataset()
        target_col = meta["target_name"]
        class_mapping = meta["class_mapping"]
        pos_class = request.positive_class or meta["positive_class"]
        pos_name = request.positive_class_name or meta["positive_class_name"]
    else:
        if not request.data_records:
            raise DataValidationError("Must provide data_records when use_builtin is false.")
        df = pd.DataFrame(request.data_records)
        target_col = request.target_column
        class_mapping = None
        pos_class = request.positive_class
        pos_name = request.positive_class_name

    res = service.train_model(
        df=df,
        target_column=target_col,
        model_id=request.model_id,
        feature_columns=request.feature_columns,
        hyperparameters=request.hyperparameters,
        test_size=request.test_size,
        random_state=request.random_state,
        class_mapping=class_mapping,
        positive_class=pos_class,
        positive_class_name=pos_name,
    )

    return TrainResponse(
        artifact_id=res.artifact_id,
        model_id=res.model_id,
        display_name=res.display_name,
        hyperparameters=res.hyperparameters,
        train_samples=res.train_samples,
        test_samples=res.test_samples,
        feature_count=res.feature_count,
        supports_probabilities=res.supports_probabilities,
        supports_decision_scores=res.supports_decision_scores,
        metrics=_format_metrics(res.metrics),
        warnings=res.warnings,
    )


@router.post("/compare", response_model=CompareResponse)
async def compare_models(
    request: CompareRequest,
    service: TrainingService = Depends(get_training_service),
) -> CompareResponse:
    """Train multiple models on the exact same split and compare performance."""
    if request.use_builtin:
        df, meta = load_builtin_dataset()
        target_col = meta["target_name"]
        class_mapping = meta["class_mapping"]
        pos_class = request.positive_class or meta["positive_class"]
        pos_name = request.positive_class_name or meta["positive_class_name"]
    else:
        if not request.data_records:
            raise DataValidationError("Must provide data_records when use_builtin is false.")
        df = pd.DataFrame(request.data_records)
        target_col = request.target_column
        class_mapping = None
        pos_class = request.positive_class
        pos_name = request.positive_class_name

    res = service.compare_models(
        df=df,
        target_column=target_col,
        model_ids=request.model_ids,
        feature_columns=request.feature_columns,
        hyperparameters_by_model=request.hyperparameters_by_model,
        test_size=request.test_size,
        random_state=request.random_state,
        class_mapping=class_mapping,
        positive_class=pos_class,
        positive_class_name=pos_name,
    )

    formatted_results = [
        TrainResponse(
            artifact_id=item.artifact_id,
            model_id=item.model_id,
            display_name=item.display_name,
            hyperparameters=item.hyperparameters,
            train_samples=item.train_samples,
            test_samples=item.test_samples,
            feature_count=item.feature_count,
            supports_probabilities=item.supports_probabilities,
            supports_decision_scores=item.supports_decision_scores,
            metrics=_format_metrics(item.metrics),
            warnings=item.warnings,
        )
        for item in res.results
    ]

    return CompareResponse(
        train_samples=res.train_samples,
        test_samples=res.test_samples,
        feature_count=res.feature_count,
        split_random_state=res.split_random_state,
        test_size=res.test_size,
        results=formatted_results,
        comparison_table=res.comparison_table,
    )
