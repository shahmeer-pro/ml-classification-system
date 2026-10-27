"""Pydantic schemas for REST API request and response validation."""

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "1.0.0"
    service: str = "ML Classification System API"


class BuiltinDatasetResponse(BaseModel):
    """Metadata response for built-in breast cancer dataset."""

    name: str
    source: str
    samples_count: int
    features_count: int
    feature_names: list[str]
    target_name: str
    class_mapping: dict[int, str]
    positive_class: int
    positive_class_name: str
    description: str
    educational_disclaimer: str


class ValidationResponse(BaseModel):
    """Validation report summary response."""

    is_valid: bool
    num_rows: int
    num_cols: int
    column_names: list[str]
    dtypes: dict[str, str]
    missing_counts: dict[str, int]
    missing_percentages: dict[str, float]
    numeric_columns: list[str]
    categorical_columns: list[str]
    constant_columns: list[str]
    target_column: str | None = None
    target_classes: list[Any] = Field(default_factory=list)
    class_distribution: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ParameterInfo(BaseModel):
    """Metadata for a single hyperparameter."""

    name: str
    param_type: str
    default: Any
    options: list[Any] | None = None
    min_value: float | None = None
    max_value: float | None = None
    description: str = ""


class ModelInfo(BaseModel):
    """Metadata for an algorithm supported by the registry."""

    model_id: str
    display_name: str
    supports_probabilities: bool
    supports_decision_scores: bool
    requires_scaling: bool
    description: str
    parameters: dict[str, ParameterInfo]


class CurveDataSchema(BaseModel):
    """ROC or PR curve coordinate data."""

    x_values: list[float]
    y_values: list[float]
    thresholds: list[float]
    score: float | None = None


class MetricsSchema(BaseModel):
    """Detailed evaluation metrics collection."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    confusion_matrix: list[list[int]]
    labels: list[str]
    per_class_metrics: dict[str, dict[str, float]]
    support: dict[str, int]
    classification_report_str: str

    is_binary: bool
    positive_class: str | None = None
    positive_class_name: str | None = None
    positive_precision: float | None = None
    positive_recall: float | None = None
    positive_f1: float | None = None

    roc_auc: float | None = None
    roc_curve: CurveDataSchema | None = None
    pr_curve: CurveDataSchema | None = None
    warnings: list[str] = Field(default_factory=list)


class TrainRequest(BaseModel):
    """Request payload for model training."""

    use_builtin: bool = True
    data_records: list[dict[str, Any]] | None = None
    target_column: str = "target"
    feature_columns: list[str] | None = None
    model_id: str = "logistic_regression"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    test_size: float = Field(default=0.20, ge=0.05, le=0.50)
    random_state: int = Field(default=42, ge=0)
    positive_class: Any | None = None
    positive_class_name: str | None = None


class TrainResponse(BaseModel):
    """Response payload following model training."""

    artifact_id: str
    model_id: str
    display_name: str
    hyperparameters: dict[str, Any]
    train_samples: int
    test_samples: int
    feature_count: int
    supports_probabilities: bool
    supports_decision_scores: bool
    metrics: MetricsSchema
    warnings: list[str] = Field(default_factory=list)


class CompareRequest(BaseModel):
    """Request payload for comparing multiple classifiers on the same split."""

    use_builtin: bool = True
    data_records: list[dict[str, Any]] | None = None
    target_column: str = "target"
    feature_columns: list[str] | None = None
    model_ids: list[str] = Field(
        default_factory=lambda: ["gaussian_nb", "logistic_regression", "linear_svm"]
    )
    hyperparameters_by_model: dict[str, dict[str, Any]] = Field(default_factory=dict)
    test_size: float = Field(default=0.20, ge=0.05, le=0.50)
    random_state: int = Field(default=42, ge=0)
    positive_class: Any | None = None
    positive_class_name: str | None = None


class CompareResponse(BaseModel):
    """Response payload for multi-model comparison."""

    train_samples: int
    test_samples: int
    feature_count: int
    split_random_state: int
    test_size: float
    results: list[TrainResponse]
    comparison_table: list[dict[str, Any]]


class PredictRequest(BaseModel):
    """Request payload for inference."""

    artifact_id: str
    records: list[dict[str, Any]]


class PredictResponse(BaseModel):
    """Response payload for inference."""

    artifact_id: str
    record_count: int
    feature_count: int
    predicted_classes: list[Any]
    predicted_class_names: list[str] | None = None
    has_probabilities: bool
    probabilities: list[list[float]] | None = None
    probability_class_labels: list[str] | None = None
    has_decision_scores: bool
    decision_scores: list[float] | None = None


class ErrorResponse(BaseModel):
    """Standardized error response."""

    error: str
    message: str
    status_code: int
