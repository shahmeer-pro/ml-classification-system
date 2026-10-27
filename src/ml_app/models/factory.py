"""Model factory for instantiating scikit-learn estimators and end-to-end pipelines."""

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml_app.exceptions import ModelTrainingError
from ml_app.models.registry import MODEL_REGISTRY, ModelMetadata
from ml_app.preprocessing.pipeline import build_preprocessing_pipeline


def validate_hyperparameters(
    model_meta: ModelMetadata,
    params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate requested hyperparameters against the model's contract.

    Args:
        model_meta: Metadata defining allowed parameters and boundaries.
        params: User-provided parameter dictionary.

    Returns:
        dict[str, Any]: Cleaned, type-cast parameter dictionary with defaults applied.

    Raises:
        ModelTrainingError: If unknown parameters or out-of-bound values are supplied.
    """
    clean_params: dict[str, Any] = {}
    user_params = params or {}

    # Check for unknown parameters
    for p_name in user_params:
        if p_name not in model_meta.parameters:
            raise ModelTrainingError(
                f"Unknown parameter '{p_name}' for model '{model_meta.display_name}'. "
                f"Valid parameters: {list(model_meta.parameters.keys())}"
            )

    # Check bounds, types, and defaults
    for p_name, p_def in model_meta.parameters.items():
        if p_name in user_params and user_params[p_name] is not None:
            raw_val = user_params[p_name]
            try:
                if p_def.param_type == "int":
                    val = int(raw_val)
                elif p_def.param_type == "float":
                    val = float(raw_val)
                elif p_def.param_type == "bool":
                    val = bool(raw_val)
                else:
                    val = str(raw_val)
            except (ValueError, TypeError) as exc:
                raise ModelTrainingError(
                    f"Parameter '{p_name}' must be of type {p_def.param_type}, got {type(raw_val).__name__}."
                ) from exc

            if p_def.min_value is not None and val < p_def.min_value:
                raise ModelTrainingError(
                    f"Parameter '{p_name}' value {val} is below minimum allowed {p_def.min_value}."
                )
            if p_def.max_value is not None and val > p_def.max_value:
                raise ModelTrainingError(
                    f"Parameter '{p_name}' value {val} is above maximum allowed {p_def.max_value}."
                )
            if p_def.options is not None and val not in p_def.options:
                raise ModelTrainingError(
                    f"Parameter '{p_name}' value '{val}' is not one of allowed options {p_def.options}."
                )

            clean_params[p_name] = val
        else:
            clean_params[p_name] = p_def.default

    return clean_params


def instantiate_estimator(
    model_id: str,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> Any:
    """Create a raw scikit-learn estimator instance with validated parameters.

    Args:
        model_id: Model identifier in MODEL_REGISTRY.
        params: Dictionary of hyperparameters.
        random_state: Deterministic random seed for reproducibility.

    Returns:
        BaseEstimator: Configured scikit-learn estimator.
    """
    if model_id not in MODEL_REGISTRY:
        raise ModelTrainingError(
            f"Model '{model_id}' is not recognized. Available: {list(MODEL_REGISTRY.keys())}"
        )

    meta = MODEL_REGISTRY[model_id]
    validated = validate_hyperparameters(meta, params)

    if model_id == "gaussian_nb":
        return GaussianNB(var_smoothing=validated["var_smoothing"])

    if model_id == "logistic_regression":
        penalty = validated.get("penalty", "l2")
        c_val = validated["C"]
        if penalty == "none":
            c_val = 1e9  # effectively unregularized
        return LogisticRegression(
            C=c_val,
            max_iter=validated["max_iter"],
            random_state=random_state,
            solver="lbfgs",
        )

    if model_id == "linear_svm":
        return LinearSVC(
            C=validated["C"],
            max_iter=validated["max_iter"],
            dual="auto",
            random_state=random_state,
        )

    if model_id == "random_forest":
        return RandomForestClassifier(
            n_estimators=validated["n_estimators"],
            max_depth=validated["max_depth"],
            random_state=random_state,
        )

    raise ModelTrainingError(f"Factory handler for '{model_id}' is not implemented.")


def create_classifier_pipeline(
    model_id: str,
    numeric_features: list[str],
    categorical_features: list[str] | None = None,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Create a unified scikit-learn Pipeline with preprocessing and classifier.

    Args:
        model_id: Model registry key.
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.
        params: Model hyperparameters.
        random_state: Deterministic random seed.

    Returns:
        Pipeline: End-to-end scikit-learn pipeline ready for fitting.
    """
    if model_id not in MODEL_REGISTRY:
        raise ModelTrainingError(f"Unknown model identifier '{model_id}'.")

    meta = MODEL_REGISTRY[model_id]
    scale_numeric = meta.requires_scaling

    preprocessor = build_preprocessing_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=scale_numeric,
    )

    estimator = instantiate_estimator(
        model_id=model_id,
        params=params,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )
    return pipeline
