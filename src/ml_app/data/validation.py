"""Data validation rules and summary generation."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ml_app.config import settings
from ml_app.exceptions import DataValidationError
from ml_app.logging_config import logger


@dataclass
class ValidationSummary:
    """Structured report detailing dataset health, dimensions, and validation warnings."""

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
    target_classes: list[Any] = field(default_factory=list)
    class_distribution: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_dataset(
    df: pd.DataFrame,
    target_column: str | None = None,
    feature_columns: list[str] | None = None,
) -> ValidationSummary:
    """Validate a DataFrame against guardrails, schema integrity, and ML requirements.

    Args:
        df: Input pandas DataFrame.
        target_column: Optional name of the target column.
        feature_columns: Optional list of selected feature columns.

    Returns:
        ValidationSummary: Detailed analysis of dataset health.

    Raises:
        DataValidationError: If any critical constraint is violated.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Row and Column Limits
    n_rows, n_cols = df.shape
    if n_rows == 0:
        raise DataValidationError("The dataset has 0 rows.")

    if n_rows < settings.MIN_ROWS:
        raise DataValidationError(
            f"Dataset has only {n_rows} rows. A minimum of {settings.MIN_ROWS} rows "
            "is required for meaningful model training and evaluation."
        )

    if n_rows > settings.MAX_ROWS:
        raise DataValidationError(
            f"Dataset has {n_rows} rows, exceeding the limit of {settings.MAX_ROWS} rows."
        )

    if n_cols > settings.MAX_COLS:
        raise DataValidationError(
            f"Dataset has {n_cols} columns, exceeding the maximum limit of {settings.MAX_COLS}."
        )

    # 2. Check duplicate column names
    col_names = [str(col) for col in df.columns]
    if len(col_names) != len(set(col_names)):
        duplicates = [col for col in col_names if col_names.count(col) > 1]
        raise DataValidationError(
            f"Duplicate column names detected: {sorted(list(set(duplicates)))}. "
            "Please ensure all column names are unique."
        )

    # 3. Categorize column types and compute missingness
    dtypes = {str(col): str(df[col].dtype) for col in df.columns}
    missing_counts = {str(col): int(df[col].isna().sum()) for col in df.columns}
    missing_percentages = {
        str(col): round(float(df[col].isna().mean() * 100), 2) for col in df.columns
    }

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    constant_columns: list[str] = []

    for col in df.columns:
        col_str = str(col)
        # Check if constant
        unique_count = df[col].nunique(dropna=True)
        if unique_count <= 1:
            constant_columns.append(col_str)
            warnings.append(f"Column '{col_str}' has {unique_count} unique value(s) (constant).")

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col_str)
            # Check for infinite values in numeric columns
            inf_count = int(np.isinf(df[col]).sum())
            if inf_count > 0:
                warnings.append(
                    f"Column '{col_str}' contains {inf_count} infinite values which will be handled."
                )
        else:
            categorical_columns.append(col_str)

    # 4. Target Column Validation (if specified)
    target_classes: list[Any] = []
    class_distribution: dict[str, int] = {}

    if target_column is not None:
        if target_column not in df.columns:
            raise DataValidationError(
                f"Specified target column '{target_column}' is not present in the dataset."
            )

        # Drop missing targets for validation analysis
        target_series = df[target_column].dropna()
        if target_series.empty:
            raise DataValidationError(
                f"Target column '{target_column}' contains only missing values."
            )

        unique_targets = sorted(target_series.unique().tolist(), key=lambda v: str(v))
        target_classes = unique_targets
        class_distribution = {str(k): int(v) for k, v in target_series.value_counts().items()}

        if len(unique_targets) < 2:
            raise DataValidationError(
                f"Classification requires at least 2 distinct target classes, but "
                f"'{target_column}' only contains 1 class: {unique_targets}."
            )

        # Check for minimum examples per class (needed for stratified split)
        min_class_samples = min(class_distribution.values())
        if min_class_samples < 2:
            warnings.append(
                f"The smallest class has only {min_class_samples} sample(s). "
                "Stratified splitting may require at least 2 samples per class."
            )

        # Severe class imbalance check
        max_class_samples = max(class_distribution.values())
        if min_class_samples > 0 and (max_class_samples / min_class_samples) > 20:
            warnings.append(
                "Severe class imbalance detected (>20:1 ratio). "
                "Accuracy may be misleading; inspect Precision, Recall, and F1."
            )

    # 5. Feature Columns Validation (if specified)
    if feature_columns is not None:
        if not feature_columns:
            raise DataValidationError("At least one feature column must be selected.")

        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise DataValidationError(f"Selected features not in dataset: {missing_features}")

        if target_column is not None and target_column in feature_columns:
            raise DataValidationError(
                f"Target column '{target_column}' cannot also be included in feature columns."
            )

    summary = ValidationSummary(
        is_valid=len(errors) == 0,
        num_rows=n_rows,
        num_cols=n_cols,
        column_names=col_names,
        dtypes=dtypes,
        missing_counts=missing_counts,
        missing_percentages=missing_percentages,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        constant_columns=constant_columns,
        target_column=target_column,
        target_classes=target_classes,
        class_distribution=class_distribution,
        warnings=warnings,
        errors=errors,
    )

    logger.info(
        "Validation completed for dataset (rows=%d, cols=%d, warnings=%d, valid=%s)",
        n_rows,
        n_cols,
        len(warnings),
        summary.is_valid,
    )
    return summary
