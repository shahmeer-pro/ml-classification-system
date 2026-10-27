"""Preprocessing pipeline construction and data splitting logic."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_app.exceptions import PreprocessingError
from ml_app.logging_config import logger


@dataclass
class PreprocessedDataSplit:
    """Container holding split data partitions and feature metadata."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    target_classes: list[Any]
    stratified: bool


def build_preprocessing_pipeline(
    numeric_features: list[str],
    categorical_features: list[str] | None = None,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Construct a scikit-learn ColumnTransformer for numeric and categorical features.

    Implements:
    - Median imputation for missing numeric values.
    - Optional standard scaling for numeric features.
    - Most frequent imputation for categorical features.
    - One-hot encoding for categorical features with unseen categories ignored.

    Args:
        numeric_features: List of column names with numeric types.
        categorical_features: List of column names with categorical/object types.
        scale_numeric: Whether to apply StandardScaler to numerical features.

    Returns:
        ColumnTransformer: An un-fitted scikit-learn preprocessor.
    """
    transformers: list[tuple[str, Any, list[str]]] = []
    safe_numeric = [str(f) for f in numeric_features] if numeric_features else []
    safe_categorical = [str(f) for f in categorical_features] if categorical_features else []

    if safe_numeric:
        num_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        num_pipeline = Pipeline(steps=num_steps)
        transformers.append(("numeric", num_pipeline, safe_numeric))

    if safe_categorical:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", cat_pipeline, safe_categorical))

    if not transformers:
        raise PreprocessingError("At least one numeric or categorical feature must be specified.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def prepare_and_split_data(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
) -> PreprocessedDataSplit:
    """Clean, split, and partition data into train and test sets without leakage.

    Splits the dataset before any transformers are fitted to ensure complete isolation
    of test data.

    Args:
        df: Input DataFrame.
        target_column: Target column name.
        feature_columns: Specific feature columns to include (or all remaining columns).
        test_size: Proportion of dataset allocated to the test split (e.g., 0.2).
        random_state: Random seed for deterministic reproducibility.

    Returns:
        PreprocessedDataSplit: Object containing X_train, X_test, y_train, y_test, and metadata.

    Raises:
        PreprocessingError: If input data is malformed or cannot be split.
    """
    if target_column not in df.columns:
        raise PreprocessingError(f"Target column '{target_column}' not found in data.")

    target_column = str(target_column)
    clean_df = df.dropna(subset=[target_column]).copy()
    clean_df.columns = [str(c) for c in clean_df.columns]
    if clean_df.empty:
        raise PreprocessingError("All rows have missing target values.")

    # Determine features
    if feature_columns is None:
        feature_columns = [col for col in clean_df.columns if col != target_column]
    else:
        feature_columns = [str(col) for col in feature_columns if str(col) != target_column]

    if not feature_columns:
        raise PreprocessingError("No feature columns available after excluding target.")

    # Validate feature presence
    missing_feats = [col for col in feature_columns if col not in clean_df.columns]
    if missing_feats:
        raise PreprocessingError(f"Feature columns missing from dataframe: {missing_feats}")

    X = clean_df[feature_columns].copy()
    y = clean_df[target_column].copy()

    # Replace +/- inf in numeric columns with NaN so imputer handles them gracefully
    num_cols = [col for col in feature_columns if pd.api.types.is_numeric_dtype(X[col])]
    cat_cols = [col for col in feature_columns if col not in num_cols]

    for col in num_cols:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)

    # Determine if stratified split is feasible
    # Stratification requires at least 2 samples per class and at least 1 in each fold
    class_counts = y.value_counts()
    min_count = class_counts.min()
    can_stratify = min_count >= 2 and len(class_counts) >= 2

    stratify_target = y if can_stratify else None
    if not can_stratify:
        logger.warning(
            "Stratified split disabled: smallest class has %d sample(s). Performing standard split.",
            min_count,
        )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target,
        )
    except Exception as exc:
        raise PreprocessingError(f"Failed to split data: {exc}") from exc

    unique_classes = sorted(y.unique().tolist(), key=lambda v: str(v))

    logger.info(
        "Data split complete: Train=%d, Test=%d, Features=%d, Stratified=%s",
        len(X_train),
        len(X_test),
        len(feature_columns),
        can_stratify,
    )

    return PreprocessedDataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_columns,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        target_classes=unique_classes,
        stratified=can_stratify,
    )
