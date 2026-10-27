"""Tests for CSV parsing, dataset validation rules, and guardrails."""

import numpy as np
import pandas as pd
import pytest

from ml_app.data.loader import load_csv_from_bytes
from ml_app.data.validation import validate_dataset
from ml_app.exceptions import DataValidationError


def test_valid_numeric_df(synthetic_binary_df: pd.DataFrame):
    summary = validate_dataset(synthetic_binary_df, target_column="target")
    assert summary.is_valid
    assert summary.num_rows == 100
    assert summary.num_cols == 7
    assert len(summary.numeric_columns) == 7
    assert summary.target_classes == [0, 1]


def test_valid_mixed_df(synthetic_mixed_df: pd.DataFrame):
    summary = validate_dataset(synthetic_mixed_df, target_column="target")
    assert summary.is_valid
    assert "cat_1" in summary.categorical_columns
    assert "num_1" in summary.numeric_columns
    assert summary.missing_counts["num_2"] > 0


def test_empty_csv_bytes():
    with pytest.raises(DataValidationError, match="is empty"):
        load_csv_from_bytes(b"")


def test_too_few_rows():
    tiny_df = pd.DataFrame({"feat_1": [1, 2, 3], "target": [0, 1, 0]})
    with pytest.raises(DataValidationError, match="minimum of 10 rows"):
        validate_dataset(tiny_df, target_column="target")


def test_missing_target_column(synthetic_binary_df: pd.DataFrame):
    with pytest.raises(DataValidationError, match="not present in the dataset"):
        validate_dataset(synthetic_binary_df, target_column="nonexistent_col")


def test_single_class_target():
    single_class_df = pd.DataFrame(
        {
            "feat_1": range(20),
            "target": [1] * 20,
        }
    )
    with pytest.raises(DataValidationError, match="requires at least 2 distinct target classes"):
        validate_dataset(single_class_df, target_column="target")


def test_duplicate_column_names():
    df_raw = pd.DataFrame([[1, 2, 3]] * 15, columns=["a", "b", "a"])
    with pytest.raises(DataValidationError, match="Duplicate column names detected"):
        validate_dataset(df_raw, target_column="b")


def test_infinite_values_warning():
    df_inf = pd.DataFrame(
        {
            "a": [1.0, np.inf, 2.0] * 5,
            "target": [0, 1, 0] * 5,
        }
    )
    summary = validate_dataset(df_inf, target_column="target")
    assert any("infinite values" in w for w in summary.warnings)
