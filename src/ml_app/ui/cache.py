"""Streamlit caching utilities for deterministic data operations."""

from typing import Any

import httpx
import pandas as pd
import streamlit as st

from ml_app.data.loader import load_builtin_dataset, load_csv_from_bytes
from ml_app.data.validation import validate_dataset


@st.cache_data(show_spinner=False)
def get_cached_builtin_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and cache the built-in Breast Cancer Wisconsin dataset.

    Cached across all user sessions as it is static, immutable reference data.
    """
    return load_builtin_dataset()


@st.cache_data(show_spinner=False, max_entries=10)
def get_cached_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse CSV bytes into a DataFrame, keyed by exact content bytes."""
    return load_csv_from_bytes(file_bytes, filename=filename)


@st.cache_data(show_spinner=False, max_entries=10)
def get_cached_validation_summary(
    file_bytes: bytes,
    filename: str,
    target_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate CSV bytes and cache the validation summary."""
    df = get_cached_csv(file_bytes, filename)
    summary = validate_dataset(df=df, target_column=target_column)

    summary_dict = {
        "is_valid": summary.is_valid,
        "num_rows": summary.num_rows,
        "num_cols": summary.num_cols,
        "column_names": summary.column_names,
        "dtypes": summary.dtypes,
        "missing_counts": summary.missing_counts,
        "missing_percentages": summary.missing_percentages,
        "numeric_columns": summary.numeric_columns,
        "categorical_columns": summary.categorical_columns,
        "constant_columns": summary.constant_columns,
        "target_column": summary.target_column,
        "target_classes": summary.target_classes,
        "class_distribution": summary.class_distribution,
        "warnings": summary.warnings,
        "errors": summary.errors,
    }
    return df, summary_dict


@st.cache_data(ttl=60, show_spinner=False)
def check_api_health_cached(base_url: str) -> dict[str, Any]:
    """Check API health with short timeout and 60-second cache TTL.

    Prevents blocking navigation on unreachable ports.
    """
    try:
        with httpx.Client(timeout=0.3) as client:
            resp = client.get(f"{base_url}/health")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass

    return {
        "status": "standalone_mode",
        "version": "1.0.0",
        "service": "Internal Python Service (Direct)",
    }
