"""Dataset loading utilities for built-in and CSV data."""

import io
from typing import Any

import pandas as pd
from sklearn.datasets import load_breast_cancer

from ml_app.config import settings
from ml_app.exceptions import DataValidationError
from ml_app.logging_config import logger


def load_builtin_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the scikit-learn Breast Cancer Wisconsin dataset.

    Returns:
        tuple[pd.DataFrame, dict[str, Any]]: A pandas DataFrame with all features and
        target column, plus a dictionary containing dataset metadata and class mapping.
    """
    raw_data = load_breast_cancer(as_frame=True)
    df = raw_data.frame.copy()
    df.columns = [str(col) for col in df.columns]

    metadata: dict[str, Any] = {
        "name": "Breast Cancer Wisconsin (Diagnostic)",
        "source": "scikit-learn (UCI Machine Learning Repository)",
        "samples_count": len(df),
        "features_count": len(raw_data.feature_names),
        "feature_names": [str(name) for name in raw_data.feature_names],
        "target_name": "target",
        # Original scikit-learn mapping: 0 = malignant, 1 = benign
        "class_mapping": {0: "malignant", 1: "benign"},
        "positive_class": 0,  # 0 (malignant) is the clinical class of interest
        "positive_class_name": "malignant",
        "description": (
            "Features are computed from a digitized image of a fine needle aspirate (FNA) "
            "of a breast mass. They describe characteristics of the cell nuclei present in the image. "
            "Classes: 0 = malignant (212 samples), 1 = benign (357 samples)."
        ),
        "educational_disclaimer": (
            "DISCLAIMER: This application and dataset are intended strictly for educational, "
            "demonstrative, and portfolio purposes. They are NOT certified or validated for clinical "
            "decision-making or medical diagnosis."
        ),
    }

    logger.info("Loaded built-in Breast Cancer dataset with %d records.", len(df))
    return df, metadata


def load_csv_from_bytes(file_bytes: bytes, filename: str = "upload.csv") -> pd.DataFrame:
    """Parse CSV content from bytes with guardrails on file size and basic parsing.

    Args:
        file_bytes: Raw bytes of uploaded file.
        filename: Original file name.

    Returns:
        pd.DataFrame: Parsed pandas DataFrame.

    Raises:
        DataValidationError: If the file is invalid, empty, or exceeds size limits.
    """
    if len(file_bytes) == 0:
        raise DataValidationError(f"The uploaded file '{filename}' is empty.")

    if len(file_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise DataValidationError(
            f"File size ({len(file_bytes) / (1024 * 1024):.2f} MB) exceeds maximum limit "
            f"of {settings.MAX_UPLOAD_SIZE_MB} MB."
        )

    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        logger.warning("Failed to parse CSV upload: %s", type(exc).__name__)
        raise DataValidationError(f"Could not parse '{filename}' as a valid CSV: {exc}") from exc

    if df.empty:
        raise DataValidationError("Uploaded CSV contains no rows.")

    df.columns = [str(col) for col in df.columns]
    logger.info("Successfully loaded CSV '%s' with shape %s", filename, df.shape)
    return df
