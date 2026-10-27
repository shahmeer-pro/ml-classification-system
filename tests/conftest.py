"""Shared pytest fixtures and configurations."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification

from ml_app.api.main import app
from ml_app.data.loader import load_builtin_dataset


@pytest.fixture
def synthetic_binary_df() -> pd.DataFrame:
    """Generate a reproducible, deterministic synthetic binary classification DataFrame."""
    X, y = make_classification(
        n_samples=100,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    feature_names = [f"feat_{i}" for i in range(6)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df


@pytest.fixture
def synthetic_mixed_df() -> pd.DataFrame:
    """Generate a DataFrame with numeric, categorical, and missing values."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame(
        {
            "num_1": np.random.randn(n),
            "num_2": [np.nan if i % 10 == 0 else np.random.randn() for i in range(n)],
            "cat_1": ["group_A" if i % 2 == 0 else "group_B" for i in range(n)],
            "target": [0 if i < 30 else 1 for i in range(n)],
        }
    )
    return df


@pytest.fixture
def builtin_cancer_data() -> tuple[pd.DataFrame, dict]:
    """Provide the real built-in Breast Cancer Wisconsin dataset."""
    return load_builtin_dataset()


@pytest.fixture
def api_client() -> TestClient:
    """FastAPI TestClient fixture."""
    return TestClient(app)
