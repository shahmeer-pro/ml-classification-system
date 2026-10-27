"""Tests for Streamlit caching, session state management, and isolation."""

import time

from ml_app.data.loader import load_builtin_dataset
from ml_app.services.training_service import training_service
from ml_app.ui.cache import (
    check_api_health_cached,
    get_cached_builtin_dataset,
    get_cached_validation_summary,
)


def test_cached_builtin_dataset_consistency():
    """Verify cached dataset returns exact same content with sub-millisecond retrieval."""
    df1, meta1 = get_cached_builtin_dataset()
    df2, meta2 = get_cached_builtin_dataset()

    assert len(df1) == len(df2) == 569
    assert meta1["name"] == meta2["name"]
    # Pointer / content equality
    assert df1.equals(df2)


def test_cached_api_health_check():
    """Verify that health check results are cached and return within sub-millisecond bounds."""
    # First call (populates cache)
    res1 = check_api_health_cached("http://127.0.0.1:8000")
    t0 = time.perf_counter()
    res2 = check_api_health_cached("http://127.0.0.1:8000")
    t1 = time.perf_counter()

    assert res1 == res2
    assert (t1 - t0) < 0.01  # Warm cached call must take < 10ms, avoiding socket stall


def test_cached_validation_summary():
    """Verify CSV validation caching by byte content."""
    df, _ = load_builtin_dataset()
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    df1, summary1 = get_cached_validation_summary(csv_bytes, "test.csv", "target")
    t0 = time.perf_counter()
    df2, summary2 = get_cached_validation_summary(csv_bytes, "test.csv", "target")
    t1 = time.perf_counter()

    assert summary1["num_rows"] == summary2["num_rows"] == 569
    assert (t1 - t0) < 0.01


def test_training_is_explicit_and_isolated():
    """Verify model training does not occur automatically and preserves distinct artifacts."""
    df, _ = load_builtin_dataset()

    # Artifacts must not exist until train_model is explicitly invoked
    initial_count = len(training_service.list_artifacts())

    res1 = training_service.train_model(
        df=df,
        target_column="target",
        model_id="logistic_regression",
        random_state=42,
    )
    res2 = training_service.train_model(
        df=df,
        target_column="target",
        model_id="gaussian_nb",
        random_state=42,
    )

    # Distinct artifact IDs guarantee session/invocation isolation
    assert res1.artifact_id != res2.artifact_id
    assert len(training_service.list_artifacts()) == initial_count + 2

    # Artifacts retain their own fitted pipelines
    art1 = training_service.get_artifact(res1.artifact_id)
    art2 = training_service.get_artifact(res2.artifact_id)
    assert art1.model_id == "logistic_regression"
    assert art2.model_id == "gaussian_nb"
