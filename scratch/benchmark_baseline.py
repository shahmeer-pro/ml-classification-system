"""Baseline performance measurement script.

Measures:
1. Module imports and top-level script execution time
2. Health check / API check latency (with and without API running)
3. Dataset loading time (builtin vs cached)
4. Preprocessing & split time
5. Model training time for each model (GaussianNB, LogisticRegression, LinearSVM, RandomForest)
6. Multi-model comparison time
7. Metric calculation & Plotly chart generation time
8. Prediction latency (batch vs single record)
"""

import sys
import time
from pathlib import Path

# Ensure src/ is in sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import pandas as pd

from ml_app.data.loader import load_builtin_dataset
from ml_app.data.validation import validate_dataset
from ml_app.evaluation.metrics import calculate_classification_metrics
from ml_app.models.registry import MODEL_REGISTRY
from ml_app.preprocessing.pipeline import prepare_and_split_data
from ml_app.services.prediction_service import prediction_service
from ml_app.services.training_service import training_service
from ml_app.ui.api_client import ClassificationApiClient
from ml_app.ui.components import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curve,
)


def benchmark():
    timings = {}

    # 1. API Health Check Latency (when backend is not running - default standalone mode)
    client = ClassificationApiClient()
    t0 = time.perf_counter()
    status = client.check_health()
    t1 = time.perf_counter()
    timings["api_check_health_standalone"] = (t1 - t0) * 1000

    # 2. Dataset Loading Time
    t0 = time.perf_counter()
    for _ in range(5):
        df, meta = load_builtin_dataset()
    t1 = time.perf_counter()
    timings["load_builtin_dataset_avg_ms"] = ((t1 - t0) / 5) * 1000

    # 3. Data Validation
    t0 = time.perf_counter()
    for _ in range(5):
        val_summary = validate_dataset(df, target_column="target")
    t1 = time.perf_counter()
    timings["validate_dataset_avg_ms"] = ((t1 - t0) / 5) * 1000

    # 4. Data Preprocessing & Splitting
    t0 = time.perf_counter()
    for _ in range(5):
        split = prepare_and_split_data(df, target_column="target", test_size=0.20, random_state=42)
    t1 = time.perf_counter()
    timings["preprocessing_split_avg_ms"] = ((t1 - t0) / 5) * 1000

    # 5. Model Training Times
    for model_id in ["gaussian_nb", "logistic_regression", "linear_svm", "random_forest"]:
        t0 = time.perf_counter()
        res = training_service.train_model(
            df=df,
            target_column="target",
            model_id=model_id,
            positive_class=0,
            positive_class_name="malignant",
        )
        t1 = time.perf_counter()
        timings[f"train_{model_id}_ms"] = (t1 - t0) * 1000

    # 6. Multi-Model Comparison Time
    t0 = time.perf_counter()
    comp_res = training_service.compare_models(
        df=df,
        target_column="target",
        model_ids=["gaussian_nb", "logistic_regression", "linear_svm", "random_forest"],
        positive_class=0,
        positive_class_name="malignant",
    )
    t1 = time.perf_counter()
    timings["compare_all_models_ms"] = (t1 - t0) * 1000

    # 7. Chart Generation Times
    # Confusion matrix
    t0 = time.perf_counter()
    fig_cm = plot_confusion_matrix(res.metrics.confusion_matrix, res.metrics.labels)
    t1 = time.perf_counter()
    timings["plot_confusion_matrix_ms"] = (t1 - t0) * 1000

    # ROC curve
    t0 = time.perf_counter()
    if res.metrics.roc_curve:
        fig_roc = plot_roc_curve(res.metrics.roc_curve, model_name=res.display_name)
    t1 = time.perf_counter()
    timings["plot_roc_curve_ms"] = (t1 - t0) * 1000

    # Comparison chart
    t0 = time.perf_counter()
    fig_comp = plot_model_comparison(comp_res.comparison_table)
    t1 = time.perf_counter()
    timings["plot_model_comparison_ms"] = (t1 - t0) * 1000

    # Class distribution chart
    t0 = time.perf_counter()
    val_counts = df["target"].value_counts().to_dict()
    fig_dist = plot_class_distribution({str(k): int(v) for k, v in val_counts.items()})
    t1 = time.perf_counter()
    timings["plot_class_distribution_ms"] = (t1 - t0) * 1000

    # 8. Prediction Latency
    artifact_id = res.artifact_id
    # Single record
    single_record = [df.drop(columns=["target"]).iloc[0].to_dict()]
    t0 = time.perf_counter()
    for _ in range(10):
        pred_single = prediction_service.predict(artifact_id, single_record)
    t1 = time.perf_counter()
    timings["prediction_single_record_avg_ms"] = ((t1 - t0) / 10) * 1000

    # Batch (114 records)
    batch_df = df.drop(columns=["target"]).iloc[:114]
    t0 = time.perf_counter()
    for _ in range(5):
        pred_batch = prediction_service.predict(artifact_id, batch_df)
    t1 = time.perf_counter()
    timings["prediction_batch_114_records_avg_ms"] = ((t1 - t0) / 5) * 1000

    # Output formatted report
    print("=" * 60)
    print("BASELINE PERFORMANCE TIMINGS (Milliseconds)")
    print("=" * 60)
    for k, v in timings.items():
        print(f"{k:40s}: {v:8.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    benchmark()
