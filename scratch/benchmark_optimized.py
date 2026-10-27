"""Post-optimization performance measurement script.

Measures:
1. Cached API Health Check (Cold vs Warm)
2. Cached Dataset Loading (Cold vs Warm)
3. Cached Validation Summary (Cold vs Warm)
4. Model Training & Comparison (Unchanged ML math, isolated execution)
5. Chart Rendering Latency
6. Prediction Latency
"""

import sys
import time
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ml_app.services.prediction_service import prediction_service
from ml_app.services.training_service import training_service
from ml_app.ui.api_client import ClassificationApiClient
from ml_app.ui.cache import (
    check_api_health_cached,
    get_cached_builtin_dataset,
    get_cached_validation_summary,
)
from ml_app.ui.components import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curve,
)


def benchmark_optimized():
    timings = {}

    # 1. API Health Check (Cached)
    client = ClassificationApiClient()
    # First call (cold)
    t0 = time.perf_counter()
    check_api_health_cached.clear()
    status_cold = client.check_health()
    t1 = time.perf_counter()
    timings["api_check_health_cold_ms"] = (t1 - t0) * 1000

    # Second call (warm - what happens on every page navigation)
    t0 = time.perf_counter()
    status_warm = client.check_health()
    t1 = time.perf_counter()
    timings["api_check_health_warm_ms"] = (t1 - t0) * 1000

    # 2. Builtin Dataset Loading (Cached)
    get_cached_builtin_dataset.clear()
    t0 = time.perf_counter()
    df, meta = get_cached_builtin_dataset()
    t1 = time.perf_counter()
    timings["load_builtin_dataset_cold_ms"] = (t1 - t0) * 1000

    # Warm call (repeated on navigation/interaction)
    t0 = time.perf_counter()
    for _ in range(100):
        df, meta = get_cached_builtin_dataset()
    t1 = time.perf_counter()
    timings["load_builtin_dataset_warm_ms"] = ((t1 - t0) / 100) * 1000

    # 3. Validation Summary (Cached)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    get_cached_validation_summary.clear()
    t0 = time.perf_counter()
    df_val, summary = get_cached_validation_summary(csv_bytes, "test.csv", "target")
    t1 = time.perf_counter()
    timings["validate_dataset_cold_ms"] = (t1 - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(100):
        df_val, summary = get_cached_validation_summary(csv_bytes, "test.csv", "target")
    t1 = time.perf_counter()
    timings["validate_dataset_warm_ms"] = ((t1 - t0) / 100) * 1000

    # 4. Model Training Time (Logistic Regression & Linear SVM)
    t0 = time.perf_counter()
    train_res = training_service.train_model(
        df=df,
        target_column="target",
        model_id="logistic_regression",
        positive_class=0,
        positive_class_name="malignant",
    )
    t1 = time.perf_counter()
    timings["train_logistic_regression_ms"] = (t1 - t0) * 1000

    # 5. Charts Generation
    t0 = time.perf_counter()
    fig_cm = plot_confusion_matrix(train_res.metrics.confusion_matrix, train_res.metrics.labels)
    t1 = time.perf_counter()
    timings["plot_confusion_matrix_ms"] = (t1 - t0) * 1000

    if train_res.metrics.roc_curve:
        t0 = time.perf_counter()
        fig_roc = plot_roc_curve(train_res.metrics.roc_curve, model_name=train_res.display_name)
        t1 = time.perf_counter()
        timings["plot_roc_curve_ms"] = (t1 - t0) * 1000

    # 6. Prediction Latency
    single_record = [df.drop(columns=["target"]).iloc[0].to_dict()]
    t0 = time.perf_counter()
    for _ in range(20):
        prediction_service.predict(train_res.artifact_id, single_record)
    t1 = time.perf_counter()
    timings["prediction_single_record_avg_ms"] = ((t1 - t0) / 20) * 1000

    print("=" * 60)
    print("OPTIMIZED PERFORMANCE TIMINGS (Milliseconds)")
    print("=" * 60)
    for k, v in timings.items():
        print(f"{k:40s}: {v:8.4f} ms")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_optimized()
