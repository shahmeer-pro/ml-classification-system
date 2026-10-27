"""Comprehensive unit tests for the ML Studio visualization module."""

import numpy as np
import pandas as pd
import pytest

from ml_app.evaluation.metrics import CurveData
from ml_app.ui.visualizations import (
    plot_actual_vs_predicted_batch,
    plot_batch_prediction_distribution,
    plot_batch_probabilities,
    plot_categorical_distribution,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_dataset_profile,
    plot_decision_boundary_concept,
    plot_feature_coefficients,
    plot_feature_scatter,
    plot_metric_ranking,
    plot_missing_values,
    plot_model_comparison,
    plot_multi_model_roc,
    plot_numerical_distribution,
    plot_pca_projection,
    plot_pr_curve,
    plot_prediction_distribution,
    plot_roc_curve,
    plot_svm_margin_concept,
    plot_threshold_tradeoff_concept,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a mock test DataFrame for visualization verification."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature_a": np.random.randn(50),
            "feature_b": np.random.rand(50) * 10,
            "feature_c": [1.0 if i % 2 == 0 else 2.0 for i in range(50)],
            "category_col": ["Type A" if i % 3 == 0 else "Type B" for i in range(50)],
            "target": [0 if i < 25 else 1 for i in range(50)],
        }
    )


def test_plot_class_distribution(sample_df: pd.DataFrame) -> None:
    """Verify class distribution bar chart generation."""
    counts = sample_df["target"].value_counts().to_dict()
    fig = plot_class_distribution(counts, title="Test Distribution")
    assert fig is not None
    assert len(fig.data) == 1
    assert "Test Distribution" in str(fig.layout.title.text)


def test_plot_dataset_profile(sample_df: pd.DataFrame) -> None:
    """Verify dataset profile summary chart."""
    fig = plot_dataset_profile(sample_df)
    assert fig is not None
    assert len(fig.data) == 1
    assert "Dataset Profile" in str(fig.layout.title.text)


def test_plot_missing_values_zero_missing(sample_df: pd.DataFrame) -> None:
    """Verify missing values returns None when no missing values exist."""
    assert plot_missing_values(sample_df) is None


def test_plot_missing_values_with_missing() -> None:
    """Verify missing values returns a valid chart when missing values exist."""
    df_missing = pd.DataFrame(
        {
            "col_a": [1.0, None, 3.0],
            "col_b": [None, None, 3.0],
            "col_c": [1.0, 2.0, 3.0],
        }
    )
    fig = plot_missing_values(df_missing)
    assert fig is not None
    assert len(fig.data) == 1


def test_plot_numerical_distribution(sample_df: pd.DataFrame) -> None:
    """Verify numerical feature distribution histogram and boxplot."""
    fig = plot_numerical_distribution(sample_df, "feature_a")
    assert fig is not None
    assert len(fig.data) == 2  # Boxplot + Histogram


def test_plot_feature_scatter(sample_df: pd.DataFrame) -> None:
    """Verify 2D feature scatter plot."""
    fig = plot_feature_scatter(sample_df, "feature_a", "feature_b", target_col="target")
    assert fig is not None
    assert len(fig.data) >= 1


def test_plot_correlation_heatmap(sample_df: pd.DataFrame) -> None:
    """Verify Pearson correlation heatmap."""
    fig = plot_correlation_heatmap(sample_df)
    assert fig is not None
    assert len(fig.data) == 1


def test_plot_categorical_distribution(sample_df: pd.DataFrame) -> None:
    """Verify categorical frequency distribution."""
    fig = plot_categorical_distribution(sample_df, "category_col")
    assert fig is not None
    assert len(fig.data) == 1


def test_plot_pca_projection(sample_df: pd.DataFrame) -> None:
    """Verify 2D PCA projection scatter plot."""
    fig = plot_pca_projection(
        sample_df, ["feature_a", "feature_b", "feature_c"], target_col="target"
    )
    assert fig is not None
    assert "Principal Component Analysis" in str(fig.layout.title.text)


def test_plot_confusion_matrix() -> None:
    """Verify confusion matrix heatmap rendering."""
    cm = [[20, 2], [1, 27]]
    fig = plot_confusion_matrix(cm, ["0", "1"], model_name="Logistic Regression")
    assert fig is not None
    assert len(fig.data) == 1
    assert "Logistic Regression" in str(fig.layout.title.text)


def test_plot_prediction_distribution() -> None:
    """Verify prediction vs ground truth distribution."""
    y_pred = [0, 1, 1, 0, 1]
    y_true = [0, 0, 1, 0, 1]
    fig = plot_prediction_distribution(y_pred, y_true=y_true, labels=["0", "1"])
    assert fig is not None
    assert len(fig.data) == 2


def test_plot_feature_coefficients() -> None:
    """Verify feature coefficient horizontal bar chart."""
    feats = ["feat_1", "feat_2", "feat_3"]
    coefs = np.array([0.45, -1.2, 0.89])
    fig = plot_feature_coefficients(feats, coefs, title="Test Coefficients", is_importance=False)
    assert fig is not None
    assert len(fig.data) == 1


def test_plot_roc_and_pr_curves() -> None:
    """Verify ROC and PR curve generators."""
    curve = CurveData(
        x_values=[0.0, 0.2, 1.0], y_values=[0.0, 0.8, 1.0], thresholds=[0.9, 0.5, 0.1], score=0.85
    )
    fig_roc = plot_roc_curve(curve, model_name="LogReg")
    assert fig_roc is not None
    assert "ROC" in str(fig_roc.layout.title.text)

    fig_pr = plot_pr_curve(curve, model_name="LogReg")
    assert fig_pr is not None
    assert "Precision-Recall" in str(fig_pr.layout.title.text)


def test_plot_model_comparison_and_ranking() -> None:
    """Verify multi-model comparison and ranking charts."""
    table = [
        {"Model": "Logistic Regression", "Accuracy": 0.96, "Macro F1": 0.95, "Macro Recall": 0.94},
        {"Model": "Linear SVM", "Accuracy": 0.97, "Macro F1": 0.96, "Macro Recall": 0.95},
    ]
    fig_comp = plot_model_comparison(table)
    assert fig_comp is not None

    fig_rank = plot_metric_ranking(table, metric_name="Macro F1")
    assert fig_rank is not None


def test_plot_multi_model_roc() -> None:
    """Verify multi-model ROC overlay."""

    class MockResult:
        def __init__(self, name: str, score: float) -> None:
            self.display_name = name

            class Metrics:
                roc_curve = CurveData(
                    [0.0, 0.1, 1.0], [0.0, 0.9, 1.0], [0.8, 0.5, 0.2], score=score
                )

            self.metrics = Metrics()

    results = [MockResult("Model A", 0.92), MockResult("Model B", 0.88)]
    fig = plot_multi_model_roc(results)
    assert fig is not None
    assert len(fig.data) == 3  # 2 models + 1 chance diagonal


def test_plot_batch_predictions_and_probabilities() -> None:
    """Verify batch inference visualization generators."""
    preds = [0, 1, 0, 0, 1, 1]
    fig_dist = plot_batch_prediction_distribution(preds, class_labels=["0", "1"])
    assert fig_dist is not None

    probs = [[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]]
    fig_probs = plot_batch_probabilities(probs, class_labels=["0", "1"])
    assert fig_probs is not None

    fig_avsp = plot_actual_vs_predicted_batch([0, 1, 0], [0, 0, 0], class_labels=["0", "1"])
    assert fig_avsp is not None


def test_educational_concept_visualizations() -> None:
    """Verify educational concept charts render without errors."""
    fig_db = plot_decision_boundary_concept()
    assert fig_db is not None
    assert len(fig_db.data) >= 3

    fig_svm = plot_svm_margin_concept()
    assert fig_svm is not None
    assert len(fig_svm.data) >= 4

    fig_thresh = plot_threshold_tradeoff_concept()
    assert fig_thresh is not None
    assert len(fig_thresh.data) == 3
