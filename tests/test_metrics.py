"""Tests for evaluation metrics, confusion matrix, and positive-class conventions."""

import numpy as np

from ml_app.evaluation.metrics import calculate_classification_metrics


def test_metrics_calculation_known_example():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    # 2 correct out of 4 -> 50% accuracy

    res = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_labels=[0, 1],
        positive_class=1,
    )

    assert res.accuracy == 0.5
    assert res.confusion_matrix == [[1, 1], [1, 1]]
    assert res.is_binary is True
    assert res.positive_precision == 0.5
    assert res.positive_recall == 0.5
    assert res.positive_f1 == 0.5


def test_malignant_positive_class_convention():
    """In Breast Cancer, class 0 is malignant and class 1 is benign."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])
    # For class 0: 2 TP, 1 FN -> Recall = 2/3 = 0.6667
    # For class 1: 2 TP, 1 FP (predicting 1 when true is 0)

    res = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_labels=[0, 1],
        positive_class=0,
        positive_class_name="malignant",
    )

    assert res.positive_class == "0"
    assert res.positive_class_name == "malignant"
    assert res.positive_recall == round(2 / 3, 4)
    assert res.positive_precision == 1.0  # Predicted 0 twice, both were 0


def test_multiclass_metrics():
    y_true = np.array(["cat", "dog", "bird", "cat", "dog"])
    y_pred = np.array(["cat", "dog", "dog", "cat", "bird"])

    res = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_labels=["bird", "cat", "dog"],
    )

    assert res.is_binary is False
    assert len(res.labels) == 3
    assert "bird" in res.per_class_metrics
    assert "cat" in res.per_class_metrics
    assert "dog" in res.per_class_metrics
    assert len(res.confusion_matrix) == 3
