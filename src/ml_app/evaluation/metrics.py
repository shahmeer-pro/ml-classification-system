"""Evaluation metrics calculation for binary and multiclass classification."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml_app.logging_config import logger


@dataclass
class CurveData:
    """Coordinates and thresholds for diagnostic curves (ROC or PR)."""

    x_values: list[float]
    y_values: list[float]
    thresholds: list[float]
    score: float | None = None


@dataclass
class ClassificationMetricsResult:
    """Comprehensive evaluation metrics collection."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    confusion_matrix: list[list[int]]
    labels: list[str]
    per_class_metrics: dict[str, dict[str, float]]
    support: dict[str, int]
    classification_report_str: str

    # Binary specific
    is_binary: bool = False
    positive_class: str | None = None
    positive_class_name: str | None = None
    positive_precision: float | None = None
    positive_recall: float | None = None
    positive_f1: float | None = None

    # Diagnostic curves (if scores/probabilities available)
    roc_auc: float | None = None
    roc_curve: CurveData | None = None
    pr_curve: CurveData | None = None
    warnings: list[str] = field(default_factory=list)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray | None = None,
    class_labels: list[Any] | None = None,
    positive_class: Any | None = None,
    positive_class_name: str | None = None,
) -> ClassificationMetricsResult:
    """Calculate classification metrics with binary and multiclass support.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted target labels.
        y_scores: Optional prediction probabilities or decision scores.
        class_labels: Complete list of expected class labels in order.
        positive_class: Label designating the positive class for binary evaluation.
        positive_class_name: Human readable label for positive class (e.g. 'malignant').

    Returns:
        ClassificationMetricsResult: Structured metrics and diagnostic curves.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # Determine unique labels
    if class_labels is None:
        detected_labels = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
        labels = [str(x) for x in sorted(detected_labels, key=lambda v: str(v))]
    else:
        labels = [str(x) for x in class_labels]

    is_binary = len(labels) == 2
    warnings: list[str] = []

    # Overall metrics
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec_macro = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    f1_mac = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    prec_weighted = float(
        precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
    )
    rec_weighted = float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    f1_wgt = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

    # Confusion matrix with explicit labels
    cm = confusion_matrix(y_true_arr.astype(str), y_pred_arr.astype(str), labels=labels).tolist()

    # Classification report
    clf_report_dict = classification_report(
        y_true_arr.astype(str),
        y_pred_arr.astype(str),
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    clf_report_text = classification_report(
        y_true_arr.astype(str),
        y_pred_arr.astype(str),
        labels=labels,
        zero_division=0,
    )

    per_class_metrics: dict[str, dict[str, float]] = {}
    support_counts: dict[str, int] = {}
    for lbl in labels:
        if lbl in clf_report_dict:
            per_class_metrics[lbl] = {
                "precision": round(float(clf_report_dict[lbl]["precision"]), 4),
                "recall": round(float(clf_report_dict[lbl]["recall"]), 4),
                "f1": round(float(clf_report_dict[lbl]["f1-score"]), 4),
            }
            support_counts[lbl] = int(clf_report_dict[lbl]["support"])

    # Binary specific handling
    pos_precision = None
    pos_recall = None
    pos_f1 = None
    roc_auc_val = None
    roc_curve_obj = None
    pr_curve_obj = None

    if is_binary:
        # Resolve target positive class
        if positive_class is None:
            resolved_pos = labels[1]
        else:
            resolved_pos = str(positive_class)

        if resolved_pos not in labels:
            resolved_pos = labels[0]

        # Calculate positive class specific metrics
        pos_precision = float(
            precision_score(
                y_true_arr.astype(str),
                y_pred_arr.astype(str),
                pos_label=resolved_pos,
                zero_division=0,
            )
        )
        pos_recall = float(
            recall_score(
                y_true_arr.astype(str),
                y_pred_arr.astype(str),
                pos_label=resolved_pos,
                zero_division=0,
            )
        )
        pos_f1 = float(
            f1_score(
                y_true_arr.astype(str),
                y_pred_arr.astype(str),
                pos_label=resolved_pos,
                zero_division=0,
            )
        )

        # ROC and PR Curves if scores are provided
        if y_scores is not None and len(y_scores) == len(y_true_arr):
            try:
                # Convert ground truth to binary 1/0 where 1 is resolved_pos
                y_true_bin = (y_true_arr.astype(str) == resolved_pos).astype(int)

                # If y_scores has 2 columns (predict_proba), take the probability of resolved_pos
                if y_scores.ndim == 2 and y_scores.shape[1] == 2:
                    pos_idx = labels.index(resolved_pos) if resolved_pos in labels else 1
                    scores_1d = y_scores[:, pos_idx]
                elif y_scores.ndim == 1:
                    # In binary LinearSVC, decision_function > 0 corresponds to labels[1].
                    # If positive class is labels[0], invert scores so higher means more likely class 0.
                    if len(labels) == 2 and resolved_pos == labels[0]:
                        scores_1d = -y_scores
                    else:
                        scores_1d = y_scores
                else:
                    scores_1d = y_scores[:, 0]

                if len(np.unique(y_true_bin)) == 2:
                    fpr, tpr, roc_thresh = roc_curve(y_true_bin, scores_1d)
                    roc_auc_val = float(roc_auc_score(y_true_bin, scores_1d))

                    # Sanitize thresholds: scikit-learn sets roc_thresh[0] = inf
                    clean_roc_thresh: list[float] = []
                    for t in roc_thresh:
                        if np.isinf(t):
                            clean_roc_thresh.append(1.0 if t > 0 else 0.0)
                        else:
                            clean_roc_thresh.append(round(float(t), 4))

                    roc_curve_obj = CurveData(
                        x_values=[round(float(x), 4) for x in fpr],
                        y_values=[round(float(y), 4) for y in tpr],
                        thresholds=clean_roc_thresh,
                        score=round(roc_auc_val, 4),
                    )

                    pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_true_bin, scores_1d)
                    clean_pr_thresh: list[float] = []
                    for t in pr_thresh:
                        if np.isinf(t):
                            clean_pr_thresh.append(1.0 if t > 0 else 0.0)
                        else:
                            clean_pr_thresh.append(round(float(t), 4))

                    pr_curve_obj = CurveData(
                        x_values=[round(float(r), 4) for r in pr_rec],
                        y_values=[round(float(p), 4) for p in pr_prec],
                        thresholds=clean_pr_thresh,
                    )
            except Exception as exc:
                warnings.append(f"ROC/PR calculation skipped: {exc}")
                logger.warning("ROC/PR curve calculation warning: %s", exc)

    logger.info(
        "Calculated metrics: Accuracy=%.4f, Macro-F1=%.4f, IsBinary=%s",
        acc,
        f1_mac,
        is_binary,
    )

    return ClassificationMetricsResult(
        accuracy=round(acc, 4),
        precision_macro=round(prec_macro, 4),
        recall_macro=round(rec_macro, 4),
        f1_macro=round(f1_mac, 4),
        precision_weighted=round(prec_weighted, 4),
        recall_weighted=round(rec_weighted, 4),
        f1_weighted=round(f1_wgt, 4),
        confusion_matrix=cm,
        labels=labels,
        per_class_metrics=per_class_metrics,
        support=support_counts,
        classification_report_str=clf_report_text,
        is_binary=is_binary,
        positive_class=str(positive_class) if positive_class is not None else None,
        positive_class_name=positive_class_name,
        positive_precision=round(pos_precision, 4) if pos_precision is not None else None,
        positive_recall=round(pos_recall, 4) if pos_recall is not None else None,
        positive_f1=round(pos_f1, 4) if pos_f1 is not None else None,
        roc_auc=round(roc_auc_val, 4) if roc_auc_val is not None else None,
        roc_curve=roc_curve_obj,
        pr_curve=pr_curve_obj,
        warnings=warnings,
    )
