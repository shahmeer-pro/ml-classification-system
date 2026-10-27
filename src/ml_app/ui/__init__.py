"""Streamlit UI components and API client."""

from ml_app.ui.api_client import ClassificationApiClient
from ml_app.ui.components import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_pr_curve,
    plot_roc_curve,
    render_disclaimer,
    render_metrics_cards,
)

__all__ = [
    "ClassificationApiClient",
    "render_disclaimer",
    "render_metrics_cards",
    "plot_confusion_matrix",
    "plot_class_distribution",
    "plot_model_comparison",
    "plot_roc_curve",
    "plot_pr_curve",
]
