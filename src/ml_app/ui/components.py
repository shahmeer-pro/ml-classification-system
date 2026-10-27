"""Streamlit visual components, Plotly chart generators, and disclaimers."""

from typing import Any

import streamlit as st

from ml_app.ui.theme import (
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    render_disclaimer_banner,
    render_kpi_card,
    render_section_title,
)
from ml_app.ui.visualizations import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_pr_curve,
    plot_roc_curve,
)

__all__ = [
    "render_disclaimer",
    "render_metrics_cards",
    "plot_class_distribution",
    "plot_confusion_matrix",
    "plot_model_comparison",
    "plot_pr_curve",
    "plot_roc_curve",
]


def render_disclaimer() -> None:
    """Render medical and educational disclaimer banner."""
    render_disclaimer_banner()


def render_metrics_cards(metrics: Any) -> None:
    """Render prominent metric scorecards with high-contrast visual indicators."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_kpi_card(
            label="Overall Accuracy",
            value=f"{metrics.accuracy * 100:.2f}%",
            subtext="Correct predictions / total",
            status_color=COLOR_SUCCESS if metrics.accuracy >= 0.90 else None,
        )
    with col2:
        render_kpi_card(
            label="Macro Precision",
            value=f"{metrics.precision_macro * 100:.2f}%",
            subtext="Unweighted mean precision",
        )
    with col3:
        render_kpi_card(
            label="Macro Recall",
            value=f"{metrics.recall_macro * 100:.2f}%",
            subtext="Unweighted mean recall",
        )
    with col4:
        render_kpi_card(
            label="Macro F1-Score",
            value=f"{metrics.f1_macro * 100:.2f}%",
            subtext="Harmonic mean of P & R",
            status_color=COLOR_PRIMARY,
        )

    # If binary positive-class metrics are present (e.g. Malignant recall)
    if getattr(metrics, "positive_recall", None) is not None:
        pos_lbl = getattr(metrics, "positive_class_name", None) or f"Class {metrics.positive_class}"
        st.write("")
        render_section_title(
            f"Target Interest Class Performance: {pos_lbl.capitalize()}", badge="Priority Focus"
        )
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        with pcol1:
            render_kpi_card(
                label=f"{pos_lbl.capitalize()} Precision",
                value=f"{metrics.positive_precision * 100:.2f}%",
                subtext="True positive / predicted pos",
            )
        with pcol2:
            render_kpi_card(
                label=f"{pos_lbl.capitalize()} Recall (Sensitivity)",
                value=f"{metrics.positive_recall * 100:.2f}%",
                subtext="Minimizes False Negatives",
                status_color=COLOR_SUCCESS if metrics.positive_recall >= 0.90 else None,
            )
        with pcol3:
            render_kpi_card(
                label=f"{pos_lbl.capitalize()} F1-Score",
                value=f"{metrics.positive_f1 * 100:.2f}%",
                subtext="Positive class balance",
            )
        with pcol4:
            if getattr(metrics, "roc_auc", None) is not None:
                render_kpi_card(
                    label="ROC-AUC Score",
                    value=f"{metrics.roc_auc:.4f}",
                    subtext="Separation capability",
                )
            else:
                render_kpi_card(
                    label="Weighted F1",
                    value=f"{metrics.f1_weighted * 100:.2f}%",
                    subtext="Sample-weighted F1",
                )
