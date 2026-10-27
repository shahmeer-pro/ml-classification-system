"""Streamlit visual components, Plotly chart generators, and disclaimers."""

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_disclaimer() -> None:
    """Render medical and educational disclaimer banner."""
    st.info(
        "⚕️ **Educational & Portfolio Disclaimer**: This application and the built-in Breast Cancer "
        "Wisconsin dataset are provided solely for educational, demonstrative, and machine learning "
        "portfolio purposes. It has not been clinically evaluated or certified for real-world diagnostic use."
    )


def render_metrics_cards(metrics: Any) -> None:
    """Render prominent metric scorecards with high-contrast visual indicators."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Overall Accuracy",
            value=f"{metrics.accuracy * 100:.2f}%",
        )
    with col2:
        st.metric(
            label="Macro Precision",
            value=f"{metrics.precision_macro * 100:.2f}%",
        )
    with col3:
        st.metric(
            label="Macro Recall",
            value=f"{metrics.recall_macro * 100:.2f}%",
        )
    with col4:
        st.metric(
            label="Macro F1-Score",
            value=f"{metrics.f1_macro * 100:.2f}%",
        )

    # If binary positive-class metrics are present (e.g. Malignant recall)
    if getattr(metrics, "positive_recall", None) is not None:
        pos_lbl = getattr(metrics, "positive_class_name", None) or f"Class {metrics.positive_class}"
        st.write("---")
        st.subheader(f"Target Interest Class Performance: **{pos_lbl.capitalize()}**")
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        with pcol1:
            st.metric(
                label=f"{pos_lbl.capitalize()} Precision",
                value=f"{metrics.positive_precision * 100:.2f}%",
            )
        with pcol2:
            st.metric(
                label=f"{pos_lbl.capitalize()} Recall (Sensitivity)",
                value=f"{metrics.positive_recall * 100:.2f}%",
            )
        with pcol3:
            st.metric(
                label=f"{pos_lbl.capitalize()} F1-Score",
                value=f"{metrics.positive_f1 * 100:.2f}%",
            )
        with pcol4:
            if getattr(metrics, "roc_auc", None) is not None:
                st.metric(label="ROC-AUC Score", value=f"{metrics.roc_auc:.4f}")
            else:
                st.metric(label="Weighted F1", value=f"{metrics.f1_weighted * 100:.2f}%")


def plot_confusion_matrix(cm: list[list[int]], labels: list[str]) -> go.Figure:
    """Generate an annotated Plotly heatmap for the confusion matrix."""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted Class", y="Actual Class", color="Sample Count"),
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        title="Confusion Matrix Heatmap",
        xaxis_title="Predicted Label",
        yaxis_title="Ground Truth Label",
        height=420,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_class_distribution(distribution: dict[str, int]) -> go.Figure:
    """Generate a clean bar chart for target class balance."""
    df_dist = pd.DataFrame(
        {"Class": list(distribution.keys()), "Count": list(distribution.values())}
    )
    fig = px.bar(
        df_dist,
        x="Class",
        y="Count",
        color="Class",
        text="Count",
        title="Target Class Distribution",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis_title="Number of Samples",
        showlegend=False,
        height=360,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig


def plot_model_comparison(comparison_table: list[dict[str, Any]]) -> go.Figure:
    """Generate side-by-side comparative bar chart across models and metrics."""
    df_comp = pd.DataFrame(comparison_table)
    metric_cols = [
        col
        for col in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
        if col in df_comp.columns
    ]

    df_melted = df_comp.melt(
        id_vars=["Model"],
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Score",
    )

    fig = px.bar(
        df_melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Model Performance Comparison on Identical Test Partition",
        range_y=[0.0, 1.05],
        text=df_melted["Score"].apply(lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else ""),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=450,
        yaxis_title="Score (0.0 to 1.0)",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_roc_curve(roc_curve_data: Any, model_name: str = "Classifier") -> go.Figure:
    """Generate interactive ROC Curve with chance diagonal line."""
    fig = go.Figure()

    # ROC curve
    fig.add_trace(
        go.Scatter(
            x=roc_curve_data.x_values,
            y=roc_curve_data.y_values,
            mode="lines",
            name=f"{model_name} (AUC = {roc_curve_data.score:.4f})"
            if roc_curve_data.score
            else model_name,
            line=dict(color="#1f77b4", width=3),
        )
    )

    # Diagonal chance line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guessing (AUC = 0.50)",
            line=dict(color="grey", dash="dash"),
        )
    )

    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Recall / Sensitivity)",
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.05]),
        height=420,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_pr_curve(pr_curve_data: Any, model_name: str = "Classifier") -> go.Figure:
    """Generate Precision-Recall Curve."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pr_curve_data.x_values,
            y=pr_curve_data.y_values,
            mode="lines",
            name=f"{model_name} PR Curve",
            line=dict(color="#2ca02c", width=3),
        )
    )

    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.05]),
        height=420,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig
