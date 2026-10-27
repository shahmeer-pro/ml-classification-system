"""Interactive, data-driven visualization components using Plotly for ML Studio.

All visualization functions are pure figure builders: they take data, compute required
statistics, and return a configured plotly.graph_objects.Figure using the studio theme.
"""

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ml_app.ui.theme import (
    CHART_PALETTE,
    COLOR_BORDER,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_TEXT,
    COLOR_WARNING,
    apply_studio_chart_theme,
)

# ---------------------------------------------------------------------------
# Consistent Color Definitions for Visualizations
# ---------------------------------------------------------------------------
COLOR_INDIGO = "#4F46E5"
COLOR_BLUE = "#2563EB"
COLOR_TEAL = "#0F766E"
COLOR_AMBER = "#D97706"
COLOR_SLATE = "#64748B"
COLOR_LIGHT_GRID = "#F1F5F9"

MODEL_COLOR_MAP: dict[str, str] = {
    "Logistic Regression": "#4F46E5",
    "Gaussian Naive Bayes": "#2563EB",
    "Linear Support Vector Machine (LinearSVC)": "#0F766E",
    "Random Forest Classifier": "#D97706",
    "Linear SVM": "#0F766E",
    "Random Forest": "#D97706",
}


def get_model_color(name: str, fallback_idx: int = 0) -> str:
    """Return a consistent color for a model name."""
    for key, color in MODEL_COLOR_MAP.items():
        if key.lower() in name.lower() or name.lower() in key.lower():
            return color
    return CHART_PALETTE[fallback_idx % len(CHART_PALETTE)]


# ===========================================================================
# 1. Dataset Explorer & Overview Visualizations
# ===========================================================================


def plot_class_distribution(
    distribution: dict[str, int] | pd.Series,
    title: str = "Target Class Distribution",
    show_percentages: bool = True,
) -> go.Figure:
    """Generate an interactive bar chart of class counts and percentages."""
    if isinstance(distribution, pd.Series):
        dist_dict = distribution.to_dict()
    else:
        dist_dict = distribution

    labels = [str(k) for k in dist_dict.keys()]
    counts = [int(v) for v in dist_dict.values()]
    total = sum(counts) if counts else 1
    pcts = [(c / total) * 100 for c in counts]

    customdata = np.stack((counts, pcts), axis=-1)

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=counts,
                text=[
                    f"{c:,} ({p:.1f}%)" if show_percentages else f"{c:,}"
                    for c, p in zip(counts, pcts, strict=False)
                ],
                textposition="outside",
                cliponaxis=False,
                marker=dict(
                    color=[CHART_PALETTE[i % len(CHART_PALETTE)] for i in range(len(labels))],
                    line=dict(color=COLOR_BORDER, width=1),
                ),
                customdata=customdata,
                hovertemplate="<b>Class:</b> %{x}<br><b>Count:</b> %{customdata[0]:,}<br><b>Proportion:</b> %{customdata[1]:.2f}%<extra></extra>",
            )
        ]
    )

    apply_studio_chart_theme(fig, title=title, height=360, show_legend=False)
    fig.update_layout(
        xaxis_title="Target Class Label",
        yaxis_title="Number of Records",
        yaxis=dict(range=[0, max(counts) * 1.18 if counts else 10]),
    )
    return fig


def plot_dataset_profile(df: pd.DataFrame) -> go.Figure:
    """Generate a compact summary chart of dataset dimensions and column types."""
    n_rows = len(df)
    n_numeric = len(df.select_dtypes(include=[np.number]).columns)
    n_categorical = len(df.select_dtypes(exclude=[np.number]).columns)
    n_missing = int(df.isna().sum().sum())

    categories = [
        "Total Rows",
        "Numeric Features",
        "Categorical Features",
        "Missing Values",
    ]
    values = [n_rows, n_numeric, n_categorical, n_missing]
    colors = [
        COLOR_INDIGO,
        COLOR_BLUE,
        COLOR_TEAL,
        COLOR_WARNING if n_missing > 0 else COLOR_SUCCESS,
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                y=categories,
                x=values,
                orientation="h",
                text=[f"{v:,}" for v in values],
                textposition="outside",
                cliponaxis=False,
                marker=dict(color=colors, line=dict(color=COLOR_BORDER, width=1)),
                hovertemplate="<b>%{y}:</b> %{x:,}<extra></extra>",
            )
        ]
    )
    apply_studio_chart_theme(fig, title="Dataset Profile Metrics", height=280, show_legend=False)
    fig.update_layout(
        xaxis_title="Count",
        yaxis_title="",
        xaxis=dict(range=[0, max(values) * 1.25 if values else 10]),
        margin=dict(l=130, r=40, t=40, b=35),
    )
    return fig


def plot_missing_values(df: pd.DataFrame, show_all: bool = False) -> go.Figure | None:
    """Generate missing-value analysis chart. Returns None if zero missing values exist."""
    null_counts = df.isna().sum()
    total_rows = len(df)
    if null_counts.sum() == 0:
        return None

    if not show_all:
        null_counts = null_counts[null_counts > 0]

    null_pct = (null_counts / total_rows) * 100
    sorted_idx = null_counts.sort_values(ascending=True).index

    cols = list(sorted_idx)
    counts = [int(null_counts[c]) for c in cols]
    pcts = [float(null_pct[c]) for c in cols]

    customdata = np.stack((counts, pcts), axis=-1)

    fig = go.Figure(
        data=[
            go.Bar(
                y=cols,
                x=counts,
                orientation="h",
                text=[f"{c:,} ({p:.1f}%)" for c, p in zip(counts, pcts, strict=False)],
                textposition="outside",
                cliponaxis=False,
                marker=dict(color=COLOR_AMBER, line=dict(color=COLOR_BORDER, width=1)),
                customdata=customdata,
                hovertemplate="<b>Feature:</b> %{y}<br><b>Missing:</b> %{customdata[0]:,}<br><b>Percentage:</b> %{customdata[1]:.2f}%<extra></extra>",
            )
        ]
    )

    height = max(300, min(800, len(cols) * 26 + 100))
    apply_studio_chart_theme(
        fig, title="Missing Value Distribution by Feature", height=height, show_legend=False
    )
    fig.update_layout(
        xaxis_title="Missing Record Count",
        yaxis_title="Feature Column",
        xaxis=dict(range=[0, max(counts) * 1.25 if counts else 10]),
        margin=dict(l=140, r=40, t=40, b=40),
    )
    return fig


def plot_numerical_distribution(df: pd.DataFrame, column: str) -> go.Figure:
    """Generate distribution histogram and boxplot for a numerical feature."""
    series = df[column].dropna()
    mean_val = float(series.mean())
    median_val = float(series.median())
    std_val = float(series.std()) if len(series) > 1 else 0.0

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.25, 0.75],
        vertical_spacing=0.08,
    )

    # Box plot on top to display median, quartiles, and outliers
    fig.add_trace(
        go.Box(
            x=series,
            name="Spread",
            marker=dict(color=COLOR_PRIMARY, size=4),
            line=dict(color=COLOR_PRIMARY),
            boxmean=True,
            hovertemplate="<b>Box Summary:</b> %{x:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Histogram on bottom
    fig.add_trace(
        go.Histogram(
            x=series,
            name="Count",
            nbinsx=35,
            marker=dict(
                color="#818CF8",
                line=dict(color=COLOR_PRIMARY, width=1),
            ),
            hovertemplate="<b>Range:</b> %{x}<br><b>Frequency:</b> %{y:,}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Add vertical mean indicator line
    fig.add_vline(
        x=mean_val,
        line_width=2,
        line_dash="dash",
        line_color=COLOR_AMBER,
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top right",
    )

    apply_studio_chart_theme(
        fig,
        title=f"Distribution & Outlier Inspection: {column}",
        height=420,
        show_legend=False,
    )
    fig.update_layout(
        xaxis2=dict(
            title=f"{column} (Mean: {mean_val:.2f}, Median: {median_val:.2f}, Std: {std_val:.2f})"
        ),
        yaxis2=dict(title="Frequency Count"),
        yaxis=dict(showticklabels=False),
        margin=dict(l=55, r=40, t=50, b=45),
    )
    return fig


def plot_feature_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    target_col: str | None = None,
) -> go.Figure:
    """Generate 2D scatter plot comparing two numerical features."""
    clean_df = df[
        [x_col, y_col] + ([target_col] if target_col and target_col in df.columns else [])
    ].dropna()

    if target_col and target_col in clean_df.columns:
        fig = px.scatter(
            clean_df,
            x=x_col,
            y=y_col,
            color=clean_df[target_col].astype(str),
            color_discrete_sequence=CHART_PALETTE,
            opacity=0.75,
            labels={target_col: "Target Class"},
        )
        fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="#FFFFFF")))
    else:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=clean_df[x_col],
                    y=clean_df[y_col],
                    mode="markers",
                    marker=dict(color=COLOR_PRIMARY, size=7, opacity=0.75),
                    hovertemplate=f"<b>{x_col}:</b> %{{x}}<br><b>{y_col}:</b> %{{y}}<extra></extra>",
                )
            ]
        )

    apply_studio_chart_theme(
        fig,
        title=f"Bivariate Scatter: {x_col} vs {y_col}",
        height=450,
        show_legend=bool(target_col and target_col in clean_df.columns),
    )
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> go.Figure:
    """Generate an annotated Pearson correlation matrix heatmap for numerical features."""
    num_df = df.select_dtypes(include=[np.number])
    if columns:
        valid_cols = [c for c in columns if c in num_df.columns]
        if valid_cols:
            num_df = num_df[valid_cols]

    if num_df.empty or num_df.shape[1] < 2:
        fig = go.Figure()
        apply_studio_chart_theme(
            fig, title="Correlation Matrix (Insufficient Numerical Features)", height=300
        )
        return fig

    # Drop constant columns to avoid NaN in Pearson correlation
    varying_cols = [c for c in num_df.columns if num_df[c].nunique() > 1]
    if len(varying_cols) >= 2:
        num_df = num_df[varying_cols]

    corr = num_df.corr(method="pearson").round(3)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        labels=dict(color="Pearson r"),
    )

    chart_size = max(450, min(850, len(corr.columns) * 36 + 120))
    apply_studio_chart_theme(
        fig,
        title=f"Pearson Correlation Matrix ({len(corr.columns)} Features)",
        height=chart_size,
        show_legend=False,
    )
    fig.update_traces(textfont=dict(size=10, color=COLOR_TEXT))
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        xaxis=dict(tickangle=-45),
        margin=dict(l=80, r=40, t=50, b=80),
    )
    return fig


def plot_categorical_distribution(df: pd.DataFrame, column: str, top_n: int = 15) -> go.Figure:
    """Generate a frequency bar chart for a categorical column."""
    counts = df[column].astype(str).value_counts().head(top_n)
    labels = list(counts.index)
    values = list(counts.values)

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                text=[f"{v:,}" for v in values],
                textposition="outside",
                cliponaxis=False,
                marker=dict(color=COLOR_TEAL, line=dict(color=COLOR_BORDER, width=1)),
                hovertemplate="<b>Category:</b> %{x}<br><b>Count:</b> %{y:,}<extra></extra>",
            )
        ]
    )

    apply_studio_chart_theme(
        fig,
        title=f"Category Distribution: {column} (Top {len(labels)})",
        height=360,
        show_legend=False,
    )
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Record Count",
        xaxis=dict(tickangle=-30 if any(len(str(x)) > 8 for x in labels) else 0),
        yaxis=dict(range=[0, max(values) * 1.18 if values else 10]),
    )
    return fig


def plot_pca_projection(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str | None = None,
) -> go.Figure:
    """Compute and display 2D PCA projection of numerical features."""
    sub_df = df[feature_cols].select_dtypes(include=[np.number]).dropna()
    if sub_df.shape[1] < 2 or len(sub_df) < 3:
        fig = go.Figure()
        apply_studio_chart_theme(fig, title="PCA Projection (Insufficient Data)", height=350)
        return fig

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(sub_df)

    pca = PCA(n_components=2)
    components = pca.fit_transform(x_scaled)
    var_exp = pca.explained_variance_ratio_ * 100

    pca_df = pd.DataFrame(
        {
            "PC1": components[:, 0],
            "PC2": components[:, 1],
        },
        index=sub_df.index,
    )

    if target_col and target_col in df.columns:
        pca_df["Target"] = df.loc[sub_df.index, target_col].astype(str)
        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Target",
            color_discrete_sequence=CHART_PALETTE,
            opacity=0.8,
            labels={"Target": "Class"},
        )
        fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="#FFFFFF")))
    else:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=pca_df["PC1"],
                    y=pca_df["PC2"],
                    mode="markers",
                    marker=dict(color=COLOR_PRIMARY, size=7, opacity=0.8),
                    hovertemplate="<b>PC1:</b> %{x:.2f}<br><b>PC2:</b> %{y:.2f}<extra></extra>",
                )
            ]
        )

    apply_studio_chart_theme(
        fig,
        title=f"Principal Component Analysis (Explains {sum(var_exp):.1f}% Variance)",
        height=450,
        show_legend=bool(target_col and target_col in df.columns),
    )
    fig.update_layout(
        xaxis_title=f"Principal Component 1 ({var_exp[0]:.1f}% Variance)",
        yaxis_title=f"Principal Component 2 ({var_exp[1]:.1f}% Variance)",
    )
    return fig


# ===========================================================================
# 2. Model Evaluation & Diagnostic Visualizations
# ===========================================================================


def plot_confusion_matrix(
    cm: list[list[int]], labels: list[str], model_name: str = "Classifier"
) -> go.Figure:
    """Generate an annotated Plotly heatmap for the confusion matrix."""
    colorscale = [
        [0.0, "#EEF2FF"],
        [0.3, "#C7D2FE"],
        [0.7, "#6366F1"],
        [1.0, "#312E81"],
    ]
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale=colorscale,
    )
    fig.update_traces(textfont=dict(size=14, color="#172033"))
    apply_studio_chart_theme(
        fig, title=f"Confusion Matrix Heatmap — {model_name}", height=420, show_legend=False
    )
    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="Actual (Ground Truth) Label",
    )
    return fig


def plot_prediction_distribution(
    y_pred: list[Any] | np.ndarray,
    y_true: list[Any] | np.ndarray | None = None,
    labels: list[str] | None = None,
    model_name: str = "Classifier",
) -> go.Figure:
    """Plot distribution of predicted classes, optionally side-by-side with ground truth."""
    pred_series = pd.Series([str(x) for x in y_pred])
    unique_labels = labels or sorted(pred_series.unique())

    pred_counts = [int(pred_series.value_counts().get(lbl, 0)) for lbl in unique_labels]

    fig = go.Figure()
    if y_true is not None:
        true_series = pd.Series([str(x) for x in y_true])
        true_counts = [int(true_series.value_counts().get(lbl, 0)) for lbl in unique_labels]

        fig.add_trace(
            go.Bar(
                name="Ground Truth Test",
                x=unique_labels,
                y=true_counts,
                text=true_counts,
                textposition="outside",
                cliponaxis=False,
                marker=dict(color=COLOR_SLATE),
                hovertemplate="<b>Actual %{x}:</b> %{y:,}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Bar(
            name=f"Predicted ({model_name})",
            x=unique_labels,
            y=pred_counts,
            text=pred_counts,
            textposition="outside",
            cliponaxis=False,
            marker=dict(color=COLOR_PRIMARY),
            hovertemplate="<b>Predicted %{x}:</b> %{y:,}<extra></extra>",
        )
    )

    apply_studio_chart_theme(
        fig,
        title="Class Frequency: Predictions vs Ground Truth",
        height=380,
        show_legend=bool(y_true is not None),
    )
    fig.update_layout(
        barmode="group",
        xaxis_title="Class Label",
        yaxis_title="Record Count",
    )
    return fig


def plot_feature_coefficients(
    feature_names: list[str],
    coefficients: np.ndarray,
    title: str = "Feature Coefficients",
    is_importance: bool = False,
    top_n: int = 20,
) -> go.Figure:
    """Generate horizontal bar chart of feature weights or importances."""
    coef_arr = np.asarray(coefficients).flatten()
    if len(coef_arr) != len(feature_names):
        # Truncate or match length safely
        min_len = min(len(coef_arr), len(feature_names))
        coef_arr = coef_arr[:min_len]
        feature_names = feature_names[:min_len]

    df_coef = (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Value": coef_arr,
                "AbsValue": np.abs(coef_arr),
            }
        )
        .sort_values(by="AbsValue", ascending=False)
        .head(top_n)
    )

    df_coef = df_coef.sort_values(by="Value", ascending=True)

    if is_importance:
        colors = [COLOR_TEAL] * len(df_coef)
        tooltip = "<b>Feature:</b> %{y}<br><b>Importance:</b> %{x:.4f}<extra></extra>"
        x_label = "Relative Feature Importance"
    else:
        colors = [COLOR_PRIMARY if v >= 0 else COLOR_AMBER for v in df_coef["Value"]]
        tooltip = "<b>Feature:</b> %{y}<br><b>Coefficient:</b> %{x:.4f}<extra></extra>"
        x_label = "Signed Weight / Coefficient (Standardized Scale)"

    fig = go.Figure(
        data=[
            go.Bar(
                y=df_coef["Feature"],
                x=df_coef["Value"],
                orientation="h",
                text=[f"{v:.3f}" for v in df_coef["Value"]],
                textposition="outside",
                cliponaxis=False,
                marker=dict(color=colors, line=dict(color=COLOR_BORDER, width=1)),
                hovertemplate=tooltip,
            )
        ]
    )

    chart_height = max(380, len(df_coef) * 24 + 100)
    apply_studio_chart_theme(
        fig, title=f"{title} (Top {len(df_coef)})", height=chart_height, show_legend=False
    )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title="Feature Name",
        margin=dict(l=150, r=40, t=50, b=45),
    )
    return fig


def plot_roc_curve(roc_curve_data: Any, model_name: str = "Classifier") -> go.Figure:
    """Generate interactive ROC Curve with chance diagonal line."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=roc_curve_data.x_values,
            y=roc_curve_data.y_values,
            mode="lines",
            name=f"{model_name} (AUC = {roc_curve_data.score:.4f})"
            if roc_curve_data.score
            else model_name,
            line=dict(color=COLOR_PRIMARY, width=2.8),
            hovertemplate="<b>FPR:</b> %{x:.3f}<br><b>TPR:</b> %{y:.3f}<extra></extra>",
        )
    )

    # Diagonal chance line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guessing (AUC = 0.50)",
            line=dict(color="#94A3B8", dash="dash", width=1.5),
            hoverinfo="skip",
        )
    )

    apply_studio_chart_theme(
        fig, title="Receiver Operating Characteristic (ROC)", height=420, show_legend=True
    )
    fig.update_layout(
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity / Recall)",
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.05]),
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
            line=dict(color=COLOR_TEAL, width=2.8),
            hovertemplate="<b>Recall:</b> %{x:.3f}<br><b>Precision:</b> %{y:.3f}<extra></extra>",
        )
    )

    apply_studio_chart_theme(fig, title="Precision-Recall Curve", height=420, show_legend=True)
    fig.update_layout(
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (Positive Predictive Value)",
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.05]),
    )
    return fig


# ===========================================================================
# 3. Multi-Model Comparison Visualizations
# ===========================================================================


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

    models = df_comp["Model"].unique()
    colors = [get_model_color(m, i) for i, m in enumerate(models)]

    fig = px.bar(
        df_melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        range_y=[0.0, 1.05],
        text=df_melted["Score"].apply(lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else ""),
        color_discrete_sequence=colors,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    apply_studio_chart_theme(
        fig, title="Multi-Model Benchmark on Identical Test Partition", height=450, show_legend=True
    )
    fig.update_layout(
        yaxis_title="Metric Score (0.00 – 1.00)",
        xaxis_title="",
    )
    return fig


def plot_metric_ranking(
    comparison_table: list[dict[str, Any]], metric_name: str = "Macro F1"
) -> go.Figure:
    """Rank models by a single chosen evaluation metric."""
    df_comp = pd.DataFrame(comparison_table)
    if metric_name not in df_comp.columns:
        metric_name = "Accuracy" if "Accuracy" in df_comp.columns else df_comp.columns[1]

    df_sorted = df_comp.sort_values(by=metric_name, ascending=True)
    models = list(df_sorted["Model"])
    scores = list(df_sorted[metric_name])
    colors = [get_model_color(m, i) for i, m in enumerate(models)]

    fig = go.Figure(
        data=[
            go.Bar(
                y=models,
                x=scores,
                orientation="h",
                text=[f"{s:.4f}" for s in scores],
                textposition="outside",
                cliponaxis=False,
                marker=dict(color=colors, line=dict(color=COLOR_BORDER, width=1)),
                hovertemplate="<b>%{y}:</b> %{x:.4f}<extra></extra>",
            )
        ]
    )

    apply_studio_chart_theme(
        fig, title=f"Model Ranking by {metric_name}", height=320, show_legend=False
    )
    fig.update_layout(
        xaxis_title=f"{metric_name} Score",
        yaxis_title="",
        xaxis=dict(range=[0.0, 1.15]),
        margin=dict(l=160, r=40, t=45, b=40),
    )
    return fig


def plot_multi_model_roc(training_results: list[Any]) -> go.Figure:
    """Overlay ROC curves for multiple models on a shared diagnostic plot."""
    fig = go.Figure()
    valid_count = 0

    for idx, tr in enumerate(training_results):
        if getattr(tr.metrics, "roc_curve", None) is not None:
            rc = tr.metrics.roc_curve
            color = get_model_color(tr.display_name, idx)
            name = f"{tr.display_name} (AUC = {rc.score:.4f})" if rc.score else tr.display_name
            fig.add_trace(
                go.Scatter(
                    x=rc.x_values,
                    y=rc.y_values,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=2.5),
                    hovertemplate=f"<b>{tr.display_name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
                )
            )
            valid_count += 1

    # Add diagonal chance reference
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guessing (AUC = 0.50)",
            line=dict(color="#94A3B8", dash="dash", width=1.5),
            hoverinfo="skip",
        )
    )

    apply_studio_chart_theme(
        fig,
        title=f"Comparative ROC Curves ({valid_count} Classifiers with Continuous Scores)",
        height=450,
        show_legend=True,
    )
    fig.update_layout(
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity / Recall)",
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.05]),
    )
    return fig


def plot_multi_model_pr(training_results: list[Any]) -> go.Figure:
    """Overlay Precision-Recall curves for multiple models on a shared plot."""
    fig = go.Figure()
    valid_count = 0

    for idx, tr in enumerate(training_results):
        if getattr(tr.metrics, "pr_curve", None) is not None:
            pr = tr.metrics.pr_curve
            color = get_model_color(tr.display_name, idx)
            fig.add_trace(
                go.Scatter(
                    x=pr.x_values,
                    y=pr.y_values,
                    mode="lines",
                    name=f"{tr.display_name} PR",
                    line=dict(color=color, width=2.5),
                    hovertemplate=f"<b>{tr.display_name}</b><br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>",
                )
            )
            valid_count += 1

    apply_studio_chart_theme(
        fig,
        title=f"Comparative Precision-Recall Curves ({valid_count} Classifiers)",
        height=450,
        show_legend=True,
    )
    fig.update_layout(
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (Positive Predictive Value)",
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.05]),
    )
    return fig


# ===========================================================================
# 4. Predictions & Inference Visualizations
# ===========================================================================


def plot_batch_prediction_distribution(
    predicted_classes: list[Any], class_labels: list[str] | None = None
) -> go.Figure:
    """Generate bar chart of class counts and percentages from batch prediction output."""
    series = pd.Series([str(x) for x in predicted_classes])
    counts_dict = series.value_counts().to_dict()
    labels = class_labels or sorted(counts_dict.keys())
    counts = [counts_dict.get(lbl, 0) for lbl in labels]
    total = sum(counts) if counts else 1
    pcts = [(c / total) * 100 for c in counts]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=counts,
                text=[f"{c:,} ({p:.1f}%)" for c, p in zip(counts, pcts, strict=False)],
                textposition="outside",
                cliponaxis=False,
                marker=dict(
                    color=[CHART_PALETTE[i % len(CHART_PALETTE)] for i in range(len(labels))]
                ),
                hovertemplate="<b>Class %{x}:</b> %{y:,} records<extra></extra>",
            )
        ]
    )

    apply_studio_chart_theme(
        fig, title="Batch Prediction Output Class Distribution", height=350, show_legend=False
    )
    fig.update_layout(
        xaxis_title="Predicted Class",
        yaxis_title="Predicted Record Count",
        yaxis=dict(range=[0, max(counts) * 1.2 if counts else 10]),
    )
    return fig


def plot_batch_probabilities(
    probabilities: list[list[float]], class_labels: list[str]
) -> go.Figure:
    """Display histogram or box distribution of predicted probabilities across batch records."""
    prob_arr = np.array(probabilities)
    fig = go.Figure()

    for col_idx, lbl in enumerate(class_labels):
        if col_idx < prob_arr.shape[1]:
            col_probs = prob_arr[:, col_idx]
            color = CHART_PALETTE[col_idx % len(CHART_PALETTE)]
            fig.add_trace(
                go.Box(
                    y=col_probs,
                    name=f"Class {lbl}",
                    marker=dict(color=color),
                    boxmean=True,
                    hovertemplate=f"<b>Class {lbl} Prob:</b> %{{y:.3f}}<extra></extra>",
                )
            )

    apply_studio_chart_theme(
        fig,
        title="Predicted Probability Distribution across Batch Records",
        height=380,
        show_legend=True,
    )
    fig.update_layout(
        yaxis_title="Assigned Probability Score (0.0 to 1.0)",
        yaxis=dict(range=[-0.05, 1.05]),
    )
    return fig


def plot_actual_vs_predicted_batch(
    actual_labels: list[Any],
    predicted_labels: list[Any],
    class_labels: list[str] | None = None,
) -> go.Figure:
    """Compare ground truth vs batch predictions when target is present in upload."""
    act_s = pd.Series([str(x) for x in actual_labels])
    pred_s = pd.Series([str(x) for x in predicted_labels])

    labels = class_labels or sorted(set(act_s.unique()).union(set(pred_s.unique())))
    act_counts = [int(act_s.value_counts().get(lbl, 0)) for lbl in labels]
    pred_counts = [int(pred_s.value_counts().get(lbl, 0)) for lbl in labels]

    fig = go.Figure(
        data=[
            go.Bar(
                name="Actual Labels",
                x=labels,
                y=act_counts,
                text=act_counts,
                textposition="outside",
                cliponaxis=False,
                marker=dict(color=COLOR_SLATE),
                hovertemplate="<b>Actual %{x}:</b> %{y:,}<extra></extra>",
            ),
            go.Bar(
                name="Predicted Labels",
                x=labels,
                y=pred_counts,
                text=pred_counts,
                textposition="outside",
                cliponaxis=False,
                marker=dict(color=COLOR_PRIMARY),
                hovertemplate="<b>Predicted %{x}:</b> %{y:,}<extra></extra>",
            ),
        ]
    )

    apply_studio_chart_theme(
        fig, title="Batch Verification: Actual vs Predicted Labels", height=380, show_legend=True
    )
    fig.update_layout(
        barmode="group",
        xaxis_title="Class Label",
        yaxis_title="Record Count",
    )
    return fig


# ===========================================================================
# 5. Educational / Conceptual Visualizations
# ===========================================================================


def plot_decision_boundary_concept() -> go.Figure:
    """Generate illustrative 2D synthetic Logistic Regression boundary for Learning Guide."""
    np.random.seed(42)
    # Synthetic 2D Gaussian clusters
    x0 = np.random.randn(50, 2) * 0.9 + np.array([-1.2, -1.0])
    x1 = np.random.randn(50, 2) * 0.9 + np.array([1.2, 1.0])
    np.vstack([x0, x1])
    np.array([0] * 50 + [1] * 50)

    # Simple fitted linear separator: w1*x1 + w2*x2 + b = 0
    w = np.array([1.2, 1.0])
    b = 0.1

    # Grid for contour
    x_grid = np.linspace(-3.5, 3.5, 100)
    y_grid = np.linspace(-3.5, 3.5, 100)
    xx, yy = np.meshgrid(x_grid, y_grid)
    z = 1 / (1 + np.exp(-(w[0] * xx + w[1] * yy + b)))

    fig = go.Figure()

    # Probability contour
    fig.add_trace(
        go.Contour(
            x=x_grid,
            y=y_grid,
            z=z,
            colorscale=[[0, "#EEF2FF"], [0.5, "#C7D2FE"], [1, "#312E81"]],
            opacity=0.6,
            showscale=False,
            contours=dict(start=0.1, end=0.9, size=0.2),
            hoverinfo="skip",
        )
    )

    # Class 0 points
    fig.add_trace(
        go.Scatter(
            x=x0[:, 0],
            y=x0[:, 1],
            mode="markers",
            name="Class 0 (Malignant)",
            marker=dict(color=COLOR_PRIMARY, size=8, line=dict(color="#FFFFFF", width=1)),
            hovertemplate="Class 0<br>x1: %{x:.2f}<br>x2: %{y:.2f}<extra></extra>",
        )
    )

    # Class 1 points
    fig.add_trace(
        go.Scatter(
            x=x1[:, 0],
            y=x1[:, 1],
            mode="markers",
            name="Class 1 (Benign)",
            marker=dict(color=COLOR_TEAL, size=8, line=dict(color="#FFFFFF", width=1)),
            hovertemplate="Class 1<br>x1: %{x:.2f}<br>x2: %{y:.2f}<extra></extra>",
        )
    )

    # Decision Boundary line w^T x + b = 0  => x2 = (-w1*x1 - b)/w2
    x_line = np.linspace(-3.0, 3.0, 50)
    y_line = (-w[0] * x_line - b) / w[1]
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Decision Boundary (P = 0.5)",
            line=dict(color="#172033", width=2.5),
            hoverinfo="skip",
        )
    )

    apply_studio_chart_theme(
        fig,
        title="[Illustrative Concept] Logistic Regression 2D Decision Boundary & Probability Contours",
        height=430,
        show_legend=True,
    )
    fig.update_layout(
        xaxis_title="Synthetic Feature Dimension 1",
        yaxis_title="Synthetic Feature Dimension 2",
        xaxis=dict(range=[-3.5, 3.5]),
        yaxis=dict(range=[-3.5, 3.5]),
    )
    return fig


def plot_svm_margin_concept() -> go.Figure:
    """Generate illustrative 2D synthetic Linear SVM margins and support vectors."""
    np.random.seed(101)
    x0 = np.random.randn(30, 2) * 0.7 + np.array([-1.5, -0.8])
    x1 = np.random.randn(30, 2) * 0.7 + np.array([1.5, 0.8])

    w = np.array([1.0, 0.8])
    b = 0.0

    x_line = np.linspace(-3.0, 3.0, 50)
    y_hyp = (-w[0] * x_line - b) / w[1]
    y_pos = (-w[0] * x_line - b + 1.0) / w[1]
    y_neg = (-w[0] * x_line - b - 1.0) / w[1]

    fig = go.Figure()

    # Hyperplane
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_hyp,
            mode="lines",
            name="Separating Hyperplane (wᵀx + b = 0)",
            line=dict(color="#172033", width=2.5),
        )
    )
    # Margins
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_pos,
            mode="lines",
            name="Positive Margin (wᵀx + b = +1)",
            line=dict(color=COLOR_TEAL, width=1.5, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_neg,
            mode="lines",
            name="Negative Margin (wᵀx + b = -1)",
            line=dict(color=COLOR_PRIMARY, width=1.5, dash="dash"),
        )
    )

    # Class points
    fig.add_trace(
        go.Scatter(
            x=x0[:, 0],
            y=x0[:, 1],
            mode="markers",
            name="Class -1",
            marker=dict(color=COLOR_PRIMARY, size=8, line=dict(color="#FFFFFF", width=1)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x1[:, 0],
            y=x1[:, 1],
            mode="markers",
            name="Class +1",
            marker=dict(color=COLOR_TEAL, size=8, line=dict(color="#FFFFFF", width=1)),
        )
    )

    apply_studio_chart_theme(
        fig,
        title="[Illustrative Concept] Linear SVM Maximum Margin & Support Vector Geometry",
        height=430,
        show_legend=True,
    )
    fig.update_layout(
        xaxis_title="Synthetic Feature 1",
        yaxis_title="Synthetic Feature 2",
        xaxis=dict(range=[-3.2, 3.2]),
        yaxis=dict(range=[-3.2, 3.2]),
    )
    return fig


def plot_threshold_tradeoff_concept() -> go.Figure:
    """Generate illustrative precision vs recall trade-off curve across classification thresholds."""
    thresholds = np.linspace(0.05, 0.95, 40)
    # Realistic monotonic trade-off curves
    precision = 1 / (1 + np.exp(-4 * (thresholds - 0.2)))
    recall = 1 / (1 + np.exp(4 * (thresholds - 0.6)))
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=precision,
            mode="lines+markers",
            name="Precision (Positive Predictive Value)",
            line=dict(color=COLOR_INDIGO, width=2.5),
            marker=dict(size=5),
            hovertemplate="<b>Threshold:</b> %{x:.2f}<br><b>Precision:</b> %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=recall,
            mode="lines+markers",
            name="Recall (Sensitivity)",
            line=dict(color=COLOR_TEAL, width=2.5),
            marker=dict(size=5),
            hovertemplate="<b>Threshold:</b> %{x:.2f}<br><b>Recall:</b> %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=f1,
            mode="lines",
            name="Harmonic F1-Score",
            line=dict(color=COLOR_AMBER, width=2, dash="dash"),
            hovertemplate="<b>Threshold:</b> %{x:.2f}<br><b>F1:</b> %{y:.3f}<extra></extra>",
        )
    )

    apply_studio_chart_theme(
        fig,
        title="[Illustrative Concept] Precision vs Recall vs Threshold Simulation",
        height=400,
        show_legend=True,
    )
    fig.update_layout(
        xaxis_title="Decision Threshold (τ)",
        yaxis_title="Score (0.0 to 1.0)",
        xaxis=dict(range=[0.0, 1.0]),
        yaxis=dict(range=[0.0, 1.05]),
    )
    return fig
