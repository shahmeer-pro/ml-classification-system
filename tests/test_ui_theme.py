"""Unit tests for the ML Studio UI design system and theme components."""

import plotly.graph_objects as go

from ml_app.ui.components import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
)
from ml_app.ui.theme import (
    COLOR_BG,
    COLOR_PRIMARY,
    COLOR_SURFACE,
    COLOR_TEXT,
    STUDIO_CSS,
    apply_studio_chart_theme,
)


def test_theme_color_tokens() -> None:
    """Verify primary and surface color tokens match specification."""
    assert COLOR_BG == "#F6F8FC"
    assert COLOR_SURFACE == "#FFFFFF"
    assert COLOR_TEXT == "#172033"
    assert COLOR_PRIMARY == "#4F46E5"


def test_studio_css_contains_design_system_rules() -> None:
    """Ensure STUDIO_CSS includes sidebar, brand, card, and button classes."""
    assert ".stApp" in STUDIO_CSS
    assert ".studio-card" in STUDIO_CSS
    assert ".studio-kpi" in STUDIO_CSS
    assert ".studio-banner" in STUDIO_CSS
    assert ".sidebar-brand" in STUDIO_CSS
    assert 'data-testid="stSidebarNavLink"' in STUDIO_CSS


def test_apply_studio_chart_theme() -> None:
    """Ensure studio chart theme sets font, backgrounds, and layout properly."""
    fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
    fig = apply_studio_chart_theme(fig, title="Test Chart", height=350, show_legend=True)

    assert fig.layout.paper_bgcolor == COLOR_SURFACE
    assert fig.layout.plot_bgcolor == COLOR_SURFACE
    assert fig.layout.height == 350
    assert "Test Chart" in str(fig.layout.title.text)
    assert fig.layout.font.color == COLOR_TEXT


def test_plot_confusion_matrix_uses_custom_theme() -> None:
    """Verify confusion matrix chart generator applies studio theme."""
    cm = [[50, 2], [3, 45]]
    labels = ["Malignant", "Benign"]
    fig = plot_confusion_matrix(cm, labels)

    assert fig.layout.height == 420
    assert fig.layout.paper_bgcolor == COLOR_SURFACE
    assert "Confusion Matrix" in str(fig.layout.title.text)


def test_plot_class_distribution() -> None:
    """Verify class balance distribution chart generator."""
    dist = {"0": 212, "1": 357}
    fig = plot_class_distribution(dist)

    assert fig.layout.height == 360
    assert fig.layout.paper_bgcolor == COLOR_SURFACE


def test_plot_model_comparison() -> None:
    """Verify model comparison bar chart generator."""
    table = [
        {"Model": "GaussianNB", "Accuracy": 0.94, "Macro F1": 0.93},
        {"Model": "LogisticRegression", "Accuracy": 0.96, "Macro F1": 0.95},
    ]
    fig = plot_model_comparison(table)

    assert fig.layout.height == 450
    assert fig.layout.paper_bgcolor == COLOR_SURFACE
