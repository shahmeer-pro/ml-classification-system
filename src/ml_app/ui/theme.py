"""Design System and UI Theme for ML Studio.

Implements clean, accessible, modern SaaS-grade styling tokens, custom CSS injection,
and reusable card, metric, and header components.
"""

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Design System Tokens
# ---------------------------------------------------------------------------
COLOR_BG = "#F6F8FC"
COLOR_SURFACE = "#FFFFFF"
COLOR_TEXT = "#172033"
COLOR_TEXT_MUTED = "#64748B"
COLOR_BORDER = "#E2E8F0"
COLOR_PRIMARY = "#4F46E5"
COLOR_PRIMARY_HOVER = "#4338CA"
COLOR_SECONDARY = "#2563EB"
COLOR_SUCCESS = "#15803D"
COLOR_WARNING = "#B45309"
COLOR_ERROR = "#B91C1C"
COLOR_SIDEBAR_BG = "#111827"
COLOR_SIDEBAR_TEXT = "#CBD5E1"
COLOR_SIDEBAR_ACTIVE = "#1F2937"

CHART_PALETTE = ["#4F46E5", "#2563EB", "#0D9488", "#F59E0B", "#64748B", "#8B5CF6"]

STUDIO_CSS = f"""
<style>
/* Main app surface background */
.stApp {{
    background-color: {COLOR_BG};
    color: {COLOR_TEXT};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}}

/* Sidebar styling */
[data-testid="stSidebar"] {{
    background-color: {COLOR_SIDEBAR_BG} !important;
    border-right: 1px solid #1F2937;
}}
[data-testid="stSidebar"] * {{
    color: {COLOR_SIDEBAR_TEXT};
}}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: #FFFFFF !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: #1F2937 !important;
}}

/* Modern Streamlit Navigation (st.navigation) styling */
[data-testid="stSidebarNavLink"] {{
    border-radius: 6px !important;
    padding: 0.5rem 0.75rem !important;
    margin: 0.15rem 0.5rem !important;
    transition: all 0.15s ease-in-out !important;
    color: {COLOR_SIDEBAR_TEXT} !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
}}
[data-testid="stSidebarNavLink"]:hover {{
    background-color: {COLOR_SIDEBAR_ACTIVE} !important;
    color: #FFFFFF !important;
}}
[data-testid="stSidebarNavLink"][aria-current="page"] {{
    background-color: {COLOR_SIDEBAR_ACTIVE} !important;
    color: #FFFFFF !important;
    border-left: 3px solid {COLOR_PRIMARY} !important;
    font-weight: 600 !important;
}}
[data-testid="stSidebarNavLink"] span {{
    color: inherit !important;
}}

/* Sidebar Brand Header */
.sidebar-brand {{
    padding: 1.25rem 1rem 0.75rem 1rem;
    border-bottom: 1px solid #1F2937;
    margin-bottom: 0.75rem;
}}
.sidebar-brand-title {{
    font-size: 1.2rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    color: #FFFFFF !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.sidebar-brand-subtitle {{
    font-size: 0.75rem;
    color: #94A3B8 !important;
    letter-spacing: 0.02em;
    margin-top: 0.2rem;
    margin-bottom: 0;
}}
.sidebar-footer {{
    padding: 1rem 0.75rem;
    margin-top: 1.5rem;
    border-top: 1px solid #1F2937;
    font-size: 0.75rem;
    color: #64748B !important;
}}

/* Studio Card Component */
.studio-card {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px 0 rgba(0, 0, 0, 0.02);
}}

/* Page Header Component */
.studio-header {{
    margin-bottom: 1.5rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid {COLOR_BORDER};
}}
.studio-header-title {{
    font-size: 1.75rem;
    font-weight: 700;
    color: {COLOR_TEXT};
    margin: 0;
    line-height: 1.2;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}}
.studio-header-subtitle {{
    font-size: 0.95rem;
    color: {COLOR_TEXT_MUTED};
    margin-top: 0.35rem;
    margin-bottom: 0;
}}
.studio-badge {{
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.55rem;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: 4px;
    background-color: #EEF2FF;
    color: {COLOR_PRIMARY};
    border: 1px solid #C7D2FE;
}}
.studio-badge-success {{
    background-color: #F0FDF4;
    color: {COLOR_SUCCESS};
    border: 1px solid #BBF7D0;
}}

/* KPI Metric Scorecard */
.studio-kpi {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 1rem 1.25rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
}}
.studio-kpi-label {{
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: {COLOR_TEXT_MUTED};
    margin-bottom: 0.35rem;
}}
.studio-kpi-value {{
    font-size: 1.65rem;
    font-weight: 700;
    color: {COLOR_TEXT};
    line-height: 1.1;
    margin-bottom: 0.25rem;
}}
.studio-kpi-subtext {{
    font-size: 0.75rem;
    color: {COLOR_TEXT_MUTED};
}}

/* Compact Information Banner */
.studio-banner {{
    background-color: #F8FAFC;
    border: 1px solid {COLOR_BORDER};
    border-left: 4px solid {COLOR_PRIMARY};
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-bottom: 1.25rem;
    font-size: 0.875rem;
    color: {COLOR_TEXT};
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
}}
.studio-banner-title {{
    font-weight: 600;
    color: {COLOR_TEXT};
    margin-bottom: 0.15rem;
}}

/* Button styling enhancements */
.stButton>button[kind="primary"] {{
    background-color: {COLOR_PRIMARY} !important;
    border-color: {COLOR_PRIMARY} !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    padding: 0.45rem 1.2rem !important;
    box-shadow: 0 1px 2px 0 rgba(79, 70, 229, 0.2) !important;
    transition: all 0.15s ease-in-out !important;
}}
.stButton>button[kind="primary"]:hover {{
    background-color: {COLOR_PRIMARY_HOVER} !important;
    border-color: {COLOR_PRIMARY_HOVER} !important;
}}
.stButton>button[kind="secondary"] {{
    background-color: {COLOR_SURFACE} !important;
    border: 1px solid {COLOR_BORDER} !important;
    color: {COLOR_TEXT} !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}}
.stButton>button[kind="secondary"]:hover {{
    border-color: {COLOR_PRIMARY} !important;
    color: {COLOR_PRIMARY} !important;
}}

/* Dataframe & Table borders */
[data-testid="stDataFrame"] {{
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
    overflow: hidden;
}}

/* Form container */
[data-testid="stForm"] {{
    border: 1px solid {COLOR_BORDER} !important;
    border-radius: 8px !important;
    background-color: {COLOR_SURFACE} !important;
    padding: 1.25rem !important;
}}

/* Section titles */
.studio-section-title {{
    font-size: 1.1rem;
    font-weight: 600;
    color: {COLOR_TEXT};
    margin-top: 1.25rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}}

/* Empty state container */
.studio-empty-state {{
    background-color: {COLOR_SURFACE};
    border: 1px dashed {COLOR_BORDER};
    border-radius: 8px;
    padding: 2.25rem 1.5rem;
    text-align: center;
    margin: 1.25rem 0;
}}
.studio-empty-state-title {{
    font-size: 1.05rem;
    font-weight: 600;
    color: {COLOR_TEXT};
    margin-bottom: 0.4rem;
}}
.studio-empty-state-text {{
    font-size: 0.85rem;
    color: {COLOR_TEXT_MUTED};
    max-width: 460px;
    margin: 0 auto;
}}
</style>
"""


def inject_custom_theme() -> None:
    """Inject studio CSS styles once per page render."""
    st.markdown(STUDIO_CSS, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str, badge: str | None = None) -> None:
    """Render a clean, responsive page header."""
    badge_html = f'<span class="studio-badge">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="studio-header">
            <div class="studio-header-title">
                {title} {badge_html}
            </div>
            <p class="studio-header-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str,
    value: str,
    subtext: str = "",
    status_color: str | None = None,
) -> None:
    """Render a clean SaaS metric card."""
    val_color = f"color: {status_color};" if status_color else f"color: {COLOR_TEXT};"
    st.markdown(
        f"""
        <div class="studio-kpi">
            <div class="studio-kpi-label">{label}</div>
            <div class="studio-kpi-value" style="{val_color}">{value}</div>
            <div class="studio-kpi-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer_banner() -> None:
    """Render educational & clinical non-use information banner."""
    st.markdown(
        """
        <div class="studio-banner">
            <div>
                <div class="studio-banner-title">Educational & Portfolio Environment</div>
                <div style="font-size: 0.8rem; color: #64748B;">
                    This workspace and the built-in Breast Cancer Wisconsin dataset are intended strictly for academic demonstration,
                    model comparison, and machine learning portfolio purposes. Not validated or certified for clinical diagnosis.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state_box(title: str, message: str, instruction: str = "") -> None:
    """Render a clean empty-state container."""
    inst_html = (
        f'<div style="margin-top: 0.5rem; font-size: 0.8rem; font-weight: 500; color: {COLOR_PRIMARY};">{instruction}</div>'
        if instruction
        else ""
    )
    st.markdown(
        f"""
        <div class="studio-empty-state">
            <div class="studio-empty-state-title">{title}</div>
            <div class="studio-empty-state-text">{message}</div>
            {inst_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_studio_chart_theme(
    fig: go.Figure,
    title: str = "",
    height: int = 400,
    show_legend: bool = True,
) -> go.Figure:
    """Apply consistent technical chart styling to Plotly figures."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=14, color=COLOR_TEXT, family="sans-serif"),
            x=0.01,
            y=0.96,
        ),
        paper_bgcolor=COLOR_SURFACE,
        plot_bgcolor=COLOR_SURFACE,
        font=dict(color=COLOR_TEXT, family="sans-serif", size=12),
        height=height,
        margin=dict(l=45, r=30, t=45, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=COLOR_TEXT_MUTED),
        )
        if show_legend
        else dict(visible=False),
        xaxis=dict(
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            linecolor=COLOR_BORDER,
            tickfont=dict(size=11, color=COLOR_TEXT_MUTED),
        ),
        yaxis=dict(
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            linecolor=COLOR_BORDER,
            tickfont=dict(size=11, color=COLOR_TEXT_MUTED),
        ),
    )
    return fig


def render_section_title(title: str, badge: str | None = None) -> None:
    """Render a clean SaaS section heading."""
    badge_html = f'<span class="studio-badge">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="studio-section-title">
            {title} {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    """Render the sidebar branding header."""
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#818CF8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                    <line x1="8" y1="21" x2="16" y2="21"></line>
                    <line x1="12" y1="17" x2="12" y2="21"></line>
                </svg>
                ML STUDIO
            </div>
            <p class="sidebar-brand-subtitle">Classification Workbench</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
