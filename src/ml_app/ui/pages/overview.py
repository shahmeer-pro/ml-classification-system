"""Overview and Architecture page."""

import streamlit as st

from ml_app.models.registry import MODEL_REGISTRY
from ml_app.ui.components import render_disclaimer, render_metrics_cards
from ml_app.ui.theme import (
    COLOR_SUCCESS,
    render_empty_state_box,
    render_kpi_card,
    render_page_header,
    render_section_title,
)
from ml_app.ui.visualizations import (
    plot_class_distribution,
    plot_dataset_profile,
    plot_model_comparison,
)


def page_overview() -> None:
    """Render the Overview and System Architecture page."""
    render_disclaimer()
    render_page_header(
        title="Overview",
        subtitle="Explore datasets, train models, evaluate diagnostics, and generate predictions.",
        badge="ML Studio v1.0",
    )

    df_current = st.session_state.get("df")
    num_samples = len(df_current) if df_current is not None else 0
    features_current = st.session_state.get("feature_columns") or []
    num_features = len(features_current)
    tr = st.session_state.get("training_result")
    trained_model_name = tr.display_name if tr else "None"
    accuracy_text = f"{tr.metrics.accuracy * 100:.1f}%" if tr else "—"

    # Overview KPI Scorecards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        render_kpi_card(
            label="Supported Algorithms",
            value=str(len(MODEL_REGISTRY)),
            subtext="GNB, LogReg, LinearSVC, RF",
        )
    with kpi_col2:
        render_kpi_card(
            label="Active Dataset Samples",
            value=f"{num_samples:,}" if num_samples else "—",
            subtext="Breast Cancer Wisconsin"
            if st.session_state.get("dataset_choice") == "builtin"
            else "Custom Upload",
        )
    with kpi_col3:
        render_kpi_card(
            label="Active Features",
            value=str(num_features) if num_features else "—",
            subtext="Normalized numeric dimensions",
        )
    with kpi_col4:
        render_kpi_card(
            label="Active Trained Model",
            value=trained_model_name
            if len(trained_model_name) <= 15
            else f"{trained_model_name[:12]}...",
            subtext=f"Accuracy: {accuracy_text}" if tr else "Pending training",
            status_color=COLOR_SUCCESS if tr else None,
        )

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Interactive Data & Session Visualizations
    # -----------------------------------------------------------------------
    target_current = st.session_state.get("target_column")
    if df_current is not None and target_current in df_current.columns:
        render_section_title("Active Dataset Overview & Balance", badge="Live Data")
        chart_col1, chart_col2 = st.columns([1.1, 0.9])
        with chart_col1:
            val_counts = df_current[target_current].value_counts()
            fig_overview_dist = plot_class_distribution(
                val_counts.to_dict(),
                title=f"Target Distribution: '{target_current}'",
                show_percentages=True,
            )
            st.plotly_chart(fig_overview_dist, use_container_width=True)

            # Class imbalance alert if applicable
            total_records = len(df_current)
            min_class_ratio = (val_counts.min() / total_records) if total_records > 0 else 0
            if min_class_ratio < 0.35 and len(val_counts) == 2:
                st.caption(
                    f"⚠️ **Moderate Class Imbalance Detected**: Minority class represents {min_class_ratio * 100:.1f}% of samples. "
                    "Evaluation should prioritize Macro-F1 and Sensitivity over raw Accuracy."
                )

        with chart_col2:
            fig_overview_profile = plot_dataset_profile(df_current)
            st.plotly_chart(fig_overview_profile, use_container_width=True)

    # Model Evaluation / Comparison Session Status
    comp_res = st.session_state.get("comparison_result")
    if comp_res is not None:
        render_section_title("Multi-Model Benchmark Summary", badge="Session Benchmark")
        fig_overview_comp = plot_model_comparison(comp_res.comparison_table)
        st.plotly_chart(fig_overview_comp, use_container_width=True)
    elif tr is not None:
        render_section_title(f"Latest Model Evaluation: {tr.display_name}", badge="Fitted Artifact")
        render_metrics_cards(tr.metrics)
    else:
        render_section_title("Session Evaluation Status")
        render_empty_state_box(
            title="No Models Evaluated in Current Session",
            message="Train an individual model in 'Train & Evaluate' or benchmark all algorithms in 'Compare Models' to view live diagnostic charts here.",
            instruction="→ Navigate to Train & Evaluate to fit your first pipeline",
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        render_section_title("Quick Actions")
        qa_col1, qa_col2 = st.columns(2)
        with qa_col1:
            st.markdown(
                """
                <div class="studio-card" style="padding: 1rem;">
                    <div style="font-weight: 600; font-size: 0.95rem; color: #172033; margin-bottom: 0.25rem;">
                        1. Inspect Dataset
                    </div>
                    <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 0.75rem;">
                        Examine class balance, feature distributions, correlations, and PCA projections.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="studio-card" style="padding: 1rem;">
                    <div style="font-weight: 600; font-size: 0.95rem; color: #172033; margin-bottom: 0.25rem;">
                        2. Train & Evaluate
                    </div>
                    <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 0.75rem;">
                        Fit isolated pipelines and inspect confusion matrices, ROC/PR curves, and coefficients.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with qa_col2:
            st.markdown(
                """
                <div class="studio-card" style="padding: 1rem;">
                    <div style="font-weight: 600; font-size: 0.95rem; color: #172033; margin-bottom: 0.25rem;">
                        3. Compare Models
                    </div>
                    <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 0.75rem;">
                        Benchmark all four classifiers on an identical stratified split with comparative ROC/PR plots.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="studio-card" style="padding: 1rem;">
                    <div style="font-weight: 600; font-size: 0.95rem; color: #172033; margin-bottom: 0.25rem;">
                        4. Run Inference
                    </div>
                    <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 0.75rem;">
                        Batch score CSV files with download options or test single manual records.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        render_section_title("Engineering Highlights")
        st.markdown(
            """
            <div class="studio-card">
                <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.875rem; color: #334155; line-height: 1.6;">
                    <li><strong>Guaranteed Leak-Free:</strong> Stratified partition is frozen before scalers and imputers are fit.</li>
                    <li><strong>High-Sensitivity Focus:</strong> Explicit tracking of Malignant class recall to minimize False Negatives.</li>
                    <li><strong>Dual Execution Mode:</strong> Fast local execution with FastAPI microservice compatibility.</li>
                    <li><strong>Sub-50ms Navigation:</strong> Plotly figures and datasets are cached in memory.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        render_section_title("Pipeline Architecture")
        st.markdown(
            """
            <div class="studio-card" style="padding: 1rem 1.25rem;">
                <div style="font-family: monospace; font-size: 0.8rem; color: #334155; line-height: 1.5; white-space: pre;">
[Dataset Source] (Built-in or Uploaded CSV)
       │
       ▼
[Data Validation Guardrails]
       │
       ▼
[Stratified Train/Test Split]
       │
       ▼
[Isolated Preprocessing Pipeline] (Scale + Impute)
       │
       ▼
[Model Fitting & Registry] (GNB, LogReg, SVM, RF)
       │
       ▼
[Evaluation Diagnostics] (CM, ROC, PR, Importances)
       │
       ▼
[Inference Engine] (Batch CSV & Manual Form)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_section_title("Roadmap Progression")
        st.markdown(
            """
            <div class="studio-card" style="font-size: 0.85rem; color: #334155;">
                <div style="margin-bottom: 0.4rem;"><strong>Stage 1:</strong> NumPy scratch derivations of Linear & Logistic Regression</div>
                <div style="margin-bottom: 0.4rem;"><strong>Stage 2:</strong> Algorithmic benchmarks on Breast Cancer Wisconsin</div>
                <div style="margin-bottom: 0.4rem;"><strong>Stage 3:</strong> Streamlit interactive application & diagnostics</div>
                <div><strong>Stage 4:</strong> FastAPI REST API, test suite, and cloud production architecture</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
