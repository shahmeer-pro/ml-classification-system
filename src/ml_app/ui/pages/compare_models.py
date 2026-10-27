"""Multi-Model Benchmark Comparison page."""

import pandas as pd
import streamlit as st

from ml_app.exceptions import MLAppException
from ml_app.models.registry import MODEL_REGISTRY
from ml_app.ui.api_client import ClassificationApiClient
from ml_app.ui.components import plot_confusion_matrix, plot_model_comparison, render_disclaimer
from ml_app.ui.theme import (
    render_empty_state_box,
    render_page_header,
    render_section_title,
)
from ml_app.ui.visualizations import (
    plot_metric_ranking,
    plot_multi_model_pr,
    plot_multi_model_roc,
)


def page_compare_models() -> None:
    """Render the Multi-Model Comparison page."""
    render_disclaimer()
    render_page_header(
        title="Compare Models",
        subtitle="Benchmark multiple classifiers on an identical stratified split to eliminate variance.",
        badge="Multi-Model Benchmark",
    )

    df_current = st.session_state.get("df")
    target_current = st.session_state.get("target_column")
    features_current = st.session_state.get("feature_columns")

    if df_current is not None:
        df_current.columns = [str(c) for c in df_current.columns]
    if target_current is not None:
        target_current = str(target_current)
    if features_current is not None:
        features_current = [str(c) for c in features_current]

    if df_current is None or not features_current:
        render_empty_state_box(
            title="No Active Dataset",
            message="Please select or upload a dataset with features in the Dataset Explorer before comparing algorithms.",
            instruction="→ Go to Dataset Explorer",
        )
        st.stop()

    render_section_title("1. Benchmark Split & Reproducibility")
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        test_split = st.slider("Comparison Test Split:", 0.10, 0.40, 0.20, 0.05, key="comp_split")
    with comp_col2:
        comp_seed = st.number_input("Comparison Random Seed:", 0, 9999, 42, key="comp_seed")

    render_section_title("2. Algorithms Under Evaluation")
    models_to_compare = st.multiselect(
        "Select Algorithms to Compare:",
        ["gaussian_nb", "logistic_regression", "linear_svm", "random_forest"],
        default=["gaussian_nb", "logistic_regression", "linear_svm", "random_forest"],
        format_func=lambda k: MODEL_REGISTRY[k].display_name,
        label_visibility="collapsed",
    )

    api_client = ClassificationApiClient()

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    if st.button("Run Multi-Model Comparison", type="primary"):
        with st.spinner("Fitting all selected classifiers on the shared test partition..."):
            try:
                is_builtin = st.session_state.get("dataset_choice") == "builtin"
                meta_dict = st.session_state.get("meta") or {}
                class_mapping = meta_dict.get("class_mapping") if is_builtin else None
                pos_class = meta_dict.get("positive_class") if is_builtin else None
                pos_name = meta_dict.get("positive_class_name") if is_builtin else None

                comp_res = api_client.compare_models(
                    df=df_current,
                    target_column=target_current,
                    model_ids=models_to_compare,
                    feature_columns=features_current,
                    test_size=test_split,
                    random_state=int(comp_seed),
                    class_mapping=class_mapping,
                    positive_class=pos_class,
                    positive_class_name=pos_name,
                )
                st.session_state.comparison_result = comp_res
                # Pre-generate comparison figure
                st.session_state.fig_comp = plot_model_comparison(comp_res.comparison_table)
                st.success("Comparison completed successfully across all selected algorithms!")

            except MLAppException as exc:
                st.error(f"Comparison error: {exc.message}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

    if st.session_state.get("comparison_result") is not None:
        comp = st.session_state.comparison_result
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        render_section_title("Comparative Benchmark Metrics", badge="Apples-to-Apples")

        df_table = pd.DataFrame(comp.comparison_table)
        st.dataframe(df_table, use_container_width=True)

        render_section_title("Comparative Visualizations", badge="Interactive Diagnostics")

        comp_tab1, comp_tab2, comp_tab3, comp_tab4 = st.tabs(
            [
                "1. Multi-Metric Benchmark",
                "2. Metric Ranking",
                "3. Overlay ROC & PR Curves",
                "4. Confusion Matrices (All Models)",
            ]
        )

        with comp_tab1:
            fig_comp = st.session_state.get("fig_comp")
            if fig_comp is None:
                fig_comp = plot_model_comparison(comp.comparison_table)
                st.session_state.fig_comp = fig_comp
            st.plotly_chart(fig_comp, use_container_width=True)

        with comp_tab2:
            r_col1, r_col2 = st.columns([1, 2])
            with r_col1:
                available_rank_metrics = [
                    c
                    for c in [
                        "Macro F1",
                        "Accuracy",
                        "Macro Recall",
                        "Macro Precision",
                        "Weighted F1",
                    ]
                    if c in df_table.columns
                ]
                selected_rank_metric = st.selectbox(
                    "Rank Models by Metric:",
                    available_rank_metrics,
                    index=0,
                    key="rank_metric_select",
                )
            with r_col2:
                fig_rank = plot_metric_ranking(
                    comp.comparison_table, metric_name=selected_rank_metric
                )
                st.plotly_chart(fig_rank, use_container_width=True)

        with comp_tab3:
            curve_col1, curve_col2 = st.columns(2)
            with curve_col1:
                render_section_title("Multi-Model ROC Curves")
                fig_multi_roc = plot_multi_model_roc(comp.results)
                st.plotly_chart(fig_multi_roc, use_container_width=True)
            with curve_col2:
                render_section_title("Multi-Model Precision-Recall Curves")
                fig_multi_pr = plot_multi_model_pr(comp.results)
                st.plotly_chart(fig_multi_pr, use_container_width=True)

        with comp_tab4:
            render_section_title("Confusion Matrices Across Evaluated Classifiers")
            cm_tabs = st.tabs([tr.display_name for tr in comp.results])
            for idx, tr in enumerate(comp.results):
                with cm_tabs[idx]:
                    cm_fig = plot_confusion_matrix(
                        tr.metrics.confusion_matrix,
                        tr.metrics.labels,
                        model_name=tr.display_name,
                    )
                    st.plotly_chart(cm_fig, use_container_width=True)
                    st.caption(
                        f"Accuracy: **{tr.metrics.accuracy * 100:.2f}%** | "
                        f"Macro F1: **{tr.metrics.f1_macro * 100:.2f}%** | "
                        f"Macro Recall: **{tr.metrics.recall_macro * 100:.2f}%**"
                    )

        st.markdown(
            """
            <div class="studio-card" style="background-color: #F8FAFC; border-left: 4px solid #2563EB;">
                <div style="font-weight: 600; font-size: 0.9rem; color: #172033; margin-bottom: 0.25rem;">
                    Clinical & Engineering Decision Context
                </div>
                <div style="font-size: 0.825rem; color: #64748B;">
                    Accuracy alone must not be the sole criteria for model selection, especially on imbalanced or medical data.
                    In oncology diagnosis, <strong>Malignant Recall (Sensitivity)</strong> is paramount to avoid missing cancerous cases (False Negatives),
                    even if it slightly compromises overall accuracy.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        render_empty_state_box(
            title="Comparison Not Yet Executed",
            message="Select two or more algorithms above and click 'Run Multi-Model Comparison' to evaluate side-by-side performance.",
            instruction="All algorithms evaluate against the identical test partition.",
        )
