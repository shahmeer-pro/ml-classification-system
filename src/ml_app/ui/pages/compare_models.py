"""Multi-Model Benchmark Comparison page."""

import pandas as pd
import streamlit as st

from ml_app.exceptions import MLAppException
from ml_app.models.registry import MODEL_REGISTRY
from ml_app.ui.api_client import ClassificationApiClient
from ml_app.ui.components import plot_model_comparison, render_disclaimer


def page_compare_models() -> None:
    """Render the Multi-Model Comparison page."""
    render_disclaimer()
    st.title("Multi-Model Benchmark Comparison")
    st.markdown(
        """
        Compare **Gaussian Naive Bayes**, **Logistic Regression**, **Linear SVM (LinearSVC)**, and **Random Forest**
        on the **exact same train/test split**. This eliminates partition variance and provides an honest,
        apples-to-apples comparison.
        """
    )

    df_current = st.session_state.get("df")
    target_current = st.session_state.get("target_column")
    features_current = st.session_state.get("feature_columns")

    if df_current is None or not features_current:
        st.warning("Please choose or upload a dataset first in Dataset Explorer.")
        st.stop()

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        test_split = st.slider("Comparison Test Split:", 0.10, 0.40, 0.20, 0.05, key="comp_split")
    with comp_col2:
        comp_seed = st.number_input("Comparison Random Seed:", 0, 9999, 42, key="comp_seed")

    models_to_compare = st.multiselect(
        "Select Algorithms to Compare:",
        ["gaussian_nb", "logistic_regression", "linear_svm", "random_forest"],
        default=["gaussian_nb", "logistic_regression", "linear_svm", "random_forest"],
        format_func=lambda k: MODEL_REGISTRY[k].display_name,
    )

    api_client = ClassificationApiClient()

    if st.button("⚡ Run Multi-Model Comparison", type="primary"):
        with st.spinner("Training all selected classifiers on identical split..."):
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
                st.success("Comparison completed successfully!")

            except MLAppException as exc:
                st.error(f"Comparison error: {exc.message}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

    if st.session_state.get("comparison_result") is not None:
        comp = st.session_state.comparison_result
        st.write("---")
        st.subheader("Comparative Results Table")

        df_table = pd.DataFrame(comp.comparison_table)
        st.dataframe(df_table, use_container_width=True)

        st.subheader("Performance Metrics Visualization")
        fig_comp = st.session_state.get("fig_comp")
        if fig_comp is None:
            fig_comp = plot_model_comparison(comp.comparison_table)
            st.session_state.fig_comp = fig_comp
        st.plotly_chart(fig_comp, use_container_width=True)

        st.info(
            "💡 **Analysis Note**: Accuracy alone should not be the sole decision metric, especially for medical "
            "classification. For malignant tumor detection, **Malignant Recall (Sensitivity)** is vital to minimize "
            "dangerous False Negatives."
        )
