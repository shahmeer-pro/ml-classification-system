"""Model Training & Evaluation page."""

from typing import Any

import streamlit as st

from ml_app.exceptions import MLAppException
from ml_app.models.registry import MODEL_REGISTRY
from ml_app.ui.api_client import ClassificationApiClient
from ml_app.ui.components import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    render_disclaimer,
    render_metrics_cards,
)


def page_train_evaluate() -> None:
    """Render the Model Training and Evaluation page."""
    render_disclaimer()
    st.title("Model Training & Evaluation")

    df_current = st.session_state.get("df")
    target_current = st.session_state.get("target_column")
    features_current = st.session_state.get("feature_columns")

    if df_current is None or not features_current:
        st.warning("Please select or upload a dataset with features in the Dataset Explorer first.")
        st.stop()

    st.subheader("1. Configure Training Parameters")
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

    with cfg_col1:
        test_size = st.slider(
            "Test Partition Ratio:", min_value=0.10, max_value=0.40, value=0.20, step=0.05
        )
    with cfg_col2:
        random_seed = st.number_input(
            "Random Seed (Reproducibility):", min_value=0, max_value=9999, value=42
        )
    with cfg_col3:
        model_options = {
            "logistic_regression": "Logistic Regression",
            "gaussian_nb": "Gaussian Naive Bayes",
            "linear_svm": "Linear Support Vector Machine (LinearSVC)",
            "random_forest": "Random Forest Classifier",
        }
        selected_model_id = st.selectbox(
            "Select Classifier:",
            list(model_options.keys()),
            format_func=lambda k: model_options[k],
        )

    # Dynamic Hyperparameters
    st.subheader("2. Hyperparameter Settings")
    meta_info = MODEL_REGISTRY[selected_model_id]
    st.caption(meta_info.description)

    param_inputs: dict[str, Any] = {}
    h_cols = st.columns(len(meta_info.parameters) or 1)
    for idx, (p_name, p_def) in enumerate(meta_info.parameters.items()):
        with h_cols[idx % len(h_cols)]:
            if p_def.options:
                param_inputs[p_name] = st.selectbox(
                    f"{p_name} ({p_def.description[:30]}...)",
                    p_def.options,
                    index=p_def.options.index(p_def.default),
                )
            elif p_def.param_type == "float":
                param_inputs[p_name] = st.number_input(
                    f"{p_name}",
                    value=float(p_def.default),
                    min_value=float(p_def.min_value) if p_def.min_value is not None else None,
                    max_value=float(p_def.max_value) if p_def.max_value is not None else None,
                    format="%.4f" if p_def.default < 0.01 else "%.2f",
                )
            elif p_def.param_type == "int":
                param_inputs[p_name] = st.number_input(
                    f"{p_name}",
                    value=int(p_def.default),
                    min_value=int(p_def.min_value) if p_def.min_value is not None else None,
                    max_value=int(p_def.max_value) if p_def.max_value is not None else None,
                )

    api_client = ClassificationApiClient()

    if st.button("🚀 Train Model & Run Evaluation", type="primary"):
        with st.spinner("Partitioning data, fitting pipeline, and evaluating..."):
            try:
                is_builtin = st.session_state.get("dataset_choice") == "builtin"
                meta_dict = st.session_state.get("meta") or {}
                class_mapping = meta_dict.get("class_mapping") if is_builtin else None
                pos_class = meta_dict.get("positive_class") if is_builtin else None
                pos_name = meta_dict.get("positive_class_name") if is_builtin else None

                train_res = api_client.train_model(
                    df=df_current,
                    target_column=target_current,
                    model_id=selected_model_id,
                    feature_columns=features_current,
                    hyperparameters=param_inputs,
                    test_size=test_size,
                    random_state=int(random_seed),
                    class_mapping=class_mapping,
                    positive_class=pos_class,
                    positive_class_name=pos_name,
                )

                st.session_state.training_result = train_res
                st.session_state.trained_artifact_id = train_res.artifact_id

                # Pre-generate and cache Plotly figures directly in session_state
                st.session_state.fig_cm = plot_confusion_matrix(
                    train_res.metrics.confusion_matrix, train_res.metrics.labels
                )
                if train_res.metrics.roc_curve:
                    st.session_state.fig_roc = plot_roc_curve(
                        train_res.metrics.roc_curve, model_name=train_res.display_name
                    )
                else:
                    st.session_state.fig_roc = None

                if train_res.metrics.pr_curve:
                    st.session_state.fig_pr = plot_pr_curve(
                        train_res.metrics.pr_curve, model_name=train_res.display_name
                    )
                else:
                    st.session_state.fig_pr = None

                st.success(
                    f"Trained {train_res.display_name}! Stored in-memory ID: `{train_res.artifact_id}`"
                )

            except MLAppException as exc:
                st.error(f"Training Failed: {exc.message}")
            except Exception as exc:
                st.error(f"Unexpected Training Error: {exc}")

    # Display Metrics if training result exists (instantaneous render)
    if st.session_state.get("training_result") is not None:
        tr = st.session_state.training_result
        st.write("---")
        st.header(f"Evaluation Diagnostics: {tr.display_name}")

        render_metrics_cards(tr.metrics)

        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            fig_cm = st.session_state.get("fig_cm")
            if fig_cm is None:
                fig_cm = plot_confusion_matrix(tr.metrics.confusion_matrix, tr.metrics.labels)
                st.session_state.fig_cm = fig_cm
            st.plotly_chart(fig_cm, use_container_width=True)

        with diag_col2:
            fig_roc = st.session_state.get("fig_roc")
            if fig_roc is not None:
                st.plotly_chart(fig_roc, use_container_width=True)
            elif tr.metrics.pr_curve:
                fig_pr = st.session_state.get("fig_pr")
                if fig_pr is None:
                    fig_pr = plot_pr_curve(tr.metrics.pr_curve, model_name=tr.display_name)
                    st.session_state.fig_pr = fig_pr
                st.plotly_chart(fig_pr, use_container_width=True)
            else:
                st.info(
                    "ROC / PR Curves are only applicable when predicted probabilities or continuous "
                    "decision scores are available."
                )

        with st.expander("Detailed Scikit-Learn Classification Report"):
            st.text(tr.metrics.classification_report_str)
