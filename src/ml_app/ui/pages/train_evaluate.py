"""Model Training & Evaluation page."""

from typing import Any

import numpy as np
import plotly.graph_objects as go
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
from ml_app.ui.theme import (
    apply_studio_chart_theme,
    render_empty_state_box,
    render_page_header,
    render_section_title,
)
from ml_app.ui.visualizations import plot_feature_coefficients


def page_train_evaluate() -> None:
    """Render the Model Training and Evaluation page."""
    render_disclaimer()
    render_page_header(
        title="Train & Evaluate",
        subtitle="Configure preprocessing, select a classifier, and evaluate diagnostics.",
        badge="Experimentation",
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
            title="No Active Dataset Available",
            message="Please navigate to Dataset Explorer to select or upload a dataset before configuring training.",
            instruction="→ Go to Dataset Explorer",
        )
        st.stop()

    render_section_title("1. Partition & Algorithm Configuration")
    cfg_col1, cfg_col2, cfg_col3 = st.columns([1, 1, 1.4])

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
    render_section_title("2. Hyperparameter Settings")
    meta_info = MODEL_REGISTRY[selected_model_id]
    st.markdown(
        f"""
        <div style="font-size: 0.8rem; color: #64748B; margin-top: -0.3rem; margin-bottom: 0.6rem;">
            {meta_info.description}
        </div>
        """,
        unsafe_allow_html=True,
    )

    param_inputs: dict[str, Any] = {}
    h_cols = st.columns(len(meta_info.parameters) or 1)
    for idx, (p_name, p_def) in enumerate(meta_info.parameters.items()):
        with h_cols[idx % len(h_cols)]:
            if p_def.options:
                param_inputs[p_name] = st.selectbox(
                    f"{p_name}:",
                    p_def.options,
                    index=p_def.options.index(p_def.default),
                    help=p_def.description,
                )
            elif p_def.param_type == "float":
                param_inputs[p_name] = st.number_input(
                    f"{p_name}:",
                    value=float(p_def.default),
                    min_value=float(p_def.min_value) if p_def.min_value is not None else None,
                    max_value=float(p_def.max_value) if p_def.max_value is not None else None,
                    format="%.4f" if p_def.default < 0.01 else "%.2f",
                    help=p_def.description,
                )
            elif p_def.param_type == "int":
                param_inputs[p_name] = st.number_input(
                    f"{p_name}:",
                    value=int(p_def.default),
                    min_value=int(p_def.min_value) if p_def.min_value is not None else None,
                    max_value=int(p_def.max_value) if p_def.max_value is not None else None,
                    help=p_def.description,
                )

    api_client = ClassificationApiClient()

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    if st.button("Train Model & Run Evaluation", type="primary"):
        with st.spinner(
            "Partitioning data, fitting isolated pipeline, and calculating diagnostics..."
        ):
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
                    f"Successfully fitted **{train_res.display_name}**! In-memory Artifact ID: `{train_res.artifact_id[:8]}...`"
                )

            except MLAppException as exc:
                st.error(f"Training Failed: {exc.message}")
            except Exception as exc:
                st.error(f"Unexpected Training Error: {exc}")

    # Display Metrics if training result exists
    if st.session_state.get("training_result") is not None:
        tr = st.session_state.training_result
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        render_section_title(f"Evaluation Diagnostics: {tr.display_name}", badge="Fitted Model")

        render_metrics_cards(tr.metrics)

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

        diag_tab1, diag_tab2, diag_tab3 = st.tabs(
            [
                "1. Confusion Matrix & Prediction Balance",
                "2. ROC & Precision-Recall Curves",
                "3. Feature Coefficients & Importances",
            ]
        )

        with diag_tab1:
            diag_col1, diag_col2 = st.columns(2)
            with diag_col1:
                render_section_title("Confusion Matrix Heatmap")
                fig_cm = st.session_state.get("fig_cm")
                if fig_cm is None:
                    fig_cm = plot_confusion_matrix(
                        tr.metrics.confusion_matrix, tr.metrics.labels, model_name=tr.display_name
                    )
                    st.session_state.fig_cm = fig_cm
                st.plotly_chart(fig_cm, use_container_width=True)

            with diag_col2:
                render_section_title("Test Set Prediction Distribution")
                cm_arr = np.array(tr.metrics.confusion_matrix)
                actual_counts = cm_arr.sum(axis=1)
                predicted_counts = cm_arr.sum(axis=0)

                fig_pred_dist = go.Figure()
                fig_pred_dist.add_trace(
                    go.Bar(
                        name="Ground Truth (y_test)",
                        x=tr.metrics.labels,
                        y=actual_counts,
                        text=actual_counts,
                        textposition="outside",
                        cliponaxis=False,
                        marker=dict(color="#64748B"),
                        hovertemplate="<b>Actual %{x}:</b> %{y:,}<extra></extra>",
                    )
                )
                fig_pred_dist.add_trace(
                    go.Bar(
                        name=f"Predicted ({tr.display_name})",
                        x=tr.metrics.labels,
                        y=predicted_counts,
                        text=predicted_counts,
                        textposition="outside",
                        cliponaxis=False,
                        marker=dict(color="#4F46E5"),
                        hovertemplate="<b>Predicted %{x}:</b> %{y:,}<extra></extra>",
                    )
                )
                apply_studio_chart_theme(
                    fig_pred_dist,
                    title="Actual vs Predicted Test Sample Counts",
                    height=420,
                    show_legend=True,
                )
                fig_pred_dist.update_layout(
                    barmode="group",
                    xaxis_title="Class Label",
                    yaxis_title="Sample Count",
                )
                st.plotly_chart(fig_pred_dist, use_container_width=True)

        with diag_tab2:
            roc_col, pr_col = st.columns(2)
            with roc_col:
                render_section_title("ROC Diagnostic Curve")
                fig_roc = st.session_state.get("fig_roc")
                if fig_roc is not None:
                    st.plotly_chart(fig_roc, use_container_width=True)
                else:
                    st.info(
                        f"ROC curves require continuous decision scores or class probabilities. "
                        f"{tr.display_name} does not expose compatible probability outputs."
                    )
            with pr_col:
                render_section_title("Precision-Recall Curve")
                fig_pr = st.session_state.get("fig_pr")
                if fig_pr is not None:
                    st.plotly_chart(fig_pr, use_container_width=True)
                else:
                    st.info(
                        "Precision-Recall curve unavailable for current estimator configuration."
                    )

        with diag_tab3:
            render_section_title(f"Model Feature Weights: {tr.display_name}")
            try:
                from ml_app.services.training_service import training_service

                artifact = training_service.get_artifact(tr.artifact_id)
                pipeline = artifact.pipeline
                classifier = pipeline.named_steps.get("classifier")
                preprocessor = pipeline.named_steps.get("preprocessor")

                if hasattr(preprocessor, "get_feature_names_out"):
                    try:
                        feature_names = list(preprocessor.get_feature_names_out())
                        feature_names = [
                            f.replace("numeric__", "").replace("remainder__", "")
                            for f in feature_names
                        ]
                    except Exception:
                        feature_names = artifact.feature_names
                else:
                    feature_names = artifact.feature_names

                if hasattr(classifier, "coef_"):
                    coefs = classifier.coef_
                    st.markdown(
                        """
                        <div style="font-size: 0.825rem; color: #64748B; margin-bottom: 0.5rem;">
                            <strong>Interpretation</strong>: Coefficients represent the change in log-odds (Logistic Regression) or hyperplane normal weights (Linear SVM) per 1-SD feature increase.
                            Positive values increase class probability; negative values decrease it.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    fig_coef = plot_feature_coefficients(
                        feature_names=feature_names,
                        coefficients=coefs[0],
                        title=f"Standardized Coefficients ({tr.display_name})",
                        is_importance=False,
                    )
                    st.plotly_chart(fig_coef, use_container_width=True)

                elif hasattr(classifier, "feature_importances_"):
                    importances = classifier.feature_importances_
                    st.markdown(
                        """
                        <div style="font-size: 0.825rem; color: #64748B; margin-bottom: 0.5rem;">
                            <strong>Interpretation</strong>: Gini impurity reduction / Mean Decrease in Impurity (MDI). Higher values indicate features that create cleaner decision splits across trees.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    fig_imp = plot_feature_coefficients(
                        feature_names=feature_names,
                        coefficients=importances,
                        title="Random Forest Feature Importance (MDI)",
                        is_importance=True,
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

                elif tr.model_id == "gaussian_nb":
                    st.markdown(
                        r"""
                        <div class="studio-card" style="margin-top: 0.5rem;">
                            <div style="font-weight: 600; color: #172033; margin-bottom: 0.25rem;">
                                Gaussian Naive Bayes Parameter Representation
                            </div>
                            <div style="font-size: 0.85rem; color: #64748B;">
                                Gaussian Naive Bayes assumes independent conditional Gaussian distributions for each feature ($X_j \sim \mathcal{N}(\mu_{c,j}, \sigma_{c,j}^2)$).
                                Unlike linear models or tree ensembles, it does not assign directional linear weights or Gini split importances.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("Feature importance extraction is not supported for this estimator.")
            except Exception as exc:
                st.info(f"Could not extract feature coefficients: {exc}")

        with st.expander("Detailed Scikit-Learn Classification Report"):
            st.code(tr.metrics.classification_report_str, language="text")

    else:
        render_empty_state_box(
            title="Model Not Yet Trained",
            message="Configure your partition and algorithm settings above, then click 'Train Model & Run Evaluation' to inspect metrics.",
            instruction="Results will appear here with zero data leakage.",
        )
