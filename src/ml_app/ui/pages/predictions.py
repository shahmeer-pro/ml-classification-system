"""Predictions & Inference page."""

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml_app.exceptions import MLAppException
from ml_app.services.training_service import training_service
from ml_app.ui.api_client import ClassificationApiClient
from ml_app.ui.components import render_disclaimer
from ml_app.ui.theme import (
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    apply_studio_chart_theme,
    render_empty_state_box,
    render_kpi_card,
    render_page_header,
    render_section_title,
)
from ml_app.ui.visualizations import (
    plot_actual_vs_predicted_batch,
    plot_batch_prediction_distribution,
    plot_batch_probabilities,
)


def page_predictions() -> None:
    """Render the Predictions and Inference page."""
    render_disclaimer()
    render_page_header(
        title="Predictions & Inference",
        subtitle="Generate predictions using an active fitted model pipeline.",
        badge="Inference Engine",
    )

    artifact_id = st.session_state.get("trained_artifact_id")

    if artifact_id is None:
        render_empty_state_box(
            title="No Trained Pipeline Available",
            message="No fitted model exists in the current session memory. Please navigate to Train & Evaluate and train a model first.",
            instruction="→ Go to Train & Evaluate",
        )
        st.stop()

    tr = st.session_state.get("training_result")
    model_name = tr.display_name if tr else "Fitted Classifier"

    st.markdown(
        f"""
        <div class="studio-card" style="padding: 0.9rem 1.25rem; margin-bottom: 1.25rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                <div>
                    <span style="font-weight: 600; color: #172033; font-size: 0.95rem;">Active Pipeline:</span>
                    <strong style="color: {COLOR_PRIMARY};"> {model_name}</strong>
                    <span style="font-size: 0.8rem; color: #64748B; margin-left: 0.5rem;">ID: <code>{artifact_id[:8]}...</code></span>
                </div>
                <span class="studio-badge studio-badge-success">Pipeline Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_title("Inference Method")
    pred_mode = st.radio(
        "Choose Prediction Method:",
        ["Batch CSV Upload", "Manual Single Record Entry"],
        horizontal=True,
        label_visibility="collapsed",
    )

    api_client = ClassificationApiClient()

    if pred_mode == "Batch CSV Upload":
        st.markdown(
            """
            <div style="font-size: 0.825rem; color: #64748B; margin-top: 0.25rem; margin-bottom: 0.5rem;">
                Upload a CSV containing all feature columns used during training. The target column is optional.
            </div>
            """,
            unsafe_allow_html=True,
        )

        pred_file = st.file_uploader(
            "Upload CSV for inference:", type=["csv"], key="pred_csv", label_visibility="collapsed"
        )
        if pred_file is not None:
            try:
                pred_df = pd.read_csv(io.BytesIO(pred_file.getvalue()))
                st.info(
                    f"Loaded **{len(pred_df):,}** records from `{pred_file.name}` for batch inference."
                )

                if st.button("Generate Batch Predictions", type="primary"):
                    with st.spinner("Executing pipeline transform and scoring..."):
                        pred_res = api_client.predict(artifact_id=artifact_id, input_data=pred_df)
                        st.session_state.latest_batch_prediction = pred_res
                        st.session_state.latest_batch_df = pred_df

                if st.session_state.get("latest_batch_prediction") is not None:
                    pred_res = st.session_state.latest_batch_prediction
                    saved_pred_df = st.session_state.get("latest_batch_df", pred_df)
                    res_df = pred_res.predictions_dataframe

                    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                    render_section_title(
                        "Scoring Results & Diagnostics", badge=f"{len(res_df):,} Predictions"
                    )

                    p_col1, p_col2 = st.columns(2)
                    with p_col1:
                        fig_bdist = plot_batch_prediction_distribution(
                            pred_res.predicted_classes,
                            class_labels=pred_res.probability_class_labels,
                        )
                        st.plotly_chart(fig_bdist, use_container_width=True)

                    with p_col2:
                        if pred_res.has_probabilities and pred_res.probabilities:
                            fig_bprobs = plot_batch_probabilities(
                                pred_res.probabilities,
                                class_labels=pred_res.probability_class_labels or ["0", "1"],
                            )
                            st.plotly_chart(fig_bprobs, use_container_width=True)
                        elif pred_res.has_decision_scores and pred_res.decision_scores:
                            fig_scores = go.Figure(
                                data=[
                                    go.Histogram(
                                        x=pred_res.decision_scores,
                                        nbinsx=30,
                                        marker=dict(color="#4F46E5"),
                                    )
                                ]
                            )
                            apply_studio_chart_theme(
                                fig_scores,
                                title="Decision Score Distribution (Hyperplane Distance)",
                                height=380,
                                show_legend=False,
                            )
                            fig_scores.update_layout(
                                xaxis_title="Decision Score", yaxis_title="Count"
                            )
                            st.plotly_chart(fig_scores, use_container_width=True)
                        else:
                            st.info(
                                "Continuous probability distributions are not exposed by the active estimator."
                            )

                    # If uploaded CSV contains true target column, offer actual vs predicted comparison
                    target_candidate = st.session_state.get("target_column")
                    if target_candidate and target_candidate in saved_pred_df.columns:
                        render_section_title("Batch Target Verification (Actual vs Predicted)")
                        fig_avsp = plot_actual_vs_predicted_batch(
                            actual_labels=saved_pred_df[target_candidate],
                            predicted_labels=pred_res.predicted_classes,
                            class_labels=pred_res.probability_class_labels,
                        )
                        st.plotly_chart(fig_avsp, use_container_width=True)

                    # Display preview
                    st.write("**Predictions Table Preview (First 20 Records):**")
                    st.dataframe(res_df.head(20), use_container_width=True)

                    if len(res_df) > 20:
                        with st.expander(f"View Full Output Table ({len(res_df):,} rows)"):
                            st.dataframe(res_df, use_container_width=True)

                    # CSV Download button
                    csv_data = res_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Scored Predictions (CSV)",
                        data=csv_data,
                        file_name="predictions_output.csv",
                        mime="text/csv",
                        type="secondary",
                    )
            except MLAppException as exc:
                st.error(f"Inference error: {exc.message}")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    else:
        render_section_title("Manual Record Entry")
        st.markdown(
            """
            <div style="font-size: 0.825rem; color: #64748B; margin-bottom: 0.75rem;">
                Adjust continuous feature inputs below to simulate an isolated clinical record. Defaults reflect dataset medians.
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            artifact = training_service.get_artifact(artifact_id)
            feat_list = artifact.feature_names
        except Exception:
            feat_list = []

        if not feat_list:
            st.error("Could not retrieve feature schema for the active model artifact.")
            st.stop()

        with st.form("manual_prediction_form"):
            input_values: dict[str, float] = {}
            cols = st.columns(3)
            df_cur = st.session_state.get("df")
            for idx, f_name in enumerate(feat_list):
                with cols[idx % 3]:
                    default_val = (
                        float(df_cur[f_name].median())
                        if df_cur is not None and f_name in df_cur.columns
                        else 0.0
                    )
                    input_values[f_name] = st.number_input(
                        f"{f_name}:",
                        value=round(default_val, 4),
                        format="%.4f",
                    )

            submitted = st.form_submit_button("Predict Single Record", type="primary")

        if submitted:
            try:
                single_df = pd.DataFrame([input_values])
                pred_res = api_client.predict(artifact_id=artifact_id, input_data=single_df)

                pred_class = pred_res.predicted_classes[0]
                pred_label = (
                    pred_res.predicted_class_names[0]
                    if pred_res.predicted_class_names
                    else str(pred_class)
                )

                st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
                render_section_title("Inference Output", badge="Classification Result")

                res_col1, res_col2 = st.columns([1, 1.2])
                with res_col1:
                    render_kpi_card(
                        label="Predicted Classification",
                        value=pred_label.capitalize(),
                        subtext=f"Numeric Class Code: {pred_class}",
                        status_color=COLOR_SUCCESS
                        if "benign" in str(pred_label).lower()
                        else COLOR_PRIMARY,
                    )

                with res_col2:
                    if pred_res.has_probabilities and pred_res.probabilities:
                        probs = pred_res.probabilities[0]
                        labels = pred_res.probability_class_labels or ["Class 0", "Class 1"]
                        prob_breakdown = {
                            lbl: f"{p * 100:.2f}%" for lbl, p in zip(labels, probs, strict=False)
                        }
                        st.markdown(
                            """
                            <div class="studio-card" style="padding: 1rem 1.25rem;">
                                <div style="font-weight: 600; font-size: 0.85rem; color: #64748B; text-transform: uppercase; margin-bottom: 0.4rem;">
                                    Class Probabilities
                                </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        p_cols = st.columns(len(prob_breakdown))
                        for p_idx, (p_name, p_val) in enumerate(prob_breakdown.items()):
                            with p_cols[p_idx]:
                                st.metric(label=p_name.capitalize(), value=p_val)
                        st.markdown("</div>", unsafe_allow_html=True)

                    elif pred_res.has_decision_scores and pred_res.decision_scores:
                        score = pred_res.decision_scores[0]
                        render_kpi_card(
                            label="Hyperplane Decision Score",
                            value=f"{score:.4f}",
                            subtext="Distance from SVM separating hyperplane",
                        )
            except MLAppException as exc:
                st.error(f"Prediction Error: {exc.message}")
