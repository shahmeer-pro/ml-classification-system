"""Predictions & Inference page."""

import io

import pandas as pd
import streamlit as st

from ml_app.exceptions import MLAppException
from ml_app.services.training_service import training_service
from ml_app.ui.api_client import ClassificationApiClient
from ml_app.ui.components import render_disclaimer


def page_predictions() -> None:
    """Render the Predictions and Inference page."""
    render_disclaimer()
    st.title("Inference & Predictions")

    artifact_id = st.session_state.get("trained_artifact_id")

    if artifact_id is None:
        st.warning(
            "⚠️ No fitted model pipeline found in the current session. "
            "Please navigate to 'Train & Evaluate' and train a model first."
        )
        st.stop()

    tr = st.session_state.get("training_result")
    model_name = tr.display_name if tr else "Fitted Classifier"
    st.success(f"Active In-Memory Model: **{model_name}** (`{artifact_id[:8]}...`)")

    pred_mode = st.radio(
        "Choose Prediction Method:", ["Batch CSV Upload", "Manual Record Entry"], horizontal=True
    )

    api_client = ClassificationApiClient()

    if pred_mode == "Batch CSV Upload":
        st.subheader("Upload Prediction Data (CSV)")
        st.caption(
            "The file must contain all required feature columns. A target column is NOT required."
        )

        pred_file = st.file_uploader("Upload CSV for inference:", type=["csv"], key="pred_csv")
        if pred_file is not None:
            try:
                pred_df = pd.read_csv(io.BytesIO(pred_file.getvalue()))
                st.write(f"Uploaded {len(pred_df)} records for inference.")

                if st.button("Generate Predictions", type="primary"):
                    with st.spinner("Applying fitted pipeline..."):
                        pred_res = api_client.predict(artifact_id=artifact_id, input_data=pred_df)
                        st.write("### Prediction Results")
                        res_df = pred_res.predictions_dataframe

                        # Display preview by default to maintain fast UI responsiveness
                        st.write("**Predictions Preview (First 20 Records):**")
                        st.dataframe(res_df.head(20), use_container_width=True)

                        if len(res_df) > 20:
                            with st.expander(f"View All {len(res_df)} Records"):
                                st.dataframe(res_df, use_container_width=True)

                        # CSV Download button
                        csv_data = res_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv_data,
                            file_name="predictions_output.csv",
                            mime="text/csv",
                        )
            except MLAppException as exc:
                st.error(f"Inference error: {exc.message}")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    else:
        st.subheader("Manual Feature Input Form")
        st.caption("Enter feature values to observe immediate classification outputs:")

        try:
            artifact = training_service.get_artifact(artifact_id)
            feat_list = artifact.feature_names
        except Exception:
            feat_list = []

        if not feat_list:
            st.error("Could not retrieve feature list for the active model artifact.")
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
                        f"{f_name}",
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

                st.subheader("Prediction Output")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.success(f"Predicted Class: **{pred_class}** ({pred_label})")

                with res_col2:
                    if pred_res.has_probabilities and pred_res.probabilities:
                        probs = pred_res.probabilities[0]
                        labels = pred_res.probability_class_labels or ["Class 0", "Class 1"]
                        prob_breakdown = {
                            lbl: f"{p * 100:.2f}%" for lbl, p in zip(labels, probs, strict=False)
                        }
                        st.json(prob_breakdown)
                    elif pred_res.has_decision_scores and pred_res.decision_scores:
                        score = pred_res.decision_scores[0]
                        st.info(f"Signed Decision Score (Hyperplane Distance): **{score:.4f}**")
            except MLAppException as exc:
                st.error(f"Prediction Error: {exc.message}")
