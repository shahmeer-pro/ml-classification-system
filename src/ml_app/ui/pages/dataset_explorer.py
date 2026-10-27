"""Dataset Explorer & Validation page."""

import streamlit as st

from ml_app.exceptions import MLAppException
from ml_app.ui.api_client import ClassificationApiClient
from ml_app.ui.cache import get_cached_builtin_dataset
from ml_app.ui.components import plot_class_distribution, render_disclaimer


def page_dataset_explorer() -> None:
    """Render the Dataset Explorer and Validation page."""
    render_disclaimer()
    st.title("Dataset Explorer & Validation")

    api_client = ClassificationApiClient()

    ds_option = st.radio(
        "Select Dataset Source:",
        ["Built-in: Breast Cancer Wisconsin", "Upload Custom CSV"],
        horizontal=True,
    )

    if ds_option == "Built-in: Breast Cancer Wisconsin":
        if st.session_state.get("dataset_choice") != "builtin" or "df" not in st.session_state:
            st.session_state.dataset_choice = "builtin"
            df, meta = get_cached_builtin_dataset()
            st.session_state.df = df
            st.session_state.meta = meta
            st.session_state.target_column = meta["target_name"]
            st.session_state.feature_columns = meta["feature_names"]
            st.session_state.pop("fig_dist", None)

        meta = st.session_state.meta
        df = st.session_state.df

        st.success(
            f"Loaded {meta['name']} with {len(df)} samples and {len(meta['feature_names'])} numeric features."
        )
        st.markdown(
            f"**Target column**: `{meta['target_name']}` | Class mapping: `0: malignant, 1: benign`"
        )

    else:
        st.session_state.dataset_choice = "uploaded"
        uploaded_file = st.file_uploader("Upload a CSV dataset (max 10MB):", type=["csv"])

        if uploaded_file is not None:
            try:
                bytes_data = uploaded_file.getvalue()
                df_up, summary = api_client.validate_csv_bytes(
                    bytes_data, filename=uploaded_file.name
                )
                st.session_state.df = df_up
                st.session_state.meta = None

                st.success(
                    f"Successfully validated '{uploaded_file.name}' ({summary['num_rows']} rows, {summary['num_cols']} columns)."
                )

                if summary["warnings"]:
                    with st.expander("Validation Warnings", expanded=True):
                        for warn in summary["warnings"]:
                            st.warning(warn)

                # Target column selector
                all_cols = list(df_up.columns)
                prev_target = st.session_state.get("target_column")
                target_idx = (
                    all_cols.index(prev_target) if prev_target in all_cols else len(all_cols) - 1
                )
                selected_target = st.selectbox(
                    "Select Target (Label) Column:", all_cols, index=target_idx
                )
                if selected_target != prev_target:
                    st.session_state.target_column = selected_target
                    st.session_state.pop("fig_dist", None)

                # Feature columns selector
                candidate_features = [c for c in all_cols if c != selected_target]
                prev_feats = st.session_state.get("feature_columns") or candidate_features
                valid_prev_feats = [
                    f for f in prev_feats if f in candidate_features
                ] or candidate_features
                selected_features = st.multiselect(
                    "Select Feature Columns for Modeling:",
                    candidate_features,
                    default=valid_prev_feats,
                )
                st.session_state.feature_columns = selected_features

            except MLAppException as exc:
                st.error(f"Validation Error: {exc.message}")
            except Exception as exc:
                st.error(f"Failed to process uploaded CSV: {exc}")

    df_current = st.session_state.get("df")
    target_current = st.session_state.get("target_column")

    if df_current is not None and target_current in df_current.columns:
        st.write("---")
        st.subheader("Dataset Summary & Sample Inspection")

        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
        col_metric1.metric("Total Rows", len(df_current))
        col_metric2.metric("Total Columns", df_current.shape[1])
        col_metric3.metric("Selected Features", len(st.session_state.get("feature_columns", [])))
        col_metric4.metric("Missing Values", int(df_current.isna().sum().sum()))

        chart_col, preview_col = st.columns([1, 2])
        with chart_col:
            # Cache the Plotly distribution figure in session state
            if "fig_dist" not in st.session_state:
                val_counts = df_current[target_current].value_counts().to_dict()
                st.session_state.fig_dist = plot_class_distribution(
                    {str(k): int(v) for k, v in val_counts.items()}
                )
            st.plotly_chart(st.session_state.fig_dist, use_container_width=True)

        with preview_col:
            st.write("**First 10 Rows Preview:**")
            st.dataframe(df_current.head(10), use_container_width=True)

            with st.expander("View Full Dataset"):
                st.dataframe(df_current, use_container_width=True)
