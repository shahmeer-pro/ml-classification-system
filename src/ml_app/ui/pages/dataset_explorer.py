import numpy as np
import pandas as pd
import streamlit as st

from ml_app.exceptions import MLAppException
from ml_app.ui.api_client import ClassificationApiClient
from ml_app.ui.cache import get_cached_builtin_dataset
from ml_app.ui.components import plot_class_distribution, render_disclaimer
from ml_app.ui.theme import (
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
    render_kpi_card,
    render_page_header,
    render_section_title,
)
from ml_app.ui.visualizations import (
    plot_categorical_distribution,
    plot_correlation_heatmap,
    plot_feature_scatter,
    plot_missing_values,
    plot_numerical_distribution,
    plot_pca_projection,
)


def page_dataset_explorer() -> None:
    """Render the Dataset Explorer and Validation page."""
    render_disclaimer()
    render_page_header(
        title="Dataset Explorer",
        subtitle="Upload, inspect, and validate your classification dataset.",
        badge="Data Ingestion",
    )

    api_client = ClassificationApiClient()

    render_section_title("Dataset Source Selection")
    ds_option = st.radio(
        "Select Dataset Source:",
        ["Built-in: Breast Cancer Wisconsin (Diagnostic)", "Upload Custom CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if ds_option == "Built-in: Breast Cancer Wisconsin (Diagnostic)":
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

        st.markdown(
            f"""
            <div class="studio-card" style="padding: 0.9rem 1.25rem; margin-top: 0.5rem; margin-bottom: 1.25rem;">
                <div style="font-weight: 600; color: #172033; font-size: 0.95rem;">
                    Active Dataset: {meta["name"]}
                </div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 0.2rem;">
                    Standard clinical benchmark with {len(df)} patient cell samples and {len(meta["feature_names"])} numeric features.
                    Target column: <code>{meta["target_name"]}</code> (0: Malignant, 1: Benign).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.session_state.dataset_choice = "uploaded"
        st.markdown(
            """
            <div style="font-size: 0.825rem; color: #64748B; margin-bottom: 0.5rem;">
                Supported format: <strong>.csv</strong> | Maximum file size: <strong>10 MB</strong>. Data is processed in-memory and never stored permanently on disk.
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload CSV file:", type=["csv"], label_visibility="collapsed"
        )

        if uploaded_file is not None:
            try:
                bytes_data = uploaded_file.getvalue()
                df_up, summary = api_client.validate_csv_bytes(
                    bytes_data, filename=uploaded_file.name
                )
                st.session_state.df = df_up
                st.session_state.meta = None

                st.success(
                    f"Successfully validated **{uploaded_file.name}** ({summary['num_rows']} rows, {summary['num_cols']} columns)."
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
                render_section_title("Column Configurations")
                col_sel1, col_sel2 = st.columns([1, 2])
                with col_sel1:
                    selected_target = st.selectbox(
                        "Target (Label) Column:", all_cols, index=target_idx
                    )
                    if selected_target != prev_target:
                        st.session_state.target_column = selected_target
                        st.session_state.pop("fig_dist", None)

                # Feature columns selector
                with col_sel2:
                    candidate_features = [c for c in all_cols if c != selected_target]
                    prev_feats = st.session_state.get("feature_columns") or candidate_features
                    valid_prev_feats = [
                        f for f in prev_feats if f in candidate_features
                    ] or candidate_features
                    selected_features = st.multiselect(
                        "Feature Columns for Modeling:",
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
        render_section_title("Dataset Summary")

        missing_count = int(df_current.isna().sum().sum())
        num_classes = int(df_current[target_current].nunique())
        num_features = len(st.session_state.get("feature_columns", []))

        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        with sum_col1:
            render_kpi_card(
                label="Total Records",
                value=f"{len(df_current):,}",
                subtext="Samples available",
                status_color=COLOR_PRIMARY,
            )
        with sum_col2:
            render_kpi_card(
                label="Total Dimensions",
                value=f"{df_current.shape[1]} cols",
                subtext=f"{num_features} modeling features",
            )
        with sum_col3:
            render_kpi_card(
                label="Target Classes",
                value=str(num_classes),
                subtext=f"Column: {target_current}",
            )
        with sum_col4:
            render_kpi_card(
                label="Missing Values",
                value=str(missing_count),
                subtext="Zero missing detected" if missing_count == 0 else "Requires imputation",
                status_color=COLOR_SUCCESS if missing_count == 0 else COLOR_WARNING,
            )

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

        with st.expander("Data Preview & Feature Schema Audit", expanded=False):
            st.write("**First 10 Sample Rows:**")
            st.dataframe(df_current.head(10), use_container_width=True)
            st.write("**Data Types & Completeness:**")
            audit_df = pd.DataFrame(
                {
                    "Data Type": df_current.dtypes.astype(str),
                    "Null Count": df_current.isna().sum(),
                    "Null Ratio (%)": (df_current.isna().mean() * 100).round(2),
                    "Unique Values": df_current.nunique(),
                }
            )
            st.dataframe(audit_df, use_container_width=True)

        render_section_title("Exploratory Data Visualizations", badge="Interactive Analysis")

        tab_target, tab_dist, tab_rel, tab_pca = st.tabs(
            [
                "1. Class Balance & Missing Values",
                "2. Feature Distributions",
                "3. Bivariate Relationships & Correlation",
                "4. PCA 2D Projection",
            ]
        )

        with tab_target:
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                render_section_title("Target Class Distribution")
                val_counts = df_current[target_current].value_counts()
                fig_dist = plot_class_distribution(
                    val_counts.to_dict(),
                    title=f"Class Frequency: '{target_current}'",
                    show_percentages=True,
                )
                st.plotly_chart(fig_dist, use_container_width=True)

                total_records = len(df_current)
                min_ratio = (val_counts.min() / total_records) if total_records > 0 else 0
                if min_ratio < 0.35 and len(val_counts) == 2:
                    st.warning(
                        f"⚠️ **Moderate Class Imbalance**: Minority class comprises {min_ratio * 100:.1f}% of total samples. "
                        "When evaluating models, evaluate Precision, Recall, and F1 rather than raw Accuracy."
                    )

            with t_col2:
                render_section_title("Missing-Value Analysis")
                if missing_count == 0:
                    st.markdown(
                        """
                        <div class="studio-card" style="background-color: #F0FDF4; border-color: #BBF7D0; margin-top: 1rem;">
                            <div style="font-weight: 600; color: #15803D; margin-bottom: 0.25rem;">
                                ✓ Zero Missing Values Detected
                            </div>
                            <div style="font-size: 0.85rem; color: #166534;">
                                All columns are 100% complete across all records. No imputation required prior to model training.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    show_all_cols = st.checkbox(
                        "Show all columns (including 0 missing)", value=False
                    )
                    fig_missing = plot_missing_values(df_current, show_all=show_all_cols)
                    if fig_missing is not None:
                        st.plotly_chart(fig_missing, use_container_width=True)

        with tab_dist:
            num_cols = list(df_current.select_dtypes(include=[np.number]).columns)
            cat_cols = list(df_current.select_dtypes(exclude=[np.number]).columns)
            # Remove target if in cat_cols to keep distinct
            cat_cols = [c for c in cat_cols if c != target_current]

            d_col1, d_col2 = st.columns([1.2, 0.8])
            with d_col1:
                if num_cols:
                    selected_num = st.selectbox(
                        "Select Numerical Feature for Distribution:",
                        num_cols,
                        index=0,
                        key="eda_num_feature",
                    )
                    fig_num = plot_numerical_distribution(df_current, selected_num)
                    st.plotly_chart(fig_num, use_container_width=True)
                else:
                    st.info("No numerical feature columns detected in this dataset.")

            with d_col2:
                if cat_cols:
                    selected_cat = st.selectbox(
                        "Select Categorical Feature:",
                        cat_cols,
                        index=0,
                        key="eda_cat_feature",
                    )
                    fig_cat = plot_categorical_distribution(df_current, selected_cat)
                    st.plotly_chart(fig_cat, use_container_width=True)
                else:
                    st.markdown(
                        """
                        <div class="studio-card" style="margin-top: 1.8rem;">
                            <div style="font-weight: 600; color: #172033; margin-bottom: 0.25rem;">
                                Categorical Attributes
                            </div>
                            <div style="font-size: 0.825rem; color: #64748B;">
                                This dataset is composed exclusively of continuous numerical dimensions. Categorical distribution plots will appear automatically when mixed datasets are ingested.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with tab_rel:
            rel_col1, rel_col2 = st.columns([1.1, 1.2])
            with rel_col1:
                render_section_title("Bivariate Feature Scatter")
                if len(num_cols) >= 2:
                    sc_c1, sc_c2 = st.columns(2)
                    with sc_c1:
                        feat_x = st.selectbox("Feature X-Axis:", num_cols, index=0, key="scat_x")
                    with sc_c2:
                        default_y = 1 if len(num_cols) > 1 else 0
                        feat_y = st.selectbox(
                            "Feature Y-Axis:", num_cols, index=default_y, key="scat_y"
                        )

                    fig_scatter = plot_feature_scatter(
                        df_current,
                        x_col=feat_x,
                        y_col=feat_y,
                        target_col=target_current,
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info(
                        "At least two numerical columns are required for a bivariate scatter plot."
                    )

            with rel_col2:
                render_section_title("Correlation Matrix Heatmap")
                if len(num_cols) >= 2:
                    # Provide subset selection if many features exist
                    default_corr_cols = num_cols[:12] if len(num_cols) > 12 else num_cols
                    corr_features = st.multiselect(
                        "Features in Correlation Heatmap:",
                        num_cols,
                        default=default_corr_cols,
                        key="corr_multiselect",
                    )
                    if len(corr_features) >= 2:
                        fig_corr = plot_correlation_heatmap(df_current, columns=corr_features)
                        st.plotly_chart(fig_corr, use_container_width=True)
                        st.caption(
                            "💡 **Statistical Notice**: Pearson correlation assesses linear association only and does not establish causality or non-linear dependencies."
                        )
                    else:
                        st.info(
                            "Please select at least two features to compute a correlation matrix."
                        )
                else:
                    st.info("Insufficient numerical features for correlation matrix.")

        with tab_pca:
            render_section_title("2D Principal Component Analysis (Exploratory Projection)")
            st.markdown(
                """
                <div style="font-size: 0.825rem; color: #64748B; margin-bottom: 0.75rem;">
                    PCA scales continuous features and linearly projects them onto the orthogonal directions of maximum variance.
                    <strong>Notice:</strong> This is an exploratory visualization tool, not a model evaluation. It may discard non-linear patterns.
                </div>
                """,
                unsafe_allow_html=True,
            )
            features_for_pca = st.session_state.get("feature_columns") or num_cols
            if len(features_for_pca) >= 2:
                fig_pca = plot_pca_projection(
                    df_current,
                    feature_cols=features_for_pca,
                    target_col=target_current,
                )
                st.plotly_chart(fig_pca, use_container_width=True)
            else:
                st.info("At least two features are required to compute PCA projection.")
