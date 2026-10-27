"""Overview and Architecture page."""

import streamlit as st

from ml_app.ui.components import render_disclaimer


def page_overview() -> None:
    """Render the Overview and System Architecture page."""
    render_disclaimer()
    st.title("Interactive Machine Learning Classification Workbench")
    st.markdown(
        """
        Welcome to the **Portfolio-Ready Machine Learning Classification System**.
        This application bridges mathematical machine learning implementations built from scratch with
        production-grade web architecture, automated testing, and interactive model diagnostics.
        """
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Key Features")
        st.markdown(
            """
            - **End-to-End Pipeline**: Isolated train/test splitting before fitting transformers to guarantee zero data leakage.
            - **Multi-Algorithm Comparison**: Gaussian Naive Bayes, Logistic Regression, Linear SVM (LinearSVC), and Random Forest.
            - **Standardized Evaluation**: Accuracy, Macro Precision, Recall, F1, Per-class support, ROC-AUC, and PR curves.
            - **Medical Positive Class Focus**: Explicit tracking of Malignant-class recall & precision for breast cancer diagnosis.
            - **Interactive Predictions**: Batch CSV inference with CSV downloads and manual record prediction.
            - **Dual Architecture**: Operates as a distributed microservice setup (FastAPI + Streamlit) or standalone single-process application.
            """
        )

        st.subheader("Roadmap Trajectory")
        st.markdown(
            """
            - **Stage 1 (Foundation)**: Mathematical derivation of Linear and Logistic Regression from scratch in NumPy.
            - **Stage 2 (Exploration)**: Comparison of Gaussian NB, Logistic Regression, and Linear SVM on Breast Cancer data.
            - **Stage 3 (Application)**: Interactive Streamlit UI, CSV validation, dynamic preprocessing, and prediction workflow.
            - **Stage 4 (Production)**: FastAPI REST service, Pydantic schemas, pytest suite, Ruff compliance, and cloud deployment guides.
            """
        )

    with col2:
        st.subheader("System Architecture")
        st.code(
            """
            [Dataset Source]
            (Built-in or User CSV)
                   │
                   ▼
            [Data Validation & Guardrails]
                   │
                   ▼
            [Train / Test Split (Stratified)]
                   │
                   ▼
            [Modular Preprocessing Pipeline]
            (Imputation + Scaling / OneHot)
                   │
                   ▼
            [Model Training & Registry]
            (GaussianNB, LogReg, LinearSVC, RF)
                   │
                   ▼
            [Evaluation Diagnostics]
            (Confusion Matrix, ROC, PR Curves)
                   │
                   ▼
            [Inference Engine]
            (Batch CSV & Manual Form)
            """,
            language="text",
        )
