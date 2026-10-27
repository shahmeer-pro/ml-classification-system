"""Educational Guide & Limitations page."""

import streamlit as st

from ml_app.ui.components import render_disclaimer


def page_educational_guide() -> None:
    """Render the Educational Reference, Mathematics, and Limitations page."""
    render_disclaimer()
    st.title("Educational Reference, Mathematics & Limitations")

    st.subheader("1. Mathematical Formulations (Stage 1 & Stage 2)")
    st.markdown(
        r"""
        ### Logistic Regression
        - **Hypothesis**: $h_\theta(x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$
        - **Cost Function (Binary Cross-Entropy)**:
          $$J(w, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$
        - **Gradients**:
          $$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (h_\theta(X) - y), \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})$$

        ---

        ### Linear Support Vector Machine (Corrected Bias Normalization)
        - **Soft-Margin Hinge Objective**:
          $$\min_{w, b} \frac{1}{2} \|w\|^2 + C \cdot \frac{1}{m} \sum_{i=1}^m \max(0, 1 - y_{\text{signed}}^{(i)}(w^T x^{(i)} + b))$$
        - **Subgradients on Active Violations** ($\text{active} = \{i : 1 - y_{\text{signed}}^{(i)}(w^T x^{(i)} + b) > 0\}$):
          $$\frac{\partial L}{\partial w} = w - \frac{C}{m} \sum_{i \in \text{active}} y_{\text{signed}}^{(i)} x^{(i)}$$
          $$\frac{\partial L}{\partial b} = -\frac{C}{m} \sum_{i \in \text{active}} y_{\text{signed}}^{(i)}$$
        *Notice the critical $\frac{1}{m}$ normalization in the bias gradient $\frac{\partial L}{\partial b}$, preventing disproportionately large bias shifts.*
        """
    )

    st.subheader("2. System Architecture & Limitations")
    st.markdown(
        """
        - **Zero Database Requirement**: Model pipelines and artifacts are held strictly in process memory.
        - **Privacy Protection**: Uploaded CSV data and prediction queries are processed transiently in RAM and never written to permanent disk storage.
        - **Process Ephemerality**: In-memory models exist for the lifetime of the running server process; platform restarts will require retraining.
        - **Clinical Non-Use**: The Breast Cancer Wisconsin dataset is utilized solely as an academic benchmark. This system must never be used for clinical diagnosis or treatment planning.
        """
    )
