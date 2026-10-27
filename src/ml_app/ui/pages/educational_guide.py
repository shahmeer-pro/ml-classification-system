"""Educational and Technical Learning Guide page."""

import streamlit as st

from ml_app.ui.components import render_disclaimer
from ml_app.ui.theme import (
    render_page_header,
    render_section_title,
)
from ml_app.ui.visualizations import (
    plot_decision_boundary_concept,
    plot_svm_margin_concept,
    plot_threshold_tradeoff_concept,
)


def page_educational_guide() -> None:
    """Render the Learning Guide, Mathematics, and System Limitations page."""
    render_disclaimer()
    render_page_header(
        title="Learning Guide",
        subtitle="Understand the algorithms, evaluation metrics, and engineering principles behind your workbench.",
        badge="Technical Reference",
    )

    tab_algo, tab_metrics, tab_arch = st.tabs(
        [
            "1. Classification Mathematics",
            "2. Metrics & Clinical Diagnosis",
            "3. System Architecture & Ethics",
        ]
    )

    with tab_algo:
        render_section_title("Core Algorithms & Formulations")

        with st.expander("Logistic Regression (Gradient Descent & Cross-Entropy)", expanded=True):
            st.markdown(
                r"""
                **Hypothesis Function**: Maps continuous feature combinations onto $(0, 1)$ via the standard logistic sigmoid:
                $$h_\theta(x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

                **Cost Function (Binary Cross-Entropy / Log Loss)**:
                $$J(w, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$

                **Analytical Gradient Updates**:
                $$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (h_\theta(X) - y), \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})$$
                """
            )
            fig_db = plot_decision_boundary_concept()
            st.plotly_chart(fig_db, use_container_width=True)

        with st.expander(
            "Linear Support Vector Machine (Soft-Margin & Normalized Subgradients)", expanded=True
        ):
            st.markdown(
                r"""
                **Primal Soft-Margin Objective**: Maximizes margin width while penalizing slack margin violations with penalty parameter $C$:
                $$\min_{w, b} \frac{1}{2} \|w\|^2 + C \cdot \frac{1}{m} \sum_{i=1}^m \max(0, 1 - y_{\text{signed}}^{(i)}(w^T x^{(i)} + b))$$

                where $y_{\text{signed}} \in \{-1, +1\}$.

                **Subgradients on Active Violations** ($\text{active} = \{i : 1 - y_{\text{signed}}^{(i)}(w^T x^{(i)} + b) > 0\}$):
                $$\frac{\partial L}{\partial w} = w - \frac{C}{m} \sum_{i \in \text{active}} y_{\text{signed}}^{(i)} x^{(i)}$$
                $$\frac{\partial L}{\partial b} = -\frac{C}{m} \sum_{i \in \text{active}} y_{\text{signed}}^{(i)}$$

                > **Mathematical Precision Note**: Notice the rigorous $\frac{1}{m}$ normalization in the bias gradient $\frac{\partial L}{\partial b}$. Omitting this factor causes disproportionately volatile bias steps that destabilize hyperplane convergence.
                """
            )
            fig_svm = plot_svm_margin_concept()
            st.plotly_chart(fig_svm, use_container_width=True)

        with st.expander("Gaussian Naive Bayes (Probabilistic Classifier)", expanded=False):
            st.markdown(
                r"""
                **Bayes' Rule**:
                $$P(Y = c \mid X) = \frac{P(Y = c) \prod_{j=1}^d P(X_j \mid Y = c)}{P(X)}$$

                **Continuous Feature Likelihood**: Features conditioned on class $c$ follow a normal distribution with class-specific mean $\mu_{c,j}$ and variance $\sigma_{c,j}^2$:
                $$P(X_j = x_j \mid Y = c) = \frac{1}{\sqrt{2\pi \sigma_{c,j}^2}} \exp\left( -\frac{(x_j - \mu_{c,j})^2}{2\sigma_{c,j}^2} \right)$$
                """
            )

        with st.expander("Random Forest Classifier (Ensemble Bagging)", expanded=False):
            st.markdown(
                r"""
                - **Bootstrap Aggregation (Bagging)**: Generates $B$ bootstrap samples from training data with replacement.
                - **Feature Subspace Randomization**: At each node split, considers only a random subset of $m = \sqrt{p}$ features.
                - **Ensemble Decision**: Majority voting across individual decision trees reduces variance without increasing bias.
                """
            )

    with tab_metrics:
        render_section_title("Evaluation Metrics & Diagnostic Priorities")

        st.markdown(
            r"""
            <div class="studio-card">
                <div style="font-weight: 600; font-size: 0.95rem; color: #172033; margin-bottom: 0.4rem;">
                    The Clinical Diagnostic Dilemma: Precision vs Recall
                </div>
                <div style="font-size: 0.85rem; color: #334155; line-height: 1.6;">
                    In cancer detection, the positive class is <strong>Malignant</strong> (0 in standard Wisconsin coding).
                    A <strong>False Negative</strong> occurs when a patient with malignant tumors is classified as benign—delaying critical therapy.
                    Conversely, a <strong>False Positive</strong> triggers follow-up biopsy anxiety but is not fatal.
                    Therefore, <strong>Malignant Recall (Sensitivity)</strong> is prioritized above raw overall accuracy.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_section_title("Decision Threshold Trade-Off Simulation")
        fig_tradeoff = plot_threshold_tradeoff_concept()
        st.plotly_chart(fig_tradeoff, use_container_width=True)

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(
                r"""
                **Overall Accuracy**:
                $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

                **Precision (Positive Predictive Value)**:
                $$\text{Precision} = \frac{TP}{TP + FP}$$
                """
            )
        with m_col2:
            st.markdown(
                r"""
                **Recall (Sensitivity / True Positive Rate)**:
                $$\text{Recall} = \frac{TP}{TP + FN}$$

                **F1-Score (Harmonic Mean)**:
                $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
                """
            )

        st.markdown(
            """
            - **ROC-AUC**: Evaluates discrimination ability across all decision thresholds. AUC = 0.5 is random chance; AUC = 1.0 is perfect separation.
            - **Precision-Recall Curve**: Highly informative on class-imbalanced datasets where TN counts dominate ROC curves.
            """
        )

    with tab_arch:
        render_section_title("Architectural Guardrails & Operating Boundaries")

        st.markdown(
            """
            <div class="studio-card">
                <div style="font-weight: 600; font-size: 0.95rem; color: #172033; margin-bottom: 0.5rem;">
                    Engineering Principles Implemented in ML Studio
                </div>
                <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.85rem; color: #334155; line-height: 1.7;">
                    <li><strong>Strict Leakage Isolation:</strong> The stratified train/test split occurs before fitting scalers or imputers. Test features are only transformed via pre-fit statistics.</li>
                    <li><strong>Stateless Process Memory:</strong> Model artifacts and datasets are held transiently in RAM. Uploaded patient or custom CSVs are never written to permanent disk storage.</li>
                    <li><strong>Dual Execution Decoupling:</strong> The Streamlit frontend uses a unified <code>ClassificationApiClient</code> capable of seamless fallback between FastAPI REST endpoints and direct in-process services.</li>
                    <li><strong>Explicit Computation:</strong> Page navigation executes lightweight rendering logic; intensive model training and evaluation are triggered exclusively by explicit user intent.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
