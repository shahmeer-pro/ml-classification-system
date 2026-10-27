# Comprehensive Model Comparison & Benchmark Report

## 1. Executive Summary

This report documents the empirical comparison of four classification algorithms evaluated on the **Breast Cancer Wisconsin (Diagnostic)** dataset under identical experimental conditions.

### Benchmark Setup
- **Dataset**: Breast Cancer Wisconsin (Diagnostic)
- **Total Samples**: 569 (212 Malignant, 357 Benign)
- **Features**: 30 numeric cell nucleus characteristics
- **Train/Test Split**: 80% Train (455 samples) / 20% Test (114 samples)
- **Split Strategy**: Stratified random split (`random_state=42`)
- **Preprocessing**: Median imputation + StandardScaler (for linear models and SVM) fitted strictly on the training partition
- **Target Convention**:
  - Class `0`: **Malignant** (positive class of clinical interest)
  - Class `1`: **Benign**

---

## 2. Real Executed Benchmark Results

All figures below represent actual values computed on the held-out test partition (114 samples) during automated execution:

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | ROC AUC | Malignant Precision | Malignant Recall (Sensitivity) | Malignant F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **98.25%** | **0.9812** | **0.9812** | **0.9812** | **0.9825** | **0.9954** | **0.9762** | **0.9762** | **0.9762** |
| **Linear SVM (LinearSVC)** | 96.49% | 0.9623 | 0.9623 | 0.9623 | 0.9649 | 0.9927 | 0.9524 | 0.9524 | 0.9524 |
| **Random Forest** | 95.61% | 0.9551 | 0.9504 | 0.9526 | 0.9560 | 0.9937 | 0.9512 | 0.9286 | 0.9398 |
| **Gaussian Naive Bayes** | 93.86% | 0.9360 | 0.9315 | 0.9337 | 0.9384 | 0.9878 | 0.9268 | 0.9048 | 0.9157 |

---

## 3. Algorithm Analysis & Trade-Offs

### A. Logistic Regression (Top Performer: 98.25% Accuracy, 97.62% Malignant Recall)
- **Mechanism**: Models log-odds using a linear combination of features through a sigmoid activation:
  $$P(y=1|x) = \frac{1}{1 + e^{-(w^T x + b)}}$$
- **Why It Excels**: Breast cancer cell nucleus attributes exhibit strong linear separability once standardized. Regularized log-loss prevents extreme coefficient magnitudes, resulting in calibrated probabilities and superior generalization.

### B. Linear Support Vector Machine (LinearSVC) (96.49% Accuracy, 95.24% Malignant Recall)
- **Mechanism**: Maximizes the geometric margin separating malignant and benign feature distributions using the soft-margin hinge loss:
  $$\min_{w, b} \frac{1}{2} \|w\|^2 + C \cdot \frac{1}{m} \sum_{i=1}^m \max(0, 1 - y_{\text{signed}}^{(i)}(w^T x^{(i)} + b))$$
- **Considerations**: LinearSVC outputs signed continuous distance from the decision boundary rather than probabilities. In clinical contexts, these scores indicate geometric margin confidence but are not calibrated percentages.

### C. Gaussian Naive Bayes (93.86% Accuracy, 90.48% Malignant Recall)
- **Mechanism**: Computes class likelihoods using independent Gaussian probability density functions per feature:
  $$P(x_i | y=c) = \frac{1}{\sqrt{2\pi \sigma_{c,i}^2}} \exp\left(-\frac{(x_i - \mu_{c,i})^2}{2\sigma_{c,i}^2}\right)$$
- **Why It Has Lower Recall**: The foundational Naive Bayes assumption posits that all 30 features are mutually independent given the class label. In reality, morphological measurements (e.g., radius, perimeter, and area) have high collinearity ($r > 0.95$), which violates this independence assumption and leads to slightly overconfident probability estimates.

---

## 4. Why Malignant Recall Outweighs Raw Accuracy

In oncology and medical diagnosis, a **False Negative** (classifying a malignant tumor as benign) can have life-threatening consequences, leading to delayed medical intervention. Conversely, a **False Positive** (classifying a benign lesion as malignant) causes emotional distress and warrants follow-up imaging or biopsy, but is substantially less catastrophic.

Therefore, **Malignant Recall (Sensitivity)**:
$$\text{Recall}_{\text{malignant}} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$
is the primary operational metric of concern. In our tests, **Logistic Regression achieved 97.62% malignant recall**, missing only 1 malignant sample out of 42 in the test set.
