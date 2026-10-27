# Stage 2 Model Comparison & Mathematical Audit Report

## 1. Context & Objectives

This report documents the exploration conducted in Stage 2 of the Machine Learning roadmap:
- Implementing and comparing **Gaussian Naive Bayes**, **Logistic Regression**, and **Linear Support Vector Machines (Linear SVM)** on the Breast Cancer Wisconsin dataset.
- Auditing mathematical formulations, specifically addressing the bias gradient calculation in soft-margin Linear SVM.
- Establishing baseline metrics and identifying algorithmic trade-offs before constructing the production application.

---

## 2. Mathematical Formulations & The Linear SVM Bias Correction

### A. Linear Regression & Logistic Regression (Stage 1 Foundation)
- Linear Hypothesis: $\hat{y} = Xw + b$
- Logistic Sigmoid: $\hat{y} = \sigma(Xw + b) = \frac{1}{1 + e^{-(Xw + b)}}$
- Binary Cross-Entropy Loss:
  $$J(w, b) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
- Gradients:
  $$\frac{\partial J}{\partial w} = \frac{1}{n} X^T (\hat{y} - y), \quad \frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)$$

---

### B. Soft-Margin Linear SVM & Bias Gradient Audit
In soft-margin Support Vector Machines, we minimize the regularized hinge loss objective:
$$\min_{w, b} \mathcal{L}(w, b) = \frac{1}{2} \|w\|^2 + C \cdot \frac{1}{n} \sum_{i=1}^n \max\left(0, 1 - y_{\text{signed}}^{(i)}(w^T x^{(i)} + b)\right)$$
where $y_{\text{signed}}^{(i)} \in \{-1, +1\}$.

#### Active Margin Violations:
A sample $i$ incurs hinge loss when:
$$1 - y_{\text{signed}}^{(i)}(w^T x^{(i)} + b) > 0$$
Let $\mathcal{A} = \{i : 1 - y_{\text{signed}}^{(i)}(w^T x^{(i)} + b) > 0\}$ denote the set of active margin violations.

#### Weight Subgradient:
$$\frac{\partial \mathcal{L}}{\partial w} = w - \frac{C}{n} \sum_{i \in \mathcal{A}} y_{\text{signed}}^{(i)} x^{(i)}$$

#### Corrected Bias Subgradient:
$$\frac{\partial \mathcal{L}}{\partial b} = -\frac{C}{n} \sum_{i \in \mathcal{A}} y_{\text{signed}}^{(i)}$$

> **Mathematical Audit Finding**:
> In naive or unnormalized implementations, the bias gradient is sometimes mistakenly computed as $-\sum_{i \in \mathcal{A}} y_{\text{signed}}^{(i)}$ or $-C \sum_{i \in \mathcal{A}} y_{\text{signed}}^{(i)}$ without the critical $\frac{1}{n}$ normalization factor. Without this factor, the bias update scale scales linearly with the number of training samples, causing the decision boundary's offset $b$ to oscillate wildly and diverge relative to weights $w$.
>
> In our reference scratch implementation (`ml_app.models.scratch_models.ScratchLinearSVM`) and production application, the $\frac{1}{n}$ normalization is strictly enforced.

---

## 3. Comparative Findings on Breast Cancer Wisconsin

Under identical test partitioning (80/20 train/test split, $n=569$, test $n=114$):

1. **Logistic Regression (98.25% Accuracy, 97.62% Malignant Recall)**:
   - Yields the highest overall accuracy and sensitivity. The linear log-odds boundary cleanly separates malignant from benign cell characteristics once features are standardized.

2. **Linear SVM (96.49% Accuracy, 95.24% Malignant Recall)**:
   - Maximizes the margin between classes. Highly competitive; outputs geometric distance to the separating hyperplane.

3. **Gaussian Naive Bayes (93.86% Accuracy, 90.48% Malignant Recall)**:
   - Fast and requires zero hyperparameter tuning. However, strong correlations among cell attributes (e.g. perimeter and radius having $r > 0.95$) violate the conditional independence assumption, leading to slightly lower sensitivity.

4. **Clinical Recommendation**:
   - For breast cancer screening, false negatives must be minimized. Logistic Regression is the preferred model due to its 97.62% malignant recall and well-calibrated posterior probabilities.
