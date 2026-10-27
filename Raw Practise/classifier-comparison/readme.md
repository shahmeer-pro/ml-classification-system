# ML Classifier Comparison: Breast Cancer Classification

## 1. Objective

The objective of this project was to implement and compare Gaussian Naive Bayes, Logistic Regression, and a linear Support Vector Machine (SVM) for binary classification.

## 2. Dataset

- **Dataset:** Breast Cancer Wisconsin dataset
- **Samples:** 569
- **Features:** 30 numerical features
- **Classes:** Malignant (0) and Benign (1)
- **Training/testing split:** 80% / 20%
- **Random state:** 42
- **Preprocessing:** StandardScaler fitted on training data only

## 3. Implementation

- **Gaussian Naive Bayes:** Implemented class priors, feature means and variances, Gaussian log-likelihoods, and class prediction.
- **Logistic Regression:** Implemented sigmoid activation, binary cross-entropy, gradient descent, and probability-based classification.
- **Linear SVM:** Implemented hinge loss, L2 regularization, subgradient descent, and decision-score-based classification.

## 4. Evaluation Metrics

The models were evaluated using the same held-out test set. Metrics included accuracy, malignant-class precision, malignant-class recall, malignant-class F1-score, and confusion matrices.

### Results

| Model | Accuracy | Malignant Precision | Malignant Recall | Malignant F1 |
|---|---|---|---|---|
| Gaussian Naive Bayes | [0.9298] | [0.9048] | [0.9048] | [0.9048] |
| Logistic Regression | [0.9737] | [0.9535] | [0.9762] | [0.9647] |
| Linear SVM | [0.9561] | [1.0000] | [0.8810] | [0.9367] |

## 5. Discussion

Explain which model achieved the highest accuracy and which achieved the highest malignant-class recall.

Discuss any differences in predictions, false negatives, false positives, and model assumptions.

Do not assume that the highest-accuracy model is automatically the best model for every use case.

## 6. Limitations

The comparison used a single train/test split and basic implementations of the algorithms. Results may vary with different splits, hyperparameters, and optimization methods.

The dataset is suitable for educational experimentation, but these models are not validated for clinical diagnosis.

## 7. Conclusion

Summarize the main differences observed among the three classifiers, identify the model that best met the chosen evaluation objective, and describe what you would improve next.
