# Stage 1 & Stage 2 Notebooks Guide

This directory preserves the educational notebooks from Stages 1 and 2:

- **Stage 1**: Mathematical implementations of Linear Regression and Logistic Regression from scratch using NumPy, including gradient descent trajectories and loss history curves.
- **Stage 2**: Implementations and baseline comparisons of Gaussian Naive Bayes, Logistic Regression, and Linear SVM evaluated on the Breast Cancer Wisconsin dataset.

### Scratch Reference Implementations in Codebase:
For programmatic verification, the NumPy implementations from Stages 1 and 2—including the verified bias gradient normalization—are maintained in:
`src/ml_app/models/scratch_models.py`

Classes available:
- `ScratchLogisticRegression`: Gradient descent on binary cross-entropy loss with sigmoid activation and loss curves.
- `ScratchLinearSVM`: Subgradient descent on soft-margin hinge loss with mathematically normalized bias update:
  `db = -(C / n_samples) * sum(y_signed[active])`

These classes are tested in `tests/test_models.py`.
