"""Tests for model factory, parameter contracts, and scratch implementations."""

import numpy as np
import pandas as pd
import pytest

from ml_app.exceptions import ModelTrainingError
from ml_app.models.factory import create_classifier_pipeline
from ml_app.models.scratch_models import ScratchLinearSVM, ScratchLogisticRegression


@pytest.mark.parametrize(
    "model_id", ["gaussian_nb", "logistic_regression", "linear_svm", "random_forest"]
)
def test_all_registered_models_train_and_predict(synthetic_binary_df: pd.DataFrame, model_id: str):
    feat_cols = [c for c in synthetic_binary_df.columns if c != "target"]
    pipeline = create_classifier_pipeline(
        model_id=model_id,
        numeric_features=feat_cols,
        categorical_features=None,
        params=None,
        random_state=42,
    )

    X = synthetic_binary_df[feat_cols]
    y = synthetic_binary_df["target"]

    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})


def test_unknown_model_raises_error():
    with pytest.raises(ModelTrainingError, match="Unknown model identifier"):
        create_classifier_pipeline(
            model_id="nonexistent_deep_net",
            numeric_features=["f1"],
        )


def test_invalid_hyperparameter_raises_error():
    with pytest.raises(ModelTrainingError, match="Unknown parameter"):
        create_classifier_pipeline(
            model_id="logistic_regression",
            numeric_features=["f1"],
            params={"invalid_hyperparam": 999},
        )

    with pytest.raises(ModelTrainingError, match="below minimum"):
        create_classifier_pipeline(
            model_id="logistic_regression",
            numeric_features=["f1"],
            params={"C": -1.0},
        )


def test_scratch_logistic_regression():
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    scratch_clf = ScratchLogisticRegression(learning_rate=0.1, n_iterations=100)
    scratch_clf.fit(X, y)

    assert len(scratch_clf.loss_history) == 100
    assert scratch_clf.loss_history[-1] < scratch_clf.loss_history[0]  # Loss must decrease

    preds = scratch_clf.predict(X)
    assert preds.shape == (50,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_scratch_linear_svm_with_normalized_bias_gradient():
    np.random.seed(42)
    X = np.random.randn(60, 4)
    y = (X[:, 0] > 0).astype(int)

    svm = ScratchLinearSVM(C=1.0, learning_rate=0.01, n_iterations=150)
    svm.fit(X, y)

    assert len(svm.loss_history) == 150
    assert svm.weights is not None
    assert svm.weights.shape == (4,)

    preds = svm.predict(X)
    assert len(preds) == 60
    assert set(np.unique(preds)).issubset({0, 1})
