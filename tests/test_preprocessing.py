"""Tests for preprocessing transformers, imputation, and splitting isolation."""

import numpy as np
import pandas as pd

from ml_app.preprocessing.pipeline import (
    build_preprocessing_pipeline,
    prepare_and_split_data,
)


def test_prepare_and_split_reproducibility(synthetic_binary_df: pd.DataFrame):
    split1 = prepare_and_split_data(
        df=synthetic_binary_df, target_column="target", test_size=0.25, random_state=123
    )
    split2 = prepare_and_split_data(
        df=synthetic_binary_df, target_column="target", test_size=0.25, random_state=123
    )

    pd.testing.assert_frame_equal(split1.X_train, split2.X_train)
    pd.testing.assert_series_equal(split1.y_train, split2.y_train)
    assert len(split1.X_train) == 75
    assert len(split1.X_test) == 25


def test_no_data_leakage_in_preprocessing():
    """Verify that scaler/imputer statistics are derived strictly from training data."""
    # Create train and test partitions manually
    train_df = pd.DataFrame({"feat": [10.0, 20.0, 30.0]})  # Mean = 20, Std ~ 8.16
    test_df = pd.DataFrame({"feat": [100.0, 200.0, 300.0]})  # Vastly different scale

    preprocessor = build_preprocessing_pipeline(
        numeric_features=["feat"],
        scale_numeric=True,
    )

    # Fit preprocessor only on train
    preprocessor.fit(train_df)

    # Inspect fitted mean from scaler
    scaler_mean = preprocessor.named_transformers_["numeric"].named_steps["scaler"].mean_[0]
    assert scaler_mean == 20.0  # Must match train mean, completely unaware of test data!

    # Transform test partition with train statistics
    transformed_test = preprocessor.transform(test_df)
    assert transformed_test.shape == (3, 1)


def test_categorical_and_unseen_values(synthetic_mixed_df: pd.DataFrame):
    preprocessor = build_preprocessing_pipeline(
        numeric_features=["num_1", "num_2"],
        categorical_features=["cat_1"],
        scale_numeric=False,
    )

    preprocessor.fit(synthetic_mixed_df[["num_1", "num_2", "cat_1"]])

    # Test with unseen category 'group_UNKNOWN'
    test_data = pd.DataFrame(
        {
            "num_1": [1.5],
            "num_2": [np.nan],  # Should be imputed
            "cat_1": ["group_UNKNOWN"],  # Should be handled gracefully without error
        }
    )

    transformed = preprocessor.transform(test_data)
    assert transformed.shape[0] == 1
    # Check that NaN in num_2 was imputed and not remaining NaN
    assert not np.isnan(transformed[0, 1])
