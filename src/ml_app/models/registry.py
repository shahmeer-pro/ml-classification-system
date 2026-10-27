"""Model registry defining supported classifiers and their hyperparameter contracts."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterDefinition:
    """Metadata for a single tunable hyperparameter."""

    name: str
    param_type: str  # "float", "int", "str", "bool"
    default: Any
    options: list[Any] | None = None
    min_value: float | None = None
    max_value: float | None = None
    description: str = ""


@dataclass(frozen=True)
class ModelMetadata:
    """Descriptor for an algorithm available in the system."""

    model_id: str
    display_name: str
    estimator_class: str
    supports_probabilities: bool
    supports_decision_scores: bool
    requires_scaling: bool
    description: str
    parameters: dict[str, ParameterDefinition]


MODEL_REGISTRY: dict[str, ModelMetadata] = {
    "gaussian_nb": ModelMetadata(
        model_id="gaussian_nb",
        display_name="Gaussian Naive Bayes",
        estimator_class="GaussianNB",
        supports_probabilities=True,
        supports_decision_scores=False,
        requires_scaling=False,
        description=(
            "Probabilistic classifier applying Bayes' Theorem with the strong assumption "
            "that continuous features follow independent Gaussian normal distributions within each class. "
            "Fast and effective baseline for high-dimensional data."
        ),
        parameters={
            "var_smoothing": ParameterDefinition(
                name="var_smoothing",
                param_type="float",
                default=1e-9,
                min_value=1e-12,
                max_value=1e-1,
                description="Portion of the largest variance of all features that is added to variances for calculation stability.",
            )
        },
    ),
    "logistic_regression": ModelMetadata(
        model_id="logistic_regression",
        display_name="Logistic Regression",
        estimator_class="LogisticRegression",
        supports_probabilities=True,
        supports_decision_scores=True,
        requires_scaling=True,
        description=(
            "Linear model for binary or multiclass classification that models log-odds using a sigmoid/softmax "
            "function. Highly interpretable; requires feature scaling for optimal gradient convergence."
        ),
        parameters={
            "C": ParameterDefinition(
                name="C",
                param_type="float",
                default=1.0,
                min_value=0.001,
                max_value=100.0,
                description="Inverse of regularization strength; smaller values specify stronger regularization.",
            ),
            "max_iter": ParameterDefinition(
                name="max_iter",
                param_type="int",
                default=1000,
                min_value=100,
                max_value=5000,
                description="Maximum number of iterations taken for the solvers to converge.",
            ),
            "penalty": ParameterDefinition(
                name="penalty",
                param_type="str",
                default="l2",
                options=["l2", "none"],
                description="Regularization penalty type.",
            ),
        },
    ),
    "linear_svm": ModelMetadata(
        model_id="linear_svm",
        display_name="Linear Support Vector Machine (LinearSVC)",
        estimator_class="LinearSVC",
        supports_probabilities=False,
        supports_decision_scores=True,
        requires_scaling=True,
        description=(
            "Linear classification via soft-margin maximum margin hyperplane using the hinge loss. "
            "Outputs uncalibrated signed decision scores representing geometric distance to the hyperplane. "
            "Does not natively produce probabilities without calibration."
        ),
        parameters={
            "C": ParameterDefinition(
                name="C",
                param_type="float",
                default=1.0,
                min_value=0.001,
                max_value=100.0,
                description="Regularization parameter trading off margin width vs training errors.",
            ),
            "max_iter": ParameterDefinition(
                name="max_iter",
                param_type="int",
                default=2000,
                min_value=500,
                max_value=10000,
                description="Maximum number of iterations.",
            ),
        },
    ),
    "random_forest": ModelMetadata(
        model_id="random_forest",
        display_name="Random Forest Classifier",
        estimator_class="RandomForestClassifier",
        supports_probabilities=True,
        supports_decision_scores=False,
        requires_scaling=False,
        description=(
            "Ensemble of randomized decision trees combining bagging and feature subsampling. "
            "Robust against outliers and non-linear patterns."
        ),
        parameters={
            "n_estimators": ParameterDefinition(
                name="n_estimators",
                param_type="int",
                default=100,
                min_value=10,
                max_value=500,
                description="The number of trees in the forest.",
            ),
            "max_depth": ParameterDefinition(
                name="max_depth",
                param_type="int",
                default=10,
                min_value=1,
                max_value=50,
                description="The maximum depth of the individual decision trees.",
            ),
        },
    ),
}
