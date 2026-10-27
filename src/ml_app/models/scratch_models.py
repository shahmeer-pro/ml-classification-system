"""Educational reference implementations from Scratch (Stage 1 & Stage 2).

Includes NumPy Linear Regression, Logistic Regression, and Linear SVM with the mathematically
verified bias-gradient normalization.

Objective for Linear SVM:
    L(w, b) = 0.5 * ||w||² + C * (1 / n_samples) * sum(max(0, 1 - y_signed * (Xw + b)))

Gradients on margin violations (active = 1 - y_signed * (Xw + b) > 0):
    dw = w - (C / n_samples) * (X[active].T @ y_signed[active])
    db = -(C / n_samples) * sum(y_signed[active])
"""

import numpy as np


class ScratchLogisticRegression:
    """Logistic Regression trained via Gradient Descent on Binary Cross-Entropy Loss."""

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchLogisticRegression":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_pred)

            # Binary cross entropy loss with epsilon clamp
            eps = 1e-15
            y_clamped = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(y * np.log(y_clamped) + (1 - y) * np.log(1 - y_clamped))
            self.loss_history.append(float(loss))

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")
        linear_pred = np.dot(X, self.weights) + self.bias
        p1 = self._sigmoid(linear_pred)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


class ScratchLinearSVM:
    """Linear Support Vector Machine trained with Subgradient Descent on Soft-Margin Hinge Loss.

    Corrected bias gradient:
        db = -(C / n_samples) * sum(y_signed[active])
    """

    def __init__(self, C: float = 1.0, learning_rate: float = 0.001, n_iterations: int = 1000):
        self.C = C
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchLinearSVM":
        n_samples, n_features = X.shape
        # Map {0, 1} to {-1, +1}
        y_signed = np.where(y <= 0, -1.0, 1.0)

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            decision = np.dot(X, self.weights) + self.bias
            margin = 1.0 - y_signed * decision
            active = margin > 0

            # Soft-margin hinge loss: 0.5 * ||w||^2 + C * mean(hinge)
            hinge_loss = np.maximum(0, margin)
            loss = 0.5 * np.dot(self.weights, self.weights) + self.C * np.mean(hinge_loss)
            self.loss_history.append(float(loss))

            # Gradients with correct 1/n_samples normalization
            if np.any(active):
                dw = self.weights - (self.C / n_samples) * np.dot(X[active].T, y_signed[active])
                db = -(self.C / n_samples) * np.sum(y_signed[active])
            else:
                dw = self.weights
                db = 0.0

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")
        return np.dot(X, self.weights) + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        decisions = self.decision_function(X)
        return (decisions >= 0).astype(int)
