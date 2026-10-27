"""Inference and prediction service using in-memory model pipelines."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ml_app.exceptions import PredictionError
from ml_app.logging_config import logger
from ml_app.services.training_service import TrainingService, training_service


@dataclass
class PredictionResult:
    """Structured response container for batch or single record predictions."""

    predicted_classes: list[Any]
    predicted_class_names: list[str] | None
    has_probabilities: bool
    probabilities: list[list[float]] | None
    probability_class_labels: list[str] | None
    has_decision_scores: bool
    decision_scores: list[float] | None
    feature_count: int
    record_count: int
    predictions_dataframe: pd.DataFrame


class PredictionService:
    """Manages prediction requests against registered model artifacts."""

    def __init__(self, t_service: TrainingService | None = None) -> None:
        self.training_service = t_service or training_service

    def predict(
        self,
        artifact_id: str,
        input_data: pd.DataFrame | list[dict[str, Any]],
    ) -> PredictionResult:
        """Execute inference on new compatible feature records using fitted pipeline.

        Args:
            artifact_id: Unique UUID of the fitted model artifact.
            input_data: pandas DataFrame or list of dict records.

        Returns:
            PredictionResult: Detailed prediction outputs, classes, and confidence/scores.

        Raises:
            PredictionError: If input data is malformed or missing required features.
            ModelNotFoundError: If artifact_id does not exist.
        """
        artifact = self.training_service.get_artifact(artifact_id)

        # Convert to DataFrame if list of dicts
        if isinstance(input_data, list):
            if not input_data:
                raise PredictionError("Input prediction record list is empty.")
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise PredictionError(f"Unsupported input data type: {type(input_data).__name__}")

        if df.empty:
            raise PredictionError("Input DataFrame for prediction is empty.")

        # 1. Feature validation: ensure all required features are present
        missing_features = [f for f in artifact.feature_names if f not in df.columns]
        if missing_features:
            raise PredictionError(
                f"Missing {len(missing_features)} required feature column(s): {missing_features}."
            )

        # 2. Extract and order columns strictly matching the training pipeline
        X_pred = df[artifact.feature_names].copy()

        # 3. Apply the fitted pipeline without refitting
        try:
            raw_predictions = artifact.pipeline.predict(X_pred)
        except Exception as exc:
            logger.error("Inference execution failed on artifact '%s': %s", artifact_id, exc)
            raise PredictionError(f"Prediction failed: {exc}") from exc

        # 4. Map class names if known
        class_names: list[str] | None = None
        if artifact.class_mapping:
            class_names = [artifact.class_mapping.get(p, str(p)) for p in raw_predictions]

        # 5. Extract probabilities or decision scores
        has_probs = False
        probabilities: list[list[float]] | None = None
        prob_labels: list[str] | None = None

        has_scores = False
        decision_scores: list[float] | None = None

        if artifact.supports_probabilities and hasattr(artifact.pipeline, "predict_proba"):
            try:
                probs_arr = artifact.pipeline.predict_proba(X_pred)
                has_probs = True
                probabilities = [[round(float(p), 4) for p in row] for row in probs_arr]
                prob_labels = [str(c) for c in artifact.target_classes]
                if artifact.class_mapping:
                    prob_labels = [
                        f"{c} ({artifact.class_mapping.get(c, str(c))})"
                        for c in artifact.target_classes
                    ]
            except Exception as exc:
                logger.warning("Could not calculate predict_proba: %s", exc)

        elif artifact.supports_decision_scores and hasattr(artifact.pipeline, "decision_function"):
            try:
                scores_arr = artifact.pipeline.decision_function(X_pred)
                has_scores = True
                decision_scores = [round(float(s), 4) for s in scores_arr]
            except Exception as exc:
                logger.warning("Could not calculate decision_function: %s", exc)

        # 6. Build combined output DataFrame
        result_df = X_pred.copy()
        result_df["predicted_class"] = raw_predictions

        if class_names is not None:
            result_df["predicted_label"] = class_names

        if has_probs and probabilities is not None and prob_labels is not None:
            for idx, label_text in enumerate(prob_labels):
                col_name = f"probability_{label_text}"
                result_df[col_name] = [row[idx] for row in probabilities]

        if has_scores and decision_scores is not None:
            # Explicitly label as decision score, NOT probability
            result_df["decision_score_raw"] = decision_scores

        logger.info(
            "Completed predictions for %d records using artifact '%s'",
            len(result_df),
            artifact_id,
        )

        # Convert numpy types to native Python types for serialization
        clean_classes = [x.item() if hasattr(x, "item") else x for x in raw_predictions]

        return PredictionResult(
            predicted_classes=clean_classes,
            predicted_class_names=class_names,
            has_probabilities=has_probs,
            probabilities=probabilities,
            probability_class_labels=prob_labels,
            has_decision_scores=has_scores,
            decision_scores=decision_scores,
            feature_count=len(artifact.feature_names),
            record_count=len(result_df),
            predictions_dataframe=result_df,
        )


prediction_service = PredictionService()
