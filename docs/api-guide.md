# REST API Guide & Integration Reference

The FastAPI service exposes endpoints for dataset exploration, validation, model training, multi-algorithm comparison, and prediction inference.

Interactive Swagger documentation is available locally at:
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

---

## 1. Health Check

### `GET /health`
Check service status and API version.

**Example Request:**
```bash
curl -X GET "http://127.0.0.1:8000/health"
```

**Example Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "service": "ML Classification System API"
}
```

---

## 2. Built-in Dataset Metadata

### `GET /api/v1/datasets/builtin`
Fetch schema, metadata, and positive class labels for the Breast Cancer Wisconsin dataset.

**Example Request:**
```bash
curl -X GET "http://127.0.0.1:8000/api/v1/datasets/builtin"
```

**Example Response (200 OK):**
```json
{
  "name": "Breast Cancer Wisconsin (Diagnostic)",
  "source": "scikit-learn (UCI Machine Learning Repository)",
  "samples_count": 569,
  "features_count": 30,
  "feature_names": [
    "mean radius", "mean texture", "mean perimeter", "mean area", ...
  ],
  "target_name": "target",
  "class_mapping": {
    "0": "malignant",
    "1": "benign"
  },
  "positive_class": 0,
  "positive_class_name": "malignant",
  "description": "Features are computed from a digitized image of a fine needle aspirate (FNA)...",
  "educational_disclaimer": "DISCLAIMER: This application and dataset are intended strictly for educational purposes..."
}
```

---

## 3. Dataset CSV Validation

### `POST /api/v1/datasets/validate`
Upload a CSV file for schema and guardrail validation.

**Example Request (multipart/form-data):**
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/datasets/validate" \
     -F "file=@sample_data.csv" \
     -F "target_column=target"
```

**Example Response (200 OK):**
```json
{
  "is_valid": true,
  "num_rows": 100,
  "num_cols": 5,
  "column_names": ["feat_1", "feat_2", "feat_3", "feat_4", "target"],
  "dtypes": {"feat_1": "float64", "target": "int64"},
  "missing_counts": {"feat_1": 0, "feat_2": 2},
  "missing_percentages": {"feat_1": 0.0, "feat_2": 2.0},
  "numeric_columns": ["feat_1", "feat_2", "feat_3", "feat_4", "target"],
  "categorical_columns": [],
  "constant_columns": [],
  "target_column": "target",
  "target_classes": [0, 1],
  "class_distribution": {"0": 50, "1": 50},
  "warnings": [],
  "errors": []
}
```

---

## 4. Model Registry Discovery

### `GET /api/v1/models`
Retrieve all registered algorithms and their valid parameter contracts.

**Example Request:**
```bash
curl -X GET "http://127.0.0.1:8000/api/v1/models"
```

---

## 5. Train Model

### `POST /api/v1/train`
Train a classifier on the built-in dataset or custom data records.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/train" \
     -H "Content-Type: application/json" \
     -d '{
       "use_builtin": true,
       "model_id": "logistic_regression",
       "hyperparameters": {"C": 1.0, "max_iter": 1000},
       "test_size": 0.20,
       "random_state": 42
     }'
```

**Example Response (200 OK):**
```json
{
  "artifact_id": "bda1c301-45c4-4687-bb6d-c2afb323ede9",
  "model_id": "logistic_regression",
  "display_name": "Logistic Regression",
  "hyperparameters": {"C": 1.0, "max_iter": 1000},
  "train_samples": 455,
  "test_samples": 114,
  "feature_count": 30,
  "supports_probabilities": true,
  "supports_decision_scores": true,
  "metrics": {
    "accuracy": 0.9825,
    "precision_macro": 0.9812,
    "recall_macro": 0.9812,
    "f1_macro": 0.9812,
    "precision_weighted": 0.9825,
    "recall_weighted": 0.9825,
    "f1_weighted": 0.9825,
    "confusion_matrix": [[41, 1], [1, 71]],
    "labels": ["0", "1"],
    "per_class_metrics": {
      "0": {"precision": 0.9762, "recall": 0.9762, "f1": 0.9762},
      "1": {"precision": 0.9861, "recall": 0.9861, "f1": 0.9861}
    },
    "support": {"0": 42, "1": 72},
    "is_binary": true,
    "positive_class": "0",
    "positive_class_name": "malignant",
    "positive_precision": 0.9762,
    "positive_recall": 0.9762,
    "positive_f1": 0.9762,
    "roc_auc": 0.9954
  },
  "warnings": []
}
```

---

## 6. Model Comparison

### `POST /api/v1/train/compare`
Train multiple algorithms on identical train/test partitions.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/train/compare" \
     -H "Content-Type: application/json" \
     -d '{
       "use_builtin": true,
       "model_ids": ["gaussian_nb", "logistic_regression", "linear_svm", "random_forest"],
       "test_size": 0.20,
       "random_state": 42
     }'
```

---

## 7. Prediction & Inference

### `POST /api/v1/predict`
Run predictions on new records using a fitted in-memory model artifact.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "artifact_id": "bda1c301-45c4-4687-bb6d-c2afb323ede9",
       "records": [
         {
           "mean radius": 17.99,
           "mean texture": 10.38,
           "mean perimeter": 122.8,
           "mean area": 1001.0,
           "mean smoothness": 0.1184,
           ... (all 30 features)
         }
       ]
     }'
```

**Example Response (200 OK):**
```json
{
  "artifact_id": "bda1c301-45c4-4687-bb6d-c2afb323ede9",
  "record_count": 1,
  "feature_count": 30,
  "predicted_classes": [0],
  "predicted_class_names": ["malignant"],
  "has_probabilities": true,
  "probabilities": [[0.9852, 0.0148]],
  "probability_class_labels": ["0 (malignant)", "1 (benign)"],
  "has_decision_scores": false,
  "decision_scores": null
}
```
