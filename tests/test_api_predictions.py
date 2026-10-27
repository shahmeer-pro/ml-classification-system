"""Tests for inference endpoint and prediction validation."""

from fastapi.testclient import TestClient


def test_prediction_workflow(api_client: TestClient):
    # 1. Train a model first
    train_payload = {
        "use_builtin": True,
        "model_id": "logistic_regression",
        "test_size": 0.20,
        "random_state": 42,
    }
    train_resp = api_client.post("/api/v1/train", json=train_payload)
    assert train_resp.status_code == 200
    artifact_id = train_resp.json()["artifact_id"]

    # 2. Get builtin dataset feature names to formulate a valid test record
    meta_resp = api_client.get("/api/v1/datasets/builtin")
    feature_names = meta_resp.json()["feature_names"]

    # Form a dummy record with 0.5 for all 30 features
    record = {f: 0.5 for f in feature_names}

    pred_payload = {
        "artifact_id": artifact_id,
        "records": [record],
    }

    pred_resp = api_client.post("/api/v1/predict", json=pred_payload)
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    assert pred_data["record_count"] == 1
    assert len(pred_data["predicted_classes"]) == 1
    assert pred_data["has_probabilities"] is True
    assert len(pred_data["probabilities"][0]) == 2


def test_prediction_with_invalid_artifact_id_returns_404(api_client: TestClient):
    pred_payload = {
        "artifact_id": "00000000-0000-0000-0000-000000000000",
        "records": [{"feat_1": 1.0}],
    }
    resp = api_client.post("/api/v1/predict", json=pred_payload)
    assert resp.status_code == 404
    assert "error" in resp.json()


def test_prediction_with_missing_features_returns_422(api_client: TestClient):
    train_resp = api_client.post(
        "/api/v1/train", json={"use_builtin": True, "model_id": "gaussian_nb"}
    )
    artifact_id = train_resp.json()["artifact_id"]

    # Incomplete records missing required 30 features
    pred_payload = {
        "artifact_id": artifact_id,
        "records": [{"only_one_feature": 1.23}],
    }
    resp = api_client.post("/api/v1/predict", json=pred_payload)
    assert resp.status_code == 422
    assert "Missing" in resp.json()["message"]
