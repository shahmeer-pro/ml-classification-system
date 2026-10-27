"""Tests for training, dataset metadata, and model comparison endpoints."""

from fastapi.testclient import TestClient


def test_builtin_dataset_endpoint(api_client: TestClient):
    response = api_client.get("/api/v1/datasets/builtin")
    assert response.status_code == 200
    data = response.json()
    assert data["samples_count"] == 569
    assert data["features_count"] == 30
    assert data["target_name"] == "target"
    assert "educational_disclaimer" in data


def test_train_model_endpoint_builtin(api_client: TestClient):
    payload = {
        "use_builtin": True,
        "model_id": "logistic_regression",
        "hyperparameters": {"C": 1.0, "max_iter": 500},
        "test_size": 0.20,
        "random_state": 42,
    }
    response = api_client.post("/api/v1/train", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "artifact_id" in data
    assert data["model_id"] == "logistic_regression"
    assert data["metrics"]["accuracy"] > 0.85
    assert data["train_samples"] == 455
    assert data["test_samples"] == 114


def test_compare_models_endpoint(api_client: TestClient):
    payload = {
        "use_builtin": True,
        "model_ids": ["gaussian_nb", "logistic_regression", "linear_svm"],
        "test_size": 0.20,
        "random_state": 42,
    }
    response = api_client.post("/api/v1/train/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 3
    assert len(data["comparison_table"]) == 3
    assert data["test_samples"] == 114


def test_train_invalid_model_returns_422(api_client: TestClient):
    payload = {
        "use_builtin": True,
        "model_id": "nonexistent_model_id",
    }
    response = api_client.post("/api/v1/train", json=payload)
    assert response.status_code == 422
    assert "error" in response.json()
