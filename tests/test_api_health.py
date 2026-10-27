"""Tests for API health endpoint."""

from fastapi.testclient import TestClient


def test_api_health_check(api_client: TestClient):
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert data["service"] == "ML Classification System API"


def test_api_models_endpoint(api_client: TestClient):
    response = api_client.get("/api/v1/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    model_ids = [m["model_id"] for m in models]
    assert "logistic_regression" in model_ids
    assert "gaussian_nb" in model_ids
    assert "linear_svm" in model_ids
