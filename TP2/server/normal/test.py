from fastapi.testclient import TestClient
from main import app

MOCK_FEATURES = [[5.1, 3.5, 1.4, 0.2], [6.2, 2.8, 4.8, 1.8]]
MOCK_MODEL_NAME = "tracking-iris-model"
MOCK_MODEL_VERSION = "1"

client = TestClient(app)

def test_predict_endpoint_success():
    response = client.post(
        "/update-model",
        json={"model_name": MOCK_MODEL_NAME, "model_version": "latest"}
    )
    assert response.status_code == 200, f"Model load failed: {response.json()}"

    response = client.post(
        "/predict",
        json={"features": MOCK_FEATURES}
    )
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) == len(MOCK_FEATURES)

def test_update_model_endpoint_success():
    response = client.post(
        "/update-model",
        json={"model_name": MOCK_MODEL_NAME, "model_version": MOCK_MODEL_VERSION}
    )
    assert response.status_code == 200
    assert f"version: {MOCK_MODEL_VERSION}" in response.json()["status"]

def test_update_model_endpoint_failure():
    response = client.post(
        "/update-model",
        json={"model_name": "nonexistent-model", "model_version": "999"}
    )
    assert response.status_code == 500
