import io

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src import predict as predict_module


@pytest.fixture()
def client(trained, monkeypatch):
    monkeypatch.setenv("HOSPITAL_MODELS_DIR", str(trained["root"] / "models"))
    monkeypatch.setattr("api.main.SAMPLE_PATH", trained["root"] / "data" / "sample_patients.csv")
    predict_module._CACHE.clear()
    return TestClient(app)


def test_health_and_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True, "detail": None}
    info = client.get("/model-info").json()
    assert "test_metrics" in info and info["threshold"] == 0.5
    assert client.get("/features").json()["n_features"] == 30


def test_health_reports_missing_model(monkeypatch, tmp_path):
    monkeypatch.setenv("HOSPITAL_MODELS_DIR", str(tmp_path))
    predict_module._CACHE.clear()
    c = TestClient(app)
    assert c.get("/health").json()["model_loaded"] is False
    assert c.get("/model-info").status_code == 503


def test_sample_can_be_posted_straight_to_predict(client):
    mal = client.get("/sample?malignant=true").json()
    ben = client.get("/sample").json()
    assert mal["actual"] == "Malignant" and len(mal["features"]) == 30
    r = client.post("/predict", json=mal)                           # extra keys (id, actual) are ignored
    assert r.status_code == 200 and 0 <= r.json()["p_malignant"] <= 1
    assert client.post("/predict", json=ben).json()["p_benign"] + client.post("/predict", json=ben).json()["p_malignant"] > 0.99


def test_predict_threshold_override(client):
    mal, ben = client.get("/sample?malignant=true").json(), client.get("/sample").json()
    assert client.post("/predict?threshold=0", json=ben).json()["prediction"] == "Malignant"   # everything is flagged
    assert client.post("/predict?threshold=1", json=ben).json()["prediction"] == "Benign"      # nothing is flagged
    assert client.post("/predict?threshold=2", json=mal).status_code == 422


def test_predict_rejects_bad_input(client):
    mal = client.get("/sample?malignant=true").json()
    missing = {"features": {k: v for k, v in mal["features"].items() if k != "radius_mean"}}
    r = client.post("/predict", json=missing)
    assert r.status_code == 422 and "radius_mean" in r.text
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"features": {"radius_mean": "abc"}}).status_code == 422


def test_predict_batch(client):
    a, b = client.get("/sample?malignant=true").json(), client.get("/sample").json()
    r = client.post("/predict/batch", json={"patients": [a, b]})
    assert r.status_code == 200
    body = r.json()
    assert body["n_patients"] == 2 and len(body["results"]) == 2 and body["n_malignant"] >= 1


def test_predict_csv(client, trained):
    csv = (trained["root"] / "data" / "sample_patients.csv").read_bytes()
    r = client.post("/predict/csv", files={"file": ("p.csv", io.BytesIO(csv), "text/csv")})
    assert r.status_code == 200 and r.json()["n_patients"] == 114
    bad = client.post("/predict/csv", files={"file": ("p.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")})
    assert bad.status_code == 422 and "Missing columns" in bad.json()["detail"]
