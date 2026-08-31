
from fastapi.testclient import TestClient

from pbsite.serve.api import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_503_without_model(monkeypatch):
    # ensure no model configured -> graceful 503, not a crash
    monkeypatch.delenv("PBSITE_MODEL_DIR", raising=False)
    import pbsite.serve.api as api

    api._predictor = None
    r = client.post("/predict", json={"sequence": "MKTAYIAK"})
    assert r.status_code == 503


def test_predict_rejects_bad_sequence():
    r = client.post("/predict", json={"sequence": "MKTZZZ123"})
    assert r.status_code == 422  # pydantic validation error
