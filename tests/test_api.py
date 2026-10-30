"""FastAPI surface: /health and /deidentify via TestClient."""

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_deidentify_label_mode():
    res = client.post(
        "/deidentify",
        json={"text": "Patient: John Smith\nMRN: 8473625", "mode": "label"},
    )
    assert res.status_code == 200
    body = res.json()
    assert "John Smith" not in body["deidentified_text"]
    assert "[PERSON]" in body["deidentified_text"]
    assert any(e["entity_type"] == "MEDICAL_RECORD_NUMBER" for e in body["entities"])


def test_deidentify_bad_mode():
    res = client.post("/deidentify", json={"text": "hello", "mode": "nope"})
    assert res.status_code == 200
    assert "error" in res.json()
