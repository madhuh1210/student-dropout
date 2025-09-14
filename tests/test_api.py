# File: tests/test_api.py
"""
Integration test for API predict endpoint.
This test assumes models/pipeline.joblib exists (create via running train.py before tests).
"""
from fastapi.testclient import TestClient
from api.app import app
import json

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict():
    payload = {
        "age_at_enrollment": 20,
        "course": "engineering",
        "tuition_fee_status": "paid",
        "semester_performance": 0.75,
        "attendance_rate": 0.9,
        "num_failed_courses": 0,
        "financial_aid": 0
    }
    r = client.post("/predict", json=payload)
    # If model missing, TestClient will raise runtime error at startup—so ensure models exist first
    assert r.status_code == 200
    data = r.json()
    assert "probability" in data and "class_" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert data["class_"] in (0,1)
