# File: api/app.py
"""
FastAPI app that loads pipeline.joblib at import/startup and serves /predict and /health.

Run:
    python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI, HTTPException
from api.schemas import PredictRequest, PredictResponse
from pathlib import Path
import joblib
import pandas as pd
import traceback

app = FastAPI(title="Student Dropout Risk API")

# Resolve model path relative to this file (repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "pipeline.joblib"

# Load model eagerly at import time so tests and TestClient always have it.
PIPE = None
try:
    if MODEL_PATH.exists():
        PIPE = joblib.load(MODEL_PATH)
        print(f"Loaded pipeline from {MODEL_PATH}")
    else:
        print(f"Model file not found at {MODEL_PATH}; PIPE remains None")
except Exception:
    PIPE = None
    print("Exception when loading pipeline at import time:")
    traceback.print_exc()

def compute_risk_score(row):
    """
    Risk formula must match the one used in src/prepare.py:
    risk_score = (1 - semester_performance)*0.6 + (1 - attendance_rate)*0.4 + (num_failed_courses * 0.1)
    """
    try:
        sp = float(row.get("semester_performance", 0.0))
        ar = float(row.get("attendance_rate", 0.0))
        nf = float(row.get("num_failed_courses", 0.0))
    except Exception:
        sp = 0.0
        ar = 0.0
        nf = 0.0
    return (1 - sp) * 0.6 + (1 - ar) * 0.4 + (nf * 0.1)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Build a DataFrame from the validated payload, compute derived features
    (risk_score) to match training-time preprocessing, then call the pipeline.
    """
    global PIPE
    if PIPE is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Build dict of input values. Use pydantic v2+ attribute 'model_fields' if available,
    # otherwise fall back to the pydantic v1 '__fields__' attribute for compatibility.
    fields = getattr(payload, "model_fields", None) or getattr(payload, "__fields__", None)
    if fields is None:
        # Very defensive fallback: attempt to use payload.__dict__
        data = {k: v for k, v in payload.__dict__.items() if not k.startswith("_")}
    else:
        data = {k: getattr(payload, k) for k in fields}

    # Compute derived feature expected by the pipeline
    data['risk_score'] = compute_risk_score(data)
    df = pd.DataFrame([data])
    try:
        proba = PIPE.predict_proba(df)[:,1][0]
        cls = int(proba >= 0.5)
    except Exception as e:
        # include the original exception message for debugging in development
        raise HTTPException(status_code=500, detail=str(e))
    return {"probability": float(proba), "class_": cls}
