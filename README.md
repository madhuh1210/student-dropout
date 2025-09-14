# Student Dropout Prediction – End-to-End ML Project

![CI](https://github.com/madhuh1210/student-dropout/actions/workflows/ci.yml/badge.svg)

## 📌 Overview
This project implements an **end-to-end machine learning pipeline** to predict whether a student is at risk of dropping out.  
It covers the full lifecycle: data preparation, model training, evaluation, interpretability, API deployment, containerization, and testing.

**Key features:**
- **Data pipeline**: ingestion, preprocessing, stratified splits, and feature engineering (`risk_score`).
- **Model training**: LightGBM classifier with class imbalance handling and hyperparameter search.
- **Interpretability**: SHAP values for feature importance visualization.
- **API**: FastAPI service (`/predict`, `/health`) to serve predictions.
- **Docker**: Production-ready container image with non-root user.
- **Testing & CI**: pytest unit/integration tests, GitHub Actions CI.

---

## 📂 Repo Structure
.
├── src/ # Data ingestion, preparation, training, evaluation
├── api/ # FastAPI app (app.py, schemas.py)
├── notebooks/ # SHAP interpretability script
├── models/ # Saved artifacts (pipeline.joblib, metrics, SHAP plots)
├── tests/ # Unit + integration tests (pytest)
├── examples/ # Sample dataset
├── Dockerfile # Multi-stage container build
├── requirements.txt # Pinned dependencies
└── README.md


---

## 🚀 Quickstart (Local)

### 1. Clone & setup environment
```powershell
git clone https://github.com/madhuh1210/student-dropout.git
cd student-dropout

# create virtualenv
python -m venv .venv
.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt

Run preprocessing
python -m src.prepare --input_csv .\examples\sample_students.csv --out_dir .\processed --random_state 42

Train model
python -m src.train --processed_dir .\processed --models_dir .\models --random_state 42

Run tests
python -m pytest -q

Start API
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload


Visit: http://127.0.0.1:8000/docs


[ci-trigger] 2025-09-14T23:34:16.7170644+05:30
