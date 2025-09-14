# File: src/train.py
"""
Train LightGBM pipeline.
- Loads processed train/val/test
- Builds sklearn ColumnTransformer + SimpleImputer + OneHotEncoder/StandardScaler pipeline
- Trains LightGBM with class_weight to handle imbalance
- Runs RandomizedSearchCV for a small param grid
- Saves pipeline.joblib and model_lgbm.joblib to models_dir
- Writes training_metrics.json

Usage:
python -m src.train --processed_dir processed --models_dir models --random_state 42
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import lightgbm as lgb
import joblib
import json
from src.utils import ensure_dir, save_json

RSEED = 42

# Build the preprocessing + estimator pipeline
def build_pipeline(feature_config):
    num_cols = feature_config['numerical']
    cat_cols = feature_config['categorical']

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

    ])

    preproc = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
    ], remainder='drop')

    clf = lgb.LGBMClassifier(random_state=RSEED, n_jobs=-1)

    pipe = Pipeline([
        ('preproc', preproc),
        ('clf', clf)
    ])
    return pipe

# Load processed CSV splits
def load_csvs(processed_dir):
    p = Path(processed_dir)
    train = pd.read_csv(p / "train.csv")
    val = pd.read_csv(p / "val.csv")
    test = pd.read_csv(p / "test.csv")
    return train, val, test

# Run a small randomized search over hyperparameters
def train_with_search(pipe, X, y, random_state=RSEED):
    param_dist = {
        'clf__num_leaves': [15, 31, 63],
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__n_estimators': [50, 100, 200],
        'clf__subsample': [0.6, 0.8, 1.0],
        'clf__colsample_bytree': [0.6, 0.8, 1.0]
    }
    rs = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=8, scoring='roc_auc',
                            cv=3, verbose=1, random_state=random_state, n_jobs=1)
    rs.fit(X, y)
    return rs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="processed")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--random_state", type=int, default=RSEED)
    args = parser.parse_args()

    train, val, test = load_csvs(args.processed_dir)
    feature_config = {
        "target": "dropped_out",
        "numerical": ["age_at_enrollment", "semester_performance", "attendance_rate", "num_failed_courses", "risk_score"],
        "categorical": ["course", "tuition_fee_status", "financial_aid"]
    }

    X_train = train[feature_config['numerical'] + feature_config['categorical']]
    y_train = train[feature_config['target']]
    X_val = val[feature_config['numerical'] + feature_config['categorical']]
    y_val = val[feature_config['target']]

    # Handle class imbalance via class_weight
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}
    print("Computed class weights:", class_weight_dict)

    pipe = build_pipeline(feature_config)

    # Set class_weight on the LightGBM classifier inside the pipeline
    pipe.named_steps['clf'].set_params(class_weight=class_weight_dict)

    # Randomized search (may take time on larger data)
    rs = train_with_search(pipe, X_train, y_train, random_state=args.random_state)
    best = rs.best_estimator_
    print("Best params:", rs.best_params_)

    # Evaluate on validation set
    y_pred_proba = best.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_val, y_pred_proba)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)

    metrics = {
        "validation_auc": float(auc),
        "validation_accuracy": float(acc),
        "classification_report": report,
        "best_params": rs.best_params_
    }

    models_dir = Path(args.models_dir)
    ensure_dir(models_dir)
    joblib.dump(best, models_dir / "pipeline.joblib")
    joblib.dump(best.named_steps['clf'], models_dir / "model_lgbm.joblib")
    save_json(metrics, models_dir / "training_metrics.json")
    print("Saved pipeline and model to", models_dir)

if __name__ == "__main__":
    main()
