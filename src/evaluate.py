# File: src/evaluate.py
"""
Evaluate trained pipeline on validation and test sets and save sample predictions.

Usage:
python src/evaluate.py --models_dir models --processed_dir processed
"""
import argparse
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--processed_dir", default="processed")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    proc_dir = Path(args.processed_dir)
    pipe = joblib.load(models_dir / "pipeline.joblib")

    val = pd.read_csv(proc_dir / "val.csv")
    test = pd.read_csv(proc_dir / "test.csv")
    features = pipe.named_steps['preproc'].transformers_[0][2] + pipe.named_steps['preproc'].transformers_[1][2]

    X_val = val[features]
    y_val = val['dropped_out']
    X_test = test[features]
    y_test = test['dropped_out']

    yv_proba = pipe.predict_proba(X_val)[:,1]
    yv = (yv_proba >= 0.5).astype(int)
    yt_proba = pipe.predict_proba(X_test)[:,1]
    yt = (yt_proba >= 0.5).astype(int)

    metrics = {
        "val_auc": float(roc_auc_score(y_val, yv_proba)),
        "val_accuracy": float(accuracy_score(y_val, yv)),
        "val_report": classification_report(y_val, yv),
        "test_auc": float(roc_auc_score(y_test, yt_proba)),
        "test_accuracy": float(accuracy_score(y_test, yt)),
        "test_report": classification_report(y_test, yt)
    }
    print(json.dumps(metrics, indent=2))

    # save sample preds
    out = models_dir / "sample_preds.csv"
    sample = test.copy()
    sample['pred_proba'] = yt_proba
    sample['pred_class'] = yt
    sample.to_csv(out, index=False)
    print("Saved sample predictions to", out)

if __name__ == "__main__":
    main()
