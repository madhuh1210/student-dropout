# File: notebooks/interpretability.ipynb.py
"""
A runnable script-version of an interpretability notebook.
Generates SHAP summary plot and saves to models/shap_summary.png

Usage:
python notebooks/interpretability.ipynb.py --models_dir models --processed_dir processed
"""
import argparse
from pathlib import Path
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--processed_dir", default="processed")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    proc_dir = Path(args.processed_dir)
    pipe = joblib.load(models_dir / "pipeline.joblib")

    # load sample data (test)
    test = pd.read_csv(proc_dir / "test.csv")
    features = pipe.named_steps['preproc'].transformers_[0][2] + pipe.named_steps['preproc'].transformers_[1][2]
    X_test = test[features]

    # We need to get a transformed matrix for SHAP's TreeExplainer. Use preproc transform.
    X_trans = pipe.named_steps['preproc'].transform(X_test)
    model = pipe.named_steps['clf']  # LightGBM model

    # Use SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)
    # shap_values for binary classification: list of two arrays or single array depending on shap version
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Feature names must match transformed features: combine numeric + onehot names
    # recreate feature names
    num_cols = pipe.named_steps['preproc'].transformers_[0][2]
    cat_pipe = pipe.named_steps['preproc'].transformers_[1][1]
    cat_cols = pipe.named_steps['preproc'].transformers_[1][2]
    # get OHE feature names
    ohe = cat_pipe.named_steps['onehot']
    try:
        ohe_features = list(ohe.get_feature_names_out(cat_cols))
    except Exception:
        # fallback
        ohe_features = [f"{c}_{i}" for c in cat_cols for i in range(2)]
    feature_names = list(num_cols) + ohe_features

    # Summary plot
    out = models_dir / "shap_summary.png"
    plt.figure(figsize=(6,4))
    shap.summary_plot(shap_vals, X_trans, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print("Saved SHAP summary to", out)

if __name__ == "__main__":
    main()
