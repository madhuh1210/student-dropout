# File: src/prepare.py
"""
Prepare pipeline:
- Reads raw CSV
- Simple processing & feature engineering
- Encodes categoricals with ordinal/one-hot via sklearn ColumnTransformer
- Splits stratified into train/val/test (70/15/15) with safe fallback for tiny datasets
- Saves split CSVs and feature config.

Usage:
python -m src.prepare --input_csv examples/sample_students.csv --out_dir processed --random_state 42
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from src.utils import ensure_dir, save_json

RSEED = 42

def preprocess_basic(df):
    # Basic feature cleaning & engineering
    df = df.copy()
    # Ensure types
    df['course'] = df['course'].astype(str)
    df['tuition_fee_status'] = df['tuition_fee_status'].astype(str)
    # Clip numeric ranges defensively
    if 'attendance_rate' in df.columns:
        df['attendance_rate'] = df['attendance_rate'].clip(0,1)
    if 'semester_performance' in df.columns:
        df['semester_performance'] = df['semester_performance'].clip(0,1)
    # small synthetic feature: risk_score (simple)
    df['risk_score'] = (1 - df.get('semester_performance', 0)) * 0.6 + (1 - df.get('attendance_rate', 0)) * 0.4 + (df.get('num_failed_courses', 0) * 0.1)
    return df

def stratified_splits(df, target='dropped_out', random_state=RSEED):
    """
    Do a 70/15/15 split with stratification where possible.
    If stratification for the second split (temp -> val/test) would fail
    because of too-few examples per class, fall back to a random split.
    """
    # 1) First split: train (70%) and temp (30%) - try stratify, fallback to random
    try:
        train, temp = train_test_split(df, test_size=0.30, stratify=df[target], random_state=random_state)
    except Exception:
        train, temp = train_test_split(df, test_size=0.30, stratify=None, random_state=random_state)

    # 2) Second split: split temp into val and test equally (15% each overall)
    counts = Counter(temp[target]) if target in temp.columns else Counter()
    # If any class has fewer than 2 samples in temp, do non-stratified split
    if counts and min(counts.values()) < 2:
        val, test = train_test_split(temp, test_size=0.5, stratify=None, random_state=random_state)
    else:
        # If counts is empty (no target) or enough samples, attempt stratify if possible
        if target in temp.columns:
            val, test = train_test_split(temp, test_size=0.5, stratify=temp[target], random_state=random_state)
        else:
            val, test = train_test_split(temp, test_size=0.5, stratify=None, random_state=random_state)

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_dir", default="processed")
    parser.add_argument("--random_state", type=int, default=RSEED)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    # quick validation
    if 'dropped_out' not in df.columns:
        raise ValueError("Input CSV must contain 'dropped_out' column.")

    df = preprocess_basic(df)
    train, val, test = stratified_splits(df, random_state=args.random_state)

    out = Path(args.out_dir)
    ensure_dir(out)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    feature_config = {
        "target": "dropped_out",
        "numerical": ["age_at_enrollment", "semester_performance", "attendance_rate", "num_failed_courses", "risk_score"],
        "categorical": ["course", "tuition_fee_status", "financial_aid"]
    }
    save_json(feature_config, out / "feature_config.json")
    print("Saved processed splits to", out)

if __name__ == "__main__":
    main()
