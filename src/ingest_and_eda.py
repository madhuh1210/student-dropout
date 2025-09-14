# File: src/ingest_and_eda.py
"""
Load CSV and print basic EDA. Saves basic summary JSON to out_dir/eda_summary.json
Usage:
  python src/ingest_and_eda.py --csv examples/sample_students.csv --out_dir data_output
"""
import argparse
import pandas as pd
from pathlib import Path
from src.utils import ensure_dir, save_json

def basic_eda(df):
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "head": df.head(3).to_dict(orient="records"),
        "class_balance": df['dropped_out'].value_counts().to_dict()
    }
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", default="data_output")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print("=== HEAD ===")
    print(df.head())
    print("\n=== CLASS BALANCE ===")
    print(df['dropped_out'].value_counts())

    summary = basic_eda(df)
    out = Path(args.out_dir)
    ensure_dir(out)
    save_json(summary, out / "eda_summary.json")
    print(f"\nSaved EDA summary to {out/'eda_summary.json'}")

if __name__ == "__main__":
    main()
