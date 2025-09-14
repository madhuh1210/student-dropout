# File: tests/test_prepare.py
"""
Unit tests for prepare module.
"""
import pandas as pd
from src.prepare import preprocess_basic, stratified_splits
import numpy as np

def test_preprocess_basic_creates_risk_score():
    df = pd.DataFrame({
        "age_at_enrollment": [18,20],
        "course": ["a","b"],
        "tuition_fee_status": ["paid","partial"],
        "semester_performance": [0.9, 0.4],
        "attendance_rate": [0.95, 0.6],
        "num_failed_courses": [0,2],
        "financial_aid": [0,1],
        "dropped_out": [0,1]
    })
    df2 = preprocess_basic(df)
    assert "risk_score" in df2.columns
    assert df2['risk_score'].dtype == float or np.issubdtype(df2['risk_score'].dtype, np.floating)

def test_stratified_splits_balance():
    df = pd.DataFrame({
        "age_at_enrollment": list(range(100)),
        "course": ["c"]*100,
        "tuition_fee_status": ["paid"]*100,
        "semester_performance": [0.7]*100,
        "attendance_rate": [0.9]*100,
        "num_failed_courses": [0]*100,
        "financial_aid": [0]*100,
        "dropped_out": ([0]*80 + [1]*20)
    })
    train, val, test = stratified_splits(df, target='dropped_out', random_state=42)
    # expect roughly same ratios
    def ratio(x): return x['dropped_out'].mean()
    assert abs(ratio(train) - 0.2) < 0.05
    assert abs(ratio(val) - 0.2) < 0.1
    assert abs(ratio(test) - 0.2) < 0.1
