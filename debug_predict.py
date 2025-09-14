# File: debug_predict.py
# small script to reproduce pipeline prediction and print any exception
import joblib, pandas as pd, traceback

def compute_risk_score(row):
    try:
        sp = float(row.get("semester_performance", 0.0))
        ar = float(row.get("attendance_rate", 0.0))
        nf = float(row.get("num_failed_courses", 0.0))
    except Exception:
        sp = 0.0; ar = 0.0; nf = 0.0
    return (1 - sp) * 0.6 + (1 - ar) * 0.4 + (nf * 0.1)

def main():
    print("Loading pipeline from models/pipeline.joblib")
    pipe = joblib.load("models/pipeline.joblib")
    payload = {
        "age_at_enrollment": 20,
        "course": "engineering",
        "tuition_fee_status": "paid",
        "semester_performance": 0.75,
        "attendance_rate": 0.9,
        "num_failed_courses": 0,
        "financial_aid": 0
    }
    payload['risk_score'] = compute_risk_score(payload)
    print("Prepared payload dict:", payload)
    df = pd.DataFrame([payload])
    print("DataFrame columns:", list(df.columns))
    print("DataFrame dtypes:")
    print(df.dtypes)
    # show first rows
    print(df.head().to_dict(orient="records"))
    try:
        proba = pipe.predict_proba(df)[:,1]
        print("predict_proba output:", proba)
    except Exception as e:
        print("Exception when calling pipeline.predict_proba():")
        traceback.print_exc()

if __name__ == "__main__":
    main()
