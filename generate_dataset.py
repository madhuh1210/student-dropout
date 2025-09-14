# File: generate_dataset.py
"""
Generate a larger synthetic dataset of student records for dropout prediction.
Outputs examples/sample_students.csv with ~5000 rows.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N = 5000
courses = ["engineering", "arts", "business", "science", "law"]
tuition_status = ["paid", "partial", "unpaid"]

ages = np.random.randint(17, 30, size=N)
course_choices = np.random.choice(courses, size=N, p=[0.3, 0.2, 0.2, 0.2, 0.1])
tuition_choices = np.random.choice(tuition_status, size=N, p=[0.5, 0.3, 0.2])

semester_perf = np.clip(np.random.normal(0.7, 0.15, size=N), 0, 1)
attendance = np.clip(np.random.normal(0.85, 0.1, size=N), 0, 1)
failed_courses = np.random.poisson(0.5, size=N)
financial_aid = np.random.choice([0, 1], size=N, p=[0.7, 0.3])

# True dropout probability (synthetic rule)
risk_score = (1 - semester_perf) * 0.6 + (1 - attendance) * 0.4 + (failed_courses * 0.05)
base_prob = np.clip(risk_score + 0.1 * (tuition_choices == "unpaid"), 0, 1)
dropped_out = np.random.binomial(1, base_prob)

df = pd.DataFrame({
    "age_at_enrollment": ages,
    "course": course_choices,
    "tuition_fee_status": tuition_choices,
    "semester_performance": semester_perf.round(2),
    "attendance_rate": attendance.round(2),
    "num_failed_courses": failed_courses,
    "financial_aid": financial_aid,
    "dropped_out": dropped_out,
})

out_path = Path("examples") / "sample_students.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print(f"Generated dataset with shape {df.shape} -> {out_path}")
print(df.head())
