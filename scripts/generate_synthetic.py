# File: scripts/generate_synthetic.py
import pandas as pd
import numpy as np
import random
random.seed(42)
np.random.seed(42)

N = 2000
courses = ['engineering','arts','business','science']
tuitions = ['paid','partial','unpaid']

rows = []
for _ in range(N):
    age = int(np.random.choice(range(17,31)))
    course = random.choice(courses)
    tuition = random.choice(tuitions)
    perf = float(np.clip(np.random.beta(5,2), 0, 1))
    attend = float(np.clip(np.random.beta(8,2), 0, 1))
    fails = int(np.random.poisson(0.3))
    aid = int(np.random.choice([0,1], p=[0.8,0.2]))
    risk = (1-perf)*0.6 + (1-attend)*0.3 + fails*0.1 + (tuition=='unpaid')*0.2 + aid*(-0.15)
    prob = 1/(1+np.exp(-4*(risk-0.4)))
    dropped = int(np.random.rand() < prob)
    rows.append([age, course, tuition, round(perf,3), round(attend,3), fails, aid, dropped])

df = pd.DataFrame(rows, columns=['age_at_enrollment','course','tuition_fee_status','semester_performance','attendance_rate','num_failed_courses','financial_aid','dropped_out'])
df.to_csv('examples/sample_students_large.csv', index=False)
print("Wrote examples/sample_students_large.csv, rows:", len(df))
