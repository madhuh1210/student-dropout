from pathlib import Path
import os

p = Path('models') / 'pipeline.joblib'
print('models/pipeline.joblib exists ->', p.exists())
print('models dir listing:')
print([f for f in os.listdir('models')])
