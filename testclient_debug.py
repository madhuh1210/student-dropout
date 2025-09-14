# File: testclient_debug.py
# Use TestClient in two modes to reveal the server error:
# 1) No propagation: show response status/text
# 2) Propagate exceptions: raise_server_exceptions=True to see traceback

from fastapi.testclient import TestClient
from api.app import app

payload = {
  "age_at_enrollment": 20,
  "course": "engineering",
  "tuition_fee_status": "paid",
  "semester_performance": 0.75,
  "attendance_rate": 0.9,
  "num_failed_courses": 0,
  "financial_aid": 0
}

print("=== Mode A: do not propagate server exceptions (show response body) ===")
client = TestClient(app, raise_server_exceptions=False)
r = client.post("/predict", json=payload)
print("status_code ->", r.status_code)
try:
    print("response json ->", r.json())
except Exception:
    print("response text ->", r.text)

print("\n=== Mode B: propagate server exceptions (will raise if server errored) ===")
try:
    client2 = TestClient(app, raise_server_exceptions=True)
    r2 = client2.post("/predict", json=payload)
    print("status_code2 ->", r2.status_code)
    print("response json2 ->", r2.json())
except Exception as e:
    print("Exception raised by TestClient with propagation turned on:")
    import traceback
    traceback.print_exc()

