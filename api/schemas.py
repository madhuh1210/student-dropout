# File: api/schemas.py
"""
Pydantic request/response schemas for API.
"""
from pydantic import BaseModel, conint, confloat
from typing import Literal

class PredictRequest(BaseModel):
    age_at_enrollment: conint(ge=15, le=90)
    course: str
    tuition_fee_status: str
    semester_performance: confloat(ge=0.0, le=1.0)
    attendance_rate: confloat(ge=0.0, le=1.0)
    num_failed_courses: conint(ge=0, le=20)
    financial_aid: conint(ge=0, le=1)

class PredictResponse(BaseModel):
    probability: float
    class_: int
