# File: Dockerfile
# Stage 1: build image with dependencies
FROM python:3.10.12-slim as builder

# set non-root user
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r /app/requirements.txt

# Stage 2: final runtime
FROM python:3.10.12-slim

# create non-root user
RUN useradd -m appuser
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels -r /app/requirements.txt || pip install --no-index --find-links=/wheels -r /app/requirements.txt

# Copy project files
COPY . /app
# set ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
ENV MODEL_PATH=/app/models/pipeline.joblib

# Default command: run uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
