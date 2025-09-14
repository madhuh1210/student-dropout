# File: Dockerfile
# Multi-stage build: build wheels in builder stage, run app in lightweight runtime stage
# Python pinned to 3.10.x (matching your local dev). Adjust if you use a different minor version.

#################################
# Builder stage (build wheels)
#################################
FROM python:3.10.12-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools needed to build wheels (LightGBM may require gcc, cmake, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

COPY requirements.txt .

# Upgrade pip and build wheels for faster runtime installation
RUN python -m pip install --upgrade pip wheel setuptools
RUN python -m pip wheel --wheel-dir=/install/wheels -r requirements.txt

#################################
# Runtime stage
#################################
FROM python:3.10.12-slim

# Create non-root runtime user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy prebuilt wheels and requirements into runtime image
COPY --from=builder /install/wheels /wheels
COPY requirements.txt /app/requirements.txt

# Try to install from wheels first (no-index). If any wheel missing, install from PyPI as fallback.
RUN python -m pip install --upgrade pip \
 && python -m pip install --no-index --find-links=/wheels -r /app/requirements.txt || true \
 && python -m pip install --upgrade pip \
 && python -m pip install -r /app/requirements.txt

# Copy project files (source code). Do not overwrite /models if you mount it at runtime.
COPY . /app

# Ensure models dir exists inside container (so mounting works). If you want to include trained models in the image,
# remove 'models' from .dockerignore and ensure models/pipeline.joblib exists before building.
RUN mkdir -p /app/models && chown -R appuser:appuser /app

USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"

EXPOSE 8000

# Default command: run uvicorn
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
