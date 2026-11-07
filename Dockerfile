# ==============================================================================
# Multi-stage Production Dockerfile for Telco Customer Churn Prediction
# Serves both the FastAPI REST backend and the Streamlit Analytics dashboard
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Base Runtime Environment
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS base

# Security & Python optimization flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH="/app"

# Install system dependencies (curl for healthchecks, certificates)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-privileged system user and group for container security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Set working directory
WORKDIR /app

# Cache dependencies layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and model artifacts
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY models/best_churn_model.joblib ./models/best_churn_model.joblib
COPY models/model_comparison_report.json ./models/model_comparison_report.json

# Enforce secure non-root file ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root execution context
USER appuser

# ------------------------------------------------------------------------------
# Stage 2: Streamlit Frontend Dashboard Target
# ------------------------------------------------------------------------------
FROM base AS frontend

# Streamlit default application port
EXPOSE 8501

# Streamlit healthcheck probe
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit in headless server mode
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

# ------------------------------------------------------------------------------
# Stage 3: FastAPI Backend Inference Target (Default Target)
# ------------------------------------------------------------------------------
FROM base AS backend

# FastAPI default REST port
EXPOSE 8000

# API healthcheck probe
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI via production ASGI server Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
