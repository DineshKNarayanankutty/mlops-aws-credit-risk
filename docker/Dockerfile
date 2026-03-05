# --- Stage 1: Builder ---
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build dependencies if any packages require C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies into a local directory
COPY inference/requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt


# --- Stage 2: Final Runtime ---
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:${PATH}"

WORKDIR /app

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy only the installed python packages from the builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code and model
COPY --chown=appuser:appuser inference/ /app/
COPY --chown=appuser:appuser model/ /app/model/

# Set ownership and switch to non-root user
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
