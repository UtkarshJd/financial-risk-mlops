# Use lightweight Python 3.12 image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1

# Create non-root user
RUN useradd -m appuser && \
    mkdir /app && \
    chown appuser:appuser /app

# Install only essential system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER appuser
WORKDIR /app

# Copy requirements first for better layer caching
COPY --chown=appuser:appuser ./deployment/requirements.txt .

# Install Python packages
RUN pip install --upgrade pip && \
    pip install --user -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Set PATH to include user Python packages
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:5000/ || exit 1

# Optimized Gunicorn config
CMD ["gunicorn", \
    "--bind", "0.0.0.0:5000", \
    "--workers", "4", \
    "--threads", "2", \
    "--worker-class", "gthread", \
    "--timeout", "120", \
    "--access-logfile", "-", \
    "--error-logfile", "-", \
    "deployment.app:app"]
