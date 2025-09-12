# Railway-optimized Dockerfile for Flask PPG Health Prediction Suite
# Explicitly use linux/amd64 platform for Railway compatibility
FROM --platform=linux/amd64 python:3.10-slim

# Set environment variables for Python optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgthread-2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY flask_requirements.txt requirements.txt

# Install Python dependencies
# Force reinstall to ensure AMD64 compatibility
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --force-reinstall -r requirements.txt

# Copy application code
COPY app.py .
COPY generate_ml_models.py .
COPY templates/ ./templates/
COPY static/ ./static/
COPY core/ ./core/
COPY models/ ./models/
COPY external/ ./external/

# Create necessary directories
RUN mkdir -p /app/models && \
    mkdir -p /app/static/uploads && \
    mkdir -p /app/logs

# Generate ML models if they don't exist
RUN python generate_ml_models.py || echo "Models already exist or generation optional"

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:${PORT}/health || exit 1

# Railway provides PORT at runtime, not build time
EXPOSE 5000

# Run with gunicorn for production with Railway-specific configuration
# Railway sets PORT environment variable at runtime
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 120 --access-logfile - --error-logfile - --log-level info app:app"]