# Railway-optimized Dockerfile for Flask PPG Health Prediction Suite
FROM python:3.10-slim

# Set environment variables for Python optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=5000

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
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

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

# Use PORT environment variable from Railway
EXPOSE ${PORT}

# Run with gunicorn for production with Railway-specific configuration
# Using exec form for better signal handling
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 2 --timeout 120 --access-logfile - --error-logfile - --log-level debug app:app"]