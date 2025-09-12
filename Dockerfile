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

# Railway uses PORT environment variable (usually 8080)
EXPOSE 8080

# Use optimized gunicorn command for WebRTC and Railway compatibility
# Increased timeout and keepalive for long-running WebRTC connections
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 180 --keepalive 5 --max-requests 1000 --worker-tmp-dir /dev/shm --access-logfile - --error-logfile - --log-level info --worker-class sync app:app"]