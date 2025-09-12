#!/bin/bash
# Railway startup script with debugging

echo "Starting Flask app on Railway..."
echo "PORT environment variable: $PORT"

if [ -z "$PORT" ]; then
    echo "ERROR: PORT environment variable is not set!"
    echo "Using default port 5000"
    export PORT=5000
fi

echo "Starting gunicorn on port $PORT..."

# Start gunicorn with explicit error handling
exec gunicorn \
    --bind "0.0.0.0:$PORT" \
    --workers 1 \
    --threads 2 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level debug \
    --preload \
    app:app