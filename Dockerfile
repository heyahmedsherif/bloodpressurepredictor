# Lightweight Railway Dockerfile for PaPaGei Blood Pressure Predictor
# Optimized for minimal size and Railway free tier

FROM python:3.10-slim

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y --auto-remove

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && pip cache purge

# Copy only necessary application files
COPY streamlit_app.py .
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Create streamlit config if not exists
RUN mkdir -p .streamlit && \
    echo '[server]\nport = 8080\naddress = "0.0.0.0"\nheadless = true\n[browser]\ngatherUsageStats = false\n[theme]\nprimaryColor = "#FF6B6B"\nbackgroundColor = "#FFFFFF"\nsecondaryBackgroundColor = "#F0F2F6"\ntextColor = "#262730"' > .streamlit/config.toml

# Expose port
EXPOSE 8080

# Run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]