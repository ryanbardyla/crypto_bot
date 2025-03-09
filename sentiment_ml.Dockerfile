# sentiment_ml.Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary files
COPY utils/ /app/utils/
COPY sentiment_ml.py /app/
COPY database_manager.py /app/

# Create necessary directories
RUN mkdir -p /app/ml_models /app/data /app/logs /app/config

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Run the service
CMD ["python", "sentiment_ml.py", "--schedule"]