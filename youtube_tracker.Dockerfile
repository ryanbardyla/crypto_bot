# youtube_tracker.Dockerfile
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
COPY youtube_tracker.py /app/
COPY sentiment_analyzer.py /app/
COPY crypto_sentiment_analyzer.py /app/
COPY database_manager.py /app/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Run the tracker
CMD ["python", "youtube_tracker.py"]