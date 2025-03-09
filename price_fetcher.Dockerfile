# price_fetcher.Dockerfile
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
COPY multi_api_price_fetcher.py /app/
COPY database_manager.py /app/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Add a wrapper script to run the price fetcher at regular intervals
COPY scripts/run_price_fetcher.py /app/

# Run the wrapper script
CMD ["python", "run_price_fetcher.py"]
