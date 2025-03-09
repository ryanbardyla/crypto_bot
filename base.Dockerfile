# base.Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command (will be overridden by specific services)
CMD ["python", "app.py"]