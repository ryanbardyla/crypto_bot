# trading_bot.Dockerfile
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
COPY live_trader.py /app/
COPY paper_trader.py /app/
COPY enhanced_strategy.py /app/
COPY crypto_analyzer.py /app/
COPY hyperliquid_api.py /app/
COPY multi_api_price_fetcher.py /app/
COPY run_bot.py /app/
COPY database_manager.py /app/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config /app/paper_trading

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Run the trading bot
CMD ["python", "run_bot.py"]