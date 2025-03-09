import os
import sys
import time
import socket
import random
import json
import math
import numpy as np
import requests
import traceback
import logging
import threading
import psutil
import gc
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from prometheus_client import start_http_server, Counter, Gauge, Summary, Histogram
import pika
import argparse

# Prometheus metrics
API_REQUESTS = Counter('price_fetcher_api_requests_total', 'Number of API requests', ['api', 'symbol', 'status'])
PRICE_UPDATES = Counter('price_fetcher_updates_total', 'Number of price updates', ['symbol', 'source'])
RATE_LIMIT_WAITS = Counter('price_fetcher_rate_limit_waits_total', 'Number of rate limit waits', ['api'])
MOCK_PRICE_USES = Counter('price_fetcher_mock_price_uses_total', 'Number of times mock price was used', ['symbol', 'reason'])
CURRENT_PRICES = Gauge('price_fetcher_current_price', 'Current price of cryptocurrency', ['symbol'])
PRICE_AGE_SECONDS = Gauge('price_fetcher_price_age_seconds', 'Age of price data in seconds', ['symbol'])
API_AVAILABILITY = Gauge('price_fetcher_api_availability', 'API availability status (1=up, 0=down)', ['api'])
PRICE_CHANGE_PCT = Gauge('price_fetcher_price_change_percent', 'Price change percentage in last period', ['symbol', 'period'])
REQUEST_DURATION = Summary('price_fetcher_request_duration_seconds', 'Duration of API requests', ['api'])
PRICE_DISTRIBUTION = Histogram('price_fetcher_price_distribution', 'Distribution of fetched prices', 
                               ['symbol'], buckets=[0.1, 1, 10, 100, 1000, 10000, 100000])

# Import logging utilities
try:
    from utils.logging_config import get_module_logger
    logger = get_module_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, calls_per_second=1, burst_limit=None):
        self.calls_per_second = calls_per_second
        self.burst_limit = burst_limit
        self.lock = threading.Lock()
        self.calls_since_reset = 0
        self.last_call = time.time()
        self.reset_time = time.time() + 1.0  # Reset counter after 1 second
    
    def wait(self, api_name="unknown"):
        with self.lock:
            if self.burst_limit is not None:
                current_time = time.time()
                if current_time >= self.reset_time:
                    self.calls_since_reset = 0
                    self.reset_time = current_time + 1.0
                
                if self.calls_since_reset >= self.burst_limit:
                    sleep_time = self.reset_time - current_time
                    if sleep_time > 0:
                        RATE_LIMIT_WAITS.labels(api_name).inc()
                        logger.debug(f"Rate limit hit for {api_name}, waiting {sleep_time:.2f}s")
                        time.sleep(max(0, sleep_time))
                    self.calls_since_reset = 0
                    self.reset_time = time.time() + 1.0
                
                self.calls_since_reset += 1
            
            elapsed = time.time() - self.last_call
            if elapsed < (1.0 / self.calls_per_second):
                sleep_time = (1.0 / self.calls_per_second) - elapsed
                if sleep_time > 0:
                    RATE_LIMIT_WAITS.labels(api_name).inc()
                    logger.debug(f"Rate limiting {api_name}, waiting {sleep_time:.2f}s")
                    time.sleep(sleep_time)
            
            self.last_call = time.time()

class CryptoPriceFetcher:
    def __init__(self, save_data=True, use_rabbitmq=False):
        self.save_data = save_data
        self.use_rabbitmq = use_rabbitmq
        self.price_history = self._load_history()
        self.data_file = "price_history.json"
        self.last_price_update = {}
        
        # Setup RabbitMQ connection if enabled
        if self.use_rabbitmq:
            try:
                self.rabbitmq_connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
                self.rabbitmq_channel = self.rabbitmq_connection.channel()
                self.rabbitmq_channel.queue_declare(queue='price_updates')
                logger.info("RabbitMQ connection established for price updates")
            except Exception as e:
                logger.error(f"Failed to connect to RabbitMQ: {str(e)}. Falling back to file-based updates.")
                self.use_rabbitmq = False
                self.rabbitmq_connection = None
                self.rabbitmq_channel = None

        # Starting mock prices for various symbols (fallback data)
        self.starting_prices = {
            "BTC": 87500.0,
            "ETH": 2200.0,
            "SOL": 142.0,
            "DOGE": 0.20,
            "ADA": 0.45,
            "XRP": 0.52,
            "DOT": 6.20,
            "AVAX": 28.50,
            "MATIC": 0.85,
            "LINK": 12.75,
            "LTC": 65.50,
            "UNI": 8.20,
            "ATOM": 7.80,
            "ETC": 21.50,
            "BCH": 320.0,
            "XLM": 0.12,
            "ALGO": 0.14,
            "FIL": 5.75,
            "AAVE": 85.0,
            "CRO": 0.085
        }
        
        # Initialize rate limiters for different APIs
        self.rate_limiters = {
            "coingecko": RateLimiter(calls_per_second=0.5, burst_limit=10),  # 10 calls per 20 seconds
            "coinbase": RateLimiter(calls_per_second=3, burst_limit=30),     # 30 calls per 10 seconds
            "kraken": RateLimiter(calls_per_second=1, burst_limit=15),       # 15 calls per 15 seconds
            "binance": RateLimiter(calls_per_second=10, burst_limit=100)     # 100 calls per 10 seconds
        }
        
        # API configuration for multiple providers
        self.apis = [
            {
                "name": "coingecko",
                "url": "https://api.coingecko.com/api/v3/simple/price?ids={id}&vs_currencies=usd",
                "symbol_map": {
                    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "DOGE": "dogecoin",
                    "ADA": "cardano", "XRP": "ripple", "DOT": "polkadot", "AVAX": "avalanche-2",
                    "MATIC": "matic-network", "LINK": "chainlink", "LTC": "litecoin", "UNI": "uniswap",
                    "ATOM": "cosmos", "ETC": "ethereum-classic", "BCH": "bitcoin-cash", "XLM": "stellar",
                    "ALGO": "algorand", "FIL": "filecoin", "AAVE": "aave", "CRO": "crypto-com-chain"
                },
                "rate_limiter": "coingecko",
                "extract": lambda data, id: float(data.get(id, {}).get("usd", 0)),
            },
            {
                "name": "coinbase",
                "url": "https://api.coinbase.com/v2/prices/{id}-USD/spot",
                "symbol_map": {
                    "BTC": "BTC", "ETH": "ETH", "SOL": "SOL", "DOGE": "DOGE",
                    "ADA": "ADA", "XRP": "XRP", "DOT": "DOT", "AVAX": "AVAX",
                    "MATIC": "MATIC", "LINK": "LINK", "LTC": "LTC", "UNI": "UNI",
                    "ATOM": "ATOM", "ETC": "ETC", "BCH": "BCH", "XLM": "XLM",
                    "ALGO": "ALGO", "FIL": "FIL", "AAVE": "AAVE", "CRO": "CRO"
                },
                "rate_limiter": "coinbase",
                "extract": lambda data, id: float(data["data"]["amount"]),
            },
            {
                "name": "kraken",
                "url": "https://api.kraken.com/0/public/Ticker?pair={id}USD",
                "symbol_map": {
                    "BTC": "XBT", "ETH": "ETH", "SOL": "SOL", "DOGE": "XDG",
                    "ADA": "ADA", "XRP": "XRP", "DOT": "DOT", "AVAX": "AVAX",
                    "MATIC": "MATIC", "LINK": "LINK", "LTC": "LTC", "UNI": "UNI",
                    "ATOM": "ATOM", "ETC": "ETC", "BCH": "BCH", "XLM": "XLM",
                    "ALGO": "ALGO", "FIL": "FIL", "AAVE": "AAVE", "CRO": "CRO"
                },
                "rate_limiter": "kraken",
                "extract": lambda data, id: float(data["result"][f"{id}USD"]["c"][0]),
            },
            {
                "name": "binance",
                "url": "https://api.binance.com/api/v3/ticker/price?symbol={id}USDT",
                "symbol_map": {
                    "BTC": "BTC", "ETH": "ETH", "SOL": "SOL", "DOGE": "DOGE",
                    "ADA": "ADA", "XRP": "XRP", "DOT": "DOT", "AVAX": "AVAX",
                    "MATIC": "MATIC", "LINK": "LINK", "LTC": "LTC", "UNI": "UNI",
                    "ATOM": "ATOM", "ETC": "ETC", "BCH": "BCH", "XLM": "XLM",
                    "ALGO": "ALGO", "FIL": "FIL", "AAVE": "AAVE", "CRO": "CRO"
                },
                "rate_limiter": "binance",
                "extract": lambda data, id: float(data["price"]),
            }
        ]
        
        # Start the metrics server for monitoring
        self._start_metrics_server()
        
        # Check API availability initially
        self._check_api_availability()
        
        # Initialize API availability metrics
        for api in self.apis:
            API_AVAILABILITY.labels(api["name"]).set(1)  # Default to available
        
        # Shuffle APIs to distribute load
        random.shuffle(self.apis)
        
        logger.info("CryptoPriceFetcher initialized")
    
    def _start_metrics_server(self):
        try:
            metrics_port = int(os.environ.get("METRICS_PORT", 8001))
            try:
                start_http_server(metrics_port)
                logger.info(f"Prometheus metrics server started on port {metrics_port}")
            except Exception as e:
                if "Address already in use" in str(e):
                    logger.info(f"Metrics server already running on port {metrics_port}")
                else:
                    raise
        except Exception as e:
            logger.error(f"Error starting metrics server: {e}")
    
    def _check_api_availability(self):
        for api in self.apis:
            try:
                api_domain = api["url"].split("//")[1].split("/")[0]
                is_available = self._is_dns_working(api_domain)
                API_AVAILABILITY.labels(api["name"]).set(1 if is_available else 0)
                if is_available:
                    logger.debug(f"API {api['name']} is available")
                else:
                    logger.warning(f"API {api['name']} appears to be unavailable")
            except Exception as e:
                logger.error(f"Error checking availability for {api['name']}: {e}")
                API_AVAILABILITY.labels(api["name"]).set(0)
    
    def _load_history(self):
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    history = json.load(f)
                logger.info(f"Loaded price history from {self.data_file}")
                
                # Initialize metrics based on loaded data
                for symbol, data in history.items():
                    if data and len(data) > 0:
                        latest_price = data[-1].get("price", 0)
                        CURRENT_PRICES.labels(symbol).set(latest_price)
                        PRICE_DISTRIBUTION.labels(symbol).observe(latest_price)
                        self._update_price_change_metrics(symbol, data)
                
                return history
            else:
                logger.info(f"No price history file found at {self.data_file}, creating new history")
                return {}
        except Exception as e:
            logger.error(f"Error loading price history: {e}")
            return {}
    
    def _update_price_change_metrics(self, symbol, price_data):
        if not price_data or len(price_data) < 2:
            return
        
        latest_price = price_data[-1].get("price", 0)
        latest_time = datetime.fromisoformat(price_data[-1].get("timestamp")) if isinstance(price_data[-1].get("timestamp"), str) else price_data[-1].get("timestamp")
        
        # Calculate price changes for different periods
        periods = [
            ("1h", timedelta(hours=1)),
            ("24h", timedelta(hours=24)),
            ("7d", timedelta(days=7)),
            ("30d", timedelta(days=30))
        ]
        
        for period_name, period_delta in periods:
            reference_time = latest_time - period_delta
            prev_price = None
            
            # Find closest data point to reference time
            for data_point in reversed(price_data):
                point_time = datetime.fromisoformat(data_point.get("timestamp")) if isinstance(data_point.get("timestamp"), str) else data_point.get("timestamp")
                if point_time <= reference_time:
                    prev_price = data_point.get("price", 0)
                    break
            
            if prev_price and prev_price > 0:
                price_change_pct = ((latest_price - prev_price) / prev_price) * 100
                PRICE_CHANGE_PCT.labels(symbol=symbol, period=period_name).set(price_change_pct)
    
    def _save_history(self):
        try:
            filename = self.data_file
            with open(filename, 'w') as f:
                json.dump(self.price_history, f, indent=2)
            logger.debug("Price history saved to file")
        except Exception as e:
            logger.error(f"Error saving price history: {e}")
    
    def _store_price(self, symbol, price, source="unknown"):
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Add to local price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            
        self.price_history[symbol].append({
            "timestamp": timestamp_str,
            "price": price,
            "source": source
        })
        
        # Limit the size of stored history
        if len(self.price_history[symbol]) > 10000:
            self.price_history[symbol] = self.price_history[symbol][-10000:]
        
        # Save to file if enabled
        if self.save_data:
            self._save_history()
        
        # Send via RabbitMQ if enabled
        if self.use_rabbitmq and hasattr(self, 'rabbitmq_channel') and self.rabbitmq_channel:
            try:
                message = json.dumps({
                    "symbol": symbol,
                    "price": price,
                    "timestamp": timestamp_str,
                    "source": source
                })
                self.rabbitmq_channel.basic_publish(
                    exchange='',
                    routing_key='price_updates',
                    body=message
                )
                logger.debug(f"Price update for {symbol} sent via RabbitMQ")
            except Exception as e:
                logger.error(f"Failed to send price update via RabbitMQ: {str(e)}")
        
        # Update metrics
        CURRENT_PRICES.labels(symbol).set(price)
        PRICE_UPDATES.labels(symbol=symbol, source=source).inc()
        PRICE_DISTRIBUTION.labels(symbol).observe(price)
        self.last_price_update[symbol] = time.time()
        
        # Update price change metrics if we have enough history
        if len(self.price_history[symbol]) > 1:
            self._update_price_change_metrics(symbol, self.price_history[symbol])
    
    def _is_dns_working(self, hostname):
        try:
            socket.gethostbyname(hostname)
            return True
        except:
            return False
    
    def _get_mock_price(self, symbol, reason="unknown"):
        try:
            if symbol in self.price_history and len(self.price_history[symbol]) > 0:
                last_price = self.price_history[symbol][-1]["price"]
                # Generate a small random change (-1% to +1%)
                change = (random.random() * 2 - 1) * 0.01
                price = last_price * (1 + change)
            else:
                # Use starting prices or a default
                price = self.starting_prices.get(symbol, 1000)
            
            self._store_price(symbol, price, source="mock_data")
            MOCK_PRICE_USES.labels(symbol=symbol, reason=reason).inc()
            logger.info(f"Using mock data for {symbol}: ${price} (reason: {reason})")
            return price
        except Exception as e:
            logger.error(f"Error generating mock price: {e}")
            price = self.starting_prices.get(symbol, 1000)
            MOCK_PRICE_USES.labels(symbol=symbol, reason="error_fallback").inc()
            logger.warning(f"Using default mock data for {symbol}: ${price}")
            return price
    
    def get_price(self, symbol="BTC", use_mock=False):
        # Check if we have recent data
        if symbol in self.last_price_update:
            age_seconds = time.time() - self.last_price_update.get(symbol, time.time())
            PRICE_AGE_SECONDS.labels(symbol).set(age_seconds)
        
        # If explicit mock data is requested, use that
        if use_mock:
            return self._get_mock_price(symbol, reason="explicitly_requested")
        
        # Check DNS resolution before making network requests
        if not self._is_dns_working("api.coingecko.com"):
            logger.warning("DNS resolution failed, using mock data")
            return self._get_mock_price(symbol, reason="dns_failure")
        
        # Randomize order to distribute load
        random.shuffle(self.apis)  # Randomize order to distribute load
        
        # Try each API in order
        for api in self.apis:
            try:
                # Check if API supports this symbol
                crypto_id = api["symbol_map"].get(symbol)
                if not crypto_id:
                    logger.debug(f"{api['name']} API doesn't support {symbol}, skipping...")
                    continue
                
                # Check if API domain is resolvable
                api_domain = api["url"].split("//")[1].split("/")[0]
                if not self._is_dns_working(api_domain):
                    logger.warning(f"DNS resolution failed for {api_domain}, skipping {api['name']} API...")
                    API_AVAILABILITY.labels(api["name"]).set(0)
                    continue
                else:
                    API_AVAILABILITY.labels(api["name"]).set(1)
                
                # Apply rate limiting
                rate_limiter_key = api.get("rate_limiter", "default")
                if rate_limiter_key in self.rate_limiters:
                    self.rate_limiters[rate_limiter_key].wait(api["name"])
                
                # Make the API request
                url = api["url"].format(id=crypto_id)
                logger.debug(f"Trying {api['name']} API...")
                
                with REQUEST_DURATION.labels(api["name"]).time():
                    response = requests.get(url, timeout=10)
                
                API_REQUESTS.labels(api=api["name"], symbol=symbol, status=str(response.status_code)).inc()
                
                if response.status_code != 200:
                    logger.warning(f"Non-200 response from {api['name']} API: {response.status_code}")
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                # Extract price using the API-specific lambda function
                price = api["extract"](data, crypto_id)
                
                if price <= 0:
                    logger.warning(f"Invalid price ({price}) from {api['name']} API")
                    continue
                
                # Store and return the price
                self._store_price(symbol, price, source=api["name"])
                logger.info(f"Successfully fetched {symbol} price from {api['name']}: ${price:.2f}")
                return price
                
            except KeyError as e:
                logger.warning(f"Error with {api['name']} API: Unexpected response format - missing key {str(e)}")
                API_REQUESTS.labels(api=api["name"], symbol=symbol, status="key_error").inc()
                continue
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error with {api['name']} API: {str(e)}")
                API_REQUESTS.labels(api=api["name"], symbol=symbol, status="request_error").inc()
                if "Connection" in str(e) or "Timeout" in str(e):
                    API_AVAILABILITY.labels(api["name"]).set(0)
                continue
            except ValueError as e:
                logger.warning(f"Error parsing data from {api['name']} API: {str(e)}")
                API_REQUESTS.labels(api=api["name"], symbol=symbol, status="parse_error").inc()
                continue
            except Exception as e:
                logger.warning(f"Unexpected error with {api['name']} API: {str(e)}")
                API_REQUESTS.labels(api=api["name"], symbol=symbol, status="other_error").inc()
                logger.debug(traceback.format_exc())
                continue
        
        # If all APIs failed, fall back to mock data
        logger.warning("All APIs failed, falling back to mock data.")
        return self._get_mock_price(symbol, reason="api_failure")
    
    def get_price_history(self, symbol):
        """Get the historical price data for a symbol"""
        if symbol in self.price_history:
            return self.price_history[symbol]
        return None
    
    def __del__(self):
        # Close RabbitMQ connection if it exists
        if hasattr(self, 'rabbitmq_connection') and self.rabbitmq_connection is not None:
            try:
                self.rabbitmq_connection.close()
                logger.info("RabbitMQ connection closed")
            except Exception as e:
                logger.error(f"Error closing RabbitMQ connection: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cryptocurrency Price Fetcher")
    parser.add_argument("--use_rabbitmq", action="store_true", help="Use RabbitMQ for price updates")
    parser.add_argument("--auto", action="store_true", help="Automatically fetch prices at regular intervals")
    parser.add_argument("--interval", type=int, default=300, help="Update interval in seconds (default: 300)")
    parser.add_argument("--symbols", type=str, default="BTC,ETH,SOL", help="Comma-separated list of symbols to fetch")
    args = parser.parse_args()
    
    fetcher = CryptoPriceFetcher(save_data=True, use_rabbitmq=args.use_rabbitmq)
    
    if args.auto:
        symbols = args.symbols.split(',')
        logger.info(f"Starting automatic price updates for {symbols} every {args.interval} seconds")
        
        try:
            while True:
                for symbol in symbols:
                    try:
                        price = fetcher.get_price(symbol)
                        logger.info(f"Updated {symbol} price: ${price:.2f}")
                    except Exception as e:
                        logger.error(f"Error updating {symbol} price: {e}")
                
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("Price fetcher stopped by user")
    else:
        # Interactive mode
        print("Cryptocurrency Price Fetcher")
        print("===========================")
        print("Enter a symbol to fetch its price (or 'quit' to exit)")
        
        while True:
            symbol = input("\nSymbol: ").strip().upper()
            if symbol.lower() == 'quit':
                break
            
            try:
                price = fetcher.get_price(symbol)
                print(f"{symbol} Price: ${price:.2f}")
            except Exception as e:
                print(f"Error: {e}")