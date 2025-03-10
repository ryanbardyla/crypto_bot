#!/usr/bin/env python3
"""
enhanced_price_fetcher.py - Enhanced Cryptocurrency Price Fetcher

This is an enhanced version of the price fetcher that uses:
- RabbitMQ for message passing (instead of file-based communication)
- Redis for caching (to reduce API calls)
- PostgreSQL for data storage (replacing SQLite)
- Improved error handling with retry mechanism

This implementation follows the improvement plan to replace file-based communication
with more reliable messaging, improve database performance, and enhance resilience.
"""

import os
import json
import time
import logging
import random
import threading
import traceback
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from functools import lru_cache
import socket
import requests
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict

# Prometheus metrics
try:
    from prometheus_client import Counter, Gauge, Summary, Histogram
    from prometheus_client import start_http_server
    PROMETHEUS_ENABLED = True
except ImportError:
    PROMETHEUS_ENABLED = False
    
    # Create dummy metric classes if Prometheus is not available
    class DummyMetric:
        def __init__(self, *args, **kwargs):
            pass
        
        def inc(self, *args, **kwargs):
            pass
            
        def dec(self, *args, **kwargs):
            pass
            
        def set(self, *args, **kwargs):
            pass
            
        def observe(self, *args, **kwargs):
            pass
            
        def labels(self, *args, **kwargs):
            return self
            
    Counter = Gauge = Summary = Histogram = DummyMetric

# Internal imports
from utils.logging_config import setup_logging
from utils.retry_utility import retry_api_call, ConnectionFailureError, RateLimitError
from message_broker import MessageBroker
from cache_manager import CacheManager
from database_manager_postgres import DatabaseManager, PriceRecord

# Configure logging
logger = setup_logging(name="price_fetcher")

# Set up Prometheus metrics
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
                             ['symbol'], buckets=(1, 10, 100, 1000, 10000, 100000))
CACHE_HITS = Counter('price_fetcher_cache_hits_total', 'Number of cache hits', ['symbol'])
CACHE_MISSES = Counter('price_fetcher_cache_misses_total', 'Number of cache misses', ['symbol'])
DB_OPERATIONS = Counter('price_fetcher_db_operations_total', 'Number of database operations', ['operation'])
MESSAGE_OPERATIONS = Counter('price_fetcher_message_operations_total', 'Number of message operations', ['operation'])

class RateLimiter:
    """Rate limiter for API requests with fair distribution among symbols."""
    
    def __init__(self, calls_per_second=1, burst_limit=None):
        self.calls_per_second = calls_per_second
        self.interval = 1.0 / calls_per_second if calls_per_second > 0 else 0
        self.burst_limit = burst_limit
        self.lock = threading.RLock()
        self.last_call = time.time()
        self.call_count = 0
        self.reset_time = time.time() + 1.0  # Reset counter after 1 second
        
    def wait(self, api_name="unknown"):
        """Wait if necessary to respect rate limits."""
        with self.lock:
            current_time = time.time()
            
            # Check if we need to reset the counter
            if current_time > self.reset_time:
                self.call_count = 0
                self.reset_time = current_time + 1.0
                
            # Check burst limit
            if self.burst_limit and self.call_count >= self.burst_limit:
                sleep_time = self.reset_time - current_time
                if sleep_time > 0:
                    RATE_LIMIT_WAITS.labels(api_name).inc()
                    logger.debug(f"Rate limit hit for {api_name}, waiting {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                    self.call_count = 0
                    self.reset_time = time.time() + 1.0
                    
            # Check time since last call
            elapsed = time.time() - self.last_call
            if elapsed < self.interval:
                sleep_time = self.interval - elapsed
                if sleep_time > 0:
                    RATE_LIMIT_WAITS.labels(api_name).inc()
                    logger.debug(f"Rate limiting {api_name}, waiting {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                    
            self.last_call = time.time()
            self.call_count += 1


class EnhancedPriceFetcher:
    """
    Enhanced cryptocurrency price fetcher with improved architecture:
    - RabbitMQ for message passing
    - Redis for caching
    - PostgreSQL for storage
    - Robust error handling and retry
    """
    
    def __init__(
        self, 
        save_data=True, 
        use_rabbitmq=False,
        use_redis=True,
        use_postgres=True,
        metrics_port=None
    ):
        """
        Initialize the price fetcher.
        
        Args:
            save_data: Whether to save price data to the database
            use_rabbitmq: Whether to use RabbitMQ for message passing
            use_redis: Whether to use Redis for caching
            use_postgres: Whether to use PostgreSQL for storage
            metrics_port: Port for Prometheus metrics server
        """
        self.save_data = save_data
        self.use_rabbitmq = use_rabbitmq
        self.use_redis = use_redis
        self.use_postgres = use_postgres
        
        # Initialize the data file path
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.data_file = os.path.join(self.data_dir, "price_history.json")
        
        # Price history (in-memory fallback)
        self.price_history = self._load_history()
        
        # Last updated timestamps
        self.last_price_update = {}
        
        # Initialize RabbitMQ if enabled
        self.message_broker = None
        if self.use_rabbitmq:
            try:
                self.message_broker = MessageBroker(
                    host=os.environ.get("RABBITMQ_HOST", "localhost"),
                    port=int(os.environ.get("RABBITMQ_PORT", "5672")),
                    username=os.environ.get("RABBITMQ_USER", "guest"),
                    password=os.environ.get("RABBITMQ_PASS", "guest")
                )
                self.message_broker.connect()
                logger.info("RabbitMQ connection established")
            except Exception as e:
                logger.error(f"Failed to connect to RabbitMQ: {str(e)}. Falling back to file-based updates.")
                self.use_rabbitmq = False
        
        # Initialize Redis if enabled
        self.cache_manager = None
        if self.use_redis:
            try:
                self.cache_manager = CacheManager(
                    host=os.environ.get("REDIS_HOST", "localhost"),
                    port=int(os.environ.get("REDIS_PORT", "6379")),
                    password=os.environ.get("REDIS_PASSWORD", None),
                    namespace="crypto_price"
                )
                # Test connection
                health = self.cache_manager.health_check()
                if health['healthy']:
                    logger.info(f"Redis connection established: {health['response_time_ms']:.2f}ms response time")
                else:
                    logger.error(f"Redis connection failed: {health['error']}. Caching disabled.")
                    self.use_redis = False
                    self.cache_manager = None
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}. Caching disabled.")
                self.use_redis = False
                self.cache_manager = None
        
        # Initialize PostgreSQL if enabled
        self.db_manager = None
        if self.use_postgres:
            try:
                self.db_manager = DatabaseManager(
                    host=os.environ.get("POSTGRES_HOST", "localhost"),
                    port=os.environ.get("POSTGRES_PORT", "5432"),
                    database=os.environ.get("POSTGRES_DB", "crypto_trading"),
                    user=os.environ.get("POSTGRES_USER", "postgres"),
                    password=os.environ.get("POSTGRES_PASSWORD", "postgres")
                )
                logger.info("PostgreSQL connection established")
            except Exception as e:
                logger.error(f"Failed to connect to PostgreSQL: {str(e)}. Falling back to file-based storage.")
                self.use_postgres = False
                self.db_manager = None
        
        # Starting prices for mock data fallback
        self.starting_prices = {
            "BTC": 50000.0,
            "ETH": 3000.0,
            "SOL": 150.0,
            "DOGE": 0.2,
            "ADA": 1.2,
            "XRP": 1.0,
            "DOT": 30.0,
            "UNI": 25.0
        }
        
        # Rate limiters for different APIs
        self.rate_limiters = {
            "coingecko": RateLimiter(calls_per_second=0.5, burst_limit=10),  # 10 calls per 20 seconds
            "coinbase": RateLimiter(calls_per_second=3, burst_limit=30),     # 30 calls per 10 seconds
            "kraken": RateLimiter(calls_per_second=1, burst_limit=15),       # 15 calls per 15 seconds
            "binance": RateLimiter(calls_per_second=10, burst_limit=100)     # 100 calls per 10 seconds
        }
        
        # API configurations
        self.apis = [
            {
                "name": "coingecko",
                "url": "https://api.coingecko.com/api/v3/simple/price?ids={id}&vs_currencies=usd",
                "symbol_map": {
                    "BTC": "bitcoin",
                    "ETH": "ethereum",
                    "SOL": "solana",
                    "DOGE": "dogecoin",
                    "ADA": "cardano",
                    "XRP": "ripple",
                    "DOT": "polkadot",
                    "UNI": "uniswap"
                },
                "rate_limiter": "coingecko",
                "extract": lambda data, id: float(data.get(id, {}).get("usd", 0))
            },
            {
                "name": "coinbase",
                "url": "https://api.coinbase.com/v2/prices/{id}-USD/spot",
                "symbol_map": {
                    "BTC": "BTC",
                    "ETH": "ETH",
                    "SOL": "SOL",
                    "DOGE": "DOGE",
                    "ADA": "ADA",
                    "XRP": "XRP",
                    "DOT": "DOT",
                    "UNI": "UNI"
                },
                "rate_limiter": "coinbase",
                "extract": lambda data, id: float(data["data"]["amount"])
            },
            {
                "name": "kraken",
                "url": "https://api.kraken.com/0/public/Ticker?pair={id}USD",
                "symbol_map": {
                    "BTC": "XXBT",
                    "ETH": "XETH",
                    "SOL": "SOL",
                    "DOGE": "XXDG",
                    "ADA": "ADA",
                    "XRP": "XXRP",
                    "DOT": "DOT",
                    "UNI": "UNI"
                },
                "rate_limiter": "kraken",
                "extract": lambda data, id: float(data["result"][f"{id}USD"]["c"][0])
            },
            {
                "name": "binance",
                "url": "https://api.binance.com/api/v3/ticker/price?symbol={id}USDT",
                "symbol_map": {
                    "BTC": "BTC",
                    "ETH": "ETH",
                    "SOL": "SOL",
                    "DOGE": "DOGE",
                    "ADA": "ADA",
                    "XRP": "XRP",
                    "DOT": "DOT",
                    "UNI": "UNI"
                },
                "rate_limiter": "binance",
                "extract": lambda data, id: float(data["price"])
            }
        ]
        
        # Start Prometheus metrics server if enabled
        if PROMETHEUS_ENABLED and metrics_port:
            self._start_metrics_server(metrics_port)
            
        # Set API availability to default (available)
        for api in self.apis:
            API_AVAILABILITY.labels(api["name"]).set(1)
            
        # Randomize API order to distribute load
        random.shuffle(self.apis)
        
        logger.info("EnhancedPriceFetcher initialized")
        
    def _start_metrics_server(self, metrics_port):
        """Start Prometheus metrics server."""
        try:
            metrics_port = int(metrics_port)
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
        """Check if APIs are available by DNS lookup."""
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
        """Load price history from file."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    history = json.load(f)
                    
                logger.info(f"Loaded price history from {self.data_file}")
                
                # Update Prometheus metrics with loaded data
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
        """Update Prometheus metrics for price changes."""
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
            # Find the closest price data point for the period
            target_time = latest_time - period_delta
            prev_price = None
            
            for data_point in reversed(price_data):
                point_time = datetime.fromisoformat(data_point.get("timestamp")) if isinstance(data_point.get("timestamp"), str) else data_point.get("timestamp")
                if point_time <= target_time:
                    prev_price = data_point.get("price", 0)
                    break
                    
            if prev_price is not None and prev_price > 0:
                price_change_pct = ((latest_price - prev_price) / prev_price) * 100
                PRICE_CHANGE_PCT.labels(symbol=symbol, period=period_name).set(price_change_pct)

    def _save_history(self):
        """Save price history to file (used as fallback when database is unavailable)."""
        if not self.save_data:
            return
            
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.price_history, f, indent=2)
                
            logger.debug("Price history saved to file")
        except Exception as e:
            logger.error(f"Error saving price history: {e}")

    def _store_price(self, symbol, price, source="unknown"):
        """
        Store price data in memory, file, database, and publish via message broker.
        
        Args:
            symbol: Cryptocurrency symbol
            price: Price value
            source: Source of the price data
        """
        # Create price data record
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        price_data = {
            "price": price,
            "timestamp": timestamp_str,
            "source": source
        }
        
        # Store in memory
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            
        self.price_history[symbol].append(price_data)
        
        # Limit history size
        if len(self.price_history[symbol]) > 10000:
            self.price_history[symbol] = self.price_history[symbol][-10000:]
            
        # Save to file (as fallback)
        if len(self.price_history[symbol]) % 10 == 0:  # Only save every 10 updates to reduce I/O
            self._save_history()
            
        # Store in PostgreSQL if enabled
        if self.use_postgres and self.db_manager:
            try:
                DB_OPERATIONS.labels("insert").inc()
                self.db_manager.save_price_data(symbol, price_data)
            except Exception as e:
                logger.error(f"Error saving price data to PostgreSQL: {str(e)}")
                DB_OPERATIONS.labels("error").inc()
                
        # Cache in Redis if enabled
        if self.use_redis and self.cache_manager:
            try:
                # Cache with a 60-second TTL (adjust as needed)
                self.cache_manager.cache_price(symbol, price_data, ttl=60)
            except Exception as e:
                logger.error(f"Error caching price data in Redis: {str(e)}")
                
        # Publish via RabbitMQ if enabled
        if self.use_rabbitmq and self.message_broker:
            try:
                MESSAGE_OPERATIONS.labels("publish").inc()
                self.message_broker.publish_price_update(symbol, price_data)
                logger.debug(f"Price update for {symbol} sent via RabbitMQ")
            except Exception as e:
                logger.error(f"Failed to send price update via RabbitMQ: {str(e)}")
                MESSAGE_OPERATIONS.labels("error").inc()
                
        # Update Prometheus metrics
        CURRENT_PRICES.labels(symbol).set(price)
        PRICE_UPDATES.labels(symbol=symbol, source=source).inc()
        PRICE_DISTRIBUTION.labels(symbol).observe(price)
        self.last_price_update[symbol] = time.time()
        
        # Update price change metrics if we have enough data
        if len(self.price_history[symbol]) > 1:
            self._update_price_change_metrics(symbol, self.price_history[symbol])
            
        logger.info(f"Stored {symbol} price: ${price:.2f} from {source}")

    def _is_dns_working(self, hostname):
        """Check if DNS resolution is working for a hostname."""
        try:
            socket.gethostbyname(hostname)
            return True
        except socket.gaierror:
            return False

    def _get_mock_price(self, symbol, reason="unknown"):
        """Generate a mock price when API calls fail."""
        try:
            # Use the last known price with a small random change
            if symbol in self.price_history and len(self.price_history[symbol]) > 0:
                last_price = self.price_history[symbol][-1]["price"]
                # Random change between -1% and +1%
                change = (random.random() * 2 - 1) * 0.01
                price = last_price * (1 + change)
            else:
                # Use a starting price with a small random change
                price = self.starting_prices.get(symbol, 1000)
                change = (random.random() * 2 - 1) * 0.01
                price = price * (1 + change)
                
            self._store_price(symbol, price, source="mock_data")
            MOCK_PRICE_USES.labels(symbol=symbol, reason=reason).inc()
            
            logger.info(f"Using mock data for {symbol}: ${price} (reason: {reason})")
            return price
            
        except Exception as e:
            logger.error(f"Error generating mock price: {e}")
            MOCK_PRICE_USES.labels(symbol=symbol, reason="error_fallback").inc()
            
            # Last resort fallback
            price = self.starting_prices.get(symbol, 1000)
            logger.warning(f"Using default mock data for {symbol}: ${price}")
            return price

    @retry_api_call(
        max_retries=3,
        backoff_factor=2.0,
        jitter=True,
        exceptions=[requests.exceptions.RequestException, ConnectionError, TimeoutError],
        circuit_breaker="price_api"
    )
    def _fetch_from_api(self, api, symbol):
        """
        Fetch price from a specific API with retry and circuit breaker.
        
        Args:
            api: API configuration dictionary
            symbol: Cryptocurrency symbol
            
        Returns:
            float: Price value
            
        Raises:
            ConnectionFailureError: If connection fails
            RateLimitError: If rate limited
            ValueError: If price cannot be extracted
        """
        crypto_id = api["symbol_map"].get(symbol)
        if not crypto_id:
            raise ValueError(f"Symbol {symbol} not supported by {api['name']} API")
            
        # Apply rate limiting
        rate_limiter_key = api.get("rate_limiter", "default")
        self.rate_limiters.get(rate_limiter_key, self.rate_limiters["coingecko"]).wait(api["name"])
        
        # Construct URL
        url = api["url"].format(id=crypto_id)
        
        logger.debug(f"Fetching {symbol} price from {api['name']} API...")
        
        # Measure request duration
        start_time = time.time()
        
        try:
            response = requests.get(url, timeout=10)
            duration = time.time() - start_time
            REQUEST_DURATION.labels(api["name"]).observe(duration)
            
            # Count request
            API_REQUESTS.labels(api=api["name"], symbol=symbol, status=str(response.status_code)).inc()
            
            # Check status code
            if response.status_code != 200:
                logger.warning(f"Non-200 response from {api['name']} API: {response.status_code}")
                if response.status_code == 429:
                    raise RateLimitError(f"Rate limited by {api['name']} API")
                elif response.status_code >= 500:
                    raise ConnectionFailureError(f"Server error from {api['name']} API: {response.status_code}")
                else:
                    raise ValueError(f"Error response from {api['name']} API: {response.status_code}")
                    
            response.raise_for_status()
            
            # Parse data
            data = response.json()
            
            # Extract price using the API-specific lambda
            price = api["extract"](data, crypto_id)
            
            # Validate price
            if not price or price <= 0:
                logger.warning(f"Invalid price ({price}) from {api['name']} API")
                raise ValueError(f"Invalid price value from {api['name']} API")
                
            return price
            
        except requests.exceptions.Timeout:
            API_REQUESTS.labels(api=api["name"], symbol=symbol, status="timeout").inc()
            raise ConnectionFailureError(f"Timeout connecting to {api['name']} API")
            
        except requests.exceptions.ConnectionError:
            API_REQUESTS.labels(api=api["name"], symbol=symbol, status="connection_error").inc()
            raise ConnectionFailureError(f"Connection error with {api['name']} API")
            
        except requests.exceptions.RequestException as e:
            API_REQUESTS.labels(api=api["name"], symbol=symbol, status="request_error").inc()
            raise ConnectionFailureError(f"Request error with {api['name']} API: {str(e)}")
            
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Error with {api['name']} API: Unexpected response format - missing key {str(e)}")
            API_REQUESTS.labels(api=api["name"], symbol=symbol, status="key_error").inc()
            raise ValueError(f"Data extraction error with {api['name']} API: {str(e)}")
            
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Error parsing data from {api['name']} API: {str(e)}")
            API_REQUESTS.labels(api=api["name"], symbol=symbol, status="parse_error").inc()
            raise ValueError(f"JSON parsing error with {api['name']} API: {str(e)}")
            
        except Exception as e:
            logger.warning(f"Unexpected error with {api['name']} API: {str(e)}")
            API_REQUESTS.labels(api=api["name"], symbol=symbol, status="other_error").inc()
            logger.debug(traceback.format_exc())
            raise

    def get_price(self, symbol="BTC", use_mock=False):
        """
        Get current price for a cryptocurrency symbol.
        
        Args:
            symbol: Cryptocurrency symbol
            use_mock: Whether to use mock data instead of API calls
            
        Returns:
            float: Current price
        """
        symbol = symbol.upper()
        
        # Update age metrics
        if symbol in self.last_price_update:
            age_seconds = time.time() - self.last_price_update.get(symbol, time.time())
            PRICE_AGE_SECONDS.labels(symbol).set(age_seconds)
            
        # Return mock data if requested
        if use_mock:
            return self._get_mock_price(symbol, reason="explicitly_requested")
            
        # Try to get from Redis cache first
        if self.use_redis and self.cache_manager:
            try:
                cached_price = self.cache_manager.get_cached_price(symbol)
                if cached_price and 'price' in cached_price:
                    # Check if cache is still fresh (less than 60 seconds old)
                    cache_time = datetime.fromisoformat(cached_price['timestamp']) if isinstance(cached_price['timestamp'], str) else cached_price['timestamp']
                    if (datetime.now() - cache_time).total_seconds() < 60:
                        CACHE_HITS.labels(symbol).inc()
                        logger.debug(f"Cache hit for {symbol}: ${cached_price['price']:.2f}")
                        return cached_price['price']
                    else:
                        logger.debug(f"Cache expired for {symbol}")
            except Exception as e:
                logger.error(f"Error retrieving from cache: {str(e)}")
                
        CACHE_MISSES.labels(symbol).inc()
        
        # Check if DNS is working before making API calls
        if not self._is_dns_working("api.coingecko.com"):
            logger.warning("DNS resolution failed, using mock data")
            return self._get_mock_price(symbol, reason="dns_failure")
            
        # Randomize API order to distribute load
        random.shuffle(self.apis)
        
        # Try each API in order
        for api in self.apis:
            try:
                # Skip if API doesn't support this symbol
                if symbol not in api["symbol_map"]:
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
                    
                # Fetch price from this API
                price = self._fetch_from_api(api, symbol)
                
                # If we got a valid price, store it and return
                self._store_price(symbol, price, source=api["name"])
                logger.info(f"Successfully fetched {symbol} price from {api['name']}: ${price:.2f}")
                return price
                
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol} price from {api['name']}: {str(e)}")
                continue
                
        # If all APIs failed, fall back to mock data
        logger.warning("All APIs failed, falling back to mock data.")
        return self._get_mock_price(symbol, reason="api_failure")

    def get_price_history(self, symbol, days=None, start_date=None, end_date=None):
        """
        Get price history for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            days: Number of days of history to retrieve (if None, return all available)
            start_date: Start date for filtering (if None, no start filter)
            end_date: End date for filtering (if None, no end filter)
            
        Returns:
            list: List of price data points
        """
        symbol = symbol.upper()
        
        # Try to get from PostgreSQL first if enabled
        if self.use_postgres and self.db_manager:
            try:
                DB_OPERATIONS.labels("query").inc()
                price_data = self.db_manager.get_price_data(
                    symbol, 
                    start_date=start_date, 
                    end_date=end_date,
                    limit=None  # No limit for history
                )
                
                if price_data:
                    logger.info(f"Retrieved {len(price_data)} price points for {symbol} from PostgreSQL")
                    return price_data
            except Exception as e:
                logger.error(f"Error retrieving price history from PostgreSQL: {str(e)}")
                DB_OPERATIONS.labels("error").inc()
        
        # Fall back to in-memory/file-based history
        if symbol in self.price_history:
            history = self.price_history[symbol]
            
            # Apply filters if provided
            if days is not None or start_date is not None or end_date is not None:
                filtered_history = []
                
                for entry in history:
                    # Convert timestamp string to datetime if needed
                    if isinstance(entry["timestamp"], str):
                        timestamp = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
                    else:
                        timestamp = entry["timestamp"]
                        
                    # Apply date filters
                    if days is not None:
                        cutoff = datetime.now() - timedelta(days=days)
                        if timestamp < cutoff:
                            continue
                            
                    if start_date is not None:
                        if isinstance(start_date, str):
                            start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                        if timestamp < start_date:
                            continue
                            
                    if end_date is not None:
                        if isinstance(end_date, str):
                            end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                        if timestamp > end_date:
                            continue
                            
                    filtered_history.append(entry)
                    
                return filtered_history
            else:
                return history
        
        # No history available
        return []

    def update_multiple_prices(self, symbols):
        """
        Update prices for multiple symbols concurrently.
        
        Args:
            symbols: List of cryptocurrency symbols
            
        Returns:
            dict: Dictionary of symbols and their prices
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(10, len(symbols))) as executor:
            future_to_symbol = {executor.submit(self.get_price, symbol): symbol for symbol in symbols}
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    price = future.result()
                    results[symbol] = price
                    logger.info(f"Updated {symbol} price: ${price:.2f}")
                except Exception as e:
                    logger.error(f"Error updating {symbol} price: {e}")
                    results[symbol] = None
                    
        return results

    def close(self):
        """Clean up resources."""
        # Close RabbitMQ connection if used
        if self.use_rabbitmq and self.message_broker:
            try:
                self.message_broker.close()
                logger.info("RabbitMQ connection closed")
            except Exception as e:
                logger.error(f"Error closing RabbitMQ connection: {str(e)}")
                
        # Close Redis connection if used
        if self.use_redis and self.cache_manager:
            try:
                self.cache_manager.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {str(e)}")
                
        # Close PostgreSQL connection if used
        if self.use_postgres and self.db_manager:
            try:
                self.db_manager.close()
                logger.info("PostgreSQL connection closed")
            except Exception as e:
                logger.error(f"Error closing PostgreSQL connection: {str(e)}")
                
        logger.info("EnhancedPriceFetcher closed")

    def __del__(self):
        """Destructor to ensure connections are closed."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Usage example
# The rest of the file remains the same...

# Remove the incorrectly placed code at the end and replace with a proper __main__ block
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Cryptocurrency Price Fetcher")
    parser.add_argument("--use_rabbitmq", action="store_true", help="Use RabbitMQ for price updates")
    parser.add_argument("--use_redis", action="store_true", help="Use Redis for caching")
    parser.add_argument("--use_postgres", action="store_true", help="Use PostgreSQL for storage")
    parser.add_argument("--auto", action="store_true", help="Automatically fetch prices at regular intervals")
    parser.add_argument("--interval", type=int, default=300, help="Update interval in seconds (default: 300)")
    parser.add_argument("--symbols", type=str, default="BTC,ETH,SOL", help="Comma-separated list of symbols to fetch")
    parser.add_argument("--metrics_port", type=int, default=8001, help="Port for Prometheus metrics")
    
    args = parser.parse_args()
    
    # Create price fetcher
    fetcher = EnhancedPriceFetcher(
        save_data=True,
        use_rabbitmq=args.use_rabbitmq,
        use_redis=args.use_redis,
        use_postgres=args.use_postgres,
        metrics_port=args.metrics_port
    )
    
    try:
        # Parse symbols
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
        
        if args.auto:
            # Automatic mode
            logger.info(f"Starting automatic price updates for {symbols} every {args.interval} seconds")
            
            while True:
                try:
                    # Update all symbols
                    results = fetcher.update_multiple_prices(symbols)
                    
                    # Display results
                    for symbol, price in results.items():
                        if price is not None:
                            logger.info(f"Updated {symbol} price: ${price:.2f}")
                        else:
                            logger.error(f"Failed to update {symbol} price")
                            
                    # Wait for next update
                    time.sleep(args.interval)
                    
                except KeyboardInterrupt:
                    logger.info("Price fetcher stopped by user")
                    break
                    
                except Exception as e:
                    logger.error(f"Error in price update loop: {e}")
                    time.sleep(10)  # Wait a bit before retrying
        else:
            # Interactive mode
            print("Enhanced Cryptocurrency Price Fetcher")
            print("===================================")
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
                    
    finally:
        # Clean up
        fetcher.close()