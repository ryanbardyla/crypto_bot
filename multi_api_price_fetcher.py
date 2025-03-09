# multi_api_price_fetcher.py (updated with metrics)

import os
import time
import json
import random
import threading
import requests
import logging
import socket
from datetime import datetime
import traceback

# Import Prometheus metrics
from prometheus_client import Counter, Gauge, Summary, Histogram, start_http_server

# Import the centralized logging configuration
from utils.logging_config import get_module_logger
from datetime import datetime, timedelta
# Set up metrics
# Counters
API_REQUESTS = Counter('price_fetcher_api_requests_total', 'Number of API requests', ['api', 'symbol', 'status'])
PRICE_UPDATES = Counter('price_fetcher_updates_total', 'Number of price updates', ['symbol', 'source'])
RATE_LIMIT_WAITS = Counter('price_fetcher_rate_limit_waits_total', 'Number of rate limit waits', ['api'])
MOCK_PRICE_USES = Counter('price_fetcher_mock_price_uses_total', 'Number of times mock price was used', ['symbol', 'reason'])

# Gauges
CURRENT_PRICES = Gauge('price_fetcher_current_price', 'Current price of cryptocurrency', ['symbol'])
PRICE_AGE_SECONDS = Gauge('price_fetcher_price_age_seconds', 'Age of price data in seconds', ['symbol'])
API_AVAILABILITY = Gauge('price_fetcher_api_availability', 'API availability status (1=up, 0=down)', ['api'])
PRICE_CHANGE_PCT = Gauge('price_fetcher_price_change_percent', 'Price change percentage in last period', ['symbol', 'period'])

# Summaries and Histograms
REQUEST_DURATION = Summary('price_fetcher_request_duration_seconds', 'Duration of API requests', ['api'])
PRICE_DISTRIBUTION = Histogram('price_fetcher_price_distribution', 'Distribution of fetched prices', 
                               ['symbol'], buckets=[1, 10, 100, 1000, 10000, 100000])

# Get logger for this module
logger = get_module_logger(__name__)

class RateLimiter:
    """Rate limiter for API calls"""
    def __init__(self, calls_per_second=1, burst_limit=None):
        self.rate = calls_per_second
        self.last_call = 0
        self.lock = threading.Lock()
        self.burst_limit = burst_limit
        self.call_count = 0
        self.reset_time = time.time() + 1.0  # Reset counter after 1 second
        
    def wait(self, api_name="unknown"):
        """Wait if necessary to maintain the rate limit"""
        with self.lock:
            # Check burst limit
            if self.burst_limit:
                current_time = time.time()
                if current_time > self.reset_time:
                    self.call_count = 0
                    self.reset_time = current_time + 1.0
                
                self.call_count += 1
                if self.call_count > self.burst_limit:
                    sleep_time = self.reset_time - current_time
                    if sleep_time > 0:
                        RATE_LIMIT_WAITS.labels(api_name).inc()
                        logger.debug(f"Rate limit hit for {api_name}, waiting {sleep_time:.2f}s")
                        time.sleep(max(0, sleep_time))
                    self.call_count = 1
                    self.reset_time = time.time() + 1.0
            
            # Regular rate limiting
            elapsed = time.time() - self.last_call
            if elapsed < (1.0 / self.rate):
                sleep_time = (1.0 / self.rate) - elapsed
                if sleep_time > 0:
                    RATE_LIMIT_WAITS.labels(api_name).inc()
                    logger.debug(f"Rate limiting {api_name}, waiting {sleep_time:.2f}s")
                    time.sleep(sleep_time)
            
            self.last_call = time.time()

class CryptoPriceFetcher:
    def __init__(self, save_data=True):
        self.data_file = "price_history.json"
        self.price_history = self._load_history()
        self.save_data = save_data
        
        # Initialize rate limiters
        self.rate_limiters = {
            "coingecko": RateLimiter(calls_per_second=0.5, burst_limit=10),  # 10 calls per 20 seconds
            "coinbase": RateLimiter(calls_per_second=3, burst_limit=30),     # 30 calls per 10 seconds
            "kraken": RateLimiter(calls_per_second=1, burst_limit=15),       # 15 calls per 15 seconds
            "binance": RateLimiter(calls_per_second=10, burst_limit=100)     # 100 calls per 10 seconds
        }
        
        # Set up API configurations
        self.apis = [
            {
                "name": "CoinGecko",
                "url": "https://api.coingecko.com/api/v3/simple/price?ids={id}&vs_currencies=usd",
                "symbol_map": {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "DOGE": "dogecoin"},
                "extract": lambda data, id: float(data.get(id, {}).get("usd", 0)),
                "rate_limiter": "coingecko"
            },
            {
                "name": "Coinbase",
                "url": "https://api.coinbase.com/v2/prices/{id}-USD/spot",
                "symbol_map": {"BTC": "BTC", "ETH": "ETH", "SOL": "SOL", "DOGE": "DOGE"},
                "extract": lambda data, id: float(data["data"]["amount"]),
                "rate_limiter": "coinbase"
            },
            {
                "name": "Kraken",
                "url": "https://api.kraken.com/0/public/Ticker?pair={id}USD",
                "symbol_map": {"BTC": "XBT", "ETH": "ETH", "SOL": "SOL", "DOGE": "DOGE"},
                "extract": lambda data, id: float(data["result"][f"{id}USD"]["c"][0]),
                "rate_limiter": "kraken"
            },
            {
                "name": "Binance",
                "url": "https://api.binance.com/api/v3/ticker/price?symbol={id}USDT",
                "symbol_map": {"BTC": "BTC", "ETH": "ETH", "SOL": "SOL", "DOGE": "DOGE"},
                "extract": lambda data, id: float(data["price"]),
                "rate_limiter": "binance"
            }
        ]
        
        # Start the metrics HTTP server if not already started
        self._start_metrics_server()
        
        # Check API availability on initialization
        self._check_api_availability()
        
        # Set initial API availability
        for api in self.apis:
            API_AVAILABILITY.labels(api["name"]).set(1)  # Default to available
        
        # Cache last price update time for each symbol
        self.last_price_update = {}
        
        # Initial call to avoid calling too many APIs at once
        random.shuffle(self.apis)
        logger.info("CryptoPriceFetcher initialized")
    
    def _start_metrics_server(self):
        """Start the metrics HTTP server if not already started"""
        try:
            # Check if a metrics server is already running by trying to bind to the port
            metrics_port = int(os.environ.get("METRICS_PORT", 8001))
            
            # Try to start the server, handle if already running
            try:
                start_http_server(metrics_port)
                logger.info(f"Prometheus metrics server started on port {metrics_port}")
            except OSError as e:
                if "Address already in use" in str(e):
                    logger.info(f"Metrics server already running on port {metrics_port}")
                else:
                    raise
        except Exception as e:
            logger.error(f"Error starting metrics server: {e}")
    
    def _check_api_availability(self):
        """Check the availability of all configured APIs"""
        for api in self.apis:
            try:
                # Use a simple DNS check to see if the API endpoint is reachable
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
        """Load price history from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    history = json.load(f)
                logger.info(f"Loaded price history from {self.data_file}")
                
                # Update metrics for loaded prices
                for symbol, data in history.items():
                    if data and len(data) > 0:
                        latest_price = data[-1].get("price", 0)
                        CURRENT_PRICES.labels(symbol).set(latest_price)
                        
                        # Update price distribution histogram
                        PRICE_DISTRIBUTION.labels(symbol).observe(latest_price)
                        
                        # Calculate price change percentages for different periods
                        self._update_price_change_metrics(symbol, data)
                return history
            return {}
        except Exception as e:
            logger.error(f"Error loading price history: {e}")
            return {}
    
    def _update_price_change_metrics(self, symbol, price_data):
        """Update metrics for price changes over different periods"""
        if not price_data or len(price_data) < 2:
            return
        
        latest_price = price_data[-1].get("price", 0)
        latest_time = datetime.fromisoformat(price_data[-1].get("timestamp")) if isinstance(price_data[-1].get("timestamp"), str) else price_data[-1].get("timestamp")
        
        # Define periods to calculate change over
        periods = [
            ("1h", timedelta(hours=1)),
            ("24h", timedelta(hours=24)),
            ("7d", timedelta(days=7)),
            ("30d", timedelta(days=30))
        ]
        
        for period_name, time_delta in periods:
            cutoff_time = latest_time - time_delta
            
            # Find the closest data point to the cutoff time
            prev_price = None
            for data_point in reversed(price_data):
                point_time = datetime.fromisoformat(data_point.get("timestamp")) if isinstance(data_point.get("timestamp"), str) else data_point.get("timestamp")
                
                if point_time <= cutoff_time:
                    prev_price = data_point.get("price", 0)
                    break
            
            # Calculate change if we found a previous price
            if prev_price and prev_price > 0:
                price_change_pct = ((latest_price / prev_price) - 1) * 100
                PRICE_CHANGE_PCT.labels(symbol=symbol, period=period_name).set(price_change_pct)
    
    def _save_history(self):
        """Save price history to file"""
        try:
            if self.save_data:
                with open(self.data_file, 'w') as f:
                    json.dump(self.price_history, f, indent=2)
                logger.debug("Price history saved to file")
        except Exception as e:
            logger.error(f"Error saving price history: {e}")
            
    def _store_price(self, symbol, price, source="unknown"):
        """Store price in history"""
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            
        self.price_history[symbol].append({
            "timestamp": timestamp_str,
            "price": price,
            "source": source
        })
        
        # Keep history manageable (last 10000 entries)
        if len(self.price_history[symbol]) > 10000:
            self.price_history[symbol] = self.price_history[symbol][-10000:]
            
        self._save_history()
        
        # Update metrics
        CURRENT_PRICES.labels(symbol).set(price)
        PRICE_UPDATES.labels(symbol=symbol, source=source).inc()
        PRICE_DISTRIBUTION.labels(symbol).observe(price)
        self.last_price_update[symbol] = time.time()
        
        # Update price change metrics if we have enough data
        if len(self.price_history[symbol]) > 1:
            self._update_price_change_metrics(symbol, self.price_history[symbol])
            
    def _is_dns_working(self, hostname):
        """Check if DNS resolution is working"""
        try:
            socket.gethostbyname(hostname)
            return True
        except:
            return False
            
    def _get_mock_price(self, symbol, reason="unknown"):
        """Generate mock price based on historical data or defaults"""
        try:
            # Base prices
            starting_prices = {
                "BTC": 85000,
                "ETH": 2200,
                "SOL": 140,
                "DOGE": 0.20
            }
            
            # Get last price if available
            last_price = None
            if symbol in self.price_history and self.price_history[symbol]:
                last_price = self.price_history[symbol][-1]["price"]
            
            # Generate new price
            if last_price:
                # Random change up to 1%
                change = (random.random() * 2 - 1) * 0.01
                price = last_price * (1 + change)
            else:
                price = starting_prices.get(symbol, 1000)
            
            # Log and update metrics    
            self._store_price(symbol, price, source="mock_data")
            MOCK_PRICE_USES.labels(symbol=symbol, reason=reason).inc()
            logger.info(f"Using mock data for {symbol}: ${price} (reason: {reason})")
            return price
        except Exception as e:
            logger.error(f"Error generating mock price: {e}")
            price = starting_prices.get(symbol, 1000)
            MOCK_PRICE_USES.labels(symbol=symbol, reason="error_fallback").inc()
            logger.warning(f"Using default mock data for {symbol}: ${price}")
            return price
            
    def get_price(self, symbol="BTC", use_mock=False):
        """Get current price for a cryptocurrency"""
        # Update price age metric if we have a recorded last update time
        if symbol in self.last_price_update:
            age_seconds = time.time() - self.last_price_update.get(symbol, time.time())
            PRICE_AGE_SECONDS.labels(symbol).set(age_seconds)
        
        if use_mock:
            return self._get_mock_price(symbol, reason="explicitly_requested")
            
        # First check if DNS is working
        if not self._is_dns_working("api.coingecko.com"):
            logger.warning("DNS resolution failed, using mock data")
            return self._get_mock_price(symbol, reason="dns_failure")
        
        # Try each API until we get a price
        random.shuffle(self.apis)  # Randomize order to distribute load
        
        for api in self.apis:
            try:
                crypto_id = api["symbol_map"].get(symbol)
                if not crypto_id:
                    logger.debug(f"{api['name']} API doesn't support {symbol}, skipping...")
                    continue
                    
                # Check if DNS resolution works for this API
                api_domain = api["url"].split("//")[1].split("/")[0]
                if not self._is_dns_working(api_domain):
                    logger.warning(f"DNS resolution failed for {api_domain}, skipping {api['name']} API...")
                    API_AVAILABILITY.labels(api["name"]).set(0)
                    continue
                else:
                    # Update API availability metric
                    API_AVAILABILITY.labels(api["name"]).set(1)
                
                # Apply rate limiting
                rate_limiter_key = api.get("rate_limiter", "default")
                if rate_limiter_key in self.rate_limiters:
                    self.rate_limiters[rate_limiter_key].wait(api["name"])
                    
                url = api["url"].format(id=crypto_id)
                logger.debug(f"Trying {api['name']} API...")
                
                # Use the request time summary to track API performance
                with REQUEST_DURATION.labels(api["name"]).time():
                    response = requests.get(url, timeout=10)
                    
                # Update API request metrics
                API_REQUESTS.labels(api=api["name"], symbol=symbol, status=str(response.status_code)).inc()
                
                if response.status_code != 200:
                    logger.warning(f"Non-200 response from {api['name']} API: {response.status_code}")
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                price = api["extract"](data, crypto_id)
                
                # Add to history
                self._store_price(symbol, price, source=api["name"])
                logger.info(f"Successfully fetched {symbol} price from {api['name']}: ${price:.2f}")
                
                return price
            except KeyError as e:
                logger.warning(f"Error with {api['name']} API: Unexpected response format - missing key {str(e)}")
                API_REQUESTS.labels(api=api["name"], symbol=symbol, status="key_error").inc()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error with {api['name']} API: {str(e)}")
                API_REQUESTS.labels(api=api["name"], symbol=symbol, status="request_error").inc()
                # Update API availability if connection failed
                if "Connection" in str(e) or "Timeout" in str(e):
                    API_AVAILABILITY.labels(api["name"]).set(0)
            except (TypeError, ValueError) as e:
                logger.warning(f"Error parsing data from {api['name']} API: {str(e)}")
                API_REQUESTS.labels(api=api["name"], symbol=symbol, status="parse_error").inc()
            except Exception as e:
                logger.warning(f"Unexpected error with {api['name']} API: {str(e)}")
                API_REQUESTS.labels(api=api["name"], symbol=symbol, status="other_error").inc()
                # Include traceback for unexpected errors
                logger.debug(traceback.format_exc())
                
        # If all APIs failed, use mock data
        logger.warning("All APIs failed, falling back to mock data.")
        return self._get_mock_price(symbol, reason="api_failure")
    
    def get_price_history(self, symbol):
        """Get historical price data for a symbol"""
        if symbol in self.price_history:
            return self.price_history[symbol]
        return []