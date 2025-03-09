# In multi_api_price_fetcher.py

import os
import time
import json
import random
import threading
import requests
import logging
import socket
from datetime import datetime

# Import the centralized logging configuration
from utils.logging_config import get_module_logger

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
        
    def wait(self):
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
                    time.sleep(max(0, sleep_time))
                    self.call_count = 1
                    self.reset_time = time.time() + 1.0
            
            # Regular rate limiting
            elapsed = time.time() - self.last_call
            if elapsed < (1.0 / self.rate):
                sleep_time = (1.0 / self.rate) - elapsed
                time.sleep(sleep_time)
            
            self.last_call = time.time()

class CryptoPriceFetcher:
    def __init__(self, save_data=True):
        self.data_file = "price_history.json"
        self.price_history = self._load_history()
        self.save_data = save_data
        self.rate_limiters = {
            "coingecko": RateLimiter(calls_per_second=0.5, burst_limit=10),  # 10 calls per 20 seconds
            "coinbase": RateLimiter(calls_per_second=3, burst_limit=30),     # 30 calls per 10 seconds
            "kraken": RateLimiter(calls_per_second=1, burst_limit=15),       # 15 calls per 15 seconds
            "binance": RateLimiter(calls_per_second=10, burst_limit=100)     # 100 calls per 10 seconds
        }
        
        # API configuration
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
        
        # Initial call to avoid calling too many APIs at once
        random.shuffle(self.apis)
        logger.info("CryptoPriceFetcher initialized")
        
    def _load_history(self):
        """Load price history from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading price history: {e}")
            return {}
    
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
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            
        self.price_history[symbol].append({
            "timestamp": timestamp,
            "price": price,
            "source": source
        })
        
        # Keep history manageable (last 10000 entries)
        if len(self.price_history[symbol]) > 10000:
            self.price_history[symbol] = self.price_history[symbol][-10000:]
            
        self._save_history()
        
    def _is_dns_working(self, hostname):
        """Check if DNS resolution is working"""
        try:
            socket.gethostbyname(hostname)
            return True
        except:
            return False
            
    def _get_mock_price(self, symbol):
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
                
            self._store_price(symbol, price, source="mock_data")
            logger.info(f"Using mock data for {symbol}: ${price}")
            return price
        except Exception as e:
            logger.error(f"Error generating mock price: {e}")
            price = starting_prices.get(symbol, 1000)
            logger.warning(f"Using default mock data for {symbol}: ${price}")
            return price
            
    def get_price(self, symbol="BTC", use_mock=False):
        """Get current price for a cryptocurrency"""
        if use_mock:
            return self._get_mock_price(symbol)
            
        # First check if DNS is working
        if not self._is_dns_working("api.coingecko.com"):
            logger.warning("DNS resolution failed, using mock data")
            return self._get_mock_price(symbol)
        
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
                    continue
                
                # Apply rate limiting
                rate_limiter_key = api.get("rate_limiter", "default")
                if rate_limiter_key in self.rate_limiters:
                    self.rate_limiters[rate_limiter_key].wait()
                    
                url = api["url"].format(id=crypto_id)
                logger.debug(f"Trying {api['name']} API...")
                
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                price = api["extract"](data, crypto_id)
                
                # Add to history
                self._store_price(symbol, price, source=api["name"])
                logger.info(f"Successfully fetched {symbol} price from {api['name']}: ${price:.2f}")
                
                return price
            except KeyError as e:
                logger.warning(f"Error with {api['name']} API: Unexpected response format - missing key {str(e)}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error with {api['name']} API: {str(e)}")
            except (TypeError, ValueError) as e:
                logger.warning(f"Error parsing data from {api['name']} API: {str(e)}")
            except Exception as e:
                logger.warning(f"Unexpected error with {api['name']} API: {str(e)}")
                
        # If all APIs failed, use mock data
        logger.warning("All APIs failed, falling back to mock data.")
        return self._get_mock_price(symbol)
    
    def get_price_history(self, symbol):
        """Get historical price data for a symbol"""
        if symbol in self.price_history:
            return self.price_history[symbol]
        return []