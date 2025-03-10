#!/usr/bin/env python3
"""
cache_manager.py - Redis Cache Manager

This module provides a robust implementation for caching data using Redis
for the crypto trading system. It helps reduce API calls and improves 
performance by caching frequently accessed data.

Features:
- Automatic key expiration
- Serialization/deserialization of complex objects
- Namespace support to avoid key collisions
- Pattern-based cache invalidation
- Cache statistics and monitoring
- Support for various data types (strings, hashes, lists, etc.)
- Resilient connection handling
"""

import os
import json
import time
import logging
import pickle
import hashlib
import threading
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from functools import wraps

import redis
from redis.exceptions import RedisError, ConnectionError

# Configure logging
from utils.logging_config import setup_logging

logger = setup_logging(name="cache_manager")

class CacheManager:
    """Redis cache manager for the crypto trading system."""
    
    def __init__(
        self, 
        host: str = None,
        port: int = None,
        db: int = 0,
        password: str = None,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        namespace: str = 'crypto',
        retry_on_timeout: bool = True,
        retry_on_error: bool = True,
        max_retries: int = 3,
        health_check_interval: int = 30,
        use_pool: bool = True,
        pool_min_connections: int = 1,
        pool_max_connections: int = 10
    ):
        """
        Initialize the CacheManager with Redis connection parameters.
        
        Args:
            host: Redis server hostname or IP
            port: Redis server port
            db: Redis database number
            password: Redis password
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connection timeout in seconds
            namespace: Namespace for cache keys
            retry_on_timeout: Whether to retry on timeout
            retry_on_error: Whether to retry on error
            max_retries: Maximum number of retries
            health_check_interval: Health check interval in seconds
            use_pool: Whether to use connection pooling
            pool_min_connections: Minimum number of connections in the pool
            pool_max_connections: Maximum number of connections in the pool
        """
        # Get connection parameters from environment variables if not provided
        self.host = host or os.environ.get('REDIS_HOST', 'localhost')
        self.port = int(port or os.environ.get('REDIS_PORT', 6379))
        self.db = db
        self.password = password or os.environ.get('REDIS_PASSWORD', None)
        
        # Connection settings
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout
        self.retry_on_error = retry_on_error
        self.max_retries = max_retries
        self.health_check_interval = health_check_interval
        
        # Namespace
        self.namespace = namespace
        
        # Connection pool settings
        self.use_pool = use_pool
        self.pool_min_connections = pool_min_connections
        self.pool_max_connections = pool_max_connections
        
        # Redis client or connection pool
        self._redis = None
        
        # Thread-local storage for Redis client
        self._local = threading.local()
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Cache statistics
        self.stats = {
            'gets': 0,
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        
        # Initialize connection
        self.initialize()
        
        logger.info("CacheManager initialized with host %s:%d, namespace %s", 
                   self.host, self.port, self.namespace)

    def initialize(self) -> bool:
        """
        Initialize the Redis connection or pool.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self._lock:
                # Common connection parameters
                params = {
                    'host': self.host,
                    'port': self.port,
                    'db': self.db,
                    'password': self.password,
                    'socket_timeout': self.socket_timeout,
                    'socket_connect_timeout': self.socket_connect_timeout,
                    'retry_on_timeout': self.retry_on_timeout,
                    'health_check_interval': self.health_check_interval,
                    'decode_responses': False
                }
                
                if self.use_pool:
                    # Create connection pool
                    self._redis = redis.ConnectionPool(
                        min_connections=self.pool_min_connections,
                        max_connections=self.pool_max_connections,
                        **params
                    )
                    logger.info("Redis connection pool initialized with %d-%d connections", 
                               self.pool_min_connections, self.pool_max_connections)
                else:
                    # Create direct client
                    self._redis = redis.Redis(**params)
                    
                # Test connection
                if self.use_pool:
                    test_client = redis.Redis(connection_pool=self._redis)
                    test_client.ping()
                else:
                    self._redis.ping()
                    
                logger.info("Redis connection successful")
                return True
                
        except (RedisError, ConnectionError) as e:
            logger.error("Failed to initialize Redis connection: %s", str(e))
            self.stats['errors'] += 1
            return False
            
        except Exception as e:
            logger.error("Unexpected error initializing Redis connection: %s", str(e))
            self.stats['errors'] += 1
            return False

    def _get_client(self) -> redis.Redis:
        """
        Get a Redis client instance.
        
        For connection pool mode, creates a new client using the pool.
        For direct connection mode, returns the existing client.
        
        Returns:
            redis.Redis: Redis client
        """
        if not self._redis:
            if not self.initialize():
                raise ConnectionError("Failed to initialize Redis connection")
        
        if self.use_pool:
            # Create a thread-local client if it doesn't exist
            if not hasattr(self._local, 'client'):
                self._local.client = redis.Redis(connection_pool=self._redis)
            return self._local.client
        else:
            return self._redis

    def _format_key(self, key: str) -> str:
        """
        Format a key with the namespace.
        
        Args:
            key: The key to format
            
        Returns:
            str: The formatted key
        """
        return f"{self.namespace}:{key}"

    def _serialize(self, value: Any) -> bytes:
        """
        Serialize a value for storage in Redis.
        
        Args:
            value: The value to serialize
            
        Returns:
            bytes: The serialized value
        """
        if value is None:
            return b'null'
            
        if isinstance(value, (str, int, float, bool)):
            return json.dumps(value).encode('utf-8')
            
        try:
            # Try JSON serialization first for better interoperability
            return json.dumps(value).encode('utf-8')
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            return pickle.dumps(value)

    def _deserialize(self, value: bytes) -> Any:
        """
        Deserialize a value from Redis.
        
        Args:
            value: The value to deserialize
            
        Returns:
            Any: The deserialized value
        """
        if value is None:
            return None
            
        if value == b'null':
            return None
            
        try:
            # Try JSON deserialization first
            return json.loads(value.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle for complex objects
            try:
                return pickle.loads(value)
            except pickle.PickleError:
                # Return as bytes if all else fails
                return value

    def _execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a Redis command with retry logic.
        
        Args:
            func: The Redis function to call
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Any: The result of the function call
        """
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                return func(*args, **kwargs)
            except (RedisError, ConnectionError) as e:
                last_error = e
                retries += 1
                
                if not self.retry_on_error:
                    break
                    
                if retries <= self.max_retries:
                    logger.warning("Redis operation failed: %s. Retrying %d/%d...", 
                                  str(e), retries, self.max_retries)
                    time.sleep(0.1 * (2 ** retries))  # Exponential backoff
                    
                    # Try to re-initialize connection
                    if isinstance(e, ConnectionError):
                        self.initialize()
                        
            except Exception as e:
                logger.error("Unexpected error in Redis operation: %s", str(e))
                self.stats['errors'] += 1
                raise
                
        logger.error("Redis operation failed after %d retries: %s", 
                    retries, str(last_error))
        self.stats['errors'] += 1
        raise last_error

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: The cache key
            default: The default value to return if the key is not found
            
        Returns:
            Any: The cached value or default
        """
        formatted_key = self._format_key(key)
        self.stats['gets'] += 1
        
        try:
            client = self._get_client()
            result = self._execute_with_retry(client.get, formatted_key)
            
            if result is None:
                self.stats['misses'] += 1
                return default
                
            self.stats['hits'] += 1
            return self._deserialize(result)
            
        except Exception as e:
            logger.error("Error getting key %s: %s", key, str(e))
            self.stats['errors'] += 1
            return default

    def set(
        self, 
        key: str, 
        value: Any, 
        ex: Optional[int] = None, 
        px: Optional[int] = None, 
        nx: bool = False, 
        xx: bool = False
    ) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ex: Expiry in seconds
            px: Expiry in milliseconds
            nx: Only set if the key does not exist
            xx: Only set if the key already exists
            
        Returns:
            bool: True if successful, False otherwise
        """
        formatted_key = self._format_key(key)
        serialized_value = self._serialize(value)
        self.stats['sets'] += 1
        
        try:
            client = self._get_client()
            result = self._execute_with_retry(
                client.set,
                formatted_key,
                serialized_value,
                ex=ex,
                px=px,
                nx=nx,
                xx=xx
            )
            
            return result
            
        except Exception as e:
            logger.error("Error setting key %s: %s", key, str(e))
            self.stats['errors'] += 1
            return False

    def delete(self, *keys) -> int:
        """
        Delete one or more keys from the cache.
        
        Args:
            *keys: The keys to delete
            
        Returns:
            int: Number of keys deleted
        """
        if not keys:
            return 0
            
        formatted_keys = [self._format_key(key) for key in keys]
        self.stats['deletes'] += 1
        
        try:
            client = self._get_client()
            count = self._execute_with_retry(client.delete, *formatted_keys)
            
            return count
            
        except Exception as e:
            logger.error("Error deleting keys %s: %s", keys, str(e))
            self.stats['errors'] += 1
            return 0

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key: The cache key
            
        Returns:
            bool: True if the key exists, False otherwise
        """
        formatted_key = self._format_key(key)
        
        try:
            client = self._get_client()
            result = self._execute_with_retry(client.exists, formatted_key)
            
            return bool(result)
            
        except Exception as e:
            logger.error("Error checking existence of key %s: %s", key, str(e))
            self.stats['errors'] += 1
            return False

    def ttl(self, key: str) -> int:
        """
        Get the time-to-live for a key in seconds.
        
        Args:
            key: The cache key
            
        Returns:
            int: TTL in seconds, -1 if no expiry, -2 if key doesn't exist
        """
        formatted_key = self._format_key(key)
        
        try:
            client = self._get_client()
            return self._execute_with_retry(client.ttl, formatted_key)
            
        except Exception as e:
            logger.error("Error getting TTL for key %s: %s", key, str(e))
            self.stats['errors'] += 1
            return -2

    def expire(self, key: str, seconds: int) -> bool:
        """
        Set the expiry time for a key.
        
        Args:
            key: The cache key
            seconds: Time in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        formatted_key = self._format_key(key)
        
        try:
            client = self._get_client()
            return self._execute_with_retry(client.expire, formatted_key, seconds)
            
        except Exception as e:
            logger.error("Error setting expiry for key %s: %s", key, str(e))
            self.stats['errors'] += 1
            return False

    def hget(self, name: str, key: str, default: Any = None) -> Any:
        """
        Get a value from a hash in the cache.
        
        Args:
            name: The hash name
            key: The hash key
            default: The default value to return if the key is not found
            
        Returns:
            Any: The cached value or default
        """
        formatted_name = self._format_key(name)
        self.stats['gets'] += 1
        
        try:
            client = self._get_client()
            result = self._execute_with_retry(client.hget, formatted_name, key)
            
            if result is None:
                self.stats['misses'] += 1
                return default
                
            self.stats['hits'] += 1
            return self._deserialize(result)
            
        except Exception as e:
            logger.error("Error getting hash key %s.%s: %s", name, key, str(e))
            self.stats['errors'] += 1
            return default

    def hset(self, name: str, key: str, value: Any) -> int:
        """
        Set a value in a hash in the cache.
        
        Args:
            name: The hash name
            key: The hash key
            value: The value to cache
            
        Returns:
            int: 1 if field is new, 0 if field was updated
        """
        formatted_name = self._format_key(name)
        serialized_value = self._serialize(value)
        self.stats['sets'] += 1
        
        try:
            client = self._get_client()
            return self._execute_with_retry(
                client.hset,
                formatted_name,
                key,
                serialized_value
            )
            
        except Exception as e:
            logger.error("Error setting hash key %s.%s: %s", name, key, str(e))
            self.stats['errors'] += 1
            return 0

    def hgetall(self, name: str) -> Dict[str, Any]:
        """
        Get all key-value pairs from a hash in the cache.
        
        Args:
            name: The hash name
            
        Returns:
            Dict[str, Any]: Dictionary of key-value pairs
        """
        formatted_name = self._format_key(name)
        self.stats['gets'] += 1
        
        try:
            client = self._get_client()
            result = self._execute_with_retry(client.hgetall, formatted_name)
            
            if not result:
                self.stats['misses'] += 1
                return {}
                
            self.stats['hits'] += 1
            
            # Deserialize all values
            return {
                key.decode('utf-8') if isinstance(key, bytes) else key: 
                self._deserialize(value)
                for key, value in result.items()
            }
            
        except Exception as e:
            logger.error("Error getting all hash keys for %s: %s", name, str(e))
            self.stats['errors'] += 1
            return {}

    def hdel(self, name: str, *keys) -> int:
        """
        Delete one or more keys from a hash in the cache.
        
        Args:
            name: The hash name
            *keys: The keys to delete
            
        Returns:
            int: Number of keys deleted
        """
        if not keys:
            return 0
            
        formatted_name = self._format_key(name)
        self.stats['deletes'] += 1
        
        try:
            client = self._get_client()
            return self._execute_with_retry(client.hdel, formatted_name, *keys)
            
        except Exception as e:
            logger.error("Error deleting hash keys %s from %s: %s", keys, name, str(e))
            self.stats['errors'] += 1
            return 0

    def incr(self, key: str, amount: int = 1) -> int:
        """
        Increment a key by a given amount.
        
        Args:
            key: The cache key
            amount: The amount to increment by
            
        Returns:
            int: The new value
        """
        formatted_key = self._format_key(key)
        
        try:
            client = self._get_client()
            if amount == 1:
                return self._execute_with_retry(client.incr, formatted_key)
            else:
                return self._execute_with_retry(client.incrby, formatted_key, amount)
            
        except Exception as e:
            logger.error("Error incrementing key %s: %s", key, str(e))
            self.stats['errors'] += 1
            return 0

    def decr(self, key: str, amount: int = 1) -> int:
        """
        Decrement a key by a given amount.
        
        Args:
            key: The cache key
            amount: The amount to decrement by
            
        Returns:
            int: The new value
        """
        formatted_key = self._format_key(key)
        
        try:
            client = self._get_client()
            if amount == 1:
                return self._execute_with_retry(client.decr, formatted_key)
            else:
                return self._execute_with_retry(client.decrby, formatted_key, amount)
            
        except Exception as e:
            logger.error("Error decrementing key %s: %s", key, str(e))
            self.stats['errors'] += 1
            return 0

    def keys(self, pattern: str) -> List[str]:
        """
        Find all keys matching a pattern.
        
        WARNING: This command should be used with caution in production as it may impact performance.
        
        Args:
            pattern: The pattern to match
            
        Returns:
            List[str]: List of matching keys
        """
        formatted_pattern = self._format_key(pattern)
        
        try:
            client = self._get_client()
            result = self._execute_with_retry(client.keys, formatted_pattern)
            
            # Remove namespace from keys
            prefix_len = len(self.namespace) + 1  # +1 for the colon
            return [
                key.decode('utf-8')[prefix_len:] if isinstance(key, bytes) else key[prefix_len:]
                for key in result
            ]
            
        except Exception as e:
            logger.error("Error finding keys matching pattern %s: %s", pattern, str(e))
            self.stats['errors'] += 1
            return []

    def flush(self, pattern: str = "*") -> int:
        """
        Delete all keys matching a pattern.
        
        Args:
            pattern: The pattern to match
            
        Returns:
            int: Number of keys deleted
        """
        keys = self.keys(pattern)
        
        if not keys:
            return 0
            
        return self.delete(*keys)

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, int]: Dictionary of cache statistics
        """
        stats = dict(self.stats)
        
        # Calculate hit rate
        total_gets = stats['hits'] + stats['misses']
        stats['hit_rate'] = (stats['hits'] / total_gets * 100) if total_gets > 0 else 0
        
        return stats

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self.stats = {
            'gets': 0,
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }

    def cache_key_hash(self, *args, **kwargs) -> str:
        """
        Generate a cache key hash from arguments.
        
        Args:
            *args: Positional arguments to include in the hash
            **kwargs: Keyword arguments to include in the hash
            
        Returns:
            str: Cache key hash
        """
        # Combine args and kwargs into a single string
        key_parts = []
        
        for arg in args:
            key_parts.append(str(arg))
            
        # Sort kwargs for consistent ordering
        for key in sorted(kwargs.keys()):
            key_parts.append(f"{key}={kwargs[key]}")
            
        # Join parts and compute hash
        key_string = ":".join(key_parts)
        key_hash = hashlib.md5(key_string.encode('utf-8')).hexdigest()
        
        return key_hash

    def cached(self, ttl: int = 3600, prefix: str = "cached", key_func: Callable = None):
        """
        Decorator for caching function results.
        
        Args:
            ttl: Time-to-live in seconds
            prefix: Prefix for cache keys
            key_func: Optional function to generate cache keys
            
        Returns:
            Callable: Decorated function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    func_name = func.__name__
                    arg_hash = self.cache_key_hash(*args, **kwargs)
                    cache_key = f"{prefix}:{func_name}:{arg_hash}"
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                
                if cached_result is not None:
                    logger.debug("Cache hit for %s", cache_key)
                    return cached_result
                
                # Call function and cache result
                logger.debug("Cache miss for %s", cache_key)
                result = func(*args, **kwargs)
                
                # Cache result if it's not None
                if result is not None:
                    self.set(cache_key, result, ex=ttl)
                    
                return result
                
            return wrapper
        return decorator

    def cache_price(self, symbol: str, price_data: Dict[str, Any], ttl: int = 60) -> bool:
        """
        Cache price data for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            price_data: Price data dictionary
            ttl: Time-to-live in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        key = f"price:{symbol.upper()}"
        
        # Add timestamp if not present
        if 'timestamp' not in price_data:
            price_data['timestamp'] = datetime.now().isoformat()
            
        return self.set(key, price_data, ex=ttl)

    def get_cached_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get cached price data for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Optional[Dict[str, Any]]: Price data dictionary or None if not cached
        """
        key = f"price:{symbol.upper()}"
        return self.get(key)

    def cache_sentiment(
        self, 
        source: str, 
        record_id: str, 
        sentiment_data: Dict[str, Any], 
        ttl: int = 3600
    ) -> bool:
        """
        Cache sentiment data.
        
        Args:
            source: Source of the sentiment data (e.g., 'youtube', 'twitter')
            record_id: Unique record ID
            sentiment_data: Sentiment data dictionary
            ttl: Time-to-live in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        key = f"sentiment:{source}:{record_id}"
        
        # Add timestamp if not present
        if 'timestamp' not in sentiment_data:
            sentiment_data['timestamp'] = datetime.now().isoformat()
            
        return self.set(key, sentiment_data, ex=ttl)

    def get_cached_sentiment(self, source: str, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached sentiment data.
        
        Args:
            source: Source of the sentiment data
            record_id: Unique record ID
            
        Returns:
            Optional[Dict[str, Any]]: Sentiment data dictionary or None if not cached
        """
        key = f"sentiment:{source}:{record_id}"
        return self.get(key)

    def cache_api_response(
        self, 
        api: str, 
        endpoint: str, 
        params: Dict[str, Any], 
        response: Any, 
        ttl: int = 60
    ) -> bool:
        """
        Cache an API response.
        
        Args:
            api: API name
            endpoint: API endpoint
            params: API parameters
            response: API response
            ttl: Time-to-live in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Generate a hash of the parameters
        param_hash = self.cache_key_hash(**params)
        key = f"api:{api}:{endpoint}:{param_hash}"
        
        return self.set(key, response, ex=ttl)

    def get_cached_api_response(
        self, 
        api: str, 
        endpoint: str, 
        params: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Get a cached API response.
        
        Args:
            api: API name
            endpoint: API endpoint
            params: API parameters
            
        Returns:
            Optional[Any]: API response or None if not cached
        """
        # Generate a hash of the parameters
        param_hash = self.cache_key_hash(**params)
        key = f"api:{api}:{endpoint}:{param_hash}"
        
        return self.get(key)

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the Redis connection.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        start_time = time.time()
        healthy = False
        error = None
        
        try:
            client = self._get_client()
            client.ping()
            healthy = True
            
        except Exception as e:
            error = str(e)
            
        end_time = time.time()
        response_time = (end_time - start_time) * 1000  # ms
        
        return {
            'healthy': healthy,
            'response_time_ms': response_time,
            'error': error,
            'stats': self.get_stats()
        }

    def close(self) -> None:
        """Close Redis connection."""
        try:
            if self.use_pool and self._redis:
                # For pool mode, close the pool
                self._redis.disconnect()
            elif self._redis:
                # For direct connection mode, close the client
                self._redis.close()
                
            logger.info("Redis connection closed")
            
        except Exception as e:
            logger.error("Error closing Redis connection: %s", str(e))

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()


# Usage example
if __name__ == "__main__":
    # Create a cache manager instance
    cache = CacheManager(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", 6379)),
        password=os.environ.get("REDIS_PASSWORD", None),
        namespace="crypto_test"
    )
    
    try:
        # Test connection
        health = cache.health_check()
        
        if health['healthy']:
            print(f"Connected to Redis: response time {health['response_time_ms']:.2f} ms")
        else:
            print(f"Redis connection failed: {health['error']}")
            exit(1)
        
        # Example: Cache price data
        sample_price = {
            "price": 50000.0,
            "timestamp": datetime.now().isoformat(),
            "volume": 1234.56,
            "source": "binance"
        }
        
        cache.cache_price("BTC", sample_price, ttl=30)
        print("Cached BTC price data")
        
        # Retrieve cached price
        cached_price = cache.get_cached_price("BTC")
        print(f"Retrieved cached BTC price: ${cached_price['price']:.2f}")
        
        # Example: Test cache decorator
        @cache.cached(ttl=60, prefix="example")
        def slow_function(x, y):
            print("Executing slow function...")
            time.sleep(1)  # Simulate slow operation
            return x + y
        
        print("First call (should be slow):")
        result1 = slow_function(5, 7)
        print(f"Result: {result1}")
        
        print("\nSecond call (should be fast, cached):")
        result2 = slow_function(5, 7)
        print(f"Result: {result2}")
        
        # Print cache statistics
        print("\nCache statistics:")
        for key, value in cache.get_stats().items():
            print(f"  {key}: {value}")
        
    finally:
        # Close cache connection
        cache.close()