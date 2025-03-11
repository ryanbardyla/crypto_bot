"""
Redis Client: Handles connections and operations with Redis.
"""

import redis
import logging
from typing import Optional, Any
from core.config_loader import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedisClient:
    """
    A client for interacting with Redis.
    """
    def __init__(self):
        """
        Initialize the Redis client using settings from config.
        """
        config = load_config()
        self.host = config["redis"]["host"]
        self.port = config["redis"]["port"]
        self.client = redis.Redis(host=self.host, port=self.port, decode_responses=True)
        try:
            self.client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def set(self, key: str, value: str) -> None:
        """
        Set a key-value pair in Redis.

        Args:
            key (str): The key to set.
            value (str): The value to associate with the key.
        """
        try:
            self.client.set(key, value)
            logger.debug(f"Set key '{key}' in Redis")
        except redis.RedisError as e:
            logger.error(f"Failed to set key '{key}': {e}")
            raise

    def get(self, key: str) -> Optional[str]:
        """
        Get the value associated with a key in Redis.

        Args:
            key (str): The key to retrieve.

        Returns:
            Optional[str]: The value if the key exists, else None.
        """
        try:
            value = self.client.get(key)
            if value:
                logger.debug(f"Retrieved key '{key}' from Redis")
            else:
                logger.warning(f"Key '{key}' not found in Redis")
            return value
        except redis.RedisError as e:
            logger.error(f"Failed to get key '{key}': {e}")
            raise

    def delete(self, key: str) -> None:
        """
        Delete a key from Redis.

        Args:
            key (str): The key to delete.
        """
        try:
            self.client.delete(key)
            logger.debug(f"Deleted key '{key}' from Redis")
        except redis.RedisError as e:
            logger.error(f"Failed to delete key '{key}': {e}")
            raise

# Example usage (for testing purposes)
if __name__ == "__main__":
    try:
        redis_client = RedisClient()
        redis_client.set("test_key", "test_value")
        value = redis_client.get("test_key")
        print(f"Retrieved value: {value}")
        redis_client.delete("test_key")
    except Exception as e:
        print(f"Error: {e}")