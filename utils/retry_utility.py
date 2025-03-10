#!/usr/bin/env python3
"""
retry_utility.py - Robust Error Handling and Retry Mechanisms

This module provides a robust implementation for handling errors and retrying
operations with exponential backoff. It's designed to be used throughout the
crypto trading system to improve resilience against API failures, network issues,
and transient errors.

Features:
- Decorator for retrying functions with customizable behavior
- Support for exponential backoff with jitter
- Specific exception handling
- Customizable retry conditions
- Detailed logging
- Circuit breaker pattern implementation
"""

import time
import random
import logging
import functools
from typing import List, Callable, Optional, Type, Any, Dict, Union
from datetime import datetime, timedelta
import threading
from enum import Enum

# Configure logging
from utils.logging_config import setup_logging

logger = setup_logging(name="retry_utility")

class RetryState(Enum):
    """Enum for circuit breaker states"""
    CLOSED = 1  # Circuit is closed, requests pass through
    OPEN = 2    # Circuit is open, requests fail fast
    HALF_OPEN = 3  # Circuit is half-open, testing if service is back


class CircuitBreaker:
    """
    Circuit breaker implementation to prevent repeated calls to failing services.
    
    When a service fails repeatedly, the circuit breaker "opens" and fails fast
    for a period of time before "half-opening" to test if the service is back.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            name: Circuit breaker name (for identification)
            failure_threshold: Number of consecutive failures before opening circuit
            recovery_timeout: Seconds to wait before transitioning from open to half-open
            half_open_max_calls: Maximum number of calls to allow in half-open state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = RetryState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        
        self._lock = threading.RLock()
        
        logger.debug(
            "Initialized circuit breaker '%s' with threshold=%d, timeout=%d",
            name, failure_threshold, recovery_timeout
        )

    def allow_request(self) -> bool:
        """
        Check if a request should be allowed through the circuit breaker.
        
        Returns:
            bool: True if request is allowed, False otherwise
        """
        with self._lock:
            if self.state == RetryState.CLOSED:
                return True
                
            if self.state == RetryState.OPEN:
                # Check if recovery timeout has elapsed
                if self.last_failure_time and datetime.now() - self.last_failure_time >= timedelta(seconds=self.recovery_timeout):
                    logger.info(
                        "Circuit '%s' transitioning from OPEN to HALF-OPEN after %d seconds",
                        self.name, self.recovery_timeout
                    )
                    self.state = RetryState.HALF_OPEN
                    self.half_open_calls = 0
                    return True
                return False
                
            # In HALF_OPEN state
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False

    def record_success(self) -> None:
        """Record a successful operation, potentially closing the circuit."""
        with self._lock:
            if self.state == RetryState.HALF_OPEN:
                logger.info("Circuit '%s' transitioning from HALF-OPEN to CLOSED after success", self.name)
                self.state = RetryState.CLOSED
                self.failure_count = 0
                self.half_open_calls = 0
            elif self.state == RetryState.CLOSED:
                self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed operation, potentially opening the circuit."""
        with self._lock:
            self.last_failure_time = datetime.now()
            
            if self.state == RetryState.CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    logger.warning(
                        "Circuit '%s' transitioning from CLOSED to OPEN after %d failures",
                        self.name, self.failure_count
                    )
                    self.state = RetryState.OPEN
                    
            elif self.state == RetryState.HALF_OPEN:
                logger.warning("Circuit '%s' transitioning from HALF-OPEN to OPEN after failure", self.name)
                self.state = RetryState.OPEN
                self.half_open_calls = 0


# Global circuit breakers registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_circuit_breakers_lock = threading.RLock()

def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name.
    
    Args:
        name: Circuit breaker name
        **kwargs: Arguments to pass to CircuitBreaker constructor if creating new
        
    Returns:
        CircuitBreaker: The circuit breaker instance
    """
    with _circuit_breakers_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, **kwargs)
        return _circuit_breakers[name]


def retry_api_call(
    max_retries: int = 3,
    backoff_factor: float = 1.5,
    jitter: bool = True,
    max_backoff: float = 60.0,
    exceptions: List[Type[Exception]] = None,
    retry_condition: Optional[Callable[[Exception], bool]] = None,
    circuit_breaker: Optional[Union[str, CircuitBreaker]] = None,
    fallback: Optional[Callable] = None
):
    """
    Decorator for retrying API calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for the backoff (e.g., 1.5 means each retry waits 1.5x longer)
        jitter: Whether to add randomness to the backoff to prevent thundering herd
        max_backoff: Maximum backoff time in seconds
        exceptions: List of exception types to catch and retry
        retry_condition: Function to determine if an exception should be retried
        circuit_breaker: Optional circuit breaker name or instance to use
        fallback: Optional function to call if all retries fail
        
    Returns:
        Callable: Decorated function
    """
    if exceptions is None:
        exceptions = [Exception]
        
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create circuit breaker if specified
            cb = None
            if circuit_breaker:
                if isinstance(circuit_breaker, str):
                    cb = get_circuit_breaker(circuit_breaker)
                else:
                    cb = circuit_breaker
                    
                # Check if circuit is open
                if not cb.allow_request():
                    logger.warning(
                        "Circuit '%s' is open, fast-failing call to %s",
                        cb.name, func.__name__
                    )
                    if fallback:
                        return fallback(*args, **kwargs)
                    raise CircuitOpenError(f"Circuit '{cb.name}' is open")
            
            retries = 0
            while True:
                try:
                    result = func(*args, **kwargs)
                    
                    # Record success if using circuit breaker
                    if cb:
                        cb.record_success()
                        
                    return result
                    
                except tuple(exceptions) as e:
                    # Check retry condition if provided
                    if retry_condition and not retry_condition(e):
                        logger.warning(
                            "Not retrying %s due to condition: %s",
                            func.__name__, str(e)
                        )
                        
                        # Record failure if using circuit breaker
                        if cb:
                            cb.record_failure()
                            
                        # Use fallback if provided
                        if fallback:
                            return fallback(*args, **kwargs)
                            
                        raise
                    
                    retries += 1
                    
                    if retries > max_retries:
                        logger.error(
                            "Failed to execute %s after %d retries: %s",
                            func.__name__, max_retries, str(e)
                        )
                        
                        # Record failure if using circuit breaker
                        if cb:
                            cb.record_failure()
                            
                        # Use fallback if provided
                        if fallback:
                            return fallback(*args, **kwargs)
                            
                        raise
                    
                    # Calculate backoff
                    backoff = min(max_backoff, backoff_factor ** retries)
                    
                    # Add jitter if enabled
                    if jitter:
                        backoff = backoff * (0.8 + 0.4 * random.random())
                    
                    logger.warning(
                        "Retrying %s in %.2f seconds after error: %s (attempt %d/%d)",
                        func.__name__, backoff, str(e), retries, max_retries
                    )
                    
                    time.sleep(backoff)
                    
                except Exception as e:
                    # Non-retryable exception
                    logger.error(
                        "Non-retryable exception in %s: %s",
                        func.__name__, str(e)
                    )
                    
                    # Record failure if using circuit breaker
                    if cb:
                        cb.record_failure()
                    
                    raise
                    
        return wrapper
    return decorator


class RetryableError(Exception):
    """Base exception for retryable errors."""
    pass


class CircuitOpenError(Exception):
    """Exception raised when a circuit breaker is open."""
    pass


class TemporaryError(RetryableError):
    """Exception for temporary issues that should be retried."""
    pass


class RateLimitError(RetryableError):
    """Exception for rate limiting issues."""
    pass


class ConnectionFailureError(RetryableError):
    """Exception for network or connection issues."""
    pass


class AuthenticationError(Exception):
    """Exception for authentication issues (not retryable)."""
    pass


# Usage examples
if __name__ == "__main__":
    # Example: Basic retry
    @retry_api_call(max_retries=3, backoff_factor=2)
    def fetch_data(url):
        print(f"Fetching data from {url}...")
        if random.random() < 0.7:  # 70% chance of failure
            raise ConnectionError("Connection failed")
        return {"data": "success"}
    
    # Example: Retry with circuit breaker
    @retry_api_call(
        max_retries=3,
        circuit_breaker="api_service",
        fallback=lambda x: {"data": "fallback"}
    )
    def call_service(endpoint):
        print(f"Calling service at {endpoint}...")
        if random.random() < 0.8:  # 80% chance of failure
            raise TimeoutError("Service timeout")
        return {"data": "service result"}
    
    # Example: Retry with specific exceptions
    @retry_api_call(
        max_retries=5,
        exceptions=[ConnectionError, TimeoutError, ValueError],
        backoff_factor=1.5,
        jitter=True
    )
    def process_data(data):
        print(f"Processing data: {data}...")
        error_type = random.randint(0, 3)
        if error_type == 0:
            raise ConnectionError("Connection lost")
        elif error_type == 1:
            raise TimeoutError("Processing timeout")
        elif error_type == 2:
            raise ValueError("Invalid data format")
        return {"result": "processed"}
    
    # Test retry with basic function
    try:
        result = fetch_data("https://api.example.com/data")
        print(f"Result: {result}")
    except Exception as e:
        print(f"All retries failed: {e}")
    
    # Test retry with circuit breaker
    for i in range(10):
        try:
            result = call_service("/api/endpoint")
            print(f"Service call #{i+1} result: {result}")
        except Exception as e:
            print(f"Service call #{i+1} failed: {e}")
    
    # Test retry with specific exceptions
    try:
        result = process_data({"id": 123})
        print(f"Processing result: {result}")
    except Exception as e:
        print(f"Processing failed after all retries: {e}")