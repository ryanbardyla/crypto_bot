#!/usr/bin/env python3
"""
message_broker.py - RabbitMQ Message Broker Implementation

This module provides a robust implementation for message passing between
components of the crypto trading system using RabbitMQ.

Features:
- Connection pooling and automatic reconnection
- Publisher confirms for reliable messaging
- Consumer acknowledgements
- Dead letter exchanges for failed message handling
- Graceful shutdown
- Comprehensive logging
- Thread-safe operations
"""

import json
import time
import threading
import logging
import functools
import signal
import os
from typing import Dict, Any, Callable, Optional, List, Union
from datetime import datetime

import pika
from pika.exceptions import (
    AMQPConnectionError, 
    AMQPChannelError, 
    ConnectionClosedByBroker,
    StreamLostError
)

# Configure logging
from utils.logging_config import setup_logging

logger = setup_logging(name="message_broker")

class MessageBroker:
    """RabbitMQ message broker implementation for the crypto trading system."""

    # Default exchange configuration
    EXCHANGES = {
        "crypto.topic": {"type": "topic", "durable": True},
        "crypto.direct": {"type": "direct", "durable": True},
        "crypto.fanout": {"type": "fanout", "durable": True},
        "crypto.dlx": {"type": "fanout", "durable": True}  # Dead letter exchange
    }

    # Default queue configuration
    QUEUES = {
        "price_updates": {
            "exchange": "crypto.topic",
            "routing_key": "market.price.#",
            "durable": True,
            "arguments": {
                "x-dead-letter-exchange": "crypto.dlx",
                "x-dead-letter-routing-key": "failed.price_updates"
            }
        },
        "sentiment_updates": {
            "exchange": "crypto.topic",
            "routing_key": "sentiment.#",
            "durable": True,
            "arguments": {
                "x-dead-letter-exchange": "crypto.dlx",
                "x-dead-letter-routing-key": "failed.sentiment_updates"
            }
        },
        "trading_signals": {
            "exchange": "crypto.topic",
            "routing_key": "signals.#",
            "durable": True,
            "arguments": {
                "x-dead-letter-exchange": "crypto.dlx",
                "x-dead-letter-routing-key": "failed.trading_signals"
            }
        },
        "system_status": {
            "exchange": "crypto.fanout",
            "routing_key": "",
            "durable": True,
            "arguments": {
                "x-dead-letter-exchange": "crypto.dlx",
                "x-dead-letter-routing-key": "failed.system_status"
            }
        },
        "failed_messages": {
            "exchange": "crypto.dlx",
            "routing_key": "failed.#",
            "durable": True,
            "arguments": {}
        }
    }

    def __init__(
        self, 
        host: str = 'localhost', 
        port: int = 5672,
        username: str = 'guest', 
        password: str = 'guest',
        virtual_host: str = '/', 
        connection_attempts: int = 3,
        retry_delay: int = 5,
        heartbeat: int = 60,
        blocked_connection_timeout: int = 300,
        prefetch_count: int = 10,
        publisher_confirms: bool = True
    ):
        """
        Initialize the MessageBroker with RabbitMQ connection parameters.

        Args:
            host: RabbitMQ server hostname or IP address
            port: RabbitMQ server port
            username: RabbitMQ username
            password: RabbitMQ password
            virtual_host: RabbitMQ virtual host
            connection_attempts: Number of connection attempts
            retry_delay: Delay between connection attempts in seconds
            heartbeat: Heartbeat interval in seconds
            blocked_connection_timeout: Timeout for blocked connections in seconds
            prefetch_count: Maximum number of messages to prefetch
            publisher_confirms: Whether to enable publisher confirms
        """
        # Connection parameters
        self.connection_params = pika.ConnectionParameters(
            host=host,
            port=port,
            virtual_host=virtual_host,
            credentials=pika.PlainCredentials(username, password),
            connection_attempts=connection_attempts,
            retry_delay=retry_delay,
            heartbeat=heartbeat,
            blocked_connection_timeout=blocked_connection_timeout
        )
        
        # Prefetch count for consumers
        self.prefetch_count = prefetch_count
        
        # Publisher confirms setting
        self.publisher_confirms = publisher_confirms
        
        # Connection and channel instances
        self._connection = None
        self._channel = None
        
        # Threads
        self._consumer_thread = None
        self._connection_monitor_thread = None
        
        # State flags
        self._running = False
        self._consuming = False
        self._reconnecting = False
        
        # Thread lock for thread safety
        self._lock = threading.RLock()
        
        # Registered callbacks for consumers
        self._consumer_callbacks = {}
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info("MessageBroker initialized")

    def connect(self) -> bool:
        """
        Establish a connection to RabbitMQ server.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        if self._connection is not None and self._connection.is_open:
            logger.debug("Already connected to RabbitMQ")
            return True
            
        try:
            logger.info("Connecting to RabbitMQ at %s:%s", 
                       self.connection_params.host, 
                       self.connection_params.port)
            
            self._connection = pika.BlockingConnection(self.connection_params)
            self._channel = self._connection.channel()
            
            if self.publisher_confirms:
                self._channel.confirm_delivery()
                
            # Set up exchanges
            self._setup_exchanges()
            
            # Set up queues
            self._setup_queues()
            
            # Set QoS for the channel
            self._channel.basic_qos(prefetch_count=self.prefetch_count)
            
            logger.info("Successfully connected to RabbitMQ")
            
            # Start connection monitor if not already running
            self._start_connection_monitor()
            
            return True
            
        except (AMQPConnectionError, ConnectionError) as e:
            logger.error("Failed to connect to RabbitMQ: %s", str(e))
            return False

    def _setup_exchanges(self) -> None:
        """Set up exchanges according to the defined configuration."""
        for exchange_name, config in self.EXCHANGES.items():
            try:
                self._channel.exchange_declare(
                    exchange=exchange_name,
                    exchange_type=config["type"],
                    durable=config["durable"]
                )
                logger.debug("Declared exchange: %s", exchange_name)
            except Exception as e:
                logger.error("Failed to declare exchange %s: %s", 
                           exchange_name, str(e))
                raise

    def _setup_queues(self) -> None:
        """Set up queues according to the defined configuration."""
        for queue_name, config in self.QUEUES.items():
            try:
                self._channel.queue_declare(
                    queue=queue_name,
                    durable=config["durable"],
                    arguments=config["arguments"]
                )
                
                self._channel.queue_bind(
                    queue=queue_name,
                    exchange=config["exchange"],
                    routing_key=config["routing_key"]
                )
                
                logger.debug("Declared and bound queue: %s", queue_name)
            except Exception as e:
                logger.error("Failed to declare or bind queue %s: %s", 
                           queue_name, str(e))
                raise

    def _start_connection_monitor(self) -> None:
        """Start the connection monitor thread."""
        if (self._connection_monitor_thread is None or 
            not self._connection_monitor_thread.is_alive()):
            self._running = True
            self._connection_monitor_thread = threading.Thread(
                target=self._monitor_connection,
                daemon=True
            )
            self._connection_monitor_thread.start()
            logger.debug("Connection monitor thread started")

    def _monitor_connection(self) -> None:
        """
        Monitor the connection to RabbitMQ and attempt to reconnect if needed.
        This method runs in a separate thread.
        """
        logger.info("Connection monitor started")
        check_interval = 5  # Check every 5 seconds
        
        while self._running:
            try:
                # Sleep first to avoid immediate checking
                time.sleep(check_interval)
                
                with self._lock:
                    if not self._running:
                        break
                        
                    if (self._connection is None or 
                        not self._connection.is_open):
                        
                        if not self._reconnecting:
                            logger.warning("Connection lost, attempting to reconnect")
                            self._reconnecting = True
                            
                            # Clean up old connection if it exists
                            if self._connection is not None:
                                try:
                                    self._connection.close()
                                except Exception:
                                    pass
                                    
                            self._connection = None
                            self._channel = None
                            
                            # Attempt to reconnect
                            reconnected = self.connect()
                            
                            if reconnected and self._consuming:
                                # Restore consumers
                                self._restore_consumers()
                                
                            self._reconnecting = False
            except Exception as e:
                logger.error("Error in connection monitor: %s", str(e))

    def _restore_consumers(self) -> None:
        """Restore consumers after a reconnection."""
        logger.info("Restoring consumers")
        
        if not self._consumer_callbacks:
            logger.info("No consumers to restore")
            return
            
        for queue_name, callback in self._consumer_callbacks.items():
            try:
                self._channel.basic_consume(
                    queue=queue_name,
                    on_message_callback=callback,
                    auto_ack=False
                )
                logger.info("Restored consumer for queue: %s", queue_name)
            except Exception as e:
                logger.error("Failed to restore consumer for queue %s: %s", 
                           queue_name, str(e))

    def publish(
        self, 
        message: Dict[str, Any], 
        exchange: str, 
        routing_key: str,
        mandatory: bool = True,
        properties: Optional[pika.BasicProperties] = None,
        retries: int = 3,
        retry_delay: int = 2
    ) -> bool:
        """
        Publish a message to the specified exchange with the given routing key.
        
        Args:
            message: The message to publish (will be converted to JSON)
            exchange: The exchange to publish to
            routing_key: The routing key for the message
            mandatory: Whether to require the message to be routable
            properties: Optional AMQP properties
            retries: Number of retry attempts if publishing fails
            retry_delay: Delay between retries in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        if properties is None:
            properties = pika.BasicProperties(
                delivery_mode=2,  # Persistent
                content_type='application/json',
                timestamp=int(time.time()),
                message_id=f"{routing_key}.{datetime.now().isoformat()}"
            )
            
        message_body = json.dumps(message).encode('utf-8')
        
        for attempt in range(retries + 1):
            try:
                with self._lock:
                    if self._connection is None or not self._connection.is_open:
                        if not self.connect():
                            logger.error("Failed to connect to RabbitMQ for publishing")
                            time.sleep(retry_delay)
                            continue
                            
                    result = self._channel.basic_publish(
                        exchange=exchange,
                        routing_key=routing_key,
                        body=message_body,
                        properties=properties,
                        mandatory=mandatory
                    )
                    
                    if self.publisher_confirms and not result:
                        logger.warning("Message was not confirmed by broker, retrying...")
                        time.sleep(retry_delay)
                        continue
                        
                    logger.debug("Successfully published message to %s with routing key %s", 
                               exchange, routing_key)
                    return True
                    
            except (AMQPConnectionError, AMQPChannelError, ConnectionClosedByBroker,
                   StreamLostError) as e:
                logger.warning("Failed to publish message (attempt %d/%d): %s", 
                             attempt + 1, retries + 1, str(e))
                
                # Try to reconnect
                self.connect()
                
                # Wait before retrying
                if attempt < retries:
                    time.sleep(retry_delay)
            except Exception as e:
                logger.error("Unexpected error publishing message: %s", str(e))
                if attempt < retries:
                    time.sleep(retry_delay)
                    
        logger.error("Failed to publish message after %d attempts", retries + 1)
        return False

    def publish_price_update(self, symbol: str, price_data: Dict[str, Any]) -> bool:
        """
        Publish a price update message.
        
        Args:
            symbol: The cryptocurrency symbol (e.g., 'BTC', 'ETH')
            price_data: Price data dictionary containing at minimum 'price' and 'timestamp'
            
        Returns:
            bool: True if successful, False otherwise
        """
        message = {
            "symbol": symbol,
            "data": price_data,
            "timestamp": price_data.get("timestamp", datetime.now().isoformat())
        }
        
        routing_key = f"market.price.{symbol.lower()}"
        return self.publish(message, "crypto.topic", routing_key)

    def publish_sentiment_update(
        self, 
        source: str, 
        sentiment_data: Dict[str, Any], 
        symbol: Optional[str] = None
    ) -> bool:
        """
        Publish a sentiment update message.
        
        Args:
            source: Source of the sentiment data (e.g., 'youtube', 'twitter')
            sentiment_data: Sentiment analysis data
            symbol: Optional cryptocurrency symbol if sentiment is related to a specific coin
            
        Returns:
            bool: True if successful, False otherwise
        """
        message = {
            "source": source,
            "data": sentiment_data,
            "timestamp": sentiment_data.get("timestamp", datetime.now().isoformat())
        }
        
        if symbol:
            message["symbol"] = symbol
            routing_key = f"sentiment.{source}.{symbol.lower()}"
        else:
            routing_key = f"sentiment.{source}"
            
        return self.publish(message, "crypto.topic", routing_key)

    def publish_trading_signal(
        self, 
        symbol: str, 
        signal_data: Dict[str, Any]
    ) -> bool:
        """
        Publish a trading signal message.
        
        Args:
            symbol: The cryptocurrency symbol (e.g., 'BTC', 'ETH')
            signal_data: Trading signal data
            
        Returns:
            bool: True if successful, False otherwise
        """
        message = {
            "symbol": symbol,
            "data": signal_data,
            "timestamp": signal_data.get("timestamp", datetime.now().isoformat())
        }
        
        routing_key = f"signals.{symbol.lower()}"
        return self.publish(message, "crypto.topic", routing_key)

    def publish_system_status(self, component: str, status_data: Dict[str, Any]) -> bool:
        """
        Publish a system status message.
        
        Args:
            component: The system component name
            status_data: Status data
            
        Returns:
            bool: True if successful, False otherwise
        """
        message = {
            "component": component,
            "data": status_data,
            "timestamp": status_data.get("timestamp", datetime.now().isoformat())
        }
        
        routing_key = f"status.{component}"
        return self.publish(message, "crypto.fanout", routing_key)

    def consume(
        self, 
        queue: str, 
        callback: Callable[[Dict[str, Any], Dict[str, Any]], None], 
        auto_ack: bool = False
    ) -> None:
        """
        Register a callback to consume messages from a queue.
        
        Args:
            queue: Queue name to consume from
            callback: Function to call when a message is received.
                     Should accept two arguments: (properties, body)
            auto_ack: Whether to automatically acknowledge messages
        """
        def internal_callback(ch, method, properties, body):
            try:
                # Parse message body
                try:
                    message = json.loads(body)
                except json.JSONDecodeError:
                    logger.warning("Received non-JSON message, processing as raw text")
                    message = {"raw": body.decode('utf-8', errors='replace')}
                
                # Extract properties as dict
                prop_dict = {
                    "content_type": properties.content_type,
                    "delivery_mode": properties.delivery_mode,
                    "correlation_id": properties.correlation_id,
                    "reply_to": properties.reply_to,
                    "message_id": properties.message_id,
                    "timestamp": properties.timestamp,
                    "routing_key": method.routing_key,
                    "exchange": method.exchange
                }
                
                # Call user callback
                callback(message, prop_dict)
                
                # Acknowledge message if not auto_ack
                if not auto_ack:
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    
            except Exception as e:
                logger.error("Error processing message: %s", str(e))
                
                # Negative acknowledge and requeue if not auto_ack
                if not auto_ack:
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        
        with self._lock:
            if self._connection is None or not self._connection.is_open:
                if not self.connect():
                    raise ConnectionError("Failed to connect to RabbitMQ")
            
            # Register the callback
            self._consumer_callbacks[queue] = internal_callback
            
            # Set up the consumer
            self._channel.basic_consume(
                queue=queue,
                on_message_callback=internal_callback,
                auto_ack=auto_ack
            )
            
            logger.info("Registered consumer for queue: %s", queue)
            
            # Start consuming if not already
            if not self._consuming:
                self._start_consuming()

    def consume_price_updates(
        self, 
        callback: Callable[[Dict[str, Any], Dict[str, Any]], None],
        symbols: Optional[List[str]] = None
    ) -> None:
        """
        Consume price update messages.
        
        Args:
            callback: Function to call when a price update is received
            symbols: Optional list of symbols to filter (if None, all symbols)
        """
        if symbols:
            # Create a temporary queue with a specific binding for the symbols
            result = self._channel.queue_declare(queue='', exclusive=True)
            temp_queue = result.method.queue
            
            for symbol in symbols:
                routing_key = f"market.price.{symbol.lower()}"
                self._channel.queue_bind(
                    exchange="crypto.topic",
                    queue=temp_queue,
                    routing_key=routing_key
                )
                
            logger.info("Created temporary queue for price updates with symbols: %s", 
                       ", ".join(symbols))
            
            # Consume from the temporary queue
            self.consume(temp_queue, callback)
        else:
            # Consume from the main price updates queue
            self.consume("price_updates", callback)

    def consume_sentiment_updates(
        self, 
        callback: Callable[[Dict[str, Any], Dict[str, Any]], None],
        sources: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None
    ) -> None:
        """
        Consume sentiment update messages.
        
        Args:
            callback: Function to call when a sentiment update is received
            sources: Optional list of sources to filter (if None, all sources)
            symbols: Optional list of symbols to filter (if None, all symbols)
        """
        if sources or symbols:
            # Create a temporary queue with specific bindings
            result = self._channel.queue_declare(queue='', exclusive=True)
            temp_queue = result.method.queue
            
            if sources and symbols:
                # Bind for specific sources and symbols
                for source in sources:
                    for symbol in symbols:
                        routing_key = f"sentiment.{source}.{symbol.lower()}"
                        self._channel.queue_bind(
                            exchange="crypto.topic",
                            queue=temp_queue,
                            routing_key=routing_key
                        )
            elif sources:
                # Bind for specific sources only
                for source in sources:
                    routing_key = f"sentiment.{source}.#"
                    self._channel.queue_bind(
                        exchange="crypto.topic",
                        queue=temp_queue,
                        routing_key=routing_key
                    )
            elif symbols:
                # Bind for specific symbols only
                for symbol in symbols:
                    routing_key = f"sentiment.#.{symbol.lower()}"
                    self._channel.queue_bind(
                        exchange="crypto.topic",
                        queue=temp_queue,
                        routing_key=routing_key
                    )
                    
            logger.info("Created temporary queue for sentiment updates with specific filters")
            
            # Consume from the temporary queue
            self.consume(temp_queue, callback)
        else:
            # Consume from the main sentiment updates queue
            self.consume("sentiment_updates", callback)

    def consume_trading_signals(
        self, 
        callback: Callable[[Dict[str, Any], Dict[str, Any]], None],
        symbols: Optional[List[str]] = None
    ) -> None:
        """
        Consume trading signal messages.
        
        Args:
            callback: Function to call when a trading signal is received
            symbols: Optional list of symbols to filter (if None, all symbols)
        """
        if symbols:
            # Create a temporary queue with a specific binding for the symbols
            result = self._channel.queue_declare(queue='', exclusive=True)
            temp_queue = result.method.queue
            
            for symbol in symbols:
                routing_key = f"signals.{symbol.lower()}"
                self._channel.queue_bind(
                    exchange="crypto.topic",
                    queue=temp_queue,
                    routing_key=routing_key
                )
                
            logger.info("Created temporary queue for trading signals with symbols: %s", 
                       ", ".join(symbols))
            
            # Consume from the temporary queue
            self.consume(temp_queue, callback)
        else:
            # Consume from the main trading signals queue
            self.consume("trading_signals", callback)

    def consume_system_status(
        self, 
        callback: Callable[[Dict[str, Any], Dict[str, Any]], None],
        components: Optional[List[str]] = None
    ) -> None:
        """
        Consume system status messages.
        
        Args:
            callback: Function to call when a system status message is received
            components: Optional list of components to filter (if None, all components)
        """
        if components:
            # Create a temporary queue with a specific binding for the components
            result = self._channel.queue_declare(queue='', exclusive=True)
            temp_queue = result.method.queue
            
            for component in components:
                routing_key = f"status.{component}"
                self._channel.queue_bind(
                    exchange="crypto.fanout",
                    queue=temp_queue,
                    routing_key=routing_key
                )
                
            logger.info("Created temporary queue for system status with components: %s", 
                       ", ".join(components))
            
            # Consume from the temporary queue
            self.consume(temp_queue, callback)
        else:
            # Consume from the main system status queue
            self.consume("system_status", callback)

    def consume_failed_messages(
        self, 
        callback: Callable[[Dict[str, Any], Dict[str, Any]], None]
    ) -> None:
        """
        Consume messages from the dead letter queue.
        
        Args:
            callback: Function to call when a failed message is received
        """
        self.consume("failed_messages", callback)

    def _start_consuming(self) -> None:
        """Start consuming messages in a separate thread."""
        self._consuming = True
        
        if self._consumer_thread is None or not self._consumer_thread.is_alive():
            self._consumer_thread = threading.Thread(
                target=self._consume_loop,
                daemon=True
            )
            self._consumer_thread.start()
            logger.info("Consumer thread started")

    def _consume_loop(self) -> None:
        """
        Main consume loop that processes messages.
        This method runs in a separate thread.
        """
        logger.info("Starting consumer loop")
        
        while self._consuming:
            if self._connection is None or not self._connection.is_open:
                logger.warning("Connection closed, waiting for reconnect...")
                time.sleep(1)
                continue
                
            try:
                with self._lock:
                    # Process messages for a short time
                    self._connection.process_data_events(time_limit=1.0)
            except (AMQPConnectionError, ConnectionClosedByBroker, StreamLostError):
                logger.warning("Connection error in consumer loop, waiting for reconnect...")
                time.sleep(1)
            except Exception as e:
                logger.error("Error in consumer loop: %s", str(e))
                time.sleep(1)
                
        logger.info("Consumer loop stopped")

    def stop_consuming(self) -> None:
        """Stop consuming messages."""
        logger.info("Stopping message consumption")
        
        with self._lock:
            self._consuming = False
            
            try:
                if self._channel is not None and self._channel.is_open:
                    self._channel.stop_consuming()
            except Exception as e:
                logger.error("Error stopping consumption: %s", str(e))
                
        # Wait for consumer thread to finish
        if self._consumer_thread is not None and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=5.0)
            
        self._consumer_thread = None
        self._consumer_callbacks.clear()
        
        logger.info("Message consumption stopped")

    def close(self) -> None:
        """Close the connection to RabbitMQ."""
        logger.info("Closing connection to RabbitMQ")
        
        # Stop consuming first
        self.stop_consuming()
        
        with self._lock:
            self._running = False
            
            if self._connection is not None and self._connection.is_open:
                try:
                    self._connection.close()
                except Exception as e:
                    logger.error("Error closing connection: %s", str(e))
                    
            self._connection = None
            self._channel = None
            
        # Wait for connection monitor thread to finish
        if (self._connection_monitor_thread is not None and 
            self._connection_monitor_thread.is_alive()):
            self._connection_monitor_thread.join(timeout=5.0)
            
        self._connection_monitor_thread = None
        
        logger.info("Connection to RabbitMQ closed")

    def _signal_handler(self, signum, frame) -> None:
        """Handle termination signals."""
        logger.info("Received signal %s, shutting down...", signum)
        self.close()

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Usage example
if __name__ == "__main__":
    # Create a broker instance
    broker = MessageBroker(
        host=os.environ.get("RABBITMQ_HOST", "localhost"),
        port=int(os.environ.get("RABBITMQ_PORT", "5672")),
        username=os.environ.get("RABBITMQ_USER", "guest"),
        password=os.environ.get("RABBITMQ_PASS", "guest")
    )
    
    # Connect to RabbitMQ
    if not broker.connect():
        logger.error("Failed to connect to RabbitMQ server")
        exit(1)
        
    try:
        # Example price update callback
        def handle_price_update(message, properties):
            logger.info("Received price update for %s: $%.2f", 
                       message.get("symbol"), 
                       message.get("data", {}).get("price", 0))
            
        # Start consuming price updates
        broker.consume_price_updates(handle_price_update, symbols=["BTC", "ETH"])
        
        # Example publishing
        sample_price = {
            "price": 50000.0,
            "timestamp": datetime.now().isoformat(),
            "volume": 1234.56,
            "source": "binance"
        }
        
        if broker.publish_price_update("BTC", sample_price):
            logger.info("Successfully published BTC price update")
        else:
            logger.error("Failed to publish BTC price update")
            
        # Keep the main thread running
        logger.info("Message broker running. Press CTRL+C to exit.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Clean up
        broker.close()