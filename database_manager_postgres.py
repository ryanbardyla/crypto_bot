#!/usr/bin/env python3
"""
database_manager_postgres.py - PostgreSQL Database Manager

This module provides a robust implementation for interacting with PostgreSQL
for the crypto trading system. It replaces the SQLite-based database_manager.py
with a more scalable and concurrent PostgreSQL solution.

Features:
- Connection pooling for efficient database access
- Transaction management
- Error handling and retry logic
- Migration support from SQLite
- Async operation support (optional)
- Comprehensive logging
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import threading
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool, sql, extras
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT, ISOLATION_LEVEL_READ_COMMITTED
import pandas as pd
from sqlalchemy import create_engine, text, Table, Column, Integer, String, Float, DateTime, Text, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
import sqlalchemy.types as types
from sqlalchemy import Index, func, cast, Date

# Configure logging
from utils.logging_config import setup_logging

logger = setup_logging(name="database_manager")

# Define SQLAlchemy Base
Base = declarative_base()

# Define database models
class SentimentRecord(Base):
    """SQLAlchemy model for the sentiment_records table."""
    __tablename__ = 'sentiment_records'
    
    id = Column(Integer, primary_key=True)
    record_id = Column(String(100), unique=True, nullable=False)  # video_id or tweet_id
    record_type = Column(String(10), nullable=False)  # 'youtube' or 'twitter'
    channel_id = Column(String(100), index=True)
    username = Column(String(100), index=True)
    title = Column(String(500))
    text = Column(Text)
    publish_date = Column(DateTime, index=True)
    processed_date = Column(DateTime, default=datetime.now, index=True)
    followers_count = Column(Integer)
    retweet_count = Column(Integer)
    like_count = Column(Integer)
    view_count = Column(Integer)
    vader_neg = Column(Float)
    vader_neu = Column(Float)
    vader_pos = Column(Float)
    vader_compound = Column(Float)
    bullish_keywords = Column(Integer)
    bearish_keywords = Column(Integer)
    keyword_sentiment = Column(Float)
    combined_score = Column(Float, index=True)
    text_length = Column(Integer)
    symbol = Column(String(10), index=True)
    source = Column(String(200))
    mentioned_cryptos = Column(Text)  # Stored as JSON string
    
    # Define table indexes
    __table_args__ = (
        Index('idx_sentiment_date_score', processed_date, combined_score),
        Index('idx_sentiment_source_date', source, processed_date),
        Index('idx_sentiment_record_type_date', record_type, processed_date),
        Index('idx_sentiment_publish_date', publish_date),
    )


class PriceRecord(Base):
    """SQLAlchemy model for the price_records table."""
    __tablename__ = 'price_records'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    price = Column(Float, nullable=False)
    volume = Column(Float)
    high = Column(Float)
    low = Column(Float)
    open = Column(Float)
    close = Column(Float)
    source = Column(String(50))
    
    # Define table indexes
    __table_args__ = (
        Index('idx_price_symbol_timestamp', symbol, timestamp),
    )


class TradeRecord(Base):
    """SQLAlchemy model for the trade_records table."""
    __tablename__ = 'trade_records'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    trade_type = Column(String(10), nullable=False)  # 'BUY' or 'SELL'
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    value = Column(Float, nullable=False)
    fee = Column(Float)
    order_id = Column(String(100))
    strategy = Column(String(50))
    signal_id = Column(String(100))
    profit_loss = Column(Float)
    profit_loss_pct = Column(Float)
    balance_after = Column(Float)
    reason = Column(String(50))
    
    # Define table indexes
    __table_args__ = (
        Index('idx_trade_symbol_timestamp', symbol, timestamp),
        Index('idx_trade_order_id', order_id),
    )


class DatabaseManager:
    """PostgreSQL database manager for the crypto trading system."""
    
    def __init__(
        self, 
        db_uri: str = None,
        host: str = None,
        port: str = None,
        database: str = None,
        user: str = None,
        password: str = None,
        min_connections: int = 1,
        max_connections: int = 10,
        enable_pooling: bool = True,
        retry_attempts: int = 3,
        retry_delay: int = 2,
        schema: str = 'public'
    ):
        """
        Initialize the DatabaseManager with PostgreSQL connection parameters.
        
        Args:
            db_uri: SQLAlchemy connection URI (if provided, other connection params are ignored)
            host: PostgreSQL server hostname or IP
            port: PostgreSQL server port
            database: Database name
            user: Database username
            password: Database password
            min_connections: Minimum number of connections in the pool
            max_connections: Maximum number of connections in the pool
            enable_pooling: Whether to use connection pooling
            retry_attempts: Number of retry attempts for database operations
            retry_delay: Delay between retries in seconds
            schema: Database schema to use
        """
        # Set connection parameters
        self.db_uri = db_uri
        
        # Use individual connection parameters if URI is not provided
        if not self.db_uri:
            # Get values from environment variables if not provided
            self.host = host or os.environ.get('POSTGRES_HOST', 'localhost')
            self.port = port or os.environ.get('POSTGRES_PORT', '5432')
            self.database = database or os.environ.get('POSTGRES_DB', 'crypto_trading')
            self.user = user or os.environ.get('POSTGRES_USER', 'postgres')
            self.password = password or os.environ.get('POSTGRES_PASSWORD', 'postgres')
            
            # Construct URI
            self.db_uri = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        
        # Connection pool settings
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.enable_pooling = enable_pooling
        
        # Retry settings
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Schema
        self.schema = schema
        
        # Connection pool
        self._pool = None
        
        # SQLAlchemy engine
        self._engine = None
        
        # Session factory
        self._session_factory = None
        
        # Thread-local storage for sessions
        self._session_local = threading.local()
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Initialize the database
        self.setup_database()
        
        logger.info("DatabaseManager initialized with URI: %s", self._safe_uri())

    def _safe_uri(self) -> str:
        """Return a safe version of the DB URI with password masked."""
        if not self.db_uri:
            return "None"
        
        parts = self.db_uri.split("://")
        if len(parts) < 2:
            return self.db_uri
            
        auth_parts = parts[1].split("@")
        if len(auth_parts) < 2:
            return self.db_uri
            
        user_pass = auth_parts[0].split(":")
        if len(user_pass) < 2:
            return self.db_uri
            
        masked_uri = f"{parts[0]}://{user_pass[0]}:***@{auth_parts[1]}"
        return masked_uri

    def setup_database(self) -> None:
        """Set up the database connection and tables."""
        try:
            with self._lock:
                # Create SQLAlchemy engine with appropriate settings
                connect_args = {
                    "connect_timeout": 10,
                    "application_name": "CryptoTradingBot"
                }
                
                self._engine = create_engine(
                    self.db_uri,
                    pool_pre_ping=True,
                    pool_size=self.min_connections,
                    max_overflow=self.max_connections - self.min_connections,
                    pool_recycle=3600,  # Recycle connections after 1 hour
                    connect_args=connect_args
                )
                
                # Create all tables if they don't exist
                Base.metadata.create_all(self._engine)
                
                # Create session factory
                self._session_factory = scoped_session(sessionmaker(bind=self._engine))
                
                # Initialize connection pool if enabled
                if self.enable_pooling:
                    self._setup_connection_pool()
                
                logger.info("Database setup successful")
                
        except Exception as e:
            logger.error("Failed to set up database: %s", str(e))
            raise

    def _setup_connection_pool(self) -> None:
        """Set up the psycopg2 connection pool."""
        try:
            self._pool = pool.ThreadedConnectionPool(
                minconn=self.min_connections,
                maxconn=self.max_connections,
                host=self.host if hasattr(self, 'host') else self.db_uri.split('@')[1].split('/')[0].split(':')[0],
                port=self.port if hasattr(self, 'port') else self.db_uri.split('@')[1].split('/')[0].split(':')[1],
                database=self.database if hasattr(self, 'database') else self.db_uri.split('/')[-1],
                user=self.user if hasattr(self, 'user') else self.db_uri.split('://')[1].split(':')[0],
                password=self.password if hasattr(self, 'password') else self.db_uri.split(':')[2].split('@')[0]
            )
            
            logger.info("Connection pool initialized with %d-%d connections", 
                       self.min_connections, self.max_connections)
                       
        except Exception as e:
            logger.error("Failed to initialize connection pool: %s", str(e))
            logger.warning("Connection pooling disabled, falling back to direct connections")
            self.enable_pooling = False

    def _get_connection(self):
        """Get a connection from the pool."""
        if not self.enable_pooling or self._pool is None:
            return psycopg2.connect(
                host=self.host if hasattr(self, 'host') else self.db_uri.split('@')[1].split('/')[0].split(':')[0],
                port=self.port if hasattr(self, 'port') else self.db_uri.split('@')[1].split('/')[0].split(':')[1],
                database=self.database if hasattr(self, 'database') else self.db_uri.split('/')[-1],
                user=self.user if hasattr(self, 'user') else self.db_uri.split('://')[1].split(':')[0],
                password=self.password if hasattr(self, 'password') else self.db_uri.split(':')[2].split('@')[0]
            )
        
        try:
            conn = self._pool.getconn()
            conn.set_isolation_level(ISOLATION_LEVEL_READ_COMMITTED)
            return conn
        except Exception as e:
            logger.error("Failed to get connection from pool: %s", str(e))
            logger.warning("Falling back to direct connection")
            return psycopg2.connect(
                host=self.host if hasattr(self, 'host') else self.db_uri.split('@')[1].split('/')[0].split(':')[0],
                port=self.port if hasattr(self, 'port') else self.db_uri.split('@')[1].split('/')[0].split(':')[1],
                database=self.database if hasattr(self, 'database') else self.db_uri.split('/')[-1],
                user=self.user if hasattr(self, 'user') else self.db_uri.split('://')[1].split(':')[0],
                password=self.password if hasattr(self, 'password') else self.db_uri.split(':')[2].split('@')[0]
            )

    def _release_connection(self, conn) -> None:
        """Release a connection back to the pool."""
        if self.enable_pooling and self._pool is not None:
            try:
                self._pool.putconn(conn)
            except Exception as e:
                logger.error("Failed to release connection to pool: %s", str(e))
                try:
                    conn.close()
                except:
                    pass
        else:
            try:
                conn.close()
            except Exception as e:
                logger.error("Failed to close connection: %s", str(e))

    @contextmanager
    def get_connection(self):
        """Context manager for getting a database connection."""
        conn = None
        try:
            conn = self._get_connection()
            yield conn
        finally:
            if conn is not None:
                self._release_connection(conn)

    @contextmanager
    def get_cursor(self, conn=None, cursor_factory=None):
        """
        Context manager for getting a database cursor.
        
        Args:
            conn: Optional existing connection to use
            cursor_factory: Optional cursor factory (e.g., psycopg2.extras.DictCursor)
        """
        connection_provided = conn is not None
        
        try:
            if not connection_provided:
                conn = self._get_connection()
                
            if cursor_factory:
                cursor = conn.cursor(cursor_factory=cursor_factory)
            else:
                cursor = conn.cursor()
                
            yield cursor
            
            if not connection_provided:
                conn.commit()
                
        except Exception as e:
            if not connection_provided and conn is not None:
                conn.rollback()
            raise e
            
        finally:
            if cursor is not None:
                cursor.close()
                
            if not connection_provided and conn is not None:
                self._release_connection(conn)

    def get_session(self):
        """Get a SQLAlchemy session."""
        # Create session if it doesn't exist for the current thread
        if not hasattr(self._session_local, 'session'):
            self._session_local.session = self._session_factory()
        
        return self._session_local.session

    def close_session(self) -> None:
        """Close the current thread's SQLAlchemy session."""
        if hasattr(self._session_local, 'session'):
            try:
                self._session_local.session.close()
            except Exception as e:
                logger.error("Error closing session: %s", str(e))
            finally:
                del self._session_local.session

    @contextmanager
    def session_scope(self):
        """Context manager for SQLAlchemy session."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Error in session scope: %s", str(e))
            raise
        finally:
            session.close()

    def execute_with_retry(self, func, *args, **kwargs):
        """
        Execute a database function with retry logic.
        
        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function call
        """
        for attempt in range(self.retry_attempts):
            try:
                return func(*args, **kwargs)
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                if attempt < self.retry_attempts - 1:
                    logger.warning("Database operation failed: %s. Retrying %d/%d...", 
                                 str(e), attempt + 1, self.retry_attempts)
                    time.sleep(self.retry_delay)
                else:
                    logger.error("Database operation failed after %d attempts: %s", 
                               self.retry_attempts, str(e))
                    raise
            except Exception as e:
                logger.error("Unexpected error in database operation: %s", str(e))
                raise

    def save_sentiment_data(self, data, record_type, record_id, override=False):
        """
        Save sentiment data to the database.
        
        Args:
            data: Sentiment data dictionary
            record_type: Type of record (e.g., 'youtube', 'twitter')
            record_id: Unique identifier for the record
            override: Whether to override existing records
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.session_scope() as session:
                # Check if record already exists
                existing = session.query(SentimentRecord).filter_by(
                    record_id=record_id,
                    record_type=record_type
                ).first()
                
                if existing and not override:
                    logger.info("%s record %s already exists", record_type.capitalize(), record_id)
                    return False
                
                if existing and override:
                    logger.info("Updating existing %s record %s", record_type.capitalize(), record_id)
                    
                    # Update existing record
                    for key, value in data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                            
                    # Update specific nested fields
                    if 'vader_sentiment' in data:
                        vader = data['vader_sentiment']
                        existing.vader_neg = vader.get('neg', 0)
                        existing.vader_neu = vader.get('neu', 0)
                        existing.vader_pos = vader.get('pos', 0)
                        existing.vader_compound = vader.get('compound', 0)
                    
                    # Set processed date to now
                    existing.processed_date = datetime.now()
                    
                    # Handle mentioned_cryptos if present
                    if 'mentioned_cryptos' in data:
                        existing.mentioned_cryptos = json.dumps(data['mentioned_cryptos'])
                        
                else:
                    # Create new record
                    logger.info("Creating new %s record %s", record_type.capitalize(), record_id)
                    
                    record = SentimentRecord(
                        record_id=record_id,
                        record_type=record_type,
                        channel_id=data.get('channel_id'),
                        username=data.get('username'),
                        title=data.get('title'),
                        text=data.get('text', '')[:10000],  # Limit text size
                        publish_date=data.get('publish_date'),
                        processed_date=datetime.now(),
                        followers_count=data.get('followers_count', 0),
                        retweet_count=data.get('retweet_count', 0),
                        like_count=data.get('like_count', 0),
                        view_count=data.get('view_count', 0),
                        vader_neg=data.get('vader_sentiment', {}).get('neg', 0),
                        vader_neu=data.get('vader_sentiment', {}).get('neu', 0),
                        vader_pos=data.get('vader_sentiment', {}).get('pos', 0),
                        vader_compound=data.get('vader_sentiment', {}).get('compound', 0),
                        bullish_keywords=data.get('bullish_keywords', 0),
                        bearish_keywords=data.get('bearish_keywords', 0),
                        keyword_sentiment=data.get('keyword_sentiment', 0),
                        combined_score=data.get('combined_score', 0),
                        text_length=data.get('text_length', 0),
                        symbol=data.get('symbol'),
                        source=data.get('source', f'{record_type}-{record_id}'),
                        mentioned_cryptos=json.dumps(data.get('mentioned_cryptos', []))
                    )
                    
                    session.add(record)
                
                session.commit()
                logger.info("Successfully saved %s record %s", record_type, record_id)
                return True
                
        except Exception as e:
            logger.error("Error saving %s record %s: %s", record_type, record_id, str(e))
            return False

    def get_sentiment_data(
        self, 
        record_type=None, 
        record_id=None, 
        symbol=None, 
        days=None, 
        limit=None, 
        offset=0,
        sort_by='processed_date',
        sort_desc=True
    ):
        """
        Get sentiment data from the database.
        
        Args:
            record_type: Optional filter by record type (e.g., 'youtube', 'twitter')
            record_id: Optional filter by record ID
            symbol: Optional filter by cryptocurrency symbol
            days: Optional filter by number of days back
            limit: Optional limit on number of records returned
            offset: Optional offset for pagination
            sort_by: Field to sort by
            sort_desc: Whether to sort in descending order
            
        Returns:
            list: List of sentiment records as dictionaries
        """
        result = []
        
        try:
            with self.session_scope() as session:
                # Build query
                query = session.query(SentimentRecord)
                
                # Apply filters
                if record_type:
                    query = query.filter(SentimentRecord.record_type == record_type)
                    
                if record_id:
                    query = query.filter(SentimentRecord.record_id == record_id)
                    
                if symbol:
                    query = query.filter(SentimentRecord.symbol == symbol)
                    
                if days:
                    cutoff_date = datetime.now() - timedelta(days=days)
                    query = query.filter(SentimentRecord.processed_date >= cutoff_date)
                
                # Apply sorting
                if sort_desc:
                    query = query.order_by(getattr(SentimentRecord, sort_by).desc())
                else:
                    query = query.order_by(getattr(SentimentRecord, sort_by))
                
                # Apply pagination
                if limit:
                    query = query.limit(limit)
                    
                if offset:
                    query = query.offset(offset)
                
                # Execute query
                records = query.all()
                
                # Convert to dictionaries
                for record in records:
                    record_dict = {c.name: getattr(record, c.name) for c in record.__table__.columns}
                    
                    # Convert datetime objects to ISO format strings
                    for key, value in record_dict.items():
                        if isinstance(value, datetime):
                            record_dict[key] = value.isoformat()
                    
                    # Parse mentioned_cryptos JSON
                    if record_dict.get('mentioned_cryptos'):
                        try:
                            record_dict['mentioned_cryptos'] = json.loads(record_dict['mentioned_cryptos'])
                        except json.JSONDecodeError:
                            record_dict['mentioned_cryptos'] = []
                    
                    # Reconstruct vader_sentiment dictionary
                    record_dict['vader_sentiment'] = {
                        'neg': record_dict.pop('vader_neg', 0),
                        'neu': record_dict.pop('vader_neu', 0),
                        'pos': record_dict.pop('vader_pos', 0),
                        'compound': record_dict.pop('vader_compound', 0)
                    }
                    
                    result.append(record_dict)
                
                logger.debug("Retrieved %d sentiment records", len(result))
                return result
                
        except Exception as e:
            logger.error("Error retrieving sentiment data: %s", str(e))
            return []

    def save_trade_record(self, trade_data):
        """
        Save a trade record to the database.
        
        Args:
            trade_data: Trade data dictionary
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.session_scope() as session:
                # Create new record
                record = TradeRecord(
                    symbol=trade_data.get('symbol'),
                    timestamp=datetime.fromisoformat(trade_data['timestamp']) if isinstance(trade_data['timestamp'], str) else trade_data['timestamp'],
                    trade_type=trade_data.get('type'),
                    quantity=trade_data.get('quantity'),
                    price=trade_data.get('price'),
                    value=trade_data.get('value', trade_data.get('quantity', 0) * trade_data.get('price', 0)),
                    fee=trade_data.get('fee', 0),
                    order_id=trade_data.get('order_id'),
                    strategy=trade_data.get('strategy'),
                    signal_id=trade_data.get('signal_id'),
                    profit_loss=trade_data.get('profit_loss'),
                    profit_loss_pct=trade_data.get('profit_loss_pct'),
                    balance_after=trade_data.get('balance_after'),
                    reason=trade_data.get('reason')
                )
                
                session.add(record)
                session.commit()
                
                logger.info("Successfully saved trade record for %s", trade_data.get('symbol'))
                return True
                
        except Exception as e:
            logger.error("Error saving trade record: %s", str(e))
            return False

    def get_trade_records(
        self, 
        symbol=None, 
        trade_type=None, 
        start_date=None, 
        end_date=None, 
        limit=None
    ):
        """
        Get trade records from the database.
        
        Args:
            symbol: Optional filter by cryptocurrency symbol
            trade_type: Optional filter by trade type ('BUY' or 'SELL')
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Optional limit on number of records returned
            
        Returns:
            list: List of trade records as dictionaries
        """
        result = []
        
        try:
            with self.session_scope() as session:
                # Build query
                query = session.query(TradeRecord)
                
                # Apply filters
                if symbol:
                    query = query.filter(TradeRecord.symbol == symbol)
                    
                if trade_type:
                    query = query.filter(TradeRecord.trade_type == trade_type)
                    
                if start_date:
                    if isinstance(start_date, str):
                        start_date = datetime.fromisoformat(start_date)
                    query = query.filter(TradeRecord.timestamp >= start_date)
                    
                if end_date:
                    if isinstance(end_date, str):
                        end_date = datetime.fromisoformat(end_date)
                    query = query.filter(TradeRecord.timestamp <= end_date)
                
                # Apply ordering
                query = query.order_by(TradeRecord.timestamp.desc())
                
                # Apply limit
                if limit:
                    query = query.limit(limit)
                
                # Execute query
                records = query.all()
                
                # Convert to dictionaries
                for record in records:
                    record_dict = {c.name: getattr(record, c.name) for c in record.__table__.columns}
                    
                    # Convert datetime objects to ISO format strings
                    for key, value in record_dict.items():
                        if isinstance(value, datetime):
                            record_dict[key] = value.isoformat()
                    
                    result.append(record_dict)
                
                logger.debug("Retrieved %d trade records", len(result))
                return result
                
        except Exception as e:
            logger.error("Error retrieving trade records: %s", str(e))
            return []

    def get_performance_metrics(self, symbol=None, start_date=None, end_date=None):
        """
        Calculate performance metrics based on trade records.
        
        Args:
            symbol: Optional filter by cryptocurrency symbol
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            dict: Dictionary of performance metrics
        """
        try:
            # Get all relevant trade records
            trades = self.get_trade_records(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if not trades:
                return {
                    "total_trades": 0,
                    "win_rate": 0,
                    "profit_factor": 0,
                    "total_profit": 0,
                    "total_loss": 0,
                    "net_pnl": 0,
                    "return_pct": 0,
                    "max_drawdown": 0
                }
            
            # Calculate metrics
            total_trades = len(trades)
            
            # Filter by trade type
            buy_trades = [t for t in trades if t['trade_type'] == 'BUY']
            sell_trades = [t for t in trades if t['trade_type'] == 'SELL']
            
            # Calculate profit/loss metrics
            winning_trades = [t for t in sell_trades if t.get('profit_loss', 0) > 0]
            losing_trades = [t for t in sell_trades if t.get('profit_loss', 0) <= 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            win_rate = (win_count / len(sell_trades)) * 100 if sell_trades else 0
            
            total_profit = sum(t.get('profit_loss', 0) for t in winning_trades)
            total_loss = abs(sum(t.get('profit_loss', 0) for t in losing_trades))
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            net_pnl = total_profit - total_loss
            
            # Calculate return percentage
            initial_balance = min(t.get('balance_after', 0) - t.get('profit_loss', 0) for t in sell_trades) if sell_trades else 0
            final_balance = max(t.get('balance_after', 0) for t in trades) if trades else 0
            
            return_pct = ((final_balance - initial_balance) / initial_balance) * 100 if initial_balance > 0 else 0
            
            # Calculate drawdown
            peak_balance = 0
            max_drawdown = 0
            
            for trade in sorted(trades, key=lambda x: x['timestamp']):
                balance = trade.get('balance_after', 0)
                
                if balance > peak_balance:
                    peak_balance = balance
                
                drawdown = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            return {
                "total_trades": total_trades,
                "buy_trades": len(buy_trades),
                "sell_trades": len(sell_trades),
                "winning_trades": win_count,
                "losing_trades": loss_count,
                "win_rate": win_rate,
                "total_profit": total_profit,
                "total_loss": total_loss,
                "profit_factor": profit_factor,
                "net_pnl": net_pnl,
                "return_pct": return_pct,
                "max_drawdown": max_drawdown,
                "initial_balance": initial_balance,
                "final_balance": final_balance
            }
                
        except Exception as e:
            logger.error("Error calculating performance metrics: %s", str(e))
            return {
                "error": str(e),
                "total_trades": 0
            }

    def migrate_from_sqlite(self, sqlite_db_path, batch_size=1000):
        """
        Migrate data from SQLite database to PostgreSQL.
        
        Args:
            sqlite_db_path: Path to SQLite database file
            batch_size: Batch size for data transfer
            
        Returns:
            dict: Dictionary of migration statistics
        """
        try:
            # Create SQLite engine
            sqlite_engine = create_engine(f"sqlite:///{sqlite_db_path}")
            
            # Create SQLite connection
            sqlite_conn = sqlite_engine.connect()
            
            # Get metadata from SQLite
            sqlite_meta = MetaData()
            sqlite_meta.reflect(bind=sqlite_engine)
            
            # Statistics
            stats = {
                "tables": {},
                "total_records": 0,
                "success": True,
                "start_time": datetime.now().isoformat(),
                "end_time": None
            }
            
            # Process each table
            for table_name in sqlite_meta.tables:
                logger.info("Migrating table: %s", table_name)
                
                # Create table statistics
                stats["tables"][table_name] = {
                    "records": 0,
                    "batches": 0,
                    "errors": 0
                }
                
                # Get SQLite table
                sqlite_table = sqlite_meta.tables[table_name]
                
                # Read data in batches
                offset = 0
                
                while True:
                    # Read batch from SQLite
                    query = f"SELECT * FROM {table_name} LIMIT {batch_size} OFFSET {offset}"
                    batch_df = pd.read_sql(query, sqlite_conn)
                    
                    # Break if no more data
                    if len(batch_df) == 0:
                        break
                    
                    # Update statistics
                    stats["tables"][table_name]["records"] += len(batch_df)
                    stats["tables"][table_name]["batches"] += 1
                    stats["total_records"] += len(batch_df)
                    
                    try:
                        # Write batch to PostgreSQL
                        batch_df.to_sql(
                            table_name,
                            self._engine,
                            if_exists='append',
                            index=False,
                            method='multi',
                            chunksize=100
                        )
                        
                        logger.info("Migrated %d records from table %s (batch %d)",
                                  len(batch_df), table_name, stats["tables"][table_name]["batches"])
                    except Exception as e:
                        stats["tables"][table_name]["errors"] += 1
                        logger.error("Error migrating batch from table %s: %s", table_name, str(e))
                    
                    # Update offset
                    offset += batch_size
            
            # Close SQLite connection
            sqlite_conn.close()
            
            # Update statistics
            stats["end_time"] = datetime.now().isoformat()
            stats["duration_seconds"] = (datetime.fromisoformat(stats["end_time"]) - datetime.fromisoformat(stats["start_time"])).total_seconds()
            
            logger.info("Migration completed: %d total records migrated", stats["total_records"])
            
            return stats
                
        except Exception as e:
            logger.error("Error migrating from SQLite: %s", str(e))
            return {
                "success": False,
                "error": str(e),
                "total_records": 0
            }

    def vacuum_database(self):
        """
        Perform database maintenance (VACUUM).
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # VACUUM can only be run outside a transaction
            with self.get_connection() as conn:
                conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                with self.get_cursor(conn) as cursor:
                    cursor.execute("VACUUM ANALYZE")
            
            logger.info("Database vacuum performed successfully")
            return True
            
        except Exception as e:
            logger.error("Error performing database vacuum: %s", str(e))
            return False

    def cleanup_old_records(self, table, days=90):
        """
        Delete old records from a table.
        
        Args:
            table: Table name
            days: Number of days to keep
            
        Returns:
            int: Number of records deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self.session_scope() as session:
                if table == 'sentiment_records':
                    count = session.query(SentimentRecord)\
                        .filter(SentimentRecord.processed_date < cutoff_date)\
                        .delete()
                elif table == 'price_records':
                    count = session.query(PriceRecord)\
                        .filter(PriceRecord.timestamp < cutoff_date)\
                        .delete()
                elif table == 'trade_records':
                    count = session.query(TradeRecord)\
                        .filter(TradeRecord.timestamp < cutoff_date)\
                        .delete()
                else:
                    logger.warning("Unknown table: %s", table)
                    return 0
                
                session.commit()
                
                logger.info("Cleaned up %d records from %s older than %d days", 
                          count, table, days)
                return count
                
        except Exception as e:
            logger.error("Error cleaning up old records from %s: %s", table, str(e))
            return 0

    def close(self):
        """Close all database connections."""
        try:
            # Close SQLAlchemy connections
            if self._engine:
                self._engine.dispose()
            
            # Close psycopg2 connection pool
            if self._pool:
                self._pool.closeall()
                
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error("Error closing database connections: %s", str(e))

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
if __name__ == "__main__":
    # Create a database manager instance
    db_manager = DatabaseManager(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        database=os.environ.get("POSTGRES_DB", "crypto_trading"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres")
    )
    
    try:
        # Test database connection
        with db_manager.get_connection() as conn:
            with db_manager.get_cursor(conn) as cursor:
                cursor.execute("SELECT version()")
                version = cursor.fetchone()[0]
                print(f"Connected to PostgreSQL: {version}")
        
        # Example: Save sentiment data
        sentiment_data = {
            "title": "Test Sentiment",
            "text": "This is a test sentiment record.",
            "vader_sentiment": {
                "neg": 0.0,
                "neu": 0.5,
                "pos": 0.5,
                "compound": 0.5
            },
            "bullish_keywords": 3,
            "bearish_keywords": 1,
            "keyword_sentiment": 2.0,
            "combined_score": 3.5,
            "text_length": 100,
            "mentioned_cryptos": ["BTC", "ETH"]
        }
        
        db_manager.save_sentiment_data(sentiment_data, "test", "test_record_1")
        
        # Example: Get sentiment data
        records = db_manager.get_sentiment_data(record_type="test", limit=5)
        print(f"Retrieved {len(records)} sentiment records")
        
    finally:
        # Close database connections
        db_manager.close()

    def get_aggregated_sentiment(self, days=7, symbol=None):
        """
        Get aggregated sentiment data by day.
        
        Args:
            days: Number of days to look back
            symbol: Optional filter by cryptocurrency symbol
            
        Returns:
            list: List of aggregated sentiment data by day
        """
        try:
            with self.session_scope() as session:
                # Calculate cutoff date
                cutoff_date = datetime.now() - timedelta(days=days)
                
                # Build query
                query = session.query(
                    cast(SentimentRecord.processed_date, Date).label('date'),
                    func.count(SentimentRecord.id).label('record_count'),
                    func.avg(SentimentRecord.combined_score).label('avg_sentiment'),
                    func.sum(SentimentRecord.bullish_keywords).label('total_bullish'),
                    func.sum(SentimentRecord.bearish_keywords).label('total_bearish'),
                    func.count(
                        func.case([(SentimentRecord.record_type == 'twitter', 1)])
                    ).label('has_twitter'),
                    func.count(
                        func.case([(SentimentRecord.record_type == 'youtube', 1)])
                    ).label('has_youtube')
                ).filter(SentimentRecord.processed_date >= cutoff_date)
                
                # Apply symbol filter if provided
                if symbol:
                    query = query.filter(SentimentRecord.symbol == symbol)
                    
                # Group by date and order by date
                query = query.group_by(cast(SentimentRecord.processed_date, Date))\
                             .order_by(cast(SentimentRecord.processed_date, Date))
                
                # Execute query
                result = [dict(row._mapping) for row in query.all()]
                
                # Convert datetime objects to ISO format strings
                for row in result:
                    if isinstance(row['date'], datetime):
                        row['date'] = row['date'].isoformat()
                
                logger.debug("Retrieved %d aggregated sentiment records", len(result))
                return result
                
        except Exception as e:
            logger.error("Error getting aggregated sentiment: %s", str(e))
            return []

    def save_price_data(self, symbol, price_data, override=False):
        """
        Save price data to the database.
        
        Args:
            symbol: Cryptocurrency symbol
            price_data: Price data dictionary
            override: Whether to override existing records
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            timestamp = datetime.fromisoformat(price_data['timestamp']) if isinstance(price_data['timestamp'], str) else price_data['timestamp']
            
            with self.session_scope() as session:
                # Check if record already exists
                existing = session.query(PriceRecord).filter_by(
                    symbol=symbol,
                    timestamp=timestamp
                ).first()
                
                if existing and not override:
                    logger.debug("Price record for %s at %s already exists", symbol, timestamp)
                    return False
                
                if existing and override:
                    logger.debug("Updating existing price record for %s at %s", symbol, timestamp)
                    
                    # Update existing record
                    existing.price = price_data.get('price', existing.price)
                    existing.volume = price_data.get('volume', existing.volume)
                    existing.high = price_data.get('high', existing.high)
                    existing.low = price_data.get('low', existing.low)
                    existing.open = price_data.get('open', existing.open)
                    existing.close = price_data.get('close', existing.close)
                    existing.source = price_data.get('source', existing.source)
                    
                else:
                    # Create new record
                    logger.debug("Creating new price record for %s at %s", symbol, timestamp)
                    
                    record = PriceRecord(
                        symbol=symbol,
                        timestamp=timestamp,
                        price=price_data.get('price', 0),
                        volume=price_data.get('volume'),
                        high=price_data.get('high'),
                        low=price_data.get('low'),
                        open=price_data.get('open'),
                        close=price_data.get('close', price_data.get('price', 0)),
                        source=price_data.get('source')
                    )
                    
                    session.add(record)
                
                session.commit()
                logger.debug("Successfully saved price record for %s at %s", symbol, timestamp)
                return True
                
        except Exception as e:
            logger.error("Error saving price record for %s: %s", symbol, str(e))
            return False

    def get_price_data(
        self, 
        symbol, 
        start_date=None, 
        end_date=None, 
        limit=None,
        interval=None
    ):
        """
        Get price data from the database.
        
        Args:
            symbol: Cryptocurrency symbol
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Optional limit on number of records returned
            interval: Optional interval for resampling (e.g., '1h', '1d')
            
        Returns:
            list: List of price records as dictionaries
        """
        result = []
        
        try:
            # For resampling, we need to use pandas
            if interval:
                # Get all data and resample with pandas
                df = pd.read_sql(
                    f"""
                    SELECT * FROM price_records 
                    WHERE symbol = '{symbol}'
                    {f"AND timestamp >= '{start_date}'" if start_date else ""}
                    {f"AND timestamp <= '{end_date}'" if end_date else ""}
                    ORDER BY timestamp
                    """, 
                    self._engine
                )
                
                if len(df) == 0:
                    return []
                
                # Convert timestamp column to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                
                # Resample data
                resampled = df.resample(interval).agg({
                    'price': 'mean',
                    'volume': 'sum',
                    'high': 'max',
                    'low': 'min',
                    'open': 'first',
                    'close': 'last',
                    'source': 'first'
                }).dropna(how='all')
                
                # Reset index and convert back to dict
                resampled = resampled.reset_index()
                
                # Apply limit if provided
                if limit:
                    resampled = resampled.tail(limit)
                    
                # Convert to list of dictionaries
                result = resampled.to_dict('records')
                for record in result:
                    record['timestamp'] = record['timestamp'].isoformat()
                    
                return result
                    
            # Otherwise, use SQLAlchemy
            with self.session_scope() as session:
                # Build query
                query = session.query(PriceRecord).filter(PriceRecord.symbol == symbol)
                
                # Apply date filters
                if start_date:
                    if isinstance(start_date, str):
                        start_date = datetime.fromisoformat(start_date)
                    query = query.filter(PriceRecord.timestamp >= start_date)
                    
                if end_date:
                    if isinstance(end_date, str):
                        end_date = datetime.fromisoformat(end_date)
                    query = query.filter(PriceRecord.timestamp <= end_date)
                
                # Apply ordering
                query = query.order_by(PriceRecord.timestamp.desc())
                
                # Apply limit
                if limit:
                    query = query.limit(limit)
                
                # Execute query
                records = query.all()
                
                # Convert to dictionaries
                for record in records:
                    record_dict = {c.name: getattr(record, c.name) for c in record.__table__.columns}
                    
                    # Convert datetime objects to ISO format strings
                    for key, value in record_dict.items():
                        if isinstance(value, datetime):
                            record_dict[key] = value.isoformat()
                    
                    result.append(record_dict)
                
                logger.debug("Retrieved %d price records for %s", len(result), symbol)
                return result
                
        except Exception as e:
            logger.error("Error retrieving price data for %s: %s", symbol, str(e))
            return []