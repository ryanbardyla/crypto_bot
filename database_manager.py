# database_manager.py

import os
import json
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("database_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DatabaseManager")

Base = declarative_base()

class SentimentRecord(Base):
    __tablename__ = 'sentiment_records'
    
    id = Column(Integer, primary_key=True)
    record_id = Column(String(100), unique=True, nullable=False)  # Could be video_id or tweet_id
    record_type = Column(String(10), nullable=False)  # 'youtube' or 'twitter'
    channel_id = Column(String(100), index=True)
    username = Column(String(100), index=True)
    title = Column(String(500))
    text = Column(Text)
    publish_date = Column(DateTime, index=True)
    processed_date = Column(DateTime, default=datetime.now)
    followers_count = Column(Integer)
    retweet_count = Column(Integer)
    like_count = Column(Integer)
    view_count = Column(Integer)
    
    # Sentiment metrics
    vader_neg = Column(Float)
    vader_neu = Column(Float)
    vader_pos = Column(Float)
    vader_compound = Column(Float)
    bullish_keywords = Column(Integer)
    bearish_keywords = Column(Integer)
    keyword_sentiment = Column(Float)
    combined_score = Column(Float, index=True)
    text_length = Column(Integer)
    
    # Symbol relation
    symbol = Column(String(10), index=True)
    
    # Source information
    source = Column(String(200))
    
    # Define indexes
    __table_args__ = (
        Index('idx_sentiment_date_score', processed_date, combined_score),
        Index('idx_sentiment_source_date', source, processed_date),
        Index('idx_sentiment_record_type_date', record_type, processed_date),
    )

class DatabaseManager:
    def __init__(self, db_path="sqlite:///sentiment_database.db", enable_pooling=True):
        self.db_path = db_path
        self.enable_pooling = enable_pooling
        self.setup_database()
        
    def setup_database(self):
        """Initialize database connection with connection pooling."""
        try:
            if self.enable_pooling:
                # Configure connection pooling
                self.engine = create_engine(
                    self.db_path,
                    poolclass=QueuePool,
                    pool_size=10,
                    max_overflow=20,
                    pool_timeout=30,
                    pool_recycle=1800  # Recycle connections every 30 minutes
                )
            else:
                self.engine = create_engine(self.db_path)
                
            # Create tables if they don't exist
            Base.metadata.create_all(self.engine)
            
            # Create a scoped session factory
            self.session_factory = scoped_session(sessionmaker(bind=self.engine))
            
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to set up database: {str(e)}")
            raise
    
    def get_session(self):
        """Get a database session."""
        return self.session_factory()
    
    def save_sentiment_data(self, data, record_type, record_id, override=False):
        """
        Save sentiment data to the database.
        
        Args:
            data (dict): The sentiment data
            record_type (str): Type of record ('youtube' or 'twitter')
            record_id (str): Unique identifier (video_id or tweet_id)
            override (bool): Whether to override existing data
        
        Returns:
            bool: Success status
        """
        session = self.get_session()
        try:
            existing = session.query(SentimentRecord).filter_by(
                record_id=record_id, 
                record_type=record_type
            ).first()
            
            if existing and not override:
                logger.info(f"{record_type.capitalize()} record {record_id} already exists")
                session.close()
                return False
                
            if existing and override:
                # Update existing record
                for key, value in data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                record = existing
            else:
                # Create new record
                record = SentimentRecord(
                    record_id=record_id,
                    record_type=record_type,
                    channel_id=data.get('channel_id'),
                    username=data.get('username'),
                    title=data.get('title'),
                    text=data.get('text', ''),
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
                    source=data.get('source', f'{record_type}-{record_id}')
                )
                session.add(record)
                
            session.commit()
            logger.info(f"Successfully saved {record_type} record {record_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving {record_type} record {record_id}: {str(e)}")
            return False
        finally:
            session.close()
    
    def get_sentiment_data(self, record_type=None, symbol=None, days=None, limit=None):
        """
        Retrieve sentiment data with optional filtering.
        
        Args:
            record_type (str, optional): Filter by record type ('youtube' or 'twitter')
            symbol (str, optional): Filter by cryptocurrency symbol
            days (int, optional): Filter by number of days back
            limit (int, optional): Limit the number of records returned
            
        Returns:
            list: List of sentiment records as dictionaries
        """
        session = self.get_session()
        try:
            query = session.query(SentimentRecord)
            
            if record_type:
                query = query.filter(SentimentRecord.record_type == record_type)
                
            if symbol:
                query = query.filter(SentimentRecord.symbol == symbol)
                
            if days:
                cutoff_date = datetime.now() - timedelta(days=days)
                query = query.filter(SentimentRecord.processed_date >= cutoff_date)
                
            query = query.order_by(SentimentRecord.processed_date.desc())
            
            if limit:
                query = query.limit(limit)
                
            records = query.all()
            
            result = []
            for record in records:
                record_dict = {
                    'record_id': record.record_id,
                    'record_type': record.record_type,
                    'channel_id': record.channel_id,
                    'username': record.username,
                    'title': record.title,
                    'publish_date': record.publish_date,
                    'processed_date': record.processed_date,
                    'vader_sentiment': {
                        'neg': record.vader_neg,
                        'neu': record.vader_neu,
                        'pos': record.vader_pos,
                        'compound': record.vader_compound
                    },
                    'bullish_keywords': record.bullish_keywords,
                    'bearish_keywords': record.bearish_keywords,
                    'keyword_sentiment': record.keyword_sentiment,
                    'combined_score': record.combined_score,
                    'text_length': record.text_length,
                    'symbol': record.symbol,
                    'source': record.source
                }
                result.append(record_dict)
                
            return result
        except Exception as e:
            logger.error(f"Error retrieving sentiment data: {str(e)}")
            return []
        finally:
            session.close()
    
    def get_aggregated_sentiment(self, days=7, symbol=None):
        """
        Get aggregated sentiment data grouped by date.
        
        Args:
            days (int): Number of days to look back
            symbol (str, optional): Filter by cryptocurrency symbol
            
        Returns:
            list: List of aggregated sentiment data by date
        """
        from sqlalchemy import func, cast, Date
        
        session = self.get_session()
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            query = session.query(
                cast(SentimentRecord.processed_date, Date).label('date'),
                func.count(SentimentRecord.id).label('record_count'),
                func.avg(SentimentRecord.combined_score).label('avg_sentiment'),
                func.sum(SentimentRecord.bullish_keywords).label('total_bullish'),
                func.sum(SentimentRecord.bearish_keywords).label('total_bearish'),
                func.count(func.case([(SentimentRecord.record_type == 'twitter', 1)])).label('has_twitter'),
                func.count(func.case([(SentimentRecord.record_type == 'youtube', 1)])).label('has_youtube')
            ).filter(SentimentRecord.processed_date >= cutoff_date)
            
            if symbol:
                query = query.filter(SentimentRecord.symbol == symbol)
                
            query = query.group_by(cast(SentimentRecord.processed_date, Date))\
                         .order_by(cast(SentimentRecord.processed_date, Date))
            
            result = [dict(row._mapping) for row in query.all()]
            return result
        except Exception as e:
            logger.error(f"Error getting aggregated sentiment: {str(e)}")
            return []
        finally:
            session.close()
    
    def migrate_json_to_db(self, json_dir="sentiment_data"):
        """
        Migrate existing JSON sentiment files to the database.
        
        Args:
            json_dir (str): Directory containing JSON sentiment files
            
        Returns:
            tuple: (success_count, failed_count)
        """
        if not os.path.exists(json_dir):
            logger.warning(f"Directory {json_dir} does not exist")
            return 0, 0
            
        success_count = 0
        failed_count = 0
        
        for filename in os.listdir(json_dir):
            if not filename.endswith('.json'):
                continue
                
            file_path = os.path.join(json_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                record_id = filename.replace('.json', '')
                record_type = 'youtube' if 'video_id' in data else 'twitter'
                
                if self.save_sentiment_data(data, record_type, record_id):
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Error migrating {filename}: {str(e)}")
                failed_count += 1
                
        logger.info(f"Migration completed: {success_count} successes, {failed_count} failures")
        return success_count, failed_count
    
    def vacuum_database(self):
        """Optimize the database by vacuuming (SQLite specific)."""
        try:
            with self.engine.connect() as conn:
                conn.execute("VACUUM")
            logger.info("Database vacuumed successfully")
            return True
        except Exception as e:
            logger.error(f"Error vacuuming database: {str(e)}")
            return False
    
    def cleanup_old_records(self, days=90):
        """
        Remove records older than specified days.
        
        Args:
            days (int): Delete records older than this many days
            
        Returns:
            int: Number of records deleted
        """
        session = self.get_session()
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            count = session.query(SentimentRecord)\
                .filter(SentimentRecord.processed_date < cutoff_date)\
                .delete()
            session.commit()
            logger.info(f"Cleaned up {count} records older than {days} days")
            return count
        except Exception as e:
            session.rollback()
            logger.error(f"Error cleaning up old records: {str(e)}")
            return 0
        finally:
            session.close()

# If run directly, perform a database check
if __name__ == "__main__":
    print("Database Manager - Performing initial setup and checks")
    db_manager = DatabaseManager()
    
    # Check connection
    session = db_manager.get_session()
    try:
        result = session.execute("SELECT 1").scalar()
        print(f"Database connection test: {'Success' if result == 1 else 'Failed'}")
    except Exception as e:
        print(f"Database connection error: {str(e)}")
    finally:
        session.close()
    
    # Check if migration is needed
    if os.path.exists("sentiment_data"):
        file_count = len([f for f in os.listdir("sentiment_data") if f.endswith('.json')])
        if file_count > 0:
            migrate = input(f"Found {file_count} JSON files. Migrate to database? (y/n): ").lower() == 'y'
            if migrate:
                success, failed = db_manager.migrate_json_to_db()
                print(f"Migration completed: {success} successes, {failed} failures")
    
    print("Database Manager setup complete")