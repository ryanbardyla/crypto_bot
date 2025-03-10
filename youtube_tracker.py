# youtube_tracker.py
import os
import json
import time
import schedule
import threading
import logging
import requests
import googleapiclient.discovery
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from prometheus_client import Counter, Gauge, Summary, Histogram, start_http_server

# Import existing utility classes
from crypto_sentiment_analyzer import CryptoSentimentAnalyzer
from database_manager import DatabaseManager
from utils.logging_config import setup_logging, get_module_logger

# Prometheus metrics
VIDEOS_PROCESSED = Counter('youtube_videos_processed_total', 'Number of videos processed')
VIDEOS_SKIPPED = Counter('youtube_videos_skipped_total', 'Number of videos skipped', ['reason'])
API_CALLS = Counter('youtube_api_calls_total', 'Number of YouTube API calls', ['endpoint'])
TRANSCRIPTS_FETCHED = Counter('youtube_transcripts_fetched_total', 'Number of transcripts fetched')
TRANSCRIPTS_FAILED = Counter('youtube_transcripts_failed_total', 'Number of transcript fetch failures', ['reason'])
DB_OPERATIONS = Counter('youtube_db_operations_total', 'Number of database operations', ['operation'])
SENTIMENT_SCORE = Gauge('youtube_sentiment_score', 'Sentiment scores by channel and video', ['channel_id', 'video_id', 'symbol'])
CHANNELS_MONITORED = Gauge('youtube_channels_monitored', 'Number of channels being monitored')
ACTIVE_WORKERS = Gauge('youtube_active_workers', 'Number of active worker threads')
BULLISH_KEYWORDS = Gauge('youtube_bullish_keywords', 'Number of bullish keywords found', ['channel_id', 'video_id'])
BEARISH_KEYWORDS = Gauge('youtube_bearish_keywords', 'Number of bearish keywords found', ['channel_id', 'video_id'])
PROCESSED_VIDEOS_LAST_RUN = Gauge('youtube_processed_videos_last_run', 'Number of videos processed in last run')
PROCESSING_TIME = Summary('youtube_processing_seconds', 'Time spent processing videos', ['operation'])
TEXT_LENGTH = Summary('youtube_text_length_bytes', 'Length of transcript text in bytes')
SENTIMENT_HISTOGRAM = Histogram('youtube_sentiment_distribution', 'Distribution of sentiment scores', 
                                ['channel_id'], buckets=[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])

# Load environment variables
load_dotenv()
logger = get_module_logger("YouTubeTracker")

# Rate limiter for API calls
class RateLimiter:
    def __init__(self, calls_per_second=1):
        self.lock = threading.Lock()
        self.calls_per_second = calls_per_second
        self.last_call = time.time()
        logger.debug(f"Initialized RateLimiter with {calls_per_second} calls per second")
    
    def wait(self):
        with self.lock:
            elapsed = time.time() - self.last_call
            if elapsed < 1.0 / self.calls_per_second:
                wait_time = (1.0 / self.calls_per_second) - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                time.sleep(wait_time)
            self.last_call = time.time()

# SQLAlchemy model
Base = declarative_base()

class SentimentRecord(Base):
    __tablename__ = 'sentiment_youtube'
    
    id = Column(Integer, primary_key=True)
    video_id = Column(String(100), unique=True)
    channel_id = Column(String(100), index=True)
    title = Column(String(500))
    publish_date = Column(DateTime)
    processed_date = Column(DateTime)
    vader_neg = Column(Float)
    vader_neu = Column(Float)
    vader_pos = Column(Float)
    vader_compound = Column(Float)
    bullish_keywords = Column(Integer)
    bearish_keywords = Column(Integer)
    keyword_sentiment = Column(Float)
    combined_score = Column(Float, index=True)
    text_length = Column(Integer)
    source = Column(String(200))
    mentioned_cryptos = Column(Text)  # Stored as JSON string
    
    def __repr__(self):
        return f"<SentimentRecord(video_id='{self.video_id}', combined_score={self.combined_score})>"

class YouTubeTracker:
    def __init__(self, config_file="youtube_tracker_config.json"):
        self.active_workers = 0
        self.db_manager = DatabaseManager(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=os.environ.get("POSTGRES_PORT", "5432"),
            database=os.environ.get("POSTGRES_DB", "trading_db"),
            user=os.environ.get("POSTGRES_USER", "bot_user"),
            password=os.environ.get("POSTGRES_PASSWORD", "secure_password")
        )
        self.load_config(config_file)
        self.setup_youtube_api()
        self.setup_database()
        self.sentiment_analyzer = CryptoSentimentAnalyzer()
        self.rate_limiter = RateLimiter(calls_per_second=0.5)  # YouTube API has quotas
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_videos)
        CHANNELS_MONITORED.set(len(self.channel_ids))
        logger.info(f"YouTubeTracker initialized with {len(self.channel_ids)} channels")
    
    def setup_database(self):
        """Set up database connections and tables"""
        try:
            # Use the engine from DatabaseManager
            Base.metadata.create_all(self.db_manager.engine)
            self.Session = sessionmaker(bind=self.db_manager.engine)
            DB_OPERATIONS.labels('setup').inc()
            logger.info(f"Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to set up database: {str(e)}")
            raise
    
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        with PROCESSING_TIME.labels('config_loading').time():
            try:
                # Check for containerized config
                container_config = os.path.join("/app/config", os.path.basename(config_file))
                if os.path.exists(container_config):
                    config_path = container_config
                    logger.info(f"Using containerized config at {container_config}")
                else:
                    config_path = config_file
                
                with open(config_path, "r") as f:
                    self.config = json.load(f)
                
                self.channel_ids = self.config.get("channel_ids", [])
                self.channel_names = self.config.get("channel_names", {})
                self.check_interval_hours = self.config.get("check_interval_hours", 6)
                self.video_age_limit_days = self.config.get("video_age_limit_days", 7)
                self.max_concurrent_channels = self.config.get("max_concurrent_channels", 5)
                self.max_concurrent_videos = self.config.get("max_concurrent_videos", 10)
                
                # Override with environment variables if available
                db_uri = os.environ.get("DB_URI")
                if db_uri:
                    logger.info(f"Using database URI from environment: {db_uri}")
                
                logger.info(f"Configuration loaded: {len(self.channel_ids)} channels, {self.check_interval_hours}h interval, {self.video_age_limit_days} days age limit")
            except Exception as e:
                logger.error(f"Failed to load configuration: {str(e)}")
                raise
    
    def setup_youtube_api(self):
        """Initialize the YouTube API client"""
        with PROCESSING_TIME.labels('api_setup').time():
            try:
                api_key = os.getenv("YOUTUBE_API_KEY")
                if not api_key:
                    logger.error("YouTube API key not found in environment variables")
                    raise ValueError("YouTube API key not found. Please set YOUTUBE_API_KEY in your .env file.")
                
                self.youtube = googleapiclient.discovery.build(
                    "youtube", "v3", developerKey=api_key,
                    cache_discovery=False
                )
                logger.info("YouTube API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize YouTube API: {str(e)}")
                raise
    
    def get_channel_uploads(self, channel_id):
        """Get the uploads playlist ID for a YouTube channel"""
        with PROCESSING_TIME.labels('channel_lookup').time():
            try:
                API_CALLS.labels('channels.list').inc()
                request = self.youtube.channels().list(
                    part="contentDetails",
                    id=channel_id
                )
                self.rate_limiter.wait()
                response = request.execute()
                
                if not response.get("items"):
                    logger.warning(f"No channel found for ID: {channel_id}")
                    VIDEOS_SKIPPED.labels('channel_not_found').inc()
                    return None
                
                uploads_playlist = response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
                logger.info(f"Found uploads playlist {uploads_playlist} for channel {channel_id}")
                return uploads_playlist
            except Exception as e:
                logger.error(f"Error getting uploads for channel {channel_id}: {str(e)}")
                VIDEOS_SKIPPED.labels('playlist_error').inc()
                return None
    
    def get_recent_videos(self, playlist_id, max_results=10):
        """Get recent videos from a playlist"""
        with PROCESSING_TIME.labels('playlist_videos').time():
            videos = []
            try:
                API_CALLS.labels('playlistItems.list').inc()
                request = self.youtube.playlistItems().list(
                    part="snippet",
                    playlistId=playlist_id,
                    maxResults=max_results
                )
                self.rate_limiter.wait()
                response = request.execute()
                
                # Filter videos to only include those within our age limit
                cutoff_date = datetime.now() - timedelta(days=self.video_age_limit_days)
                
                for item in response.get("items", []):
                    snippet = item["snippet"]
                    # Convert ISO 8601 format to datetime
                    dt = datetime.fromisoformat(snippet["publishedAt"].replace("Z", "+00:00"))
                    publish_date = dt.replace(tzinfo=None)  # Remove timezone info
                    
                    if publish_date >= cutoff_date:
                        videos.append({
                            "video_id": snippet["resourceId"]["videoId"],
                            "channel_id": snippet["channelId"],
                            "title": snippet["title"],
                            "publish_date": publish_date
                        })
                
                logger.info(f"Retrieved {len(videos)} recent videos from playlist {playlist_id}")
                return videos
            except Exception as e:
                logger.error(f"Error getting videos from playlist {playlist_id}: {str(e)}")
                VIDEOS_SKIPPED.labels('playlist_fetch_error').inc()
                return []
    
    def get_youtube_transcript(self, video_id):
        """Get transcript for a YouTube video"""
        with PROCESSING_TIME.labels('transcript_fetch').time():
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = ' '.join([item['text'] for item in transcript_list])
                text_length = len(transcript_text)
                TRANSCRIPTS_FETCHED.inc()
                TEXT_LENGTH.observe(text_length)
                logger.info(f"Successfully retrieved transcript for video {video_id}: {text_length} characters")
                return transcript_text
            except Exception as e:
                logger.warning(f"Could not retrieve transcript for video {video_id}: {str(e)}")
                TRANSCRIPTS_FAILED.labels('api_error').inc()
                return None
    
    def process_video(self, video_info):
        """Process a video to extract and analyze its transcript"""
        self.active_workers += 1
        ACTIVE_WORKERS.set(self.active_workers)
        
        with PROCESSING_TIME.labels('video_processing').time():
            session = self.Session()
            try:
                DB_OPERATIONS.labels('query').inc()
                existing = session.query(SentimentRecord).filter_by(video_id=video_info["video_id"]).first()
                
                if existing:
                    logger.info(f"Video {video_info['video_id']} already processed - skipping")
                    VIDEOS_SKIPPED.labels('already_processed').inc()
                    session.close()
                    self.active_workers -= 1
                    return False
                
                # Get transcript
                transcript = self.get_youtube_transcript(video_info["video_id"])
                if not transcript:
                    logger.warning(f"No transcript available for {video_info['video_id']} - skipping")
                    VIDEOS_SKIPPED.labels('no_transcript').inc()
                    session.close()
                    self.active_workers -= 1
                    return False
                
                # Analyze sentiment
                source_id = f"youtube-{video_info['video_id']}"
                start_time = time.time()
                sentiment = self.sentiment_analyzer.analyze_text(transcript, source=source_id)
                sentiment_time = time.time() - start_time
                PROCESSING_TIME.labels('sentiment_analysis').observe(sentiment_time)
                
                # Store data in sentiment metrics
                combined_score = sentiment.get("combined_score", 0)
                SENTIMENT_HISTOGRAM.labels(
                    channel_id=video_info["channel_id"]
                ).observe(combined_score)
                
                # Create database record
                record = SentimentRecord(
                    video_id=video_info["video_id"],
                    channel_id=video_info["channel_id"],
                    title=video_info["title"],
                    publish_date=video_info["publish_date"],
                    processed_date=datetime.now(),
                    vader_neg=sentiment.get("vader_sentiment", {}).get("neg", 0),
                    vader_neu=sentiment.get("vader_sentiment", {}).get("neu", 0),
                    vader_pos=sentiment.get("vader_sentiment", {}).get("pos", 0),
                    vader_compound=sentiment.get("vader_sentiment", {}).get("compound", 0),
                    bullish_keywords=sentiment.get("bullish_keywords", 0),
                    bearish_keywords=sentiment.get("bearish_keywords", 0),
                    keyword_sentiment=sentiment.get("keyword_sentiment", 0),
                    combined_score=combined_score,
                    text_length=sentiment.get("text_length", 0),
                    source=sentiment.get("source", source_id),
                    mentioned_cryptos=json.dumps(sentiment.get("mentioned_cryptos", []))
                )
                
                DB_OPERATIONS.labels('insert').inc()
                session.add(record)
                session.commit()
                VIDEOS_PROCESSED.inc()
                
                # Update metrics
                BULLISH_KEYWORDS.labels(
                    channel_id=video_info["channel_id"],
                    video_id=video_info["video_id"]
                ).set(sentiment.get('bullish_keywords', 0))
                
                BEARISH_KEYWORDS.labels(
                    channel_id=video_info["channel_id"],
                    video_id=video_info["video_id"]
                ).set(sentiment.get('bearish_keywords', 0))
                
                # Set sentiment scores for each mentioned crypto
                mentioned_cryptos = sentiment.get("mentioned_cryptos", [])
                for crypto in mentioned_cryptos:
                    SENTIMENT_SCORE.labels(
                        channel_id=video_info["channel_id"],
                        video_id=video_info["video_id"],
                        symbol=crypto
                    ).set(combined_score)
                
                logger.info(f"Successfully processed video {video_info['video_id']} - Score: {combined_score:.2f}")
                
                # Also save to file for backward compatibility
                os.makedirs("sentiment_data", exist_ok=True)
                with open(f"sentiment_data/{video_info['video_id']}.json", "w") as f:
                    sentiment["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    sentiment["video_id"] = video_info["video_id"]
                    sentiment["title"] = video_info["title"]
                    json.dump(sentiment, f, indent=2)
                
                return True
                
            except Exception as e:
                session.rollback()
                DB_OPERATIONS.labels('rollback').inc()
                logger.error(f"Error processing video {video_info.get('video_id', 'unknown')}: {str(e)}")
                return False
                
            finally:
                session.close()
                self.active_workers -= 1
    
    def run_tracker(self):
        """Main tracking function to process all channels and videos"""
        with PROCESSING_TIME.labels('tracker_run').time():
            logger.info("Starting YouTube tracker run")
            processed_count = 0
            skipped_count = 0
            
            try:
                for channel_id in self.channel_ids:
                    try:
                        channel_name = self.channel_names.get(channel_id, f"Channel {channel_id}")
                        logger.info(f"Processing channel: {channel_name} (ID: {channel_id})")
                        
                        # Get channel uploads playlist
                        uploads_playlist = self.get_channel_uploads(channel_id)
                        if not uploads_playlist:
                            logger.warning(f"Could not get uploads playlist for channel {channel_name} - skipping")
                            skipped_count += 1
                            continue
                        
                        # Get recent videos
                        videos = self.get_recent_videos(uploads_playlist)
                        logger.info(f"Found {len(videos)} recent videos from {channel_name}")
                        
                        # Process each video
                        for video in videos:
                            if self.process_video(video):
                                processed_count += 1
                            else:
                                skipped_count += 1
                        
                        # Sleep briefly between channels to avoid API rate limits
                        time.sleep(2)
                    except Exception as e:
                        logger.error(f"Error processing channel {channel_id}: {str(e)}")
                        skipped_count += 1
                
                PROCESSED_VIDEOS_LAST_RUN.set(processed_count)
                logger.info(f"Tracker run complete. Processed {processed_count} new videos, skipped {skipped_count}")
                
            except Exception as e:
                logger.error(f"Error in tracker run: {str(e)}")
    
    def start_scheduler(self):
        """Start the scheduler for periodic tracking"""
        try:
            # Start metrics server if available
            metrics_port = int(os.environ.get("METRICS_PORT", 8000))
            start_http_server(metrics_port)
            logger.info(f"Prometheus metrics server started on port {metrics_port}")
            
            logger.info(f"Starting YouTube tracker scheduler with {self.check_interval_hours} hour interval")
            self.run_tracker()
            
            # Schedule regular runs
            schedule.every(self.check_interval_hours).hours.do(self.run_tracker)
            
            # Schedule daily database maintenance
            schedule.every().day.at("03:00").do(self.vacuum_database)
            
            # Run the scheduler
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("YouTube tracker scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {str(e)}")
    
    def vacuum_database(self):
        """Perform database maintenance"""
        logger.info("Running database maintenance...")
        try:
            # For PostgreSQL, use VACUUM ANALYZE
            with self.db_manager.engine.connect() as conn:
                conn.execute("VACUUM ANALYZE sentiment_youtube")
            
            logger.info("Database maintenance completed successfully")
            DB_OPERATIONS.labels('vacuum').inc()
        except Exception as e:
            logger.error(f"Error during database maintenance: {e}")

# Main execution
if __name__ == "__main__":
    try:
        tracker = YouTubeTracker()
        tracker.start_scheduler()
    except Exception as e:
        logger.error(f"Failed to start YouTube tracker: {str(e)}")