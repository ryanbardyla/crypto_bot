# youtube_tracker.py (updated with Prometheus metrics)
import os
import json
import time
import logging
import schedule
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import googleapiclient.discovery
from youtube_transcript_api import YouTubeTranscriptApi
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from crypto_sentiment_analyzer import CryptoSentimentAnalyzer

# Import the centralized logging configuration
from utils.logging_config import get_module_logger

# Import Prometheus metrics
from prometheus_client import Counter, Gauge, Summary, Histogram, start_http_server

# Set up metrics
# Counters (these only go up)
VIDEOS_PROCESSED = Counter('youtube_videos_processed_total', 'Number of videos processed')
VIDEOS_SKIPPED = Counter('youtube_videos_skipped_total', 'Number of videos skipped', ['reason'])
API_CALLS = Counter('youtube_api_calls_total', 'Number of YouTube API calls', ['endpoint'])
TRANSCRIPTS_FETCHED = Counter('youtube_transcripts_fetched_total', 'Number of transcripts fetched')
TRANSCRIPTS_FAILED = Counter('youtube_transcripts_failed_total', 'Number of transcript fetch failures', ['reason'])
DB_OPERATIONS = Counter('youtube_db_operations_total', 'Number of database operations', ['operation'])

# Gauges (these can go up and down)
SENTIMENT_SCORE = Gauge('youtube_sentiment_score', 'Sentiment scores by channel and video', ['channel_id', 'video_id', 'symbol'])
CHANNELS_MONITORED = Gauge('youtube_channels_monitored', 'Number of channels being monitored')
ACTIVE_WORKERS = Gauge('youtube_active_workers', 'Number of active worker threads')
BULLISH_KEYWORDS = Gauge('youtube_bullish_keywords', 'Number of bullish keywords found', ['channel_id', 'video_id'])
BEARISH_KEYWORDS = Gauge('youtube_bearish_keywords', 'Number of bearish keywords found', ['channel_id', 'video_id'])
PROCESSED_VIDEOS_LAST_RUN = Gauge('youtube_processed_videos_last_run', 'Number of videos processed in last run')

# Summaries (these track distributions)
PROCESSING_TIME = Summary('youtube_processing_seconds', 'Time spent processing videos', ['operation'])
TEXT_LENGTH = Summary('youtube_text_length_bytes', 'Length of transcript text in bytes')

# Histograms (these provide more detailed distributions)
SENTIMENT_HISTOGRAM = Histogram('youtube_sentiment_distribution', 'Distribution of sentiment scores', 
                               buckets=[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])

# Load environment variables
load_dotenv()

# Get logger for this module
logger = get_module_logger("YouTubeTracker")

# Define rate limiter class
class RateLimiter:
    """Rate limiter to prevent hitting API rate limits"""
    def __init__(self, calls_per_second=1):
        self.rate = calls_per_second
        self.last_call = 0
        self.lock = threading.Lock()
        logger.debug(f"Initialized RateLimiter with {calls_per_second} calls per second")
        
    def wait(self):
        """Wait if needed to respect the rate limit"""
        with self.lock:
            elapsed = time.time() - self.last_call
            if elapsed < (1.0 / self.rate):
                wait_time = (1.0 / self.rate) - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                time.sleep(wait_time)
            self.last_call = time.time()

# Define database model
Base = declarative_base()

class SentimentRecord(Base):
    __tablename__ = 'sentiment_youtube'
    
    id = Column(Integer, primary_key=True)
    video_id = Column(String, unique=True)
    channel_id = Column(String)
    title = Column(String)
    publish_date = Column(DateTime)
    processed_date = Column(DateTime)
    vader_neg = Column(Float)
    vader_neu = Column(Float)
    vader_pos = Column(Float)
    vader_compound = Column(Float)
    bullish_keywords = Column(Integer)
    bearish_keywords = Column(Integer)
    keyword_sentiment = Column(Float)
    combined_score = Column(Float)
    text_length = Column(Integer)
    source = Column(String)
    mentioned_cryptos = Column(String)  # Stored as JSON string

class YouTubeTracker:
    def __init__(self, config_file="youtube_tracker_config.json"):
        # Track active workers for metrics
        self.active_workers = 0
        ACTIVE_WORKERS.set(self.active_workers)
        
        # Load configuration and set up components
        self.load_config(config_file)
        self.setup_youtube_api()
        self.setup_database()
        self.sentiment_analyzer = CryptoSentimentAnalyzer()
        self.rate_limiter = RateLimiter(calls_per_second=0.5)  # YouTube API has quotas
        
        # Initialize executor with worker tracking
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_videos)
        
        # Update channels monitored metric
        CHANNELS_MONITORED.set(len(self.channel_ids))
        
        logger.info(f"YouTubeTracker initialized with {len(self.channel_ids)} channels")
        
    def load_config(self, config_file):
        # Track config loading time
        with PROCESSING_TIME.labels('config_loading').time():
            try:
                # First check for config file in the config directory (for containerized environments)
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
                self.db_path = self.config.get("db_path", "sqlite:///sentiment_database.db")
                self.max_concurrent_channels = self.config.get("max_concurrent_channels", 5)
                self.max_concurrent_videos = self.config.get("max_concurrent_videos", 10)
                
                # Handle database connection string from environment variable
                db_uri = os.environ.get("DB_URI")
                if db_uri:
                    self.db_path = db_uri
                    logger.info(f"Using database URI from environment: {db_uri}")
                
                # Set prometheus metrics
                CHANNELS_MONITORED.set(len(self.channel_ids))
                
                logger.info(f"Configuration loaded: {len(self.channel_ids)} channels, {self.check_interval_hours}h interval, {self.video_age_limit_days} days age limit")
            except Exception as e:
                logger.error(f"Failed to load configuration: {str(e)}")
                raise
    
    def setup_youtube_api(self):
        with PROCESSING_TIME.labels('api_setup').time():
            try:
                # Get API key from environment variable
                api_key = os.getenv("YOUTUBE_API_KEY")
                if not api_key:
                    logger.error("YouTube API key not found in environment variables")
                    raise ValueError("YouTube API key not found. Please set YOUTUBE_API_KEY in your .env file.")
                
                # Initialize the YouTube API client
                self.youtube = googleapiclient.discovery.build(
                    "youtube", "v3", developerKey=api_key, cache_discovery=False
                )
                logger.info("YouTube API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize YouTube API: {str(e)}")
                raise
    
    def setup_database(self):
        with PROCESSING_TIME.labels('db_setup').time():
            try:
                # Ensure the db_path is correctly formatted for SQLite
                if not self.db_path.startswith("sqlite:///") and not self.db_path.startswith("postgresql://"):
                    logger.warning(f"Fixing database path format: {self.db_path}")
                    if self.db_path.startswith("sqlite://"):
                        self.db_path = self.db_path.replace("sqlite://", "sqlite:///")
                    else:
                        self.db_path = f"sqlite:///{self.db_path}"
                
                # Create database engine and tables
                self.engine = create_engine(self.db_path)
                Base.metadata.create_all(self.engine)
                self.Session = sessionmaker(bind=self.engine)
                
                # Record success in metrics
                DB_OPERATIONS.labels('setup').inc()
                
                logger.info(f"Database connection established at {self.db_path}")
            except Exception as e:
                logger.error(f"Failed to set up database: {str(e)}")
                raise
    
    def get_channel_uploads(self, channel_id):
        """
        Get the uploads playlist ID for a YouTube channel
        """
        # Use context manager to time this operation
        with PROCESSING_TIME.labels('channel_lookup').time():
            try:
                # Record the API call
                API_CALLS.labels('channels.list').inc()
                
                # Request channel details
                request = self.youtube.channels().list(
                    part="contentDetails",
                    id=channel_id
                )
                response = request.execute()
                
                # Check if the channel was found
                if not response.get("items"):
                    logger.warning(f"No channel found for ID: {channel_id}")
                    VIDEOS_SKIPPED.labels('channel_not_found').inc()
                    return None
                
                # Extract uploads playlist ID
                uploads_playlist = response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
                logger.info(f"Found uploads playlist {uploads_playlist} for channel {channel_id}")
                return uploads_playlist
            except Exception as e:
                logger.error(f"Error getting uploads for channel {channel_id}: {str(e)}")
                VIDEOS_SKIPPED.labels('playlist_error').inc()
                return None
    
    def get_recent_videos(self, playlist_id, max_results=10):
        """
        Get recent videos from a YouTube playlist
        """
        with PROCESSING_TIME.labels('playlist_videos').time():
            try:
                videos = []
                # Record the API call
                API_CALLS.labels('playlistItems.list').inc()
                
                # Request playlist items
                request = self.youtube.playlistItems().list(
                    part="snippet",
                    playlistId=playlist_id,
                    maxResults=max_results
                )
                response = request.execute()
                
                # Define cutoff date for video age filtering
                cutoff_date = datetime.now() - timedelta(days=self.video_age_limit_days)
                
                # Process each video in the playlist
                for item in response.get("items", []):
                    snippet = item["snippet"]
                    # Convert ISO 8601 timestamp to datetime
                    dt = datetime.fromisoformat(snippet["publishedAt"].replace("Z", "+00:00"))
                    publish_date = dt.replace(tzinfo=None)  # Remove timezone info
                    
                    # Only include videos within the age limit
                    if publish_date >= cutoff_date:
                        videos.append({
                            "video_id": snippet["resourceId"]["videoId"],
                            "title": snippet["title"],
                            "channel_id": snippet["channelId"],
                            "publish_date": publish_date
                        })
                
                logger.info(f"Retrieved {len(videos)} recent videos from playlist {playlist_id}")
                return videos
            except Exception as e:
                logger.error(f"Error getting videos from playlist {playlist_id}: {str(e)}")
                VIDEOS_SKIPPED.labels('playlist_fetch_error').inc()
                return []
    
    def get_youtube_transcript(self, video_id):
        """
        Get transcript for a YouTube video
        """
        with PROCESSING_TIME.labels('transcript_fetch').time():
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = ' '.join([item['text'] for item in transcript_list])
                text_length = len(transcript_text)
                
                # Record metrics
                TRANSCRIPTS_FETCHED.inc()
                TEXT_LENGTH.observe(text_length)
                
                logger.info(f"Successfully retrieved transcript for video {video_id}: {text_length} characters")
                return transcript_text
            except Exception as e:
                logger.warning(f"Could not retrieve transcript for video {video_id}: {str(e)}")
                TRANSCRIPTS_FAILED.labels('api_error').inc()
                return None
    
    def process_video(self, video_info):
        """
        Process a YouTube video - get transcript and analyze sentiment
        """
        # Update active workers metric
        self.active_workers += 1
        ACTIVE_WORKERS.set(self.active_workers)
        
        # Use context manager to time the entire processing operation
        with PROCESSING_TIME.labels('video_processing').time():
            session = self.Session()
            try:
                # Check if video already processed
                DB_OPERATIONS.labels('query').inc()
                existing = session.query(SentimentRecord).filter_by(video_id=video_info["video_id"]).first()
                if existing:
                    logger.info(f"Video {video_info['video_id']} already processed - skipping")
                    VIDEOS_SKIPPED.labels('already_processed').inc()
                    session.close()
                    return False
                
                # Get video transcript
                transcript = self.get_youtube_transcript(video_info["video_id"])
                
                if not transcript:
                    logger.warning(f"No transcript available for {video_info['video_id']} - skipping")
                    VIDEOS_SKIPPED.labels('no_transcript').inc()
                    session.close()
                    return False
                
                # Analyze sentiment
                start_time = time.time()
                source_id = f"youtube-{video_info['video_id']}"
                sentiment = self.sentiment_analyzer.analyze_text(transcript, source=source_id)
                sentiment_time = time.time() - start_time
                PROCESSING_TIME.labels('sentiment_analysis').observe(sentiment_time)
                
                # Get the sentiment score and add to histogram
                combined_score = sentiment.get("combined_score", 0)
                SENTIMENT_HISTOGRAM.observe(combined_score)
                
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
                
                # Save to database
                DB_OPERATIONS.labels('insert').inc()
                session.add(record)
                session.commit()
                
                # Update more detailed metrics
                VIDEOS_PROCESSED.inc()
                BULLISH_KEYWORDS.labels(
                    channel_id=video_info["channel_id"],
                    video_id=video_info["video_id"]
                ).set(sentiment.get('bullish_keywords', 0))
                
                BEARISH_KEYWORDS.labels(
                    channel_id=video_info["channel_id"],
                    video_id=video_info["video_id"]
                ).set(sentiment.get('bearish_keywords', 0))
                
                # Record sentiment scores for each mentioned crypto
                mentioned_cryptos = sentiment.get("mentioned_cryptos", [])
                if not mentioned_cryptos:
                    # If no specific crypto mentioned, use a general "CRYPTO" label
                    SENTIMENT_SCORE.labels(
                        channel_id=video_info["channel_id"],
                        video_id=video_info["video_id"],
                        symbol="CRYPTO"
                    ).set(combined_score)
                else:
                    # Record sentiment for each mentioned crypto
                    for crypto in mentioned_cryptos:
                        SENTIMENT_SCORE.labels(
                            channel_id=video_info["channel_id"],
                            video_id=video_info["video_id"],
                            symbol=crypto
                        ).set(combined_score)
                
                logger.info(f"Successfully processed video {video_info['video_id']} - Score: {combined_score:.2f}")
                
                # Save sentiment data to file for easier access
                os.makedirs("sentiment_data", exist_ok=True)
                with open(f"sentiment_data/{video_info['video_id']}.json", "w") as f:
                    sentiment["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    sentiment["video_id"] = video_info['video_id']
                    sentiment["title"] = video_info['title']
                    json.dump(sentiment, f, indent=2)
                
                return True
            except Exception as e:
                session.rollback()
                DB_OPERATIONS.labels('rollback').inc()
                logger.error(f"Error processing video {video_info.get('video_id', 'unknown')}: {str(e)}")
                return False
            finally:
                session.close()
                # Update active workers metric
                self.active_workers -= 1
                ACTIVE_WORKERS.set(self.active_workers)
    
    def run_tracker(self):
        """
        Main function to process YouTube channels and videos
        """
        with PROCESSING_TIME.labels('tracker_run').time():
            logger.info("Starting YouTube tracker run")
            processed_count = 0
            skipped_count = 0
            
            for channel_id in self.channel_ids:
                try:
                    channel_name = self.channel_names.get(channel_id, f"Channel {channel_id}")
                    logger.info(f"Processing channel: {channel_name} (ID: {channel_id})")
                    
                    # Get uploads playlist
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
                    
                    # Avoid hitting API rate limits
                    time.sleep(2)
                except Exception as e:
                    logger.error(f"Error processing channel {channel_id}: {str(e)}")
                    skipped_count += 1
            
            # Update metrics for this run
            PROCESSED_VIDEOS_LAST_RUN.set(processed_count)
            
            logger.info(f"Tracker run complete. Processed {processed_count} new videos, skipped {skipped_count}")
            return processed_count
    
    def start_scheduler(self):
        """
        Start the scheduler to run at regular intervals
        """
        try:
            # Start Prometheus metrics server
            metrics_port = int(os.environ.get("METRICS_PORT", 8000))
            start_http_server(metrics_port)
            logger.info(f"Prometheus metrics server started on port {metrics_port}")
            
            logger.info(f"Starting YouTube tracker scheduler with {self.check_interval_hours} hour interval")
            
            # Run immediately on startup
            self.run_tracker()
            
            # Schedule regular runs
            schedule.every(self.check_interval_hours).hours.do(self.run_tracker)
            
            # Add a daily database vacuum operation to maintain performance
            schedule.every().day.at("03:00").do(self.vacuum_database)
            
            # Keep the scheduler running
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("YouTube tracker scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {str(e)}")
    
    def vacuum_database(self):
        """Perform database maintenance (vacuum)"""
        logger.info("Running database maintenance...")
        try:
            if self.db_path.startswith("sqlite:///"):
                with self.engine.connect() as conn:
                    conn.execute("VACUUM")
                logger.info("Database vacuumed successfully")
                DB_OPERATIONS.labels('vacuum').inc()
            else:
                logger.info("Vacuum operation skipped (not a SQLite database)")
            return True
        except Exception as e:
            logger.error(f"Error during database maintenance: {e}")
            return False

# Main execution
if __name__ == "__main__":
    try:
        tracker = YouTubeTracker()
        tracker.start_scheduler()
    except Exception as e:
        logger.error(f"Failed to start YouTube tracker: {str(e)}")