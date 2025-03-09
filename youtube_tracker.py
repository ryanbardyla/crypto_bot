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
        self.load_config(config_file)
        self.setup_youtube_api()
        self.setup_database()
        self.sentiment_analyzer = CryptoSentimentAnalyzer()
        self.rate_limiter = RateLimiter(calls_per_second=0.5)  # YouTube API has quotas
        self.executor = ThreadPoolExecutor(max_workers=10)  # Adjust based on your needs
        
    def load_config(self, config_file):
        try:
            with open(config_file, "r") as f:
                self.config = json.load(f)
                
            self.channel_ids = self.config.get("channel_ids", [])
            self.channel_names = self.config.get("channel_names", {})
            self.check_interval_hours = self.config.get("check_interval_hours", 6)
            self.video_age_limit_days = self.config.get("video_age_limit_days", 7)
            self.db_path = self.config.get("db_path", "sqlite:///sentiment_database.db")
            self.max_concurrent_channels = self.config.get("max_concurrent_channels", 5)
            self.max_concurrent_videos = self.config.get("max_concurrent_videos", 10)
            
            logger.info(f"Tracking {len(self.channel_ids)} YouTube channels")
            logger.info(f"Configuration loaded: check interval {self.check_interval_hours}h, video age limit {self.video_age_limit_days} days")
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def setup_youtube_api(self):
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
        try:
            # Ensure the db_path is correctly formatted
            if not self.db_path.startswith("sqlite:///"):
                logger.warning(f"Fixing database path format: {self.db_path}")
                if self.db_path.startswith("sqlite://"):
                    self.db_path = self.db_path.replace("sqlite://", "sqlite:///")
                else:
                    self.db_path = f"sqlite:///{self.db_path}"
            
            # Create database engine and tables
            self.engine = create_engine(self.db_path)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            logger.info(f"Database connection established at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to set up database: {str(e)}")
            raise
    
    def get_channel_uploads(self, channel_id):
        """
        Get the uploads playlist ID for a YouTube channel
        """
        try:
            # Request channel details
            request = self.youtube.channels().list(
                part="contentDetails",
                id=channel_id
            )
            response = request.execute()
            
            # Check if the channel was found
            if not response.get("items"):
                logger.warning(f"No channel found for ID: {channel_id}")
                return None
            
            # Extract uploads playlist ID
            uploads_playlist = response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
            logger.info(f"Found uploads playlist {uploads_playlist} for channel {channel_id}")
            return uploads_playlist
        except Exception as e:
            logger.error(f"Error getting uploads for channel {channel_id}: {str(e)}")
            return None
    
    def get_recent_videos(self, playlist_id, max_results=10):
        """
        Get recent videos from a YouTube playlist
        """
        try:
            videos = []
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
            return []
    
    def get_youtube_transcript(self, video_id):
        """
        Get transcript for a YouTube video
        """
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = ' '.join([item['text'] for item in transcript_list])
            logger.info(f"Successfully retrieved transcript for video {video_id}: {len(transcript_text)} characters")
            return transcript_text
        except Exception as e:
            logger.warning(f"Could not retrieve transcript for video {video_id}: {str(e)}")
            return None
    
    def process_video(self, video_info):
        """
        Process a YouTube video - get transcript and analyze sentiment
        """
        session = self.Session()
        try:
            # Check if video already processed
            existing = session.query(SentimentRecord).filter_by(video_id=video_info["video_id"]).first()
            if existing:
                logger.info(f"Video {video_info['video_id']} already processed - skipping")
                session.close()
                return False
            
            # Get video transcript
            transcript = self.get_youtube_transcript(video_info["video_id"])
            
            if not transcript:
                logger.warning(f"No transcript available for {video_info['video_id']} - skipping")
                session.close()
                return False
            
            # Analyze sentiment
            source_id = f"youtube-{video_info['video_id']}"
            sentiment = self.sentiment_analyzer.analyze_text(transcript, source=source_id)
            
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
                combined_score=sentiment.get("combined_score", 0),
                text_length=sentiment.get("text_length", 0),
                source=sentiment.get("source", source_id),
                mentioned_cryptos=json.dumps(sentiment.get("mentioned_cryptos", []))
            )
            
            # Save to database
            session.add(record)
            session.commit()
            logger.info(f"Successfully processed video {video_info['video_id']} - Score: {sentiment.get('combined_score', 0):.2f}")
            
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
            logger.error(f"Error processing video {video_info.get('video_id', 'unknown')}: {str(e)}")
            return False
        finally:
            session.close()
    
    def run_tracker(self):
        """
        Main function to process YouTube channels and videos
        """
        logger.info("Starting YouTube tracker run")
        processed_count = 0
        
        for channel_id in self.channel_ids:
            try:
                channel_name = self.channel_names.get(channel_id, f"Channel {channel_id}")
                logger.info(f"Processing channel: {channel_name} (ID: {channel_id})")
                
                # Get uploads playlist
                uploads_playlist = self.get_channel_uploads(channel_id)
                
                if not uploads_playlist:
                    logger.warning(f"Could not get uploads playlist for channel {channel_name} - skipping")
                    continue
                
                # Get recent videos
                videos = self.get_recent_videos(uploads_playlist)
                logger.info(f"Found {len(videos)} recent videos from {channel_name}")
                
                # Process each video
                for video in videos:
                    if self.process_video(video):
                        processed_count += 1
                
                # Avoid hitting API rate limits
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error processing channel {channel_id}: {str(e)}")
        
        logger.info(f"Tracker run complete. Processed {processed_count} new videos")
        return processed_count
    
    def start_scheduler(self):
        """
        Start the scheduler to run at regular intervals
        """
        logger.info(f"Starting YouTube tracker scheduler with {self.check_interval_hours} hour interval")
        
        # Run immediately on startup
        self.run_tracker()
        
        # Schedule regular runs
        schedule.every(self.check_interval_hours).hours.do(self.run_tracker)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("YouTube tracker scheduler stopped by user")

# Main execution
if __name__ == "__main__":
    try:
        tracker = YouTubeTracker()
        tracker.start_scheduler()
    except Exception as e:
        logger.error(f"Failed to start YouTube tracker: {str(e)}")