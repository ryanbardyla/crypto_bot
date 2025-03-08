import os
import json
import logging
import schedule
import time
import googleapiclient.discovery
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
        handlers=[
            logging.FileHandler("youtube_tracker.log"),
            logging.StreamHandler()
        ]
    )
logger = logging.getLogger("YouTubeTracker")
Base = declarative_base()

class SentimentRecord(Base):
    __tablename__ = "sentiment_youtube"
    
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

class YouTubeTracker:
    def __init__(self, config_file="youtube_tracker_config.json"):
        self.load_config(config_file)
        self.setup_youtube_api()
        self.setup_database()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def load_config(self, config_file):
        try:
            with open(config_file, "r") as f:
                self.config = json.load(f)
            self.channel_ids = self.config.get("channel_ids", [])
            self.check_interval_hours = self.config.get("check_interval_hours", 6)
            self.video_age_limit_days = self.config.get("video_age_limit_days", 7)
            self.db_path = self.config.get("db_path", "sqlite:///sentiment_database.db")
            logger.info(f"Tracking {len(self.channel_ids)} YouTube channels")
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def setup_youtube_api(self):
        try:
            # Get API key from environment variable
            api_key = os.getenv("YOUTUBE_API_KEY")
            if not api_key:
                logger.error("YouTube API key not found in environment variables")
                raise ValueError("YouTube API key not found")
                
            self.youtube = googleapiclient.discovery.build(
                "youtube", "v3", developerKey=api_key
            )
            logger.info("YouTube API client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize YouTube API: {str(e)}")
            raise
    
    def setup_database(self):
        try:
            self.engine = create_engine(self.db_path)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to set up database: {str(e)}")
            raise
    
    def get_channel_uploads(self, channel_id):
        try:
            request = self.youtube.channels().list(
                part="contentDetails",
                id=channel_id
            )
            response = request.execute()
            
            if not response.get("items"):
                logger.warning(f"No channel found for ID: {channel_id}")
                return None
                
            uploads_playlist = response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
            return uploads_playlist
        except Exception as e:
            logger.error(f"Error getting uploads for channel {channel_id}: {str(e)}")
            return None
    
    def get_recent_videos(self, playlist_id, max_results=10):
        try:
            videos = []
            request = self.youtube.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=max_results
            )
            response = request.execute()
            
            cutoff_date = datetime.now() - timedelta(days=self.video_age_limit_days)
            
            for item in response.get("items", []):
                snippet = item["snippet"]
                dt = datetime.fromisoformat(snippet["publishedAt"].replace("Z", "+00:00"))
                publish_date = dt.replace(tzinfo=None)  # Remove timezone info
                
                if publish_date >= cutoff_date:
                    videos.append({
                        "video_id": snippet["resourceId"]["videoId"],
                        "title": snippet["title"],
                        "channel_id": snippet["channelId"],
                        "publish_date": publish_date
                    })
            
            return videos
        except Exception as e:
            logger.error(f"Error getting videos from playlist {playlist_id}: {str(e)}")
            return []
    
    def process_video(self, video_info):
        session = self.Session()
        try:
            existing = session.query(SentimentRecord).filter_by(video_id=video_info["video_id"]).first()
            if existing:
                logger.info(f"Video {video_info['video_id']} already processed")
                session.close()
                return False
                
            video_url = f"https://www.youtube.com/watch?v={video_info['video_id']}"
            transcript = self.sentiment_analyzer.get_youtube_transcript(video_url)
            
            if not transcript:
                logger.warning(f"Could not retrieve transcript for {video_info['video_id']}")
                session.close()
                return False
                
            sentiment = self.sentiment_analyzer.analyze_text(transcript, source=f"youtube-{video_info['video_id']}")
            
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
                source=sentiment.get("source", "youtube")
            )
            
            session.add(record)
            session.commit()
            logger.info(f"Successfully processed video {video_info['video_id']} - Score: {sentiment['combined_score']:.2f}")
            
            # Save sentiment data to file for easier access
            os.makedirs("sentiment_data", exist_ok=True)
            with open(f"sentiment_data/{video_info['video_id']}.json", "w") as f:
                sentiment["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sentiment["video_id"] = video_info['video_id']
                json.dump(sentiment, f, indent=2)
                
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error processing video {video_info['video_id']}: {str(e)}")
            return False
        finally:
            session.close()
    
    def run_tracker(self):
        logger.info("Starting tracker run")
        processed_count = 0
        
        for channel_id in self.channel_ids:
            try:
                logger.info(f"Processing channel: {channel_id}")
                uploads_playlist = self.get_channel_uploads(channel_id)
                
                if not uploads_playlist:
                    logger.warning(f"Could not get uploads playlist for channel {channel_id}")
                    continue
                    
                videos = self.get_recent_videos(uploads_playlist)
                logger.info(f"Found {len(videos)} recent videos for channel {channel_id}")
                
                for video in videos:
                    if self.process_video(video):
                        processed_count += 1
                        
                # Avoid hitting API rate limits
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error processing channel {channel_id}: {str(e)}")
                
        logger.info(f"Tracker run complete. Processed {processed_count} new videos")
    
    def start_scheduler(self):
        logger.info(f"Starting scheduler with {self.check_interval_hours} hour interval")
        self.run_tracker()
        schedule.every(self.check_interval_hours).hours.do(self.run_tracker)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")

# Import SentimentAnalyzer here to avoid circular imports
from sentiment_analyzer import SentimentAnalyzer

if __name__ == "__main__":
    tracker = YouTubeTracker()
    tracker.start_scheduler()