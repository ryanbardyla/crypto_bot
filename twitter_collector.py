# twitter_collector.py
import os
import json
import time
import tweepy
import schedule
import threading
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sentiment_analyzer import SentimentAnalyzer
from database_manager import DatabaseManager
from utils.logging_config import get_module_logger

# Load environment variables
load_dotenv()
logger = get_module_logger("TwitterCollector")

# SQLAlchemy base
Base = declarative_base()

class TwitterSentiment(Base):
    __tablename__ = 'sentiment_twitter'
    
    id = Column(Integer, primary_key=True)
    tweet_id = Column(String(100), unique=True)
    username = Column(String(100))
    text = Column(Text)
    created_at = Column(DateTime)
    processed_date = Column(DateTime)
    followers_count = Column(Integer)
    retweet_count = Column(Integer)
    like_count = Column(Integer)
    vader_neg = Column(Float)
    vader_neu = Column(Float)
    vader_pos = Column(Float)
    vader_compound = Column(Float)
    bullish_keywords = Column(Integer)
    bearish_keywords = Column(Integer)
    keyword_sentiment = Column(Float)
    combined_score = Column(Float)
    symbol = Column(String(10))
    source = Column(String(200))
    mentioned_cryptos = Column(Text)  # Stored as JSON string
    
    def __repr__(self):
        return f"<TwitterSentiment(tweet_id='{self.tweet_id}', combined_score={self.combined_score})>"

class TwitterCollector:
    def __init__(self, config_file="twitter_collector_config.json"):
        self.load_config(config_file)
        self.setup_twitter_api()
        self.db_manager = DatabaseManager(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=os.environ.get("POSTGRES_PORT", "5432"),
            database=os.environ.get("POSTGRES_DB", "trading_db"),
            user=os.environ.get("POSTGRES_USER", "bot_user"),
            password=os.environ.get("POSTGRES_PASSWORD", "secure_password")
        )
        self.setup_database()
        self.sentiment_analyzer = SentimentAnalyzer()

    def setup_database(self):
        """Set up database connection and create tables if needed"""
        Base.metadata.create_all(self.db_manager.engine)
        self.Session = sessionmaker(bind=self.db_manager.engine)
        logger.info("Database setup complete")

    def load_config(self, config_file):
        """Load configuration from JSON file"""
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
            
            self.search_terms = self.config.get("search_terms", [])
            self.crypto_symbols = self.config.get("crypto_symbols", ["BTC", "ETH"])
            self.accounts_to_follow = self.config.get("accounts_to_follow", [])
            self.check_interval_minutes = self.config.get("check_interval_minutes", 30)
            self.max_tweets_per_search = self.config.get("max_tweets_per_search", 100)
            
            # Override with environment variables if available
            db_uri = os.environ.get("DB_URI")
            if db_uri:
                logger.info(f"Using database URI from environment: {db_uri}")
            
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise

    def setup_twitter_api(self):
        """Initialize Twitter API connection"""
        try:
            consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
            consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
            access_token = os.getenv("TWITTER_ACCESS_TOKEN")
            access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
            
            if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
                logger.warning("Twitter API credentials not fully configured in environment variables")
                self.api = None
                return
                
            auth = tweepy.OAuth1UserHandler(
                consumer_key, consumer_secret,
                access_token, access_token_secret
            )
            self.api = tweepy.API(auth)
            
            # Test the connection
            try:
                self.api.get_rate_limit_status()
                logger.info("Twitter API connection established")
            except Exception as e:
                logger.warning(f"Could not verify API status: {e}")
                logger.info("Twitter API connection initialized but not fully verified")
        except Exception as e:
            logger.error(f"Failed to initialize Twitter API: {str(e)}")
            self.api = None

    def _convert_v2_to_v1_format(self, tweet, client, user_info=None):
        """Convert Twitter API v2 response to v1-like format for compatibility"""
        class TweetObj:
            def __init__(self):
                self.id = None
                self.user = None
                
        tweet_obj = TweetObj()
        tweet_obj.id = getattr(tweet, 'id', '')
        tweet_obj.text = getattr(tweet, 'text', '')
        
        if isinstance(tweet.created_at, str):
            tweet_obj.created_at = datetime.fromisoformat(tweet.created_at.replace('Z', '+00:00'))
        else:
            tweet_obj.created_at = getattr(tweet, 'created_at', datetime.now())
            
        metrics = getattr(tweet, 'public_metrics', {}) or {}
        tweet_obj.retweet_count = metrics.get('retweet_count', 0)
        tweet_obj.favorite_count = metrics.get('like_count', 0)
        
        # User info
        user_obj = TweetObj()
        try:
            if user_info:
                user_obj.screen_name = getattr(user_info, 'username', '')
                user_metrics = getattr(user_info, 'public_metrics', {}) or {}
                user_obj.followers_count = user_metrics.get('followers_count', 0)
            else:
                if hasattr(tweet, 'author_id'):
                    user = client.get_user(id=tweet.author_id, user_fields=['public_metrics'])
                    if hasattr(user, 'data'):
                        user_obj.screen_name = getattr(user.data, 'username', '')
                        user_metrics = getattr(user.data, 'public_metrics', {}) or {}
                        user_obj.followers_count = user_metrics.get('followers_count', 0)
        except Exception as e:
            logger.warning(f"Couldn't get user data for tweet {tweet.id}: {e}")
            
        tweet_obj.user = user_obj
        return tweet_obj

    def collect_tweets_by_search(self, search_term, max_tweets=100):
        """Collect tweets matching a search term"""
        tweets = []
        
        if not self.api:
            logger.warning("Twitter API not initialized, skipping search")
            return tweets
            
        try:
            logger.info(f"Searching for tweets with term: {search_term}")
            
            # Try using v2 API first with tweepy client
            try:
                client = tweepy.Client(
                    bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
                    consumer_key=os.getenv("TWITTER_CONSUMER_KEY"),
                    consumer_secret=os.getenv("TWITTER_CONSUMER_SECRET"),
                    access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
                    access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
                )
                
                response = client.search_recent_tweets(
                    query=search_term,
                    max_results=min(max_tweets, 100),
                    tweet_fields=['created_at', 'public_metrics', 'author_id']
                )
                
                if hasattr(response, 'data') and response.data:
                    for tweet in response.data:
                        tweet_obj = self._convert_v2_to_v1_format(tweet, client)
                        tweets.append(tweet_obj)
            except Exception as e:
                logger.error(f"Error in v2 search API, falling back to v1: {str(e)}")
                
                # Fall back to v1 API if v2 fails
                try:
                    cursor = tweepy.Cursor(
                        self.api.search_tweets,
                        q=search_term,
                        tweet_mode='extended',
                        count=max_tweets
                    )
                    for tweet in cursor.items(max_tweets):
                        tweets.append(tweet)
                except Exception as e2:
                    logger.error(f"V1 fallback also failed: {str(e2)}")
            
            logger.info(f"Found {len(tweets)} tweets for search term: {search_term}")
            return tweets
            
        except Exception as e:
            logger.error(f"Error searching for tweets: {str(e)}")
            return []

    def collect_tweets_by_user(self, username, max_tweets=50):
        """Collect tweets from a specific user"""
        tweets = []
        
        if not self.api:
            logger.warning("Twitter API not initialized, skipping user timeline fetch")
            return tweets
            
        try:
            logger.info(f"Collecting tweets from user: {username}")
            
            # Try v2 API first
            try:
                client = tweepy.Client(
                    bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
                    consumer_key=os.getenv("TWITTER_CONSUMER_KEY"),
                    consumer_secret=os.getenv("TWITTER_CONSUMER_SECRET"),
                    access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
                    access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
                )
                
                user = client.get_user(username=username, user_fields=['public_metrics'])
                if not hasattr(user, 'data'):
                    logger.warning(f"User {username} not found")
                    return tweets
                    
                user_info = user.data
                
                response = client.get_users_tweets(
                    id=user_info.id,
                    max_results=min(max_tweets, 100),
                    tweet_fields=['created_at', 'public_metrics']
                )
                
                if hasattr(response, 'data') and response.data:
                    for tweet in response.data:
                        tweet_obj = self._convert_v2_to_v1_format(tweet, client, user_info)
                        tweets.append(tweet_obj)
            except Exception as e:
                logger.error(f"Error in v2 user timeline API, falling back to v1: {str(e)}")
                
                # Fall back to v1 API
                try:
                    for tweet in tweepy.Cursor(self.api.user_timeline, screen_name=username, tweet_mode='extended').items(max_tweets):
                        tweets.append(tweet)
                except Exception as e2:
                    logger.error(f"V1 fallback also failed: {str(e2)}")
            
            logger.info(f"Found {len(tweets)} tweets from user: {username}")
            return tweets
            
        except Exception as e:
            logger.error(f"Error collecting tweets from user: {str(e)}")
            return []

    def process_tweet(self, tweet, search_term=None):
        """Process a tweet to extract sentiment and save to database"""
        session = self.Session()
        try:
            # Check if tweet already exists in database
            existing = session.query(TwitterSentiment).filter_by(tweet_id=str(tweet.id)).first()
            if existing:
                logger.debug(f"Tweet {tweet.id} already processed")
                session.close()
                return False
                
            # Extract tweet text
            if hasattr(tweet, 'full_text'):
                text = tweet.full_text
            else:
                text = tweet.text
                
            # Check if tweet mentions any of our tracked cryptocurrencies
            tweet_symbols = []
            for symbol in self.crypto_symbols:
                if symbol.lower() in text.lower() or symbol.upper() in text.upper():
                    tweet_symbols.append(symbol)
            
            # If search term contains a symbol, use that as primary
            primary_symbol = None
            if search_term:
                for symbol in self.crypto_symbols:
                    if symbol.lower() in search_term.lower():
                        primary_symbol = symbol
                        break
            
            # Use first mentioned symbol or search term symbol
            crypto_symbol = primary_symbol or (tweet_symbols[0] if tweet_symbols else None)
            
            if not crypto_symbol and not tweet_symbols:
                logger.debug(f"Tweet {tweet.id} doesn't mention any tracked crypto")
                session.close()
                return False
                
            # Analyze sentiment
            sentiment = self.sentiment_analyzer.analyze_text(text, source=f"twitter-{tweet.id}")
            
            # Store in database
            record = TwitterSentiment(
                tweet_id=str(tweet.id),
                username=tweet.user.screen_name if hasattr(tweet.user, 'screen_name') else "unknown",
                text=text,
                created_at=tweet.created_at,
                processed_date=datetime.now(),
                followers_count=tweet.user.followers_count if hasattr(tweet.user, 'followers_count') else 0,
                retweet_count=tweet.retweet_count if hasattr(tweet, 'retweet_count') else 0,
                like_count=tweet.favorite_count if hasattr(tweet, 'favorite_count') else 0,
                vader_neg=sentiment.get("vader_sentiment", {}).get("neg", 0),
                vader_neu=sentiment.get("vader_sentiment", {}).get("neu", 0),
                vader_pos=sentiment.get("vader_sentiment", {}).get("pos", 0),
                vader_compound=sentiment.get("vader_sentiment", {}).get("compound", 0),
                bullish_keywords=sentiment.get("bullish_keywords", 0),
                bearish_keywords=sentiment.get("bearish_keywords", 0),
                keyword_sentiment=sentiment.get("keyword_sentiment", 0),
                combined_score=sentiment.get("combined_score", 0),
                symbol=crypto_symbol,
                source=f"twitter-{tweet.id}",
                mentioned_cryptos=json.dumps(tweet_symbols)
            )
            
            session.add(record)
            session.commit()
            logger.info(f"Successfully processed tweet {tweet.id} from @{tweet.user.screen_name}")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error processing tweet {tweet.id}: {str(e)}")
            return False
            
        finally:
            session.close()

    def run_collection(self):
        """Run a collection cycle for both search terms and user accounts"""
        logger.info("Starting Twitter collection run")
        processed_count = 0
        
        # Collect tweets by search terms
        for term in self.search_terms:
            try:
                tweets = self.collect_tweets_by_search(term, self.max_tweets_per_search)
                for tweet in tweets:
                    if self.process_tweet(tweet, search_term=term):
                        processed_count += 1
                
                # Sleep to respect rate limits
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error during search collection for term '{term}': {e}")
        
        # Collect tweets by user
        for username in self.accounts_to_follow:
            try:
                tweets = self.collect_tweets_by_user(username, max_tweets=50)
                for tweet in tweets:
                    if self.process_tweet(tweet):
                        processed_count += 1
                
                # Sleep to respect rate limits
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error during user collection for '{username}': {e}")
        
        logger.info(f"Collection run complete. Processed {processed_count} new tweets")
        return processed_count

    def start_scheduler(self):
        """Start the scheduler for periodic collection"""
        logger.info(f"Starting scheduler with {self.check_interval_minutes} minute interval")
        self.run_collection()
        schedule.every(self.check_interval_minutes).minutes.do(self.run_collection)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")

# Main execution
if __name__ == "__main__":
    collector = TwitterCollector()
    collector.start_scheduler()