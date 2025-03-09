import os
import json
import time
import tweepy
import schedule
import logging
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sentiment_analyzer import SentimentAnalyzer

# Import the centralized logging configuration
from utils.logging_config import get_module_logger

# Load environment variables
load_dotenv()

# Get logger for this module
logger = get_module_logger("TwitterCollector")

# Define database model
Base = declarative_base()

class TwitterSentiment(Base):
    __tablename__ = 'sentiment_twitter'
    
    id = Column(Integer, primary_key=True)
    tweet_id = Column(String, unique=True)
    username = Column(String)
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
    symbol = Column(String)
    source = Column(String)

class TwitterCollector:
    def __init__(self, config_file="twitter_collector_config.json"):
        self.load_config(config_file)
        self.setup_twitter_api()
        self.setup_database()
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def load_config(self, config_file):
        try:
            with open(config_file, "r") as f:
                self.config = json.load(f)
                
            self.search_terms = self.config.get("search_terms", [])
            self.crypto_symbols = self.config.get("crypto_symbols", ["BTC", "ETH"])
            self.accounts_to_follow = self.config.get("accounts_to_follow", [])
            self.check_interval_minutes = self.config.get("check_interval_minutes", 30)
            self.max_tweets_per_search = self.config.get("max_tweets_per_search", 100)
            self.db_path = self.config.get("db_path", "sqlite:///sentiment_database.db")
            
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def setup_twitter_api(self):
        try:
            # Get credentials from environment variables
            consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
            consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
            access_token = os.getenv("TWITTER_ACCESS_TOKEN")
            access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
            
            # Check if all required credentials are available
            if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
                logger.warning("Twitter API credentials not fully configured in environment variables")
                self.api = None
                return
                
            auth = tweepy.OAuth1UserHandler(
                consumer_key, consumer_secret,
                access_token, access_token_secret
            )
            self.api = tweepy.API(auth)
            
            try:
                self.api.get_rate_limit_status()
                logger.info("Twitter API connection established")
            except Exception as e:
                logger.warning(f"Could not verify API status: {e}")
                logger.info("Twitter API connection initialized but not fully verified")
        except Exception as e:
            logger.error(f"Failed to initialize Twitter API: {str(e)}")
            self.api = None
    
    def setup_database(self):
        try:
            self.engine = create_engine(self.db_path)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to set up database: {str(e)}")
            raise
    
    def _convert_v2_to_v1_format(self, tweet, client, user_info=None):
        """Convert Twitter API v2 response to a format compatible with v1"""
        class TweetObj:
            pass
            
        tweet_obj = TweetObj()
        tweet_obj.id = tweet.id
        tweet_obj.text = getattr(tweet, 'text', '')
        
        if isinstance(tweet.created_at, str):
            tweet_obj.created_at = datetime.fromisoformat(tweet.created_at.replace('Z', '+00:00'))
        else:
            tweet_obj.created_at = tweet.created_at
            
        metrics = getattr(tweet, 'public_metrics', {}) or {}
        tweet_obj.retweet_count = metrics.get('retweet_count', 0)
        tweet_obj.favorite_count = metrics.get('like_count', 0)
        
        user_obj = TweetObj()
        user_obj.screen_name = ''
        user_obj.followers_count = 0
        
        try:
            if user_info:
                user_obj.screen_name = getattr(user_info, 'username', '')
                user_metrics = getattr(user_info, 'public_metrics', {}) or {}
                user_obj.followers_count = user_metrics.get('followers_count', 0)
            else:
                if hasattr(tweet, 'author_id'):
                    user = client.get_user(id=tweet.author_id, user_fields=['public_metrics'])
                    if user and user.data:
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
            
            # Try using Twitter API v2 first
            client = tweepy.Client(
                bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
                consumer_key=os.getenv("TWITTER_CONSUMER_KEY"),
                consumer_secret=os.getenv("TWITTER_CONSUMER_SECRET"),
                access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
                access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
            )
            
            try:
                response = client.search_recent_tweets(
                    query=search_term,
                    max_results=min(max_tweets, 100),  # API limit is 100 per request
                    tweet_fields=['created_at', 'public_metrics']
                )
                
                if response.data:
                    for tweet in response.data:
                        tweet_obj = self._convert_v2_to_v1_format(tweet, client)
                        tweets.append(tweet_obj)
            except Exception as e:
                logger.error(f"Error in v2 search API, falling back to v1: {str(e)}")
                
                try:
                    cursor = tweepy.Cursor(
                        self.api.search_tweets,
                        q=search_term,
                        tweet_mode='extended',
                        count=100
                    )
                    for tweet in cursor.items(max_tweets):
                        tweets.append(tweet)
                except Exception as e2:
                    logger.error(f"V1 fallback also failed: {str(e2)}")
                
            logger.info(f"Found {len(tweets)} tweets for search term: {search_term}")
            return tweets
            
        except Exception as e:
            logger.error(f"Error searching for tweets: {str(e)}")
            return tweets
    
    def collect_tweets_by_user(self, username, max_tweets=50):
        """Collect tweets from a specific user"""
        tweets = []
        
        if not self.api:
            logger.warning("Twitter API not initialized, skipping user timeline fetch")
            return tweets
            
        try:
            logger.info(f"Collecting tweets from user: {username}")
            
            # Try using Twitter API v2
            client = tweepy.Client(
                bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
                consumer_key=os.getenv("TWITTER_CONSUMER_KEY"),
                consumer_secret=os.getenv("TWITTER_CONSUMER_SECRET"),
                access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
                access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
            )
            
            try:
                user = client.get_user(username=username, user_fields=['public_metrics'])
                if not user or not user.data:
                    logger.warning(f"User {username} not found")
                    return tweets
                    
                user_info = user.data
                
                response = client.get_users_tweets(
                    id=user_info.id,
                    max_results=min(max_tweets, 100),  # API limit
                    tweet_fields=['created_at', 'public_metrics']
                )
                
                if response.data:
                    for tweet in response.data:
                        tweet_obj = self._convert_v2_to_v1_format(tweet, client, user_info)
                        tweets.append(tweet_obj)
            except Exception as e:
                logger.error(f"Error in v2 user timeline API, falling back to v1: {str(e)}")
                
                try:
                    for tweet in tweepy.Cursor(self.api.user_timeline, screen_name=username, tweet_mode='extended').items(max_tweets):
                        tweets.append(tweet)
                except Exception as e2:
                    logger.error(f"V1 fallback also failed: {str(e2)}")
            
            logger.info(f"Found {len(tweets)} tweets from user: {username}")
            return tweets
            
        except Exception as e:
            logger.error(f"Error collecting tweets from user: {str(e)}")
            return tweets
    
    def process_tweet(self, tweet, search_term=None):
        """Process a tweet and store sentiment analysis results"""
        session = self.Session()
        processed = False
        
        try:
            # Check if tweet already exists in database
            existing = session.query(TwitterSentiment).filter_by(tweet_id=str(tweet.id)).first()
            if existing:
                logger.debug(f"Tweet {tweet.id} already processed")
                return False
                
            # Extract tweet text based on API version
            if hasattr(tweet, 'full_text'):
                text = tweet.full_text
            else:
                text = tweet.text
                
            # Determine which crypto symbols are mentioned
            tweet_symbols = []
            for symbol in self.crypto_symbols:
                if symbol.lower() in text.lower() or symbol.upper() in text.upper():
                    tweet_symbols.append(symbol)
                    
            if not tweet_symbols and search_term:
                # Assign based on search term if no symbol found in text
                for symbol in self.crypto_symbols:
                    if symbol.lower() in search_term.lower():
                        tweet_symbols.append(symbol)
                        
            if not tweet_symbols:
                logger.debug(f"Tweet {tweet.id} doesn't mention any tracked crypto")
                return False
                
            # Run sentiment analysis
            sentiment = self.sentiment_analyzer.analyze_text(text, source=f"twitter-{tweet.id}")
            
            # Store for each mentioned symbol
            for symbol in tweet_symbols:
                record = TwitterSentiment(
                    tweet_id=str(tweet.id),
                    username=tweet.user.screen_name,
                    text=text,
                    created_at=tweet.created_at,
                    processed_date=datetime.now(),
                    followers_count=tweet.user.followers_count,
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
                    symbol=symbol,
                    source=f"twitter-{tweet.id}"
                )
                session.add(record)
                processed = True
                
            session.commit()
            logger.info(f"Successfully processed tweet {tweet.id} from @{tweet.user.screen_name}")
            return processed
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error processing tweet {tweet.id}: {str(e)}")
            return False
        finally:
            session.close()
    
    def run_collection(self):
        """Run a full collection cycle for searches and user timelines"""
        logger.info("Starting Twitter collection run")
        processed_count = 0
        
        # Process search terms
        for term in self.search_terms:
            try:
                tweets = self.collect_tweets_by_search(term, self.max_tweets_per_search)
                for tweet in tweets:
                    if self.process_tweet(tweet, search_term=term):
                        processed_count += 1
                        
                # Be nice to API rate limits
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error during search collection for term '{term}': {e}")
        
        # Process user timelines
        for username in self.accounts_to_follow:
            try:
                tweets = self.collect_tweets_by_user(username, max_tweets=50)
                for tweet in tweets:
                    if self.process_tweet(tweet):
                        processed_count += 1
            except Exception as e:
                logger.error(f"Error during user collection for '{username}': {e}")
                
        logger.info(f"Collection run complete. Processed {processed_count} new tweets")
    
    def start_scheduler(self):
        """Start the scheduled collection process"""
        logger.info(f"Starting scheduler with {self.check_interval_minutes} minute interval")
        self.run_collection()
        schedule.every(self.check_interval_minutes).minutes.do(self.run_collection)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")

# Import this after class definition to avoid circular imports
from sentiment_analyzer import SentimentAnalyzer

if __name__ == "__main__":
    collector = TwitterCollector()
    collector.start_scheduler()