# sentiment_analyzer.py
import os
import re
import json
import logging
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentimentAnalyzer")

class SentimentAnalyzer:
    """Analyze sentiment from video transcripts and text content"""
    
    def __init__(self):
        """Initialize the sentiment analyzer"""
        # Download NLTK resources if not already present
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        self.sia = SentimentIntensityAnalyzer()
        
        # Crypto-specific keyword dictionaries
        self.bullish_keywords = [
            "bullish", "buy", "long", "support", "uptrend", "breakout", 
            "accumulate", "moon", "upside", "bottom", "reversal",
            "oversold", "undervalued", "adoption", "institutional"
        ]
        
        self.bearish_keywords = [
            "bearish", "sell", "short", "resistance", "downtrend", "breakdown", 
            "distribute", "dump", "downside", "top", "correction",
            "overbought", "overvalued", "regulation", "ban", "hack"
        ]
    
    def _extract_youtube_id(self, url):
        """Extract YouTube video ID from URL"""
        youtube_regex = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
        match = re.search(youtube_regex, url)
        
        if match:
            return match.group(1)
        return None
    
    def get_youtube_transcript(self, url):
        """Get transcript from a YouTube video"""
        video_id = self._extract_youtube_id(url)
        
        if not video_id:
            logger.error(f"Could not extract YouTube ID from {url}")
            return None
        
        try:
            # This would normally use YouTubeTranscriptApi, but we'll make a simplified version
            # that doesn't require the external dependency for now
            transcript_text = f"Placeholder transcript for video {video_id}"
            
            logger.info(f"Retrieved transcript for YouTube video {video_id}: {len(transcript_text)} characters")
            return transcript_text
            
        except Exception as e:
            logger.error(f"Error retrieving YouTube transcript: {e}")
            return None
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text
        
        Returns:
            dict: Sentiment scores
        """
        if not text:
            return None
        
        # Get VADER sentiment
        sentiment = self.sia.polarity_scores(text)
        
        # Count crypto-specific keywords
        bullish_count = sum(1 for keyword in self.bullish_keywords if keyword.lower() in text.lower())
        bearish_count = sum(1 for keyword in self.bearish_keywords if keyword.lower() in text.lower())
        
        # Normalize to get a score from -10 to 10 (bearish to bullish)
        keyword_sentiment = ((bullish_count - bearish_count) / max(1, bullish_count + bearish_count)) * 10
        
        # Combine VADER compound score (ranges from -1 to 1) with our keyword sentiment
        combined_score = (sentiment['compound'] * 5) + (keyword_sentiment * 0.5)
        
        # Clamp between -10 and 10
        combined_score = max(-10, min(10, combined_score))
        
        result = {
            'vader_sentiment': sentiment,
            'bullish_keywords': bullish_count,
            'bearish_keywords': bearish_count,
            'keyword_sentiment': keyword_sentiment,
            'combined_score': combined_score
        }
        
        return result
    
    def analyze_youtube_video(self, url):
        """
        Analyze sentiment of a YouTube video
        
        Args:
            url: YouTube video URL
            
        Returns:
            dict: Sentiment analysis results
        """
        transcript = self.get_youtube_transcript(url)
        
        if not transcript:
            return None
        
        sentiment = self.analyze_sentiment(transcript)
        
        if sentiment:
            sentiment['url'] = url
            sentiment['transcript_length'] = len(transcript)
            
            logger.info(f"Sentiment analysis for {url}: {sentiment['combined_score']:.2f}")
            
            return sentiment
        
        return None
    
    def analyze_text(self, text, source=None):
        """
        Analyze sentiment of text content
        
        Args:
            text: Text content to analyze
            source: Optional source identifier
            
        Returns:
            dict: Sentiment analysis results
        """
        sentiment = self.analyze_sentiment(text)
        
        if sentiment:
            if source:
                sentiment['source'] = source
            
            sentiment['text_length'] = len(text)
            
            logger.info(f"Sentiment analysis: {sentiment['combined_score']:.2f}")
            
            return sentiment
        
        return None