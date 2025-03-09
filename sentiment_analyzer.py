import re
import nltk
import logging
import sys
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# Import the centralized logging configuration
from utils.logging_config import get_module_logger

# Get logger for this module
logger = get_module_logger(__name__)

# Download VADER lexicon if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.youtube_regex = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
        
        # Lists of keywords for sentiment analysis
        self.bullish_keywords = [
            "bullish", "buy", "long", "uptrend", "growth", "positive", "rally", "surge",
            "climb", "breakout", "support", "oversold", "accumulate", "hodl", "moon",
            "progress", "gain", "potential", "opportunity", "undervalued"
        ]
        
        self.bearish_keywords = [
            "bearish", "sell", "short", "downtrend", "decline", "negative", "crash", "drop",
            "fall", "breakdown", "resistance", "overbought", "distribute", "dump", "tank",
            "risk", "loss", "danger", "concern", "overvalued"
        ]
        
        logger.info("SentimentAnalyzer initialized")
        
    def _extract_youtube_id(self, url):
        """Extract the YouTube video ID from a URL"""
        match = re.search(self.youtube_regex, url)
        if match:
            return match.group(1)
        return None


    def get_youtube_transcript(self, url):
        """Get transcript from YouTube video."""
        video_id = self._extract_youtube_id(url)
        if not video_id:
            logger.error(f"Could not extract YouTube ID from {url}")
            return None
            
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = ' '.join([item['text'] for item in transcript_list])
            logger.info(f"Retrieved transcript for YouTube video {video_id}: {len(transcript_text)} characters")
            return transcript_text
        except TranscriptsDisabled:
            logger.warning(f"Transcripts are disabled for video {video_id}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving YouTube transcript: {e}")
            return None

    def analyze_sentiment(self, text):
        """Analyze sentiment of text using VADER and keyword analysis."""
        # VADER sentiment analysis
        sentiment = self.sia.polarity_scores(text)
        
        # Keyword-based sentiment analysis
        bullish_count = sum(1 for keyword in self.bullish_keywords if keyword.lower() in text.lower())
        bearish_count = sum(1 for keyword in self.bearish_keywords if keyword.lower() in text.lower())
        
        # Calculate keyword sentiment score (scale -10 to 10)
        keyword_difference = bullish_count - bearish_count
        keyword_total = bullish_count + bearish_count
        keyword_sentiment = 0
        if keyword_total > 0:
            keyword_sentiment = (keyword_difference / keyword_total) * 10
            
        # Combine VADER and keyword sentiment
        # VADER compound score ranges from -1 to 1, multiply by 5 for -5 to 5 scale
        vader_scaled = sentiment['compound'] * 5
        
        # Combined score - weighted average (50% VADER, 50% keyword)
        combined_score = (vader_scaled + keyword_sentiment) / 2
        
        # Ensure combined score stays within -10 to 10 range
        combined_score = max(-10, min(10, combined_score))
        
        return {
            "vader_sentiment": sentiment,
            "bullish_keywords": bullish_count,
            "bearish_keywords": bearish_count,
            "keyword_sentiment": keyword_sentiment,
            "combined_score": combined_score
        }

    def analyze_youtube_video(self, url):
        """Analyze sentiment of a YouTube video transcript."""
        transcript = self.get_youtube_transcript(url)
        if not transcript:
            return None
            
        sentiment = self.analyze_sentiment(transcript)
        sentiment['text_length'] = len(transcript)
        sentiment['transcript'] = transcript  # Include full transcript
        sentiment['video_id'] = self._extract_youtube_id(url)
        logger.info(f"Sentiment analysis for {url}: {sentiment['combined_score']:.2f}")
        return sentiment

    def analyze_text(self, text, source=None):
        """Analyze sentiment of any text."""
        if not text:
            return {"combined_score": 0, "vader_sentiment": {"compound": 0, "neg": 0, "neu": 1, "pos": 0}}
            
        sentiment = self.analyze_sentiment(text)
        sentiment['text_length'] = len(text)
        if source:
            sentiment['source'] = source
            
        logger.info(f"Sentiment analysis: {sentiment['combined_score']:.2f}")
        return sentiment

# If run directly, perform a simple test
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Test with a sample text
    test_text = """
    Bitcoin is looking incredibly bullish right now. The technical indicators are showing 
    strong support levels and I think we're going to see a significant rally in the coming weeks.
    Smart money is accumulating while weak hands are selling. This is a great opportunity to buy the dip
    and hold for the long term. I'm very optimistic about the future price action.
    """
    
    result = analyzer.analyze_text(test_text)
    print("\nText sentiment analysis:")
    print(f"VADER: {result['vader_sentiment']}")
    print(f"Bullish keywords: {result['bullish_keywords']}")
    print(f"Bearish keywords: {result['bearish_keywords']}")
    print(f"Keyword sentiment: {result['keyword_sentiment']:.2f}")
    print(f"Combined score: {result['combined_score']:.2f}")
    
    # Test with a YouTube URL (if provided as argument)
    import sys
    if len(sys.argv) > 1:
        youtube_url = sys.argv[1]
        print(f"\nAnalyzing YouTube video: {youtube_url}")
        video_result = analyzer.analyze_youtube_video(youtube_url)
        if video_result:
            print(f"VADER: {video_result['vader_sentiment']}")
            print(f"Bullish keywords: {video_result['bullish_keywords']}")
            print(f"Bearish keywords: {video_result['bearish_keywords']}")
            print(f"Keyword sentiment: {video_result['keyword_sentiment']:.2f}")
            print(f"Combined score: {video_result['combined_score']:.2f}")
            print(f"Text length: {video_result['text_length']}")
        else:
            print("Could not analyze video. Check if it has available transcripts.")