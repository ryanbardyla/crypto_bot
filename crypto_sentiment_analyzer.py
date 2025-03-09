import nltk
import re
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime

# Download VADER lexicon if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CryptoSentimentAnalyzer")

class CryptoSentimentAnalyzer:
    def __init__(self):
        # Initialize the VADER sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        # Extend the VADER lexicon with crypto-specific terms
        self.extend_vader_lexicon()
        
        # Define crypto-specific keywords
        self.bullish_keywords = [
            "bullish", "moon", "mooning", "to the moon", "ath", "all-time high", "all time high",
            "hodl", "buy the dip", "buying the dip", "accumulate", "accumulating", 
            "long", "going long", "rally", "breakout", "breaking out", "uptrend", "support",
            "undervalued", "growth", "opportunity", "buy", "buying", "bought",
            "potential", "gain", "gains", "profit", "profitable", "roi", "return on investment",
            "staking", "staked", "yield", "earning", "earnings", "passive income",
            "adoption", "mainstream", "institutional", "halving", "supply shock"
        ]
        
        self.bearish_keywords = [
            "bearish", "crash", "crashing", "dip", "dipping", "correction", "correcting", 
            "dump", "dumping", "sell", "selling", "sold", "short", "shorting", "shorted",
            "downtrend", "resistance", "overhead resistance", "rejection", "rejected",
            "overvalued", "bubble", "hype", "fud", "fear", "uncertainty", "doubt",
            "scam", "ponzi", "rugpull", "rug pull", "manipulation", "manipulated",
            "regulation", "regulated", "sec", "ban", "banned", "illegal", "hack", "hacked",
            "risk", "risky", "loss", "losses", "liquidation", "liquidated", "margin call"
        ]
        
        # Regex for finding YouTube video IDs
        self.youtube_regex = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
        
        # Crypto symbols for entity recognition
        self.crypto_symbols = [
            "BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "DOGE", "AVAX", "MATIC", "LINK",
            "UNI", "AAVE", "SNX", "MKR", "CRV", "YFI", "COMP", "SUSHI", "GRT", "FTT",
            "Bitcoin", "Ethereum", "Solana", "Cardano", "Ripple", "Polkadot", "Dogecoin"
        ]
        
        logger.info("CryptoSentimentAnalyzer initialized")
    
    def extend_vader_lexicon(self):
        """Extend VADER lexicon with crypto-specific sentiment values"""
        # Positive/bullish terms
        self.sia.lexicon.update({
            'bullish': 3.0,
            'moon': 3.0,
            'mooning': 3.0,
            'hodl': 2.0,
            'staking': 1.5,
            'adoption': 2.0,
            'institutional': 1.5,
            'support': 1.0,
            'breakout': 2.5,
            'accumulate': 1.5,
            'undervalued': 2.0,
            'halving': 1.5,
            'ath': 2.5,
            'dip': -0.5,  # Reduced negative value since "buy the dip" is positive
        })
        
        # Negative/bearish terms
        self.sia.lexicon.update({
            'bearish': -3.0,
            'crash': -3.0,
            'dump': -2.5,
            'rugpull': -4.0,
            'scam': -3.5,
            'ponzi': -3.5,
            'fud': -2.0,
            'resistance': -1.0,
            'correction': -1.5,
            'bubble': -2.0,
            'overvalued': -2.0,
            'liquidation': -3.0,
            'hack': -3.0,
            'banned': -2.5,
            'sec': -1.0,  # SEC mentions often have negative context
        })
    
    def extract_youtube_id(self, url):
        """Extract YouTube ID from a URL"""
        match = re.search(self.youtube_regex, url)
        if match:
            return match.group(1)
        return None
    
    def get_youtube_transcript(self, url):
        """Get transcript for a YouTube video"""
        from youtube_transcript_api import YouTubeTranscriptApi
        
        video_id = self.extract_youtube_id(url)
        if not video_id:
            logger.error(f"Could not extract YouTube ID from {url}")
            return None
        
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = ' '.join([item['text'] for item in transcript_list])
            logger.info(f"Retrieved transcript for YouTube video {video_id}: {len(transcript_text)} characters")
            return transcript_text
        except Exception as e:
            logger.warning(f"Transcripts are disabled for video {video_id}")
            logger.error(f"Error retrieving YouTube transcript: {e}")
            return None
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment in text using VADER and custom crypto patterns
        """
        # VADER sentiment analysis
        sentiment = self.sia.polarity_scores(text)
        
        # Count bullish/bearish keywords
        text_lower = text.lower()
        bullish_count = sum(1 for keyword in self.bullish_keywords if keyword.lower() in text_lower)
        bearish_count = sum(1 for keyword in self.bearish_keywords if keyword.lower() in text_lower)
        
        # Calculate keyword sentiment (range -10 to +10)
        total_keywords = bullish_count + bearish_count
        if total_keywords > 0:
            keyword_sentiment = ((bullish_count - bearish_count) / total_keywords) * 10
        else:
            keyword_sentiment = 0
        
        # Calculate combined score
        # Weight: 40% VADER, 60% keyword-based for crypto-specific context
        combined_score = (0.4 * sentiment['compound'] * 10) + (0.6 * keyword_sentiment)
        
        # Extract mentioned cryptocurrencies
        mentioned_cryptos = []
        for symbol in self.crypto_symbols:
            if re.search(r'\b' + re.escape(symbol) + r'\b', text, re.IGNORECASE):
                mentioned_cryptos.append(symbol)
        
        # Analyze price predictions
        price_predictions = self.extract_price_predictions(text)
        
        # Return complete analysis
        return {
            'vader_sentiment': sentiment,
            'bullish_keywords': bullish_count,
            'bearish_keywords': bearish_count,
            'keyword_sentiment': keyword_sentiment,
            'combined_score': combined_score,
            'mentioned_cryptos': mentioned_cryptos,
            'price_predictions': price_predictions,
            'text_length': len(text)
        }
    
    def extract_price_predictions(self, text):
        """
        Extract price predictions from text
        Returns a list of (crypto, price, timeframe) tuples
        """
        predictions = []
        
        # Look for patterns like "$50,000" or "50k" or "50,000 dollars" near crypto symbols
        price_pattern = r'(?:\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:k|K|thousand|million|m|M|billion|B)?)|\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:k|K|thousand|million|m|M|billion|B)?)\s*(?:dollars|USD|BTC|ETH|SOL))'
        timeframe_pattern = r'\b(?:by|in|next|coming|this)\s+(\w+\s+\w+|\w+)\b'
        
        for symbol in self.crypto_symbols:
            # Look for the symbol in the text
            symbol_matches = re.finditer(r'\b' + re.escape(symbol) + r'\b', text, re.IGNORECASE)
            
            for match in symbol_matches:
                # Get the context around this symbol mention (50 chars before and after)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                # Find price mentions in this context
                price_matches = re.finditer(price_pattern, context)
                for price_match in price_matches:
                    price = price_match.group(1) or price_match.group(2)
                    
                    # Look for timeframe mentions
                    timeframe = None
                    timeframe_matches = re.search(timeframe_pattern, context)
                    if timeframe_matches:
                        timeframe = timeframe_matches.group(1)
                    
                    predictions.append({
                        'crypto': symbol,
                        'price': price,
                        'timeframe': timeframe
                    })
        
        return predictions
    
    def analyze_youtube_video(self, url):
        """
        Analyze sentiment in a YouTube video's transcript
        """
        transcript = self.get_youtube_transcript(url)
        if not transcript:
            return None
        
        sentiment = self.analyze_sentiment(transcript)
        sentiment['video_id'] = self.extract_youtube_id(url)
        sentiment['source'] = f"youtube-{sentiment['video_id']}"
        
        logger.info(f"Sentiment analysis for {url}: {sentiment['combined_score']:.2f}")
        return sentiment
    
    def analyze_text(self, text, source=None):
        """
        Analyze sentiment in raw text
        """
        sentiment = self.analyze_sentiment(text)
        
        if source:
            sentiment['source'] = source
        
        logger.info(f"Sentiment analysis: {sentiment['combined_score']:.2f}")
        return sentiment

# For testing
if __name__ == "__main__":
    analyzer = CryptoSentimentAnalyzer()
    
    # Test with a sample text
    test_text = """
    Bitcoin is looking extremely bullish right now. I think we're going to see BTC hit $100k by the end of this year.
    The on-chain metrics are strong, and institutional adoption is increasing. ETH could reach $10k soon too!
    However, there are some bearish signals in the altcoin market. Be careful with smaller caps as they might dump.
    Overall, I'm accumulating BTC and ETH during any dips, but staying cautious with everything else.
    HODL strong, and don't get liquidated!
    """
    
    result = analyzer.analyze_text(test_text)
    
    print("\nCrypto Sentiment Analysis:")
    print(f"VADER: {result['vader_sentiment']}")
    print(f"Bullish keywords: {result['bullish_keywords']}")
    print(f"Bearish keywords: {result['bearish_keywords']}")
    print(f"Keyword sentiment: {result['keyword_sentiment']:.2f}")
    print(f"Combined score: {result['combined_score']:.2f}")
    print(f"Mentioned cryptocurrencies: {result['mentioned_cryptos']}")
    print(f"Price predictions: {result['price_predictions']}")
    
    # Test with a YouTube URL if provided
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
            print(f"Mentioned cryptocurrencies: {video_result['mentioned_cryptos']}")
            print(f"Price predictions: {video_result['price_predictions']}")
        else:
            print("Could not analyze video. Check if it has available transcripts.")