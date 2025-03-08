# enhanced_strategy.py
import os
import json
import logging
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedStrategy")

class EnhancedStrategy:
    def __init__(self, config_file="enhanced_strategy_config.json"):
        self.load_config(config_file)
        self.setup_database()
        
    def load_config(self, config_file):
        try:
            with open(config_file, "r") as f:
                self.config = json.load(f)
                
            # Channel credibility weights
            self.channel_weights = self.config.get("channel_weights", {})
            
            # Default weight if channel not in list
            self.default_weight = self.config.get("default_weight", 1.0)
            
            # Market condition thresholds
            self.volatility_threshold = self.config.get("volatility_threshold", 2.0)
            self.trend_threshold = self.config.get("trend_threshold", 1.5)
            
            # Sentiment thresholds
            self.bullish_threshold = self.config.get("bullish_threshold", 3.0)
            self.bearish_threshold = self.config.get("bearish_threshold", -3.0)
            
            # Adaptive parameter ranges
            self.position_size_range = self.config.get("position_size_range", [0.5, 1.0])
            self.stop_loss_range = self.config.get("stop_loss_range", [2.0, 5.0]) 
            self.take_profit_range = self.config.get("take_profit_range", [5.0, 15.0])
            
            # Database path
            self.db_path = self.config.get("db_path", "sqlite:///sentiment_database.db")
            
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
            
    def setup_database(self):
        try:
            self.engine = create_engine(self.db_path)
            self.Session = sessionmaker(bind=self.engine)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to set up database: {str(e)}")
            raise
            
    def get_market_condition(self, price_data, lookback_days=14):
        """
        Determine market condition based on recent price action
        Returns: Dict with market state metrics
        """
        if price_data is None or len(price_data) < lookback_days:
            logger.warning("Not enough price data to determine market condition")
            return {"volatility": 1.0, "trend": 0.0, "condition": "neutral"}
            
        # Get recent price data
        recent_data = price_data.iloc[-lookback_days:]
        
        # Calculate volatility (standard deviation of returns)
        returns = recent_data['price'].pct_change().dropna()
        volatility = returns.std() * 100  # Convert to percentage
        
        # Calculate trend (linear regression slope)
        x = np.arange(len(recent_data))
        y = recent_data['price'].values
        slope, _ = np.polyfit(x, y, 1)
        
        # Normalize slope as percentage of mean price
        mean_price = recent_data['price'].mean()
        trend = (slope * len(recent_data)) / mean_price * 100
        
        # Determine market condition
        if volatility > self.volatility_threshold:
            if trend > self.trend_threshold:
                condition = "volatile_bullish"
            elif trend < -self.trend_threshold:
                condition = "volatile_bearish"
            else:
                condition = "volatile_neutral"
        else:
            if trend > self.trend_threshold:
                condition = "stable_bullish"
            elif trend < -self.trend_threshold:
                condition = "stable_bearish"
            else:
                condition = "stable_neutral"
                
        return {
            "volatility": volatility,
            "trend": trend,
            "condition": condition
        }
        
    def get_weighted_sentiment(self, days=3):
        session = self.Session()
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Modified query to handle potential absence of Twitter data
            query = """
            SELECT 
                s.source, 
                s.channel_id,
                s.text_length,
                s.combined_score,
                s.bullish_keywords,
                s.bearish_keywords
            FROM sentiment_records s
            WHERE s.processed_date >= :cutoff_date
            """
            
            result = session.execute(query, {"cutoff_date": cutoff_date})
            records = [dict(row) for row in result]
            
            if not records:
                logger.warning(f"No sentiment records found in the last {days} days")
                return 0.0
            
            # Check if we have any data
            youtube_records = [r for r in records if 'youtube' in str(r.get('source', '')).lower()]
            twitter_records = [r for r in records if 'twitter' in str(r.get('source', '')).lower()]
            
            if not twitter_records:
                logger.info("No Twitter data available - using YouTube data only")
            
            if not youtube_records and not twitter_records:
                logger.warning("No valid sentiment data available")
                return 0.0
            
            # Process any available data
            total_weighted_score = 0.0
            total_weight = 0.0
            
            for record in records:
                source_type = 'youtube' if 'youtube' in str(record.get('source', '')).lower() else 'twitter'
                channel_id = record.get('channel_id', 'unknown')
                
                # Apply source type and channel weights
                source_weight = 1.0  # Default weight for all sources
                channel_weight = self.channel_weights.get(channel_id, self.default_weight)
                
                # Adjust weight based on text length (more content = more reliable)
                length_factor = min(2.0, max(0.5, record.get("text_length", 500) / 5000))
                
                # Calculate final weight for this record
                weight = source_weight * channel_weight * length_factor
                score = record.get("combined_score", 0.0)
                
                total_weighted_score += score * weight
                total_weight += weight
            
            # Calculate final score
            avg_weighted_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
            normalized_score = max(-10, min(10, avg_weighted_score))
            
            logger.info(f"Calculated weighted sentiment: {normalized_score:.2f} from {len(records)} records")
            logger.info(f"Data sources: YouTube({len(youtube_records)}), Twitter({len(twitter_records)})")
            
            return normalized_score
        except Exception as e:
            logger.error(f"Error calculating weighted sentiment: {str(e)}")
            return 0.0
        finally:
            session.close()
            
    def load_sentiment_data(self):
        sentiment_dir = "sentiment_data"
        sentiment_data = []
        
        if not os.path.exists(sentiment_dir):
            logger.warning(f"Sentiment data directory {sentiment_dir} not found")
            return sentiment_data
        
        try:
            # Count files by source
            youtube_files = 0
            twitter_files = 0
            
            for filename in os.listdir(sentiment_dir):
                if not filename.endswith('.json'):
                    continue
                    
                try:
                    with open(os.path.join(sentiment_dir, filename), 'r') as f:
                        data = json.load(f)
                        sentiment_data.append(data)
                        
                        # Track file source
                        source = data.get('source', '').lower()
                        if 'youtube' in source:
                            youtube_files += 1
                        elif 'twitter' in source:
                            twitter_files += 1
                except Exception as e:
                    logger.error(f"Error loading sentiment data from {filename}: {e}")
            
            # Log data source information
            if len(sentiment_data) > 0:
                logger.info(f"Loaded {len(sentiment_data)} sentiment records: YouTube({youtube_files}), Twitter({twitter_files})")
                
                if twitter_files == 0 and youtube_files > 0:
                    logger.info("Operating with YouTube data only - Twitter data not available")
            else:
                logger.warning("No sentiment data loaded")
                
            return sentiment_data
        except Exception as e:
            logger.error(f"Error in load_sentiment_data: {e}")
            return []
            
    def get_combined_sentiment_score(self):
        sentiment_data = self.load_sentiment_data()
        
        if not sentiment_data:
            logger.warning("No sentiment data available for scoring")
            return 0.0
        
        # Check data sources
        youtube_data = [d for d in sentiment_data if 'youtube' in str(d.get('source', '')).lower()]
        twitter_data = [d for d in sentiment_data if 'twitter' in str(d.get('source', '')).lower()]
        
        if not twitter_data and youtube_data:
            logger.info("Calculating sentiment using YouTube data only")
        
        # Process all available sentiment data
        total_score = 0.0
        total_weight = 0.0
        
        for data in sentiment_data:
            score = data.get('combined_score', 0.0)
            source = str(data.get('source', '')).lower()
            
            # Base weight - all sources equal by default
            weight = 1.0
            
            # Adjust weight based on source type
            if 'youtube' in source:
                # YouTube videos often have more detailed content
                # We could weight them higher when Twitter is absent
                weight *= 1.0
                
                # Use predefined channel weights if available
                video_id = data.get('video_id', '')
                if video_id:
                    for channel_id, channel_weight in self.channel_weights.items():
                        if channel_id in source:
                            weight *= channel_weight
                            break
                    else:
                        # Default weight if no matching channel found
                        weight *= self.default_weight
                        
                # Consider text length (longer content might be more informative)
                text_length = data.get('text_length', 1000)
                length_factor = min(2.0, max(0.5, text_length / 5000))
                weight *= length_factor
                
            # Apply time decay - more recent data is more valuable
            timestamp = data.get('timestamp')
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    else:
                        dt = timestamp
                        
                    hours_ago = (datetime.now() - dt).total_seconds() / 3600
                    # Exponential decay with half-life of 24 hours
                    time_factor = math.exp(-0.029 * hours_ago)  # ln(2)/24 ≈ 0.029
                    weight *= time_factor
                except Exception as e:
                    logger.warning(f"Error calculating time decay: {e}")
            
            # Add to weighted average
            total_score += score * weight
            total_weight += weight
        
        # Calculate final score
        if total_weight > 0:
            final_score = total_score / total_weight
            # Normalize to range [-10, 10]
            final_score = max(-10, min(10, final_score))
            logger.info(f"Calculated combined sentiment score: {final_score:.2f}")
            return final_score
        else:
            logger.warning("No valid sentiment weights calculated")
            return 0.0
            
    def calculate_adapted_parameters(self, sentiment_score, market_condition):
        """
        Calculate strategy parameters adapted to current market conditions
        """
        # Base parameter values
        position_size_pct = 0.75  # Default 75% of allowed size
        stop_loss_pct = 3.5       # Default 3.5%
        take_profit_pct = 7.0     # Default 7.0%
        
        # Adjust for market condition
        if "volatile" in market_condition["condition"]:
            # Reduce position size, tighten stops in volatile markets
            position_size_pct *= 0.8
            stop_loss_pct *= 0.9
            take_profit_pct *= 1.2
        elif "stable" in market_condition["condition"]:
            # Increase position size, widen stops in stable markets
            position_size_pct *= 1.1
            stop_loss_pct *= 1.1
            take_profit_pct *= 0.9
            
        # Adjust for trend direction
        if "bullish" in market_condition["condition"]:
            # Bias toward longs in bullish trends
            position_size_pct *= (1.0 + 0.2 * (sentiment_score > 0))
            position_size_pct *= (1.0 - 0.3 * (sentiment_score < 0))
        elif "bearish" in market_condition["condition"]:
            # Bias toward shorts in bearish trends
            position_size_pct *= (1.0 + 0.2 * (sentiment_score < 0))
            position_size_pct *= (1.0 - 0.3 * (sentiment_score > 0))
        
        # Adjust take-profit based on volatility
        take_profit_pct = max(take_profit_pct, market_condition["volatility"] * 1.5)
        
        # Ensure we stay within configured ranges
        position_size_pct = min(self.position_size_range[1], max(self.position_size_range[0], position_size_pct))
        stop_loss_pct = min(self.stop_loss_range[1], max(self.stop_loss_range[0], stop_loss_pct))
        take_profit_pct = min(self.take_profit_range[1], max(self.take_profit_range[0], take_profit_pct))
        
        return {
            "position_size_pct": position_size_pct,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct
        }
        
    def generate_enhanced_signals(self, symbol, price_data):
        """
        Generate enhanced trading signals based on sentiment and market conditions
        """
        try:
            # Get market condition
            market_condition = self.get_market_condition(price_data)
            
            # Get sentiment score
            sentiment_score = self.get_weighted_sentiment()
            
            # Determine signal direction
            signal = 0  # Neutral by default
            
            if sentiment_score > self.bullish_threshold:
                if "bullish" in market_condition["condition"]:
                    # Strong bullish signal when sentiment and trend align
                    signal = 1
                elif "bearish" in market_condition["condition"]:
                    # Potential reversal signal - reduced strength
                    signal = 0.5
            elif sentiment_score < self.bearish_threshold:
                if "bearish" in market_condition["condition"]:
                    # Strong bearish signal when sentiment and trend align
                    signal = -1
                elif "bullish" in market_condition["condition"]:
                    # Potential reversal signal - reduced strength
                    signal = -0.5
                    
            # Calculate adaptive parameters
            params = self.calculate_adapted_parameters(sentiment_score, market_condition)
            
            result = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "sentiment_score": sentiment_score,
                "market_condition": market_condition["condition"],
                "volatility": market_condition["volatility"],
                "trend": market_condition["trend"],
                "signal": signal,
                "position_size_pct": params["position_size_pct"],
                "stop_loss_pct": params["stop_loss_pct"],
                "take_profit_pct": params["take_profit_pct"],
                "confidence": abs(sentiment_score) / 10.0  # Normalized confidence score
            }
            
            logger.info(f"Generated signal for {symbol}: {signal} (confidence: {result['confidence']:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return None

if __name__ == "__main__":
    # Test the strategy
    from crypto_analyzer import CryptoAnalyzer
    from multi_api_price_fetcher import CryptoPriceFetcher
    
    try:
        # Initialize components
        fetcher = CryptoPriceFetcher()
        analyzer = CryptoAnalyzer(fetcher)
        strategy = EnhancedStrategy()
        
        # Get price data
        symbol = "BTC"
        df = analyzer.convert_history_to_dataframe(symbol)
        
        # Generate signals
        signals = strategy.generate_enhanced_signals(symbol, df)
        
        if signals:
            print(f"\nEnhanced Strategy Signals for {symbol}:")
            print(f"Signal: {signals['signal']}")
            print(f"Sentiment Score: {signals['sentiment_score']:.2f}")
            print(f"Market Condition: {signals['market_condition']}")
            print(f"Volatility: {signals['volatility']:.2f}%")
            print(f"Trend: {signals['trend']:.2f}%")
            print(f"Confidence: {signals['confidence']:.2f}")
            print("\nAdapted Parameters:")
            print(f"Position Size: {signals['position_size_pct']:.2f}%")
            print(f"Stop Loss: {signals['stop_loss_pct']:.2f}%")
            print(f"Take Profit: {signals['take_profit_pct']:.2f}%")
    except Exception as e:
        print(f"Error testing strategy: {str(e)}")