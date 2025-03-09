import os
import sys
import math
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from database_manager import DatabaseManager
from multi_api_price_fetcher import CryptoPriceFetcher

# Import the centralized logging configuration
from utils.logging_config import get_module_logger

# Get logger for this module
logger = get_module_logger("EnhancedStrategy")

class EnhancedStrategy:
    def __init__(self, config_file="enhanced_strategy_config.json"):
        self.load_config(config_file)
        self.db_manager = DatabaseManager(self.db_path)
        
    def load_config(self, config_file):
        try:
            with open(config_file, "r") as f:
                self.config = json.load(f)
                
            self.channel_weights = self.config.get("channel_weights", {})
            self.default_weight = self.config.get("default_weight", 1.0)
            self.volatility_threshold = self.config.get("volatility_threshold", 2.0)
            self.trend_threshold = self.config.get("trend_threshold", 1.5)
            self.bullish_threshold = self.config.get("bullish_threshold", 3.0)
            self.bearish_threshold = self.config.get("bearish_threshold", -3.0)
            self.position_size_range = self.config.get("position_size_range", [0.5, 1.0])
            self.stop_loss_range = self.config.get("stop_loss_range", [2.0, 5.0]) 
            self.take_profit_range = self.config.get("take_profit_range", [5.0, 15.0])
            self.db_path = self.config.get("db_path", "sqlite:///sentiment_database.db")
            
            logger.info("Strategy configuration loaded")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def get_market_condition(self, price_data, lookback_days=14):
        """
        Analyze market conditions based on price data.
        
        Args:
            price_data (DataFrame): Price data with timestamp and price columns
            lookback_days (int): Number of days to analyze
            
        Returns:
            dict: Market condition metrics
        """
        if price_data is None or len(price_data) < lookback_days:
            logger.warning("Not enough price data to determine market condition")
            return {"volatility": 0, "trend": 0, "condition": "unknown"}
        
        # Filter to recent data
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_data = price_data[price_data['timestamp'] >= cutoff_date].copy()
        
        if len(recent_data) < 3:
            logger.warning(f"Insufficient recent data points ({len(recent_data)})")
            return {"volatility": 0, "trend": 0, "condition": "unknown"}
        
        # Calculate volatility
        returns = recent_data['price'].pct_change().dropna()
        volatility = returns.std() * 100  # Convert to percentage
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_data))
        y = recent_data['price'].values
        slope, _ = np.polyfit(x, y, 1)
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
        """
        Calculate weighted sentiment score from database.
        
        Args:
            days (int): Number of days to look back
            
        Returns:
            float: Weighted sentiment score
        """
        # Get recent sentiment records from database
        sentiment_records = self.db_manager.get_sentiment_data(days=days)
        
        if not sentiment_records:
            logger.warning(f"No sentiment records found in the last {days} days")
            return 0.0
        
        # Separate YouTube and Twitter records
        youtube_records = [r for r in sentiment_records if r.get('record_type') == 'youtube']
        twitter_records = [r for r in sentiment_records if r.get('record_type') == 'twitter']
        
        # Check data availability
        if not youtube_records and not twitter_records:
            logger.warning("No valid sentiment data available")
            return 0.0
            
        if not twitter_records:
            logger.info("No Twitter data available - using YouTube data only")
        
        # Calculate weighted scores
        weighted_scores = []
        total_weight = 0
        
        for record in sentiment_records:
            # Apply channel/user weights
            source_type = record.get('record_type', 'unknown')
            channel_id = record.get('channel_id', 'unknown')
            channel_weight = self.channel_weights.get(channel_id, self.default_weight)
            
            # Additional weighting factors
            # 1. Content length factor (longer content may be more significant)
            length_factor = min(2.0, max(0.5, record.get("text_length", 500) / 5000))
            
            # 2. Time decay factor (more recent content is more relevant)
            time_factor = 1.0
            if 'processed_date' in record:
                try:
                    processed_date = record['processed_date']
                    if isinstance(processed_date, str):
                        processed_date = datetime.strptime(processed_date, "%Y-%m-%d %H:%M:%S")
                    
                    hours_ago = (datetime.now() - processed_date).total_seconds() / 3600
                    time_factor = math.exp(-0.01 * hours_ago)  # Exponential decay
                except Exception as e:
                    logger.warning(f"Error calculating time decay: {e}")
            
            # Calculate weight and score
            weight = channel_weight * length_factor * time_factor
            score = record.get("combined_score", 0.0)
            
            weighted_scores.append(weight * score)
            total_weight += weight
        
        # Calculate weighted average
        if total_weight > 0:
            avg_weighted_score = sum(weighted_scores) / total_weight
            # Normalize to -10 to 10 scale
            normalized_score = max(-10, min(10, avg_weighted_score))
            
            logger.info(f"Calculated weighted sentiment: {normalized_score:.2f} from {len(sentiment_records)} records")
            logger.info(f"Data sources: YouTube({len(youtube_records)}), Twitter({len(twitter_records)})")
            
            return normalized_score
        else:
            logger.warning("No valid sentiment weights calculated")
            return 0.0

    def calculate_adapted_parameters(self, sentiment_score, market_condition):
        """
        Calculate trading parameters adapted to market conditions and sentiment.
        
        Args:
            sentiment_score (float): Weighted sentiment score
            market_condition (dict): Market condition metrics
            
        Returns:
            dict: Adapted trading parameters
        """
        # Base position size based on sentiment
        sentiment_strength = abs(sentiment_score) / 10.0  # Scale to 0-1
        position_size_pct = self.position_size_range[0] + (self.position_size_range[1] - self.position_size_range[0]) * sentiment_strength
        
        # Adjust based on market condition
        if market_condition["condition"].startswith("volatile"):
            # Reduce position size in volatile markets
            position_size_pct *= 0.8
            
            # Tighten stop loss in volatile markets
            stop_loss_pct = self.stop_loss_range[0] + (market_condition["volatility"] / 10)
            
            # Wider take profit in volatile markets
            take_profit_pct = self.take_profit_range[0] + (market_condition["volatility"] / 5)
        else:
            # Standard stop loss in stable markets
            stop_loss_pct = self.stop_loss_range[0] + (self.stop_loss_range[1] - self.stop_loss_range[0]) * 0.5
            
            # Standard take profit in stable markets
            take_profit_pct = self.take_profit_range[0] + (self.take_profit_range[1] - self.take_profit_range[0]) * 0.5
        
        # Ensure take profit is at least 1.5x stop loss
        take_profit_pct = max(take_profit_pct, market_condition["volatility"] * 1.5)
        
        # Ensure parameters are within configured ranges
        position_size_pct = min(self.position_size_range[1], max(self.position_size_range[0], position_size_pct))
        stop_loss_pct = min(self.stop_loss_range[1], max(self.stop_loss_range[0], stop_loss_pct))
        take_profit_pct = min(self.take_profit_range[1], max(self.take_profit_range[0], take_profit_pct))
        
        return {
            "position_size_pct": position_size_pct * 100,  # Convert to percentage
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct
        }

    def generate_enhanced_signals(self, symbol, price_data):
        """
        Generate trading signals based on sentiment and market conditions.
        
        Args:
            symbol (str): Cryptocurrency symbol
            price_data (DataFrame): Price data with timestamp and price columns
            
        Returns:
            dict: Trading signals and parameters
        """
        try:
            # Get market condition
            market_condition = self.get_market_condition(price_data)
            
            # Get sentiment score
            sentiment_score = self.get_weighted_sentiment()
            
            # Calculate adapted parameters
            params = self.calculate_adapted_parameters(sentiment_score, market_condition)
            
            # Generate signal based on sentiment and market condition
            signal = "NEUTRAL"
            if sentiment_score > self.bullish_threshold:
                signal = "BUY"
            elif sentiment_score < self.bearish_threshold:
                signal = "SELL"
            
            # Create result
            result = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "signal": signal,
                "sentiment_score": sentiment_score,
                "market_condition": market_condition["condition"],
                "volatility": market_condition["volatility"],
                "trend": market_condition["trend"],
                "position_size_pct": params["position_size_pct"],
                "stop_loss_pct": params["stop_loss_pct"],
                "take_profit_pct": params["take_profit_pct"],
                "confidence": abs(sentiment_score) / 10.0  # Normalized confidence score
            }
            
            logger.info(f"Generated signal for {symbol}: {signal} (confidence: {result['confidence']:.2f})")
            return result
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "signal": "ERROR",
                "error": str(e)
            }

# If run directly, test the strategy
if __name__ == "__main__":
    try:
        import sys
        
        # Get symbol from command line argument or default to BTC
        symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC"
        
        print(f"Testing enhanced strategy for {symbol}...")
        
        # Initialize dependencies
        fetcher = CryptoPriceFetcher()
        price_data = fetcher.get_price_history(symbol)
        
        if not price_data:
            print(f"No price data available for {symbol}")
            sys.exit(1)
            
        # Convert to DataFrame
        df = pd.DataFrame(price_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Initialize and test strategy
        strategy = EnhancedStrategy()
        signals = strategy.generate_enhanced_signals(symbol, df)
        
        # Display results
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