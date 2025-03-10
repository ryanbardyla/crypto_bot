# alert_system.py
import os
import json
import time
import schedule
import requests
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from multi_api_price_fetcher import CryptoPriceFetcher
from database_manager import DatabaseManager
from utils.logging_config import get_module_logger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("alert_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AlertSystem")

class AlertSystem:
    def __init__(self, config_file="alert_system_config.json"):
        self.load_config(config_file)
        self.db_manager = DatabaseManager(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=os.environ.get("POSTGRES_PORT", "5432"),
            database=os.environ.get("POSTGRES_DB", "trading_db"),
            user=os.environ.get("POSTGRES_USER", "bot_user"),
            password=os.environ.get("POSTGRES_PASSWORD", "secure_password")
        )
        self.setup_database()
        self.last_alerts = {}
        self.load_last_state()
    
    def setup_database(self):
        """Set up database connection"""
        self.Session = sessionmaker(bind=self.db_manager.engine)
        logger.info("Database connection established via DatabaseManager")
    
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, "r") as f:
                self.config = json.load(f)
            
            # Discord webhook setup
            self.discord_webhook_url = self.config.get("discord_webhook_url", "")
            if not self.discord_webhook_url:
                logger.warning("Discord webhook URL is not configured")
            
            # Alert thresholds
            self.sentiment_shift_threshold = self.config.get("sentiment_shift_threshold", 2.0)
            self.extreme_sentiment_threshold = self.config.get("extreme_sentiment_threshold", 5.0)
            self.price_divergence_threshold = self.config.get("price_divergence_threshold", 3.0)
            
            # Timing settings
            self.check_interval_minutes = self.config.get("check_interval_minutes", 60)
            self.lookback_hours = self.config.get("lookback_hours", 24)
            self.min_alert_interval_hours = self.config.get("min_alert_interval_hours", 6)
            
            # Feature toggles
            self.enable_sentiment_shift = self.config.get("enable_sentiment_shift", True)
            self.enable_extreme_sentiment = self.config.get("enable_extreme_sentiment", True)
            self.enable_price_divergence = self.config.get("enable_price_divergence", True)
            
            # Symbols to monitor
            self.symbols = self.config.get("symbols", ["BTC", "ETH", "SOL"])
            
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def load_last_state(self):
        """Load previous alert state from file"""
        try:
            if os.path.exists("alert_state.json"):
                with open("alert_state.json", "r") as f:
                    self.last_alerts = json.load(f)
            logger.info("Alert state loaded")
        except Exception as e:
            logger.error(f"Failed to load alert state: {str(e)}")
            self.last_alerts = {}
    
    def save_last_state(self):
        """Save current alert state to file"""
        try:
            with open("alert_state.json", "w") as f:
                json.dump(self.last_alerts, f, indent=2)
            logger.info("Alert state saved")
        except Exception as e:
            logger.error(f"Failed to save alert state: {str(e)}")
    
    def get_recent_sentiment(self, hours=24):
        """Get recent sentiment data from the database"""
        try:
            session = self.Session()
            
            # Define the cutoff date
            cutoff_date = datetime.now() - timedelta(hours=hours)
            
            # Query to get hourly sentiment averages
            query = """
            SELECT 
                DATE_TRUNC('hour', processed_date) as hour,
                AVG(combined_score) as avg_score,
                COUNT(*) as record_count,
                SUM(CASE WHEN record_type = 'twitter' THEN 1 ELSE 0 END) as has_twitter,
                SUM(CASE WHEN record_type = 'youtube' THEN 1 ELSE 0 END) as has_youtube
            FROM 
                sentiment_youtube
            WHERE 
                processed_date >= :cutoff_date
            GROUP BY 
                DATE_TRUNC('hour', processed_date)
            ORDER BY 
                hour ASC
            """
            
            result = session.execute(text(query), {"cutoff_date": cutoff_date})
            sentiment_trend = [dict(row) for row in result]
            
            if not sentiment_trend:
                logger.warning(f"No sentiment data found in the last {hours} hours")
                return []
            
            # Check if we have both Twitter and YouTube data
            has_twitter = any(record.get('has_twitter', 0) > 0 for record in sentiment_trend)
            has_youtube = any(record.get('has_youtube', 0) > 0 for record in sentiment_trend)
            
            if not has_twitter and has_youtube:
                logger.info("Operating with YouTube data only - Twitter data not available")
            
            return sentiment_trend
        except Exception as e:
            logger.error(f"Error retrieving recent sentiment: {str(e)}")
            return []
        finally:
            session.close()
    
    def get_price_data(self, symbol, hours=24):
        """Get recent price data for a symbol"""
        try:
            fetcher = CryptoPriceFetcher()
            
            # Get all price history and filter for recent data
            cutoff_time = datetime.now() - timedelta(hours=hours)
            all_history = fetcher.get_price_history(symbol)
            
            # Filter for recent data
            recent_data = []
            for entry in all_history:
                try:
                    timestamp = entry.get('timestamp')
                    if isinstance(timestamp, str):
                        timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    
                    if timestamp >= cutoff_time:
                        recent_data.append(entry)
                except:
                    pass
            
            return recent_data
        except Exception as e:
            logger.error(f"Error retrieving price data for {symbol}: {str(e)}")
            return []
    
    def detect_sentiment_shift(self, sentiment_trend):
        """Detect significant shifts in sentiment"""
        if not sentiment_trend or len(sentiment_trend) < 2:
            return None
        
        # Calculate recent sentiment vs previous period
        recent_hours = min(6, len(sentiment_trend))
        previous_start = max(0, len(sentiment_trend) - 2*recent_hours)
        previous_end = len(sentiment_trend) - recent_hours
        
        recent = sentiment_trend[-recent_hours:]
        previous = sentiment_trend[previous_start:previous_end]
        
        if not previous:  # Not enough data for comparison
            return None
        
        # Calculate averages
        recent_avg = sum(r["avg_score"] for r in recent) / len(recent)
        previous_avg = sum(p["avg_score"] for p in previous) / len(previous)
        
        # Calculate the shift
        shift = recent_avg - previous_avg
        
        # Check if shift exceeds threshold
        if abs(shift) >= self.sentiment_shift_threshold:
            direction = "positive" if shift > 0 else "negative"
            return {
                "type": "sentiment_shift",
                "direction": direction,
                "shift_value": shift,
                "recent_avg": recent_avg,
                "previous_avg": previous_avg,
                "hours_compared": recent_hours,
                "time": datetime.now().isoformat()
            }
        
        return None
    
    def detect_extreme_sentiment(self, sentiment_trend):
        """Detect extremely positive or negative sentiment"""
        if not sentiment_trend or len(sentiment_trend) < 1:
            return None
        
        # Get the latest sentiment score
        latest = sentiment_trend[-1]["avg_score"]
        
        # Check if it exceeds the threshold
        if abs(latest) >= self.extreme_sentiment_threshold:
            direction = "extremely_positive" if latest > 0 else "extremely_negative"
            return {
                "type": "extreme_sentiment",
                "direction": direction,
                "value": latest,
                "threshold": self.extreme_sentiment_threshold,
                "time": datetime.now().isoformat()
            }
        
        return None
    
    def detect_price_divergence(self, symbol, sentiment_trend, price_data):
        """Detect divergence between price movement and sentiment"""
        if not sentiment_trend or not price_data or len(sentiment_trend) < 6 or len(price_data) < 6:
            return None
        
        # Get price change
        start_price = price_data[0]["price"]
        end_price = price_data[-1]["price"]
        price_change = end_price - start_price
        price_change_pct = (price_change / start_price) * 100 if start_price > 0 else 0
        
        # Get sentiment change for the same period
        recent_sentiment = sentiment_trend[-6:]  # Last 6 hours
        start_sentiment = recent_sentiment[0]["avg_score"]
        end_sentiment = recent_sentiment[-1]["avg_score"]
        sentiment_change = end_sentiment - start_sentiment
        
        # Look for divergence - price going up but sentiment down, or vice versa
        divergence = price_change_pct * sentiment_change < 0  # Opposite signs
        significance = abs(sentiment_change) + abs(price_change_pct / 5)
        
        if divergence and significance >= self.price_divergence_threshold:
            # Check if sentiment is purely from YouTube or includes Twitter
            youtube_only = all(r.get('has_twitter', 0) == 0 and r.get('has_youtube', 0) > 0 for r in recent_sentiment)
            
            return {
                "type": "price_divergence",
                "symbol": symbol,
                "price_change_pct": price_change_pct,
                "sentiment_change": sentiment_change,
                "significance": significance,
                "price_movement": "up" if price_change > 0 else "down",
                "sentiment_movement": "up" if sentiment_change > 0 else "down",
                "youtube_only": youtube_only,
                "time": datetime.now().isoformat()
            }
        
        return None
    
    def send_discord_alert(self, alert):
        """Send alert to Discord webhook"""
        if not self.discord_webhook_url:
            logger.warning("Discord webhook URL not configured, skipping alert")
            return False
        
        try:
            # Format message based on alert type
            if alert["type"] == "sentiment_shift":
                title = f"🔄 Sentiment Shift Alert: {alert['direction'].title()}"
                description = (
                    f"Sentiment has shifted {alert['direction']} by {abs(alert['shift_value']):.2f} points\n"
                    f"Recent average: {alert['recent_avg']:.2f}\n"
                    f"Previous average: {alert['previous_avg']:.2f}\n"
                    f"Hours compared: {alert['hours_compared']}"
                )
                color = 65280 if alert["direction"] == "positive" else 16711680  # Green or red
            
            elif alert["type"] == "extreme_sentiment":
                title = f"⚠️ Extreme Sentiment Alert: {alert['direction'].replace('_', ' ').title()}"
                description = (
                    f"Current sentiment: {alert['value']:.2f}\n"
                    f"Threshold: ±{alert['threshold']:.2f}"
                )
                color = 65280 if "positive" in alert["direction"] else 16711680  # Green or red
            
            elif alert["type"] == "price_divergence":
                title = f"↔️ Price-Sentiment Divergence Alert: {alert['symbol']}"
                description = (
                    f"Price movement: {alert['price_movement'].title()} ({alert['price_change_pct']:.2f}%)\n"
                    f"Sentiment movement: {alert['sentiment_movement'].title()} ({alert['sentiment_change']:.2f})\n"
                    f"Divergence significance: {alert['significance']:.2f}"
                )
                color = 16776960  # Yellow
                
                if alert.get("youtube_only", False):
                    description += "\n\n*Note: This alert is based on YouTube data only*"
            
            else:
                title = f"🔔 Trading Alert"
                description = str(alert)
                color = 3447003  # Blue
            
            # Create Discord embed
            payload = {
                "embeds": [
                    {
                        "title": title,
                        "description": description,
                        "color": color,
                        "footer": {
                            "text": f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        }
                    }
                ]
            }
            
            # Send to webhook
            response = requests.post(self.discord_webhook_url, json=payload)
            
            if response.status_code == 204:
                logger.info(f"Discord alert sent: {alert['type']}")
                return True
            else:
                logger.error(f"Failed to send Discord alert: {response.status_code} {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Discord alert: {str(e)}")
            return False
    
    def check_for_alerts(self):
        """Main method to check for all alert conditions"""
        logger.info("Checking for alert conditions...")
        alerts = []
        
        try:
            # Get recent sentiment data
            sentiment_trend = self.get_recent_sentiment(hours=self.lookback_hours)
            
            if not sentiment_trend:
                logger.warning("No sentiment data available for alerts")
                return
            
            # Mark if we only have YouTube data (no Twitter)
            youtube_only = all(r.get('has_twitter', 0) == 0 and r.get('has_youtube', 0) > 0 for r in sentiment_trend)
            if youtube_only:
                logger.info("Alert system running with YouTube data only")
            
            # Check for sentiment shift
            if self.enable_sentiment_shift:
                last_alert = self.last_alerts.get("sentiment_shift", {})
                last_time = last_alert.get("time", "")
                
                # Only alert if minimum interval has passed since last alert
                if not last_time or (datetime.now() - datetime.fromisoformat(last_time)).total_seconds() / 3600 >= self.min_alert_interval_hours:
                    shift_alert = self.detect_sentiment_shift(sentiment_trend)
                    if shift_alert:
                        alerts.append(shift_alert)
                        self.last_alerts["sentiment_shift"] = {
                            "time": datetime.now().isoformat(),
                            "value": shift_alert["shift_value"]
                        }
            
            # Check for extreme sentiment
            if self.enable_extreme_sentiment:
                last_alert = self.last_alerts.get("extreme_sentiment", {})
                last_time = last_alert.get("time", "")
                
                if not last_time or (datetime.now() - datetime.fromisoformat(last_time)).total_seconds() / 3600 >= self.min_alert_interval_hours:
                    extreme_alert = self.detect_extreme_sentiment(sentiment_trend)
                    if extreme_alert:
                        alerts.append(extreme_alert)
                        self.last_alerts["extreme_sentiment"] = {
                            "time": datetime.now().isoformat(),
                            "value": extreme_alert["value"]
                        }
            
            # Check for price divergence for each symbol
            if self.enable_price_divergence:
                for symbol in self.symbols:
                    last_alert = self.last_alerts.get(f"price_divergence_{symbol}", {})
                    last_time = last_alert.get("time", "")
                    
                    if not last_time or (datetime.now() - datetime.fromisoformat(last_time)).total_seconds() / 3600 >= self.min_alert_interval_hours:
                        price_data = self.get_price_data(symbol, hours=self.lookback_hours)
                        divergence_alert = self.detect_price_divergence(symbol, sentiment_trend, price_data)
                        
                        if divergence_alert:
                            alerts.append(divergence_alert)
                            self.last_alerts[f"price_divergence_{symbol}"] = {
                                "time": datetime.now().isoformat(),
                                "value": divergence_alert["significance"]
                            }
            
            # Send alerts
            for alert in alerts:
                self.send_discord_alert(alert)
            
            # Save state
            self.save_last_state()
            
            logger.info(f"Alert check complete. Found {len(alerts)} alerts.")
            
        except Exception as e:
            logger.error(f"Error checking for alerts: {str(e)}")
    
    def start_scheduler(self):
        """Start the scheduler for periodic alert checks"""
        logger.info(f"Starting alert system with {self.check_interval_minutes} minute interval")
        
        # Run initial check
        self.check_for_alerts()
        
        # Schedule regular checks
        schedule.every(self.check_interval_minutes).minutes.do(self.check_for_alerts)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Alert system stopped by user")

# Main execution
if __name__ == "__main__":
    alert_system = AlertSystem()
    alert_system.start_scheduler()