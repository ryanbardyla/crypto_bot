import os
import json
import logging
import requests
import schedule
import time
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from database_manager import DatabaseManager

# Configure logging
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
        self.db_manager = DatabaseManager()  # Defaults to PostgreSQL
        self.setup_database()
        self.load_last_state()

    def setup_database(self):
        self.Session = sessionmaker(bind=self.db_manager.engine)
        logger.info("Database connection established via DatabaseManager")

    def load_config(self, config_file):
        try:
            with open(config_file, "r") as f:
                self.config = json.load(f)
            
            # Discord webhook URL
            self.discord_webhook_url = self.config.get("discord_webhook_url", "")
            if not self.discord_webhook_url:
                logger.warning("Discord webhook URL is not configured")
            
            # Alert thresholds
            self.sentiment_shift_threshold = self.config.get("sentiment_shift_threshold", 2.0)
            self.extreme_sentiment_threshold = self.config.get("extreme_sentiment_threshold", 5.0)
            self.price_divergence_threshold = self.config.get("price_divergence_threshold", 3.0)
            
            # Alert intervals
            self.check_interval_minutes = self.config.get("check_interval_minutes", 60)
            self.lookback_hours = self.config.get("lookback_hours", 24)
            self.min_alert_interval_hours = self.config.get("min_alert_interval_hours", 6)
            
            # Alert types to enable
            self.enable_sentiment_shift = self.config.get("enable_sentiment_shift", True)
            self.enable_extreme_sentiment = self.config.get("enable_extreme_sentiment", True)
            self.enable_price_divergence = self.config.get("enable_price_divergence", True)
            
            # Symbol settings
            self.symbols = self.config.get("symbols", ["BTC", "ETH", "SOL"])
            
            # Database path
            self.db_path = self.config.get("db_path", "sqlite:///sentiment_database.db")
            
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise

    def load_last_state(self):
        """Load the last alert state to prevent duplicate alerts"""
        self.last_alerts = {}
        try:
            if os.path.exists("alert_state.json"):
                with open("alert_state.json", "r") as f:
                    self.last_alerts = json.load(f)
            logger.info("Alert state loaded")
        except Exception as e:
            logger.error(f"Failed to load alert state: {str(e)}")
            self.last_alerts = {}

    def save_last_state(self):
        """Save the current alert state"""
        try:
            with open("alert_state.json", "w") as f:
                json.dump(self.last_alerts, f, indent=2)
            logger.info("Alert state saved")
        except Exception as e:
            logger.error(f"Failed to save alert state: {str(e)}")

    def get_recent_sentiment(self, hours=24):
        """Get recent sentiment data from the database"""
        session = self.Session()
        try:
            cutoff_date = datetime.now() - timedelta(hours=hours)
            
            query = """
            SELECT 
                date(processed_date) as date,
                avg(combined_score) as avg_score,
                count(*) as record_count,
                max(case when source like '%youtube%' then 1 else 0 end) as has_youtube,
                max(case when source like '%twitter%' then 1 else 0 end) as has_twitter
            FROM sentiment_records
            WHERE processed_date >= :cutoff_date
            GROUP BY date(processed_date)
            ORDER BY date(processed_date)
            """
            
            result = session.execute(text(query), {"cutoff_date": cutoff_date})
            sentiment_trend = [dict(row) for row in result]
            
            if not sentiment_trend:
                logger.warning(f"No sentiment data found in the last {hours} hours")
                return []
                
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
        """Get recent price data"""
        from multi_api_price_fetcher import CryptoPriceFetcher
        
        try:
            fetcher = CryptoPriceFetcher()
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            all_history = fetcher.get_price_history(symbol)
            
            recent_data = []
            for entry in all_history:
                if "timestamp" in entry:
                    timestamp = entry["timestamp"]
                    if isinstance(timestamp, str):
                        timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    
                    if timestamp >= cutoff_time:
                        recent_data.append(entry)
            
            return recent_data
        except Exception as e:
            logger.error(f"Error retrieving price data for {symbol}: {str(e)}")
            return []

    def detect_sentiment_shift(self, sentiment_trend):
        """Detect significant shifts in sentiment over time"""
        if not sentiment_trend or len(sentiment_trend) < 2:
            return None
            
        recent_hours = min(6, len(sentiment_trend))
        recent = sentiment_trend[-recent_hours:]
        
        previous_start = max(0, len(sentiment_trend) - 2*recent_hours)
        previous_end = len(sentiment_trend) - recent_hours
        previous = sentiment_trend[previous_start:previous_end]
        
        if not previous:
            return None
            
        recent_avg = sum(r["avg_score"] for r in recent) / len(recent)
        previous_avg = sum(p["avg_score"] for p in previous) / len(previous)
        
        shift = recent_avg - previous_avg
        
        if abs(shift) >= self.sentiment_shift_threshold:
            direction = "bullish" if shift > 0 else "bearish"
            return {
                "type": "sentiment_shift",
                "direction": direction,
                "shift": shift,
                "recent_avg": recent_avg,
                "previous_avg": previous_avg,
                "description": f"Significant {direction} sentiment shift detected: {shift:.2f}"
            }
        return None

    def detect_extreme_sentiment(self, sentiment_trend):
        """Detect extremely positive or negative sentiment"""
        if not sentiment_trend or len(sentiment_trend) < 1:
            return None
            
        latest = sentiment_trend[-1]["avg_score"]
        
        if abs(latest) >= self.extreme_sentiment_threshold:
            direction = "bullish" if latest > 0 else "bearish"
            return {
                "type": "extreme_sentiment",
                "direction": direction,
                "score": latest,
                "description": f"Extreme {direction} sentiment detected: {latest:.2f}"
            }
        return None

    def detect_price_divergence(self, symbol, sentiment_trend, price_data):
        """Detect divergence between price movement and sentiment"""
        if not sentiment_trend or not price_data or len(sentiment_trend) < 6 or len(price_data) < 6:
            return None
            
        recent_hours = min(6, len(sentiment_trend))
        recent_sentiment = sentiment_trend[-recent_hours:]
        
        first_sentiment = recent_sentiment[0]["avg_score"]
        last_sentiment = recent_sentiment[-1]["avg_score"]
        sentiment_change = last_sentiment - first_sentiment
        
        first_price = price_data[0]["price"]
        last_price = price_data[-1]["price"]
        price_change_pct = (last_price - first_price) / first_price * 100
        
        divergence = sentiment_change * price_change_pct < 0
        significance = abs(sentiment_change) + abs(price_change_pct / 5)
        
        if divergence and significance >= self.price_divergence_threshold:
            sentiment_direction = "bullish" if sentiment_change > 0 else "bearish"
            price_direction = "bearish" if price_change_pct < 0 else "bullish"
            
            youtube_only = all(r.get('has_twitter', 0) == 0 and r.get('has_youtube', 0) > 0 for r in recent_sentiment)
            source_info = " (YouTube only)" if youtube_only else ""
            
            return {
                "type": "price_divergence",
                "symbol": symbol,
                "sentiment_direction": sentiment_direction,
                "price_direction": price_direction,
                "sentiment_change": sentiment_change,
                "price_change_pct": price_change_pct,
                "significance": significance,
                "youtube_only": youtube_only,
                "description": f"{symbol}: {sentiment_direction} sentiment{source_info} but {price_direction} price movement"
            }
        return None

    def send_discord_alert(self, alert):
        """Send an alert to Discord"""
        if not self.discord_webhook_url:
            logger.warning("Discord webhook URL not configured, skipping alert")
            return False
            
        try:
            if alert["type"] == "sentiment_shift":
                emoji = "🚀" if alert["direction"] == "bullish" else "🐻"
                title = f"{emoji} Significant Sentiment Shift Detected"
                description = alert["description"]
                fields = [
                    {"name": "Recent Average", "value": f"{alert['recent_avg']:.2f}", "inline": True},
                    {"name": "Previous Average", "value": f"{alert['previous_avg']:.2f}", "inline": True},
                    {"name": "Shift", "value": f"{alert['shift']:.2f}", "inline": True}
                ]
                color = 0x00FF00 if alert["direction"] == "bullish" else 0xFF0000
                
            elif alert["type"] == "extreme_sentiment":
                emoji = "🔥" if alert["direction"] == "bullish" else "❄️"
                title = f"{emoji} Extreme Sentiment Detected"
                description = alert["description"]
                fields = [
                    {"name": "Sentiment Score", "value": f"{alert['score']:.2f}", "inline": True},
                ]
                color = 0x00FF00 if alert["direction"] == "bullish" else 0xFF0000
                
            elif alert["type"] == "price_divergence":
                emoji = "⚠️"
                title = f"{emoji} Price-Sentiment Divergence Detected"
                if alert.get("youtube_only", False):
                    title += " (YouTube Data Only)"
                description = alert["description"]
                fields = [
                    {"name": "Symbol", "value": alert["symbol"], "inline": True},
                    {"name": "Sentiment Change", "value": f"{alert['sentiment_change']:.2f}", "inline": True},
                    {"name": "Price Change", "value": f"{alert['price_change_pct']:.2f}%", "inline": True},
                    {"name": "Significance", "value": f"{alert['significance']:.2f}", "inline": True}
                ]
                color = 0xFFAA00
                
            else:
                title = "Crypto Sentiment Alert"
                description = alert["description"]
                fields = []
                color = 0x0000FF
                
            payload = {
                "embeds": [
                    {
                        "title": title,
                        "description": description,
                        "color": color,
                        "fields": fields,
                        "footer": {
                            "text": f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        }
                    }
                ]
            }
            
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
        """Check for conditions that should trigger alerts"""
        logger.info("Checking for alert conditions...")
        
        try:
            sentiment_trend = self.get_recent_sentiment(hours=self.lookback_hours)
            
            has_twitter = any(record.get('has_twitter', 0) > 0 for record in sentiment_trend)
            has_youtube = any(record.get('has_youtube', 0) > 0 for record in sentiment_trend)
            
            if not has_twitter and has_youtube:
                logger.info("Alert system running with YouTube data only")
            
            alerts = []
            
            if self.enable_sentiment_shift:
                last_alert = self.last_alerts.get("sentiment_shift", {})
                last_time = last_alert.get("time", "")
                
                if not last_time or (datetime.now() - datetime.fromisoformat(last_time)).total_seconds() / 3600 >= self.min_alert_interval_hours:
                    shift_alert = self.detect_sentiment_shift(sentiment_trend)
                    if shift_alert:
                        alerts.append(shift_alert)
                        self.last_alerts["sentiment_shift"] = {
                            "time": datetime.now().isoformat(),
                            "alert": shift_alert
                        }
            
            if self.enable_extreme_sentiment:
                last_alert = self.last_alerts.get("extreme_sentiment", {})
                last_time = last_alert.get("time", "")
                
                if not last_time or (datetime.now() - datetime.fromisoformat(last_time)).total_seconds() / 3600 >= self.min_alert_interval_hours:
                    extreme_alert = self.detect_extreme_sentiment(sentiment_trend)
                    if extreme_alert:
                        if not has_twitter and has_youtube:
                            extreme_alert["description"] += " (YouTube data only)"
                        alerts.append(extreme_alert)
                        self.last_alerts["extreme_sentiment"] = {
                            "time": datetime.now().isoformat(),
                            "alert": extreme_alert
                        }
            
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
                                "alert": divergence_alert
                            }
            
            for alert in alerts:
                self.send_discord_alert(alert)
            
            self.save_last_state()
            
            logger.info(f"Alert check complete. Found {len(alerts)} alerts.")
        except Exception as e:
            logger.error(f"Error checking for alerts: {str(e)}")

    def start_scheduler(self):
        """Start the scheduled alert checker"""
        logger.info(f"Starting alert system with {self.check_interval_minutes} minute interval")
        
        self.check_for_alerts()
        
        schedule.every(self.check_interval_minutes).minutes.do(self.check_for_alerts)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Alert system stopped by user")

if __name__ == "__main__":
    alert_system = AlertSystem()
    alert_system.start_scheduler()