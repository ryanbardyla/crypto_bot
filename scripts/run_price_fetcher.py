# scripts/run_price_fetcher.py
import time
import os
import logging
from datetime import datetime
from multi_api_price_fetcher import CryptoPriceFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/app/logs/price_fetcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PriceFetcher")

def run_price_update():
    """Run price update for all configured symbols"""
    try:
        logger.info("Starting price update")
        symbols = ["BTC", "ETH", "SOL", "DOGE"]  # Default symbols
        
        # Try to load symbols from config
        try:
            if os.path.exists("/app/config/price_fetcher_config.json"):
                import json
                with open("/app/config/price_fetcher_config.json", "r") as f:
                    config = json.load(f)
                    if "symbols" in config and isinstance(config["symbols"], list):
                        symbols = config["symbols"]
        except Exception as e:
            logger.error(f"Error loading symbols from config: {e}")
        
        # Initialize price fetcher
        fetcher = CryptoPriceFetcher(save_data=True)
        
        # Update prices for all symbols
        for symbol in symbols:
            try:
                price = fetcher.get_price(symbol)
                logger.info(f"Updated {symbol} price: ${price:.2f}")
            except Exception as e:
                logger.error(f"Error updating {symbol} price: {e}")
        
        logger.info("Price update completed")
    except Exception as e:
        logger.error(f"Error in price update: {e}")

# Main loop
if __name__ == "__main__":
    # Initial run
    run_price_update()
    
    # Get update interval from environment or use default
    update_interval = int(os.environ.get("UPDATE_INTERVAL_SECONDS", 300))  # Default: 5 minutes
    
    logger.info(f"Price fetcher running with {update_interval} second interval")
    
    try:
        while True:
            time.sleep(update_interval)
            run_price_update()
    except KeyboardInterrupt:
        logger.info("Price fetcher stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error in price fetcher: {e}")