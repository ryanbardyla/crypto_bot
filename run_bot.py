import os
import time
import json  # Add this import
import argparse
import threading
import logging
from datetime import datetime

# Import the centralized logging configuration
from utils.logging_config import setup_logging, get_module_logger

# Set up logging for the application
setup_logging(log_dir="logs")
logger = get_module_logger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hyperliquid Trading Bot")
    parser.add_argument(
        "--mode", choices=["live", "paper", "backtest"], default="paper",
        help="Trading mode: live (real trading), paper (simulated), or backtest (historical)"
    )
    parser.add_argument(
        "--config", type=str, default="config.json",
        help="Configuration file path"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return
    
    logger.info(f"Starting trading bot in {args.mode} mode")
    logger.info(f"Using configuration from {args.config}")
    
    # Import the modules dynamically based on the mode
    try:
        from sentiment_analyzer import SentimentAnalyzer
        sentiment_analyzer = SentimentAnalyzer()
        
        if args.mode == "live":
            from live_trader import HyperliquidTrader
            trader = HyperliquidTrader(config_file=args.config)
            try:
                trader.start()
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, stopping bot...")
                trader.stop()
                logger.info("Bot stopped")
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                
        elif args.mode == "paper":
            from paper_trader import PaperTrader
            trader = PaperTrader()
            trader.load_state()
            try:
                with open(args.config, 'r') as f:
                    config = json.load(f)
                interval = config.get('update_interval_minutes', 15)
            except Exception as e:
                logger.warning(f"Could not read interval from config: {e}")
                interval = 15
                
            logger.info(f"Starting paper trader with {interval} minute intervals")
            trader.run_scheduler(interval_minutes=interval)
            
        elif args.mode == "backtest":
            from simple_backtester import SimpleBacktester
            trader = SimpleBacktester()
            # Add backtest configuration here
    except Exception as e:
        logger.error(f"Failed to initialize bot: {e}", exc_info=True)

if __name__ == "__main__":
    main()