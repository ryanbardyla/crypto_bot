# run_bot.py
import argparse
import json
import logging
import os
import time
from datetime import datetime

from live_trader import HyperliquidTrader
from sentiment_analyzer import SentimentAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingBot")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Hyperliquid Trading Bot")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["live", "paper", "backtest"], 
        default="paper",
        help="Trading mode"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return
    
    # Log startup
    logger.info(f"Starting trading bot in {args.mode} mode")
    logger.info(f"Using configuration from {args.config}")
    
    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer()
    
    # Initialize trader based on mode
    if args.mode == "live":
        trader = HyperliquidTrader(config_file=args.config)
        
        # Start live trader
        try:
            trader.start()
            
            # Keep main thread alive
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping bot...")
            trader.stop()
            logger.info("Bot stopped")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            trader.stop()
            
    elif args.mode == "paper":
        # Use PaperTrader which has different methods for starting/stopping
        from paper_trader import PaperTrader
        trader = PaperTrader()
        trader.load_state()
        
        # Set up interval from config
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            interval = config.get('update_interval_minutes', 15)
        except Exception as e:
            logger.warning(f"Could not read interval from config: {e}")
            interval = 15
            
        # Start the paper trader (with its own method)
        try:
            # Instead of calling start(), call run_scheduler()
            logger.info(f"Starting paper trader with {interval} minute intervals")
            trader.run_scheduler(interval_minutes=interval)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping bot...")
            logger.info("Bot stopped")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            
    elif args.mode == "backtest":
        # Fall back to using your existing backtester
        from simple_backtester import SimpleBacktester
        trader = SimpleBacktester()
        # You would need to add code here to run the backtester

if __name__ == "__main__":
    main()