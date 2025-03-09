# example_with_improved_logging.py

import os
import json
import time
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.logging_config import setup_logging, get_module_logger

# Application entry point: set up logging once
setup_logging(log_dir="logs")

# Get logger for this module
logger = get_module_logger(__name__)

class RateLimiter:
    """Rate limiter to prevent hitting API rate limits"""
    def __init__(self, calls_per_second=1):
        self.rate = calls_per_second
        self.last_call = 0
        self.lock = threading.Lock()
        logger.debug(f"Initialized RateLimiter with {calls_per_second} calls per second")
        
    def wait(self):
        """Wait if needed to respect the rate limit"""
        with self.lock:
            elapsed = time.time() - self.last_call
            if elapsed < (1.0 / self.rate):
                wait_time = (1.0 / self.rate) - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                time.sleep(wait_time)
            self.last_call = time.time()

class DataProcessor:
    """Example class using the improved logging and concurrency"""
    def __init__(self, config_file="config.json"):
        self.load_config(config_file)
        self.rate_limiter = RateLimiter(calls_per_second=self.api_rate_limit)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        logger.info(f"Initialized DataProcessor with {self.max_workers} workers")
        
    def load_config(self, config_file):
        """Load configuration from file"""
        try:
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from {config_file}")
            else:
                logger.warning(f"Config file {config_file} not found, using defaults")
                self.config = {}
                
            self.api_rate_limit = self.config.get("api_rate_limit", 1.0)
            self.max_workers = self.config.get("max_workers", 5)
            self.input_directory = self.config.get("input_directory", "data")
            self.output_directory = self.config.get("output_directory", "output")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}", exc_info=True)
            # Set defaults
            self.api_rate_limit = 1.0
            self.max_workers = 5
            self.input_directory = "data"
            self.output_directory = "output"
    
    def process_item(self, item):
        """Process a single item"""
        try:
            logger.debug(f"Processing item: {item}")
            self.rate_limiter.wait()  # Respect rate limits
            
            # Simulate processing
            processing_time = 0.5 + (hash(item) % 10) / 10.0  # Random time between 0.5-1.5s
            time.sleep(processing_time)
            
            result = f"Processed {item} in {processing_time:.2f}s"
            logger.debug(f"Completed processing item: {item}")
            return result
        except Exception as e:
            logger.error(f"Error processing item {item}: {str(e)}", exc_info=True)
            return None
    
    def process_batch(self, items):
        """Process a batch of items concurrently"""
        logger.info(f"Processing batch of {len(items)} items")
        results = []
        
        try:
            # Submit all tasks
            future_to_item = {self.executor.submit(self.process_item, item): item for item in items}
            
            # Collect results as they complete
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Exception in future for item {item}: {str(e)}", exc_info=True)
            
            logger.info(f"Batch processing complete. Processed {len(results)} of {len(items)} items")
            return results
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}", exc_info=True)
            return results
    
    def shutdown(self):
        """Shutdown the processor"""
        logger.info("Shutting down processor")
        self.executor.shutdown(wait=True)
        logger.debug("Executor shutdown complete")


if __name__ == "__main__":
    # Example usage
    logger.info("Starting example application")
    
    processor = DataProcessor()
    
    # Create sample data
    sample_items = [f"item_{i}" for i in range(50)]
    
    # Process in batches
    batch_size = 10
    for i in range(0, len(sample_items), batch_size):
        batch = sample_items[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(sample_items) + batch_size - 1)//batch_size}")
        results = processor.process_batch(batch)
        logger.info(f"Got {len(results)} results from batch")
    
    processor.shutdown()
    logger.info("Application completed")