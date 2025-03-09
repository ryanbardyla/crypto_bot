# utils/logging_config.py

import os
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

def setup_logging(name=None, log_dir="logs", console_level=logging.INFO, file_level=logging.DEBUG, 
                 max_size=10*1024*1024, backup_count=5, rotating_type="size"):
    """
    Set up a centralized logging configuration
    
    Args:
        name (str, optional): Logger name. If None, returns the root logger.
        log_dir (str): Directory to store log files.
        console_level (int): Log level for console output.
        file_level (int): Log level for file output.
        max_size (int): Maximum size of log file in bytes before rotation (for size-based rotation).
        backup_count (int): Number of backup log files to keep.
        rotating_type (str): Type of log rotation: "size" or "time".
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Get logger (root logger if name is None)
    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level))  # Set to the more detailed level
    
    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # Determine log filename
    log_filename = os.path.join(log_dir, f"{name or 'root'}.log")
    
    # Add file handler with rotation
    if rotating_type.lower() == "time":
        # Time-based rotation (daily at midnight)
        file_handler = TimedRotatingFileHandler(
            log_filename, 
            when='midnight',
            interval=1,
            backupCount=backup_count
        )
    else:
        # Size-based rotation
        file_handler = RotatingFileHandler(
            log_filename, 
            maxBytes=max_size, 
            backupCount=backup_count
        )
    
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add a special handler for errors and above to go to a separate errors log
    error_log = os.path.join(log_dir, f"{name or 'root'}_errors.log")
    error_handler = RotatingFileHandler(
        error_log,
        maxBytes=max_size,
        backupCount=backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    logger.debug(f"Logging configured: console={console_level}, file={file_level}")
    return logger

def get_module_logger(module_name):
    """
    Get a logger for a specific module
    
    This is a convenience function to get a logger with the module name
    already configured.
    
    Args:
        module_name (str): Name of the module (typically __name__)
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(module_name)

# Example usage in any module:
"""
from utils.logging_config import setup_logging, get_module_logger

# In your application entry point (only once):
setup_logging(log_dir="logs")

# Then in each module:
logger = get_module_logger(__name__)
logger.info("This is an info message")
logger.debug("This is a debug message")
logger.error("This is an error message")
"""