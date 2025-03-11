"""
Configuration Loader: Loads settings from config/app_config.yaml.
"""

import yaml
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/app_config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Configuration settings.

    Raises:
        FileNotFoundError: If the config file is not found.
        yaml.YAMLError: If the YAML file is invalid.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise

# Example usage (for testing purposes)
if __name__ == "__main__":
    try:
        config = load_config()
        print(config)
    except Exception as e:
        print(f"Failed to load config: {e}")