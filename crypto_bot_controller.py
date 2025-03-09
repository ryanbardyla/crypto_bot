import os
import sys
import time
import json
import argparse
import subprocess
import logging

# Import the centralized logging configuration
from utils.logging_config import setup_logging, get_module_logger

# Set up logging for the application
setup_logging(log_dir="logs")
logger = get_module_logger("CryptoBotController")

def load_config(config_file="controller_config.json"):
    """Load configuration from file"""
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}

def start_component(component, config):
    """Start a specific component"""
    try:
        logger.info(f"Starting component: {component}")
        script_map = {
            "youtube-tracker": "youtube_tracker.py",
            "twitter-collector": "twitter_collector.py",
            "sentiment-ml": "sentiment_ml.py",
            "alert-system": "alert_system.py",
            "paper-trader": "paper_trader.py",
            "dashboard": "app.py"
        }
        
        if component not in script_map:
            logger.error(f"Unknown component: {component}")
            return False
            
        script = script_map[component]
        cmd = ["python3", script]
        extra_args = []
        
        if component == "sentiment-ml":
            extra_args.append("--schedule")
        elif component == "paper-trader":
            extra_args.append("--auto")
            
        cmd.extend(extra_args)
        
        if config.get("run_in_background", True):
            # Start in background
            if os.name == 'nt':  # Windows
                subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:  # Unix
                subprocess.Popen(cmd, start_new_session=True)
            logger.info(f"Component {component} started in background")
            return True
        else:
            # Start in foreground
            result = subprocess.run(cmd)
            if result.returncode != 0:
                logger.error(f"Component {component} exited with code {result.returncode}")
                return False
            logger.info(f"Component {component} finished")
            return True
    except Exception as e:
        logger.error(f"Error starting component {component}: {str(e)}")
        return False

def stop_component(component):
    """Stop a specific component"""
    try:
        logger.info(f"Stopping component: {component}")
        script_map = {
            "youtube-tracker": "youtube_tracker.py",
            "twitter-collector": "twitter_collector.py",
            "sentiment-ml": "sentiment_ml.py",
            "alert-system": "alert_system.py",
            "paper-trader": "paper_trader.py",
            "dashboard": "app.py"
        }
        
        if component not in script_map:
            logger.error(f"Unknown component: {component}")
            return False
            
        script = script_map[component]
        
        if os.name == 'nt':  # Windows
            cmd = f"taskkill /F /FI \"IMAGENAME eq python*\" /FI \"WINDOWTITLE eq *{script}*\""
        else:  # Unix
            cmd = f"pkill -f {script}"
            
        os.system(cmd)
        logger.info(f"Component {component} stopped")
        return True
    except Exception as e:
        logger.error(f"Error stopping component {component}: {str(e)}")
        return False

def start_all(config):
    """Start all enabled components"""
    logger.info("Starting all components")
    components = ["youtube-tracker", "twitter-collector", "sentiment-ml", 
                 "alert-system", "paper-trader", "dashboard"]
                 
    for component in components:
        if config.get(f"enable_{component.replace('-', '_')}", True):
            start_component(component, config)
    
    logger.info("All components started")

def stop_all():
    """Stop all components"""
    logger.info("Stopping all components")
    components = ["youtube-tracker", "twitter-collector", "sentiment-ml", 
                 "alert-system", "paper-trader", "dashboard"]
                 
    for component in components:
        stop_component(component)
    
    logger.info("All components stopped")

def status():
    """Check the status of all components"""
    logger.info("Checking component status")
    components = ["youtube-tracker", "twitter-collector", "sentiment-ml", 
                 "alert-system", "paper-trader", "dashboard"]
    
    script_map = {
        "youtube-tracker": "youtube_tracker.py",
        "twitter-collector": "twitter_collector.py",
        "sentiment-ml": "sentiment_ml.py",
        "alert-system": "alert_system.py",
        "paper-trader": "paper_trader.py",
        "dashboard": "app.py"
    }
    
    status_info = {}
    
    for component in components:
        script = script_map.get(component, "unknown.py")
        status_info[component] = {
            "script": script,
            "running": False
        }
        
        try:
            if os.name == 'nt':  # Windows
                cmd = f"tasklist /FI \"IMAGENAME eq python*\" /FI \"WINDOWTITLE eq *{script}*\""
            else:  # Unix
                cmd = f"pgrep -f {script}"
                
            output = os.popen(cmd).read()
            running = bool(output.strip())
            status_info[component]["running"] = running
        except Exception as e:
            logger.error(f"Error checking status for {component}: {str(e)}")
    
    print("\nComponent Status:")
    print("=" * 50)
    print(f"{'Component':<20} {'Status':<10} {'Script':<20}")
    print("-" * 50)
    for component, info in status_info.items():
        status_str = "Running" if info["running"] else "Stopped"
        print(f"{component:<20} {status_str:<10} {info['script']:<20}")
    
    return status_info

def main():
    """Main entry point for the controller"""
    parser = argparse.ArgumentParser(description="Crypto Bot Controller")
    parser.add_argument("--config", type=str, default="controller_config.json", help="Controller configuration file")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    start_parser = subparsers.add_parser("start", help="Start components")
    start_parser.add_argument("component", nargs="?", help="Component to start (omit for all)")
    
    stop_parser = subparsers.add_parser("stop", help="Stop components")
    stop_parser.add_argument("component", nargs="?", help="Component to stop (omit for all)")
    
    subparsers.add_parser("status", help="Check component status")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    if args.command == "start":
        if args.component:
            start_component(args.component, config)
        else:
            start_all(config)
    elif args.command == "stop":
        if args.component:
            stop_component(args.component)
        else:
            stop_all()
    elif args.command == "status":
        status()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()