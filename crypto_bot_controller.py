# crypto_bot_controller.py
import os
import sys
import json
import logging
import argparse
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("controller.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CryptoBotController")

def load_config(config_file="controller_config.json"):
    """Load controller configuration"""
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
        
        # Determine the script file
        script_mapping = {
            "youtube-tracker": "youtube_tracker.py",
            "twitter-collector": "twitter_collector.py",
            "sentiment-ml": "sentiment_ml.py",
            "alert-system": "alert_system.py",
            "paper-trader": "paper_trader.py",
            "dashboard": "app.py",
            "performance-tracker": "performance_tracker.py"
        }
        
        if component not in script_mapping:
            logger.error(f"Unknown component: {component}")
            return False
            
        script = script_mapping[component]
        
        # Add component-specific arguments
        extra_args = []
        if component == "sentiment-ml":
            extra_args.append("--schedule")
        elif component == "paper-trader":
            extra_args.append("--auto")
        elif component == "performance-tracker":
            extra_args.append("--schedule")
            
        # Start the component
        cmd = [sys.executable, script] + extra_args
        
        if config.get("run_in_background", True):
            # Start in background
            if os.name == "nt":  # Windows
                import subprocess
                subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:  # Unix/Linux
                import subprocess
                subprocess.Popen(cmd, start_new_session=True)
                
            logger.info(f"Component {component} started in background")
        else:
            # Run in foreground
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
        
        # Determine the script file
        script_mapping = {
            "youtube-tracker": "youtube_tracker.py",
            "twitter-collector": "twitter_collector.py",
            "sentiment-ml": "sentiment_ml.py",
            "alert-system": "alert_system.py",
            "paper-trader": "paper_trader.py",
            "dashboard": "app.py",
            "performance-tracker": "performance_tracker.py"
        }
        
        if component not in script_mapping:
            logger.error(f"Unknown component: {component}")
            return False
            
        script = script_mapping[component]
        
        # Find and kill the process
        if os.name == "nt":  # Windows
            cmd = f'taskkill /f /im python.exe /fi "WINDOWTITLE eq {script}"'
            os.system(cmd)
        else:  # Unix/Linux
            cmd = f"pkill -f {script}"
            os.system(cmd)
            
        logger.info(f"Component {component} stopped")
        return True
    except Exception as e:
        logger.error(f"Error stopping component {component}: {str(e)}")
        return False

def start_all(config):
    """Start all components"""
    logger.info("Starting all components")
    
    components = [
        "youtube-tracker",
        "twitter-collector",
        "sentiment-ml",
        "alert-system",
        "paper-trader",
        "dashboard",
        "performance-tracker"
    ]
    
    for component in components:
        if config.get(f"enable_{component.replace('-', '_')}", True):
            start_component(component, config)
            
    logger.info("All components started")

def stop_all():
    """Stop all components"""
    logger.info("Stopping all components")
    
    components = [
        "youtube-tracker",
        "twitter-collector",
        "sentiment-ml",
        "alert-system",
        "paper-trader",
        "dashboard",
        "performance-tracker"
    ]
    
    for component in components:
        stop_component(component)
            
    logger.info("All components stopped")

def status():
    """Check status of all components"""
    logger.info("Checking component status")
    
    components = [
        "youtube-tracker",
        "twitter-collector",
        "sentiment-ml",
        "alert-system",
        "paper-trader",
        "dashboard",
        "performance-tracker"
    ]
    
    script_mapping = {
        "youtube-tracker": "youtube_tracker.py",
        "twitter-collector": "twitter_collector.py",
        "sentiment-ml": "sentiment_ml.py",
        "alert-system": "alert_system.py",
        "paper-trader": "paper_trader.py",
        "dashboard": "app.py",
        "performance-tracker": "performance_tracker.py"
    }
    
    status_info = {}
    
    for component in components:
        script = script_mapping[component]
        
        # Check if process is running
        if os.name == "nt":  # Windows
            cmd = f'tasklist /fi "IMAGENAME eq python.exe" /fi "WINDOWTITLE eq {script}" /fo csv'
            output = os.popen(cmd).read()
            running = "python.exe" in output
        else:  # Unix/Linux
            cmd = f"pgrep -f {script}"
            output = os.popen(cmd).read()
            running = bool(output.strip())
            
        status_info[component] = {
            "running": running,
            "script": script
        }
        
    # Print status table
    print("\nComponent Status:")
    print("=" * 50)
    print(f"{'Component':<20} {'Status':<10} {'Script':<20}")
    print("-" * 50)
    
    for component, info in status_info.items():
        status_str = "RUNNING" if info["running"] else "STOPPED"
        print(f"{component:<20} {status_str:<10} {info['script']:<20}")
    
    print("=" * 50)
    
    return status_info

def main():
    parser = argparse.ArgumentParser(description="Crypto Bot Controller")
    parser.add_argument("--config", type=str, default="controller_config.json", help="Controller configuration file")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start components")
    start_parser.add_argument("component", nargs="?", help="Component to start (omit for all)")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop components")
    stop_parser.add_argument("component", nargs="?", help="Component to stop (omit for all)")
    
    # Status command
    subparsers.add_parser("status", help="Check component status")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Execute command
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