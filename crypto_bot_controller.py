import os
import argparse
import subprocess
import time
import threading
import logging
import sys
import json

# Set up logging
try:
    from utils.logging_config import setup_logging
    setup_logging(log_dir="logs")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    os.makedirs("logs", exist_ok=True)
    
logger = logging.getLogger("CryptoBotController")

class CryptoBotController:
    def __init__(self, config_file="controller_config.json"):
        self.load_config(config_file)
        self.connection = None
        self.channel = None
        
        # Initialize RabbitMQ connection only if it's enabled in config
        if self.config.get("use_rabbitmq", False):
            self.setup_rabbitmq()

    def setup_rabbitmq(self):
        try:
            # Try to import pika here to handle import errors gracefully
            import pika
            
            self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
            self.channel = self.connection.channel()
            
            # Declare queues for different data types
            self.channel.queue_declare(queue='price_updates')
            self.channel.queue_declare(queue='sentiment_updates')
            self.channel.queue_declare(queue='trading_signals')
            self.channel.queue_declare(queue='system_status')
            
            logger.info("RabbitMQ connection established successfully")
            return True
        except ImportError:
            logger.error("Pika package not installed. Please install it with: pip install pika")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
            logger.info("Continuing with file-based communication as fallback")
            return False

    def load_config(self, config_file):
        try:
            with open(config_file, "r") as f:
                self.config = json.load(f)
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            self.config = {
                "run_in_background": True,
                "enable_youtube_tracker": True,
                "enable_twitter_collector": False,
                "enable_sentiment_ml": True,
                "enable_alert_system": True,
                "enable_paper_trader": True,
                "enable_dashboard": True,
                "enable_performance_tracker": True,
                "use_rabbitmq": False
            }

    def start_component(self, component, config):
        try:
            logger.info(f"Starting component: {component}")
            
            # Base command
            cmd = ["python", f"{component.replace('-', '_')}.py"]
            extra_args = []
            
            # Add RabbitMQ flag if enabled in config
            if config.get("use_rabbitmq", False):
                extra_args.append("--use_rabbitmq")
            
            # Component-specific arguments
            if component == "sentiment-ml":
                extra_args.append("--schedule")
            elif component == "price-fetcher":
                extra_args.append("--auto")
            elif component == "paper-trader" or component == "live-trader":
                extra_args.append("--auto")
            
            cmd.extend(extra_args)
            
            # Run in background or foreground based on config
            if config.get("run_in_background", True):
                if sys.platform == 'win32':
                    subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
                else:
                    subprocess.Popen(cmd, start_new_session=True)
                logger.info(f"Component {component} started in background with args: {extra_args}")
            else:
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    logger.error(f"Component {component} exited with code {result.returncode}")
                logger.info(f"Component {component} finished")
        except Exception as e:
            logger.error(f"Error starting component {component}: {str(e)}")

    def stop_component(self, component):
        try:
            logger.info(f"Stopping component: {component}")
            
            # Send stop command via RabbitMQ if available
            if self.connection is not None and self.channel is not None:
                try:
                    import pika  # Make sure pika is available
                    message = json.dumps({"command": "stop", "component": component})
                    self.channel.basic_publish(
                        exchange='',
                        routing_key='system_status',
                        body=message
                    )
                    logger.info(f"Stop command sent to {component} via RabbitMQ")
                    return
                except Exception as e:
                    logger.error(f"Failed to send stop command via RabbitMQ: {str(e)}")
            
            # Fallback to OS-specific process termination
            if sys.platform == 'win32':
                cmd = f"taskkill /F /IM python.exe /FI \"WINDOWTITLE eq {component.replace('-', '_')}.py\""
            else:
                cmd = f"pkill -f {component.replace('-', '_')}.py"
            
            os.system(cmd)
            logger.info(f"Component {component} stopped")
        except Exception as e:
            logger.error(f"Error stopping component {component}: {str(e)}")

    def start_all(self, config):
        logger.info("Starting all components")
        components = [
            "youtube-tracker",
            "sentiment-ml",
            "price-fetcher",
            "alert-system",
            "paper-trader",
            "dashboard",
            "performance-tracker"
        ]
        
        for component in components:
            config_key = f"enable_{component.replace('-', '_')}"
            if config.get(config_key, True):
                self.start_component(component, config)
        
        logger.info("All components started")

    def stop_all(self):
        logger.info("Stopping all components")
        components = [
            "youtube-tracker",
            "sentiment-ml",
            "price-fetcher",
            "alert-system",
            "paper-trader",
            "dashboard",
            "performance-tracker"
        ]
        
        for component in components:
            self.stop_component(component)
        
        logger.info("All components stopped")

    def status(self):
        logger.info("Checking component status")
        components = [
            "youtube-tracker",
            "sentiment-ml",
            "price-fetcher",
            "alert-system",
            "paper-trader",
            "dashboard",
            "performance-tracker"
        ]
        
        script_map = {
            "youtube-tracker": "youtube_tracker.py",
            "sentiment-ml": "sentiment_ml.py",
            "price-fetcher": "multi_api_price_fetcher.py",
            "alert-system": "alert_system.py",
            "paper-trader": "paper_trader.py",
            "dashboard": "app.py",
            "performance-tracker": "performance_tracker.py"
        }
        
        status_info = {}
        
        for component in components:
            script = script_map.get(component, "unknown.py")
            try:
                if sys.platform == 'win32':
                    cmd = f"tasklist | findstr /i python"
                    output = os.popen(cmd).read()
                    # Check if the script name appears in the process list
                    process_info = subprocess.check_output(["wmic", "process", "where", 
                                                          f"CommandLine like '%{script}%'", 
                                                          "get", "ProcessId"]).decode()
                    running = len(process_info.strip().split('\n')) > 1
                else:
                    cmd = f"pgrep -f {script}"
                    output = os.popen(cmd).read()
                    running = bool(output.strip())
                
                status_info[component] = {
                    "running": running,
                    "script": script
                }
            except Exception as e:
                logger.error(f"Error checking status for {component}: {str(e)}")
                status_info[component] = {
                    "running": False,
                    "script": script,
                    "error": str(e)
                }
        
        print("\nComponent Status:")
        print("=" * 50)
        print(f"{'Component':<20} {'Status':<10} {'Script':<20}")
        print("-" * 50)
        for component, info in status_info.items():
            status_str = "RUNNING" if info.get("running", False) else "STOPPED"
            print(f"{component:<20} {status_str:<10} {info['script']:<20}")
    
    def __del__(self):
        # Close RabbitMQ connection if it exists
        if hasattr(self, 'connection') and self.connection is not None:
            try:
                self.connection.close()
                logger.info("RabbitMQ connection closed")
            except Exception as e:
                logger.error(f"Error closing RabbitMQ connection: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Crypto Bot Controller")
    parser.add_argument("--config", type=str, default="controller_config.json", help="Controller configuration file")
    parser.add_argument("--use_rabbitmq", action="store_true", help="Enable RabbitMQ for all components")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    start_parser = subparsers.add_parser("start", help="Start components")
    start_parser.add_argument("component", nargs="?", help="Component to start (omit for all)")
    
    stop_parser = subparsers.add_parser("stop", help="Stop components")
    stop_parser.add_argument("component", nargs="?", help="Component to stop (omit for all)")
    
    subparsers.add_parser("status", help="Check component status")
    
    args = parser.parse_args()
    
    controller = CryptoBotController(args.config)
    config = controller.config
    
    # If RabbitMQ is enabled, update the config
    if args.use_rabbitmq:
        config["use_rabbitmq"] = True
        if not controller.connection:
            # Try to set up RabbitMQ if it wasn't set up in the constructor
            rabbitmq_success = controller.setup_rabbitmq()
            if not rabbitmq_success:
                logger.warning("Continuing without RabbitMQ")
    
    if args.command == "start":
        if args.component:
            controller.start_component(args.component, config)
        else:
            controller.start_all(config)
    elif args.command == "stop":
        if args.component:
            controller.stop_component(args.component)
        else:
            controller.stop_all()
    elif args.command == "status":
        controller.status()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()