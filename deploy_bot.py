# deploy_bot.py
import os
import sys
import json
import argparse
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deployment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DeployBot")

class BotDeployer:
    def __init__(self, config_file="deployment_config.json"):
        self.load_config(config_file)
        
    def load_config(self, config_file):
        try:
            with open(config_file, "r") as f:
                self.config = json.load(f)
                
            # Deployment settings
            self.environment = self.config.get("environment", "development")
            self.deploy_mode = self.config.get("deploy_mode", "docker")
            self.docker_image = self.config.get("docker_image", "crypto-sentiment-bot")
            self.docker_tag = self.config.get("docker_tag", "latest")
            
            # Cloud settings
            self.cloud_provider = self.config.get("cloud_provider", "aws")
            self.aws_region = self.config.get("aws_region", "us-east-1")
            self.aws_instance_type = self.config.get("aws_instance_type", "t2.micro")
            
            # Bot components to deploy
            self.deploy_youtube_tracker = self.config.get("deploy_youtube_tracker", True)
            self.deploy_twitter_collector = self.config.get("deploy_twitter_collector", True)
            self.deploy_sentiment_ml = self.config.get("deploy_sentiment_ml", True)
            self.deploy_alert_system = self.config.get("deploy_alert_system", True)
            self.deploy_paper_trader = self.config.get("deploy_paper_trader", True)
            self.deploy_dashboard = self.config.get("deploy_dashboard", True)
            
            # Monitoring settings
            self.enable_monitoring = self.config.get("enable_monitoring", True)
            self.alert_email = self.config.get("alert_email", "")
            
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
            
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        try:
            # Check Docker
            if self.deploy_mode == "docker":
                result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error("Docker not found. Please install Docker to continue.")
                    return False
                logger.info(f"Docker version: {result.stdout.strip()}")
                
            # Check AWS CLI if using AWS
            if self.cloud_provider == "aws":
                result = subprocess.run(["aws", "--version"], capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error("AWS CLI not found. Please install AWS CLI to continue.")
                    return False
                logger.info(f"AWS CLI version: {result.stdout.strip()}")
                
            # Check Python requirements
            requirements_file = "requirements.txt"
            if not os.path.exists(requirements_file):
                # Generate requirements file if it doesn't exist
                self.generate_requirements()
                
            # Check if all required source files exist
            required_files = [
                "youtube_tracker.py",
                "sentiment_analyzer.py",
                "multi_api_price_fetcher.py",
                "enhanced_strategy.py",
                "paper_trader.py"
            ]
            
            for file in required_files:
                if not os.path.exists(file):
                    logger.error(f"Required file not found: {file}")
                    return False
                    
            logger.info("All dependencies checked successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error checking dependencies: {str(e)}")
            return False
            
    def generate_requirements(self):
        """Generate requirements.txt file"""
        try:
            requirements = [
                "numpy",
                "pandas",
                "scikit-learn",
                "matplotlib",
                "plotly",
                "dash",
                "schedule",
                "requests",
                "tweepy",
                "sqlalchemy",
                "youtube-transcript-api",
                "nltk",
                "joblib",
                "python-dotenv"
            ]
            
            with open("requirements.txt", "w") as f:
                for req in requirements:
                    f.write(f"{req}\n")
                    
            logger.info("Generated requirements.txt file")
        except Exception as e:
            logger.error(f"Error generating requirements.txt: {str(e)}")
            
    def create_docker_compose(self):
        """Create a docker-compose.yml file"""
        try:
            compose = {
                "version": "3",
                "services": {}
            }
            
            # Create services for each component
            if self.deploy_youtube_tracker:
                compose["services"]["youtube-tracker"] = {
                    "image": f"{self.docker_image}-youtube-tracker:{self.docker_tag}",
                    "restart": "always",
                    "environment": [
                        "PYTHONUNBUFFERED=1"
                    ],
                    "volumes": [
                        "./data:/app/data",
                        "./sentiment_data:/app/sentiment_data",
                        "./config:/app/config"
                    ]
                }
                
            if self.deploy_twitter_collector:
                compose["services"]["twitter-collector"] = {
                    "image": f"{self.docker_image}-twitter-collector:{self.docker_tag}",
                    "restart": "always",
                    "environment": [
                        "PYTHONUNBUFFERED=1",
                        "TWITTER_API_KEY=${TWITTER_API_KEY}",
                        "TWITTER_API_SECRET=${TWITTER_API_SECRET}",
                        "TWITTER_ACCESS_TOKEN=${TWITTER_ACCESS_TOKEN}",
                        "TWITTER_ACCESS_SECRET=${TWITTER_ACCESS_SECRET}"
                    ],
                    "volumes": [
                        "./data:/app/data",
                        "./sentiment_data:/app/sentiment_data",
                        "./config:/app/config"
                    ]
                }
                
            if self.deploy_sentiment_ml:
                compose["services"]["sentiment-ml"] = {
                    "image": f"{self.docker_image}-sentiment-ml:{self.docker_tag}",
                    "restart": "always",
                    "environment": [
                        "PYTHONUNBUFFERED=1"
                    ],
                    "volumes": [
                        "./data:/app/data",
                        "./sentiment_data:/app/sentiment_data",
                        "./ml_models:/app/ml_models",
                        "./config:/app/config"
                    ]
                }
                
            if self.deploy_alert_system:
                compose["services"]["alert-system"] = {
                    "image": f"{self.docker_image}-alert-system:{self.docker_tag}",
                    "restart": "always",
                    "environment": [
                        "PYTHONUNBUFFERED=1",
                        "DISCORD_WEBHOOK_URL=${DISCORD_WEBHOOK_URL}"
                    ],
                    "volumes": [
                        "./data:/app/data",
                        "./sentiment_data:/app/sentiment_data",
                        "./config:/app/config"
                    ]
                }
                
            if self.deploy_paper_trader:
                compose["services"]["paper-trader"] = {
                    "image": f"{self.docker_image}-paper-trader:{self.docker_tag}",
                    "restart": "always",
                    "environment": [
                        "PYTHONUNBUFFERED=1"
                    ],
                    "volumes": [
                        "./data:/app/data",
                        "./paper_trading:/app/paper_trading",
                        "./config:/app/config"
                    ]
                }
                
            if self.deploy_dashboard:
                compose["services"]["dashboard"] = {
                    "image": f"{self.docker_image}-dashboard:{self.docker_tag}",
                    "restart": "always",
                    "ports": [
                        "8050:8050"
                    ],
                    "environment": [
                        "PYTHONUNBUFFERED=1"
                    ],
                    "volumes": [
                        "./data:/app/data",
                        "./sentiment_data:/app/sentiment_data",
                        "./paper_trading:/app/paper_trading",
                        "./config:/app/config"
                    ]
                }
                
            # Add database service
            compose["services"]["db"] = {
                "image": "sqlite3",
                "restart": "always",
                "volumes": [
                    "./data:/data"
                ]
            }
            
            # Add monitoring if enabled
            if self.enable_monitoring:
                compose["services"]["prometheus"] = {
                    "image": "prom/prometheus",
                    "restart": "always",
                    "ports": [
                        "9090:9090"
                    ],
                    "volumes": [
                        "./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml"
                    ]
                }
                
                compose["services"]["grafana"] = {
                    "image": "grafana/grafana",
                    "restart": "always",
                    "ports": [
                        "3000:3000"
                    ],
                    "volumes": [
                        "./monitoring/grafana:/var/lib/grafana"
                    ],
                    "depends_on": [
                        "prometheus"
                    ]
                }
                
            # Write to file
            with open("docker-compose.yml", "w") as f:
                yaml.dump(compose, f, default_flow_style=False)
                
            logger.info("Created docker-compose.yml")
            return True
        except Exception as e:
            logger.error(f"Error creating docker-compose.yml: {str(e)}")
            return False
            
    def create_dockerfile(self, component):
        """Create a Dockerfile for a specific component"""
        try:
            # Define component-specific settings
            entrypoint = {
                "youtube-tracker": "python3 youtube_tracker.py",
                "twitter-collector": "python3 twitter_collector.py",
                "sentiment-ml": "python3 sentiment_ml.py --schedule",
                "alert-system": "python3 alert_system.py",
                "paper-trader": "python3 paper_trader.py --auto",
                "dashboard": "python3 app.py"
            }
            
            # Create Dockerfile content
            dockerfile = f"""FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Create necessary directories
RUN mkdir -p data sentiment_data paper_trading ml_models predictions performance_reports

# Run the application
CMD [{entrypoint.get(component, "python3")}]
"""
            
            # Write to component-specific Dockerfile
            with open(f"Dockerfile.{component}", "w") as f:
                f.write(dockerfile)
                
            logger.info(f"Created Dockerfile for {component}")
            return True
        except Exception as e:
            logger.error(f"Error creating Dockerfile for {component}: {str(e)}")
            return False
            
    def build_docker_images(self):
        """Build Docker images for all components"""
        try:
            components = []
            
            if self.deploy_youtube_tracker:
                components.append("youtube-tracker")
            
            if self.deploy_twitter_collector:
                components.append("twitter-collector")
                
            if self.deploy_sentiment_ml:
                components.append("sentiment-ml")
                
            if self.deploy_alert_system:
                components.append("alert-system")
                
            if self.deploy_paper_trader:
                components.append("paper-trader")
                
            if self.deploy_dashboard:
                components.append("dashboard")
                
            for component in components:
                # Create Dockerfile
                self.create_dockerfile(component)
                
                # Build image
                logger.info(f"Building Docker image for {component}")
                
                result = subprocess.run(
                    ["docker", "build", 
                     "-t", f"{self.docker_image}-{component}:{self.docker_tag}",
                     "-f", f"Dockerfile.{component}", "."],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.error(f"Failed to build image for {component}: {result.stderr}")
                    return False
                    
                logger.info(f"Successfully built Docker image for {component}")
                
            logger.info("All Docker images built successfully")
            return True
        except Exception as e:
            logger.error(f"Error building Docker images: {str(e)}")
            return False
            
    def setup_aws_infrastructure(self):
        """Set up AWS infrastructure using CloudFormation"""
        try:
            if self.cloud_provider != "aws":
                logger.warning(f"Cloud provider is set to {self.cloud_provider}, not AWS")
                return False
                
            # Create CloudFormation template
            template = {
                "AWSTemplateFormatVersion": "2010-09-09",
                "Description": "Crypto Sentiment Bot Infrastructure",
                "Resources": {
                    "BotInstance": {
                        "Type": "AWS::EC2::Instance",
                        "Properties": {
                            "InstanceType": self.aws_instance_type,
                            "ImageId": "ami-0c55b159cbfafe1f0",  # Amazon Linux 2 AMI (adjust for your region)
                            "SecurityGroups": ["BotSecurityGroup"],
                            "UserData": {
                                "Fn::Base64": """#!/bin/bash
                                yum update -y
                                amazon-linux-extras install docker -y
                                service docker start
                                systemctl enable docker
                                curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
                                chmod +x /usr/local/bin/docker-compose
                                mkdir -p /opt/crypto-bot
                                cd /opt/crypto-bot
                                # Clone the repository or download the bot files
                                """
                            }
                        }
                    },
                    "BotSecurityGroup": {
                        "Type": "AWS::EC2::SecurityGroup",
                        "Properties": {
                            "GroupDescription": "Security group for the bot",
                            "SecurityGroupIngress": [
                                {
                                    "IpProtocol": "tcp",
                                    "FromPort": 22,
                                    "ToPort": 22,
                                    "CidrIp": "0.0.0.0/0"  # Restrict this in production
                                },
                                {
                                    "IpProtocol": "tcp",
                                    "FromPort": 8050,
                                    "ToPort": 8050,
                                    "CidrIp": "0.0.0.0/0"
                                },
                                {
                                    "IpProtocol": "tcp",
                                    "FromPort": 3000,
                                    "ToPort": 3000,
                                    "CidrIp": "0.0.0.0/0"
                                }
                            ]
                        }
                    }
                },
                "Outputs": {
                    "InstancePublicIP": {
                        "Description": "Public IP of the bot instance",
                        "Value": {"Fn::GetAtt": ["BotInstance", "PublicIp"]}
                    },
                    "DashboardURL": {
                        "Description": "URL for the dashboard",
                        "Value": {"Fn::Join": ["", ["http://", {"Fn::GetAtt": ["BotInstance", "PublicDnsName"]}, ":8050"]]}
                    }
                }
            }
            
            # Write template to file
            with open("cloudformation-template.json", "w") as f:
                json.dump(template, f, indent=2)
                
            logger.info("Created CloudFormation template")
            
            # Deploy using AWS CLI
            stack_name = f"crypto-sentiment-bot-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            result = subprocess.run(
                ["aws", "cloudformation", "create-stack",
                 "--stack-name", stack_name,
                 "--template-body", "file://cloudformation-template.json",
                 "--region", self.aws_region],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to create CloudFormation stack: {result.stderr}")
                return False
                
            logger.info(f"CloudFormation stack {stack_name} creation initiated")
            logger.info("Please check the AWS CloudFormation console for stack creation progress")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up AWS infrastructure: {str(e)}")
            return False
            
    def deploy_local_docker(self):
        """Deploy using Docker locally"""
        try:
            # Build Docker images
            if not self.build_docker_images():
                return False
                
            # Create docker-compose.yml
            if not self.create_docker_compose():
                return False
                
            # Start containers
            logger.info("Starting Docker containers")
            
            result = subprocess.run(
                ["docker-compose", "up", "-d"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to start Docker containers: {result.stderr}")
                return False
                
            logger.info("Docker containers started successfully")
            
            # Create monitoring directory and config if enabled
            if self.enable_monitoring:
                os.makedirs("monitoring", exist_ok=True)
                
                # Create prometheus.yml
                prometheus_config = """
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'crypto-bot'
    static_configs:
      - targets: ['youtube-tracker:8000', 'twitter-collector:8000', 'sentiment-ml:8000', 'alert-system:8000', 'paper-trader:8000', 'dashboard:8050']
"""
                with open("monitoring/prometheus.yml", "w") as f:
                    f.write(prometheus_config)
                    
                logger.info("Created monitoring configuration")
                
            return True
            
        except Exception as e:
            logger.error(f"Error deploying with Docker: {str(e)}")
            return False
            
    def create_systemd_service(self):
        """Create systemd service files for each component"""
        try:
            components = []
            
            if self.deploy_youtube_tracker:
                components.append("youtube-tracker")
            
            if self.deploy_twitter_collector:
                components.append("twitter-collector")
                
            if self.deploy_sentiment_ml:
                components.append("sentiment-ml")
                
            if self.deploy_alert_system:
                components.append("alert-system")
                
            if self.deploy_paper_trader:
                components.append("paper-trader")
                
            if self.deploy_dashboard:
                components.append("dashboard")
                
            for component in components:
                service_content = f"""[Unit]
Description=Crypto Sentiment Bot - {component}
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/crypto-bot
ExecStart=/usr/bin/python3 {component.replace('-', '_')}.py
Restart=always
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
"""
                
                service_file = f"crypto-{component}.service"
                with open(service_file, "w") as f:
                    f.write(service_content)
                    
                logger.info(f"Created systemd service file for {component}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error creating systemd service files: {str(e)}")
            return False
            
    def deploy(self):
        """Run the deployment process"""
        try:
            logger.info(f"Starting deployment in {self.environment} environment")
            
            # Check dependencies
            if not self.check_dependencies():
                logger.error("Dependency check failed. Aborting deployment.")
                return False
                
            # Deploy based on mode
            if self.deploy_mode == "docker":
                if self.cloud_provider == "aws":
                    return self.setup_aws_infrastructure()
                else:
                    return self.deploy_local_docker()
            else:  # bare metal deployment
                return self.create_systemd_service()
                
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            return False

if __name__ == "__main__":
    # Import yaml for docker-compose generation
    import yaml
    
    parser = argparse.ArgumentParser(description="Deploy Crypto Sentiment Bot")
    parser.add_argument("--config", type=str, default="deployment_config.json", help="Deployment configuration file")
    parser.add_argument("--environment", type=str, choices=["development", "production"], help="Deployment environment")
    
    args = parser.parse_args()
    
    deployer = BotDeployer(args.config)
    
    if args.environment:
        deployer.environment = args.environment
        
    if deployer.deploy():
        logger.info("Deployment completed successfully")
        sys.exit(0)
    else:
        logger.error("Deployment failed")
        sys.exit(1)