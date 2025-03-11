# Project Baseline Format

This file defines the standard format for all code in the project. Following these rules keeps files small, readable, and consistent across the team.

## 1. File Naming
- Use lowercase with underscores (e.g., `sentiment_agent.py`).
- Name files after their purpose (e.g., `redis_client.py` for Redis stuff).
- Keep each file focused on one job (e.g., one agent per file).

crypto-trading-bot/
│
├── agents/                   # Agent-specific logic
│   ├── sentiment_agent.py    # Handles YouTube sentiment analysis
│   ├── trading_agent.py      # Manages trading decisions
│   └── supervisory_agent.py  # Oversees agents and validates actions
│
├── config/                   # Configuration files
│   ├── app_config.yaml       # App settings (e.g., Redis, InfluxDB)
│   └── logging_config.yaml   # Logging setup
│
├── core/                     # Shared utilities
│   ├── config_loader.py      # Loads config from YAML
│   ├── redis_client.py       # Redis operations
│   ├── influxdb_client.py    # InfluxDB operations
│   └── rabbitmq_client.py    # RabbitMQ messaging
│
├── data/                     # Temporary storage (to be phased out)
│   ├── sentiment_data/       # JSON files for sentiment
│   └── paper_trading/        # JSON logs for trades
│
├── tests/                    # Tests for each component
│   ├── test_sentiment.py     # Sentiment agent tests
│   ├── test_trading.py       # Trading agent tests
│   └── test_supervisory.py   # Supervisory agent tests
│
├── docker-compose.yml        # Docker setup for services
├── main.py                   # Entry point to run agents
├── requirements.txt          # Python dependencies
└── README.md                 # Project overview

## 2. Code Structure
- Start every Python file with a docstring explaining what it does.
- Use classes for agents and big components.
- Keep functions short (<50 lines) and focused on one task.
- Avoid deep nesting—keep code flat and simple.

## 3. Configuration
- Store all settings in `config/app_config.yaml`.
- Use `core/config_loader.py` to read configs—don’t hardcode values!
- Use environment variables for secrets (e.g., API keys).

## 4. Logging
- Use Python’s `logging` module with `config/logging_config.yaml`.
- Log important stuff (e.g., agent startup, errors) at the right level (INFO, ERROR).

## 5. Error Handling
- Wrap external calls (e.g., APIs, databases) in try-except blocks.
- Log errors with clear messages.
- Handle failures gracefully (e.g., retry if a network call fails).

## 6. Documentation
- Add a docstring at the top of each file.
- Document classes and functions with their purpose and inputs/outputs.
- Use type hints (e.g., `def run(self) -> None:`).

## 7. Testing
- Write tests for every agent and utility in `tests/`.
- Use `pytest` and name tests like `test_sentiment.py`.
- Aim for high coverage (>80%).

## 8. Version Control
- Use Git with clear commit messages (e.g., "Add sentiment agent").
- Branch for features; merge with pull requests.
- Tag releases (e.g., `v1.0.0`) for production.

## 9. Dependencies
- List all packages in `requirements.txt`.
- Use a virtual environment (`venv`) to keep things clean.

## 10. Docker
- Use `docker-compose.yml` for services like Redis and InfluxDB.
- Configure services with environment variables.

---

**Example File:**

```python
# agents/sentiment_agent.py
"""
SentimentAgent: Scrapes YouTube for crypto sentiment and sends it to RabbitMQ.
"""

import logging
from core.config_loader import load_config
from core.rabbitmq_client import RabbitMQClient

class SentimentAgent:
    """Analyzes YouTube sentiment for trading signals."""
    def __init__(self):
        self.config = load_config()
        self.rabbit_client = RabbitMQClient(self.config["rabbitmq"])
        self.logger = logging.getLogger(__name__)

    def run(self) -> None:
        """Start scraping and publishing sentiment."""
        self.logger.info("Sentiment agent started")
        # Add scraping logic here
        pass

if __name__ == "__main__":
    agent = SentimentAgent()
    agent.run()