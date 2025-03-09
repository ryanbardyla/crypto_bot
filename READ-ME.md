# CryptoBot - Cryptocurrency Trading System

## Overview

CryptoBot is a comprehensive cryptocurrency trading system that combines technical analysis with sentiment analysis from social media and video content. The system is designed to analyze market trends, gather sentiment data from various sources, and execute trades based on configurable strategies.

## Key Features

- **Multi-Source Data Collection**:
  - Historical and real-time price data for cryptocurrencies
  - YouTube video transcript analysis
  - Twitter/X data collection and sentiment analysis (currently disabled)

- **Advanced Trading Strategies**:
  - Technical analysis indicators (RSI, MACD, Bollinger Bands, etc.)
  - Sentiment-based trading signals
  - Customizable strategy parameters

- **Trading Execution**:
  - Paper trading for strategy testing
  - Live trading via Hyperliquid API
  - Risk management with configurable stop-loss and take-profit levels

- **Monitoring and Analytics**:
  - Web dashboard for performance monitoring
  - Automated alerts via Discord
  - Performance tracking with key metrics (returns, drawdowns, Sharpe ratio)

- **Machine Learning Integration**:
  - Sentiment analysis models
  - Price prediction based on historical and sentiment data

## System Architecture

The system is organized into several interconnected modules:

1. **Data Collection**:
   - `youtube_tracker.py`: Collects and analyzes YouTube video transcripts
   - `twitter_collector.py`: Gathers tweets from specified accounts and search terms
   - `multi_api_price_fetcher.py`: Fetches price data from multiple sources

2. **Analysis**:
   - `sentiment_analyzer.py`: Performs sentiment analysis on text data
   - `crypto_analyzer.py`: Analyzes price data and generates technical signals
   - `sentiment_ml.py`: Machine learning for sentiment-based price prediction

3. **Trading**:
   - `paper_trader.py`: Simulates trades without real money
   - `live_trader.py`: Executes real trades via exchange APIs
   - `enhanced_strategy.py`: Implements advanced trading strategies

4. **Monitoring**:
   - `performance_tracker.py`: Tracks trading performance metrics
   - `alert_system.py`: Sends notifications about important events
   - `app.py`: Web dashboard for monitoring system performance

5. **Infrastructure**:
   - `database_manager.py`: Manages the sentiment and trading databases
   - `crypto_bot_controller.py`: Controls all system components
   - `deploy_bot.py`: Handles deployment and infrastructure setup

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python packages listed in `requirements.txt`
- API keys for:
  - YouTube Data API
  - Twitter API (optional)
  - Hyperliquid API (for live trading)

### Installation

1. Clone the repository:
   ```
   git clone [repository-url]
   cd cryptobot
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   YOUTUBE_API_KEY=your_youtube_api_key
   TWITTER_CONSUMER_KEY=your_twitter_consumer_key
   TWITTER_CONSUMER_SECRET=your_twitter_consumer_secret
   TWITTER_ACCESS_TOKEN=your_twitter_access_token
   TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
   DISCORD_TOKEN=your_discord_token
   DISCORD_WEBHOOK_URL=your_discord_webhook_url
   ```

4. Configure system settings in the configuration files:
   - `config.json`: General settings
   - `controller_config.json`: Component activation
   - `youtube_tracker_config.json`: YouTube channel settings
   - `twitter_collector_config.json`: Twitter settings
   - `enhanced_strategy_config.json`: Trading strategy parameters

### Running the System

To start the system with all components:

```
python crypto_bot_controller.py start
```

To start individual components:

```
python crypto_bot_controller.py start youtube-tracker
python crypto_bot_controller.py start sentiment-ml
python crypto_bot_controller.py start paper-trader
```

To check system status:

```
python crypto_bot_controller.py status
```

### Running in Paper Trading Mode

For testing without real money:

```
python paper_trader.py
```

This will start a paper trading session using current market data and configured strategies.

## Component Details

### YouTube Tracker

- Collects transcripts from specified YouTube channels
- Analyzes sentiment in cryptocurrency discussions
- Stores results in the sentiment database

### Sentiment Analyzer

- Processes text data from various sources
- Uses VADER sentiment analysis and keyword detection
- Outputs numerical sentiment scores for trading decisions

### Paper Trader

- Simulates trading with virtual funds
- Tests strategies without financial risk
- Provides performance metrics for strategy evaluation

### Alert System

- Monitors market conditions and sentiment shifts
- Sends notifications via Discord for significant events
- Configurable thresholds for various alert types

## Database Structure

The system uses SQLite databases to store:

- Sentiment data from YouTube and Twitter
- Trading performance metrics
- Historical price data
- Backtesting results

## Future Development

- Integration with additional data sources
- Enhanced machine learning models for price prediction
- Portfolio optimization algorithms
- Support for additional exchanges and cryptocurrencies

