import os
import sys
import json
import time
import logging
import argparse
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import pika
from database_manager import DatabaseManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingStrategy")

class TradingStrategy:
    def __init__(self, config_file="trading_strategy_config.json", use_rabbitmq=False):
        self.load_config(config_file)
        self.db_manager = DatabaseManager()
        self.setup_database()
        self.use_rabbitmq = use_rabbitmq
        
    def setup_database(self):
        self.engine = self.db_manager.engine  # Use if direct engine access is needed

        # Setup RabbitMQ connection if enabled
        if self.use_rabbitmq:
            try:
                self.setup_rabbitmq()
                logger.info("RabbitMQ connection established for trading strategy")
            except Exception as e:
                logger.error(f"Failed to connect to RabbitMQ: {str(e)}. Falling back to file-based updates.")
                self.use_rabbitmq = False
    
    def setup_rabbitmq(self):
        self.rabbitmq_connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        self.rabbitmq_channel = self.rabbitmq_connection.channel()
        
        # Declare queues
        self.rabbitmq_channel.queue_declare(queue='price_updates')
        self.rabbitmq_channel.queue_declare(queue='sentiment_updates')
        self.rabbitmq_channel.queue_declare(queue='trading_signals')
        
        # Setup consumers
        self.rabbitmq_channel.basic_consume(
            queue='price_updates',
            on_message_callback=self.handle_price_update,
            auto_ack=True
        )
        
        self.rabbitmq_channel.basic_consume(
            queue='sentiment_updates',
            on_message_callback=self.handle_sentiment_update,
            auto_ack=True
        )
        
        # Start consumer thread
        self.consumer_thread = threading.Thread(target=self.start_consuming)
        self.consumer_thread.daemon = True
        self.consumer_thread.start()
    
    def start_consuming(self):
        try:
            logger.info("Starting RabbitMQ message consumption")
            self.rabbitmq_channel.start_consuming()
        except Exception as e:
            logger.error(f"Error in RabbitMQ consumer: {str(e)}")
    
    def handle_price_update(self, ch, method, properties, body):
        try:
            data = json.loads(body)
            symbol = data['symbol']
            price = data['price']
            timestamp = data.get('timestamp')
            
            logger.info(f"Received price update via RabbitMQ: {symbol} at ${price}")
            
            # Trigger signal generation with new price data
            self.generate_signals(symbol)
        except Exception as e:
            logger.error(f"Error handling price update: {str(e)}")
    
    def handle_sentiment_update(self, ch, method, properties, body):
        try:
            data = json.loads(body)
            symbol = data.get('symbol')
            sentiment_score = data.get('combined_score', 0)
            
            logger.info(f"Received sentiment update via RabbitMQ: {symbol} with score {sentiment_score}")
            
            # Update sentiment data and regenerate signals
            if symbol:
                self.generate_signals(symbol)
        except Exception as e:
            logger.error(f"Error handling sentiment update: {str(e)}")
    
    def load_config(self, config_file):
        try:
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from {config_file}")
            else:
                logger.warning(f"Config file {config_file} not found, using defaults")
                self.config = {
                    "db_path": "sqlite:///sentiment_database.db",
                    "symbols": ["BTC", "ETH", "SOL"],
                    "sentiment_lookback_days": 7,
                    "sentiment_weight": 0.6,
                    "price_weight": 0.4,
                    "sentiment_threshold": 2.0,
                    "volatility_adjustment": True,
                    "max_position_size": 0.2,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.1
                }
            
            self.db_path = self.config.get("db_path", "sqlite:///sentiment_database.db")
            self.symbols = self.config.get("symbols", ["BTC", "ETH", "SOL"])
            self.sentiment_lookback_days = self.config.get("sentiment_lookback_days", 7)
            self.sentiment_weight = self.config.get("sentiment_weight", 0.6)
            self.price_weight = self.config.get("price_weight", 0.4)
            self.sentiment_threshold = self.config.get("sentiment_threshold", 2.0)
            self.volatility_adjustment = self.config.get("volatility_adjustment", True)
            self.max_position_size = self.config.get("max_position_size", 0.2)  # 20% of portfolio
            self.stop_loss_pct = self.config.get("stop_loss_pct", 0.05)  # 5%
            self.take_profit_pct = self.config.get("take_profit_pct", 0.1)  # 10%
            
            logger.info("Strategy configuration loaded")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def setup_database(self):
        try:
            self.engine = create_engine(self.db_path)
            logger.info(f"Database connection established to {self.db_path}")
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
    
    def load_price_data(self, symbol, days=30):
        try:
            # First try to load from price_history.json
            if os.path.exists("price_history.json"):
                with open("price_history.json", "r") as f:
                    all_prices = json.load(f)
                    if symbol in all_prices:
                        price_data = pd.DataFrame(all_prices[symbol])
                        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
                        cutoff_date = datetime.now() - timedelta(days=days)
                        recent_data = price_data[price_data['timestamp'] >= cutoff_date]
                        logger.info(f"Loaded {len(recent_data)} price points for {symbol}")
                        return recent_data
            
            # If no data in price_history.json, try CSV file
            csv_path = f"data/{symbol}_price_data.csv"
            if os.path.exists(csv_path):
                price_data = pd.read_csv(csv_path)
                logger.info(f"Loaded {len(price_data)} price points for {symbol} from {csv_path}")
                return price_data
            
            logger.warning(f"No price data found for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error loading price data for {symbol}: {str(e)}")
            return None
    
    def get_sentiment_data(self, symbol=None, days=None):
        try:
            days = days or self.sentiment_lookback_days
            cutoff_date = datetime.now() - timedelta(days=days)
            
            query = """
            SELECT
                CAST(processed_date AS DATE) as date,
                AVG(combined_score) as sentiment_score,
                SUM(bullish_keywords) as total_bullish,
                SUM(bearish_keywords) as total_bearish,
                COUNT(*) as record_count
            FROM sentiment_record
            WHERE processed_date >= :cutoff_date
            """
            
            if symbol:
                query += " AND (symbol = :symbol OR symbol IS NULL)"
                
            query += " GROUP BY CAST(processed_date AS DATE) ORDER BY date"
            
            df = pd.read_sql(query, self.engine, params={"cutoff_date": cutoff_date, "symbol": symbol})
            
            if len(df) == 0:
                logger.warning(f"No sentiment data found for the last {days} days{' for ' + symbol if symbol else ''}")
                return None
            
            if len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                df['keyword_ratio'] = df['total_bullish'] / df['total_bearish'].replace(0, 1)
                df['sentiment_momentum'] = df['sentiment_score'].diff()
                
                if len(df) >= 3:
                    df['sentiment_ma_3d'] = df['sentiment_score'].rolling(window=3, min_periods=1).mean()
                
                logger.info(f"Retrieved sentiment data: {len(df)} days{' for ' + symbol if symbol else ''}")
                return df
            else:
                logger.warning(f"No sentiment data found")
                return None
        except Exception as e:
            logger.error(f"Error retrieving sentiment data: {str(e)}")
            return None
    
    def integrate_price_and_sentiment(self, symbol):
        try:
            price_df = self.load_price_data(symbol)
            sentiment_df = self.get_sentiment_data(symbol)
            
            if price_df is None or sentiment_df is None:
                logger.warning(f"Missing data for {symbol}, cannot integrate")
                return None
            
            # Convert price data to daily granularity
            daily_price = price_df.copy()
            
            # Ensure timestamp is in datetime format
            if 'timestamp' in daily_price.columns:
                daily_price['timestamp'] = pd.to_datetime(daily_price['timestamp'])
                daily_price['date'] = daily_price['timestamp'].dt.date
                daily_price['date'] = pd.to_datetime(daily_price['date'])
            
            if 'date' not in daily_price.columns:
                logger.warning("No date column in price data, cannot integrate")
                return None
            
            # Aggregate price data by day
            daily_price = daily_price.groupby('date').agg({
                'price': ['open', 'high', 'low', 'close', 'mean'],
                'source': 'first'  # Keep the first source for reference
            }).reset_index()
            
            # Flatten the column names after groupby
            daily_price.columns = ['date'] + [f'price_{col[1]}' if col[0] == 'price' else col[0] for col in daily_price.columns[1:]]
            
            # Calculate price metrics
            daily_price['price_change'] = daily_price['price_close'].pct_change() * 100
            daily_price['price_volatility'] = daily_price['price_change'].rolling(window=3, min_periods=1).std()
            
            # Merge the dataframes
            combined_df = pd.merge(daily_price, sentiment_df, on='date', how='left')
            
            # Forward fill missing sentiment data
            combined_df['sentiment_score'] = combined_df['sentiment_score'].fillna(method='ffill')
            
            # If we have enough data, calculate additional features
            if len(combined_df) >= 3:
                # Calculate momentum metrics
                combined_df['price_momentum'] = combined_df['price_change'] - combined_df['price_change'].rolling(window=3, min_periods=1).mean()
                
                # Calculate divergence between price and sentiment
                if len(combined_df) > 1:
                    # Normalize signals to compare
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    
                    # Ensure there are no NaN values
                    sentiment_momentum = combined_df['sentiment_momentum'].fillna(0).values.reshape(-1, 1)
                    price_momentum = combined_df['price_momentum'].fillna(0).values.reshape(-1, 1)
                    
                    if len(sentiment_momentum) > 1 and len(price_momentum) > 1:
                        combined_df['norm_sentiment_momentum'] = scaler.fit_transform(sentiment_momentum)
                        combined_df['norm_price_momentum'] = scaler.fit_transform(price_momentum)
                        combined_df['divergence'] = combined_df['norm_sentiment_momentum'] - combined_df['norm_price_momentum']
                        
                        # Signal if there's significant divergence
                        combined_df['divergence_signal'] = np.where(combined_df['divergence'] < -0.5, True, False)
            
            logger.info(f"Successfully integrated price and sentiment data for {symbol}")
            return combined_df
        except Exception as e:
            logger.error(f"Error integrating price and sentiment data for {symbol}: {str(e)}")
            return None
    
    def generate_signals(self, symbol):
        try:
            combined_df = self.integrate_price_and_sentiment(symbol)
            
            if combined_df is None or len(combined_df) < 3:
                logger.warning(f"Insufficient data to generate signals for {symbol}")
                return None
            
            signals = combined_df.copy()
            signals['signal'] = 0  # 1 for buy, -1 for sell, 0 for neutral
            signals['signal_strength'] = 0.0  # 0.0 to 1.0
            signals['position_size'] = 0.0  # % of portfolio
            signals['stop_loss'] = 0.0  # Price level
            signals['take_profit'] = 0.0  # Price level
            
            # Process each day
            for i in range(1, len(signals)):
                # Use today's sentiment, price, etc. to generate signal
                today = signals.iloc[i]
                yesterday = signals.iloc[i-1]
                
                # Skip if no sentiment data
                if pd.isna(today['sentiment_score']):
                    continue
                
                # Base signal on sentiment
                if today['sentiment_score'] >= self.sentiment_threshold:
                    signals.loc[signals.index[i], 'signal'] = 1  # Buy signal
                elif today['sentiment_score'] <= -self.sentiment_threshold:
                    signals.loc[signals.index[i], 'signal'] = -1  # Sell signal
                
                # Calculate signal strength based on sentiment score and momentum
                if signals.iloc[i]['signal'] != 0:
                    # Base strength on normalized sentiment score
                    base_strength = min(abs(today['sentiment_score']) / 10, 1.0)
                    
                    # Add bonus for strong momentum
                    momentum_boost = 0.0
                    if not pd.isna(today.get('sentiment_momentum', None)) and abs(today['sentiment_momentum']) > 0.5:
                        momentum_boost = min(abs(today['sentiment_momentum']) / 5, 0.3)
                    
                    # Add bonus for divergence signal
                    divergence_boost = 0.0
                    if 'divergence_signal' in today and today['divergence_signal']:
                        divergence_boost = 0.2
                    
                    # Combine the factors with capped maximum
                    signals.loc[signals.index[i], 'signal_strength'] = min(base_strength + momentum_boost + divergence_boost, 1.0)
                    
                    # Adjust for volatility if enabled
                    if self.volatility_adjustment:
                        volatility = today.get('price_volatility', None)
                        if not pd.isna(volatility) and volatility > 0:
                            # Scale position size down for high volatility, up for low volatility
                            volatility_factor = min(1 + (volatility / 5), 2.0)
                            signals.loc[signals.index[i], 'signal_strength'] /= volatility_factor
                    
                    # Calculate position size
                    signals.loc[signals.index[i], 'position_size'] = signals.iloc[i]['signal_strength'] * self.max_position_size
                    
                    # Calculate stop loss and take profit levels
                    current_price = today['price_close']
                    if signals.iloc[i]['signal'] == 1:  # Long position
                        signals.loc[signals.index[i], 'stop_loss'] = current_price * (1 - self.stop_loss_pct)
                        signals.loc[signals.index[i], 'take_profit'] = current_price * (1 + self.take_profit_pct)
                    else:  # Short position
                        signals.loc[signals.index[i], 'stop_loss'] = current_price * (1 + self.stop_loss_pct)
                        signals.loc[signals.index[i], 'take_profit'] = current_price * (1 - self.take_profit_pct)
            
            # Log the latest signal
            if len(signals) > 0:
                latest = signals.iloc[-1]
                signal_message = "NEUTRAL"
                if latest['signal'] == 1:
                    signal_message = f"BUY with strength {latest['signal_strength']:.2f}"
                elif latest['signal'] == -1:
                    signal_message = f"SELL with strength {latest['signal_strength']:.2f}"
                
                logger.info(f"Latest signal for {symbol}: {signal_message}")
                
                # If RabbitMQ is enabled, publish the signal
                if self.use_rabbitmq:
                    self.publish_trading_signal(symbol, {
                        "date": latest['date'].strftime("%Y-%m-%d"),
                        "price": float(latest['price_close']),
                        "sentiment_score": float(latest['sentiment_score']),
                        "signal": int(latest['signal']),
                        "signal_strength": float(latest['signal_strength']),
                        "position_size": float(latest['position_size']),
                        "stop_loss": float(latest['stop_loss']),
                        "take_profit": float(latest['take_profit'])
                    })
            
            return signals
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {str(e)}")
            return None
    
    def publish_trading_signal(self, symbol, signal_data):
        if not self.use_rabbitmq or not hasattr(self, 'rabbitmq_channel'):
            return
        
        try:
            message = json.dumps({
                "symbol": symbol,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "signal": signal_data
            })
            
            self.rabbitmq_channel.basic_publish(
                exchange='',
                routing_key='trading_signals',
                body=message
            )
            
            logger.info(f"Published trading signal for {symbol} via RabbitMQ")
        except Exception as e:
            logger.error(f"Error publishing trading signal: {str(e)}")
    
    def plot_signals(self, symbol, save_to_file=None):
        signals = self.generate_signals(symbol)
        
        if signals is None or len(signals) < 3:
            logger.warning(f"No signals available for {symbol}")
            return
        
        try:
            plt.figure(figsize=(12, 10))
            
            # Plot 1: Price and signals
            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(signals['date'], signals['price_close'], 'b-', label='Price')
            
            # Add buy/sell markers
            buy_signals = signals[signals['signal'] == 1]
            sell_signals = signals[signals['signal'] == -1]
            
            if len(buy_signals) > 0:
                ax1.scatter(buy_signals['date'], buy_signals['price_close'], marker='^', color='g', s=100, label='Buy Signal')
            
            if len(sell_signals) > 0:
                ax1.scatter(sell_signals['date'], sell_signals['price_close'], marker='v', color='r', s=100, label='Sell Signal')
            
            ax1.set_title(f'{symbol} Price and Signals')
            ax1.set_ylabel('Price ($)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: Sentiment
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            ax2.plot(signals['date'], signals['sentiment_score'], 'g-', label='Sentiment Score')
            ax2.axhline(y=self.sentiment_threshold, color='g', linestyle='--', alpha=0.5, label='Bullish Threshold')
            ax2.axhline(y=-self.sentiment_threshold, color='r', linestyle='--', alpha=0.5, label='Bearish Threshold')
            ax2.set_ylabel('Sentiment Score')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Plot 3: Momentum and Divergence
            if 'norm_sentiment_momentum' in signals.columns and 'norm_price_momentum' in signals.columns:
                ax3 = plt.subplot(3, 1, 3, sharex=ax1)
                ax3.plot(signals['date'], signals['norm_sentiment_momentum'], 'g-', label='Sentiment Momentum')
                ax3.plot(signals['date'], signals['norm_price_momentum'], 'b-', label='Price Momentum')
                
                if 'divergence' in signals.columns:
                    # Add divergence highlights
                    divergences = signals[signals['divergence_signal'] == True]
                    if len(divergences) > 0:
                        ax3.scatter(divergences['date'], [0] * len(divergences), marker='o', color='purple', s=50, label='Divergence')
                
                ax3.set_ylabel('Normalized Momentum')
                ax3.set_xlabel('Date')
                ax3.grid(True, alpha=0.3)
                ax3.legend()
            
            plt.tight_layout()
            
            if save_to_file:
                plt.savefig(save_to_file)
                logger.info(f"Saved signal chart to {save_to_file}")
            
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting signals for {symbol}: {str(e)}")
    
    def run_strategy(self, symbols=None):
        symbols = symbols or self.symbols
        results = {}
        
        for symbol in symbols:
            logger.info(f"Running strategy for {symbol}")
            signals = self.generate_signals(symbol)
            
            if signals is not None and len(signals) > 0:
                latest = signals.iloc[-1]
                
                # Record the results
                results[symbol] = {
                    "date": latest['date'].strftime("%Y-%m-%d"),
                    "price": latest['price_close'],
                    "sentiment_score": latest['sentiment_score'],
                    "signal": int(latest['signal']),
                    "signal_strength": float(latest['signal_strength']),
                    "position_size": float(latest['position_size']),
                    "stop_loss": float(latest['stop_loss']),
                    "take_profit": float(latest['take_profit'])
                }
                
                # Generate and save chart
                os.makedirs("charts", exist_ok=True)
                chart_file = f"charts/{symbol}_strategy_{datetime.now().strftime('%Y%m%d')}.png"
                self.plot_signals(symbol, save_to_file=chart_file)
            else:
                results[symbol] = {
                    "error": "Insufficient data"
                }
        
        # Save results to file
        results_file = f"strategy_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "signals": results
            }, f, indent=2)
        
        logger.info(f"Strategy results saved to {results_file}")
        return results
    
    def __del__(self):
        # Close RabbitMQ connection if it exists
        if hasattr(self, 'rabbitmq_connection') and self.rabbitmq_connection is not None:
            try:
                self.rabbitmq_connection.close()
                logger.info("RabbitMQ connection closed")
            except Exception as e:
                logger.error(f"Error closing RabbitMQ connection: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Trading Strategy")
    parser.add_argument("--use_rabbitmq", action="store_true", help="Use RabbitMQ for data communication")
    parser.add_argument("--symbol", type=str, help="Symbol to analyze (default: all configured symbols)")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data to analyze")
    args = parser.parse_args()
    
    strategy = TradingStrategy(use_rabbitmq=args.use_rabbitmq)
    
    if args.symbol:
        signals = strategy.generate_signals(args.symbol)
        if signals is not None:
            strategy.plot_signals(args.symbol)
    else:
        results = strategy.run_strategy()
        print("\nTrading Strategy Results:")
        print("=" * 50)
        for symbol, data in results.items():
            if "error" in data:
                print(f"{symbol}: {data['error']}")
            else:
                signal_type = "BUY" if data['signal'] > 0 else "SELL" if data['signal'] < 0 else "NEUTRAL"
                print(f"{symbol} @ ${data['price']:.2f}: {signal_type} (Strength: {data['signal_strength']:.2f})")
                if data['signal'] != 0:
                    print(f"  Position Size: {data['position_size']:.1%} of capital")
                    print(f"  Stop Loss: ${data['stop_loss']:.2f}, Take Profit: ${data['take_profit']:.2f}")
        
        print("\nDetailed results and charts saved to the 'charts' directory.")