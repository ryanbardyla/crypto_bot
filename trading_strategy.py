import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

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
    def __init__(self, config_file="trading_strategy_config.json"):
        self.load_config(config_file)
        self.setup_database()
        self.price_data = {}
        self.sentiment_data = {}
        self.signals = {}
        
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    self.config = json.load(f)
            else:
                logger.warning(f"Config file {config_file} not found, using defaults")
                self.config = {}
            
            # Trading parameters with defaults
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
        """Set up database connection"""
        try:
            self.engine = create_engine(self.db_path)
            logger.info(f"Database connection established to {self.db_path}")
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
    
    def load_price_data(self, symbol, days=30):
        """Load price data for a symbol"""
        try:
            # Try to load from price_history.json first
            if os.path.exists("price_history.json"):
                with open("price_history.json", "r") as f:
                    all_prices = json.load(f)
                    if symbol in all_prices:
                        price_data = pd.DataFrame(all_prices[symbol])
                        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
                        # Filter to the requested timeframe
                        cutoff_date = datetime.now() - timedelta(days=days)
                        price_data = price_data[price_data['timestamp'] >= cutoff_date]
                        
                        # Save to instance cache
                        self.price_data[symbol] = price_data
                        logger.info(f"Loaded {len(price_data)} price points for {symbol}")
                        return price_data
            
            # If symbol not found in price_history.json, look for CSV files
            for csv_path in [f"data/{symbol}_historical_{days}d.csv", f"data/{symbol}_historical.csv"]:
                if os.path.exists(csv_path):
                    price_data = pd.read_csv(csv_path)
                    price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
                    
                    # Save to instance cache
                    self.price_data[symbol] = price_data
                    logger.info(f"Loaded {len(price_data)} price points for {symbol} from {csv_path}")
                    return price_data
            
            logger.warning(f"No price data found for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error loading price data for {symbol}: {str(e)}")
            return None
    
    def get_sentiment_data(self, symbol=None, days=None):
        """Get sentiment data from database"""
        if days is None:
            days = self.sentiment_lookback_days
            
        try:
            # Build SQL query
            query = """
            SELECT 
                date(processed_date) as date,
                avg(combined_score) as sentiment_score,
                count(*) as video_count,
                sum(bullish_keywords) as total_bullish,
                sum(bearish_keywords) as total_bearish
            FROM 
                sentiment_youtube
            WHERE 
                processed_date >= :cutoff_date
            """
            
            # Add symbol filter if specified
            symbol_filter = ""
            if symbol:
                # If mentioned_cryptos is properly populated
                symbol_filter = f" AND (mentioned_cryptos LIKE '%{symbol}%' OR title LIKE '%{symbol}%')"
                query += symbol_filter
            
            # Group and order
            query += " GROUP BY date(processed_date) ORDER BY date(processed_date)"
            
            # Execute query
            cutoff_date = datetime.now() - timedelta(days=days)
            df = pd.read_sql(query, self.engine, params={"cutoff_date": cutoff_date})
            
            if len(df) == 0:
                logger.warning(f"No sentiment data found for the last {days} days{' for ' + symbol if symbol else ''}")
                return None
            
            # Calculate additional metrics
            if len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                df['net_keywords'] = df['total_bullish'] - df['total_bearish']
                df['keyword_ratio'] = df['total_bullish'] / df['total_bearish'].replace(0, 1)
                
                # Calculate rolling averages and momentum
                if len(df) >= 3:
                    df['sentiment_ma_3d'] = df['sentiment_score'].rolling(window=3, min_periods=1).mean()
                    df['sentiment_momentum'] = df['sentiment_score'] - df['sentiment_ma_3d']
                
                # Store in instance cache
                key = symbol if symbol else "all"
                self.sentiment_data[key] = df
                
                logger.info(f"Retrieved sentiment data: {len(df)} days{' for ' + symbol if symbol else ''}")
                return df
            else:
                logger.warning(f"No sentiment data found")
                return None
        except Exception as e:
            logger.error(f"Error retrieving sentiment data: {str(e)}")
            return None
    
    def integrate_price_and_sentiment(self, symbol):
        """Combine price and sentiment data"""
        try:
            price_df = self.load_price_data(symbol)
            sentiment_df = self.get_sentiment_data(symbol)
            
            if price_df is None or sentiment_df is None:
                logger.warning(f"Missing data for {symbol}, cannot integrate")
                return None
            
            # Convert price data to daily for merging
            price_df['date'] = price_df['timestamp'].dt.date
            daily_price = price_df.groupby('date').agg({
                'price': ['first', 'last', 'min', 'max', 'mean'],
                'timestamp': 'first'  # Keep timestamp for reference
            }).reset_index()
            
            # Flatten multi-level columns
            daily_price.columns = ['date', 'price_open', 'price_close', 'price_low', 'price_high', 'price_mean', 'timestamp']
            daily_price['date'] = pd.to_datetime(daily_price['date'])
            
            # Calculate price metrics
            daily_price['price_change'] = daily_price['price_close'].pct_change() * 100
            daily_price['price_volatility'] = daily_price['price_change'].rolling(window=3, min_periods=1).std()
            
            # Merge sentiment and price data
            combined_df = pd.merge(daily_price, sentiment_df, on='date', how='left')
            
            # Forward fill missing sentiment data (use most recent sentiment)
            combined_df['sentiment_score'] = combined_df['sentiment_score'].fillna(method='ffill')
            
            # Calculate combined metrics
            if 'sentiment_momentum' in combined_df.columns:
                combined_df['price_momentum'] = combined_df['price_change'] - combined_df['price_change'].rolling(window=3, min_periods=1).mean()
                
                # Normalize sentiment and price momentum to compare them
                scaler = MinMaxScaler(feature_range=(-1, 1))
                if len(combined_df) > 1:
                    combined_df['norm_sentiment_momentum'] = scaler.fit_transform(combined_df[['sentiment_momentum']].fillna(0))
                    combined_df['norm_price_momentum'] = scaler.fit_transform(combined_df[['price_momentum']].fillna(0))
                    
                    # Detect divergence (sentiment and price moving in opposite directions)
                    combined_df['divergence'] = combined_df['norm_sentiment_momentum'] * combined_df['norm_price_momentum']
                    combined_df['divergence_signal'] = np.where(combined_df['divergence'] < -0.5, True, False)
            
            logger.info(f"Successfully integrated price and sentiment data for {symbol}")
            return combined_df
        except Exception as e:
            logger.error(f"Error integrating price and sentiment data for {symbol}: {str(e)}")
            return None
    
    def generate_signals(self, symbol):
        """Generate trading signals for a symbol"""
        try:
            combined_df = self.integrate_price_and_sentiment(symbol)
            if combined_df is None or len(combined_df) < 3:
                logger.warning(f"Insufficient data to generate signals for {symbol}")
                return None
            
            # Create signals dataframe
            signals = combined_df.copy()
            signals['signal'] = 0  # 0 = no signal, 1 = buy, -1 = sell
            signals['signal_strength'] = 0.0  # 0.0 to 1.0
            signals['stop_loss'] = 0.0
            signals['take_profit'] = 0.0
            signals['position_size'] = 0.0
            
            # Generate signals based on sentiment and price
            for i in range(1, len(signals)):
                today = signals.iloc[i]
                yesterday = signals.iloc[i-1]
                
                # Skip if missing sentiment data
                if pd.isna(today['sentiment_score']):
                    continue
                
                # Base conditions
                sentiment_bullish = today['sentiment_score'] > self.sentiment_threshold
                sentiment_bearish = today['sentiment_score'] < -self.sentiment_threshold
                sentiment_rising = today['sentiment_score'] > yesterday['sentiment_score']
                sentiment_falling = today['sentiment_score'] < yesterday['sentiment_score']
                price_rising = today['price_change'] > 0
                price_falling = today['price_change'] < 0
                
                # Additional variables if available
                sentiment_momentum_bullish = False
                sentiment_momentum_bearish = False
                divergence_bullish = False
                divergence_bearish = False
                
                if 'sentiment_momentum' in signals.columns:
                    sentiment_momentum_bullish = today['sentiment_momentum'] > 0
                    sentiment_momentum_bearish = today['sentiment_momentum'] < 0
                
                if 'divergence_signal' in signals.columns and today['divergence_signal']:
                    divergence_bullish = today['norm_sentiment_momentum'] > 0 and today['norm_price_momentum'] < 0
                    divergence_bearish = today['norm_sentiment_momentum'] < 0 and today['norm_price_momentum'] > 0
                
                # Signal logic
                if sentiment_bullish and (sentiment_rising or sentiment_momentum_bullish or divergence_bullish):
                    signals.loc[signals.index[i], 'signal'] = 1
                    # Signal strength based on sentiment score and momentum
                    base_strength = min(abs(today['sentiment_score']) / 10, 1.0)
                    momentum_boost = 0.2 if sentiment_momentum_bullish else 0
                    divergence_boost = 0.3 if divergence_bullish else 0
                    signals.loc[signals.index[i], 'signal_strength'] = min(base_strength + momentum_boost + divergence_boost, 1.0)
                
                elif sentiment_bearish and (sentiment_falling or sentiment_momentum_bearish or divergence_bearish):
                    signals.loc[signals.index[i], 'signal'] = -1
                    # Signal strength based on sentiment score and momentum
                    base_strength = min(abs(today['sentiment_score']) / 10, 1.0)
                    momentum_boost = 0.2 if sentiment_momentum_bearish else 0
                    divergence_boost = 0.3 if divergence_bearish else 0
                    signals.loc[signals.index[i], 'signal_strength'] = min(base_strength + momentum_boost + divergence_boost, 1.0)
                
                # Risk management calculations
                if signals.loc[signals.index[i], 'signal'] != 0:
                    price = today['price_close']
                    volatility_factor = 1.0
                    
                    # Adjust for volatility
                    if self.volatility_adjustment and 'price_volatility' in signals.columns:
                        # Higher volatility = smaller position size, wider stops
                        volatility = today['price_volatility']
                        if not pd.isna(volatility) and volatility > 0:
                            # Normalize volatility - higher volatility = higher factor
                            volatility_factor = min(1 + (volatility / 5), 2.0)
                    
                    # Set position size based on signal strength and volatility
                    signal_strength = signals.loc[signals.index[i], 'signal_strength']
                    position_size = self.max_position_size * signal_strength / volatility_factor
                    signals.loc[signals.index[i], 'position_size'] = position_size
                    
                    # Set stop loss and take profit based on volatility
                    direction = signals.loc[signals.index[i], 'signal']
                    stop_loss_pct = self.stop_loss_pct * volatility_factor
                    take_profit_pct = self.take_profit_pct * volatility_factor
                    
                    if direction == 1:  # Buy signal
                        signals.loc[signals.index[i], 'stop_loss'] = price * (1 - stop_loss_pct)
                        signals.loc[signals.index[i], 'take_profit'] = price * (1 + take_profit_pct)
                    else:  # Sell signal
                        signals.loc[signals.index[i], 'stop_loss'] = price * (1 + stop_loss_pct)
                        signals.loc[signals.index[i], 'take_profit'] = price * (1 - take_profit_pct)
            
            # Save to instance cache
            self.signals[symbol] = signals
            
            # Look at the latest signal
            latest_signal = signals.iloc[-1]
            signal_message = "NEUTRAL"
            if latest_signal['signal'] == 1:
                signal_message = f"BUY {symbol} with {latest_signal['position_size']:.1%} of capital"
                signal_message += f" (Stop Loss: ${latest_signal['stop_loss']:.2f}, Take Profit: ${latest_signal['take_profit']:.2f})"
            elif latest_signal['signal'] == -1:
                signal_message = f"SELL {symbol} with {latest_signal['position_size']:.1%} of capital"
                signal_message += f" (Stop Loss: ${latest_signal['stop_loss']:.2f}, Take Profit: ${latest_signal['take_profit']:.2f})"
                
            logger.info(f"Latest signal for {symbol}: {signal_message}")
            return signals
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {str(e)}")
            return None
    
    def plot_signals(self, symbol, save_to_file=None):
        """Plot price, sentiment and signals for a symbol"""
        if symbol not in self.signals or self.signals[symbol] is None:
            logger.warning(f"No signals available for {symbol}")
            return
        
        signals = self.signals[symbol]
        
        try:
            plt.figure(figsize=(12, 10))
            
            # Plot 1: Price and Signals
            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(signals['date'], signals['price_close'], 'b-', label='Price')
            
            # Plot buy signals
            buy_signals = signals[signals['signal'] == 1]
            if len(buy_signals) > 0:
                ax1.scatter(buy_signals['date'], buy_signals['price_close'], marker='^', color='g', s=100, label='Buy Signal')
            
            # Plot sell signals
            sell_signals = signals[signals['signal'] == -1]
            if len(sell_signals) > 0:
                ax1.scatter(sell_signals['date'], sell_signals['price_close'], marker='v', color='r', s=100, label='Sell Signal')
                
            ax1.set_title(f'{symbol} Price and Signals')
            ax1.set_ylabel('Price ($)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: Sentiment Score
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            ax2.plot(signals['date'], signals['sentiment_score'], 'g-', label='Sentiment Score')
            ax2.axhline(y=self.sentiment_threshold, color='g', linestyle='--', alpha=0.5, label='Bullish Threshold')
            ax2.axhline(y=-self.sentiment_threshold, color='r', linestyle='--', alpha=0.5, label='Bearish Threshold')
            ax2.set_ylabel('Sentiment Score')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Plot 3: Sentiment vs Price Momentum (if available)
            if 'norm_sentiment_momentum' in signals.columns and 'norm_price_momentum' in signals.columns:
                ax3 = plt.subplot(3, 1, 3, sharex=ax1)
                ax3.plot(signals['date'], signals['norm_sentiment_momentum'], 'g-', label='Sentiment Momentum')
                ax3.plot(signals['date'], signals['norm_price_momentum'], 'b-', label='Price Momentum')
                
                # Highlight divergences
                if 'divergence_signal' in signals.columns:
                    divergences = signals[signals['divergence_signal']]
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
        """Run the strategy for all specified symbols"""
        if symbols is None:
            symbols = self.symbols
            
        results = {}
        for symbol in symbols:
            logger.info(f"Running strategy for {symbol}")
            signals = self.generate_signals(symbol)
            
            if signals is not None and len(signals) > 0:
                # Get the latest signal
                latest = signals.iloc[-1]
                
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
                
                # Create charts directory if doesn't exist
                os.makedirs("charts", exist_ok=True)
                chart_file = f"charts/{symbol}_strategy_{datetime.now().strftime('%Y%m%d')}.png"
                self.plot_signals(symbol, save_to_file=chart_file)
            else:
                results[symbol] = {"error": "Insufficient data"}
        
        # Save results to file
        results_file = f"strategy_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "signals": results
            }, f, indent=2)
            
        logger.info(f"Strategy results saved to {results_file}")
        return results

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Crypto Trading Strategy")
    parser.add_argument("--symbol", type=str, help="Symbol to analyze (default: all configured symbols)")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data to analyze")
    args = parser.parse_args()
    
    strategy = TradingStrategy()
    
    if args.symbol:
        # Run for a single symbol
        signals = strategy.generate_signals(args.symbol)
        if signals is not None:
            strategy.plot_signals(args.symbol)
    else:
        # Run for all configured symbols
        results = strategy.run_strategy()
        
        # Print a summary of results
        print("\nTrading Strategy Results:")
        print("=" * 50)
        for symbol, data in results.items():
            if "error" in data:
                print(f"{symbol}: {data['error']}")
            else:
                signal_type = "BUY" if data["signal"] == 1 else "SELL" if data["signal"] == -1 else "NEUTRAL"
                print(f"{symbol} @ ${data['price']:.2f}: {signal_type} (Strength: {data['signal_strength']:.2f})")
                if data["signal"] != 0:
                    print(f"  Position Size: {data['position_size']:.1%} of capital")
                    print(f"  Stop Loss: ${data['stop_loss']:.2f}, Take Profit: ${data['take_profit']:.2f}")
        print("\nDetailed results and charts saved to the 'charts' directory.")