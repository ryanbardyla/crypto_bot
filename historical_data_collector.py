# historical_data_collector.py
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from multi_api_price_fetcher import CryptoPriceFetcher

class HistoricalDataCollector:
    """Collect historical cryptocurrency price data for backtesting"""
    
    def __init__(self, save_to_file=True):
        """Initialize the data collector"""
        self.save_to_file = save_to_file
        self.price_fetcher = CryptoPriceFetcher(save_data=save_to_file)
        self.history_file = "price_history.json"
        
        # Create charts directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
    
    def load_history(self):
        """Load existing price history"""
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def save_history(self, history):
        """Save price history to file"""
        if self.save_to_file:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
            print(f"Saved price history to {self.history_file}")
    
    def fetch_historical_data_coingecko(self, symbol, days=30):
        """
        Fetch historical data from CoinGecko API
        Returns DataFrame with timestamp and price
        """
        # Map common symbols to CoinGecko IDs
        symbol_map = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "SOL": "solana",
            "DOGE": "dogecoin",
            "XRP": "ripple"
        }
        
        if symbol not in symbol_map:
            print(f"Symbol {symbol} not supported for historical data")
            return None
        
        coin_id = symbol_map[symbol]
        
        # CoinGecko API URL for historical market data
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'  # Use hourly for more granular data
        }
        
        try:
            print(f"Fetching {days} days of historical data for {symbol} from CoinGecko...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # CoinGecko returns prices as [timestamp, price] pairs
            if 'prices' in data and len(data['prices']) > 0:
                # Convert to DataFrame
                df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                
                # Convert millisecond timestamps to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                print(f"Successfully fetched {len(df)} data points")
                return df
            else:
                print("No price data found in response")
                return None
                
        except requests.RequestException as e:
            print(f"Error fetching historical data: {e}")
            return None
    
    def fetch_historical_data_alternative(self, symbol, days=30):
        """
        Alternative method to fetch historical data if CoinGecko fails
        Uses Coinbase API for recent daily prices
        """
        # Map symbols if needed
        symbol_map = {
            "BTC": "BTC",
            "ETH": "ETH",
            "SOL": "SOL",
            "DOGE": "DOGE",
            "XRP": "XRP"
        }
        
        if symbol not in symbol_map:
            print(f"Symbol {symbol} not supported for historical data")
            return None
        
        api_symbol = symbol_map[symbol]
        
        # Calculate start and end dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for API
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Coinbase API URL for historical prices
        url = f"https://api.coinbase.com/v2/prices/{api_symbol}-USD/spot/price"
        
        # We'll collect daily data by making requests with different dates
        prices = []
        current_date = start_date
        
        print(f"Fetching historical data for {symbol} using alternative method...")
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            try:
                # Add date parameter to URL
                params = {'date': date_str}
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and 'amount' in data['data']:
                        price = float(data['data']['amount'])
                        prices.append([current_date, price])
                        print(f"Got price for {date_str}: ${price}")
                    else:
                        print(f"No price data for {date_str}")
                else:
                    print(f"Failed to get price for {date_str}: {response.status_code}")
                
                # Sleep to avoid rate limiting
                time.sleep(0.5)
                
            except requests.RequestException as e:
                print(f"Error for {date_str}: {e}")
            
            # Move to next day
            current_date += timedelta(days=1)
        
        if prices:
            # Convert to DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            print(f"Successfully fetched {len(df)} data points")
            return df
        else:
            print("Failed to fetch any historical data")
            return None
    
    def generate_synthetic_data(self, symbol, days=30, volatility=0.02, trend=0.001):
        """
        Generate synthetic price data for testing when APIs fail
        This creates fake but somewhat realistic price movements
        """
        print(f"Generating {days} days of synthetic data for {symbol}...")
        
        # Set starting price based on symbol
        starting_prices = {
            "BTC": 90000,
            "ETH": 2200,
            "SOL": 145,
            "DOGE": 0.12,
            "XRP": 0.55
        }
        
        start_price = starting_prices.get(symbol, 1000)
        
        # Generate dates and initialize prices
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate random price movements
        prices = [start_price]
        for i in range(1, len(date_range)):
            # Random component (volatility)
            random_change = np.random.normal(0, volatility)
            
            # Trend component (slight upward or downward bias)
            trend_change = trend
            
            # Calculate new price
            last_price = prices[-1]
            new_price = last_price * (1 + random_change + trend_change)
            
            # Ensure price doesn't go negative
            prices.append(max(0.01, new_price))
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': date_range,
            'price': prices
        })
        
        print(f"Generated {len(df)} synthetic data points")
        return df
    
    def update_price_history(self, symbol, df):
        """
        Update price history with new data points
        Integrates with the existing price history format
        """
        history = self.load_history()
        
        # Initialize entry for this symbol if needed
        if symbol not in history:
            history[symbol] = []
        
        # Convert DataFrame rows to history format
        for _, row in df.iterrows():
            timestamp = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            price = float(row['price'])
            
            # Add to history if this timestamp doesn't exist
            existing_timestamps = [entry['timestamp'] for entry in history[symbol]]
            if timestamp not in existing_timestamps:
                history[symbol].append({
                    'timestamp': timestamp,
                    'price': price,
                    'source': 'historical'
                })
        
        # Sort by timestamp
        history[symbol] = sorted(history[symbol], key=lambda x: x['timestamp'])
        
        # Save updated history
        self.save_history(history)
        
        return len(history[symbol])
    
    def collect_data(self, symbol, days=30, use_synthetic=False):
        """
        Main method to collect historical data from any available source
        Returns number of data points collected
        """
        if use_synthetic:
            df = self.generate_synthetic_data(symbol, days)
        else:
            # Try CoinGecko first
            df = self.fetch_historical_data_coingecko(symbol, days)
            
            # If CoinGecko fails, try alternative
            if df is None or len(df) < 10:
                print("CoinGecko data insufficient, trying alternative source...")
                df = self.fetch_historical_data_alternative(symbol, days)
                
                # If all APIs fail, use synthetic data
                if df is None or len(df) < 10:
                    print("API data collection failed, generating synthetic data...")
                    df = self.generate_synthetic_data(symbol, days)
        
        if df is not None and len(df) > 0:
            # Save to CSV for reference
            csv_file = f"data/{symbol}_historical_{days}d.csv"
            df.to_csv(csv_file, index=False)
            print(f"Saved raw data to {csv_file}")
            
            # Update price history
            num_points = self.update_price_history(symbol, df)
            print(f"Updated price history for {symbol}: {num_points} total data points")
            return num_points
        else:
            print("Failed to collect any data")
            return 0


# Simple demonstration
if __name__ == "__main__":
    print("Historical Cryptocurrency Data Collector")
    print("---------------------------------------")
    
    collector = HistoricalDataCollector()
    
    # Get user input
    symbol = input("Enter cryptocurrency symbol to collect data for (default: BTC): ").upper() or "BTC"
    days = input("Number of days of historical data to collect (default: 30): ")
    days = int(days) if days.isdigit() else 30
    
    # Ask about synthetic data
    use_synthetic = False
    synthetic_option = input("Use synthetic data instead of API? (y/n, default: n): ").lower()
    if synthetic_option in ['y', 'yes']:
        use_synthetic = True
    
    # Collect data
    num_points = collector.collect_data(symbol, days, use_synthetic)
    
    if num_points > 0:
        print(f"\nSuccessfully collected {num_points} data points for {symbol}")
        print(f"You can now run the backtester with enough historical data")
    else:
        print("\nFailed to collect enough data. Please try again later or use synthetic data.")