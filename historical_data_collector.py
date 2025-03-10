# historical_data_collector.py
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from multi_api_price_fetcher import CryptoPriceFetcher
from database_manager import DatabaseManager
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class PriceHistory(Base):
    __tablename__ = 'price_history'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    price = Column(Float, nullable=False)
    volume = Column(Float)
    high = Column(Float)
    low = Column(Float)
    open = Column(Float)
    close = Column(Float)
    source = Column(String(50))
    
    __table_args__ = (
        Index('idx_price_symbol_timestamp', symbol, timestamp),
    )
    
    def __repr__(self):
        return f"<PriceHistory(symbol='{self.symbol}', timestamp='{self.timestamp}', price={self.price})>"
    
class HistoricalDataCollector:
    """Collect historical cryptocurrency price data for backtesting"""
    
    def __init__(self, save_to_file=True):
        """Initialize the data collector"""
        self.save_to_file = save_to_file
        self.price_fetcher = CryptoPriceFetcher(save_data=save_to_file)
        self.history_file = "price_history.json"
        
        # CoinGecko API specific parameters
        self.coingecko_rate_limit_wait = 10  # seconds to wait between CoinGecko API calls
        self.coingecko_last_call = 0  # timestamp of last CoinGecko API call
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Map common symbols to CoinGecko IDs
        self.symbol_map = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "SOL": "solana",
            "DOGE": "dogecoin",
            "XRP": "ripple",
            "LINK": "chainlink",
            "DOT": "polkadot",
            "AVAX": "avalanche-2",
            "MATIC": "polygon",
            "ADA": "cardano",
            "BNB": "binancecoin",
            "SHIB": "shiba-inu",
            "LTC": "litecoin",
            "UNI": "uniswap",
            "ATOM": "cosmos"
        }
    
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
    
    def respect_rate_limit(self):
        """Respect CoinGecko API rate limits to avoid getting blocked"""
        current_time = time.time()
        time_since_last_call = current_time - self.coingecko_last_call
        
        if time_since_last_call < self.coingecko_rate_limit_wait:
            wait_time = self.coingecko_rate_limit_wait - time_since_last_call
            print(f"Waiting {wait_time:.2f}s to respect API rate limits...")
            time.sleep(wait_time)
        
        self.coingecko_last_call = time.time()
    
    def fetch_historical_data_coingecko(self, symbol, days=30):
        """
        Fetch historical data from CoinGecko API
        Returns DataFrame with timestamp and price
        """
        if symbol not in self.symbol_map:
            print(f"Symbol {symbol} not supported for historical data")
            return None
        
        coin_id = self.symbol_map[symbol]
        
        # CoinGecko has limitations on the 'days' parameter:
        # - For 'daily' interval, max is 365 days per request
        # - For no specified interval, you can get full history but granularity varies
        
        # If days > 365, we'll make multiple requests and combine
        if days <= 365:
            return self._fetch_coingecko_single_period(coin_id, symbol, days)
        else:
            # Split into multiple requests
            all_data = []
            remaining_days = days
            
            # First get the most recent 365 days
            df_recent = self._fetch_coingecko_single_period(coin_id, symbol, 365)
            if df_recent is not None:
                all_data.append(df_recent)
                remaining_days -= 365
            
            # Then get remaining data in max chunks
            if remaining_days > 0:
                print(f"Fetching additional {remaining_days} days of historical data...")
                # For remaining history, use the 'max' parameter 
                # Note: This gives all available history, granularity varies
                df_older = self._fetch_coingecko_max_history(coin_id, symbol)
                
                if df_older is not None:
                    # Filter to only keep the older portion we need
                    if len(all_data) > 0 and len(all_data[0]) > 0:
                        oldest_date_in_recent = all_data[0]['timestamp'].min()
                        df_older = df_older[df_older['timestamp'] < oldest_date_in_recent]
                        
                        # Further filter to only keep the days we need
                        cutoff_date = datetime.now() - timedelta(days=days)
                        df_older = df_older[df_older['timestamp'] >= cutoff_date]
                    
                    all_data.append(df_older)
            
            # Combine all dataframes
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                return combined_df
            else:
                return None
    
    def _fetch_coingecko_single_period(self, coin_id, symbol, days):
        """Helper method to fetch a single period from CoinGecko"""
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        
        try:
            print(f"Fetching {days} days of historical data for {symbol} from CoinGecko...")
            self.respect_rate_limit()
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 429:
                print("Rate limit exceeded. Waiting 60 seconds before retrying...")
                time.sleep(60)
                self.respect_rate_limit()
                response = requests.get(url, params=params, timeout=15)
            
            response.raise_for_status()
            
            data = response.json()
            
            if 'prices' in data and len(data['prices']) > 0:
                df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                print(f"Successfully fetched {len(df)} data points")
                return df
            else:
                print("No price data found in response")
                return None
                
        except requests.RequestException as e:
            print(f"Error fetching historical data: {e}")
            print(f"Response: {response.text if 'response' in locals() else 'No response'}")
            return None
    
    def _fetch_coingecko_max_history(self, coin_id, symbol):
        """Helper method to fetch maximum available history from CoinGecko"""
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        
        params = {
            'vs_currency': 'usd',
            'days': 'max',  # This gets all available history
        }
        
        try:
            print(f"Fetching maximum available history for {symbol} from CoinGecko...")
            self.respect_rate_limit()
            response = requests.get(url, params=params, timeout=20)
            
            if response.status_code == 429:
                print("Rate limit exceeded. Waiting 60 seconds before retrying...")
                time.sleep(60)
                self.respect_rate_limit()
                response = requests.get(url, params=params, timeout=20)
                
            response.raise_for_status()
            
            data = response.json()
            
            if 'prices' in data and len(data['prices']) > 0:
                df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                print(f"Successfully fetched {len(df)} historical data points")
                return df
            else:
                print("No historical price data found in response")
                return None
                
        except requests.RequestException as e:
            print(f"Error fetching max historical data: {e}")
            print(f"Response: {response.text if 'response' in locals() else 'No response'}")
            return None
    
    def fetch_historical_data_alternative(self, symbol, days=30):
        """
        Alternative method to fetch historical data if CoinGecko fails
        Uses Coinbase API for recent daily prices
        """
        # No need to remap symbols for Coinbase
        api_symbol = symbol
        
        # Calculate start and end dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # For Coinbase, we'll use a different approach with a single request
        # for recent data due to rate limits
        if days > 300:
            print(f"Coinbase alternative method works best for shorter timeframes. Limiting to 300 days.")
            days = 300
            start_date = end_date - timedelta(days=days)
        
        # Coinbase API URL for historical prices - using the candles endpoint
        url = f"https://api.exchange.coinbase.com/products/{api_symbol}-USD/candles"
        
        # Parameters - note that Coinbase uses seconds for timestamps
        params = {
            'granularity': 86400,  # Daily candles (86400 seconds in a day)
            'start': start_date.isoformat(),
            'end': end_date.isoformat()
        }
        
        try:
            print(f"Fetching historical data for {symbol} using Coinbase API...")
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and isinstance(data, list):
                    # Coinbase format: [timestamp, low, high, open, close, volume]
                    prices = []
                    for candle in data:
                        timestamp = datetime.fromtimestamp(candle[0])
                        price = candle[4]  # Close price
                        prices.append([timestamp, price])
                    
                    if prices:
                        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                        df = df.sort_values('timestamp')  # Ensure chronological order
                        print(f"Successfully fetched {len(df)} data points from Coinbase")
                        return df
                    else:
                        print("No usable price data from Coinbase")
                else:
                    print("Invalid response format from Coinbase")
            else:
                print(f"Failed to get data from Coinbase: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.RequestException as e:
            print(f"Error fetching from Coinbase: {e}")
        
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
            "XRP": 0.55,
            "LINK": 14.50,
            "DOT": 7.20,
            "AVAX": 32.50,
            "MATIC": 0.85,
            "ADA": 0.45
        }
        
        start_price = starting_prices.get(symbol, 1000)
        
        # Generate dates and initialize prices
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate random price movements with more realistic behavior
        prices = [start_price]
        
        # We'll use a simple random walk with memory (mean reversion)
        for i in range(1, len(date_range)):
            # Random component (volatility)
            random_change = np.random.normal(0, volatility)
            
            # Trend component (slight upward or downward bias)
            trend_change = trend
            
            # Mean reversion component
            # If price has gone up a lot recently, it's more likely to go down
            reversion_strength = 0.01
            price_deviation = prices[-1] / start_price - 1  # How far from starting price
            reversion = -price_deviation * reversion_strength  # Pull back toward mean
            
            # Market cycles - add some cyclical behavior
            cycle_phase = i % 100  # 100-day cycle
            cycle_component = 0.0005 * np.sin(2 * np.pi * cycle_phase / 100)
            
            # Occasional "big" moves (market shocks)
            shock = 0
            if np.random.random() < 0.01:  # 1% chance each day
                shock = np.random.normal(0, volatility * 5)  # 5x normal volatility
            
            # Calculate new price with all components
            last_price = prices[-1]
            new_price = last_price * (1 + random_change + trend_change + reversion + cycle_component + shock)
            
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
    
        db_manager = DatabaseManager(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=os.environ.get("POSTGRES_PORT", "5432"),
            database=os.environ.get("POSTGRES_DB", "trading_db"),
            user=os.environ.get("POSTGRES_USER", "bot_user"),
            password=os.environ.get("POSTGRES_PASSWORD", "secure_password")
        )
    
        session = db_manager.get_session()
        records_added = 0
    
        try:
            for _, row in df.iterrows():
                # Convert timestamp to datetime if it's a string
                if isinstance(row['timestamp'], str):
                    timestamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
                else:
                    timestamp = row['timestamp']
                
                price = float(row['price'])
                
                # Check if entry exists
                existing = session.query(PriceHistory).filter_by(
                    symbol=symbol, 
                    timestamp=timestamp
                ).first()
                
                if existing:
                    # Update existing record
                    existing.price = price
                    if 'volume' in row:
                        existing.volume = float(row.get('volume', 0))
                    if 'high' in row:
                        existing.high = float(row.get('high', price))
                    if 'low' in row:
                        existing.low = float(row.get('low', price))
                    if 'open' in row:
                        existing.open = float(row.get('open', price))
                    if 'close' in row:
                        existing.close = float(row.get('close', price))
                    existing.source = 'historical'
                else:
                    # Create new record
                    record = PriceHistory(
                        symbol=symbol,
                        timestamp=timestamp,
                        price=price,
                        volume=float(row.get('volume', 0)) if 'volume' in row else None,
                        high=float(row.get('high', price)) if 'high' in row else None,
                        low=float(row.get('low', price)) if 'low' in row else None,
                        open=float(row.get('open', price)) if 'open' in row else None,
                        close=float(row.get('close', price)) if 'close' in row else None,
                        source='historical'
                    )
                    session.add(record)
                    records_added += 1
                    
            session.commit()
            print(f"Updated price history for {symbol} in database: {records_added} new records")
            return records_added
        except Exception as e:
            session.rollback()
            print(f"Error updating database: {e}")
            return 0
        finally:
            session.close()
    
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
            if df is None or len(df) < days * 0.5:  # If we got less than 50% of requested days
                print("CoinGecko data insufficient, trying alternative source...")
                df_alt = self.fetch_historical_data_alternative(symbol, days)
                
                # Combine results if we got data from both sources
                if df is not None and df_alt is not None:
                    # Combine and remove duplicates
                    combined = pd.concat([df, df_alt]).drop_duplicates('timestamp').sort_values('timestamp')
                    print(f"Combined data from multiple sources: {len(combined)} data points")
                    df = combined
                elif df_alt is not None:
                    df = df_alt
                
                # If all APIs fail, use synthetic data
                if df is None or len(df) < days * 0.3:  # If we got less than 30% of requested days
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
            
    def collect_multi_data(self, symbols, days=30, use_synthetic=False, max_workers=3):
        """
        Collect data for multiple symbols in parallel
        Limits parallel requests to avoid rate limiting issues
        """
        results = {}
        
        # Use ThreadPoolExecutor for parallel processing with a limit on concurrent workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_symbol = {
                executor.submit(self.collect_data, symbol, days, use_synthetic): symbol
                for symbol in symbols
            }
            
            # Process results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    num_points = future.result()
                    results[symbol] = num_points
                except Exception as e:
                    print(f"Error collecting data for {symbol}: {str(e)}")
                    results[symbol] = 0
        
        return results


# Enhanced command-line interface
if __name__ == "__main__":
    print("Enhanced Historical Cryptocurrency Data Collector")
    print("-----------------------------------------------")
    
    parser = argparse.ArgumentParser(description="Collect historical cryptocurrency price data")
    parser.add_argument("--symbols", type=str, default="BTC", 
                        help="Comma-separated list of cryptocurrency symbols (default: BTC)")
    parser.add_argument("--days", type=int, default=720, 
                        help="Number of days of historical data to collect (default: 720)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of API data")
    parser.add_argument("--all", action="store_true",
                        help="Collect data for all supported cryptocurrencies")
    
    args = parser.parse_args()
    
    collector = HistoricalDataCollector()
    
    # Determine which symbols to collect
    if args.all:
        symbols = list(collector.symbol_map.keys())
        print(f"Collecting data for all {len(symbols)} supported cryptocurrencies")
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
        print(f"Collecting data for {len(symbols)} cryptocurrencies: {', '.join(symbols)}")
    
    # Check for unsupported symbols
    unsupported = [s for s in symbols if s not in collector.symbol_map and not args.synthetic]
    if unsupported and not args.synthetic:
        print(f"Warning: The following symbols are not supported by our API: {', '.join(unsupported)}")
        print("They will use synthetic data instead of real historical data.")
    
    # Confirm before starting a large data collection
    if args.days > 90 and len(symbols) > 3:
        confirm = input(f"You're about to collect {args.days} days of data for {len(symbols)} symbols. This may take a while. Continue? (y/n): ")
        if confirm.lower() not in ('y', 'yes'):
            print("Operation cancelled.")
            exit()
    
    # Collect data
    print(f"\nCollecting {args.days} days of historical data...")
    
    # Start the timer
    start_time = time.time()
    
    results = collector.collect_multi_data(symbols, args.days, args.synthetic, max_workers=2)
    
    # Calculate time elapsed
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n=== Data Collection Summary ===")
    total_points = sum(results.values())
    success_count = sum(1 for count in results.values() if count > 0)
    
    print(f"Time elapsed: {elapsed_time:.1f} seconds")
    print(f"Total data points collected: {total_points}")
    print(f"Successful symbols: {success_count}/{len(symbols)}")
    
    print("\nSymbol-specific results:")
    for symbol, count in results.items():
        status = "✅ Success" if count > 0 else "❌ Failed"
        print(f"  {symbol}: {count} data points - {status}")
    
    print("\nData Collection complete!")
    print("Historical data is now available for backtesting and analysis.")