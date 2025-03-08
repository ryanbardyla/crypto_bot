# multi_api_price_fetcher.py
import requests
import time
import json
import socket
from datetime import datetime

class CryptoPriceFetcher:
    """Fetch cryptocurrency prices from multiple APIs with fallback"""
    
    def __init__(self, save_data=True):
        self.save_data = save_data
        self.data_file = "price_history.json"
        self.price_history = self._load_history()
        
        # API configuration - multiple sources for redundancy
        self.apis = [
            {
                "name": "CoinGecko",
                "url": "https://api.coingecko.com/api/v3/simple/price?ids={id}&vs_currencies=usd",
                "mapping": {  # Map common symbols to their CoinGecko IDs
                    "BTC": "bitcoin",
                    "ETH": "ethereum",
                    "SOL": "solana",
                    "DOGE": "dogecoin",
                    "XRP": "ripple"
                },
                "extract": lambda data, id: data[id]["usd"]
            },
            {
                "name": "Coinbase",
                "url": "https://api.coinbase.com/v2/prices/{id}-USD/spot",
                "mapping": {  # Map common symbols as they appear in Coinbase API
                    "BTC": "BTC",
                    "ETH": "ETH",
                    "SOL": "SOL",
                    "DOGE": "DOGE",
                    "XRP": "XRP"
                },
                "extract": lambda data, id: float(data["data"]["amount"])
            }
        ]
        
        # Mock prices for testing when all APIs fail
        self.mock_prices = {
            "BTC": 48500.25,
            "ETH": 3250.75,
            "SOL": 120.50,
            "DOGE": 0.12,
            "XRP": 0.57
        }
    
    def _load_history(self):
        """Load existing price history if available"""
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_history(self):
        """Save price history to file"""
        if self.save_data:
            with open(self.data_file, 'w') as f:
                json.dump(self.price_history, f, indent=2)
    
    def _store_price(self, symbol, price, source="unknown"):
        """Store a price in history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            "timestamp": timestamp,
            "price": price,
            "source": source
        })
        
        self._save_history()
    
    def _is_dns_working(self, hostname):
        """Check if DNS resolution works for a given hostname"""
        try:
            socket.gethostbyname(hostname)
            return True
        except socket.gaierror:
            return False
    
    def _get_mock_price(self, symbol):
        """Return mock price data for testing"""
        if symbol in self.mock_prices:
            price = self.mock_prices[symbol]
            self._store_price(symbol, price, source="mock_data")
            print(f"Using mock data for {symbol}: ${price}")
            return price
        else:
            print(f"No mock data available for {symbol}")
            return None
    
    def get_price(self, symbol="BTC", use_mock=False):
        """Get current price for a cryptocurrency from any available API"""
        if use_mock:
            return self._get_mock_price(symbol)
        
        # Try each API in sequence until one works
        for api in self.apis:
            try:
                # Check if this API supports the requested symbol
                if symbol not in api["mapping"]:
                    print(f"{api['name']} API doesn't support {symbol}, skipping...")
                    continue
                
                # Extract domain from URL for DNS check
                api_domain = api["url"].split("//")[1].split("/")[0]
                
                # Check DNS resolution before making the request
                if not self._is_dns_working(api_domain):
                    print(f"DNS resolution failed for {api_domain}, skipping {api['name']} API...")
                    continue
                
                # Prepare the URL with the mapped ID
                crypto_id = api["mapping"][symbol]
                url = api["url"].format(id=crypto_id)
                
                # Make the request
                print(f"Trying {api['name']} API...")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Extract the price using the API-specific extraction function
                data = response.json()
                price = api["extract"](data, crypto_id)
                
                # Store successful result
                self._store_price(symbol, price, source=api["name"])
                print(f"Successfully fetched {symbol} price from {api['name']}")
                return price
                
            except requests.exceptions.RequestException as e:
                print(f"Error with {api['name']} API: {str(e)}")
            except (KeyError, ValueError, TypeError) as e:
                print(f"Error parsing data from {api['name']} API: {str(e)}")
            except Exception as e:
                print(f"Unexpected error with {api['name']} API: {str(e)}")
        
        # If all APIs fail, fall back to mock data
        print("All APIs failed, falling back to mock data.")
        return self._get_mock_price(symbol)
    
    def print_latest_prices(self):
        """Print the latest price for each tracked symbol"""
        if not self.price_history:
            print("No price data available yet.")
            return
            
        for symbol, history in self.price_history.items():
            if history:
                latest = history[-1]
                print(f"{symbol}: ${latest['price']:.2f} at {latest['timestamp']} (Source: {latest.get('source', 'unknown')})")
            else:
                print(f"{symbol}: No price data available")
    
    def get_price_history(self, symbol):
        """Return the price history for a symbol"""
        return self.price_history.get(symbol, [])


# Simple demonstration
if __name__ == "__main__":
    print("Multi-API Cryptocurrency Price Fetcher")
    print("--------------------------------------")
    
    # Allow user to decide if they want to use mock data
    use_mock = False
    response = input("Do you want to use mock data for testing? (y/n, default=n): ").lower()
    if response in ['y', 'yes']:
        use_mock = True
        print("Using mock data for this session.")
    
    fetcher = CryptoPriceFetcher()
    
    # List of cryptocurrencies to fetch
    cryptos = ["BTC", "ETH", "SOL", "DOGE"]
    
    print("\nFetching current prices...")
    for crypto in cryptos:
        price = fetcher.get_price(crypto, use_mock=use_mock)
        if price is not None:
            print(f"Current {crypto} price: ${price:.2f}")
        else:
            print(f"Could not retrieve {crypto} price from any source.")
    
    # Print summary of all prices
    print("\nSummary of all prices:")
    fetcher.print_latest_prices()