# crypto_analyzer.py
import json
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import os
from multi_api_price_fetcher import CryptoPriceFetcher

class CryptoAnalyzer:
    """Analyze cryptocurrency price data and generate basic signals"""
    
    def __init__(self, price_fetcher=None):
        """Initialize with an existing price fetcher or create a new one"""
        if price_fetcher is None:
            self.price_fetcher = CryptoPriceFetcher()
        else:
            self.price_fetcher = price_fetcher
    
    def fetch_latest_prices(self, symbols=None):
        """Fetch latest prices for all specified symbols"""
        if symbols is None:
            symbols = ["BTC", "ETH", "SOL", "DOGE"]
        
        results = {}
        for symbol in symbols:
            price = self.price_fetcher.get_price(symbol)
            if price is not None:
                results[symbol] = price
        
        return results
    
    def convert_history_to_dataframe(self, symbol):
        """Convert price history for a symbol to pandas DataFrame"""
        history = self.price_fetcher.get_price_history(symbol)
        
        if not history:
            print(f"No price history available for {symbol}")
            return None
        
        # Convert history to DataFrame
        df = pd.DataFrame(history)
        
        # Convert timestamp strings to datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        return df
    
    def calculate_basic_indicators(self, symbol):
        """Calculate basic technical indicators for a symbol"""
        df = self.convert_history_to_dataframe(symbol)
        
        if df is None or len(df) < 2:
            print(f"Not enough data to calculate indicators for {symbol}")
            return None
            
        # Make a copy to avoid SettingWithCopyWarning
        result = df.copy()
        
        # Calculate percentage change
        result['pct_change'] = result['price'].pct_change() * 100
        
        # Calculate simple moving averages if we have enough data
        if len(result) >= 3:
            result['sma_3'] = result['price'].rolling(window=3).mean()
        
        if len(result) >= 5:
            result['sma_5'] = result['price'].rolling(window=5).mean()
        
        # Calculate volatility (standard deviation over recent periods)
        if len(result) >= 5:
            result['volatility'] = result['price'].rolling(window=5).std()
        
        return result
    
    def get_simple_signals(self, symbol):
        """Generate simple trading signals based on indicators"""
        df = self.calculate_basic_indicators(symbol)
        
        if df is None or len(df) < 2:  # Need at least some data points
            return {"status": "Not enough data for signals", "symbol": symbol}
        
        signals = {
            "symbol": symbol,
            "current_price": 0,
            "signals": []
        }
        
        try:
            latest = df.iloc[-1]
            signals["current_price"] = latest['price']
            
            # Make sure we have at least 2 data points
            if len(df) >= 2:
                prev = df.iloc[-2]
            else:
                prev = latest  # Use latest as fallback if only one point
        
                    # Check for price crossing above/below moving averages
            if 'sma_5' in latest and 'sma_5' in prev:
                if latest['price'] > latest['sma_5'] and prev['price'] <= prev['sma_5']:
                    signals["signals"].append("BULLISH: Price crossed above 5-period SMA")
                elif latest['price'] < latest['sma_5'] and prev['price'] >= prev['sma_5']:
                    signals["signals"].append("BEARISH: Price crossed below 5-period SMA")
            
            # Check for significant price movements
            if 'pct_change' in latest:
                if latest['pct_change'] > 3:
                    signals["signals"].append(f"ALERT: Large price increase ({latest['pct_change']:.2f}%)")
                elif latest['pct_change'] < -3:
                    signals["signals"].append(f"ALERT: Large price decrease ({latest['pct_change']:.2f}%)")
            
            # Add simple trend analysis
            if len(df) >= 5:
                recent_prices = df['price'].tail(5)
                if recent_prices.is_monotonic_increasing:
                    signals["signals"].append("BULLISH: Price consistently increasing over last 5 periods")
                elif recent_prices.is_monotonic_decreasing:
                    signals["signals"].append("BEARISH: Price consistently decreasing over last 5 periods")
            
            # If no specific signals, add a neutral message
            if not signals["signals"]:
                signals["signals"].append("NEUTRAL: No clear signals detected")
        
        except Exception as e:
            print(f"Error generating signals for {symbol}: {e}")
            signals["signals"].append(f"ERROR: {str(e)}")
            
        return signals
    
    def plot_price_history(self, symbol, save_to_file=None):
        """Plot price history with indicators"""
        df = self.calculate_basic_indicators(symbol)
        
        if df is None or len(df) < 2:
            print(f"Not enough data to plot for {symbol}")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot price
        plt.plot(df.index, df['price'], label='Price', color='blue', linewidth=2)
        
        # Plot moving averages if available
        if 'sma_3' in df.columns:
            plt.plot(df.index, df['sma_3'], label='3-period SMA', color='red', linestyle='--')
        
        if 'sma_5' in df.columns:
            plt.plot(df.index, df['sma_5'], label='5-period SMA', color='green', linestyle='--')
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.title(f'{symbol} Price History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Rotate date labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save to file if requested
        if save_to_file:
            plt.savefig(save_to_file)
            print(f"Chart saved to {save_to_file}")
        else:
            plt.show()
    
    def generate_summary_report(self, symbols=None):
        """Generate a summary report for multiple cryptocurrencies"""
        if symbols is None:
            symbols = ["BTC", "ETH", "SOL", "DOGE"]
        
        report = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbols_analyzed": symbols,
            "price_data": {},
            "signals": {}
        }
        
        for symbol in symbols:
            # Get latest price - reuse existing prices when possible to avoid rate limits
            try:
                # First check if we already have this price in memory
                df = self.convert_history_to_dataframe(symbol)
                if df is not None and not df.empty:
                    price = df['price'].iloc[-1]
                    print(f"Using cached price for {symbol}: ${price:.2f}")
                else:
                    # If not, fetch a new price
                    price = self.price_fetcher.get_price(symbol)
                
                if price is not None:
                    report["price_data"][symbol] = price
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")
            
            # Get signals
            try:
                signals = self.get_simple_signals(symbol)
                if signals:
                    # Handle both formats of signal responses
                    if "signals" in signals:
                        report["signals"][symbol] = signals["signals"]
                    elif "status" in signals:
                        report["signals"][symbol] = [signals["status"]]
            except Exception as e:
                print(f"Error generating signals for {symbol}: {e}")
                report["signals"][symbol] = [f"Error: {str(e)}"]
        
        return report


# Simple demonstration
if __name__ == "__main__":
    print("Cryptocurrency Price Analyzer")
    print("-----------------------------")
    
    # Create price fetcher and analyzer
    fetcher = CryptoPriceFetcher()
    analyzer = CryptoAnalyzer(fetcher)
    
    # List of cryptocurrencies to analyze
    cryptos = ["BTC", "ETH", "SOL", "DOGE"]
    
    # Fetch current prices first to ensure we have some data
    print("\nFetching current prices...")
    latest_prices = analyzer.fetch_latest_prices(cryptos)
    
            # Generate and print signals
    print("\nGenerating trading signals...")
    for symbol in cryptos:
        if symbol in latest_prices:
            try:
                signals = analyzer.get_simple_signals(symbol)
                print(f"\n{symbol} Analysis:")
                print(f"Current Price: ${latest_prices[symbol]:.2f}")
                
                if "signals" in signals:
                    print("Signals:")
                    for signal in signals["signals"]:
                        print(f"  - {signal}")
                elif "status" in signals:
                    print(f"Status: {signals['status']}")
            except Exception as e:
                print(f"\n{symbol} Analysis Error: {e}")
    
    # Ask if user wants to see charts
    create_charts = input("\nDo you want to generate price charts? (y/n): ").lower()
    if create_charts in ['y', 'yes']:
        # Create charts directory if it doesn't exist
        os.makedirs("charts", exist_ok=True)
        
        for symbol in cryptos:
            if symbol in latest_prices:
                print(f"Generating chart for {symbol}...")
                analyzer.plot_price_history(symbol, save_to_file=f"charts/{symbol}_price_chart.png")
    
            # Generate summary report
    print("\nGenerating summary report...")
    try:
        report = analyzer.generate_summary_report(cryptos)
    except Exception as e:
        print(f"Error generating report: {e}")
        report = {
            "error": str(e),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Save report to file
    with open("crypto_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("Analysis complete! Report saved to crypto_analysis_report.json")
    
    if create_charts in ['y', 'yes']:
        print("Charts saved to the 'charts' directory")