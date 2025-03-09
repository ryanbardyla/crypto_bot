# crypto_analyzer.py (updated with concurrent processing)

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the centralized logging configuration
from utils.logging_config import get_module_logger

# Get logger for this module
logger = get_module_logger(__name__)

class CryptoAnalyzer:
    def __init__(self, price_fetcher=None):
        if price_fetcher is None:
            from multi_api_price_fetcher import CryptoPriceFetcher
            self.price_fetcher = CryptoPriceFetcher()
        else:
            self.price_fetcher = price_fetcher
        self.max_workers = 4  # Adjust based on your system capabilities
            
    def fetch_latest_prices(self, symbols=None):
        """Fetch latest prices for multiple symbols concurrently"""
        if symbols is None:
            symbols = ["BTC", "ETH", "SOL", "DOGE"]
            
        prices = {}
        
        # Use ThreadPoolExecutor for concurrent API requests
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_symbol = {executor.submit(self.price_fetcher.get_price, symbol): symbol for symbol in symbols}
            
            # Process results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    price = future.result()
                    if price is not None:
                        prices[symbol] = price
                        logger.info(f"Fetched price for {symbol}: ${price:.2f}")
                    else:
                        logger.warning(f"Could not get price for {symbol}")
                except Exception as e:
                    logger.error(f"Error getting price for {symbol}: {e}")
                    
        return prices
    
    def convert_history_to_dataframe(self, symbol):
        """Convert price history to pandas DataFrame"""
        try:
            history = self.price_fetcher.get_price_history(symbol)
            if not history or len(history) < 2:
                logger.warning(f"No price history available for {symbol}")
                return None
                
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            return df
        except Exception as e:
            logger.error(f"Error converting history to dataframe: {e}")
            return None
    
    def calculate_basic_indicators(self, symbol):
        """Calculate basic technical indicators for a symbol"""
        df = self.convert_history_to_dataframe(symbol)
        if df is None or len(df) < 2:
            logger.warning(f"Not enough data to calculate indicators for {symbol}")
            return None
            
        result = df.copy()
        result['pct_change'] = result['price'].pct_change() * 100
        
        # Calculate moving averages if enough data points
        if len(result) >= 3:
            result['sma_3'] = result['price'].rolling(window=3).mean()
        if len(result) >= 5:
            result['sma_5'] = result['price'].rolling(window=5).mean()
            result['volatility'] = result['price'].rolling(window=5).std()
        if len(result) >= 14:
            result['sma_14'] = result['price'].rolling(window=14).mean()
            
        # Add Bollinger Bands if enough data
        if len(result) >= 20:
            result['sma_20'] = result['price'].rolling(window=20).mean()
            result['upper_band'] = result['sma_20'] + (result['price'].rolling(window=20).std() * 2)
            result['lower_band'] = result['sma_20'] - (result['price'].rolling(window=20).std() * 2)
            
        return result
    
    def analyze_multiple_symbols(self, symbols=None):
        """Analyze multiple symbols concurrently"""
        if symbols is None:
            symbols = ["BTC", "ETH", "SOL", "DOGE"]
            
        results = {}
        
        # Use ThreadPoolExecutor for concurrent analysis
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_symbol = {executor.submit(self.calculate_basic_indicators, symbol): symbol for symbol in symbols}
            
            # Process results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    indicator_df = future.result()
                    if indicator_df is not None:
                        results[symbol] = indicator_df
                        logger.info(f"Analyzed {symbol}: {len(indicator_df)} data points")
                    else:
                        logger.warning(f"Could not analyze {symbol}")
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    
        return results
    
    def get_simple_signals(self, symbol):
        """Generate simple trading signals based on indicators"""
        df = self.calculate_basic_indicators(symbol)
        if df is None or len(df) < 2:  # Need at least some data points
            logger.warning(f"Not enough data for signal generation for {symbol}")
            return {
                "symbol": symbol,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "signals": ["ERROR: Insufficient data"],
                "status": "error"
            }
            
        signals = {
            "symbol": symbol,
            "date": df.index[-1].strftime("%Y-%m-%d"),
            "price": df['price'].iloc[-1],
            "signals": [],
            "status": "neutral"  # Default status
        }
            
        try:
            latest = df.iloc[-1]
            
            # Check for signals if we have enough data
            if len(df) >= 5:
                # Moving average crossover
                if 'sma_3' in df.columns and 'sma_5' in df.columns:
                    if df['sma_3'].iloc[-2] <= df['sma_5'].iloc[-2] and df['sma_3'].iloc[-1] > df['sma_5'].iloc[-1]:
                        signals["signals"].append("BULLISH: Price crossed above 5-period SMA")
                        signals["status"] = "bullish"
                    elif df['sma_3'].iloc[-2] >= df['sma_5'].iloc[-2] and df['sma_3'].iloc[-1] < df['sma_5'].iloc[-1]:
                        signals["signals"].append("BEARISH: Price crossed below 5-period SMA")
                        signals["status"] = "bearish"
                
                # Large price movements
                if 'pct_change' in latest and abs(latest['pct_change']) > 5:
                    if latest['pct_change'] > 0:
                        signals["signals"].append(f"ALERT: Large price increase ({latest['pct_change']:.2f}%)")
                    else:
                        signals["signals"].append(f"ALERT: Large price decrease ({latest['pct_change']:.2f}%)")
                        
                # Check recent price direction
                recent_prices = df['price'].tail(5)
                if all(recent_prices.diff().dropna() > 0):
                    signals["signals"].append("BULLISH: Price consistently increasing over last 5 periods")
                    signals["status"] = "bullish"
                elif all(recent_prices.diff().dropna() < 0):
                    signals["signals"].append("BEARISH: Price consistently decreasing over last 5 periods")
                    signals["status"] = "bearish"
                    
            # Add Bollinger Band signals if available
            if 'upper_band' in df.columns and 'lower_band' in df.columns:
                if df['price'].iloc[-1] > df['upper_band'].iloc[-1]:
                    signals["signals"].append("BEARISH: Price above upper Bollinger Band (overbought)")
                    signals["status"] = "bearish"
                elif df['price'].iloc[-1] < df['lower_band'].iloc[-1]:
                    signals["signals"].append("BULLISH: Price below lower Bollinger Band (oversold)")
                    signals["status"] = "bullish"
                
            # If no specific signals were found
            if not signals["signals"]:
                signals["signals"].append("NEUTRAL: No clear signals detected")
                
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            signals["signals"].append(f"ERROR: {str(e)}")
            signals["status"] = "error"
            
        return signals
    
    def get_signals_for_multiple_symbols(self, symbols=None):
        """Generate signals for multiple symbols concurrently"""
        if symbols is None:
            symbols = ["BTC", "ETH", "SOL", "DOGE"]
            
        results = {}
        
        # Use ThreadPoolExecutor for concurrent signal generation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_symbol = {executor.submit(self.get_simple_signals, symbol): symbol for symbol in symbols}
            
            # Process results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    signals = future.result()
                    results[symbol] = signals
                    logger.info(f"Generated signals for {symbol}: {signals['status']}")
                except Exception as e:
                    logger.error(f"Error generating signals for {symbol}: {e}")
                    results[symbol] = {
                        "symbol": symbol,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "signals": [f"Error: {str(e)}"],
                        "status": "error"
                    }
                    
        return results