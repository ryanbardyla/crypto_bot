# Add these imports to simple_backtester.py
import os
import gc
import json
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm  # For progress bars

# Import the necessary classes
from multi_api_price_fetcher import CryptoPriceFetcher
from crypto_analyzer import CryptoAnalyzer

class ParallelBacktester:
    def __init__(self, initial_capital=10000, chunk_size=1000, memory_threshold=80, max_workers=None):
        self.initial_capital = initial_capital
        self.chunk_size = chunk_size
        self.memory_threshold = memory_threshold
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)  # Default to CPU count - 1
        self.price_fetcher = CryptoPriceFetcher()
        self.analyzer = CryptoAnalyzer(self.price_fetcher)
        self.data_dir = "data"
        self.results_dir = "backtest_results"
        
        # Create necessary directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"Initialized ParallelBacktester with {self.max_workers} workers")
    
    def check_memory_usage(self, critical=False):
        """Monitor memory usage and take action if it exceeds threshold"""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent
        
        message = f"Memory usage: {usage_percent:.1f}% of total"
        if usage_percent > self.memory_threshold:
            message = f"WARNING: High memory usage ({usage_percent:.1f}%)"
            if critical:
                print(message)
                gc.collect()
                if usage_percent > 95:  # Critical threshold
                    raise MemoryError(f"Memory usage too high ({usage_percent:.1f}%). Stopping operation.")
        return usage_percent < self.memory_threshold
    
    def get_all_symbols(self):
        """Get all available symbols for backtesting"""
        symbols = []
        
        # Check price_history.json
        if os.path.exists("price_history.json"):
            try:
                with open("price_history.json", "r") as f:
                    price_history = json.load(f)
                    symbols.extend(list(price_history.keys()))
            except Exception as e:
                print(f"Error loading price_history.json: {e}")
        
        # Check data directory for CSV files
        if os.path.exists(self.data_dir):
            for file in os.listdir(self.data_dir):
                if file.endswith(".csv"):
                    # Try to extract symbol from filename patterns like "BTC_historical.csv"
                    parts = file.split("_")
                    if len(parts) > 0 and len(parts[0]) <= 5:  # Most crypto symbols are short
                        symbol = parts[0].upper()
                        if symbol not in symbols:
                            symbols.append(symbol)
        
        return sorted(list(set(symbols)))  # Remove duplicates and sort
    
    def backtest_symbol(self, symbol, strategy_func=None, days=None, file_path=None):
        """Backtest a single symbol with the specified strategy"""
        print(f"Starting backtest for {symbol}...")
        
        # Initialize a separate backtester instance for this symbol
        # This avoids any shared state issues
        from simple_backtester import SimpleBacktester
        backtester = SimpleBacktester(
            initial_capital=self.initial_capital,
            chunk_size=self.chunk_size,
            memory_threshold=self.memory_threshold
        )
        
        # Determine the file path if not provided
        if file_path is None:
            file_path = os.path.join(self.data_dir, f"{symbol}_historical.csv")
            if not os.path.exists(file_path):
                file_path = None  # Let backtester use price_history.json
        
        # Run the backtest
        try:
            result = backtester.backtest(symbol, strategy_func, file_path)
            
            if result:
                # Save the results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_file = os.path.join(self.results_dir, f"{symbol}_backtest_{timestamp}.json")
                with open(result_file, "w") as f:
                    json.dump({
                        "symbol": symbol,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "metrics": result,
                        "trades_count": len(backtester.trades)
                    }, f, indent=2, default=str)
                
                # Generate chart if requested
                chart_file = os.path.join(self.results_dir, f"{symbol}_backtest_{timestamp}.png")
                backtester.plot_performance(symbol, save_to_file=chart_file)
                
                return {
                    "symbol": symbol,
                    "success": True,
                    "metrics": result,
                    "chart_file": chart_file,
                    "result_file": result_file
                }
            else:
                return {
                    "symbol": symbol,
                    "success": False,
                    "error": "Backtest returned no results"
                }
        except Exception as e:
            print(f"Error backtesting {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "success": False,
                "error": str(e)
            }
    
    def backtest_multiple(self, symbols=None, strategy_func=None, days=None, parallel=True):
        """Backtest multiple symbols in parallel or sequentially"""
        if symbols is None:
            symbols = self.get_all_symbols()
            
        if not symbols:
            print("No symbols found for backtesting")
            return {}
            
        print(f"Backtesting {len(symbols)} symbols: {', '.join(symbols)}")
        
        results = {}
        
        if parallel and len(symbols) > 1:
            # Use ProcessPoolExecutor for true parallelism
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all backtest tasks
                futures = {
                    executor.submit(self.backtest_symbol, symbol, strategy_func, days): symbol 
                    for symbol in symbols
                }
                
                # Process results as they complete
                for future in tqdm(as_completed(futures), total=len(futures), desc="Backtesting Progress"):
                    symbol = futures[future]
                    try:
                        result = future.result()
                        results[symbol] = result
                        
                        if result["success"]:
                            metrics = result["metrics"]
                            print(f"\n{symbol} Backtest Results:")
                            print(f"Return: {metrics['return_pct']:.2f}%")
                            print(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
                            print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
                            print(f"Chart saved to: {result['chart_file']}")
                        else:
                            print(f"\n{symbol} Backtest Failed: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        print(f"Error processing results for {symbol}: {e}")
                        results[symbol] = {
                            "symbol": symbol,
                            "success": False,
                            "error": str(e)
                        }
        else:
            # Sequential processing
            for symbol in symbols:
                results[symbol] = self.backtest_symbol(symbol, strategy_func, days)
                
                if results[symbol]["success"]:
                    metrics = results[symbol]["metrics"]
                    print(f"\n{symbol} Backtest Results:")
                    print(f"Return: {metrics['return_pct']:.2f}%")
                    print(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
                    print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
                    print(f"Chart saved to: {results[symbol]['chart_file']}")
                else:
                    print(f"\n{symbol} Backtest Failed: {results[symbol].get('error', 'Unknown error')}")
        
        # Create a summary report
        self.generate_summary_report(results)
        
        return results
    
    def generate_summary_report(self, results):
        """Generate a summary report of all backtest results"""
        successful_results = {s: r for s, r in results.items() if r["success"]}
        
        if not successful_results:
            print("No successful backtest results to summarize")
            return
        
        # Prepare summary data
        summary_data = []
        for symbol, result in successful_results.items():
            metrics = result["metrics"]
            summary_data.append({
                "symbol": symbol,
                "return_pct": metrics.get("return_pct", 0),
                "win_rate": metrics.get("win_rate", 0),
                "profit_factor": metrics.get("profit_factor", 0),
                "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
                "trades": metrics.get("total_trades", 0)
            })
        
        # Convert to DataFrame for easier sorting and display
        df = pd.DataFrame(summary_data)
        
        # Sort by return percentage (descending)
        df = df.sort_values("return_pct", ascending=False)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(self.results_dir, f"backtest_summary_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        
        # Display summary
        print("\n=== Backtest Summary ===")
        print(f"Total symbols tested: {len(results)}")
        print(f"Successful backtests: {len(successful_results)}")
        print(f"Failed backtests: {len(results) - len(successful_results)}")
        
        print("\nTop Performing Symbols:")
        for i, row in df.head(5).iterrows():
            print(f"{row['symbol']}: {row['return_pct']:.2f}% return, {row['win_rate']:.2f}% win rate")
        
        print(f"\nSummary saved to: {csv_file}")
        
        # Generate comparison chart
        self.plot_performance_comparison(df)
    
    def plot_performance_comparison(self, summary_df):
        """Create a comparative performance chart for all symbols"""
        if len(summary_df) <= 1:
            return  # Need at least 2 symbols to compare
            
        plt.figure(figsize=(12, 8))
        
        # Plot returns
        ax1 = plt.subplot(2, 1, 1)
        summary_df.plot(x='symbol', y='return_pct', kind='bar', color='green', ax=ax1)
        ax1.set_title('Return Percentage by Symbol')
        ax1.set_ylabel('Return %')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot win rate and drawdown
        ax2 = plt.subplot(2, 1, 2)
        summary_df.plot(x='symbol', y='win_rate', kind='bar', color='blue', ax=ax2, position=1, width=0.3)
        summary_df.plot(x='symbol', y='max_drawdown_pct', kind='bar', color='red', ax=ax2, position=0, width=0.3)
        ax2.set_title('Win Rate and Max Drawdown by Symbol')
        ax2.set_ylabel('Percentage')
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend(['Win Rate', 'Max Drawdown'])
        
        plt.tight_layout()
        
        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_file = os.path.join(self.results_dir, f"performance_comparison_{timestamp}.png")
        plt.savefig(chart_file)
        plt.close()
        
        print(f"Performance comparison chart saved to: {chart_file}")

def find_best_opportunities(initial_capital=10000, strategy_func=None, lookback_days=180):
    """Find the most profitable tokens based on backtest results"""
    # Initialize the parallel backtester
    backtester = ParallelBacktester(initial_capital=initial_capital)
    
    # Get all available symbols
    symbols = backtester.get_all_symbols()
    print(f"Found {len(symbols)} symbols for opportunity analysis")
    
    # Run backtests in parallel
    results = backtester.backtest_multiple(symbols, strategy_func=strategy_func, days=lookback_days)
    
    # Filter successful results and rank by performance
    successful = {s: r for s, r in results.items() if r["success"]}
    
    if not successful:
        print("No successful backtests to analyze")
        return []
    
    # Create a DataFrame with metrics for comparison
    metrics_data = []
    for symbol, result in successful.items():
        metrics = result["metrics"]
        
        # Calculate a combined score based on multiple factors
        # 60% weight on returns, 20% on win rate, 20% on inverse of max drawdown
        return_pct = metrics.get("return_pct", 0)
        win_rate = metrics.get("win_rate", 0)
        max_drawdown = metrics.get("max_drawdown_pct", 100) if metrics.get("max_drawdown_pct") else 100
        drawdown_factor = 100 / max(1, max_drawdown)  # Inverse of drawdown (higher is better)
        
        combined_score = (0.6 * return_pct) + (0.2 * win_rate) + (0.2 * drawdown_factor * 10)
        
        metrics_data.append({
            "symbol": symbol,
            "return_pct": return_pct,
            "win_rate": win_rate,
            "max_drawdown_pct": max_drawdown,
            "profit_factor": metrics.get("profit_factor", 0),
            "total_trades": metrics.get("total_trades", 0),
            "combined_score": combined_score
        })
    
    # Create DataFrame and sort by combined score
    df = pd.DataFrame(metrics_data)
    df = df.sort_values("combined_score", ascending=False)
    
    # Save the opportunity analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"opportunity_analysis_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    
    # Display top opportunities
    print("\n=== Top Trading Opportunities ===")
    for i, row in df.head(10).iterrows():
        print(f"{i+1}. {row['symbol']}: Score {row['combined_score']:.2f}, Return {row['return_pct']:.2f}%, " +
              f"Win Rate {row['win_rate']:.2f}%, Max DD {row['max_drawdown_pct']:.2f}%")
    
    print(f"\nComplete analysis saved to: {csv_file}")
    
    # Return the top opportunities
    return df.head(10).to_dict('records')


# Demo usage
if __name__ == "__main__":
    print("Multi-Token Parallel Backtester")
    print("------------------------------")
    
    # Get user input for configuration
    capital = float(input("Enter initial capital (default: $10000): ") or 10000)
    chunk_size = int(input("Enter chunk size (default: 1000): ") or 1000)
    max_workers = int(input("Enter max parallel workers (default: auto): ") or 0)
    
    # Create backtester
    backtester = ParallelBacktester(
        initial_capital=capital,
        chunk_size=chunk_size,
        max_workers=max_workers if max_workers > 0 else None
    )
    
    # Get all available symbols
    all_symbols = backtester.get_all_symbols()
    print(f"Found {len(all_symbols)} symbols for backtesting: {', '.join(all_symbols)}")
    
    # Ask which symbols to test
    symbols_input = input("Enter symbols to test (comma-separated, or 'all' for all symbols): ")
    if symbols_input.lower() == 'all':
        symbols = all_symbols
    else:
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
    
    # Ask about parallel processing
    parallel = input("Use parallel processing? (y/n, default: y): ").lower() != 'n'
    
    # Run backtests
    results = backtester.backtest_multiple(symbols, parallel=parallel)
    
    print("\nBacktesting complete!")