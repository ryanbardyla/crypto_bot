import os
import gc
import json
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Import the necessary classes
from multi_api_price_fetcher import CryptoPriceFetcher
from crypto_analyzer import CryptoAnalyzer

class SimpleBacktester:
    def __init__(self, initial_capital=10000, chunk_size=1000, memory_threshold=80):
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.position = 0
        self.trades = []
        self.performance = []
        self.metrics = {}
        self.price_fetcher = CryptoPriceFetcher()
        self.analyzer = CryptoAnalyzer(self.price_fetcher)
        self.chunk_size = chunk_size
        self.memory_threshold = memory_threshold
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
    
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
    
    def get_data_iterator(self, symbol, file_path=None):
        """Generator that yields data in chunks to conserve memory"""
        if file_path and os.path.exists(file_path):
            print(f"Loading {symbol} price data from {file_path} in chunks")
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                if 'timestamp' in chunk.columns and isinstance(chunk['timestamp'].iloc[0], str):
                    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
                yield chunk
        else:
            print(f"Loading {symbol} price data from internal history")
            df = self.analyzer.convert_history_to_dataframe(symbol)
            if df is None or len(df) < 10:
                print(f"Warning: Not enough price data for {symbol}. Need at least 10 data points.")
                return
            
            total_rows = len(df)
            for i in range(0, total_rows, self.chunk_size):
                end_idx = min(i + self.chunk_size, total_rows)
                yield df.iloc[i:end_idx].copy()
                
                # Check memory after processing each chunk
                self.check_memory_usage()
    
    def load_price_data(self, symbol, file_path=None, memory_efficient=True):
        """Load price data efficiently based on memory constraints"""
        if memory_efficient:
            return self.get_data_iterator(symbol, file_path)
        else:
            try:
                print(f"Loading {symbol} price data from {file_path}")
                df = pd.read_csv(file_path)
                if 'timestamp' in df.columns and isinstance(df['timestamp'].iloc[0], str):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            except Exception as e:
                print(f"Error loading price data: {e}")
                return self.analyzer.convert_history_to_dataframe(symbol)
    
    def simple_strategy(self, df):
        """Simple moving average crossover strategy"""
        if len(df) < 20:  # Need at least 20 data points for meaningful strategy
            print("Not enough data for strategy calculation")
            return None
            
        result = df.copy()
        result['short_ma'] = result['price'].rolling(window=5).mean()
        result['long_ma'] = result['price'].rolling(window=15).mean()
        result['signal'] = 0
        result['signal_strength'] = 0
        
        for i in range(1, len(result)):
            if pd.notna(result['short_ma'].iloc[i]) and pd.notna(result['long_ma'].iloc[i]):
                # Crossing above: Buy signal
                if result['short_ma'].iloc[i] > result['long_ma'].iloc[i] and result['short_ma'].iloc[i-1] <= result['long_ma'].iloc[i-1]:
                    result.loc[result.index[i], 'signal'] = 1
                    # Signal strength based on distance between MAs
                    strength = (result['short_ma'].iloc[i] - result['long_ma'].iloc[i]) / result['price'].iloc[i] * 100
                    result.loc[result.index[i], 'signal_strength'] = min(1.0, max(0.1, strength / 2))
                
                # Crossing below: Sell signal
                elif result['short_ma'].iloc[i] < result['long_ma'].iloc[i] and result['short_ma'].iloc[i-1] >= result['long_ma'].iloc[i-1]:
                    result.loc[result.index[i], 'signal'] = -1
                    # Signal strength based on distance between MAs
                    strength = (result['long_ma'].iloc[i] - result['short_ma'].iloc[i]) / result['price'].iloc[i] * 100
                    result.loc[result.index[i], 'signal_strength'] = min(1.0, max(0.1, strength / 2))
        
        return result
    
    def backtest_chunk(self, strategy_df, starting_balance, starting_position, initial=False):
        """Process a chunk of data for backtesting"""
        balance = starting_balance
        position = starting_position
        local_trades = []
        local_performance = []
        max_balance = balance
        current_drawdown = 0
        
        for i in range(0, len(strategy_df)):
            price = strategy_df['price'].iloc[i]
            signal = strategy_df['signal'].iloc[i]
            signal_strength = strategy_df['signal_strength'].iloc[i]
            timestamp = strategy_df.index[i] if isinstance(strategy_df.index[i], datetime) else i
            
            # Calculate position value
            position_value = position * price
            total_value = balance + position_value
            
            # Calculate drawdown
            if total_value > max_balance:
                max_balance = total_value
                current_drawdown = 0
            else:
                current_drawdown = (max_balance - total_value) / max_balance * 100
            
            # Record performance
            local_performance.append({
                'timestamp': timestamp,
                'balance': balance,
                'position': position,
                'position_value': position_value,
                'total_value': total_value,
                'return_pct': (total_value / self.initial_capital - 1) * 100
            })
            
            # Process signals
            if signal == 1 and position == 0:  # Buy signal
                position_size = balance * signal_strength
                position = position_size / price
                balance -= position_size
                
                local_trades.append({
                    'type': 'BUY',
                    'timestamp': timestamp,
                    'price': price,
                    'quantity': position,
                    'value': position_size,
                    'balance_after': balance
                })
                
            elif signal == -1 and position > 0:  # Sell signal
                sale_value = position * price
                profit_loss = sale_value - (position * local_trades[-1]['price'])
                profit_loss_pct = profit_loss / (position * local_trades[-1]['price']) * 100
                
                balance += sale_value
                
                local_trades.append({
                    'type': 'SELL',
                    'timestamp': timestamp,
                    'price': price,
                    'quantity': position,
                    'value': sale_value,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct,
                    'balance_after': balance
                })
                
                position = 0
            
            # Update max drawdown in metrics
            if len(self.metrics) > 0:  # Update max drawdown if metrics exist
                self.metrics["max_drawdown_pct"] = max(self.metrics.get("max_drawdown_pct", 0), current_drawdown)
        
        return balance, position, local_trades, local_performance, max_balance
    
    def backtest(self, symbol, strategy_func=None, file_path=None, memory_efficient=True):
        """Backtest a trading strategy on historical data"""
        self.balance = self.initial_capital
        self.position = 0
        self.trades = []
        self.performance = []
        self.metrics = {}
        
        if strategy_func is None:
            strategy_func = self.simple_strategy
        
        self.check_memory_usage()
        
        try:
            if memory_efficient:
                data_iterator = self.load_price_data(symbol, file_path, memory_efficient=True)
                if data_iterator is None:
                    print("Cannot run backtest: No price data available")
                    return None
                
                chunks_processed = 0
                for chunk in data_iterator:
                    chunks_processed += 1
                    print(f"Processing chunk {chunks_processed} ({len(chunk)} rows)")
                    
                    # Check memory before processing each chunk
                    self.check_memory_usage(critical=True)
                    
                    strategy_df = strategy_func(chunk)
                    if strategy_df is None or len(strategy_df) == 0:
                        print("Cannot process chunk: Strategy failed to generate signals")
                        continue
                    
                    # Process chunk
                    self.balance, self.position, chunk_trades, chunk_performance, max_balance = self.backtest_chunk(
                        strategy_df, self.balance, self.position, initial=(chunks_processed == 1)
                    )
                    
                    self.trades.extend(chunk_trades)
                    self.performance.extend(chunk_performance)
                    
                    # Force garbage collection after each chunk
                    del strategy_df, chunk_trades, chunk_performance
                    gc.collect()
            else:
                # Non-chunked processing (legacy mode)
                df = self.load_price_data(symbol, file_path, memory_efficient=False)
                strategy_df = strategy_func(df)
                if strategy_df is None:
                    print("Cannot run backtest: Strategy failed to generate signals")
                    return None
                
                self.balance, self.position, self.trades, self.performance, _ = self.backtest_chunk(
                    strategy_df, self.balance, self.position, initial=True
                )
            
            # Calculate metrics
            if len(self.trades) == 0:
                print("No trades executed during backtest")
                return {"return_pct": 0, "win_rate": 0, "profit_factor": 0}
            
            # Calculate performance metrics
            sell_trades = [t for t in self.trades if t.get("type") == "SELL"]
            self.metrics["total_trades"] = len(sell_trades)
            self.metrics["winning_trades"] = sum(1 for t in sell_trades if t.get("profit_loss", 0) > 0)
            self.metrics["losing_trades"] = sum(1 for t in sell_trades if t.get("profit_loss", 0) <= 0)
            
            if self.metrics["total_trades"] > 0:
                self.metrics["win_rate"] = (self.metrics["winning_trades"] / self.metrics["total_trades"]) * 100
            else:
                self.metrics["win_rate"] = 0
            
            # Calculate return percentage
            if self.position > 0 and len(self.performance) > 0:
                final_value = self.performance[-1]["total_value"]
            else:
                final_value = self.balance
                
            self.metrics["starting_balance"] = self.initial_capital
            self.metrics["ending_balance"] = final_value
            self.metrics["return"] = final_value - self.initial_capital
            self.metrics["return_pct"] = (final_value / self.initial_capital - 1) * 100
            
            # Calculate profit factor (sum of profits divided by sum of losses)
            total_profits = sum(t.get("profit_loss", 0) for t in sell_trades if t.get("profit_loss", 0) > 0)
            total_losses = abs(sum(t.get("profit_loss", 0) for t in sell_trades if t.get("profit_loss", 0) < 0))
            self.metrics["profit_factor"] = total_profits / total_losses if total_losses > 0 else float('inf')
            
            print(f"\nBacktest completed for {symbol}:")
            print(f"Initial capital: ${self.initial_capital:.2f}")
            print(f"Final capital: ${final_value:.2f}")
            print(f"Return: {self.metrics['return_pct']:.2f}%")
            print(f"Win Rate: {self.metrics['win_rate']:.2f}%")
            print(f"Max Drawdown: {self.metrics.get('max_drawdown_pct', 0):.2f}%")
            
            return self.metrics
            
        except Exception as e:
            print(f"Error during backtest: {e}")
            return None

       
    def plot_performance(self, symbol, save_to_file=None):
        """
        Plot backtest performance results.
        
        Args:
            symbol (str): The cryptocurrency symbol
            save_to_file (str, optional): Path to save the performance chart
        """
        if not self.performance:
            print("No performance data to plot")
            return
        
        # Check memory before plotting
        self.check_memory_usage()
        
        # Convert to DataFrame for plotting
        df = pd.DataFrame(self.performance)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot equity curve
        df['total_value'].plot(ax=ax1, label='Total Value')
        df['balance'].plot(ax=ax1, label='Cash Balance')
        ax1.set_title('Paper Trading Account Performance')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot return percentage
        if 'return_pct' in df.columns:
            df['return_pct'].plot(ax=ax2, color='green')
            ax2.set_ylabel('Return (%)')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add trade markers if available
        buy_trades = [t for t in self.trades if t.get('type') == 'BUY']
        sell_trades = [t for t in self.trades if t.get('type') == 'SELL']
        
        try:
            if buy_trades and 'timestamp' in buy_trades[0]:
                buy_times = pd.to_datetime([trade['timestamp'] for trade in buy_trades])
                buy_values = [trade.get('value', 0) for trade in buy_trades]
                ax1.scatter(buy_times, buy_values, marker='^', color='green', s=100, label='Buy')
                
            if sell_trades and 'timestamp' in sell_trades[0]:
                sell_times = pd.to_datetime([trade['timestamp'] for trade in sell_trades])
                sell_values = [trade.get('value', 0) for trade in sell_trades]
                ax1.scatter(sell_times, sell_values, marker='v', color='red', s=100, label='Sell')
                
                # Add profit/loss annotations
                for trade in sell_trades:
                    if 'profit_loss' in trade and 'timestamp' in trade:
                        time = pd.to_datetime(trade['timestamp'])
                        price = trade.get('value', 0)
                        pnl = trade.get('profit_loss', 0)
                        pnl_pct = (pnl / price * 100) if price else 0
                        
                        ax1.annotate(f"${pnl:.0f}\n({pnl_pct:.1f}%)", 
                                    xy=(time, price),
                                    xytext=(10, 20),
                                    textcoords='offset points',
                                    arrowprops=dict(arrowstyle='->', color='black'),
                                    color='green' if pnl > 0 else 'red')
                        
        except Exception as e:
            print(f"Error adding trade markers: {e}")
            
        plt.tight_layout()
        
        if save_to_file:
            plt.savefig(save_to_file)
            print(f"Performance chart saved to {save_to_file}")
        else:
            plt.show()
        
        # Clean up to free memory
        plt.close(fig)
        gc.collect()
        
    def save_backtest_results(self, symbol, filename=None):
        """
        Save backtest results to a JSON file.
        
        Args:
            symbol (str): The cryptocurrency symbol
            filename (str, optional): Custom filename for results
            
        Returns:
            str: Path to the saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{symbol}_{timestamp}.json"
        
        results = {
            "symbol": symbol,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "initial_capital": self.initial_capital,
            "final_balance": self.balance,
            "position": self.position,
            "trades": self.trades,
            "performance_summary": {
                "total_trades": self.metrics["total_trades"],
                "winning_trades": self.metrics["winning_trades"],
                "losing_trades": self.metrics["losing_trades"],
                "win_rate": (self.metrics["winning_trades"] / self.metrics["total_trades"] * 100) if self.metrics["total_trades"] > 0 else 0,
                "profit_factor": self.metrics.get("profit_factor", 0),
                "max_drawdown_pct": self.metrics["max_drawdown_pct"]
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Backtest results saved to {filename}")
        return filename

# Example usage
if __name__ == "__main__":
    print("Simple Cryptocurrency Backtester with Memory Management")
    print("------------------------------------------------------")
    
    capital = float(input("Enter starting capital (default: $10000): ") or 10000)
    chunk_size = int(input("Enter processing chunk size (default: 1000): ") or 1000)
    memory_threshold = int(input("Enter memory usage threshold % (default: 80): ") or 80)
    
    backtester = SimpleBacktester(initial_capital=capital, chunk_size=chunk_size, memory_threshold=memory_threshold)
    
    symbol = input("Enter cryptocurrency symbol to backtest (default: BTC): ").upper() or "BTC"
    memory_efficient = input("Use memory-efficient mode? (y/n, default: y): ").lower() != 'n'
    
    print(f"\nRunning backtest for {symbol} in {'memory-efficient' if memory_efficient else 'standard'} mode...")
    results = backtester.backtest(symbol, memory_efficient=memory_efficient)
    
    if results:
        print("\nBacktest Results Summary:")
        print(f"Starting Balance: ${results['starting_balance']:.2f}")
        print(f"Ending Balance: ${results['ending_balance']:.2f}")
        print(f"Return: {results['return_pct']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        
        if results['total_trades'] > 0:
            print(f"Win Rate: {results['win_rate']:.2f}% ({results['winning_trades']}/{results['total_trades']})")
            print(f"Profit Factor: {results['profit_factor']:.2f}")
            
        print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        
        generate_chart = input("\nGenerate performance chart? (y/n): ").lower() == 'y'
        if generate_chart:
            os.makedirs("charts", exist_ok=True)
            chart_path = f"charts/backtest_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            print("\nGenerating backtest chart...")
            backtester.plot_performance(symbol, save_to_file=chart_path)
        
        save_results = input("\nDo you want to save detailed backtest results? (y/n): ").lower() == 'y'
        if save_results:
            backtester.save_backtest_results(symbol)
    else:
        print("\nBacktest failed. Please check error messages above.")