# simple_backtester.py
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from multi_api_price_fetcher import CryptoPriceFetcher
from crypto_analyzer import CryptoAnalyzer

class SimpleBacktester:
    """Simple backtesting system for cryptocurrency trading strategies"""
    
    def __init__(self, initial_capital=10000):
        """Initialize with starting capital amount"""
        self.initial_capital = initial_capital
        self.price_fetcher = CryptoPriceFetcher()
        self.analyzer = CryptoAnalyzer(self.price_fetcher)
        
        # Trading performance metrics
        self.metrics = {
            "starting_balance": initial_capital,
            "ending_balance": initial_capital,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "profit_factor": 0,
            "max_drawdown_pct": 0
        }
        
        # Trade history
        self.trades = []
        
        # Performance log (balance over time)
        self.performance = []
    
    def load_price_data(self, symbol, file_path=None):
        """Load price data for backtesting either from API history or from file"""
        if file_path and os.path.exists(file_path):
            # Load from CSV file if provided
            print(f"Loading {symbol} price data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Convert timestamp strings to datetime objects if needed
            if 'timestamp' in df.columns and isinstance(df['timestamp'].iloc[0], str):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            # Make sure we have a 'price' column
            if 'price' not in df.columns and 'close' in df.columns:
                df['price'] = df['close']  # Use 'close' as 'price' if available
                
            return df
        else:
            # Use existing data collected by the price fetcher
            print(f"Loading {symbol} price data from internal history")
            df = self.analyzer.convert_history_to_dataframe(symbol)
            
            if df is None or len(df) < 10:
                print(f"Warning: Not enough price data for {symbol}. Need at least 10 data points.")
                return None
                
            return df
    
    def simple_strategy(self, df):
        """
        Implement a simple moving average crossover strategy
        Returns DataFrame with buy/sell signals
        """
        if len(df) < 20:  # Need at least 20 data points for meaningful strategy
            print("Not enough data for strategy calculation")
            return None
            
        # Create a copy to avoid SettingWithCopyWarning
        result = df.copy()
        
        # Calculate short and long term moving averages
        result['short_ma'] = result['price'].rolling(window=5).mean()
        result['long_ma'] = result['price'].rolling(window=15).mean()
        
        # Initialize signal column
        result['signal'] = 0
        
        # Generate buy signal (1) when short MA crosses above long MA
        # Generate sell signal (-1) when short MA crosses below long MA
        for i in range(1, len(result)):
            if pd.notna(result['short_ma'].iloc[i]) and pd.notna(result['long_ma'].iloc[i]):
                # Buy signal
                if (result['short_ma'].iloc[i] > result['long_ma'].iloc[i] and 
                    result['short_ma'].iloc[i-1] <= result['long_ma'].iloc[i-1]):
                    result.loc[result.index[i], 'signal'] = 1
                
                # Sell signal
                elif (result['short_ma'].iloc[i] < result['long_ma'].iloc[i] and 
                      result['short_ma'].iloc[i-1] >= result['long_ma'].iloc[i-1]):
                    result.loc[result.index[i], 'signal'] = -1
        
        return result
    
    def backtest(self, symbol, strategy_func=None, file_path=None):
        """Run backtest on historical data using specified strategy"""
        # Reset performance metrics
        self.metrics = {
            "starting_balance": self.initial_capital,
            "ending_balance": self.initial_capital,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "profit_factor": 0,
            "max_drawdown_pct": 0
        }
        self.trades = []
        self.performance = []
        
        # Load price data
        df = self.load_price_data(symbol, file_path)
        if df is None:
            print("Cannot run backtest: No price data available")
            return None
        
        # Use provided strategy or default to simple_strategy
        if strategy_func is None:
            strategy_func = self.simple_strategy
        
        # Apply strategy to get signals
        strategy_df = strategy_func(df)
        if strategy_df is None:
            print("Cannot run backtest: Strategy failed to generate signals")
            return None
        
        # Initialize variables for backtesting
        balance = self.initial_capital
        position = 0
        entry_price = 0
        max_balance = balance
        
        # Record initial performance
        self.performance.append({
            "timestamp": strategy_df.index[0] if isinstance(strategy_df.index[0], datetime) else None,
            "balance": balance,
            "position": position
        })
        
        # Iterate through the data to simulate trades
        for i in range(1, len(strategy_df)):
            current_row = strategy_df.iloc[i]
            signal = current_row['signal']
            price = current_row['price']
            
            # Record timestamp for the current step
            timestamp = strategy_df.index[i] if isinstance(strategy_df.index[i], datetime) else i
            
            # Buy signal and not in position
            if signal == 1 and position == 0:
                position_size = balance * 0.95  # Use 95% of balance
                position = position_size / price
                entry_price = price
                balance -= position_size
                
                # Record trade
                self.trades.append({
                    "type": "BUY",
                    "timestamp": timestamp,
                    "price": price,
                    "quantity": position,
                    "value": position_size
                })
                
                print(f"BUY: {position:.4f} {symbol} at ${price:.2f}")
            
            # Sell signal and in position
            elif signal == -1 and position > 0:
                position_value = position * price
                profit_loss = position_value - (position * entry_price)
                balance += position_value
                
                # Update metrics
                self.metrics["total_trades"] += 1
                if profit_loss > 0:
                    self.metrics["winning_trades"] += 1
                else:
                    self.metrics["losing_trades"] += 1
                
                # Record trade
                self.trades.append({
                    "type": "SELL",
                    "timestamp": timestamp,
                    "price": price,
                    "quantity": position,
                    "value": position_value,
                    "profit_loss": profit_loss
                })
                
                print(f"SELL: {position:.4f} {symbol} at ${price:.2f}, P/L: ${profit_loss:.2f}")
                
                # Reset position
                position = 0
            
            # Update max balance
            total_value = balance + (position * price)
            max_balance = max(max_balance, total_value)
            
            # Calculate current drawdown
            current_drawdown = (max_balance - total_value) / max_balance * 100 if max_balance > 0 else 0
            self.metrics["max_drawdown_pct"] = max(self.metrics["max_drawdown_pct"], current_drawdown)
            
            # Record performance for this step
            self.performance.append({
                "timestamp": timestamp,
                "balance": balance,
                "position": position,
                "position_value": position * price,
                "total_value": total_value
            })
        
        # Calculate final metrics
        final_balance = balance + (position * strategy_df.iloc[-1]['price'])
        self.metrics["ending_balance"] = final_balance
        self.metrics["return_pct"] = (final_balance / self.initial_capital - 1) * 100
        
        # Calculate profit factor if we have any trades
        if self.metrics["total_trades"] > 0:
            total_profits = sum(trade["profit_loss"] for trade in self.trades if "profit_loss" in trade and trade["profit_loss"] > 0)
            total_losses = abs(sum(trade["profit_loss"] for trade in self.trades if "profit_loss" in trade and trade["profit_loss"] < 0))
            self.metrics["profit_factor"] = total_profits / total_losses if total_losses > 0 else float('inf')
        
        return self.metrics
    
    def plot_backtest_results(self, symbol, save_to_file=None):
        """Plot backtest results including equity curve and trades"""
        if not self.performance:
            print("No backtest results to plot")
            return
        
        # Convert performance log to DataFrame
        perf_df = pd.DataFrame(self.performance)
        
        # Check if we have timestamp data
        has_timestamps = 'timestamp' in perf_df.columns and not pd.isna(perf_df['timestamp'].iloc[0])
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot equity curve
        if has_timestamps:
            perf_df.set_index('timestamp', inplace=True)
            
        perf_df['total_value'].plot(ax=ax1, title=f'Equity Curve - {symbol}')
        ax1.set_ylabel('Account Value ($)')
        ax1.grid(True, alpha=0.3)
        
        # Plot buy and sell signals
        for trade in self.trades:
            if trade['type'] == 'BUY':
                ax1.scatter(trade['timestamp'], trade['value'], marker='^', color='green', s=100)
            else:  # SELL
                ax1.scatter(trade['timestamp'], trade['value'], marker='v', color='red', s=100)
        
        # Plot drawdowns
        if has_timestamps:
            perf_df.reset_index(inplace=True)
            
        # Calculate running maximum
        perf_df['max_value'] = perf_df['total_value'].cummax()
        
        # Calculate drawdown percentage
        perf_df['drawdown'] = (perf_df['max_value'] - perf_df['total_value']) / perf_df['max_value'] * 100
        
        # Plot drawdown
        if has_timestamps:
            perf_df.set_index('timestamp', inplace=True)
            
        perf_df['drawdown'].plot(ax=ax2, color='red', title='Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.invert_yaxis()  # Invert y-axis to show drawdowns as going down
        ax2.grid(True, alpha=0.3)
        
        # Add summary text
        summary_text = (
            f"Starting Balance: ${self.metrics['starting_balance']:.2f}\n"
            f"Ending Balance: ${self.metrics['ending_balance']:.2f}\n"
            f"Return: {self.metrics['return_pct']:.2f}%\n"
            f"Total Trades: {self.metrics['total_trades']}\n"
            f"Win Rate: {self.metrics['winning_trades'] / self.metrics['total_trades'] * 100:.2f}% "
            f"({self.metrics['winning_trades']}/{self.metrics['total_trades']})\n"
            f"Profit Factor: {self.metrics['profit_factor']:.2f}\n"
            f"Max Drawdown: {self.metrics['max_drawdown_pct']:.2f}%"
        )
        
        plt.figtext(0.15, 0.01, summary_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.5))
        
        # Adjust layout and display
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # Make room for text
        
        # Save to file if requested
        if save_to_file:
            plt.savefig(save_to_file)
            print(f"Chart saved to {save_to_file}")
        else:
            plt.show()
    
    def save_backtest_results(self, symbol, filename=None):
        """Save backtest results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{symbol}_{timestamp}.json"
        
        results = {
            "symbol": symbol,
            "metrics": self.metrics,
            "trades": self.trades
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Backtest results saved to {filename}")


# Simple demonstration
if __name__ == "__main__":
    print("Simple Cryptocurrency Backtester")
    print("-------------------------------")
    
    # Initialize backtester with starting capital
    capital = 10000
    backtester = SimpleBacktester(initial_capital=capital)
    
    # Select cryptocurrency to backtest
    symbol = input("Enter cryptocurrency symbol to backtest (default: BTC): ").upper() or "BTC"
    
    # Run backtest
    print(f"\nRunning backtest for {symbol}...")
    results = backtester.backtest(symbol)
    
    if results:
        print("\nBacktest Results Summary:")
        print(f"Starting Balance: ${results['starting_balance']:.2f}")
        print(f"Ending Balance: ${results['ending_balance']:.2f}")
        print(f"Return: {results['return_pct']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        
        if results['total_trades'] > 0:
            win_rate = results['winning_trades'] / results['total_trades'] * 100
            print(f"Win Rate: {win_rate:.2f}% ({results['winning_trades']}/{results['total_trades']})")
            print(f"Profit Factor: {results['profit_factor']:.2f}")
        
        print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        
        # Create charts directory if it doesn't exist
        os.makedirs("charts", exist_ok=True)
        
        # Plot results
        print("\nGenerating backtest chart...")
        chart_path = f"charts/{symbol}_backtest_results.png"
        backtester.plot_backtest_results(symbol, save_to_file=chart_path)
        
        # Save detailed results
        save_results = input("\nDo you want to save detailed backtest results? (y/n): ").lower()
        if save_results in ['y', 'yes']:
            backtester.save_backtest_results(symbol)
    else:
        print("\nBacktest failed. Please check error messages above.")