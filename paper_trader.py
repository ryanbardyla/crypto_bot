# paper_trader.py
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import os
import schedule
import threading
from multi_api_price_fetcher import CryptoPriceFetcher
from crypto_analyzer import CryptoAnalyzer

class PaperTrader:
    """
    Paper trading system that simulates cryptocurrency trades without using real money.
    Uses real-time price data and trading signals to make decisions.
    """
    
    def __init__(self, initial_capital=10000):
        """Initialize with starting capital"""
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.positions = {}  # symbol -> {'quantity': qty, 'entry_price': price}
        self.trade_history = []
        self.performance_log = []
        
        # Create our core components
        self.price_fetcher = CryptoPriceFetcher()
        self.analyzer = CryptoAnalyzer(self.price_fetcher)
        
        # Trading parameters (can be adjusted)
        self.position_size_pct = 0.95  # Use 95% of available balance for each position
        self.stop_loss_pct = 5.0       # 5% stop loss
        self.take_profit_pct = 10.0    # 10% take profit
        self.max_positions = 3         # Maximum number of simultaneous positions
        
        # Auto-save settings
        self.auto_save = True
        self.save_interval = 10  # minutes
        self.data_dir = "paper_trading"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Log initial state
        self._log_performance()
    
    def _log_performance(self):
        """Record current performance metrics"""
        # Calculate total value including open positions
        total_value = self.balance
        for symbol, position in self.positions.items():
            current_price = self._get_current_price(symbol)
            position_value = position['quantity'] * current_price
            total_value += position_value
        
        # Create performance record
        record = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'balance': self.balance,
            'positions_value': total_value - self.balance,
            'total_value': total_value,
            'return_pct': (total_value / self.initial_capital - 1) * 100
        }
        
        self.performance_log.append(record)
    
    def _get_current_price(self, symbol):
        """Get current price for a symbol, handling potential errors"""
        try:
            price = self.price_fetcher.get_price(symbol)
            if price is None:
                # If price fetch fails, check if we have recent price in history
                df = self.analyzer.convert_history_to_dataframe(symbol)
                if df is not None and not df.empty:
                    price = df['price'].iloc[-1]
                    print(f"Using most recent stored price for {symbol}: ${price:.2f}")
                else:
                    print(f"Warning: Could not get price for {symbol}")
                    return None
            return price
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return None
    
    def open_position(self, symbol, price=None, quantity=None):
        """
        Open a new position in the specified symbol
        If price is None, fetch current market price
        If quantity is None, calculate based on position size percentage
        """
        # Check if we're already holding this symbol
        if symbol in self.positions:
            print(f"Already holding {symbol}, cannot open new position")
            return False
        
        # Check if we've reached the maximum number of positions
        if len(self.positions) >= self.max_positions:
            print(f"Maximum number of positions reached ({self.max_positions})")
            return False
        
        # Get current price if not provided
        if price is None:
            price = self._get_current_price(symbol)
            if price is None:
                return False
        
        # Calculate quantity based on position size if not provided
        if quantity is None:
            # Use percentage of available balance
            position_value = self.balance * self.position_size_pct
            
            # If we have very little balance left, use all of it
            if position_value < 100:
                position_value = self.balance
                
            quantity = position_value / price
        else:
            position_value = quantity * price
        
        # Check if we have enough balance
        if position_value > self.balance:
            print(f"Insufficient balance to open position in {symbol}")
            return False
        
        # Open the position
        self.positions[symbol] = {
            'quantity': quantity,
            'entry_price': price,
            'entry_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'stop_loss': price * (1 - self.stop_loss_pct/100),
            'take_profit': price * (1 + self.take_profit_pct/100)
        }
        
        # Update balance
        self.balance -= position_value
        
        # Log the trade
        trade = {
            'type': 'BUY',
            'symbol': symbol,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'price': price,
            'quantity': quantity,
            'value': position_value,
            'balance_after': self.balance
        }
        self.trade_history.append(trade)
        
        print(f"Opened {symbol} position: {quantity:.6f} @ ${price:.2f} = ${position_value:.2f}")
        self._log_performance()
        return True
    
    def close_position(self, symbol, price=None, reason="Manual"):
        """
        Close an existing position in the specified symbol
        If price is None, fetch current market price
        """
        # Check if we're holding this symbol
        if symbol not in self.positions:
            print(f"No open position for {symbol}")
            return False
        
        # Get current price if not provided
        if price is None:
            price = self._get_current_price(symbol)
            if price is None:
                return False
        
        # Get position details
        position = self.positions[symbol]
        quantity = position['quantity']
        entry_price = position['entry_price']
        
        # Calculate position value and profit/loss
        position_value = quantity * price
        entry_value = quantity * entry_price
        profit_loss = position_value - entry_value
        profit_loss_pct = (price / entry_price - 1) * 100
        
        # Update balance
        self.balance += position_value
        
        # Log the trade
        trade = {
            'type': 'SELL',
            'symbol': symbol,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'price': price,
            'quantity': quantity,
            'value': position_value,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'balance_after': self.balance,
            'reason': reason
        }
        self.trade_history.append(trade)
        
        # Remove the position
        del self.positions[symbol]
        
        print(f"Closed {symbol} position: {quantity:.6f} @ ${price:.2f} = ${position_value:.2f}")
        print(f"Profit/Loss: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
        self._log_performance()
        return True
    
    def check_stop_loss_take_profit(self):
        """Check if any positions have hit stop loss or take profit levels"""
        for symbol in list(self.positions.keys()):
            price = self._get_current_price(symbol)
            if price is None:
                continue
                
            position = self.positions[symbol]
            
            # Check stop loss
            if price <= position['stop_loss']:
                print(f"Stop loss triggered for {symbol} at ${price:.2f}")
                self.close_position(symbol, price, reason="Stop Loss")
            
            # Check take profit
            elif price >= position['take_profit']:
                print(f"Take profit triggered for {symbol} at ${price:.2f}")
                self.close_position(symbol, price, reason="Take Profit")
    
    def generate_signals(self, symbols=None):
        """Generate trading signals for the specified symbols"""
        if symbols is None:
            symbols = ["BTC", "ETH", "SOL", "DOGE"]
        
        results = {}
        for symbol in symbols:
            try:
                signals = self.analyzer.get_simple_signals(symbol)
                results[symbol] = signals
            except Exception as e:
                print(f"Error generating signals for {symbol}: {e}")
        
        return results
    
    def process_signals(self, signals):
        """Process trading signals to make trade decisions"""
        for symbol, signal_data in signals.items():
            if "signals" not in signal_data:
                continue
                
            signal_list = signal_data["signals"]
            bullish_signals = sum(1 for s in signal_list if "BULLISH" in s)
            bearish_signals = sum(1 for s in signal_list if "BEARISH" in s)
            
            # Simple decision logic
            if symbol in self.positions:
                # Already have a position - check if we should close it
                if bearish_signals >= 2:
                    print(f"Strong bearish signal for {symbol}, closing position")
                    self.close_position(symbol, reason="Bearish Signal")
            else:
                # No position - check if we should open one
                if bullish_signals >= 2:
                    print(f"Strong bullish signal for {symbol}, opening position")
                    self.open_position(symbol)
    
    def save_state(self):
        """Save current state to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trade history
        trades_file = f"{self.data_dir}/trade_history.json"
        with open(trades_file, 'w') as f:
            json.dump(self.trade_history, f, indent=2)
        
        # Save performance log
        perf_file = f"{self.data_dir}/performance_log.json"
        with open(perf_file, 'w') as f:
            json.dump(self.performance_log, f, indent=2)
        
        # Save current state
        state_file = f"{self.data_dir}/current_state.json"
        state = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'initial_capital': self.initial_capital,
            'balance': self.balance,
            'positions': self.positions,
            'total_trades': len(self.trade_history),
            'params': {
                'position_size_pct': self.position_size_pct,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
                'max_positions': self.max_positions
            }
        }
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
            
        print(f"Saved trader state at {timestamp}")
    
    def load_state(self, state_file=None):
        """Load state from files"""
        if state_file is None:
            state_file = f"{self.data_dir}/current_state.json"
            
        if not os.path.exists(state_file):
            print(f"No saved state found at {state_file}")
            return False
            
        try:
            # Load current state
            with open(state_file, 'r') as f:
                state = json.load(f)
                
            self.balance = state['balance']
            self.positions = state['positions']
            
            # Load trade history if it exists
            trades_file = f"{self.data_dir}/trade_history.json"
            if os.path.exists(trades_file):
                with open(trades_file, 'r') as f:
                    self.trade_history = json.load(f)
            
            # Load performance log if it exists
            perf_file = f"{self.data_dir}/performance_log.json"
            if os.path.exists(perf_file):
                with open(perf_file, 'r') as f:
                    self.performance_log = json.load(f)
                    
            print(f"Loaded trader state from {state_file}")
            print(f"Current balance: ${self.balance:.2f}")
            print(f"Open positions: {len(self.positions)}")
            return True
            
        except Exception as e:
            print(f"Error loading state: {e}")
            return False
    
    def update_prices(self, symbols=None):
        """Fetch latest prices for the specified symbols"""
        if symbols is None:
            symbols = ["BTC", "ETH", "SOL", "DOGE"]
            
        for symbol in symbols:
            price = self.price_fetcher.get_price(symbol)
            if price is not None:
                print(f"Updated {symbol} price: ${price:.2f}")
    
    def print_status(self):
        """Print current account status"""
        total_value = self.balance
        
        print("\n=== Paper Trading Account Status ===")
        print(f"Cash balance: ${self.balance:.2f}")
        
        if self.positions:
            print("\nOpen Positions:")
            for symbol, position in self.positions.items():
                current_price = self._get_current_price(symbol)
                if current_price is None:
                    continue
                    
                position_value = position['quantity'] * current_price
                total_value += position_value
                
                entry_value = position['quantity'] * position['entry_price']
                profit_loss = position_value - entry_value
                profit_loss_pct = (current_price / position['entry_price'] - 1) * 100
                
                print(f"  {symbol}: {position['quantity']:.6f} @ ${position['entry_price']:.2f} (now ${current_price:.2f})")
                print(f"     Value: ${position_value:.2f} | P/L: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
                print(f"     Stop-Loss: ${position['stop_loss']:.2f} | Take-Profit: ${position['take_profit']:.2f}")
        else:
            print("\nNo open positions")
            
        print(f"\nTotal Account Value: ${total_value:.2f}")
        print(f"Return: {(total_value / self.initial_capital - 1) * 100:.2f}%")
        print(f"Total Trades: {len(self.trade_history)}")
        print("===================================\n")
    
    def plot_performance(self, save_to_file=None):
        """Plot account performance over time"""
        if not self.performance_log:
            print("No performance data to plot")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(self.performance_log)
        
        # Convert timestamp strings to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot total value
        df['total_value'].plot(ax=ax1, label='Total Value')
        df['balance'].plot(ax=ax1, label='Cash Balance')
        ax1.set_title('Paper Trading Account Performance')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot return percentage
        df['return_pct'].plot(ax=ax2, color='green')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, alpha=0.3)
        
        # Add zero line to return plot
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add trades to the plot if we have trade history
        if self.trade_history:
            buys = [trade for trade in self.trade_history if trade['type'] == 'BUY']
            sells = [trade for trade in self.trade_history if trade['type'] == 'SELL']
            
            if buys:
                buy_times = pd.to_datetime([trade['timestamp'] for trade in buys])
                buy_values = [trade['value'] for trade in buys]
                ax1.scatter(buy_times, buy_values, marker='^', color='green', s=100, label='Buy')
                
            if sells:
                sell_times = pd.to_datetime([trade['timestamp'] for trade in sells])
                sell_values = [trade['value'] for trade in sells]
                ax1.scatter(sell_times, sell_values, marker='v', color='red', s=100, label='Sell')
                
                # Add profit/loss annotations
                for trade in sells:
                    time = pd.to_datetime(trade['timestamp'])
                    value = trade['value']
                    pnl = trade['profit_loss']
                    pnl_pct = trade['profit_loss_pct']
                    
                    color = 'green' if pnl > 0 else 'red'
                    ax1.annotate(f"${pnl:.0f}\n({pnl_pct:.1f}%)", 
                                xy=(time, value),
                                xytext=(10, 10),
                                textcoords="offset points",
                                color=color,
                                fontweight='bold')
        
        plt.tight_layout()
        
        # Save to file if requested
        if save_to_file:
            plt.savefig(save_to_file)
            print(f"Performance chart saved to {save_to_file}")
        else:
            plt.show()
    
    def auto_save_task(self):
        """Task to automatically save state at regular intervals"""
        self.save_state()
    
    def run_scheduled_update(self):
        """Run a scheduled update of prices, signals, and positions"""
        symbols = ["BTC", "ETH", "SOL", "DOGE"]
        
        print(f"\n=== Scheduled Update {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        
        # Update prices
        print("Updating prices...")
        self.update_prices(symbols)
        
        # Check stop loss / take profit
        print("Checking stop loss / take profit levels...")
        self.check_stop_loss_take_profit()
        
        # Generate and process signals
        print("Generating trading signals...")
        signals = self.generate_signals(symbols)
        self.process_signals(signals)
        
        # Log performance
        self._log_performance()
        
        # Print current status
        self.print_status()
        
        print("Update complete")
    
    def run_scheduler(self, interval_minutes=15):
        """
        Run the paper trading system with a scheduler
        This will update at regular intervals
        """
        # Schedule regular updates
        schedule.every(interval_minutes).minutes.do(self.run_scheduled_update)
        
        # Schedule auto-save if enabled
        if self.auto_save:
            schedule.every(self.save_interval).minutes.do(self.auto_save_task)
        
        print(f"Starting paper trading scheduler with {interval_minutes} minute intervals")
        print("Press Ctrl+C to stop")
        
        # Run initial update immediately
        self.run_scheduled_update()
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nScheduler stopped by user")
            self.save_state()
    
    def run_manual(self):
        """
        Run the paper trading system in manual mode
        This provides an interactive command-line interface
        """
        commands = {
            'help': 'Show available commands',
            'status': 'Show current account status',
            'update': 'Update prices and check positions',
            'buy <symbol> [quantity]': 'Buy the specified symbol',
            'sell <symbol>': 'Sell the specified symbol',
            'signals <symbol>': 'Get trading signals for a symbol',
            'params': 'Show/edit trading parameters',
            'plot': 'Plot account performance',
            'save': 'Save current state',
            'exit': 'Exit the program'
        }
        
        print("\nPaper Trading System - Manual Mode")
        print("Type 'help' to see available commands")
        
        while True:
            cmd = input("\nCommand: ").strip().lower()
            
            if cmd == 'help':
                print("\nAvailable Commands:")
                for c, desc in commands.items():
                    print(f"  {c:<25} - {desc}")
                    
            elif cmd == 'status':
                self.print_status()
                
            elif cmd == 'update':
                self.run_scheduled_update()
                
            elif cmd.startswith('buy '):
                parts = cmd.split()
                if len(parts) < 2:
                    print("Usage: buy <symbol> [quantity]")
                    continue
                    
                symbol = parts[1].upper()
                quantity = float(parts[2]) if len(parts) > 2 else None
                
                self.open_position(symbol, quantity=quantity)
                
            elif cmd.startswith('sell '):
                parts = cmd.split()
                if len(parts) < 2:
                    print("Usage: sell <symbol>")
                    continue
                    
                symbol = parts[1].upper()
                self.close_position(symbol)
                
            elif cmd.startswith('signals '):
                parts = cmd.split()
                if len(parts) < 2:
                    print("Usage: signals <symbol>")
                    continue
                    
                symbol = parts[1].upper()
                signals = self.analyzer.get_simple_signals(symbol)
                
                print(f"\nSignals for {symbol}:")
                if "signals" in signals:
                    for signal in signals["signals"]:
                        print(f"  - {signal}")
                else:
                    print("  No signals generated")
                    
            elif cmd == 'params':
                print("\nTrading Parameters:")
                print(f"  Position Size: {self.position_size_pct}% of balance")
                print(f"  Stop Loss: {self.stop_loss_pct}%")
                print(f"  Take Profit: {self.take_profit_pct}%")
                print(f"  Max Positions: {self.max_positions}")
                
                edit = input("\nEdit parameters? (y/n): ").lower()
                if edit == 'y':
                    try:
                        size = input(f"Position Size % ({self.position_size_pct}): ")
                        if size:
                            self.position_size_pct = float(size)
                            
                        sl = input(f"Stop Loss % ({self.stop_loss_pct}): ")
                        if sl:
                            self.stop_loss_pct = float(sl)
                            
                        tp = input(f"Take Profit % ({self.take_profit_pct}): ")
                        if tp:
                            self.take_profit_pct = float(tp)
                            
                        mp = input(f"Max Positions ({self.max_positions}): ")
                        if mp:
                            self.max_positions = int(mp)
                            
                        print("Parameters updated")
                    except ValueError:
                        print("Invalid input. Parameters not updated.")
                        
            elif cmd == 'plot':
                os.makedirs("charts", exist_ok=True)
                chart_file = f"charts/paper_trading_performance.png"
                self.plot_performance(save_to_file=chart_file)
                
            elif cmd == 'save':
                self.save_state()
                
            elif cmd == 'exit':
                print("Saving state before exit...")
                self.save_state()
                print("Exiting paper trading system")
                break
                
            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' to see available commands")


# Simple demonstration
# Add this code to the end of paper_trader.py, replacing the existing __main__ block
# This correctly handles the interval variable

if __name__ == "__main__":
    print("Cryptocurrency Paper Trading System")
    print("----------------------------------")
    
    # Initialize with starting capital
    capital = float(input("Enter starting capital (default: $10000): ") or 10000)
    trader = PaperTrader(initial_capital=capital)
    
    # Check for saved state
    load_state = input("Load previous trading state? (y/n, default: n): ").lower()
    if load_state in ['y', 'yes']:
        trader.load_state()
    
    # Select mode
    mode = input("Select mode (manual/auto, default: manual): ").lower()
    
    if mode == 'auto':
        # Set update interval
        interval_str = input("Update interval in minutes (default: 15): ")
        interval = int(interval_str) if interval_str.isdigit() else 15
        
        # Apply enhanced strategy if user wants
        use_enhanced = input("Use enhanced strategy for limited data? (y/n, default: y): ").lower()
        if use_enhanced != 'n':
            try:
                from enhanced_strategy import apply_enhanced_strategy
                trader = apply_enhanced_strategy(trader)
            except ImportError:
                print("Enhanced strategy module not found. Using default strategy.")
        
        # Run in scheduled mode
        trader.run_scheduler(interval_minutes=interval)
    else:
        # Run in manual mode
        trader.run_manual()