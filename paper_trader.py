# paper_trader.py (updated with metrics)
import os
import json
import time
import schedule
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import threading
import traceback
from datetime import datetime, timedelta

# Import Prometheus metrics
from prometheus_client import Counter, Gauge, Summary, Histogram, start_http_server

# Import the centralized logging configuration
from utils.logging_config import get_module_logger

# Set up metrics
# Counters (these only go up)
TRADES_EXECUTED = Counter('paper_trader_trades_total', 'Number of trades executed', ['symbol', 'direction'])
SIGNALS_RECEIVED = Counter('paper_trader_signals_received_total', 'Number of trading signals received', ['symbol', 'signal_type'])
POSITIONS_CLOSED = Counter('paper_trader_positions_closed_total', 'Number of positions closed', ['symbol', 'reason'])
SAVE_OPERATIONS = Counter('paper_trader_save_operations_total', 'Number of state save operations')
LOAD_OPERATIONS = Counter('paper_trader_load_operations_total', 'Number of state load operations')
STRATEGY_RUNS = Counter('paper_trader_strategy_runs_total', 'Number of strategy execution runs')
PRICE_CHECKS = Counter('paper_trader_price_checks_total', 'Number of price check operations', ['symbol'])
ERRORS = Counter('paper_trader_errors_total', 'Number of errors encountered', ['operation', 'symbol'])

# Gauges (these can go up and down)
ACCOUNT_BALANCE = Gauge('paper_trader_balance', 'Current account balance')
ACCOUNT_EQUITY = Gauge('paper_trader_equity', 'Total account equity including positions')
POSITION_VALUE = Gauge('paper_trader_position_value', 'Value of open positions', ['symbol'])
POSITION_SIZE = Gauge('paper_trader_position_size', 'Size of open positions in units', ['symbol'])
POSITION_ENTRY_PRICE = Gauge('paper_trader_position_entry_price', 'Entry price of open positions', ['symbol'])
OPEN_POSITIONS_COUNT = Gauge('paper_trader_open_positions_count', 'Number of open positions')
UNREALIZED_PNL = Gauge('paper_trader_unrealized_pnl', 'Unrealized profit/loss of open positions', ['symbol'])
UNREALIZED_PNL_PCT = Gauge('paper_trader_unrealized_pnl_pct', 'Unrealized profit/loss percentage', ['symbol'])
DRAWDOWN_PCT = Gauge('paper_trader_drawdown_pct', 'Current drawdown as percentage')
POSITION_DURATION = Gauge('paper_trader_position_duration_hours', 'Duration of open positions in hours', ['symbol'])
TARGET_POSITION_SIZE_PCT = Gauge('paper_trader_target_position_size_pct', 'Target position size as percentage of equity')
STOP_LOSS_PCT = Gauge('paper_trader_stop_loss_pct', 'Stop loss percentage setting')
TAKE_PROFIT_PCT = Gauge('paper_trader_take_profit_pct', 'Take profit percentage setting')
MAX_POSITIONS = Gauge('paper_trader_max_positions', 'Maximum number of positions allowed')

# Summaries & Histograms (for distributions)
TRADE_EXECUTION_TIME = Summary('paper_trader_trade_execution_seconds', 'Time taken to execute trades', ['operation'])
TRADE_PNL = Histogram('paper_trader_trade_pnl', 'Distribution of trade profit/loss', 
                     ['symbol'], buckets=[-1000, -500, -200, -100, -50, 0, 50, 100, 200, 500, 1000])
TRADE_PNL_PCT = Histogram('paper_trader_trade_pnl_pct', 'Distribution of trade profit/loss percentage', 
                        ['symbol'], buckets=[-50, -25, -10, -5, -2, 0, 2, 5, 10, 25, 50])
POSITION_HOLD_TIME = Histogram('paper_trader_position_hold_time_hours', 'Distribution of position hold times in hours',
                             ['symbol'], buckets=[1, 6, 12, 24, 48, 72, 168, 336, 720])

# Get logger for this module
logger = get_module_logger("PaperTrader")

class PaperTrader:
    def __init__(self, initial_capital=10000):
        # Initialize core trading attributes
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.positions = {}
        self.trade_history = []
        self.performance_log = []
        self.position_size_pct = 0.95
        self.stop_loss_pct = 5.0
        self.take_profit_pct = 10.0
        self.max_positions = 3
        self.save_interval = 30  # minutes
        self.last_update_time = datetime.now()
        self.max_drawdown = 0.0
        self.peak_equity = initial_capital
        
        # Update strategy gauges
        TARGET_POSITION_SIZE_PCT.set(self.position_size_pct * 100)
        STOP_LOSS_PCT.set(self.stop_loss_pct)
        TAKE_PROFIT_PCT.set(self.take_profit_pct)
        MAX_POSITIONS.set(self.max_positions)
        
        # Initialize tools
        self.price_fetcher = self._init_price_fetcher()
        self.analyzer = self._init_analyzer()
        
        # Initialize data directories
        self.data_dir = "paper_trading"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create default state file if it doesn't exist
        self.current_state_file = os.path.join(self.data_dir, "current_state.json")
        if not os.path.exists(self.current_state_file):
            self.save_state()
            
        # Start metrics server
        self._start_metrics_server()
        
        # Update initial metrics
        self._log_performance()
        ACCOUNT_BALANCE.set(self.balance)
        ACCOUNT_EQUITY.set(self.balance)
        OPEN_POSITIONS_COUNT.set(0)
        
        logger.info(f"Initialized paper trader with ${initial_capital:.2f} starting capital")
    
    def _init_price_fetcher(self):
        """Initialize the price fetcher with error handling"""
        try:
            from multi_api_price_fetcher import CryptoPriceFetcher
            return CryptoPriceFetcher()
        except ImportError:
            logger.error("Could not import CryptoPriceFetcher. Make sure multi_api_price_fetcher.py is in the path.")
            ERRORS.labels(operation="initialization", symbol="all").inc()
            raise
    
    def _init_analyzer(self):
        """Initialize the crypto analyzer with error handling"""
        try:
            from crypto_analyzer import CryptoAnalyzer
            return CryptoAnalyzer(self.price_fetcher)
        except ImportError:
            logger.error("Could not import CryptoAnalyzer. Make sure crypto_analyzer.py is in the path.")
            ERRORS.labels(operation="initialization", symbol="all").inc()
            raise
    
    def _start_metrics_server(self):
        """Start the Prometheus metrics server"""
        try:
            metrics_port = int(os.environ.get("METRICS_PORT", 8003))
            
            # Try to start the server, handle if already running
            try:
                start_http_server(metrics_port)
                logger.info(f"Prometheus metrics server started on port {metrics_port}")
            except OSError as e:
                if "Address already in use" in str(e):
                    logger.info(f"Metrics server already running on port {metrics_port}")
                else:
                    raise
        except Exception as e:
            logger.error(f"Error starting metrics server: {e}")
            ERRORS.labels(operation="metrics_server", symbol="all").inc()
    
    def _log_performance(self):
        """Record current performance metrics"""
        try:
            # Calculate total value including open positions
            total_value = self.balance
            unrealized_pnl = 0
            
            for symbol, position in self.positions.items():
                current_price = self._get_current_price(symbol)
                if current_price is None:
                    continue
                    
                position_value = position['quantity'] * current_price
                total_value += position_value
                
                # Calculate unrealized P&L for this position
                entry_value = position['quantity'] * position['entry_price']
                position_pnl = position_value - entry_value
                unrealized_pnl += position_pnl
                
                # Update position-specific metrics
                POSITION_VALUE.labels(symbol=symbol).set(position_value)
                POSITION_SIZE.labels(symbol=symbol).set(position['quantity'])
                POSITION_ENTRY_PRICE.labels(symbol=symbol).set(position['entry_price'])
                UNREALIZED_PNL.labels(symbol=symbol).set(position_pnl)
                UNREALIZED_PNL_PCT.labels(symbol=symbol).set((position_pnl / entry_value) * 100 if entry_value > 0 else 0)
                
                # Calculate position duration
                if 'entry_time' in position:
                    entry_time = datetime.strptime(position['entry_time'], "%Y-%m-%d %H:%M:%S")
                    duration_hours = (datetime.now() - entry_time).total_seconds() / 3600
                    POSITION_DURATION.labels(symbol=symbol).set(duration_hours)
            
            # Update account metrics
            ACCOUNT_BALANCE.set(self.balance)
            ACCOUNT_EQUITY.set(total_value)
            OPEN_POSITIONS_COUNT.set(len(self.positions))
            
            # Update drawdown tracking
            if total_value > self.peak_equity:
                self.peak_equity = total_value
            
            current_drawdown_pct = ((self.peak_equity - total_value) / self.peak_equity) * 100 if self.peak_equity > 0 else 0
            DRAWDOWN_PCT.set(current_drawdown_pct)
            
            # Update max drawdown if needed
            if current_drawdown_pct > self.max_drawdown:
                self.max_drawdown = current_drawdown_pct
            
            # Create performance record
            record = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'balance': self.balance,
                'positions_value': total_value - self.balance,
                'unrealized_pnl': unrealized_pnl,
                'total_value': total_value,
                'return_pct': (total_value / self.initial_capital - 1) * 100,
                'drawdown_pct': current_drawdown_pct,
                'max_drawdown_pct': self.max_drawdown
            }
            
            self.performance_log.append(record)
            return record
        except Exception as e:
            logger.error(f"Error logging performance: {e}")
            ERRORS.labels(operation="performance_logging", symbol="all").inc()
            return None
    
    def _get_current_price(self, symbol):
        """Get current price for a symbol, handling potential errors"""
        PRICE_CHECKS.labels(symbol=symbol).inc()
        try:
            price = self.price_fetcher.get_price(symbol)
            if price is None:
                # If price fetch fails, check if we have recent price in history
                df = self.analyzer.convert_history_to_dataframe(symbol)
                if df is not None and not df.empty:
                    price = df['price'].iloc[-1]
                    logger.info(f"Using most recent stored price for {symbol}: ${price:.2f}")
                else:
                    logger.warning(f"Could not get price for {symbol}")
                    ERRORS.labels(operation="price_fetch", symbol=symbol).inc()
                    return None
            return price
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            ERRORS.labels(operation="price_fetch", symbol=symbol).inc()
            return None
    
    def open_position(self, symbol, price=None, quantity=None):
        """
        Open a new position in the specified symbol
        If price is None, fetch current market price
        If quantity is None, calculate based on position size percentage
        """
        # Time the execution
        with TRADE_EXECUTION_TIME.labels(operation="open").time():
            try:
                # Check if we're already holding this symbol
                if symbol in self.positions:
                    logger.info(f"Already holding {symbol}, cannot open new position")
                    return False
                
                # Check if we've reached the maximum number of positions
                if len(self.positions) >= self.max_positions:
                    logger.info(f"Maximum number of positions reached ({self.max_positions})")
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
                    logger.info(f"Insufficient balance to open position in {symbol}")
                    return False
                
                # Open the position
                self.positions[symbol] = {
                    'quantity': quantity,
                    'entry_price': price,
                    'entry_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'stop_loss': price * (1 - self.stop_loss_pct/100),
                    'take_profit': price * (1 + self.take_profit_pct/100),
                    'current_price': price
                }
                
                # Update balance
                self.balance -= position_value
                
                # Update metrics
                ACCOUNT_BALANCE.set(self.balance)
                POSITION_VALUE.labels(symbol=symbol).set(position_value)
                POSITION_SIZE.labels(symbol=symbol).set(quantity)
                POSITION_ENTRY_PRICE.labels(symbol=symbol).set(price)
                POSITION_DURATION.labels(symbol=symbol).set(0)  # Just opened
                OPEN_POSITIONS_COUNT.set(len(self.positions))
                TRADES_EXECUTED.labels(symbol=symbol, direction="buy").inc()
                
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
                
                logger.info(f"Opened {symbol} position: {quantity:.6f} @ ${price:.2f} = ${position_value:.2f}")
                self._log_performance()
                return True
            except Exception as e:
                logger.error(f"Error opening position for {symbol}: {e}")
                logger.debug(traceback.format_exc())
                ERRORS.labels(operation="position_open", symbol=symbol).inc()
                return False
    
    def close_position(self, symbol, price=None, reason="Manual"):
        """
        Close an existing position in the specified symbol
        If price is None, fetch current market price
        """
        # Time the execution
        with TRADE_EXECUTION_TIME.labels(operation="close").time():
            try:
                # Check if we're holding this symbol
                if symbol not in self.positions:
                    logger.info(f"No open position for {symbol}")
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
                
                # Update metrics
                ACCOUNT_BALANCE.set(self.balance)
                POSITION_VALUE.labels(symbol=symbol).set(0)  # Position closed
                POSITION_SIZE.labels(symbol=symbol).set(0)
                POSITION_ENTRY_PRICE.labels(symbol=symbol).set(0)
                POSITION_DURATION.labels(symbol=symbol).set(0)
                OPEN_POSITIONS_COUNT.set(len(self.positions) - 1)  # Will be removed
                TRADES_EXECUTED.labels(symbol=symbol, direction="sell").inc()
                POSITIONS_CLOSED.labels(symbol=symbol, reason=reason).inc()
                
                # Log the trade
                trade = {
                    'type': 'SELL',
                    'symbol': symbol,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'price': price,
                    'quantity': quantity,
                    'value': position_value,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct / 100,  # Store as decimal
                    'balance_after': self.balance,
                    'reason': reason
                }
                self.trade_history.append(trade)
                
                # Update trade metrics histograms
                TRADE_PNL.labels(symbol=symbol).observe(profit_loss)
                TRADE_PNL_PCT.labels(symbol=symbol).observe(profit_loss_pct)
                
                # Calculate hold time and update histogram
                if 'entry_time' in position:
                    entry_time = datetime.strptime(position['entry_time'], "%Y-%m-%d %H:%M:%S")
                    hold_time_hours = (datetime.now() - entry_time).total_seconds() / 3600
                    POSITION_HOLD_TIME.labels(symbol=symbol).observe(hold_time_hours)
                
                # Remove the position
                del self.positions[symbol]
                
                logger.info(f"Closed {symbol} position: {quantity:.6f} @ ${price:.2f} = ${position_value:.2f}")
                logger.info(f"Profit/Loss: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
                self._log_performance()
                return True
            except Exception as e:
                logger.error(f"Error closing position for {symbol}: {e}")
                logger.debug(traceback.format_exc())
                ERRORS.labels(operation="position_close", symbol=symbol).inc()
                return False
    
    def check_stop_loss_take_profit(self):
        """Check if any positions have hit stop loss or take profit levels"""
        for symbol in list(self.positions.keys()):
            price = self._get_current_price(symbol)
            if price is None:
                continue
                
            position = self.positions[symbol]
            
            # Update current price in position data
            position['current_price'] = price
            
            # Update unrealized P&L metrics
            entry_value = position['quantity'] * position['entry_price']
            current_value = position['quantity'] * price
            unrealized_pnl = current_value - entry_value
            unrealized_pnl_pct = (price / position['entry_price'] - 1) * 100
            
            UNREALIZED_PNL.labels(symbol=symbol).set(unrealized_pnl)
            UNREALIZED_PNL_PCT.labels(symbol=symbol).set(unrealized_pnl_pct)
            
            # Check stop loss
            if price <= position['stop_loss']:
                logger.info(f"Stop loss triggered for {symbol} at ${price:.2f}")
                self.close_position(symbol, price, reason="Stop Loss")
            
            # Check take profit
            elif price >= position['take_profit']:
                logger.info(f"Take profit triggered for {symbol} at ${price:.2f}")
                self.close_position(symbol, price, reason="Take Profit")
    
    def generate_signals(self, symbols=None):
        """Generate trading signals for the specified symbols"""
        STRATEGY_RUNS.inc()
        
        if symbols is None:
            symbols = ["BTC", "ETH", "SOL", "DOGE"]
        
        results = {}
        for symbol in symbols:
            try:
                signals = self.analyzer.get_simple_signals(symbol)
                results[symbol] = signals
                
                # Track signal metrics
                if signals and "status" in signals:
                    SIGNALS_RECEIVED.labels(symbol=symbol, signal_type=signals["status"]).inc()
                
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {e}")
                ERRORS.labels(operation="signal_generation", symbol=symbol).inc()
        
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
                    logger.info(f"Strong bearish signal for {symbol}, closing position")
                    self.close_position(symbol, reason="Bearish Signal")
            else:
                # No position - check if we should open one
                if bullish_signals >= 2:
                    logger.info(f"Strong bullish signal for {symbol}, opening position")
                    self.open_position(symbol)
    
    def save_state(self):
        """Save current state to files"""
        try:
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
                'peak_equity': self.peak_equity,
                'max_drawdown': self.max_drawdown,
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
                
            SAVE_OPERATIONS.inc()
            logger.info(f"Saved trader state at {timestamp}")
            return True
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            ERRORS.labels(operation="save_state", symbol="all").inc()
            return False
    
    def load_state(self, state_file=None):
        """Load state from files"""
        try:
            if state_file is None:
                state_file = f"{self.data_dir}/current_state.json"
                
            if not os.path.exists(state_file):
                logger.info(f"No saved state found at {state_file}")
                return False
                
            # Load current state
            with open(state_file, 'r') as f:
                state = json.load(f)
                
            self.balance = state['balance']
            self.positions = state['positions']
            
            # Load peak equity and max drawdown if available
            if 'peak_equity' in state:
                self.peak_equity = state['peak_equity']
            if 'max_drawdown' in state:
                self.max_drawdown = state['max_drawdown']
            
            # Load parameters if available
            if 'params' in state:
                params = state['params']
                self.position_size_pct = params.get('position_size_pct', self.position_size_pct)
                self.stop_loss_pct = params.get('stop_loss_pct', self.stop_loss_pct)
                self.take_profit_pct = params.get('take_profit_pct', self.take_profit_pct)
                self.max_positions = params.get('max_positions', self.max_positions)
                
                # Update parameter metrics
                TARGET_POSITION_SIZE_PCT.set(self.position_size_pct * 100)
                STOP_LOSS_PCT.set(self.stop_loss_pct)
                TAKE_PROFIT_PCT.set(self.take_profit_pct)
                MAX_POSITIONS.set(self.max_positions)
            
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
            
            # Update metrics
            self._log_performance()
            LOAD_OPERATIONS.inc()
                    
            logger.info(f"Loaded trader state from {state_file}")
            logger.info(f"Current balance: ${self.balance:.2f}")
            logger.info(f"Open positions: {len(self.positions)}")
            return True
                
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            ERRORS.labels(operation="load_state", symbol="all").inc()
            return False
    
    def update_prices(self, symbols=None):
        """Fetch latest prices for the specified symbols"""
        if symbols is None:
            symbols = ["BTC", "ETH", "SOL", "DOGE"]
                
        for symbol in symbols:
            price = self.price_fetcher.get_price(symbol)
            if price is not None:
                logger.info(f"Updated {symbol} price: ${price:.2f}")
                
                # If we have a position in this symbol, update its current price
                if symbol in self.positions:
                    self.positions[symbol]['current_price'] = price
    
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
                
                # Format entry time and calculate duration
                entry_time = datetime.strptime(position['entry_time'], '%Y-%m-%d %H:%M:%S')
                duration = datetime.now() - entry_time
                duration_hours = duration.total_seconds() / 3600
                
                print(f"  {symbol}: {position['quantity']:.6f} @ ${position['entry_price']:.2f} (now ${current_price:.2f})")
                print(f"     Value: ${position_value:.2f} | P/L: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
                print(f"     Duration: {duration_hours:.1f} hours")
                print(f"     Stop-Loss: ${position['stop_loss']:.2f} | Take-Profit: ${position['take_profit']:.2f}")
        else:
            print("\nNo open positions")
            
        # Calculate and print return metrics
        return_pct = (total_value / self.initial_capital - 1) * 100
        
        # Calculate drawdown
        drawdown_pct = ((self.peak_equity - total_value) / self.peak_equity) * 100 if self.peak_equity > 0 else 0
        
        print(f"\nTotal Account Value: ${total_value:.2f}")
        print(f"Return: {return_pct:.2f}%")
        print(f"Current Drawdown: {drawdown_pct:.2f}%")
        print(f"Max Drawdown: {self.max_drawdown:.2f}%")
        
        # Calculate and print win rate
        sell_trades = [t for t in self.trade_history if t['type'] == 'SELL']
        if sell_trades:
            winning_trades = [t for t in sell_trades if t.get('profit_loss', 0) > 0]
            win_rate = len(winning_trades) / len(sell_trades) * 100
            
            total_profit = sum(t.get('profit_loss', 0) for t in winning_trades)
            total_loss = abs(sum(t.get('profit_loss', 0) for t in sell_trades if t.get('profit_loss', 0) <= 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            print(f"Win Rate: {win_rate:.2f}% ({len(winning_trades)}/{len(sell_trades)})")
            print(f"Profit Factor: {profit_factor:.2f}")
        
        print(f"Total Trades: {len(self.trade_history)}")
        print("===================================\n")
    
    def plot_performance(self, save_to_file=None):
        """Plot account performance over time"""
        if not self.performance_log:
            logger.warning("No performance data to plot")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(self.performance_log)
        
        # Convert timestamp strings to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1.5, 1.5]})
        
        # Plot 1: Equity curve
        ax1.plot(df.index, df['total_value'], 'b-', linewidth=2, label='Total Equity')
        ax1.plot(df.index, df['balance'], 'g--', linewidth=1.5, label='Cash Balance')
        
        if 'positions_value' in df.columns:
            ax1.fill_between(df.index, df['balance'], df['total_value'], 
                            color='lightblue', alpha=0.3, label='Position Value')
        
        ax1.set_title('Paper Trading Account Performance', fontsize=14)
        ax1.set_ylabel('Value ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Add trades to the equity curve
        if self.trade_history:
            buy_trades = [trade for trade in self.trade_history if trade['type'] == 'BUY']
            sells = [trade for trade in self.trade_history if trade['type'] == 'SELL']
            
            if buy_trades:
                buy_times = pd.to_datetime([trade['timestamp'] for trade in buy_trades])
                buy_values = [trade['value'] for trade in buy_trades]
                ax1.scatter(buy_times, buy_values, marker='^', color='green', s=80, label='Buy')
                
            if sells:
                sell_times = pd.to_datetime([trade['timestamp'] for trade in sells])
                sell_values = [trade['value'] for trade in sells]
                ax1.scatter(sell_times, sell_values, marker='v', color='red', s=80, label='Sell')
                
                # Add profit/loss annotations
                for trade in sells:
                    time = pd.to_datetime(trade['timestamp'])
                    value = trade['value']
                    pnl = trade.get('profit_loss', 0)
                    pnl_pct = trade.get('profit_loss_pct', 0) * 100
                    reason = trade.get('reason', 'Manual')
                    
                    # Color code based on profit/loss
                    color = 'green' if pnl > 0 else 'red'
                    
                    # Add annotation with arrow
                    ax1.annotate(f"${pnl:.0f}\n({pnl_pct:.1f}%)\n{reason}", 
                                xy=(time, value),
                                xytext=(10, 20),
                                textcoords="offset points",
                                arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                                color=color,
                                fontweight='bold')
                
            # Update legend with trade markers
            if buy_trades or sells:
                ax1.legend(loc='upper left')
        
        # Plot 2: Return percentage
        if 'return_pct' in df.columns:
            ax2.plot(df.index, df['return_pct'], 'b-', label='Return %')
            ax2.set_ylabel('Return (%)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            ax2.legend(loc='upper left')
            
            # Fill above/below zero
            ax2.fill_between(df.index, 0, df['return_pct'], 
                            where=(df['return_pct'] >= 0), 
                            color='green', alpha=0.2)
            ax2.fill_between(df.index, 0, df['return_pct'], 
                            where=(df['return_pct'] < 0), 
                            color='red', alpha=0.2)
            
        # Plot 3: Drawdown
        if 'drawdown_pct' in df.columns:
            ax3.fill_between(df.index, 0, df['drawdown_pct'], color='red', alpha=0.3)
            ax3.plot(df.index, df['drawdown_pct'], 'r-', linewidth=1.5, label='Drawdown')
            
            # Add max drawdown line
            if 'max_drawdown_pct' in df.columns and not df['max_drawdown_pct'].empty:
                max_dd = df['max_drawdown_pct'].max()
                ax3.axhline(y=max_dd, color='darkred', linestyle='--', 
                           label=f'Max Drawdown: {max_dd:.2f}%')
                
            ax3.set_title('Drawdown (%)', fontsize=12)
            ax3.set_ylabel('Drawdown (%)', fontsize=12)
            ax3.set_ylim(top=1, bottom=max(df['drawdown_pct'].max() * 1.5, 5))  # Ensure we see the drawdown
            ax3.invert_yaxis()  # Invert so drawdowns go down
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='lower left')
        
        # Add performance metrics summary
        if len(self.trade_history) > 0:
            sell_trades = [t for t in self.trade_history if t['type'] == 'SELL']
            if sell_trades:
                winning_trades = [t for t in sell_trades if t.get('profit_loss', 0) > 0]
                win_rate = len(winning_trades) / len(sell_trades) * 100
                
                total_profit = sum(t.get('profit_loss', 0) for t in winning_trades)
                total_loss = abs(sum(t.get('profit_loss', 0) for t in sell_trades if t.get('profit_loss', 0) <= 0))
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                
                # Add metrics as text
                metrics = [
                    f"Initial Capital: ${self.initial_capital:,.2f}",
                    f"Current Equity: ${df['total_value'].iloc[-1]:,.2f}",
                    f"Total Return: {df['return_pct'].iloc[-1]:.2f}%",
                    f"Win Rate: {win_rate:.2f}% ({len(winning_trades)}/{len(sell_trades)})",
                    f"Profit Factor: {profit_factor:.2f}",
                    f"Max Drawdown: {df['max_drawdown_pct'].max():.2f}%",
                    f"Total Trades: {len(self.trade_history)}"
                ]
                
                # Add metrics to plot
                plt.figtext(0.15, 0.01, '\n'.join(metrics), fontsize=12, 
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)  # Make room for metrics
        
        # Save to file if requested
        if save_to_file:
            plt.savefig(save_to_file, dpi=300, bbox_inches='tight')
            logger.info(f"Performance chart saved to {save_to_file}")
        else:
            plt.show()
            
        # Close the plot to free memory
        plt.close(fig)
    
    def run_scheduled_update(self):
        """Run a scheduled update to check prices and signals"""
        try:
            update_start = time.time()
            logger.info(f"=== Scheduled Update {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            
            symbols = list(self.positions.keys())
            if len(symbols) < self.max_positions:
                # Add some symbols to check if we have room for more positions
                for s in ["BTC", "ETH", "SOL", "DOGE"]:
                    if s not in symbols:
                        symbols.append(s)
            
            # Update market data
            logger.info("Updating market data...")
            self.update_prices(symbols)
            
            # Check open positions
            logger.info("Checking open positions...")
            self.check_stop_loss_take_profit()
            
            # Generate and process signals
            logger.info("Generating trading signals...")
            signals = self.generate_signals(symbols)
            self.process_signals(signals)
            
            # Log performance
            self._log_performance()
            
            # Save state every so often
            time_since_save = (datetime.now() - self.last_update_time).total_seconds() / 60
            if time_since_save >= self.save_interval:
                self.save_state()
                self.last_update_time = datetime.now()
            
            update_duration = time.time() - update_start
            logger.info(f"Update complete in {update_duration:.2f} seconds")
        except Exception as e:
            logger.error(f"Error in scheduled update: {e}")
            logger.debug(traceback.format_exc())
            ERRORS.labels(operation="scheduled_update", symbol="all").inc()
    
    def run_scheduler(self, interval_minutes=15):
        """
        Run the paper trading system with a scheduler
        This will update at regular intervals
        """
        try:
            # Set auto-save flag
            self.auto_save = True
            
            # Schedule regular updates
            schedule.every(interval_minutes).minutes.do(self.run_scheduled_update)
            
            # Calculate start of the next hour for report generation
            next_hour = datetime.now().replace(minute=0, second=0) + timedelta(hours=1)
            midnight = datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)
            
            # Schedule report generation at the start of every hour
            schedule.every().hour.at(":00").do(self._generate_hourly_report)
            
            # Schedule daily report at midnight
            schedule.every().day.at("00:00").do(self._generate_daily_report)
            
            logger.info(f"Starting paper trading scheduler with {interval_minutes} minute intervals")
            logger.info("Hourly reports will be generated at the start of each hour")
            logger.info("Daily reports will be generated at midnight")
            logger.info("Press Ctrl+C to stop")
            
            # Run initial update immediately
            self.run_scheduled_update()
            
            try:
                while True:
                    schedule.run_pending()
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("\nScheduler stopped by user")
                self.save_state()
                
        except Exception as e:
            logger.error(f"Error in scheduler: {e}")
            logger.debug(traceback.format_exc())
            ERRORS.labels(operation="scheduler", symbol="all").inc()
    
    def _generate_hourly_report(self):
        """Generate hourly performance report"""
        try:
            # Generate timestamp for the report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            
            # Create reports directory
            report_dir = os.path.join(self.data_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            
            # Save performance chart
            chart_file = os.path.join(report_dir, f"performance_{timestamp}.png")
            self.plot_performance(save_to_file=chart_file)
            
            # Log performance metrics
            perf = self._log_performance()
            
            # Save summary to text file
            summary_file = os.path.join(report_dir, f"summary_{timestamp}.txt")
            
            with open(summary_file, 'w') as f:
                f.write(f"Paper Trading Hourly Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("Current Portfolio Status:\n")
                f.write(f"Cash Balance: ${self.balance:.2f}\n")
                
                if perf:
                    f.write(f"Total Equity: ${perf['total_value']:.2f}\n")
                    f.write(f"Return: {perf['return_pct']:.2f}%\n")
                    f.write(f"Drawdown: {perf['drawdown_pct']:.2f}%\n")
                    f.write(f"Max Drawdown: {perf['max_drawdown_pct']:.2f}%\n\n")
                
                # Add position information
                if self.positions:
                    f.write("Open Positions:\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"{'Symbol':<8} {'Quantity':<12} {'Entry Price':<12} {'Current':<10} {'P/L':<10} {'P/L %':<8}\n")
                    
                    for symbol, position in self.positions.items():
                        current_price = position.get('current_price', self._get_current_price(symbol))
                        if current_price:
                            entry_price = position['entry_price']
                            quantity = position['quantity']
                            pnl = (current_price - entry_price) * quantity
                            pnl_pct = (current_price / entry_price - 1) * 100
                            
                            f.write(f"{symbol:<8} {quantity:<12.6f} ${entry_price:<11.2f} ${current_price:<9.2f} ${pnl:<9.2f} {pnl_pct:<8.2f}%\n")
                    f.write("-" * 60 + "\n\n")
                else:
                    f.write("No open positions\n\n")
                
                # Add recent trades
                recent_trades = self.trade_history[-5:] if len(self.trade_history) > 5 else self.trade_history
                if recent_trades:
                    f.write("Recent Trades:\n")
                    f.write("-" * 60 + "\n")
                    
                    for trade in reversed(recent_trades):
                        trade_type = trade['type']
                        symbol = trade['symbol']
                        timestamp = trade['timestamp']
                        price = trade['price']
                        quantity = trade['quantity']
                        value = trade['value']
                        
                        trade_info = f"{timestamp} - {trade_type} {quantity:.6f} {symbol} @ ${price:.2f} = ${value:.2f}"
                        
                        if trade_type == 'SELL':
                            profit_loss = trade.get('profit_loss', 0)
                            profit_loss_pct = trade.get('profit_loss_pct', 0) * 100
                            reason = trade.get('reason', 'Manual')
                            
                            trade_info += f" | P/L: ${profit_loss:.2f} ({profit_loss_pct:.2f}%) - {reason}"
                            
                        f.write(f"{trade_info}\n")
                
            logger.info(f"Hourly report generated at {report_dir}")
            return True
        except Exception as e:
            logger.error(f"Error generating hourly report: {e}")
            logger.debug(traceback.format_exc())
            ERRORS.labels(operation="report_generation", symbol="all").inc()
            return False
    
    def _generate_daily_report(self):
        """Generate daily performance report with more detailed analytics"""
        try:
            # Create reports directory
            report_dir = os.path.join(self.data_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            
            # Generate timestamp for the report
            timestamp = datetime.now().strftime("%Y%m%d")
            
            # Save performance chart
            chart_file = os.path.join(report_dir, f"daily_performance_{timestamp}.png")
            self.plot_performance(save_to_file=chart_file)
            
            # Generate detailed analytics if we have trade history
            if self.trade_history:
                # Convert to DataFrame for analysis
                trades_df = pd.DataFrame(self.trade_history)
                
                # Filter sell trades for P&L analysis
                sell_trades = trades_df[trades_df['type'] == 'SELL'].copy()
                
                if not sell_trades.empty:
                    # Calculate trade metrics
                    sell_trades['timestamp'] = pd.to_datetime(sell_trades['timestamp'])
                    
                    # Generate trade analytics
                    win_count = len(sell_trades[sell_trades['profit_loss'] > 0])
                    loss_count = len(sell_trades) - win_count
                    win_rate = win_count / len(sell_trades) * 100 if len(sell_trades) > 0 else 0
                    
                    total_profit = sell_trades[sell_trades['profit_loss'] > 0]['profit_loss'].sum()
                    total_loss = sell_trades[sell_trades['profit_loss'] <= 0]['profit_loss'].sum()
                    net_pnl = total_profit + total_loss
                    
                    avg_win = total_profit / win_count if win_count > 0 else 0
                    avg_loss = total_loss / loss_count if loss_count > 0 else 0
                    
                    profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
                    
                    # Group by symbol
                    symbol_performance = sell_trades.groupby('symbol').agg({
                        'profit_loss': 'sum',
                        'type': 'count'
                    }).rename(columns={'type': 'trade_count'})
                    
                    # Group by reason
                    reason_counts = sell_trades['reason'].value_counts().to_dict()
                    
                    # Save analytics to file
                    analytics_file = os.path.join(report_dir, f"daily_analytics_{timestamp}.json")
                    
                    analytics = {
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'total_trades': len(trades_df),
                        'total_buys': len(trades_df[trades_df['type'] == 'BUY']),
                        'total_sells': len(sell_trades),
                        'win_count': int(win_count),
                        'loss_count': int(loss_count),
                        'win_rate': float(win_rate),
                        'total_profit': float(total_profit),
                        'total_loss': float(total_loss),
                        'net_pnl': float(net_pnl),
                        'avg_win': float(avg_win),
                        'avg_loss': float(avg_loss),
                        'profit_factor': float(profit_factor),
                        'initial_capital': float(self.initial_capital),
                        'current_balance': float(self.balance),
                        'equity': float(self._log_performance()['total_value']),
                        'return_pct': float(self._log_performance()['return_pct']),
                        'max_drawdown': float(self.max_drawdown),
                        'symbol_performance': symbol_performance.to_dict(),
                        'exit_reasons': reason_counts
                    }
                    
                    with open(analytics_file, 'w') as f:
                        json.dump(analytics, f, indent=2, default=str)
                
                # Create trade distribution charts
                if len(sell_trades) >= 5:  # Only if we have enough trades for meaningful charts
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # P&L Distribution
                    sell_trades['profit_loss'].hist(bins=10, ax=ax1, color='skyblue', edgecolor='black')
                    ax1.set_title('Profit/Loss Distribution', fontsize=14)
                    ax1.set_xlabel('Profit/Loss ($)', fontsize=12)
                    ax1.set_ylabel('Number of Trades', fontsize=12)
                    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                    ax1.grid(True, alpha=0.3)
                    
                    # P&L by Symbol
                    symbol_pnl = sell_trades.groupby('symbol')['profit_loss'].sum()
                    colors = ['green' if x > 0 else 'red' for x in symbol_pnl]
                    symbol_pnl.sort_values().plot(kind='barh', ax=ax2, color=colors)
                    ax2.set_title('Profit/Loss by Symbol', fontsize=14)
                    ax2.set_xlabel('Profit/Loss ($)', fontsize=12)
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    chart_file = os.path.join(report_dir, f"trade_distribution_{timestamp}.png")
                    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
                    plt.close()
            
            logger.info(f"Daily report generated at {report_dir}")
            return True
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
            logger.debug(traceback.format_exc())
            ERRORS.labels(operation="daily_report", symbol="all").inc()
            return False
    
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
            'report': 'Generate performance report',
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
                print(f"  Position Size: {self.position_size_pct * 100}% of balance")
                print(f"  Stop Loss: {self.stop_loss_pct}%")
                print(f"  Take Profit: {self.take_profit_pct}%")
                print(f"  Max Positions: {self.max_positions}")
                
                edit = input("\nEdit parameters? (y/n): ").lower()
                if edit == 'y':
                    try:
                        size = input(f"Position Size % ({self.position_size_pct * 100}): ")
                        if size:
                            new_size = float(size) / 100
                            self.position_size_pct = new_size
                            TARGET_POSITION_SIZE_PCT.set(new_size * 100)
                            
                        sl = input(f"Stop Loss % ({self.stop_loss_pct}): ")
                        if sl:
                            new_sl = float(sl)
                            self.stop_loss_pct = new_sl
                            STOP_LOSS_PCT.set(new_sl)
                            
                        tp = input(f"Take Profit % ({self.take_profit_pct}): ")
                        if tp:
                            new_tp = float(tp)
                            self.take_profit_pct = new_tp
                            TAKE_PROFIT_PCT.set(new_tp)
                            
                        mp = input(f"Max Positions ({self.max_positions}): ")
                        if mp:
                            new_mp = int(mp)
                            self.max_positions = new_mp
                            MAX_POSITIONS.set(new_mp)
                            
                        print("Parameters updated")
                    except ValueError:
                        print("Invalid input. Parameters not updated.")
                        
            elif cmd == 'plot':
                os.makedirs("charts", exist_ok=True)
                chart_file = f"charts/paper_trading_performance.png"
                self.plot_performance(save_to_file=chart_file)
                print(f"Performance chart saved to {chart_file}")
                
            elif cmd == 'save':
                self.save_state()
                
            elif cmd == 'report':
                self._generate_hourly_report()
                print("Report generated in the reports directory")
                
            elif cmd == 'exit':
                print("Saving state before exit...")
                self.save_state()
                print("Exiting paper trading system")
                break
                
            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' to see available commands")


# Add this code to the end of paper_trader.py
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cryptocurrency Paper Trading System")
    parser.add_argument("--capital", type=float, default=10000, help="Starting capital")
    parser.add_argument("--mode", choices=["auto", "manual"], default="manual", help="Trading mode")
    parser.add_argument("--interval", type=int, default=15, help="Update interval in minutes (auto mode)")
    parser.add_argument("--config", type=str, help="Configuration file (not required)")
    parser.add_argument("--metrics-port", type=int, default=8003, help="Port for Prometheus metrics")
    args = parser.parse_args()
    
    # Set metrics port in environment
    os.environ["METRICS_PORT"] = str(args.metrics_port)
    
    print("Cryptocurrency Paper Trading System")
    print("----------------------------------")
    
    # Initialize with starting capital
    trader = PaperTrader(initial_capital=args.capital)
    
    # Check for saved state
    load_state = input("Load previous trading state? (y/n, default: n): ").lower()
    if load_state in ['y', 'yes']:
        trader.load_state()
    
    if args.mode == 'auto':
        # Apply enhanced strategy if user wants
        use_enhanced = input("Use enhanced trading strategy? (y/n, default: y): ").lower()
        if use_enhanced != 'n':
            try:
                from enhanced_strategy import apply_enhanced_strategy
                trader = apply_enhanced_strategy(trader)
                logger.info("Enhanced trading strategy applied")
            except ImportError:
                logger.warning("Enhanced strategy module not found. Using default strategy.")
        
        # Run in scheduled mode
        trader.run_scheduler(interval_minutes=args.interval)
    else:
        # Run in manual mode
        trader.run_manual()