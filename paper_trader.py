import os
import json
import time
import schedule
import pandas as pd
import matplotlib.pyplot as plt
import threading
from datetime import datetime, timedelta

# Import the centralized logging configuration
from utils.logging_config import get_module_logger

# Get logger for this module
logger = get_module_logger("PaperTrader")

# Import other necessary modules
from multi_api_price_fetcher import CryptoPriceFetcher
from crypto_analyzer import CryptoAnalyzer

class StrategyManager:
    """Manages and loads trading strategies dynamically"""
    
    def __init__(self):
        self.available_strategies = {}
        self.active_strategy = None
        self.strategy_params = {}
        
        # Register built-in strategies
        self.register_built_in_strategies()
    
    def register_built_in_strategies(self):
        """Register the built-in strategies"""
        # Simple moving average crossover strategy
        self.register_strategy(
            "ma_crossover", 
            self.ma_crossover_strategy,
            {"short_period": 5, "long_period": 20}
        )
        
        # RSI strategy
        self.register_strategy(
            "rsi", 
            self.rsi_strategy,
            {"period": 14, "oversold": 30, "overbought": 70}
        )
        
        # Sentiment-based strategy
        self.register_strategy(
            "sentiment", 
            self.sentiment_strategy,
            {"sentiment_threshold": 3.0, "lookback_days": 3}
        )
        
        # Enhanced hybrid strategy
        self.register_strategy(
            "enhanced_hybrid", 
            self.enhanced_hybrid_strategy,
            {"sentiment_weight": 0.6, "technical_weight": 0.4}
        )
    
    def register_strategy(self, name, strategy_func, default_params=None):
        """Register a new strategy"""
        self.available_strategies[name] = {
            "function": strategy_func,
            "default_params": default_params or {}
        }
        print(f"Registered strategy: {name}")
    
    def load_external_strategy(self, module_name, strategy_name=None):
        """Dynamically load a strategy from an external Python module"""
        try:
            import importlib
            module = importlib.import_module(module_name)
            
            if strategy_name is None:
                # If no specific strategy name is provided, look for a 'get_strategy' function
                if hasattr(module, 'get_strategy'):
                    strategy_func = module.get_strategy()
                    name = module_name.split('.')[-1]
                    params = getattr(module, 'default_params', {})
                    self.register_strategy(name, strategy_func, params)
                    return name
                else:
                    raise ValueError(f"Module {module_name} does not contain a get_strategy function")
            else:
                # Load a specific strategy from the module
                if hasattr(module, strategy_name):
                    strategy_func = getattr(module, strategy_name)
                    params = getattr(module, 'default_params', {})
                    self.register_strategy(strategy_name, strategy_func, params)
                    return strategy_name
                else:
                    raise ValueError(f"Module {module_name} does not contain strategy {strategy_name}")
        except Exception as e:
            print(f"Error loading external strategy: {e}")
            return None
    
    def get_available_strategies(self):
        """Get list of available strategies"""
        return list(self.available_strategies.keys())
    
    def set_active_strategy(self, strategy_name, custom_params=None):
        """Set the active strategy with optional custom parameters"""
        if strategy_name not in self.available_strategies:
            print(f"Strategy '{strategy_name}' not found")
            return False
        
        self.active_strategy = strategy_name
        
        # Start with default parameters
        self.strategy_params = self.available_strategies[strategy_name]["default_params"].copy()
        
        # Update with custom parameters if provided
        if custom_params:
            self.strategy_params.update(custom_params)
            
        print(f"Activated strategy: {strategy_name} with parameters: {self.strategy_params}")
        return True
    
    def generate_signals(self, df):
        """Generate trading signals using the active strategy"""
        if not self.active_strategy:
            print("No active strategy selected")
            return None
        
        strategy_func = self.available_strategies[self.active_strategy]["function"]
        return strategy_func(df, self.strategy_params)
    
    # Built-in strategy implementations
    def ma_crossover_strategy(self, df, params):
        """Moving average crossover strategy"""
        if len(df) < params["long_period"]:
            return None
            
        result = df.copy()
        result['short_ma'] = result['price'].rolling(window=params["short_period"]).mean()
        result['long_ma'] = result['price'].rolling(window=params["long_period"]).mean()
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
    
    def rsi_strategy(self, df, params):
        """RSI-based strategy"""
        if len(df) < params["period"]:
            return None
            
        result = df.copy()
        
        # Calculate RSI
        delta = result['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=params["period"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=params["period"]).mean()
        
        rs = gain / loss.replace(0, 1e-10)  # Avoid division by zero
        result['rsi'] = 100 - (100 / (1 + rs))
        
        result['signal'] = 0
        result['signal_strength'] = 0
        
        for i in range(1, len(result)):
            if pd.isna(result['rsi'].iloc[i]):
                continue
                
            # Buy signal: RSI crosses above oversold level
            if result['rsi'].iloc[i] > params["oversold"] and result['rsi'].iloc[i-1] <= params["oversold"]:
                result.loc[result.index[i], 'signal'] = 1
                # Signal strength increases as RSI gets more oversold
                strength = (params["oversold"] - min(result['rsi'].iloc[i-1], params["oversold"])) / params["oversold"]
                result.loc[result.index[i], 'signal_strength'] = min(1.0, max(0.1, 0.5 + strength))
                
            # Sell signal: RSI crosses below overbought level
            elif result['rsi'].iloc[i] < params["overbought"] and result['rsi'].iloc[i-1] >= params["overbought"]:
                result.loc[result.index[i], 'signal'] = -1
                # Signal strength increases as RSI gets more overbought
                strength = (min(result['rsi'].iloc[i-1], 100) - params["overbought"]) / (100 - params["overbought"])
                result.loc[result.index[i], 'signal_strength'] = min(1.0, max(0.1, 0.5 + strength))
        
        return result
    
    def sentiment_strategy(self, df, params):
        """Pure sentiment-based strategy"""
        from database_manager import DatabaseManager
        
        result = df.copy()
        result['signal'] = 0
        result['signal_strength'] = 0
        
        try:
            # Get recent sentiment data
            db = DatabaseManager()
            sentiment_data = db.get_aggregated_sentiment(days=params["lookback_days"])
            
            if not sentiment_data:
                print("No sentiment data available")
                return result
            
            # Calculate average sentiment score
            total_score = sum(record.get('avg_sentiment', 0) for record in sentiment_data)
            avg_sentiment = total_score / len(sentiment_data)
            
            # Apply signals based on sentiment threshold
            if avg_sentiment > params["sentiment_threshold"]:
                # Strong bullish sentiment
                for i in range(len(result) - 1, max(0, len(result) - 5), -1):
                    # Apply to recent data points
                    result.loc[result.index[i], 'signal'] = 1
                    result.loc[result.index[i], 'signal_strength'] = min(1.0, avg_sentiment / 10)
            elif avg_sentiment < -params["sentiment_threshold"]:
                # Strong bearish sentiment
                for i in range(len(result) - 1, max(0, len(result) - 5), -1):
                    # Apply to recent data points
                    result.loc[result.index[i], 'signal'] = -1
                    result.loc[result.index[i], 'signal_strength'] = min(1.0, abs(avg_sentiment) / 10)
        except Exception as e:
            print(f"Error in sentiment strategy: {e}")
        
        return result
    
    def enhanced_hybrid_strategy(self, df, params):
        """Hybrid strategy combining technical and sentiment signals"""
        # Get technical signals (MA crossover)
        tech_params = {"short_period": 5, "long_period": 20}
        tech_result = self.ma_crossover_strategy(df, tech_params)
        
        if tech_result is None:
            return None
            
        # Get sentiment signals
        sentiment_params = {"sentiment_threshold": 2.0, "lookback_days": 3}
        sentiment_result = self.sentiment_strategy(df, sentiment_params)
        
        # Combine signals
        result = tech_result.copy()
        
        # For each row where we have both technical and sentiment data
        for i in range(len(result)):
            tech_signal = tech_result['signal'].iloc[i]
            tech_strength = tech_result['signal_strength'].iloc[i]
            
            sentiment_signal = 0
            sentiment_strength = 0
            
            if i < len(sentiment_result):
                sentiment_signal = sentiment_result['signal'].iloc[i]
                sentiment_strength = sentiment_result['signal_strength'].iloc[i]
            
            # If both signals agree, boost the signal
            if tech_signal != 0 and tech_signal == sentiment_signal:
                # Weighted combination of signal strengths
                combined_strength = (params["technical_weight"] * tech_strength + 
                                     params["sentiment_weight"] * sentiment_strength)
                result.loc[result.index[i], 'signal_strength'] = min(1.0, combined_strength)
            elif sentiment_signal != 0 and tech_signal == 0:
                # Only sentiment signal exists - apply with sentiment weight
                result.loc[result.index[i], 'signal'] = sentiment_signal
                result.loc[result.index[i], 'signal_strength'] = sentiment_strength * params["sentiment_weight"]
            elif tech_signal != 0 and sentiment_signal == 0:
                # Only technical signal exists - keep as is, scaled by technical weight
                result.loc[result.index[i], 'signal_strength'] = tech_strength * params["technical_weight"]
            elif tech_signal != 0 and sentiment_signal != 0 and tech_signal != sentiment_signal:
                # Signals conflict - use the one with higher priority based on weights
                if params["technical_weight"] >= params["sentiment_weight"]:
                    # Technical signals have priority
                    result.loc[result.index[i], 'signal'] = tech_signal
                    result.loc[result.index[i], 'signal_strength'] = tech_strength * 0.7  # Reduce strength due to conflict
                else:
                    # Sentiment signals have priority
                    result.loc[result.index[i], 'signal'] = sentiment_signal
                    result.loc[result.index[i], 'signal_strength'] = sentiment_strength * 0.7  # Reduce strength due to conflict
        
        return result


class PaperTrader:
    def __init__(self, initial_capital=10000):
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
        self.auto_save = True  # Added this flag to fix run_scheduler
        
        self.price_fetcher = CryptoPriceFetcher()
        self.analyzer = CryptoAnalyzer(self.price_fetcher)
        self.strategy_manager = StrategyManager()
        
        self.data_dir = "paper_trading"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create default state file if it doesn't exist
        self.current_state_file = os.path.join(self.data_dir, "current_state.json")
        if not os.path.exists(self.current_state_file):
            self.save_state()
            
        self._log_performance()
        logger.info(f"Initialized paper trader with ${initial_capital:.2f} starting capital")
    
    def _log_performance(self):
        """Record current performance metrics"""
        # Calculate total value including open positions
        total_value = self.balance
        for symbol, position in self.positions.items():
            current_price = self._get_current_price(symbol)
            if current_price is not None:  # Added check to prevent NoneType error
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
            position_value = self.balance * self.position_size_pct / 100  # Fixed: Convert percentage to decimal
            
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
        """Generate trading signals for the specified symbols using the active strategy"""
        if symbols is None:
            symbols = ["BTC", "ETH", "SOL", "DOGE"]
        
        results = {}
        
        for symbol in symbols:
            # Get historical data
            df = self.analyzer.convert_history_to_dataframe(symbol)
            
            if df is not None and len(df) > 10:  # Need at least some data points
                # Apply strategy
                if self.strategy_manager.active_strategy:
                    signal_df = self.strategy_manager.generate_signals(df)
                    
                    if signal_df is not None:
                        # Extract latest signals
                        latest = signal_df.iloc[-1]
                        results[symbol] = {
                            'signal': latest.get('signal', 0),
                            'signal_strength': latest.get('signal_strength', 0),
                            'current_price': latest.get('price', 0)
                        }
                        
                        logger.info(f"Signals for {symbol}: Signal={latest.get('signal', 0)}, Strength={latest.get('signal_strength', 0):.2f}")
                else:
                    # Use the default strategy from the analyzer
                    signals = self.analyzer.get_simple_signals(symbol)
                    if signals:
                        results[symbol] = {
                            'signal': 1 if "BULLISH" in str(signals) else (-1 if "BEARISH" in str(signals) else 0),
                            'signal_strength': 0.5,  # Default strength
                            'current_price': signals.get('price', 0)
                        }
            else:
                logger.warning(f"Not enough data for signal generation for {symbol}")
        
        return results
    
    def process_signals(self, signals):
        """Process trading signals to make trade decisions"""
        for symbol, signal_data in signals.items():
            if "signals" not in signal_data:
                # Check if we have the new format
                if "signal" in signal_data:
                    signal = signal_data["signal"]
                    signal_strength = signal_data.get("signal_strength", 0.5)
                    
                    # Simple decision logic
                    if symbol in self.positions:
                        # Already have a position - check if we should close it
                        if signal == -1 and signal_strength > 0.5:
                            print(f"Strong sell signal for {symbol}, closing position")
                            self.close_position(symbol, reason="Bearish Signal")
                    else:
                        # No position - check if we should open one
                        if signal == 1 and signal_strength > 0.5:
                            print(f"Strong buy signal for {symbol}, opening position")
                            self.open_position(symbol)
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
            'strategies': 'List available trading strategies',
            'strategy <name> [params]': 'Set active trading strategy',
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
                signals = self.generate_signals([symbol])
                
                if symbol in signals:
                    signal = signals[symbol]
                    signal_type = "BUY" if signal['signal'] == 1 else "SELL" if signal['signal'] == -1 else "NEUTRAL"
                    print(f"\nSignals for {symbol}:")
                    print(f"  Signal: {signal_type}")
                    print(f"  Strength: {signal['signal_strength']:.2f}")
                    print(f"  Current Price: ${signal['current_price']:.2f}")
                else:
                    print(f"No signals generated for {symbol}")
                    
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
                    
            elif cmd == 'strategies':
                available_strategies = self.strategy_manager.get_available_strategies()
                active_strategy = self.strategy_manager.active_strategy
                
                print("\nAvailable Trading Strategies:")
                for strategy in available_strategies:
                    if strategy == active_strategy:
                        print(f"  * {strategy} (ACTIVE)")
                    else:
                        print(f"  - {strategy}")
                        
                print("\nUse 'strategy <name>' to set the active strategy")
                
            elif cmd.startswith('strategy '):
                parts = cmd.split()
                if len(parts) < 2:
                    print("Usage: strategy <name> [params]")
                    continue
                    
                strategy_name = parts[1]
                
                # Check if there are custom parameters (simple key=value format)
                custom_params = {}
                if len(parts) > 2:
                    for param in parts[2:]:
                        if '=' in param:
                            key, value = param.split('=', 1)
                            try:
                                # Try to convert to appropriate type
                                if value.isdigit():
                                    value = int(value)
                                elif '.' in value and all(c.isdigit() for c in value.replace('.', '', 1)):
                                    value = float(value)
                                custom_params[key] = value
                            except ValueError:
                                custom_params[key] = value
                
                if self.strategy_manager.set_active_strategy(strategy_name, custom_params):
                    print(f"Strategy set to '{strategy_name}'")
                    if custom_params:
                        print("With custom parameters:", custom_params)
                else:
                    print(f"Failed to set strategy '{strategy_name}'")
                    
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
    
    # Select trading strategy
    print("\nAvailable strategies:")
    for i, strategy in enumerate(trader.strategy_manager.get_available_strategies()):
        print(f"  {i+1}. {strategy}")
    print("  0. Load external strategy")
    
    strategy_choice = input("\nSelect strategy (default: ma_crossover): ").strip()
    
    if strategy_choice == '0':
        # Load external strategy
        module_path = input("Enter strategy module path: ")
        strategy_name = input("Enter strategy function name (optional): ") or None
        loaded_strategy = trader.strategy_manager.load_external_strategy(module_path, strategy_name)
        if loaded_strategy:
            trader.strategy_manager.set_active_strategy(loaded_strategy)
    elif strategy_choice.isdigit() and int(strategy_choice) > 0:
        strategies = trader.strategy_manager.get_available_strategies()
        idx = int(strategy_choice) - 1
        if idx < len(strategies):
            trader.strategy_manager.set_active_strategy(strategies[idx])
    else:
        # Default to MA crossover
        trader.strategy_manager.set_active_strategy("ma_crossover")
    
    # Select mode
    mode = input("Select mode (manual/auto, default: manual): ").lower()
    
    if mode == 'auto':
        # Set update interval
        interval_str = input("Update interval in minutes (default: 15): ")
        interval = int(interval_str) if interval_str.isdigit() else 15
        
        # Run in scheduled mode
        trader.run_scheduler(interval_minutes=interval)
    else:
        # Run in manual mode
        trader.run_manual()