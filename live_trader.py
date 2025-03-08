# live_trader.py
import time
import json
import pandas as pd
import numpy as np
import os
import schedule
import threading
from datetime import datetime
import logging

from multi_api_price_fetcher import CryptoPriceFetcher
from crypto_analyzer import CryptoAnalyzer
from advanced_strategy import AdvancedStrategy
from hyperliquid_api import HyperliquidAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HyperliquidTrader")

class HyperliquidTrader:
    """
    Live trading bot for Hyperliquid exchange
    """
    
    def __init__(self, config_file="config.json"):
        """Initialize with configuration from file"""
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.price_fetcher = CryptoPriceFetcher()
        self.analyzer = CryptoAnalyzer(self.price_fetcher)
        self.strategy = AdvancedStrategy(risk_level=self.config.get('risk_level', 'medium'))
        
        # Initialize API connection
        api_key = self.config.get('api_key', None)
        api_secret = self.config.get('api_secret', None)
        testnet = self.config.get('use_testnet', True)
        
        self.api = HyperliquidAPI(api_key=api_key, api_secret=api_secret, testnet=testnet)
        
        # Trading settings
        self.symbols = self.config.get('symbols', ["BTC", "ETH", "SOL"])
        self.update_interval = self.config.get('update_interval_minutes', 5)
        self.active = False
        self.max_positions = self.config.get('max_positions', 3)
        
        # State variables
        self.positions = {}  # Current positions
        self.pending_orders = {}  # Orders that are not yet filled
        
        # Create data directory if it doesn't exist
        self.data_dir = "live_trading_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info(f"Initialized Hyperliquid Trader with {len(self.symbols)} markets")
    
    def _load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading config: {e}")
            # Return default config
            return {
                "risk_level": "low",
                "symbols": ["BTC", "ETH", "SOL"],
                "update_interval_minutes": 5,
                "max_positions": 3,
                "use_testnet": True
            }
    
    def _save_state(self):
        """Save current trading state to file"""
        state = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'positions': self.positions,
            'pending_orders': self.pending_orders
        }
        
        filename = f"{self.data_dir}/trading_state.json"
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Saved trading state to {filename}")
    
    def _load_state(self):
        """Load trading state from file"""
        filename = f"{self.data_dir}/trading_state.json"
        if not os.path.exists(filename):
            logger.info("No saved state found")
            return False
        
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            self.positions = state.get('positions', {})
            self.pending_orders = state.get('pending_orders', {})
            
            logger.info(f"Loaded trading state from {filename}")
            logger.info(f"Current positions: {len(self.positions)}")
            return True
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False
    
    def update_market_data(self, symbol):
        """Fetch latest market data for a symbol"""
        # Get ticker data from Hyperliquid
        ticker_data = self.api.get_ticker(symbol)
        
        if not ticker_data:
            logger.warning(f"Failed to get ticker data for {symbol}")
            return None
        
        # Also get price data from our existing price fetcher as backup
        price = self.price_fetcher.get_price(symbol)
        
        if not price and 'lastPrice' in ticker_data:
            price = float(ticker_data['lastPrice'])
        
        if price:
            logger.info(f"Updated {symbol} price: ${price:.2f}")
            return price
        else:
            logger.warning(f"Could not get price for {symbol}")
            return None
    
    def check_open_positions(self):
        """Check status of open positions and manage them"""
        # Get account data from API
        account_data = self.api.get_account_balance()
        
        if not account_data:
            logger.warning("Failed to get account data")
            return
        
        # Update positions from API
        api_positions = {}
        if 'positions' in account_data:
            for position in account_data['positions']:
                symbol = position.get('symbol')
                if symbol:
                    api_positions[symbol] = position
        
        # Reconcile with our tracked positions
        for symbol, position in list(self.positions.items()):
            if symbol in api_positions:
                # Update our position data
                api_position = api_positions[symbol]
                
                # Check if stop loss or take profit hit
                current_price = position.get('current_price', 0)
                entry_price = position.get('entry_price', 0)
                
                if current_price and entry_price:
                    if position['side'] == 'long':
                        # Check stop loss
                        if current_price <= position['stop_loss']:
                            logger.info(f"Stop loss triggered for {symbol} long at ${current_price:.2f}")
                            self.close_position(symbol, 'stop_loss')
                        
                        # Check take profit
                        elif current_price >= position['take_profit']:
                            logger.info(f"Take profit triggered for {symbol} long at ${current_price:.2f}")
                            self.close_position(symbol, 'take_profit')
                    
                    else:  # short position
                        # Check stop loss
                        if current_price >= position['stop_loss']:
                            logger.info(f"Stop loss triggered for {symbol} short at ${current_price:.2f}")
                            self.close_position(symbol, 'stop_loss')
                        
                        # Check take profit
                        elif current_price <= position['take_profit']:
                            logger.info(f"Take profit triggered for {symbol} short at ${current_price:.2f}")
                            self.close_position(symbol, 'take_profit')
            else:
                # Position no longer exists in API
                logger.info(f"Position for {symbol} no longer exists in account")
                if symbol in self.positions:
                    del self.positions[symbol]
    
    def process_signals(self, signals):
        """Process trading signals to open/close positions"""
        for symbol, data in signals.items():
            current_price = self.update_market_data(symbol)
            if not current_price:
                continue
            
            # Check for long signal
            if data.get('long_signal', 0) == 1:
                if symbol not in self.positions and len(self.positions) < self.max_positions:
                    logger.info(f"Long signal for {symbol}, opening position")
                    self.open_position(symbol, 'long', current_price)
            
            # Check for short signal
            elif data.get('short_signal', 0) == 1:
                if symbol not in self.positions and len(self.positions) < self.max_positions:
                    logger.info(f"Short signal for {symbol}, opening position")
                    self.open_position(symbol, 'short', current_price)
            
            # Check for exit signals
            if symbol in self.positions:
                position = self.positions[symbol]
                
                # Exit long position on short signal
                if position['side'] == 'long' and data.get('short_signal', 0) == 1:
                    logger.info(f"Exit signal for {symbol} long position")
                    self.close_position(symbol, 'signal')
                
                # Exit short position on long signal
                elif position['side'] == 'short' and data.get('long_signal', 0) == 1:
                    logger.info(f"Exit signal for {symbol} short position")
                    self.close_position(symbol, 'signal')
    
    def generate_trading_signals(self):
        """Generate trading signals for all symbols"""
        signals = {}
        
        for symbol in self.symbols:
            # Get historical data
            df = self.analyzer.convert_history_to_dataframe(symbol)
            
            if df is not None and len(df) > 30:  # Need enough data
                # Apply strategy
                signal_df = self.strategy.generate_signals(df)
                
                if signal_df is not None:
                    # Extract latest signals
                    latest = signal_df.iloc[-1]
                    signals[symbol] = {
                        'long_signal': latest.get('long_signal', 0),
                        'short_signal': latest.get('short_signal', 0),
                        'signal_strength': latest.get('signal_strength', 0),
                        'current_price': latest.get('close', latest.get('price', 0))
                    }
                    
                    logger.info(f"Signals for {symbol}: Long={latest.get('long_signal', 0)}, Short={latest.get('short_signal', 0)}")
            else:
                logger.warning(f"Not enough data for {symbol} to generate signals")
        
        return signals
    
    # in live_trader.py
def open_position(self, symbol, side, price):
    """
    Open a new position
    
    Args:
        symbol: Market symbol
        side: 'long' or 'short'
        price: Current price
    """
    if not self.api.api_key:
        logger.warning("API key required to open positions")
        return False
    
    # Get account balance
    account_data = self.api.get_account_balance()
    if not account_data or 'balance' not in account_data:
        logger.warning("Could not get account balance")
        return False
    
    available_balance = float(account_data['balance'].get('available', 0))
    
    # Calculate position size
    quantity, max_loss = self.strategy.calculate_position_size(available_balance, price)
    
    if quantity <= 0:
        logger.warning(f"Invalid position size calculated: {quantity}")
        return False
    
    # Calculate stop loss and take profit
    stop_loss = self.strategy.determine_stop_loss(price, side)
    take_profit = self.strategy.determine_take_profit(price, side)
    
    # Determine order side for API (buy/sell)
    api_side = "buy" if side == "long" else "sell"
    
    # Calculate leverage based on risk profile
    leverage = 1  # Default to 1x leverage
    if self.config.get('use_leverage', False):
        leverage = self.config.get('leverage', 1)
    
    # Place the order
    order_result = self.api.place_order(
        symbol=symbol,
        side=api_side,
        quantity=quantity,
        order_type="market",
        leverage=leverage
    )
    
    if not order_result or 'orderId' not in order_result:
        logger.error(f"Failed to place {side} order for {symbol}")
        return False
    
    # Record the position
    self.positions[symbol] = {
        'symbol': symbol,
        'side': side,
        'entry_price': price,
        'quantity': quantity,
        'entry_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'order_id': order_result['orderId'],
        'current_price': price,
        'leverage': leverage
    }
    
    logger.info(f"Opened {side} position for {symbol}: {quantity:.6f} @ ${price:.2f}")
    
    # Save state
    self._save_state()
    return True

def close_position(self, symbol, reason="manual"):
    """
    Close an existing position
    
    Args:
        symbol: Market symbol
        reason: Reason for closing (manual, stop_loss, take_profit, signal)
    """
    if symbol not in self.positions:
        logger.warning(f"No position found for {symbol}")
        return False
    
    position = self.positions[symbol]
    
    # Determine order side (opposite of position side)
    api_side = "sell" if position['side'] == "long" else "buy"
    
    # Get current price
    current_price = self.update_market_data(symbol)
    if not current_price:
        logger.warning(f"Could not get current price for {symbol}")
        return False
    
    # Place the order
    order_result = self.api.place_order(
        symbol=symbol,
        side=api_side,
        quantity=position['quantity'],
        order_type="market"
    )
    
    if not order_result or 'orderId' not in order_result:
        logger.error(f"Failed to close position for {symbol}")
        return False
    
    # Calculate profit/loss
    if position['side'] == "long":
        pnl = (current_price - position['entry_price']) * position['quantity']
        pnl_pct = (current_price / position['entry_price'] - 1) * 100
    else:  # short
        pnl = (position['entry_price'] - current_price) * position['quantity']
        pnl_pct = (position['entry_price'] / current_price - 1) * 100
    
    logger.info(f"Closed {position['side']} position for {symbol}: {position['quantity']:.6f} @ ${current_price:.2f}")
    logger.info(f"P/L: ${pnl:.2f} ({pnl_pct:.2f}%) - Reason: {reason}")
    
    # Remove from positions
    del self.positions[symbol]
    
    # Save state
    self._save_state()
    return True

def run_scheduled_update(self):
    """Run a scheduled update to check prices and signals"""
    logger.info(f"=== Scheduled Update {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    try:
        # Update market data
        logger.info("Updating market data...")
        for symbol in self.symbols:
            self.update_market_data(symbol)
        
        # Check open positions
        logger.info("Checking open positions...")
        self.check_open_positions()
        
        # Generate and process signals
        logger.info("Generating trading signals...")
        signals = self.generate_trading_signals()
        self.process_signals(signals)
        
        # Save state
        self._save_state()
        
        logger.info("Update complete")
    except Exception as e:
        logger.error(f"Error in scheduled update: {e}")

def start(self):
    """Start the trading bot"""
    if self.active:
        logger.warning("Trading bot is already running")
        return
    
    # Load state
    self._load_state()
    
    # Schedule regular updates
    schedule.every(self.update_interval).minutes.do(self.run_scheduled_update)
    
    # Run initial update
    self.run_scheduled_update()
    
    # Mark as active
    self.active = True
    
    # Start the scheduler in a separate thread
    self.scheduler_thread = threading.Thread(target=self._run_scheduler)
    self.scheduler_thread.daemon = True
    self.scheduler_thread.start()
    
    logger.info(f"Trading bot started with {self.update_interval} minute intervals")
    logger.info(f"Trading {len(self.symbols)} symbols: {', '.join(self.symbols)}")
    logger.info(f"Max positions: {self.max_positions}")
    logger.info(f"Risk level: {self.config.get('risk_level', 'medium')}")

def stop(self):
    """Stop the trading bot"""
    if not self.active:
        logger.warning("Trading bot is not running")
        return
    
    # Mark as inactive
    self.active = False
    
    # Save state
    self._save_state()
    
    logger.info("Trading bot stopped")

def _run_scheduler(self):
    """Run the scheduler loop"""
    while self.active:
        schedule.run_pending()
        time.sleep(1)