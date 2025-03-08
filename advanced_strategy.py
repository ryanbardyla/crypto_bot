# advanced_strategy.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

# Import ta indicators
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange

class AdvancedStrategy:
    """Advanced trading strategy with multiple indicators and entry/exit signals"""
    
    def __init__(self, risk_level='medium'):
        """
        Initialize with risk level
        risk_level options: 'low', 'medium', 'high'
        """
        self.risk_level = risk_level
        self.risk_multipliers = {
            'low': 0.5,
            'medium': 1.0,
            'high': 2.0
        }
        
        # Strategy parameters (can be adjusted based on risk level)
        self.set_parameters()
    
    def set_parameters(self):
        """Set strategy parameters based on risk level"""
        multiplier = self.risk_multipliers.get(self.risk_level, 1.0)
        
        # Position sizing and risk management
        self.position_size_pct = 0.1 * multiplier  # Percentage of capital per trade
        self.max_positions = max(1, int(3 * multiplier))  # Maximum number of simultaneous positions
        self.stop_loss_pct = 5.0 / multiplier      # Stop loss percentage
        self.take_profit_pct = 10.0 * multiplier   # Take profit percentage
        
        # Technical indicators
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.fast_ma = 10
        self.slow_ma = 30
        self.bollinger_period = 20
        self.bollinger_std = 2.0
    
    def generate_signals(self, df, additional_data=None):
        """
        Generate trading signals from price data
        
        Args:
            df: DataFrame with OHLCV data
            additional_data: Optional dict with alternative data signals
            
        Returns:
            DataFrame with signals added
        """
        if df is None or len(df) < self.slow_ma:
            return None
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Ensure we have proper columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # If we only have 'price', create other columns
        if 'price' in result.columns and 'close' not in result.columns:
            result['close'] = result['price']
            # Create synthetic OHLC data if missing
            if 'open' not in result.columns:
                result['open'] = result['price'].shift(1)
                result.loc[result.index[0], 'open'] = result['price'].iloc[0]
            if 'high' not in result.columns:
                result['high'] = result['price']
            if 'low' not in result.columns:
                result['low'] = result['price']
            if 'volume' not in result.columns:
                result['volume'] = 0  # Default volume if missing
        
        # Calculate technical indicators
        try:
            # RSI
            rsi_indicator = RSIIndicator(close=result['close'], window=self.rsi_period)
            result['rsi'] = rsi_indicator.rsi()
            
            # Moving Averages
            fast_ma_indicator = SMAIndicator(close=result['close'], window=self.fast_ma)
            slow_ma_indicator = SMAIndicator(close=result['close'], window=self.slow_ma)
            result['fast_ma'] = fast_ma_indicator.sma_indicator()
            result['slow_ma'] = slow_ma_indicator.sma_indicator()
            
            # Bollinger Bands
            bollinger = BollingerBands(close=result['close'], window=self.bollinger_period, window_dev=self.bollinger_std)
            result['bb_upper'] = bollinger.bollinger_hband()
            result['bb_middle'] = bollinger.bollinger_mavg()
            result['bb_lower'] = bollinger.bollinger_lband()
            
            # MACD
            macd_indicator = MACD(close=result['close'])
            result['macd'] = macd_indicator.macd()
            result['macd_signal'] = macd_indicator.macd_signal()
            result['macd_hist'] = macd_indicator.macd_diff()
            
            # Calculate volatility (ATR)
            atr_indicator = AverageTrueRange(high=result['high'], low=result['low'], close=result['close'])
            result['atr'] = atr_indicator.average_true_range()
            
            # Stochastic Oscillator
            stoch = StochasticOscillator(high=result['high'], low=result['low'], close=result['close'])
            result['stoch_k'] = stoch.stoch()
            result['stoch_d'] = stoch.stoch_signal()
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return None
        
        # Initialize signal columns
        result['long_signal'] = 0
        result['short_signal'] = 0
        result['signal_strength'] = 0  # -10 to +10 scale, positive for long, negative for short
        
        # Generate signals
        for i in range(1, len(result)):
            # Skip if we have NaN values
            if pd.isna(result['rsi'].iloc[i]) or pd.isna(result['macd'].iloc[i]):
                continue
            
            # Initialize signal strength for this bar
            signal_points = 0
            
            # === Long Signals ===
            
            # RSI oversold and turning up
            if (result['rsi'].iloc[i-1] < self.rsi_oversold and 
                result['rsi'].iloc[i] > result['rsi'].iloc[i-1]):
                signal_points += 2
            
            # Price crosses above slow MA
            if (result['close'].iloc[i] > result['slow_ma'].iloc[i] and 
                result['close'].iloc[i-1] <= result['slow_ma'].iloc[i-1]):
                signal_points += 3
            
            # Fast MA crosses above slow MA
            if (result['fast_ma'].iloc[i] > result['slow_ma'].iloc[i] and 
                result['fast_ma'].iloc[i-1] <= result['slow_ma'].iloc[i-1]):
                signal_points += 3
            
            # Price bounces off lower Bollinger Band
            if (result['close'].iloc[i-1] < result['bb_lower'].iloc[i-1] and 
                result['close'].iloc[i] > result['bb_lower'].iloc[i]):
                signal_points += 2
            
            # MACD histogram turns positive
            if (result['macd_hist'].iloc[i] > 0 and 
                result['macd_hist'].iloc[i-1] <= 0):
                signal_points += 2
            
            # Stochastic oversold and turning up
            if (result['stoch_k'].iloc[i-1] < 20 and 
                result['stoch_k'].iloc[i] > result['stoch_k'].iloc[i-1]):
                signal_points += 2
            
            # === Short Signals ===
            
            # RSI overbought and turning down
            if (result['rsi'].iloc[i-1] > self.rsi_overbought and 
                result['rsi'].iloc[i] < result['rsi'].iloc[i-1]):
                signal_points -= 2
            
            # Price crosses below slow MA
            if (result['close'].iloc[i] < result['slow_ma'].iloc[i] and 
                result['close'].iloc[i-1] >= result['slow_ma'].iloc[i-1]):
                signal_points -= 3
            
            # Fast MA crosses below slow MA
            if (result['fast_ma'].iloc[i] < result['slow_ma'].iloc[i] and 
                result['fast_ma'].iloc[i-1] >= result['slow_ma'].iloc[i-1]):
                signal_points -= 3
            
            # Price bounces off upper Bollinger Band
            if (result['close'].iloc[i-1] > result['bb_upper'].iloc[i-1] and 
                result['close'].iloc[i] < result['bb_upper'].iloc[i]):
                signal_points -= 2
            
            # MACD histogram turns negative
            if (result['macd_hist'].iloc[i] < 0 and 
                result['macd_hist'].iloc[i-1] >= 0):
                signal_points -= 2
            
            # Stochastic overbought and turning down
            if (result['stoch_k'].iloc[i-1] > 80 and 
                result['stoch_k'].iloc[i] < result['stoch_k'].iloc[i-1]):
                signal_points -= 2
            
            # === Incorporate additional data if provided ===
            if additional_data and isinstance(additional_data, dict):
                # Example: sentiment score from -10 to 10
                if 'sentiment_score' in additional_data:
                    signal_points += additional_data['sentiment_score']
            else:
                # If no additional data was provided, try to load sentiment from files
                sentiment_score = self.get_combined_sentiment_score()
                if sentiment_score != 0:
                    signal_points += sentiment_score
                    print(f"Adding sentiment score to signals: {sentiment_score:.2f}")
            
            # Record the signal strength
            result.loc[result.index[i], 'signal_strength'] = signal_points
            
            # Generate actual signals based on strength
            # Thresholds can be adjusted based on risk level
            if signal_points >= 6:  # Strong long signal
                result.loc[result.index[i], 'long_signal'] = 1
            elif signal_points <= -6:  # Strong short signal
                result.loc[result.index[i], 'short_signal'] = 1
        
        return result
    
    def calculate_position_size(self, capital, price, risk_per_trade=None):
        """
        Calculate appropriate position size based on capital and risk
        
        Args:
            capital: Available capital
            price: Current price
            risk_per_trade: Optional override for risk percentage
            
        Returns:
            tuple: (quantity, max_loss_amount)
        """
        if risk_per_trade is None:
            risk_per_trade = self.position_size_pct
        
        position_value = capital * risk_per_trade
        quantity = position_value / price
        
        # Calculate maximum loss amount
        max_loss = position_value * (self.stop_loss_pct / 100)
        
        return quantity, max_loss
    
    def determine_stop_loss(self, entry_price, side, atr_value=None):
        """
        Calculate stop loss price
        
        Args:
            entry_price: Entry price
            side: 'long' or 'short'
            atr_value: Optional ATR value for dynamic stop loss
            
        Returns:
            float: Stop loss price
        """
        if side == 'long':
            if atr_value:
                # Dynamic stop loss based on ATR
                stop_loss = entry_price - (atr_value * 2)
            else:
                # Fixed percentage stop loss
                stop_loss = entry_price * (1 - self.stop_loss_pct / 100)
        else:  # short
            if atr_value:
                stop_loss = entry_price + (atr_value * 2)
            else:
                stop_loss = entry_price * (1 + self.stop_loss_pct / 100)
        
        return stop_loss
    
    def determine_take_profit(self, entry_price, side):
        """
        Calculate take profit price
        
        Args:
            entry_price: Entry price
            side: 'long' or 'short'
            
        Returns:
            float: Take profit price
        """
        if side == 'long':
            take_profit = entry_price * (1 + self.take_profit_pct / 100)
        else:  # short
            take_profit = entry_price * (1 - self.take_profit_pct / 100)
        
        return take_profit
    
    def load_sentiment_data(self):
        """Load all available sentiment data"""
        sentiment_dir = "sentiment_data"
        sentiment_data = []
        
        if not os.path.exists(sentiment_dir):
            return sentiment_data
        
        for filename in os.listdir(sentiment_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(sentiment_dir, filename), 'r') as f:
                        data = json.load(f)
                        sentiment_data.append(data)
                except Exception as e:
                    print(f"Error loading sentiment data: {e}")
        
        return sentiment_data

    def get_combined_sentiment_score(self):
        """Calculate a combined sentiment score from all available data"""
        sentiment_data = self.load_sentiment_data()
        
        if not sentiment_data:
            return 0
        
        # Calculate weighted average of sentiment scores
        total_score = 0
        total_weight = 0
        
        for data in sentiment_data:
            # Newer sentiment has more weight (could be based on timestamp)
            weight = 1
            total_score += data['combined_score'] * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0