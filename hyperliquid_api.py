import hmac
import json
import time
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

class HyperliquidAPI:
    """
    Python client for the Hyperliquid API.
    
    This class provides methods to interact with the Hyperliquid API for cryptocurrency trading,
    including account management, market data retrieval, and order execution.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, testnet: bool = True):
        """
        Initialize the Hyperliquid API client.
        
        Args:
            api_key: Your Hyperliquid API key
            api_secret: Your Hyperliquid API secret
            testnet: Whether to use the testnet (True) or mainnet (False)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Set base URLs based on testnet flag
        if testnet:
            self.base_url = "https://api-testnet.hyperliquid.xyz"
        else:
            self.base_url = "https://api.hyperliquid.xyz"
            
        self.session = requests.Session()
        
        # Setup logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        logger.info(f"Initialized Hyperliquid API client (testnet: {testnet})")

    def _generate_signature(self, payload: Dict) -> Dict[str, Any]:
        """
        Generate HMAC signature for authenticated API requests.
        
        Args:
            payload: The request payload to sign
            
        Returns:
            Dict containing timestamp and signature
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret required for authenticated requests")
            
        timestamp = int(time.time() * 1000)
        message = f"{timestamp}{json.dumps(payload)}"
        
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            digestmod='sha256'
        ).hexdigest()
        
        return {
            "timestamp": timestamp,
            "signature": signature
        }

    def get_markets(self) -> List[Dict]:
        """
        Get information about all available markets.
        
        Returns:
            List of market information dictionaries
        """
        url = f"{self.base_url}/info/markets"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []

    def get_ticker(self, symbol: str) -> Dict:
        """
        Get current ticker information for a specific symbol.
        
        Args:
            symbol: The trading symbol (e.g., "BTC")
            
        Returns:
            Dictionary containing ticker information
        """
        url = f"{self.base_url}/info/ticker"
        params = {"symbol": symbol.upper()}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return {}

    def get_orderbook(self, symbol: str, depth: int = 10) -> Dict:
        """
        Get the current orderbook for a specific symbol.
        
        Args:
            symbol: The trading symbol (e.g., "BTC")
            depth: Depth of the orderbook to return
            
        Returns:
            Dictionary containing orderbook data
        """
        url = f"{self.base_url}/info/orderbook"
        params = {"symbol": symbol.upper(), "depth": depth}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return {}

    def get_account_balance(self) -> Dict:
        """
        Get the account balance and positions.
        
        Returns:
            Dictionary containing account information
        """
        if not self.api_key:
            logger.error("API key required for account balance")
            return {}
            
        url = f"{self.base_url}/account/balance"
        payload = {"timestamp": int(time.time() * 1000)}
        
        signature_headers = self._generate_signature(payload)
        headers = {
            "X-HL-ApiKey": self.api_key,
            "X-HL-Signature": signature_headers["signature"],
            "X-HL-Timestamp": str(signature_headers["timestamp"])
        }
        
        try:
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            return {}

    def place_order(self, symbol: str, side: str, quantity: float, 
                  order_type: str = "market", price: Optional[float] = None, 
                  leverage: int = 1, time_in_force: str = "GTC",
                  reduce_only: bool = False) -> Dict:
        """
        Place a new order.
        
        Args:
            symbol: Trading symbol (e.g., "BTC")
            side: Order side ("buy" or "sell")
            quantity: Order quantity
            order_type: Order type ("market" or "limit")
            price: Order price (required for limit orders)
            leverage: Leverage to use (1 = no leverage)
            time_in_force: Time in force ("GTC", "IOC", or "FOK")
            reduce_only: Whether this order is reduce-only
            
        Returns:
            Dictionary containing order information
        """
        if not self.api_key:
            logger.error("API key required for placing orders")
            return {"error": "API key required"}
            
        side = side.lower()
        order_type = order_type.lower()
        
        if side not in ["buy", "sell"]:
            logger.error(f"Invalid side: {side}. Must be 'buy' or 'sell'")
            return {"error": "Invalid side"}
            
        if order_type not in ["market", "limit"]:
            logger.error(f"Invalid order type: {order_type}. Must be 'market' or 'limit'")
            return {"error": "Invalid order type"}
            
        if order_type == "limit" and price is None:
            logger.error("Price required for limit orders")
            return {"error": "Price required for limit orders"}
            
        url = f"{self.base_url}/trade/order"
        
        payload = {
            "symbol": symbol.upper(),
            "side": side,
            "type": order_type,
            "quantity": float(quantity),
            "leverage": int(leverage),
            "timeInForce": time_in_force,
            "reduceOnly": reduce_only
        }
        
        if order_type == "limit":
            payload["price"] = float(price)
            
        signature_headers = self._generate_signature(payload)
        headers = {
            "X-HL-ApiKey": self.api_key,
            "X-HL-Signature": signature_headers["signature"],
            "X-HL-Timestamp": str(signature_headers["timestamp"]),
            "Content-Type": "application/json"
        }
        
        try:
            response = self.session.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Successfully placed {side} {order_type} order for {quantity} {symbol}")
            return result
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"error": str(e)}

    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel an existing order.
        
        Args:
            order_id: The ID of the order to cancel
            
        Returns:
            Dictionary containing the cancel result
        """
        if not self.api_key:
            logger.error("API key required for cancelling orders")
            return {"error": "API key required"}
            
        url = f"{self.base_url}/trade/cancel"
        payload = {"orderId": order_id}
        
        signature_headers = self._generate_signature(payload)
        headers = {
            "X-HL-ApiKey": self.api_key,
            "X-HL-Signature": signature_headers["signature"],
            "X-HL-Timestamp": str(signature_headers["timestamp"]),
            "Content-Type": "application/json"
        }
        
        try:
            response = self.session.delete(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Successfully cancelled order {order_id}")
            return result
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return {"error": str(e)}

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get all open orders, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol to filter orders
            
        Returns:
            List of open order dictionaries
        """
        if not self.api_key:
            logger.error("API key required for fetching open orders")
            return []
            
        url = f"{self.base_url}/account/openOrders"
        payload = {}
        
        if symbol:
            payload["symbol"] = symbol.upper()
            
        signature_headers = self._generate_signature(payload)
        headers = {
            "X-HL-ApiKey": self.api_key,
            "X-HL-Signature": signature_headers["signature"],
            "X-HL-Timestamp": str(signature_headers["timestamp"])
        }
        
        try:
            response = self.session.get(url, headers=headers, params=payload, timeout=10)
            response.raise_for_status()
            orders = response.json()
            logger.info(f"Retrieved {len(orders)} open orders{' for ' + symbol if symbol else ''}")
            return orders
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return []

    def get_order_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """
        Get order history, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol to filter orders
            limit: Maximum number of orders to return
            
        Returns:
            List of order history dictionaries
        """
        if not self.api_key:
            logger.error("API key required for fetching order history")
            return []
            
        url = f"{self.base_url}/account/orderHistory"
        payload = {"limit": limit}
        
        if symbol:
            payload["symbol"] = symbol.upper()
            
        signature_headers = self._generate_signature(payload)
        headers = {
            "X-HL-ApiKey": self.api_key,
            "X-HL-Signature": signature_headers["signature"],
            "X-HL-Timestamp": str(signature_headers["timestamp"])
        }
        
        try:
            response = self.session.get(url, headers=headers, params=payload, timeout=10)
            response.raise_for_status()
            orders = response.json()
            logger.info(f"Retrieved {len(orders)} historical orders{' for ' + symbol if symbol else ''}")
            return orders
        except Exception as e:
            logger.error(f"Error fetching order history: {e}")
            return []

    def get_trade_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """
        Get trade history, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol to filter trades
            limit: Maximum number of trades to return
            
        Returns:
            List of trade history dictionaries
        """
        if not self.api_key:
            logger.error("API key required for fetching trade history")
            return []
            
        url = f"{self.base_url}/account/tradeHistory"
        payload = {"limit": limit}
        
        if symbol:
            payload["symbol"] = symbol.upper()
            
        signature_headers = self._generate_signature(payload)
        headers = {
            "X-HL-ApiKey": self.api_key,
            "X-HL-Signature": signature_headers["signature"],
            "X-HL-Timestamp": str(signature_headers["timestamp"])
        }
        
        try:
            response = self.session.get(url, headers=headers, params=payload, timeout=10)
            response.raise_for_status()
            trades = response.json()
            logger.info(f"Retrieved {len(trades)} historical trades{' for ' + symbol if symbol else ''}")
            return trades
        except Exception as e:
            logger.error(f"Error fetching trade history: {e}")
            return []

    def get_market_ticker_price(self, symbol: str) -> Optional[float]:
        """
        Get the current market price of a symbol from ticker data.
        
        Args:
            symbol: The trading symbol
            
        Returns:
            Current price as a float, or None if not available
        """
        ticker_data = self.get_ticker(symbol)
        
        if not ticker_data or 'lastPrice' not in ticker_data:
            logger.warning(f"No ticker data available for {symbol}")
            return None
            
        try:
            return float(ticker_data['lastPrice'])
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing price for {symbol}: {e}")
            return None

    def get_historical_klines(self, symbol: str, interval: str = '1h', 
                            start_time: Optional[int] = None, 
                            end_time: Optional[int] = None, 
                            limit: int = 500) -> List[Dict]:
        """
        Get historical candlestick data.
        
        Args:
            symbol: The trading symbol
            interval: Candlestick interval ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
            start_time: Optional start time in milliseconds
            end_time: Optional end time in milliseconds
            limit: Maximum number of candles to return
            
        Returns:
            List of candle dictionaries
        """
        url = f"{self.base_url}/info/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = start_time
            
        if end_time:
            params["endTime"] = end_time
            
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            klines = response.json()
            logger.info(f"Retrieved {len(klines)} {interval} candles for {symbol}")
            return klines
        except Exception as e:
            logger.error(f"Error fetching historical klines for {symbol}: {e}")
            return []

    def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """
        Set leverage for a specific symbol.
        
        Args:
            symbol: The trading symbol
            leverage: The leverage value to set
            
        Returns:
            Dictionary containing the result
        """
        if not self.api_key:
            logger.error("API key required for setting leverage")
            return {"error": "API key required"}
            
        url = f"{self.base_url}/trade/leverage"
        payload = {
            "symbol": symbol.upper(),
            "leverage": int(leverage)
        }
        
        signature_headers = self._generate_signature(payload)
        headers = {
            "X-HL-ApiKey": self.api_key,
            "X-HL-Signature": signature_headers["signature"],
            "X-HL-Timestamp": str(signature_headers["timestamp"]),
            "Content-Type": "application/json"
        }
        
        try:
            response = self.session.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Successfully set leverage for {symbol} to {leverage}x")
            return result
        except Exception as e:
            logger.error(f"Error setting leverage for {symbol}: {e}")
            return {"error": str(e)}

    def get_funding_rate(self, symbol: str) -> Dict:
        """
        Get the current funding rate for a symbol.
        
        Args:
            symbol: The trading symbol
            
        Returns:
            Dictionary containing funding rate information
        """
        url = f"{self.base_url}/info/funding"
        params = {"symbol": symbol.upper()}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching funding rate for {symbol}: {e}")
            return {}

    def check_connection(self) -> bool:
        """
        Check if the API connection is working.
        
        Returns:
            True if connection is working, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/info/ping", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False

    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get current position for a specific symbol.
        
        Args:
            symbol: The trading symbol
            
        Returns:
            Position dictionary or None if not found
        """
        account_data = self.get_account_balance()
        
        if not account_data or "positions" not in account_data:
            logger.warning("Could not get account data")
            return None
            
        for position in account_data["positions"]:
            if position.get("symbol") == symbol.upper():
                return position
                
        return None

    def close_position(self, symbol: str, reduce_only: bool = True) -> Dict:
        """
        Close a position for a symbol.
        
        Args:
            symbol: The trading symbol
            reduce_only: Whether to use reduce_only flag
            
        Returns:
            Dictionary containing order result
        """
        position = self.get_position(symbol)
        
        if not position:
            logger.warning(f"No position found for {symbol}")
            return {"error": "No position found"}
            
        position_size = abs(float(position.get("size", 0)))
        position_side = "long" if float(position.get("size", 0)) > 0 else "short"
        
        if position_size <= 0:
            logger.warning(f"Position size for {symbol} is zero or negative")
            return {"error": "Invalid position size"}
            
        close_side = "sell" if position_side == "long" else "buy"
        
        return self.place_order(
            symbol=symbol,
            side=close_side,
            quantity=position_size,
            order_type="market",
            reduce_only=reduce_only
        )