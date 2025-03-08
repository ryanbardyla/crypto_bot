# hyperliquid_api.py
import json
import hmac
import hashlib
import time
import requests
from datetime import datetime

class HyperliquidAPI:
    """Interface with Hyperliquid exchange API"""
    
    def __init__(self, api_key=None, api_secret=None, testnet=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Set base URL based on testnet or mainnet
        if testnet:
            self.base_url = "https://api-testnet.hyperliquid.xyz"
        else:
            self.base_url = "https://api.hyperliquid.xyz"
            
        self.session = requests.Session()
    
    def _generate_signature(self, payload):
        """Generate API signature for authenticated requests"""
        if not self.api_secret:
            return None
            
        timestamp = int(time.time() * 1000)
        message = f"{timestamp}{json.dumps(payload)}"
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "timestamp": timestamp,
            "signature": signature
        }
    
    def get_markets(self):
        """Get list of available markets"""
        endpoint = "/info/markets"
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching markets: {e}")
            return None
    
    def get_ticker(self, symbol):
        """Get current ticker information for a symbol"""
        endpoint = "/info/ticker"
        url = f"{self.base_url}{endpoint}"
        params = {"symbol": symbol}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching ticker for {symbol}: {e}")
            return None
    
    def get_account_balance(self):
        """Get account balance and positions"""
        if not self.api_key:
            print("API key required for account balance")
            return None
            
        endpoint = "/api/v1/account"
        url = f"{self.base_url}{endpoint}"
        
        payload = {}
        signature_headers = self._generate_signature(payload)
        
        headers = {
            "X-HL-API-Key": self.api_key,
            "X-HL-Signature": signature_headers["signature"],
            "X-HL-Timestamp": str(signature_headers["timestamp"])
        }
        
        try:
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching account balance: {e}")
            return None
    
    def place_order(self, symbol, side, quantity, order_type="market", price=None, leverage=1):
        """
        Place an order on Hyperliquid
        
        Args:
            symbol (str): Market symbol (e.g., 'BTC-USDT')
            side (str): 'buy' or 'sell'
            quantity (float): Order quantity
            order_type (str): 'market' or 'limit'
            price (float): Required for limit orders
            leverage (int): Leverage multiplier (1-100)
        
        Returns:
            dict: Order response or None if error
        """
        if not self.api_key:
            print("API key required for placing orders")
            return None
            
        endpoint = "/api/v1/order"
        url = f"{self.base_url}{endpoint}"
        
        # Validate inputs
        if side not in ["buy", "sell"]:
            print(f"Invalid side: {side}. Must be 'buy' or 'sell'")
            return None
            
        if order_type not in ["market", "limit"]:
            print(f"Invalid order type: {order_type}. Must be 'market' or 'limit'")
            return None
            
        if order_type == "limit" and price is None:
            print("Price required for limit orders")
            return None
        
        # Prepare order payload
        payload = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": float(quantity),
            "leverage": int(leverage)
        }
        
        if order_type == "limit":
            payload["price"] = float(price)
        
        # Add authentication
        signature_headers = self._generate_signature(payload)
        
        headers = {
            "X-HL-API-Key": self.api_key,
            "X-HL-Signature": signature_headers["signature"],
            "X-HL-Timestamp": str(signature_headers["timestamp"]),
            "Content-Type": "application/json"
        }
        
        try:
            response = self.session.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error placing order: {e}")
            return None
            
    def cancel_order(self, order_id):
        """Cancel an open order"""
        if not self.api_key:
            print("API key required for cancelling orders")
            return None
            
        endpoint = "/api/v1/order"
        url = f"{self.base_url}{endpoint}"
        
        payload = {
            "orderId": order_id
        }
        
        # Add authentication
        signature_headers = self._generate_signature(payload)
        
        headers = {
            "X-HL-API-Key": self.api_key,
            "X-HL-Signature": signature_headers["signature"],
            "X-HL-Timestamp": str(signature_headers["timestamp"]),
            "Content-Type": "application/json"
        }
        
        try:
            response = self.session.delete(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error cancelling order {order_id}: {e}")
            return None
            
    def get_open_orders(self, symbol=None):
        """Get all open orders, optionally filtered by symbol"""
        if not self.api_key:
            print("API key required for fetching open orders")
            return None
            
        endpoint = "/api/v1/openOrders"
        url = f"{self.base_url}{endpoint}"
        
        payload = {}
        if symbol:
            payload["symbol"] = symbol
        
        # Add authentication
        signature_headers = self._generate_signature(payload)
        
        headers = {
            "X-HL-API-Key": self.api_key,
            "X-HL-Signature": signature_headers["signature"],
            "X-HL-Timestamp": str(signature_headers["timestamp"])
        }
        
        try:
            response = self.session.get(url, headers=headers, params=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching open orders: {e}")
            return None