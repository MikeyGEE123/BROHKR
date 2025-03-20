# data_ingestion/binance_connector_v2.py

import asyncio
import time
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from binance import AsyncClient, BinanceAPIException
from binance.helpers import convert_ts_str

from data_ingestion.exchange_connector_base import ExchangeConnector, OrderType, OrderSide

# Set up logging
logger = logging.getLogger(__name__)

class BinanceConnector(ExchangeConnector):
    """
    Implementation of ExchangeConnector for the Binance exchange.
    
    This class provides methods to interact with the Binance API for trading and data retrieval.
    """
    
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True):
        """Initialize the Binance connector with API credentials."""
        super().__init__(api_key, api_secret, testnet)
        self.name = "Binance"
        self.rate_limits = {"requests_per_minute": 1200, "last_request_time": 0, "request_count": 0}
        self.client = None
    
    async def initialize(self) -> bool:
        """
        Initialize the connection to Binance.
        
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            self.client = await AsyncClient.create(
                api_key=self.api_key, 
                api_secret=self.api_secret,
                testnet=self.testnet
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {e}")
            return False
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get the current ticker price for a symbol.
        
        Args:
            symbol (str): The trading pair symbol, e.g., 'BTCUSDT'.
            
        Returns:
            Dict[str, Any]: Dictionary containing ticker information.
        """
        await self._handle_rate_limits()
        try:
            ticker = await self.client.get_symbol_ticker(symbol=symbol)
            return self._standardize_response(ticker, "ticker")
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting ticker for {symbol}: {e}")
            return {"error": str(e), "code": e.code}
        except Exception as e:
            logger.error(f"Unexpected error getting ticker for {symbol}: {e}")
            return {"error": str(e)}
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get the order book for a symbol.
        
        Args:
            symbol (str): The trading pair symbol.
            limit (int): Maximum number of orders to return (default: 100, max: 5000).
            
        Returns:
            Dict[str, Any]: Dictionary containing order book data.
        """
        await self._handle_rate_limits()
        try:
            order_book = await self.client.get_order_book(symbol=symbol, limit=limit)
            return self._standardize_response(order_book, "order_book")
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting order book for {symbol}: {e}")
            return {"error": str(e), "code": e.code}
        except Exception as e:
            logger.error(f"Unexpected error getting order book for {symbol}: {e}")
            return {"error": str(e)}
    
    async def get_historical_klines(self, symbol: str, interval: str, 
                                   start_time: Optional[int] = None, 
                                   end_time: Optional[int] = None, 
                                   limit: int = 500) -> List[Dict[str, Any]]:
        """
        Get historical klines/candlesticks for a symbol.
        
        Args:
            symbol (str): The trading pair symbol.
            interval (str): Kline interval (e.g., '1m', '1h', '1d').
            start_time (int, optional): Start time in milliseconds.
            end_time (int, optional): End time in milliseconds.
            limit (int): Maximum number of klines to return (default: 500, max: 1000).
            
        Returns:
            List[Dict[str, Any]]: List of klines with OHLCV data.
        """
        await self._handle_rate_limits()
        try:
            # Convert datetime to timestamp if provided
            if isinstance(start_time, datetime):
                start_time = int(start_time.timestamp() * 1000)
            if isinstance(end_time, datetime):
                end_time = int(end_time.timestamp() * 1000)
                
            klines = await self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_time,
                end_str=end_time,
                limit=limit
            )
            
            # Format klines into a more readable format
            formatted_klines = []
            for k in klines:
                formatted_kline = {
                    "open_time": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": k[6],
                    "quote_asset_volume": float(k[7]),
                    "number_of_trades": int(k[8]),
                    "taker_buy_base_asset_volume": float(k[9]),
                    "taker_buy_quote_asset_volume": float(k[10])
                }
                formatted_klines.append(formatted_kline)
                
            return self._standardize_response(formatted_klines, "klines")
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting klines for {symbol}: {e}")
            return {"error": str(e), "code": e.code}
        except Exception as e:
            logger.error(f"Unexpected error getting klines for {symbol}: {e}")
            return {"error": str(e)}
    
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                         quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place an order on Binance.
        
        Args:
            symbol (str): The trading pair symbol.
            side (OrderSide): Whether to buy or sell.
            order_type (OrderType): Type of order (market, limit, etc.).
            quantity (float): Quantity to buy/sell.
            price (float, optional): Price for limit orders.
            
        Returns:
            Dict[str, Any]: Dictionary containing order information.
        """
        if not self.api_key or not self.api_secret:
            return {"error": "API key and secret are required for trading"}
        
        await self._handle_rate_limits()
        try:
            params = {
                "symbol": symbol,
                "side": side.value.upper(),
                "type": order_type.value.upper(),
                "quantity": quantity
            }
            
            # Add price for limit orders
            if order_type in [OrderType.LIMIT, OrderType.STOP_LOSS_LIMIT, OrderType.TAKE_PROFIT_LIMIT]:
                if price is None:
                    return {"error": "Price is required for limit orders"}
                params["price"] = price
                params["timeInForce"] = "GTC"  # Good Till Cancelled
            
            order = await self.client.create_order(**params)
            return self._standardize_response(order, "order")
        except BinanceAPIException as e:
            logger.error(f"Binance API error placing order: {e}")
            return {"error": str(e), "code": e.code}
        except Exception as e:
            logger.error(f"Unexpected error placing order: {e}")
            return {"error": str(e)}
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information, including balances.
        
        Returns:
            Dict[str, Any]: Dictionary containing account information.
        """
        if not self.api_key or not self.api_secret:
            return {"error": "API key and secret are required for account information"}
        
        await self._handle_rate_limits()
        try:
            account = await self.client.get_account()
            
            # Format balances to only include non-zero balances
            non_zero_balances = [
                {
                    "asset": balance["asset"],
                    "free": float(balance["free"]),
                    "locked": float(balance["locked"]),
                    "total": float(balance["free"]) + float(balance["locked"])
                }
                for balance in account["balances"]
                if float(balance["free"]) > 0 or float(balance["locked"]) > 0
            ]
            
            account["non_zero_balances"] = non_zero_balances
            return self._standardize_response(account, "account")
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting account info: {e}")
            return {"error": str(e), "code": e.code}
        except Exception as e:
            logger.error(f"Unexpected error getting account info: {e}")
            return {"error": str(e)}
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange information, including trading rules and symbol information.
        
        Returns:
            Dict[str, Any]: Dictionary containing exchange information.
        """
        await self._handle_rate_limits()
        try:
            exchange_info = await self.client.get_exchange_info()
            return self._standardize_response(exchange_info, "exchange_info")
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting exchange info: {e}")
            return {"error": str(e), "code": e.code}
        except Exception as e:
            logger.error(f"Unexpected error getting exchange info: {e}")
            return {"error": str(e)}
    
    async def close(self) -> None:
        """Close the connection to Binance."""
        if self.client:
            await self.client.close_connection()
    
    def _standardize_response(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """
        Standardize the response data from Binance to have a consistent format.
        
        Args:
            data (Dict[str, Any]): The raw response data from Binance.
            data_type (str): The type of data.
            
        Returns:
            Dict[str, Any]: Standardized data.
        """
        standardized = {
            "exchange": self.name,
            "timestamp": int(time.time() * 1000),
            "data_type": data_type,
        }
        
        # Add specific formatting based on data type
        if data_type == "ticker":
            standardized["symbol"] = data.get("symbol")
            standardized["price"] = float(data.get("price", 0))
        elif data_type == "order_book":
            standardized["symbol"] = data.get("symbol")
            standardized["last_update_id"] = data.get("lastUpdateId")
            standardized["bids"] = [[float(p), float(q)] for p, q in data.get("bids", [])]
            standardized["asks"] = [[float(p), float(q)] for p, q in data.get("asks", [])]
        elif data_type == "klines":
            standardized["data"] = data
        elif data_type == "order":
            standardized["order_id"] = data.get("orderId")
            standardized["client_order_id"] = data.get("clientOrderId")
            standardized["symbol"] = data.get("symbol")
            standardized["status"] = data.get("status")
            standardized["price"] = float(data.get("price", 0))
            standardized["original_qty"] = float(data.get("origQty", 0))
            standardized["executed_qty"] = float(data.get("executedQty", 0))
        elif data_type == "account":
            standardized["account_type"] = data.get("accountType")
            standardized["balances"] = data.get("non_zero_balances", [])
            standardized["permissions"] = data.get("permissions", [])
        elif data_type == "exchange_info":
            standardized["timezone"] = data.get("timezone")
            standardized["server_time"] = data.get("serverTime")
            
            # Extract important trading pair info
            trading_pairs = []
            for symbol in data.get("symbols", []):
                pair_info = {
                    "symbol": symbol.get("symbol"),
                    "base_asset": symbol.get("baseAsset"),
                    "quote_asset": symbol.get("quoteAsset"),
                    "status": symbol.get("status"),
                    "min_notional": None,
                    "min_qty": None,
                    "max_qty": None,
                    "step_size": None,
                    "tick_size": None,
                }
                
                # Extract the numeric filters
                for filter_obj in symbol.get("filters", []):
                    if filter_obj.get("filterType") == "PRICE_FILTER":
                        pair_info["tick_size"] = float(filter_obj.get("tickSize", 0))
                    elif filter_obj.get("filterType") == "LOT_SIZE":
                        pair_info["min_qty"] = float(filter_obj.get("minQty", 0))
                        pair_info["max_qty"] = float(filter_obj.get("maxQty", 0))
                        pair_info["step_size"] = float(filter_obj.get("stepSize", 0))
                    elif filter_obj.get("filterType") == "MIN_NOTIONAL":
                        pair_info["min_notional"] = float(filter_obj.get("minNotional", 0))
                
                trading_pairs.append(pair_info)
            
            standardized["trading_pairs"] = trading_pairs
        
        # Add the raw data for reference
        standardized["raw_data"] = data
        
        return standardized

# Example usage
async def example_usage():
    connector = BinanceConnector(testnet=True)
    if await connector.initialize():
        # Get ticker
        ticker = await connector.get_ticker("BTCUSDT")
        print(f"BTC/USDT price: ${ticker['price']}")
        
        # Get account info (requires API key/secret)
        account = await connector.get_account_info()
        if "error" not in account:
            print(f"Account balances: {account['balances']}")
        
        await connector.close()

if __name__ == "__main__":
    asyncio.run(example_usage())
