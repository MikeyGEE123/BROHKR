# data_ingestion/exchange_connector_base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging
import time
from enum import Enum

# Set up logging
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Enum representing different order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_LIMIT = "stop_loss_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"

class OrderSide(Enum):
    """Enum representing order sides."""
    BUY = "buy"
    SELL = "sell"

class ExchangeConnector(ABC):
    """
    Abstract base class for all exchange connectors.
    
    This class defines the interface that all exchange connectors should implement,
    ensuring consistent behavior across different exchanges.
    """
    
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True):
        """
        Initialize the exchange connector.
        
        Args:
            api_key (str): The API key for the exchange.
            api_secret (str): The API secret for the exchange.
            testnet (bool): Whether to use the testnet (sandbox) environment.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = None
        self.name = self.__class__.__name__
        self.rate_limits = {"requests_per_minute": 60, "last_request_time": 0, "request_count": 0}
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the connection to the exchange.
        
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get the current ticker for a symbol.
        
        Args:
            symbol (str): The trading pair symbol, e.g., 'BTCUSDT'.
            
        Returns:
            Dict[str, Any]: Dictionary containing ticker information.
        """
        pass
    
    @abstractmethod
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get the order book for a symbol.
        
        Args:
            symbol (str): The trading pair symbol.
            limit (int): Maximum number of orders to return.
            
        Returns:
            Dict[str, Any]: Dictionary containing order book.
        """
        pass
    
    @abstractmethod
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
            limit (int): Maximum number of klines to return.
            
        Returns:
            List[Dict[str, Any]]: List of klines.
        """
        pass
    
    @abstractmethod
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                          quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place an order on the exchange.
        
        Args:
            symbol (str): The trading pair symbol.
            side (OrderSide): Whether to buy or sell.
            order_type (OrderType): Type of order (market, limit, etc.).
            quantity (float): Quantity to buy/sell.
            price (float, optional): Price for limit orders.
            
        Returns:
            Dict[str, Any]: Dictionary containing order information.
        """
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information and balances.
        
        Returns:
            Dict[str, Any]: Dictionary containing account information.
        """
        pass
    
    @abstractmethod
    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange information, including trading rules and symbol information.
        
        Returns:
            Dict[str, Any]: Dictionary containing exchange information.
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the connection to the exchange."""
        pass
    
    async def _handle_rate_limits(self) -> None:
        """
        Handle rate limiting to avoid exceeding exchange limits.
        This is a helper method that should be called before making API requests.
        """
        # Simple rate limiting implementation
        current_time = time.time()
        minute_passed = current_time - self.rate_limits["last_request_time"] >= 60
        
        if minute_passed:
            # Reset counter if a minute has passed
            self.rate_limits["request_count"] = 0
            self.rate_limits["last_request_time"] = current_time
        elif self.rate_limits["request_count"] >= self.rate_limits["requests_per_minute"]:
            # Wait until the next minute if limit exceeded
            wait_time = 60 - (current_time - self.rate_limits["last_request_time"])
            logger.warning(f"Rate limit reached for {self.name}. Waiting {wait_time:.2f} seconds.")
            await asyncio.sleep(wait_time)
            self.rate_limits["request_count"] = 0
            self.rate_limits["last_request_time"] = time.time()
        
        # Increment request counter
        self.rate_limits["request_count"] += 1
        
    def _standardize_response(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """
        Standardize the response data from the exchange to have a consistent format.
        
        Args:
            data (Dict[str, Any]): The raw response data from the exchange.
            data_type (str): The type of data (e.g., 'ticker', 'order_book', 'account').
            
        Returns:
            Dict[str, Any]: Standardized data.
        """
        # Default implementation - subclasses should override this
        return {
            "exchange": self.name,
            "timestamp": int(time.time() * 1000),
            "data_type": data_type,
            "raw_data": data
        }
