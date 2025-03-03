# trading_execution/exchange_interface.py

import aiohttp
import logging

logger = logging.getLogger(__name__)

class ExchangeInterface:
    """
    Provides an interface for executing trade orders on the exchange.
    """

    def __init__(self, base_url: str, api_key: str = None, api_secret: str = None):
        """
        Initializes the ExchangeInterface with API credentials.

        Args:
            base_url (str): The base URL for the exchange API.
            api_key (str, optional): The API key provided by the exchange.
            api_secret (str, optional): The API secret provided by the exchange.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.api_secret = api_secret

    async def create_order(self, order_type: str, symbol: str, side: str, quantity: float, price: float = None) -> dict:
        """
        Creates an order on the exchange.

        Args:
            order_type (str): The type of order ('market' or 'limit').
            symbol (str): The trading pair symbol (e.g., 'BTCUSD').
            side (str): The side of the order ('buy' or 'sell').
            quantity (float): The amount to trade.
            price (float, optional): The price for limit orders.

        Returns:
            dict: Response from the exchange API containing order details.
        """
        url = f"{self.base_url}/order"
        payload = {
            "type": order_type,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
        }
        if order_type.lower() == "limit" and price is not None:
            payload["price"] = price

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        """
        Cancels an existing order by its ID.

        Args:
            order_id (str): The identifier of the order to cancel.
            symbol (str): The trading pair symbol.

        Returns:
            dict: Response from the exchange API confirming cancellation.
        """
        url = f"{self.base_url}/order/{order_id}/cancel"
        payload = {"symbol": symbol}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            raise

    async def get_order_status(self, order_id: str, symbol: str) -> dict:
        """
        Retrieves the status of an existing order.

        Args:
            order_id (str): The identifier of the order.
            symbol (str): The trading pair symbol.

        Returns:
            dict: Order status details from the exchange API.
        """
        url = f"{self.base_url}/order/{order_id}/status"
        params = {"symbol": symbol}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data
        except Exception as e:
            logger.error(f"Error fetching status for order {order_id}: {e}")
            raise
