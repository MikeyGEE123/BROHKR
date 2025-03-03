# trading_execution/order_manager.py

import logging
import asyncio
from trading_execution.exchange_interface import ExchangeInterface

logger = logging.getLogger(__name__)

class OrderManager:
    """
    Handles order operations, including creation, cancellation, and status inquiries,
    by leveraging the ExchangeInterface.
    """
    
    def __init__(self, exchange_interface: ExchangeInterface):
        """
        Initializes the OrderManager with an ExchangeInterface instance.
        
        Args:
            exchange_interface (ExchangeInterface): The interface to interact with the exchange.
        """
        self.exchange = exchange_interface

    async def place_order(self, side: str, symbol: str, quantity: float, price: float = None, order_type: str = "market") -> dict:
        """
        Places an order for a given symbol and quantity.

        Args:
            side (str): "buy" or "sell" indicating order direction.
            symbol (str): The trading pair symbol (e.g., 'BTCUSD').
            quantity (float): The amount to trade.
            price (float, optional): The order price for limit orders.
            order_type (str): The type of order, either "market" or "limit".

        Returns:
            dict: The order details returned by the exchange.
        """
        try:
            order_data = await self.exchange.create_order(order_type, symbol, side, quantity, price)
            logger.info(f"Order placed: {order_data}")
            return order_data
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        """
        Cancels an existing order.

        Args:
            order_id (str): The ID of the order to cancel.
            symbol (str): The trading pair symbol.

        Returns:
            dict: Response confirming cancellation.
        """
        try:
            cancel_response = await self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Order cancelled: {order_id}")
            return cancel_response
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            raise

    async def get_order_status(self, order_id: str, symbol: str) -> dict:
        """
        Retrieves the status of a given order.

        Args:
            order_id (str): The order identifier.
            symbol (str): The trading pair symbol.

        Returns:
            dict: The current status of the order.
        """
        try:
            status_response = await self.exchange.get_order_status(order_id, symbol)
            logger.info(f"Order status for {order_id}: {status_response}")
            return status_response
        except Exception as e:
            logger.error(f"Error retrieving status for order {order_id}: {e}")
            raise

# Example usage for testing purposes.
if __name__ == "__main__":
    import os

    # Configure basic logging for demonstration
    import logging
    logging.basicConfig(level=logging.INFO)

    async def main():
        # Replace with appropriate base URL and credentials
        base_url = os.getenv("API_BASE_URL", "https://api.example.com")
        api_key = os.getenv("API_KEY", "your_api_key")
        api_secret = os.getenv("API_SECRET", "your_api_secret")

        # Initialize the exchange interface
        exchange_interface = ExchangeInterface(base_url, api_key, api_secret)
        order_manager = OrderManager(exchange_interface)

        # Demo: Place a market order for demonstration purposes.
        try:
            order_details = await order_manager.place_order(side="buy", symbol="BTCUSD", quantity=0.01)
            print("Order Details:", order_details)

            # Assuming the order details include an 'order_id'; you can check order status:
            order_id = order_details.get("order_id", "demo_order_id")
            status = await order_manager.get_order_status(order_id, "BTCUSD")
            print("Order Status:", status)

            # Optionally cancel the order (if supported and necessary):
            canceled = await order_manager.cancel_order(order_id, "BTCUSD")
            print("Cancel Response:", canceled)
        except Exception as ex:
            print("An error occurred during order processing:", ex)

    asyncio.run(main())
