# tests/test_order_manager.py

import pytest
import asyncio
from trading_execution.order_manager import OrderManager
from trading_execution.exchange_interface import ExchangeInterface

# Create a dummy exchange interface to simulate API responses.
class DummyExchangeInterface(ExchangeInterface):
    def __init__(self):
        # No need for actual API credentials
        pass

    async def create_order(self, order_type: str, symbol: str, side: str, quantity: float, price: float = None) -> dict:
        return {"order_id": "dummy_order_id", "status": "created"}

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        return {"order_id": order_id, "status": "cancelled"}

    async def get_order_status(self, order_id: str, symbol: str) -> dict:
        return {"order_id": order_id, "status": "filled"}

@pytest.mark.asyncio
async def test_place_order():
    dummy_interface = DummyExchangeInterface()
    order_manager = OrderManager(dummy_interface)
    result = await order_manager.place_order(side="buy", symbol="BTCUSD", quantity=0.1)
    assert result["order_id"] == "dummy_order_id"
    assert result["status"] == "created"

@pytest.mark.asyncio
async def test_cancel_order():
    dummy_interface = DummyExchangeInterface()
    order_manager = OrderManager(dummy_interface)
    result = await order_manager.cancel_order("dummy_order_id", "BTCUSD")
    assert result["order_id"] == "dummy_order_id"
    assert result["status"] == "cancelled"

@pytest.mark.asyncio
async def test_get_order_status():
    dummy_interface = DummyExchangeInterface()
    order_manager = OrderManager(dummy_interface)
    result = await order_manager.get_order_status("dummy_order_id", "BTCUSD")
    assert result["order_id"] == "dummy_order_id"
    assert result["status"] == "filled"
