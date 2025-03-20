# tests/test_data_ingestion.py

import pytest
from data_ingestion import data_parser, data_manager

def test_parse_market_data_success():
    raw_data = {
        "ticker": {"symbol": "BTCUSD", "price": "10000"},
        "timestamp": "2023-10-15T10:00:00Z"
    }
    parsed = data_parser.parse_market_data(raw_data)
    assert parsed["symbol"] == "BTCUSD"
    assert parsed["price"] == 10000.0
    assert parsed["timestamp"] == "2023-10-15T10:00:00Z"

@pytest.mark.asyncio
async def test_fetch_and_parse_data():
    # Define a dummy connector that simulates fetching market data.
    class DummyConnector:
        async def fetch_market_data(self, endpoint: str, params: dict = None):
            return {
                "ticker": {"symbol": "BTCUSD", "price": "10000"},
                "timestamp": "2023-10-15T10:00:00Z"
            }
    
    dummy_connector = DummyConnector()
    dm = data_manager.DataManager([dummy_connector])
    results = await dm.fetch_all_market_data()
    assert len(results) == 1

    result = results[0]
    assert result["symbol"] == "BTCUSD"
    assert result["price"] == 10000.0
    assert result["timestamp"] == "2023-10-15T10:00:00Z"
