# tests/test_binance_integration.py

import pytest
import asyncio
from data_ingestion.binance_connector import BinanceConnector

@pytest.mark.asyncio
async def test_binance_connection():
    """Test basic connection to Binance API."""
    connector = BinanceConnector()
    try:
        # Test connection by fetching BTC/USDT ticker
        ticker = await connector.get_market_data("BTCUSDT")
        
        # Assert we got a valid response
        assert ticker is not None
        assert "symbol" in ticker
        assert "price" in ticker
        assert ticker["symbol"] == "BTCUSDT"
        assert float(ticker["price"]) > 0
        
        print(f"Successfully connected to Binance API. BTC price: {ticker['price']} USDT")
    finally:
        # Ensure we close the connection properly
        await connector.close_connection()

@pytest.mark.asyncio
async def test_multiple_symbols():
    """Test fetching data for multiple trading pairs."""
    connector = BinanceConnector()
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    results = {}
    
    try:
        for symbol in symbols:
            ticker = await connector.get_market_data(symbol)
            results[symbol] = ticker
            
            # Verify the response
            assert ticker is not None
            assert ticker["symbol"] == symbol
            assert float(ticker["price"]) > 0
            
        # Print a summary of prices
        for symbol, ticker in results.items():
            print(f"{symbol}: {ticker['price']} USDT")
    finally:
        await connector.close_connection()

# This allows running the tests directly from command line
if __name__ == "__main__":
    asyncio.run(test_binance_connection())
    asyncio.run(test_multiple_symbols())
