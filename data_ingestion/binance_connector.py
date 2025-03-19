from binance.client import AsyncClient
from binance import BinanceAPIException, BinanceOrderException
from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET
import logging

logger = logging.getLogger(__name__)

class BinanceConnector:
    def __init__(self):
        self.client = AsyncClient(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True) # Initialize async client

    async def get_market_data(self, symbol="BTCUSDT"):
        """
        Fetch the current price for a given symbol (e.g., BTC/USDT) using async client.
        """
        try:
            ticker = await self.client.get_symbol_ticker(symbol=symbol) # Use await for async call
            return ticker
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching market data for {symbol}: {e}")
            return None
        except BinanceOrderException as e:
            logger.error(f"Binance order error fetching market data for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching market data for {symbol}: {e}")
            return None

    async def close_connection(self):
        await self.client.close_connection()
