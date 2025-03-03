# data_ingestion/api_connector.py

import aiohttp
import asyncio
import logging

logger = logging.getLogger(__name__)

class ExchangeAPIConnector:
    """
    Connects to a cryptocurrency exchange API to fetch market data asynchronously.
    """

    def __init__(self, base_url: str):
        """
        Initializes the connector with a base URL.
        
        Args:
            base_url (str): The base URL for the exchange API.
        """
        self.base_url = base_url

    async def fetch_market_data(self, endpoint: str, params: dict = None) -> dict:
        """
        Fetches market data from a specified endpoint.

        Args:
            endpoint (str): The API endpoint for fetching data (e.g., '/api/v1/ticker').
            params (dict): Optional URL parameters.

        Returns:
            dict: Parsed JSON data from the API.

        Raises:
            Exception: If an error occurs during the HTTP request.
        """
        url = f"{self.base_url}{endpoint}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()  # Throws an exception for 4XX/5XX errors
                    data = await response.json()
                    return data
        except Exception as e:
            logger.error(f"Error fetching market data from {url}: {e}")
            raise
