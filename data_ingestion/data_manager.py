# data_ingestion/data_manager.py

import asyncio
import logging
from data_ingestion.api_connector import ExchangeAPIConnector
from data_ingestion.data_parser import parse_market_data

logger = logging.getLogger(__name__)

class DataManager:
    """
    Orchestrates the fetching and parsing of market data from multiple exchange connectors.
    """

    def __init__(self, connectors: list):
        """
        Initializes the DataManager with a list of ExchangeAPIConnector instances.
        
        Args:
            connectors (list): A list of instantiated ExchangeAPIConnector objects.
        """
        self.connectors = connectors

    async def fetch_all_market_data(self) -> list:
        """
        Fetches and normalizes market data from all configured connectors concurrently.

        Returns:
            list: A list of normalized data dictionaries.
        """
        tasks = []
        for connector in self.connectors:
            # For demonstration, we assume the same endpoint for all connectors.
            task = asyncio.create_task(self.fetch_and_parse_data(connector, endpoint="/api/v3/ticker/24hr"))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        normalized_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error during data fetch and parse: {result}")
            else:
                normalized_results.append(result)
        return normalized_results

    async def fetch_and_parse_data(self, connector: ExchangeAPIConnector, endpoint: str) -> dict:
        """
        Fetches raw data from the given connector and returns normalized market data.

        Args:
            connector (ExchangeAPIConnector): The connector to use for fetching data.
            endpoint (str): The API endpoint to access.
        
        Returns:
            dict: Normalized market data.
        """
        raw_data = await connector.fetch_market_data(endpoint)
        normalized_data = parse_market_data(raw_data)
        return normalized_data

# Example usage for testing purposes
if __name__ == "__main__":
    import logging

    # Configure basic logging to output to console
    logging.basicConfig(level=logging.INFO)

    async def main():
        # Placeholder base URL; replace with an actual API endpoint as needed.
        base_url = "https://api.example.com"
        connector = ExchangeAPIConnector(base_url)
        data_manager = DataManager([connector])
        data = await data_manager.fetch_all_market_data()
        print("Normalized Market Data:", data)
    
    asyncio.run(main())
