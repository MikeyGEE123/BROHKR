# main.py

import asyncio
import logging
from config import initialize_app  # Import configuration initializer
from data_ingestion.api_connector import ExchangeAPIConnector
from data_ingestion.data_manager import DataManager

def main():
    """
    Entry point for the BROKR platform.
    
    This function:
    - Initializes configuration and logging.
    - Sets up the necessary connectors (e.g., for data ingestion).
    - Orchestrates a sample data fetching routine.
    """
    # Initialize configuration and logging.
    settings = initialize_app()
    logger = logging.getLogger(__name__)
    logger.info("Starting BROKR Platform...")

    # Create an API connector instance using the base URL from configuration.
    connector = ExchangeAPIConnector(settings.api_base_url)
    
    # Instantiate the DataManager with the API connector.
    data_manager = DataManager([connector])
    
    # Use asyncio to fetch market data asynchronously.
    try:
        market_data = asyncio.run(data_manager.fetch_all_market_data())
        logger.info(f"Fetched Market Data: {market_data}")
    except Exception as e:
        logger.error(f"Error during market data fetch: {e}")

    # Future steps:
    # - Integrate trading strategies.
    # - Initiate order execution routines.
    # - Expand error handling and recovery.
    logger.info("BROKR Platform execution completed.")

if __name__ == "__main__":
    main()
