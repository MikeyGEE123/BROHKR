# data_ingestion/data_parser.py

import logging

logger = logging.getLogger(__name__)

def parse_market_data(raw_data: dict) -> dict:
    """
    Parses and normalizes raw market data from an exchange API.

    Args:
        raw_data (dict): Raw JSON data retrieved from the API.

    Returns:
        dict: A normalized dictionary containing key market data elements such as:
              - 'symbol': Trading pair symbol
              - 'price': Latest traded price as a float
              - 'timestamp': The time the data was fetched

    Raises:
        Exception: If the expected data fields are not present or conversion fails.
    """
    try:
        ticker = raw_data.get("ticker", {})
        normalized_data = {
            "symbol": ticker.get("symbol", "UNKNOWN"),
            "price": float(ticker.get("price", 0.0)),
            "timestamp": raw_data.get("timestamp")
        }
        return normalized_data
    except Exception as e:
        logger.error(f"Error parsing market data: {e}")
        raise
