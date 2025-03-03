# analysis/data_analyzer.py

import statistics
import logging

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """
    Analyzes market data and computes relevant statistics for each trading symbol.
    """

    def __init__(self):
        pass

    def analyze_data(self, data: list) -> dict:
        """
        Analyze a list of normalized market data dictionaries.

        Args:
            data (list): List of market data records (e.g.,
                         [{"symbol": "BTCUSD", "price": 10000, "timestamp": "2023-10-15T10:00:00Z"}, ...])
        
        Returns:
            dict: Mapping of symbols to their computed analysis including:
                  - average price
                  - minimum price
                  - maximum price
                  - volatility (standard deviation)
                  - moving averages (computed with a default window)
                  - raw data points (prices)
        """
        results = {}
        data_by_symbol = {}

        # Group data by symbol.
        for record in data:
            symbol = record.get("symbol", "UNKNOWN")
            price = record.get("price", 0.0)
            data_by_symbol.setdefault(symbol, []).append(price)

        # Process each symbol.
        for symbol, prices in data_by_symbol.items():
            if not prices:
                continue

            average_price = sum(prices) / len(prices)
            min_price = min(prices)
            max_price = max(prices)
            stdev_price = statistics.stdev(prices) if len(prices) > 1 else 0.0
            # Compute moving averages with a default window of 3.
            moving_average = self.compute_moving_average(prices, window=3)

            results[symbol] = {
                "average": average_price,
                "min": min_price,
                "max": max_price,
                "volatility": stdev_price,
                "moving_average": moving_average,
                "data_points": prices,
            }
            logger.info(f"Analysis for {symbol}: {results[symbol]}")

        return results

    def compute_moving_average(self, prices: list, window: int = 3) -> list:
        """
        Compute the simple moving average over a given window for a list of prices.

        Args:
            prices (list): List of price floats.
            window (int): Number of points to include in each average.
        
        Returns:
            list: List of computed moving averages.
        """
        if len(prices) < window:
            return []
        moving_averages = []
        for i in range(len(prices) - window + 1):
            window_prices = prices[i:i + window]
            avg = sum(window_prices) / window
            moving_averages.append(avg)
        return moving_averages

# Example usage for testing purposes.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = DataAnalyzer()
    sample_data = [
        {"symbol": "BTCUSD", "price": 10000, "timestamp": "2023-10-15T10:00:00Z"},
        {"symbol": "BTCUSD", "price": 10100, "timestamp": "2023-10-15T10:01:00Z"},
        {"symbol": "BTCUSD", "price": 10200, "timestamp": "2023-10-15T10:02:00Z"},
        {"symbol": "ETHUSD", "price": 300, "timestamp": "2023-10-15T10:00:00Z"},
        {"symbol": "ETHUSD", "price": 305, "timestamp": "2023-10-15T10:01:00Z"},
        {"symbol": "ETHUSD", "price": 310, "timestamp": "2023-10-15T10:02:00Z"},
    ]
    analysis_results = analyzer.analyze_data(sample_data)
    print("Analysis Results:")
    for symbol, stats in analysis_results.items():
        print(f"{symbol}: {stats}")
