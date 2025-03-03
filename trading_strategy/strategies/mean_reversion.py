# trading_strategy/strategies/mean_reversion.py

from trading_strategy.base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion trading strategy.
    
    Signals:
      - "buy" when the current price is significantly below the moving average.
      - "sell" when it is significantly above.
      - "hold" when the price is close to the moving average.
    """

    def __init__(self, threshold=0.05, window_size=10):
        """
        Args:
            threshold (float): Percentage deviation (5% default) for signaling.
            window_size (int): Number of data points to compute the moving average.
        """
        self.threshold = threshold
        self.window_size = window_size

    def calculate_signal(self, data):
        """
        Calculate trading signal based on mean reversion logic.
        
        Args:
            data (dict): Should contain:
                - 'historical_prices': list of prices
                - 'current_price': current price as float
            
        Returns:
            str: "buy", "sell", or "hold"
        """
        historical_prices = data.get("historical_prices", [])
        current_price = data.get("current_price")

        if len(historical_prices) < self.window_size or current_price is None:
            return "hold"

        # Compute moving average over the defined window.
        recent_prices = historical_prices[-self.window_size:]
        moving_average = sum(recent_prices) / len(recent_prices)

        if current_price < moving_average * (1 - self.threshold):
            return "buy"
        elif current_price > moving_average * (1 + self.threshold):
            return "sell"
        else:
            return "hold"
