# trading_strategy/strategies/momentum.py

from trading_strategy.base_strategy import BaseStrategy

class MomentumStrategy(BaseStrategy):
    """
    Momentum-based trading strategy.
    
    Signals:
      - "buy" when the current price is above the historical average by a small margin.
      - "sell" when it is below the historical average.
      - "hold" when within a neutral range.
    """

    def __init__(self, lookback_period=5):
        self.lookback_period = lookback_period

    def calculate_signal(self, data):
        """
        Calculate trading signal based on momentum.
        
        Args:
            data (dict): Should contain:
                - 'historical_prices': list of prices
                - 'current_price': current price as float
            
        Returns:
            str: "buy", "sell", or "hold"
        """
        historical_prices = data.get("historical_prices", [])
        current_price = data.get("current_price")

        # If there is insufficient data, we cannot generate a signal.
        if not historical_prices or current_price is None or len(historical_prices) < self.lookback_period:
            return "hold"

        # Calculate average from the last 'lookback_period' values.
        recent_prices = historical_prices[-self.lookback_period:]
        average_price = sum(recent_prices) / len(recent_prices)

        # Determine signal based on a 1% deviation from the average.
        if current_price < average_price * 0.99:
            return "sell"
        elif current_price > average_price * 1.01:
            return "buy"
        else:
            return "hold"
