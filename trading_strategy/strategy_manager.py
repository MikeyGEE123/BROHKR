# trading_strategy/strategy_manager.py

import logging
from trading_strategy.strategies.momentum import MomentumStrategy
from trading_strategy.strategies.mean_reversion import MeanReversionStrategy

logger = logging.getLogger(__name__)

class StrategyManager:
    """
    Manages and evaluates multiple trading strategies.
    """

    def __init__(self):
        self.strategies = []

    def register_strategy(self, strategy):
        """
        Add a new strategy to the manager.
        
        Args:
            strategy (BaseStrategy): A strategy instance.
        """
        self.strategies.append(strategy)

    def evaluate_strategies(self, data):
        """
        Evaluate all registered strategies and return their signals.
        
        Args:
            data (dict): Market data to evaluate signals.
            
        Returns:
            dict: Strategy names mapped to their generated signals.
        """
        signals = {}
        for strategy in self.strategies:
            strategy_name = strategy.__class__.__name__
            try:
                signal = strategy.calculate_signal(data)
                signals[strategy_name] = signal
                logger.info(f"{strategy_name} signal: {signal}")
            except Exception as e:
                logger.error(f"Error in {strategy_name}: {e}")
                signals[strategy_name] = "error"
        return signals

# Example usage for testing purposes.
if __name__ == "__main__":
    # Sample data for testing purposes.
    sample_data = {
        "historical_prices": [100, 102, 101, 103, 105, 107, 106, 108, 110, 111],
        "current_price": 112
    }

    manager = StrategyManager()
    # Register the strategies with desired parameters.
    manager.register_strategy(MomentumStrategy(lookback_period=5))
    manager.register_strategy(MeanReversionStrategy(threshold=0.05, window_size=5))

    # Evaluate all strategies.
    signals = manager.evaluate_strategies(sample_data)
    print("Trading Signals:", signals)
