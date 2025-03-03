# trading_strategy/base_strategy.py

from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    """
    
    @abstractmethod
    def calculate_signal(self, data):
        """
        Calculate the trading signal based on input data.
        
        Args:
            data (dict): A dictionary containing market data.
            
        Returns:
            str: Trading signal (e.g., "buy", "sell", "hold").
        """
        pass
