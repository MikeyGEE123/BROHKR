# tests/test_trading_strategy.py

import pytest
from trading_strategy.strategies.momentum import MomentumStrategy
from trading_strategy.strategies.mean_reversion import MeanReversionStrategy

def test_momentum_strategy_buy():
    # Set up historical data and a current price high enough to trigger a "buy" signal.
    strat = MomentumStrategy(lookback_period=5)
    data = {
        "historical_prices": [100, 101, 102, 103, 104],
        "current_price": 105  # Average is 102; 102 * 1.01 = 103.02; 105 > 103.02 → "buy"
    }
    signal = strat.calculate_signal(data)
    assert signal == "buy"

def test_momentum_strategy_sell():
    strat = MomentumStrategy(lookback_period=5)
    data = {
        "historical_prices": [100, 101, 102, 103, 104],
        "current_price": 100  # Average is 102; 102 * 0.99 ≈ 100.98; 100 < 100.98 → "sell"
    }
    signal = strat.calculate_signal(data)
    assert signal == "sell"

def test_momentum_strategy_hold():
    strat = MomentumStrategy(lookback_period=5)
    data = {
        "historical_prices": [100, 101, 102, 103, 104],
        "current_price": 102  # Within the neutral range → "hold"
    }
    signal = strat.calculate_signal(data)
    assert signal == "hold"

def test_mean_reversion_strategy_buy():
    strat = MeanReversionStrategy(threshold=0.05, window_size=5)
    data = {
        "historical_prices": [100, 102, 101, 103, 105],
        "current_price": 94   # Moving average = ~102.2; lower threshold = 102.2 * (1 - 0.05) ≈ 97.09; 94 < 97.09 → "buy"
    }
    signal = strat.calculate_signal(data)
    assert signal == "buy"

def test_mean_reversion_strategy_sell():
    strat = MeanReversionStrategy(threshold=0.05, window_size=5)
    data = {
        "historical_prices": [100, 102, 101, 103, 105],
        "current_price": 110  # 110 > 102.2 * (1 + 0.05) ≈ 107.31 → "sell"
    }
    signal = strat.calculate_signal(data)
    assert signal == "sell"

def test_mean_reversion_strategy_hold():
    strat = MeanReversionStrategy(threshold=0.05, window_size=5)
    data = {
        "historical_prices": [100, 102, 101, 103, 105],
        "current_price": 103  # Close to the moving average → "hold"
    }
    signal = strat.calculate_signal(data)
    assert signal == "hold"
