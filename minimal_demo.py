#!/usr/bin/env python
# minimal_demo.py - Minimal demonstration of BROKR functionality

from trading_strategy.strategies.mean_reversion import MeanReversionStrategy
from trading_strategy.base_strategy import BaseStrategy
import json
from datetime import datetime

# Sample historical price data (no pandas/numpy required)
historical_prices = [
    50000, 50500, 51000, 52000, 51500, 51200, 50800, 50400, 
    50600, 51200, 52000, 53000, 52500, 52200, 51800, 51400,
    51800, 52200, 53000, 52500
]

def run_simple_backtest():
    """Run a minimal backtest with the Mean Reversion strategy."""
    print("BROKR - Simple Backtest Demonstration")
    print("=====================================")
    
    # Create strategy instance
    strategy = MeanReversionStrategy(threshold=0.03, window_size=5)
    print(f"Strategy: {strategy.__class__.__name__}")
    
    # Calculate and show signals for each price point
    signals = []
    for i in range(len(historical_prices)):
        if i < 5:  # Need enough data for strategy
            signal = "hold"
        else:
            data = {
                "historical_prices": historical_prices[:i],
                "current_price": historical_prices[i]
            }
            signal = strategy.calculate_signal(data)
        signals.append(signal)
    
    # Print results in table format
    print("\nPrice Data and Signals:")
    print("-----------------------")
    print("  Day  |   Price   |  Signal  ")
    print("-------|-----------|----------")
    for i, (price, signal) in enumerate(zip(historical_prices, signals)):
        print(f"  {i+1:2d}   |  ${price:6,d}  |   {signal:<6s}  ")
    
    # Count signal types
    buy_count = signals.count("buy")
    sell_count = signals.count("sell")
    hold_count = signals.count("hold")
    
    print("\nSignal Summary:")
    print(f"- Buy signals:  {buy_count}")
    print(f"- Sell signals: {sell_count}")
    print(f"- Hold signals: {hold_count}")
    
    # Save results to file
    results = {
        "timestamp": datetime.now().isoformat(),
        "strategy": strategy.__class__.__name__,
        "parameters": {
            "threshold": 0.03,
            "window_size": 5
        },
        "prices": historical_prices,
        "signals": signals,
        "summary": {
            "buy_count": buy_count,
            "sell_count": sell_count,
            "hold_count": hold_count
        }
    }
    
    with open("minimal_backtest_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to minimal_backtest_results.json")

if __name__ == "__main__":
    run_simple_backtest()
