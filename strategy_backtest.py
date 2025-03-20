#!/usr/bin/env python
# strategy_backtest.py

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

from trading_strategy.strategies.mean_reversion import MeanReversionStrategy
from trading_strategy.strategies.momentum import MomentumStrategy
from analysis.visualization import TradingVisualizer

class StrategyBacktester:
    """
    Class for backtesting trading strategies using historical data.
    """
    
    def __init__(self, strategy, initial_capital=10000):
        """
        Initialize the backtester with a strategy and initial capital.
        
        Args:
            strategy: A trading strategy instance
            initial_capital (float): Initial capital in USD
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.holdings = 0  # Number of coins/shares held
        self.trades = []
        self.portfolio_values = []
        
    def run_backtest(self, historical_data):
        """
        Run a backtest on historical price data.
        
        Args:
            historical_data (pd.DataFrame): DataFrame with historical price data containing:
                - 'timestamp' or 'date'
                - 'close' or 'price'
                
        Returns:
            dict: Backtest results
        """
        # Reset state for new backtest
        self.current_capital = self.initial_capital
        self.holdings = 0
        self.trades = []
        self.portfolio_values = []
        
        price_col = 'close' if 'close' in historical_data.columns else 'price'
        time_col = 'timestamp' if 'timestamp' in historical_data.columns else 'date'
        
        # Extract timestamps and prices
        timestamps = historical_data[time_col].tolist()
        prices = historical_data[price_col].tolist()
        
        # Track signals for each timestamp
        signals = []
        
        # Process each data point
        for i in range(len(historical_data)):
            # Calculate trading signal
            if i < 10:  # Need enough data for strategy
                signal = "hold"
            else:
                data = {
                    "historical_prices": prices[:i],
                    "current_price": prices[i]
                }
                signal = self.strategy.calculate_signal(data)
            
            signals.append(signal)
            
            # Execute trade based on signal
            current_price = prices[i]
            if signal == "buy" and self.current_capital > 0:
                # Buy with 90% of available capital
                buy_amount = self.current_capital * 0.9
                self.holdings += buy_amount / current_price
                self.current_capital -= buy_amount
                
                self.trades.append({
                    'timestamp': timestamps[i],
                    'type': 'buy',
                    'price': current_price,
                    'amount': buy_amount,
                    'holdings': self.holdings,
                    'capital': self.current_capital
                })
                
            elif signal == "sell" and self.holdings > 0:
                # Sell 90% of holdings
                sell_amount = self.holdings * 0.9
                self.current_capital += sell_amount * current_price
                self.holdings -= sell_amount
                
                self.trades.append({
                    'timestamp': timestamps[i],
                    'type': 'sell',
                    'price': current_price,
                    'amount': sell_amount * current_price,
                    'holdings': self.holdings,
                    'capital': self.current_capital
                })
            
            # Calculate portfolio value
            portfolio_value = self.current_capital + (self.holdings * current_price)
            self.portfolio_values.append(portfolio_value)
        
        # Calculate benchmark (buy and hold)
        coins_bought = self.initial_capital / prices[0]
        benchmark_values = [coins_bought * price for price in prices]
        
        # Calculate performance metrics
        roi = (self.portfolio_values[-1] / self.initial_capital - 1) * 100
        max_drawdown = self._calculate_max_drawdown(self.portfolio_values)
        
        return {
            'timestamps': timestamps,
            'prices': prices,
            'signals': signals,
            'portfolio_values': self.portfolio_values,
            'benchmark_values': benchmark_values,
            'trades': self.trades,
            'roi': roi,
            'max_drawdown': max_drawdown
        }
    
    @staticmethod
    def _calculate_max_drawdown(portfolio_values):
        """Calculate maximum drawdown from a list of portfolio values."""
        max_so_far = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > max_so_far:
                max_so_far = value
            else:
                drawdown = (max_so_far - value) / max_so_far * 100
                max_drawdown = max(max_drawdown, drawdown)
                
        return max_drawdown
    
    def generate_report(self, results, output_dir="backtest_results"):
        """
        Generate visual reports and save them to the output directory.
        
        Args:
            results (dict): Results from run_backtest
            output_dir (str): Directory to save the output files
            
        Returns:
            list: Paths to generated files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        fig1 = TradingVisualizer.plot_price_history(
            results['timestamps'], 
            results['prices'], 
            "Historical Price Data"
        )
        fig1.savefig(f"{output_dir}/price_history.png")
        
        fig2 = TradingVisualizer.plot_strategy_signals(
            results['timestamps'], 
            results['prices'], 
            results['signals'], 
            "Trading Signals"
        )
        fig2.savefig(f"{output_dir}/trading_signals.png")
        
        fig3 = TradingVisualizer.plot_strategy_performance(
            results['timestamps'], 
            results['portfolio_values'],
            results['benchmark_values'],
            "Strategy vs. Buy & Hold"
        )
        fig3.savefig(f"{output_dir}/performance_comparison.png")
        
        # Generate summary
        summary_path = f"{output_dir}/backtest_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("===== BACKTEST SUMMARY =====\n\n")
            f.write(f"Strategy Type: {type(self.strategy).__name__}\n")
            f.write(f"Initial Capital: ${self.initial_capital:,.2f}\n")
            f.write(f"Final Portfolio Value: ${results['portfolio_values'][-1]:,.2f}\n")
            f.write(f"Return on Investment: {results['roi']:.2f}%\n")
            f.write(f"Maximum Drawdown: {results['max_drawdown']:.2f}%\n\n")
            
            f.write(f"Number of Trades: {len(self.trades)}\n")
            f.write(f"Buy Trades: {sum(1 for trade in self.trades if trade['type'] == 'buy')}\n")
            f.write(f"Sell Trades: {sum(1 for trade in self.trades if trade['type'] == 'sell')}\n\n")
            
            benchmark_roi = (results['benchmark_values'][-1] / results['benchmark_values'][0] - 1) * 100
            f.write(f"Buy & Hold Return: {benchmark_roi:.2f}%\n")
            f.write(f"Strategy Outperformance: {results['roi'] - benchmark_roi:.2f}%\n")
        
        return [
            f"{output_dir}/price_history.png",
            f"{output_dir}/trading_signals.png",
            f"{output_dir}/performance_comparison.png",
            summary_path
        ]


def generate_sample_historical_data(days=60, volatility=0.02, start_price=50000):
    """Generate sample Bitcoin price data for backtesting."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Start with an actual BTC trend but add randomness
    returns = np.random.normal(0.001, volatility, days)  # Slight upward bias
    price_series = start_price * np.cumprod(1 + returns)
    
    # Create a DataFrame with the sample data
    df = pd.DataFrame({
        'date': timestamps,
        'price': price_series
    })
    
    return df


if __name__ == "__main__":
    # Generate sample data
    print("Generating sample historical data...")
    historical_data = generate_sample_historical_data(days=180)
    
    # Create strategies
    mean_reversion_strategy = MeanReversionStrategy(threshold=0.03, window_size=14)
    momentum_strategy = MomentumStrategy(lookback_period=10, threshold=0.02)
    
    # Test mean reversion strategy
    print("\nBacktesting Mean Reversion strategy...")
    backtester_mr = StrategyBacktester(mean_reversion_strategy)
    results_mr = backtester_mr.run_backtest(historical_data)
    output_files_mr = backtester_mr.generate_report(results_mr, "backtest_results/mean_reversion")
    
    # Test momentum strategy
    print("\nBacktesting Momentum strategy...")
    backtester_mom = StrategyBacktester(momentum_strategy)
    results_mom = backtester_mom.run_backtest(historical_data)
    output_files_mom = backtester_mom.generate_report(results_mom, "backtest_results/momentum")
    
    print("\nBacktesting complete. Results saved to:")
    for f in output_files_mr + output_files_mom:
        print(f"- {f}")
