# analysis/visualization.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go


class TradingVisualizer:
    """Utility class for visualizing trading data and strategy performance."""
    
    @staticmethod
    def plot_price_history(timestamps, prices, title="Price History", figsize=(12, 6)):
        """
        Plot historical price data with matplotlib.
        
        Args:
            timestamps (list): List of timestamps (datetime objects or strings)
            prices (list): List of price values
            title (str): Plot title
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        plt.figure(figsize=figsize)
        plt.plot(timestamps, prices, 'b-')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True)
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_strategy_signals(timestamps, prices, signals, title="Trading Signals", figsize=(12, 6)):
        """
        Plot price history with buy/sell signals.
        
        Args:
            timestamps (list): List of timestamps
            prices (list): List of price values
            signals (list): List of signals ('buy', 'sell', or 'hold')
            title (str): Plot title
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        buy_times = [timestamps[i] for i, signal in enumerate(signals) if signal == 'buy']
        buy_prices = [prices[i] for i, signal in enumerate(signals) if signal == 'buy']
        
        sell_times = [timestamps[i] for i, signal in enumerate(signals) if signal == 'sell']
        sell_prices = [prices[i] for i, signal in enumerate(signals) if signal == 'sell']
        
        plt.figure(figsize=figsize)
        plt.plot(timestamps, prices, 'b-', label='Price')
        plt.scatter(buy_times, buy_prices, color='green', marker='^', s=100, label='Buy Signal')
        plt.scatter(sell_times, sell_prices, color='red', marker='v', s=100, label='Sell Signal')
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_strategy_performance(timestamps, portfolio_values, benchmark_values=None, 
                                  title="Strategy Performance", figsize=(12, 6)):
        """
        Plot strategy performance over time with optional benchmark comparison.
        
        Args:
            timestamps (list): List of timestamps
            portfolio_values (list): List of portfolio values over time
            benchmark_values (list, optional): List of benchmark values (e.g., "buy and hold" strategy)
            title (str): Plot title
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        plt.figure(figsize=figsize)
        plt.plot(timestamps, portfolio_values, 'b-', label='Strategy')
        
        if benchmark_values:
            plt.plot(timestamps, benchmark_values, 'r--', label='Benchmark')
            
        # Calculate and display performance metrics
        if len(portfolio_values) > 1:
            returns = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
            plt.annotate(f'Return: {returns:.2f}%', 
                         xy=(0.02, 0.95), 
                         xycoords='axes fraction',
                         fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def create_candlestick_chart(df, title="Price Chart"):
        """
        Create an interactive candlestick chart using Plotly.
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLC data. Must contain columns:
                - 'timestamp' or 'date'
                - 'open', 'high', 'low', 'close'
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Interactive Plotly figure
        """
        time_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        
        fig = go.Figure(data=[go.Candlestick(
            x=df[time_col],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def generate_sample_data(days=30, volatility=0.02, start_price=100):
        """
        Generate sample price data for testing visualizations.
        
        Args:
            days (int): Number of days to generate
            volatility (float): Price volatility factor
            start_price (float): Starting price
            
        Returns:
            tuple: (timestamps, prices, ohlc_df)
        """
        # Generate timestamps
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        timestamps = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate random price movements
        returns = np.random.normal(0, volatility, days)
        price_series = start_price * (1 + np.cumsum(returns))
        
        # Create OHLC data
        ohlc_data = []
        for i, timestamp in enumerate(timestamps):
            base_price = price_series[i]
            daily_volatility = base_price * volatility * 0.5
            
            ohlc_data.append({
                'date': timestamp,
                'open': base_price,
                'high': base_price + abs(np.random.normal(0, daily_volatility)),
                'low': base_price - abs(np.random.normal(0, daily_volatility)),
                'close': base_price + np.random.normal(0, daily_volatility)
            })
        
        ohlc_df = pd.DataFrame(ohlc_data)
        return timestamps, price_series, ohlc_df


# Example usage
if __name__ == "__main__":
    # Generate sample data
    timestamps, prices, ohlc_df = TradingVisualizer.generate_sample_data(days=60)
    
    # Generate random trading signals (just for demonstration)
    signals = ['hold'] * len(prices)
    for i in range(5, len(signals)):
        if prices[i] > prices[i-5] * 1.05:
            signals[i] = 'sell'
        elif prices[i] < prices[i-5] * 0.95:
            signals[i] = 'buy'
    
    # Create and show visualizations
    TradingVisualizer.plot_price_history(timestamps, prices, "Sample Price History")
    plt.savefig("price_history.png")
    
    TradingVisualizer.plot_strategy_signals(timestamps, prices, signals, "Sample Trading Signals")
    plt.savefig("trading_signals.png")
    
    # Simulate portfolio growth
    portfolio = [100]  # Start with $100
    for signal in signals[1:]:
        if signal == 'buy':
            change = np.random.uniform(0.001, 0.02)
        elif signal == 'sell':
            change = np.random.uniform(-0.02, -0.001)
        else:
            change = np.random.uniform(-0.005, 0.005)
        
        portfolio.append(portfolio[-1] * (1 + change))
    
    TradingVisualizer.plot_strategy_performance(timestamps, portfolio, title="Sample Portfolio Performance")
    plt.savefig("portfolio_performance.png")
    
    # Candlestick chart using Plotly
    fig = TradingVisualizer.create_candlestick_chart(ohlc_df, "Sample Candlestick Chart")
    fig.write_html("candlestick.html")
    
    print("Visualization examples saved to files.")
