# examples/visualization_demo.py

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from config.config_manager import ConfigManager
from data_ingestion.exchange_manager import ExchangeManager
from data_ingestion.exchange_connector_base import OrderSide, OrderType

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketVisualizer:
    """Class for visualizing market data"""
    
    def __init__(self, theme="dark_background"):
        """Initialize the visualizer with a theme"""
        self.theme = theme
        plt.style.use(theme)
    
    def plot_price_chart(self, data, title="Price Chart"):
        """Plot price chart from OHLCV data"""
        if data is None or len(data) == 0:
            logger.error("No data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot candlestick chart
        width = 0.6
        width2 = width * 0.8
        
        up = data[data['close'] >= data['open']]
        down = data[data['close'] < data['open']]
        
        # Plot up candles
        ax1.bar(up.index, up['high'] - up['low'], width=width2, bottom=up['low'], color='green', alpha=0.5)
        ax1.bar(up.index, up['close'] - up['open'], width=width, bottom=up['open'], color='green')
        
        # Plot down candles
        ax1.bar(down.index, down['high'] - down['low'], width=width2, bottom=down['low'], color='red', alpha=0.5)
        ax1.bar(down.index, down['open'] - down['close'], width=width, bottom=down['close'], color='red')
        
        # Plot volume
        ax2.bar(data.index, data['volume'], color='blue', alpha=0.5)
        
        # Add moving averages
        if len(data) > 20:
            ma20 = data['close'].rolling(window=20).mean()
            ax1.plot(data.index, ma20, color='blue', label='MA20')
        
        if len(data) > 50:
            ma50 = data['close'].rolling(window=50).mean()
            ax1.plot(data.index, ma50, color='orange', label='MA50')
        
        # Set labels and title
        ax1.set_title(title)
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_order_book(self, bids, asks, title="Order Book"):
        """Plot order book visualization"""
        if not bids or not asks:
            logger.error("No order book data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert to numpy arrays for easier manipulation
        bid_prices = np.array([float(bid[0]) for bid in bids])
        bid_quantities = np.array([float(bid[1]) for bid in bids])
        ask_prices = np.array([float(ask[0]) for ask in asks])
        ask_quantities = np.array([float(ask[1]) for ask in asks])
        
        # Calculate cumulative quantities
        bid_cum = np.cumsum(bid_quantities)
        ask_cum = np.cumsum(ask_quantities)
        
        # Plot bids (green)
        ax.step(bid_prices, bid_cum, where='post', color='green', label='Bids')
        ax.fill_between(bid_prices, 0, bid_cum, step='post', alpha=0.3, color='green')
        
        # Plot asks (red)
        ax.step(ask_prices, ask_cum, where='post', color='red', label='Asks')
        ax.fill_between(ask_prices, 0, ask_cum, step='post', alpha=0.3, color='red')
        
        # Calculate mid price
        mid_price = (bid_prices[0] + ask_prices[0]) / 2
        ax.axvline(x=mid_price, color='yellow', linestyle='--', label=f'Mid Price: {mid_price:.2f}')
        
        # Set labels and title
        ax.set_title(title)
        ax.set_xlabel('Price')
        ax.set_ylabel('Cumulative Quantity')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_exchange_comparison(self, data_dict, title="Exchange Price Comparison"):
        """Plot price comparison between exchanges"""
        if not data_dict:
            logger.error("No comparison data to plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        for exchange_id, price in data_dict.items():
            if isinstance(price, dict) and "price" in price:
                plt.scatter(exchange_id, float(price["price"]), s=100)
                plt.text(exchange_id, float(price["price"]), f"${float(price['price']):.2f}", 
                         ha='center', va='bottom')
        
        plt.title(title)
        plt.ylabel('Price (USD)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

async def fetch_historical_data(exchange, symbol, timeframe='1h', limit=100):
    """Fetch historical OHLCV data from an exchange"""
    try:
        data = await exchange.get_historical_data(symbol, timeframe, limit)
        
        if isinstance(data, dict) and "error" in data:
            logger.error(f"Error fetching historical data: {data['error']}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    except Exception as e:
        logger.error(f"Error in fetch_historical_data: {e}")
        return None

async def main():
    # Initialize config manager
    config = ConfigManager()
    
    # Initialize exchange manager with config
    exchange_manager = ExchangeManager(config)
    
    # Initialize visualizer
    visualizer = MarketVisualizer(theme="dark_background")
    
    # Initialize all exchanges
    init_results = await exchange_manager.initialize_all()
    logger.info(f"Exchange initialization results: {init_results}")
    
    # Get available exchanges
    exchange_ids = exchange_manager.get_exchange_ids()
    logger.info(f"Available exchanges: {exchange_ids}")
    
    if exchange_ids:
        # Get default exchange
        default_exchange = exchange_manager.get_exchange()
        logger.info(f"Default exchange: {default_exchange.name if default_exchange else 'None'}")
        
        # Get ticker for BTC/USDT from all exchanges
        ticker_results = await exchange_manager.get_ticker_all("BTCUSDT")
        
        # Visualize price comparison between exchanges
        logger.info("\nVisualizing price comparison between exchanges...")
        visualizer.plot_exchange_comparison(ticker_results, title="BTC/USDT Price Comparison")
        
        # Fetch historical data from default exchange
        if default_exchange:
            logger.info(f"\nFetching historical data from {default_exchange.name}...")
            historical_data = await fetch_historical_data(default_exchange, "BTCUSDT", timeframe="1h", limit=100)
            
            if historical_data is not None:
                logger.info("Visualizing price chart...")
                visualizer.plot_price_chart(historical_data, title=f"BTC/USDT Price Chart - {default_exchange.name}")
        
        # Get order book data from default exchange
        if default_exchange:
            logger.info(f"\nFetching order book from {default_exchange.name}...")
            order_book = await default_exchange.get_order_book("BTCUSDT")
            
            if "error" not in order_book:
                logger.info("Visualizing order book...")
                visualizer.plot_order_book(
                    order_book.get("bids", [])[:20],  # Top 20 bids
                    order_book.get("asks", [])[:20],  # Top 20 asks
                    title=f"BTC/USDT Order Book - {default_exchange.name}"
                )
    else:
        logger.warning("No exchanges available. Please configure at least one exchange.")
    
    # Close all exchange connections
    await exchange_manager.close_all()

if __name__ == "__main__":
    asyncio.run(main())