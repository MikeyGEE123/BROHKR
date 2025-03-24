# examples/web_visualizer.py

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from flask import Flask, render_template, jsonify, request
import threading
from config.config_manager import ConfigManager
from data_ingestion.exchange_manager import ExchangeManager
from data_ingestion.exchange_connector_base import OrderSide, OrderType

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            static_folder='../web/static',
            template_folder='../web/templates')

# Global variables
config = ConfigManager()
exchange_manager = None
loop = asyncio.new_event_loop()

class WebMarketVisualizer:
    """Class for creating web-based market visualizations"""
    
    def create_price_chart(self, data, title="Price Chart"):
        """Create price chart from OHLCV data using Plotly"""
        if data is None or len(data) == 0:
            logger.error("No data to plot")
            return None
        
        # Create subplot with 2 rows
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, 
                           row_heights=[0.7, 0.3],
                           subplot_titles=(title, "Volume"))
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="Price"
        ), row=1, col=1)
        
        # Add volume bar chart
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['volume'],
            name="Volume",
            marker_color='blue',
            opacity=0.5
        ), row=2, col=1)
        
        # Add moving averages
        if len(data) > 20:
            ma20 = data['close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=ma20,
                name="MA20",
                line=dict(color='blue', width=1)
            ), row=1, col=1)
        
        if len(data) > 50:
            ma50 = data['close'].rolling(window=50).mean()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=ma50,
                name="MA50",
                line=dict(color='orange', width=1)
            ), row=1, col=1)
        
        # Update layout
        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=800,
            width=1000,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        return fig
    
    def create_order_book_chart(self, bids, asks, title="Order Book"):
        """Create order book visualization using Plotly"""
        if not bids or not asks:
            logger.error("No order book data to plot")
            return None
        
        # Convert to numpy arrays for easier manipulation
        bid_prices = np.array([float(bid[0]) for bid in bids])
        bid_quantities = np.array([float(bid[1]) for bid in bids])
        ask_prices = np.array([float(ask[0]) for ask in asks])
        ask_quantities = np.array([float(ask[1]) for ask in asks])
        
        # Calculate cumulative quantities
        bid_cum = np.cumsum(bid_quantities)
        ask_cum = np.cumsum(ask_quantities)
        
        # Create figure
        fig = go.Figure()
        
        # Add bid line
        fig.add_trace(go.Scatter(
            x=bid_prices,
            y=bid_cum,
            fill='tozeroy',
            mode='lines',
            line=dict(color='green', width=1),
            fillcolor='rgba(0, 255, 0, 0.3)',
            name='Bids'
        ))
        
        # Add ask line
        fig.add_trace(go.Scatter(
            x=ask_prices,
            y=ask_cum,
            fill='tozeroy',
            mode='lines',
            line=dict(color='red', width=1),
            fillcolor='rgba(255, 0, 0, 0.3)',
            name='Asks'
        ))
        
        # Calculate mid price
        mid_price = (bid_prices[0] + ask_prices[0]) / 2
        
        # Add mid price line
        fig.add_trace(go.Scatter(
            x=[mid_price, mid_price],
            y=[0, max(max(bid_cum), max(ask_cum))],
            mode='lines',
            line=dict(color='yellow', width=1, dash='dash'),
            name=f'Mid Price: {mid_price:.2f}'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Price',
            yaxis_title='Cumulative Quantity',
            template="plotly_dark",
            height=600,
            width=1000
        )
        
        return fig
    
    def create_exchange_comparison_chart(self, data_dict, title="Exchange Price Comparison"):
        """Create price comparison between exchanges using Plotly"""
        if not data_dict:
            logger.error("No comparison data to plot")
            return None
        
        # Extract exchange names and prices
        exchanges = []
        prices = []
        
        for exchange_id, price_data in data_dict.items():
            if isinstance(price_data, dict) and "price" in price_data:
                exchanges.append(exchange_id)
                prices.append(float(price_data["price"]))
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Bar(
            x=exchanges,
            y=prices,
            marker_color='lightblue',
            text=[f"${price:.2f}" for price in prices],
            textposition='outside'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            yaxis_title='Price (USD)',
            template="plotly_dark",
            height=500,
            width=1000
        )
        
        return fig

async def fetch_historical_data(exchange, symbol, timeframe='1h', limit=100):
    """Fetch historical OHLCV data from an exchange"""
    try:
        data = await exchange.get_ohlcv(symbol, timeframe, limit)
        
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

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/exchanges')
def get_exchanges():
    """Get available exchanges"""
    if not exchange_manager:
        return jsonify({"error": "Exchange manager not initialized"})
    
    exchange_ids = exchange_manager.get_exchange_ids()
    return jsonify({"exchanges": exchange_ids})

@app.route('/api/price_chart')
def get_price_chart():
    """Get price chart data"""
    exchange_id = request.args.get('exchange', '')
    symbol = request.args.get('symbol', 'BTCUSDT')
    timeframe = request.args.get('timeframe', '1h')
    limit = int(request.args.get('limit', 100))
    
    if not exchange_manager:
        return jsonify({"error": "Exchange manager not initialized"})
    
    exchange = exchange_manager.get_exchange(exchange_id)
    if not exchange:
        return jsonify({"error": f"Exchange {exchange_id} not found"})
    
    # Run async function in the event loop
    future = asyncio.run_coroutine_threadsafe(
        fetch_historical_data(exchange, symbol, timeframe, limit), 
        loop
    )
    data = future.result()
    
    if data is None:
        return jsonify({"error": "Failed to fetch historical data"})
    
    # Create chart
    visualizer = WebMarketVisualizer()
    fig = visualizer.create_price_chart(data, title=f"{symbol} Price Chart - {exchange.name}")
    
    return jsonify({"chart": fig.to_json()})

@app.route('/api/order_book')
def get_order_book():
    """Get order book visualization"""
    exchange_id = request.args.get('exchange', '')
    symbol = request.args.get('symbol', 'BTCUSDT')
    
    if not exchange_manager:
        return jsonify({"error": "Exchange manager not initialized"})
    
    exchange = exchange_manager.get_exchange(exchange_id)
    if not exchange:
        return jsonify({"error": f"Exchange {exchange_id} not found"})
    
    # Run async function in the event loop
    future = asyncio.run_coroutine_threadsafe(
        exchange.get_order_book(symbol), 
        loop
    )
    order_book = future.result()
    
    if "error" in order_book:
        return jsonify({"error": f"Failed to fetch order book: {order_book['error']}"})
    
    # Create chart
    visualizer = WebMarketVisualizer()
    fig = visualizer.create_order_book_chart(
        order_book.get("bids", [])[:20],  # Top 20 bids
        order_book.get("asks", [])[:20],  # Top 20 asks
        title=f"{symbol} Order Book - {exchange.name}"
    )
    
    return jsonify({"chart": fig.to_json()})

@app.route('/api/exchange_comparison')
def get_exchange_comparison():
    """Get exchange price comparison"""
    symbol = request.args.get('symbol', 'BTCUSDT')
    
    if not exchange_manager:
        return jsonify({"error": "Exchange manager not initialized"})
    
    # Run async function in the event loop
    future = asyncio.run_coroutine_threadsafe(
        exchange_manager.get_ticker_all(symbol), 
        loop
    )
    ticker_results = future.result()
    
    # Create chart
    visualizer = WebMarketVisualizer()
    fig = visualizer.create_exchange_comparison_chart(
        ticker_results, 
        title=f"{symbol} Price Comparison"
    )
    
    return jsonify({"chart": fig.to_json()})

def initialize_exchange_manager():
    """Initialize the exchange manager"""
    global exchange_manager
    
    if exchange_manager is None:
        exchange_manager = ExchangeManager(config)
        init_future = asyncio.run_coroutine_threadsafe(exchange_manager.initialize_all(), loop)
        init_future.result()

def start_async_loop():
    """Start the async event loop in a separate thread"""
    asyncio.set_event_loop(loop)
    loop.run_forever()

if __name__ == "__main__":
    # Start async loop in a separate thread
    async_thread = threading.Thread(target=start_async_loop, daemon=True)
    async_thread.start()
    
    # Initialize exchange manager
    initialize_exchange_manager()
    
    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=8080)