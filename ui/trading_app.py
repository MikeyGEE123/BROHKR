# ui/trading_app.py

import tkinter as tk
from tkinter import ttk, messagebox
import asyncio
import threading
import logging
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_manager import ConfigManager
from data_ingestion.exchange_manager import ExchangeManager
from data_ingestion.exchange_connector_base import OrderSide, OrderType

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AsyncTkApp:
    """Base class for handling async operations in Tkinter"""
    
    def __init__(self, loop=None):
        self.loop = loop or asyncio.new_event_loop()
        self.async_thread = None
    
    def start_async_thread(self):
        """Start the async event loop in a separate thread"""
        if self.async_thread is not None and self.async_thread.is_alive():
            return
            
        def run_async_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
            
        self.async_thread = threading.Thread(target=run_async_loop, daemon=True)
        self.async_thread.start()
    
    def stop_async_thread(self):
        """Stop the async event loop"""
        if self.async_thread is None:
            return
            
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.async_thread.join(timeout=1.0)
        self.async_thread = None
    
    def run_async(self, coro):
        """Run a coroutine in the async event loop and return a future"""
        if self.async_thread is None:
            self.start_async_thread()
            
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

class TradingApp(AsyncTkApp):
    """Main trading application UI"""
    
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.root.title("BROHKR Trading Platform")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Initialize components
        self.config = ConfigManager()
        self.exchange_manager = None
        
        # Create UI
        self.create_ui()
        
        # Start async thread
        self.start_async_thread()
        
        # Initialize exchange manager
        self.run_async(self.initialize_exchange_manager())
    
    async def initialize_exchange_manager(self):
        """Initialize the exchange manager"""
        try:
            self.exchange_manager = ExchangeManager(self.config)
            init_results = await self.exchange_manager.initialize_all()
            
            # Update exchange selector
            exchange_ids = self.exchange_manager.get_exchange_ids()
            self.exchange_var.set("")
            self.exchange_selector['values'] = exchange_ids
            
            if exchange_ids:
                self.exchange_var.set(exchange_ids[0])
                self.status_var.set(f"Connected to {len(exchange_ids)} exchanges")
                
                # Update market data
                await self.update_market_data()
            else:
                self.status_var.set("No exchanges configured")
                
            # Log initialization results
            for exchange_id, success in init_results.items():
                status = "Success" if success else "Failed"
                self.log(f"Exchange {exchange_id} initialization: {status}")
                
        except Exception as e:
            self.log(f"Error initializing exchange manager: {e}")
            self.status_var.set(f"Error: {str(e)}")
    
    def create_ui(self):
        """Create the main UI components"""
        # Create notebook with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.market_tab = ttk.Frame(self.notebook)
        self.trading_tab = ttk.Frame(self.notebook)
        self.account_tab = ttk.Frame(self.notebook)
        self.log_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.market_tab, text="Market Data")
        self.notebook.add(self.trading_tab, text="Trading")
        self.notebook.add(self.account_tab, text="Account")
        self.notebook.add(self.log_tab, text="Logs")
        
        # Setup market tab
        self.setup_market_tab()
        
        # Setup trading tab
        self.setup_trading_tab()
        
        # Setup account tab
        self.setup_account_tab()
        
        # Setup log tab
        self.setup_log_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Initializing...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_market_tab(self):
        """Setup the market data tab"""
        # Control frame
        control_frame = ttk.Frame(self.market_tab)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Exchange selector
        ttk.Label(control_frame, text="Exchange:").pack(side=tk.LEFT, padx=5)
        self.exchange_var = tk.StringVar()
        self.exchange_selector = ttk.Combobox(control_frame, textvariable=self.exchange_var, state="readonly", width=15)
        self.exchange_selector.pack(side=tk.LEFT, padx=5)
        self.exchange_selector.bind("<<ComboboxSelected>>", lambda e: self.run_async(self.update_market_data()))
        
        # Symbol selector
        ttk.Label(control_frame, text="Symbol:").pack(side=tk.LEFT, padx=5)
        self.symbol_var = tk.StringVar(value="BTCUSDT")
        ttk.Entry(control_frame, textvariable=self.symbol_var, width=15).pack(side=tk.LEFT, padx=5)
        
        # Timeframe selector
        ttk.Label(control_frame, text="Timeframe:").pack(side=tk.LEFT, padx=5)
        self.timeframe_var = tk.StringVar(value="1h")
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        ttk.Combobox(control_frame, textvariable=self.timeframe_var, values=timeframes, state="readonly", width=10).pack(side=tk.LEFT, padx=5)
        
        # Refresh button
        ttk.Button(control_frame, text="Refresh", command=lambda: self.run_async(self.update_market_data())).pack(side=tk.LEFT, padx=10)
        
        # Chart frame
        chart_frame = ttk.LabelFrame(self.market_tab, text="Price Chart")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure and canvas
        self.fig = plt.figure(figsize=(10, 6), facecolor='#1e1e1e')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#2d2d2d')
        self.ax.tick_params(colors='#00ff9d')
        self.ax.spines['bottom'].set_color('#00ff9d')
        self.ax.spines['top'].set_color('#00ff9d')
        self.ax.spines['left'].set_color('#00ff9d')
        self.ax.spines['right'].set_color('#00ff9d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Order book frame
        order_book_frame = ttk.LabelFrame(self.market_tab, text="Order Book")
        order_book_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create bid and ask frames
        bid_frame = ttk.Frame(order_book_frame)
        bid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ask_frame = ttk.Frame(order_book_frame)
        ask_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Bid table
        ttk.Label(bid_frame, text="Bids", anchor=tk.CENTER).pack(fill=tk.X)
        self.bid_tree = ttk.Treeview(bid_frame, columns=("price", "quantity", "total"), show="headings", height=5)
        self.bid_tree.heading("price", text="Price")
        self.bid_tree.heading("quantity", text="Quantity")
        self.bid_tree.heading("total", text="Total")
        self.bid_tree.column("price", width=100, anchor=tk.E)
        self.bid_tree.column("quantity", width=100, anchor=tk.E)
        self.bid_tree.column("total", width=100, anchor=tk.E)
        self.bid_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Ask table
        ttk.Label(ask_frame, text="Asks", anchor=tk.CENTER).pack(fill=tk.X)
        self.ask_tree = ttk.Treeview(ask_frame, columns=("price", "quantity", "total"), show="headings", height=5)
        self.ask_tree.heading("price", text="Price")
        self.ask_tree.heading("quantity", text="Quantity")
        self.ask_tree.heading("total", text="Total")
        self.ask_tree.column("price", width=100, anchor=tk.E)
        self.ask_tree.column("quantity", width=100, anchor=tk.E)
        self.ask_tree.column("total", width=100, anchor=tk.E)
        self.ask_tree.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    def setup_trading_tab(self):
        """Setup the trading tab"""
        # Split into left and right frames
        left_frame = ttk.Frame(self.trading_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        right_frame = ttk.Frame(self.trading_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Order form
        order_frame = ttk.LabelFrame(left_frame, text="Place Order")
        order_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Order form fields
        form_frame = ttk.Frame(order_frame)
        form_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Symbol
        ttk.Label(form_frame, text="Symbol:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.trade_symbol_var = tk.StringVar(value="BTCUSDT")
        ttk.Entry(form_frame, textvariable=self.trade_symbol_var, width=15).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Side
        ttk.Label(form_frame, text="Side:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.side_var = tk.StringVar(value="BUY")
        ttk.Radiobutton(form_frame, text="Buy", variable=self.side_var, value="BUY").grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(form_frame, text="Sell", variable=self.side_var, value="SELL").grid(row=1, column=2, sticky=tk.W, pady=5)
        
        # Order type
        ttk.Label(form_frame, text="Order Type:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.order_type_var = tk.StringVar(value="MARKET")
        ttk.Radiobutton(form_frame, text="Market", variable=self.order_type_var, value="MARKET").grid(row=2, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(form_frame, text="Limit", variable=self.order_type_var, value="LIMIT").grid(row=2, column=2, sticky=tk.W, pady=5)
        
        # Quantity
        ttk.Label(form_frame, text="Quantity:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.quantity_var = tk.StringVar(value="0.001")
        ttk.Entry(form_frame, textvariable=self.quantity_var, width=15).grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Price (for limit orders)
        ttk.Label(form_frame, text="Price:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.price_var = tk.StringVar()
        self.price_entry = ttk.Entry(form_frame, textvariable=self.price_var, width=15)
        self.price_entry.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        # Place order button
        ttk.Button(form_frame, text="Place Order", command=self.place_order).grid(row=5, column=0, columnspan=3, pady=10)
        
        # Order history
        history_frame = ttk.LabelFrame(right_frame, text="Order History")
        history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Order history table
        self.order_tree = ttk.Treeview(history_frame, columns=("time", "symbol", "side", "type", "price", "quantity", "status"), show="headings", height=10)
        self.order_tree.heading("time", text="Time")
        self.order_tree.heading("symbol", text="Symbol")
        self.order_tree.heading("side", text="Side")
        self.order_tree.heading("type", text="Type")
        self.order_tree.heading("price", text="Price")
        self.order_tree.heading("quantity", text="Quantity")
        self.order_tree.heading("status", text="Status")
        
        self.order_tree.column("time", width=150)
        self.order_tree.column("symbol", width=80)
        self.order_tree.column("side", width=60)
        self.order_tree.column("type", width=80)
        self.order_tree.column("price", width=100)
        self.order_tree.column("quantity", width=100)
        self.order_tree.column("status", width=80)
        
        self.order_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar for order history
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.order_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.order_tree.configure(yscrollcommand=scrollbar.set)
        
        # Refresh button
        ttk.Button(history_frame, text="Refresh Orders", command=lambda: self.run_async(self.refresh_orders())).pack(pady=5)
    
    def setup_account_tab(self):
        """Setup the account tab"""
        # Account info frame
        info_frame = ttk.LabelFrame(self.account_tab, text="Account Information")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Account info fields
        account_frame = ttk.Frame(info_frame)
        account_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Exchange selector
        ttk.Label(account_frame, text="Exchange:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.account_exchange_var = tk.StringVar()
        self.account_exchange_selector = ttk.Combobox(account_frame, textvariable=self.account_exchange_var, state="readonly", width=15)
        self.account_exchange_selector.grid(row=0, column=1, sticky=tk.W, pady=5)
        self.account_exchange_selector.bind("<<ComboboxSelected>>", lambda e: self.run_async(self.update_account_info()))
        
        # Account type
        ttk.Label(account_frame, text="Account Type:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.account_type_var = tk.StringVar()
        ttk.Label(account_frame, textvariable=self.account_type_var).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Total balance
        ttk.Label(account_frame, text="Total Balance (USDT):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.total_balance_var = tk.StringVar()
        ttk.Label(account_frame, textvariable=self.total_balance_var).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Refresh button
        ttk.Button(account_frame, text="Refresh", command=lambda: self.run_async(self.update_account_info())).grid(row=3, column=0, columnspan=2, pady=10)
        
        # Balances frame
        balances_frame = ttk.LabelFrame(self.account_tab, text="Asset Balances")
        balances_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Balances table
        self.balance_tree = ttk.Treeview(balances_frame, columns=("asset", "free", "locked", "total"), show="headings", height=15)
        self.balance_tree.heading("asset", text="Asset")
        self.balance_tree.heading("free", text="Available")
        self.balance_tree.heading("locked", text="In Order")
        self.balance_tree.heading("total", text="Total")
        
        self.balance_tree.column("asset", width=100)
        self.balance_tree.column("free", width=150)
        self.balance_tree.column("locked", width=150)
        self.balance_tree.column("total", width=150)
        
        self.balance_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar for balances
        scrollbar = ttk.Scrollbar(balances_frame, orient=tk.VERTICAL, command=self.balance_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.balance_tree.configure(yscrollcommand=scrollbar.set)
    
    def setup_log_tab(self):
        """Setup the log tab"""
        # Log frame
        log_frame = ttk.Frame(self.log_tab)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log text widget
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=20, bg='#1e1e1e', fg='#00ff9d')
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar for log
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        # Clear button
        ttk.Button(self.log_tab, text="Clear Logs", command=self.clear_logs).pack(pady=5)
    
    async def update_market_data(self):
        """Update market data based on selected exchange and symbol"""
        try:
            exchange_id = self.exchange_var.get()
            symbol = self.symbol_var.get()
            timeframe = self.timeframe_var.get()
            
            if not exchange_id or not symbol:
                return
                
            self.status_var.set(f"Fetching data for {symbol} from {exchange_id}...")
            
            # Get exchange
            exchange = self.exchange_manager.get_exchange(exchange_id)
            
            if not exchange:
                self.log(f"Exchange {exchange_id} not found")
                return
                
            # Get ticker
            ticker = await exchange.get_ticker(symbol)
            
            if "error" in ticker:
                self.log(f"Error getting ticker: {ticker['error']}")
                return
                
            # Get order book
            order_book = await exchange.get_order_book(symbol)
            
            if "error" in order_book:
                self.log(f"Error getting order book: {order_book['error']}")
                return
                
            # Get historical data
            ohlcv = await exchange.get_ohlcv(symbol, timeframe)
            
            if "error" in ohlcv:
                self.log(f"Error getting OHLCV data: {ohlcv['error']}")
                return
                
            # Update UI with data
            self.update_chart(ohlcv)
            self.update_order_book(order_book)
            
            self.status_var.set(f"{symbol} last price: {ticker.get('last', 'N/A')}")
            self.log(f"Updated market data for {symbol} on {exchange_id}")
            
        except Exception as e:
            self.log(f"Error updating market data: {e}")
            self.status_var.set(f"Error: {str(e)}")