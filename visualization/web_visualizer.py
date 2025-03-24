# At the bottom of your file, modify the run method:

def start_async_loop():
    """Start the asyncio event loop in a separate thread"""
    asyncio.set_event_loop(loop)
    loop.run_forever()

def initialize_exchange_manager():
    """Initialize the exchange manager"""
    global exchange_manager
    exchange_manager = ExchangeManager(config)
    # Initialize exchanges
    exchange_manager.initialize_exchanges()

if __name__ == "__main__":
    # Start async loop in a separate thread
    async_thread = threading.Thread(target=start_async_loop, daemon=True)
    async_thread.start()
    
    # Initialize exchange manager
    initialize_exchange_manager()
    
    # For development only
    app.run(host='0.0.0.0', port=8080, debug=True)