# examples/exchange_manager_demo.py

import asyncio
import logging
from config.config_manager import ConfigManager
from data_ingestion.exchange_manager import ExchangeManager
from data_ingestion.exchange_connector_base import OrderSide, OrderType

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    # Initialize config manager
    config = ConfigManager()
    
    # Initialize exchange manager with config
    exchange_manager = ExchangeManager(config)
    
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
        
        for exchange_id, ticker in ticker_results.items():
            if "error" in ticker:
                logger.error(f"{exchange_id.upper()} error: {ticker['error']}")
            else:
                logger.info(f"{exchange_id.upper()} BTC/USDT price: ${ticker.get('price', 'N/A')}")
        
        # Get order book data
        logger.info("\nFetching order book data...")
        order_book_results = await exchange_manager.get_order_book_all("BTCUSDT", limit=5)
        
        for exchange_id, order_book in order_book_results.items():
            if "error" in order_book:
                logger.error(f"{exchange_id.upper()} order book error: {order_book['error']}")
            else:
                top_bid = order_book.get("bids", [[0, 0]])[0] if order_book.get("bids") else [0, 0]
                top_ask = order_book.get("asks", [[0, 0]])[0] if order_book.get("asks") else [0, 0]
                logger.info(f"{exchange_id.upper()} Top Bid: ${top_bid[0]} ({top_bid[1]} BTC), Top Ask: ${top_ask[0]} ({top_ask[1]} BTC)")
        
        # Get account information
        logger.info("\nFetching account information...")
        account_results = await exchange_manager.get_account_info_all()
        
        for exchange_id, account_info in account_results.items():
            if "error" in account_info:
                logger.error(f"{exchange_id.upper()} account info error: {account_info['error']}")
            else:
                balances = account_info.get("balances", [])
                if balances:
                    logger.info(f"{exchange_id.upper()} Account Balances:")
                    for balance in balances[:3]:  # Show first 3 balances
                        logger.info(f"  {balance.get('asset')}: Free={balance.get('free')}, Locked={balance.get('locked')}")
                    if len(balances) > 3:
                        logger.info(f"  ... and {len(balances) - 3} more assets")
                else:
                    logger.info(f"{exchange_id.upper()} No balance information available")
        
        # Example: Place a paper trading order
        if config.get_setting("trading", "paper_trading", True):
            logger.info("\nPlacing a paper trading order example...")
            order_result = await exchange_manager.place_order(
                exchange_id=None,  # Use default exchange
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.001  # Small BTC amount
            )
            
            if "error" in order_result:
                logger.error(f"Order placement error: {order_result['error']}")
            else:
                logger.info(f"Order placed successfully: {order_result.get('order_id', 'N/A')}")
                logger.info(f"Order details: {order_result}")
        
        # Example: Update exchange credentials
        # Note: This is just a demonstration - in a real app, you'd get these from user input
        logger.info("\nUpdating exchange credentials example (with dummy values)...")
        update_result = await exchange_manager.update_exchange_credentials(
            exchange_id="binance",
            api_key="dummy_api_key",
            api_secret="dummy_api_secret"
        )
        logger.info(f"Credential update result: {update_result}")
        
        # Example: Set default exchange
        if len(exchange_ids) > 1:
            new_default = exchange_ids[0]
            logger.info(f"\nSetting default exchange to: {new_default}")
            exchange_manager.set_default_exchange(new_default)
            logger.info(f"New default exchange: {exchange_manager.default_exchange}")
    else:
        # If no exchanges are configured, show how to add one
        logger.info("No exchanges configured. Adding a demo exchange...")
        demo_added = exchange_manager.add_exchange("binance", {
            "api_key": "demo_key",
            "api_secret": "demo_secret"
        })
        logger.info(f"Demo exchange added: {demo_added}")
        
        if demo_added:
            # Initialize the newly added exchange
            await exchange_manager.initialize_all()
            
            # Get ticker for BTC/USDT
            logger.info("\nFetching ticker data from demo exchange...")
            ticker = await exchange_manager.get_exchange().get_ticker("BTCUSDT")
            
            if "error" in ticker:
                logger.error(f"Demo exchange error: {ticker['error']}")
            else:
                logger.info(f"Demo exchange BTC/USDT price: ${ticker.get('price', 'N/A')}")
    
    # Close all exchange connections
    await exchange_manager.close_all()

if __name__ == "__main__":
    asyncio.run(main())