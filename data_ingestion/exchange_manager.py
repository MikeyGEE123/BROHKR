# data_ingestion/exchange_manager.py

import asyncio
import logging
from typing import Dict, List, Any, Optional, Type, Union
import json
import os
from datetime import datetime

from data_ingestion.exchange_connector_base import ExchangeConnector, OrderSide, OrderType
from data_ingestion.binance_connector_v2 import BinanceConnector
from data_ingestion.uniswap_connector import UniswapV3Connector

# Set up logging
logger = logging.getLogger(__name__)

class ExchangeManager:
    """
    Manager class for handling multiple exchange connections.
    
    This class provides a unified interface to interact with different exchanges,
    including CEXes (like Binance) and DEXes (like Uniswap).
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the exchange manager with a configuration.
        
        Args:
            config_path (str, optional): Path to a JSON configuration file.
                If provided, exchanges will be loaded from this config.
        """
        self.exchanges: Dict[str, ExchangeConnector] = {}
        self.default_exchange: Optional[str] = None
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> bool:
        """
        Load exchange configuration from a JSON file.
        
        Args:
            config_path (str): Path to a JSON configuration file.
            
        Returns:
            bool: True if loading was successful, False otherwise.
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Process exchanges config
            exchanges_config = config.get('exchanges', [])
            for exchange_config in exchanges_config:
                exchange_type = exchange_config.get('type')
                exchange_id = exchange_config.get('id')
                
                if not exchange_type or not exchange_id:
                    logger.warning(f"Invalid exchange config, missing type or id: {exchange_config}")
                    continue
                
                # Add the exchange based on type
                if exchange_type.lower() == 'binance':
                    self.add_binance_exchange(
                        exchange_id,
                        api_key=exchange_config.get('api_key', ''),
                        api_secret=exchange_config.get('api_secret', ''),
                        testnet=exchange_config.get('testnet', True)
                    )
                elif exchange_type.lower() == 'uniswap':
                    self.add_uniswap_exchange(
                        exchange_id,
                        web3_provider=exchange_config.get('web3_provider', ''),
                        wallet_address=exchange_config.get('wallet_address', ''),
                        private_key=exchange_config.get('private_key', ''),
                        testnet=exchange_config.get('testnet', False)
                    )
                else:
                    logger.warning(f"Unknown exchange type: {exchange_type}")
            
            # Set default exchange if specified
            if 'default_exchange' in config:
                self.default_exchange = config['default_exchange']
                if self.default_exchange not in self.exchanges:
                    logger.warning(f"Default exchange '{self.default_exchange}' not found in config")
            
            return True
        except Exception as e:
            logger.error(f"Error loading exchange config: {e}")
            return False
    
    def save_config(self, config_path: str) -> bool:
        """
        Save current exchange configuration to a JSON file.
        
        Args:
            config_path (str): Path to save the configuration.
            
        Returns:
            bool: True if saving was successful, False otherwise.
        """
        try:
            config = {
                'exchanges': [],
                'default_exchange': self.default_exchange
            }
            
            # Build exchange configs
            for exchange_id, exchange in self.exchanges.items():
                exchange_config = {
                    'id': exchange_id,
                }
                
                # Add exchange-specific details
                if isinstance(exchange, BinanceConnector):
                    exchange_config.update({
                        'type': 'binance',
                        'testnet': exchange.testnet,
                        # Don't save API keys in plain text in real applications!
                        # This is just for demonstration
                        'api_key': exchange.api_key[:5] + '...' if exchange.api_key else '',
                        'api_secret': '...' if exchange.api_secret else ''
                    })
                elif isinstance(exchange, UniswapV3Connector):
                    exchange_config.update({
                        'type': 'uniswap',
                        'testnet': exchange.testnet,
                        'web3_provider': exchange.web3_provider,
                        'wallet_address': exchange.wallet_address,
                        # Don't save private keys!
                        'private_key': '...' if exchange.private_key else ''
                    })
                
                config['exchanges'].append(exchange_config)
            
            # Write to file
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            return True
        except Exception as e:
            logger.error(f"Error saving exchange config: {e}")
            return False
    
    def add_binance_exchange(self, exchange_id: str, api_key: str = "", api_secret: str = "", 
                             testnet: bool = True) -> None:
        """
        Add a Binance exchange connector.
        
        Args:
            exchange_id (str): Unique identifier for this exchange connection.
            api_key (str): Binance API key.
            api_secret (str): Binance API secret.
            testnet (bool): Whether to use testnet.
        """
        self.exchanges[exchange_id] = BinanceConnector(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet
        )
        
        # Set as default if it's the first exchange
        if not self.default_exchange:
            self.default_exchange = exchange_id
    
    def add_uniswap_exchange(self, exchange_id: str, web3_provider: str = "", 
                           wallet_address: str = "", private_key: str = "",
                           testnet: bool = False) -> None:
        """
        Add a Uniswap V3 exchange connector.
        
        Args:
            exchange_id (str): Unique identifier for this exchange connection.
            web3_provider (str): Ethereum node provider URL.
            wallet_address (str): Ethereum wallet address.
            private_key (str): Private key for the wallet (only needed for trading).
            testnet (bool): Whether to use testnet (Goerli).
        """
        self.exchanges[exchange_id] = UniswapV3Connector(
            web3_provider=web3_provider,
            wallet_address=wallet_address,
            private_key=private_key,
            testnet=testnet
        )
        
        # Set as default if it's the first exchange
        if not self.default_exchange:
            self.default_exchange = exchange_id
    
    def remove_exchange(self, exchange_id: str) -> bool:
        """
        Remove an exchange connector.
        
        Args:
            exchange_id (str): Identifier of the exchange to remove.
            
        Returns:
            bool: True if the exchange was removed, False if it doesn't exist.
        """
        if exchange_id in self.exchanges:
            del self.exchanges[exchange_id]
            
            # Update default exchange if needed
            if self.default_exchange == exchange_id:
                self.default_exchange = next(iter(self.exchanges)) if self.exchanges else None
            
            return True
        
        return False
    
    def get_exchange(self, exchange_id: Optional[str] = None) -> Optional[ExchangeConnector]:
        """
        Get an exchange connector by ID or the default exchange.
        
        Args:
            exchange_id (str, optional): Exchange identifier. If not provided, returns the default exchange.
            
        Returns:
            ExchangeConnector: The exchange connector, or None if not found.
        """
        if exchange_id:
            return self.exchanges.get(exchange_id)
        elif self.default_exchange:
            return self.exchanges.get(self.default_exchange)
        
        return None
    
    def set_default_exchange(self, exchange_id: str) -> bool:
        """
        Set the default exchange.
        
        Args:
            exchange_id (str): Identifier of the exchange to set as default.
            
        Returns:
            bool: True if successful, False if the exchange doesn't exist.
        """
        if exchange_id in self.exchanges:
            self.default_exchange = exchange_id
            return True
        
        return False
    
    def list_exchanges(self) -> List[Dict[str, Any]]:
        """
        Get a list of all configured exchanges.
        
        Returns:
            List[Dict[str, Any]]: List of exchange details.
        """
        result = []
        for exchange_id, exchange in self.exchanges.items():
            exchange_info = {
                'id': exchange_id,
                'name': exchange.name,
                'type': type(exchange).__name__,
                'is_default': exchange_id == self.default_exchange
            }
            
            # Add exchange-specific details
            if isinstance(exchange, BinanceConnector):
                exchange_info['testnet'] = exchange.testnet
                exchange_info['has_api_key'] = bool(exchange.api_key)
            elif isinstance(exchange, UniswapV3Connector):
                exchange_info['testnet'] = exchange.testnet
                exchange_info['has_wallet'] = bool(exchange.wallet_address)
            
            result.append(exchange_info)
        
        return result
    
    async def initialize_all(self) -> Dict[str, bool]:
        """
        Initialize all exchange connections.
        
        Returns:
            Dict[str, bool]: Dictionary mapping exchange IDs to initialization success status.
        """
        results = {}
        for exchange_id, exchange in self.exchanges.items():
            try:
                success = await exchange.initialize()
                results[exchange_id] = success
                if not success:
                    logger.warning(f"Failed to initialize exchange {exchange_id}")
            except Exception as e:
                logger.error(f"Error initializing exchange {exchange_id}: {e}")
                results[exchange_id] = False
        
        return results
    
    async def close_all(self) -> None:
        """Close all exchange connections."""
        for exchange_id, exchange in self.exchanges.items():
            try:
                await exchange.close()
            except Exception as e:
                logger.error(f"Error closing exchange {exchange_id}: {e}")
    
    async def get_ticker(self, symbol: str, exchange_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get ticker information for a symbol.
        
        Args:
            symbol (str): The trading pair symbol.
            exchange_id (str, optional): Exchange to use. Uses default if not specified.
            
        Returns:
            Dict[str, Any]: Ticker information or error.
        """
        exchange = self.get_exchange(exchange_id)
        if not exchange:
            return {"error": f"Exchange not found: {exchange_id or 'default'}"}
        
        return await exchange.get_ticker(symbol)
    
    async def get_all_tickers(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """
        Get ticker information for a symbol from all exchanges.
        
        Args:
            symbol (str): The trading pair symbol.
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping exchange IDs to ticker information.
        """
        results = {}
        for exchange_id, exchange in self.exchanges.items():
            try:
                results[exchange_id] = await exchange.get_ticker(symbol)
            except Exception as e:
                logger.error(f"Error getting ticker for {symbol} on {exchange_id}: {e}")
                results[exchange_id] = {"error": str(e)}
        
        return results
    
    async def get_order_book(self, symbol: str, limit: int = 100, 
                          exchange_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get order book for a symbol.
        
        Args:
            symbol (str): The trading pair symbol.
            limit (int): Maximum number of orders to return.
            exchange_id (str, optional): Exchange to use. Uses default if not specified.
            
        Returns:
            Dict[str, Any]: Order book information or error.
        """
        exchange = self.get_exchange(exchange_id)
        if not exchange:
            return {"error": f"Exchange not found: {exchange_id or 'default'}"}
        
        return await exchange.get_order_book(symbol, limit)
    
    async def get_historical_klines(self, symbol: str, interval: str, 
                                 start_time: Optional[int] = None, 
                                 end_time: Optional[int] = None, 
                                 limit: int = 500,
                                 exchange_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get historical klines/candlesticks for a symbol.
        
        Args:
            symbol (str): The trading pair symbol.
            interval (str): Kline interval (e.g., '1m', '1h', '1d').
            start_time (int, optional): Start time in milliseconds.
            end_time (int, optional): End time in milliseconds.
            limit (int): Maximum number of klines to return.
            exchange_id (str, optional): Exchange to use. Uses default if not specified.
            
        Returns:
            List[Dict[str, Any]]: List of klines or error.
        """
        exchange = self.get_exchange(exchange_id)
        if not exchange:
            return {"error": f"Exchange not found: {exchange_id or 'default'}"}
        
        return await exchange.get_historical_klines(
            symbol, interval, start_time, end_time, limit
        )
    
    async def place_order(self, symbol: str, side: Union[OrderSide, str], 
                       order_type: Union[OrderType, str], 
                       quantity: float, price: Optional[float] = None,
                       exchange_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Place an order on an exchange.
        
        Args:
            symbol (str): The trading pair symbol.
            side (OrderSide or str): Whether to buy or sell.
            order_type (OrderType or str): Type of order (market, limit, etc.).
            quantity (float): Quantity to buy/sell.
            price (float, optional): Price for limit orders.
            exchange_id (str, optional): Exchange to use. Uses default if not specified.
            
        Returns:
            Dict[str, Any]: Order information or error.
        """
        exchange = self.get_exchange(exchange_id)
        if not exchange:
            return {"error": f"Exchange not found: {exchange_id or 'default'}"}
        
        # Convert string enums to enum values if needed
        if isinstance(side, str):
            side = OrderSide[side.upper()]
        if isinstance(order_type, str):
            order_type = OrderType[order_type.upper()]
        
        return await exchange.place_order(symbol, side, order_type, quantity, price)
    
    async def get_account_info(self, exchange_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get account information from an exchange.
        
        Args:
            exchange_id (str, optional): Exchange to use. Uses default if not specified.
            
        Returns:
            Dict[str, Any]: Account information or error.
        """
        exchange = self.get_exchange(exchange_id)
        if not exchange:
            return {"error": f"Exchange not found: {exchange_id or 'default'}"}
        
        return await exchange.get_account_info()
    
    async def get_exchange_info(self, exchange_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get exchange information.
        
        Args:
            exchange_id (str, optional): Exchange to use. Uses default if not specified.
            
        Returns:
            Dict[str, Any]: Exchange information or error.
        """
        exchange = self.get_exchange(exchange_id)
        if not exchange:
            return {"error": f"Exchange not found: {exchange_id or 'default'}"}
        
        return await exchange.get_exchange_info()
    
    async def get_arbitrage_opportunities(self, symbol: str, min_profit_percent: float = 1.0) -> List[Dict[str, Any]]:
        """
        Find arbitrage opportunities for a symbol across exchanges.
        
        Args:
            symbol (str): The trading pair symbol.
            min_profit_percent (float): Minimum profit percentage to consider an opportunity.
            
        Returns:
            List[Dict[str, Any]]: List of arbitrage opportunities.
        """
        # Get ticker data from all exchanges
        all_tickers = await self.get_all_tickers(symbol)
        
        # Find arbitrage opportunities
        opportunities = []
        exchanges = list(all_tickers.keys())
        
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                exchange1 = exchanges[i]
                exchange2 = exchanges[j]
                
                ticker1 = all_tickers[exchange1]
                ticker2 = all_tickers[exchange2]
                
                # Skip if any ticker has an error
                if "error" in ticker1 or "error" in ticker2:
                    continue
                
                price1 = ticker1.get("price")
                price2 = ticker2.get("price")
                
                if price1 and price2:
                    # Calculate profit percentage in both directions
                    profit1to2 = (price2 / price1 - 1) * 100
                    profit2to1 = (price1 / price2 - 1) * 100
                    
                    # Check for opportunities
                    if profit1to2 >= min_profit_percent:
                        opportunities.append({
                            "buy_exchange": exchange1,
                            "sell_exchange": exchange2,
                            "symbol": symbol,
                            "buy_price": price1,
                            "sell_price": price2,
                            "profit_percent": profit1to2,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    if profit2to1 >= min_profit_percent:
                        opportunities.append({
                            "buy_exchange": exchange2,
                            "sell_exchange": exchange1,
                            "symbol": symbol,
                            "buy_price": price2,
                            "sell_price": price1,
                            "profit_percent": profit2to1,
                            "timestamp": datetime.now().isoformat()
                        })
        
        # Sort by profit percentage
        opportunities.sort(key=lambda x: x["profit_percent"], reverse=True)
        
        return opportunities

# Example usage
async def example_usage():
    # Create exchange manager
    manager = ExchangeManager()
    
    # Add exchanges
    manager.add_binance_exchange("binance", testnet=True)
    manager.add_uniswap_exchange(
        "uniswap", 
        web3_provider="https://mainnet.infura.io/v3/your-infura-key"
    )
    
    # Initialize exchanges
    init_results = await manager.initialize_all()
    print(f"Initialization results: {init_results}")
    
    # Get ticker from default exchange
    ticker = await manager.get_ticker("BTC/USDT")
    print(f"BTC/USDT price: ${ticker.get('price', 'N/A')}")
    
    # Find arbitrage opportunities
    opportunities = await manager.get_arbitrage_opportunities("ETH/USDT", min_profit_percent=0.5)
    for opp in opportunities:
        print(f"Arbitrage opportunity: Buy on {opp['buy_exchange']} at ${opp['buy_price']:.2f}, "
              f"sell on {opp['sell_exchange']} at ${opp['sell_price']:.2f} "
              f"for {opp['profit_percent']:.2f}% profit")
    
    # Close all exchange connections
    await manager.close_all()

if __name__ == "__main__":
    asyncio.run(example_usage())
