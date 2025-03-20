# data_ingestion/uniswap_connector.py

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp
from web3 import Web3

from data_ingestion.exchange_connector_base import ExchangeConnector, OrderType, OrderSide

# Set up logging
logger = logging.getLogger(__name__)

# Uniswap V3 Factory and Router addresses (Ethereum Mainnet)
UNISWAP_V3_FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
UNISWAP_V3_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
UNISWAP_V3_QUOTER = "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6"

# ABI snippets for key interactions
FACTORY_ABI = json.loads('''[
    {"inputs":[],"name":"owner","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"address","name":"tokenA","type":"address"},{"internalType":"address","name":"tokenB","type":"address"},{"internalType":"uint24","name":"fee","type":"uint24"}],"name":"getPool","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"}
]''')

POOL_ABI = json.loads('''[
    {"inputs":[],"name":"slot0","outputs":[{"internalType":"uint160","name":"sqrtPriceX96","type":"uint160"},{"internalType":"int24","name":"tick","type":"int24"},{"internalType":"uint16","name":"observationIndex","type":"uint16"},{"internalType":"uint16","name":"observationCardinality","type":"uint16"},{"internalType":"uint16","name":"observationCardinalityNext","type":"uint16"},{"internalType":"uint8","name":"feeProtocol","type":"uint8"},{"internalType":"bool","name":"unlocked","type":"bool"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"uint256","name":"","type":"uint256"}],"name":"observations","outputs":[{"internalType":"uint32","name":"blockTimestamp","type":"uint32"},{"internalType":"int56","name":"tickCumulative","type":"int56"},{"internalType":"uint160","name":"secondsPerLiquidityCumulativeX128","type":"uint160"},{"internalType":"bool","name":"initialized","type":"bool"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"liquidity","outputs":[{"internalType":"uint128","name":"","type":"uint128"}],"stateMutability":"view","type":"function"}
]''')

ERC20_ABI = json.loads('''[
    {"inputs":[],"name":"decimals","outputs":[{"internalType":"uint8","name":"","type":"uint8"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"symbol","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"address","name":"account","type":"address"}],"name":"balanceOf","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]''')

# Common token addresses for reference
TOKEN_ADDRESSES = {
    "ETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
    "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F"
}

class UniswapV3Connector(ExchangeConnector):
    """
    Implementation of ExchangeConnector for Uniswap V3 DEX.
    
    This class provides methods to interact with Uniswap V3 for data retrieval and trading.
    """
    
    def __init__(self, 
                web3_provider: str = "https://mainnet.infura.io/v3/your-infura-key", 
                wallet_address: str = "", 
                private_key: str = "",
                testnet: bool = False):
        """
        Initialize the Uniswap V3 connector.
        
        Args:
            web3_provider (str): The Ethereum node provider URL.
            wallet_address (str): The Ethereum wallet address for trading.
            private_key (str): The private key for the wallet (only needed for trading).
            testnet (bool): Whether to use testnet (Goerli).
        """
        super().__init__(wallet_address, private_key, testnet)
        self.name = "UniswapV3"
        self.web3_provider = web3_provider
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.testnet = testnet
        self.web3 = None
        self.factory_contract = None
        self.session = None
        self.rate_limits = {"requests_per_minute": 100, "last_request_time": 0, "request_count": 0}
        
        # Mapping of token symbols to addresses (for easy reference)
        self.token_addresses = TOKEN_ADDRESSES.copy()
        
        # Testnet (Goerli) configuration
        if testnet:
            # Update with testnet addresses (Goerli)
            self.factory_address = "0x1F98431c8aD98523631AE4a59f267346ea31F984"  # Same as mainnet
            self.router_address = "0xE592427A0AEce92De3Edee1F18E0157C05861564"  # Same as mainnet
            # Testnet token addresses would be different
            # TODO: Update with Goerli testnet token addresses
        else:
            # Mainnet configuration
            self.factory_address = UNISWAP_V3_FACTORY
            self.router_address = UNISWAP_V3_ROUTER
    
    async def initialize(self) -> bool:
        """
        Initialize the connection to Uniswap V3.
        
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            # Initialize Web3
            self.web3 = Web3(Web3.HTTPProvider(self.web3_provider))
            if not self.web3.is_connected():
                logger.error("Failed to connect to Ethereum node")
                return False
            
            # Initialize contracts
            self.factory_contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(self.factory_address), 
                abi=FACTORY_ABI
            )
            
            # Initialize HTTP session for API calls
            self.session = aiohttp.ClientSession()
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Uniswap connector: {e}")
            return False
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get the current price for a trading pair on Uniswap.
        
        Args:
            symbol (str): The trading pair symbol, e.g., 'ETH/USDC'.
            
        Returns:
            Dict[str, Any]: Dictionary containing price information.
        """
        await self._handle_rate_limits()
        try:
            # Parse symbol to get token addresses
            base_token, quote_token = self._parse_trading_pair(symbol)
            
            # Default to 0.3% fee tier
            fee_tier = 3000  # 0.3%
            
            # Get pool address
            pool_address = self.factory_contract.functions.getPool(
                self.web3.to_checksum_address(base_token),
                self.web3.to_checksum_address(quote_token),
                fee_tier
            ).call()
            
            if pool_address == "0x0000000000000000000000000000000000000000":
                # Try other fee tiers if pool not found
                for fee in [500, 10000]:  # 0.05%, 1%
                    pool_address = self.factory_contract.functions.getPool(
                        self.web3.to_checksum_address(base_token),
                        self.web3.to_checksum_address(quote_token),
                        fee
                    ).call()
                    if pool_address != "0x0000000000000000000000000000000000000000":
                        fee_tier = fee
                        break
            
            if pool_address == "0x0000000000000000000000000000000000000000":
                return {
                    "error": f"Pool not found for {symbol} with any fee tier"
                }
            
            # Get pool contract
            pool_contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(pool_address),
                abi=POOL_ABI
            )
            
            # Get current price from slot0
            slot0 = pool_contract.functions.slot0().call()
            sqrt_price_x96 = slot0[0]
            
            # Convert sqrtPriceX96 to price
            price = (sqrt_price_x96 ** 2) / (2 ** 192)
            
            # Get decimals for both tokens
            base_token_contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(base_token),
                abi=ERC20_ABI
            )
            quote_token_contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(quote_token),
                abi=ERC20_ABI
            )
            
            base_decimals = base_token_contract.functions.decimals().call()
            quote_decimals = quote_token_contract.functions.decimals().call()
            
            # Adjust price based on token decimals
            price = price * (10 ** (quote_decimals - base_decimals))
            
            # Get pool liquidity
            liquidity = pool_contract.functions.liquidity().call()
            
            return self._standardize_response({
                "symbol": symbol,
                "price": price,
                "pool_address": pool_address,
                "fee_tier": fee_tier,
                "liquidity": liquidity,
                "sqrt_price_x96": sqrt_price_x96
            }, "ticker")
        except Exception as e:
            logger.error(f"Uniswap error getting ticker for {symbol}: {e}")
            return {"error": str(e)}
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get a simulated order book for Uniswap V3.
        
        Note: Uniswap doesn't have a traditional order book, so this is approximated
        by simulating positions at different price points based on pool liquidity distribution.
        
        Args:
            symbol (str): The trading pair symbol.
            limit (int): Maximum number of price levels to return.
            
        Returns:
            Dict[str, Any]: Dictionary containing order book simulation.
        """
        await self._handle_rate_limits()
        try:
            # Parse symbol to get token addresses
            base_token, quote_token = self._parse_trading_pair(symbol)
            
            # Get current price
            ticker_data = await self.get_ticker(symbol)
            if "error" in ticker_data:
                return ticker_data
            
            current_price = ticker_data["price"]
            pool_address = ticker_data["pool_address"]
            
            # Get pool contract
            pool_contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(pool_address),
                abi=POOL_ABI
            )
            
            # Get current tick from slot0
            slot0 = pool_contract.functions.slot0().call()
            current_tick = slot0[1]
            
            # Simulate order book by distributing liquidity around the current price
            spread = 0.001  # 0.1% spread
            bid_price = current_price * (1 - spread)
            ask_price = current_price * (1 + spread)
            
            # Calculate liquidity distribution (simplified)
            total_liquidity = pool_contract.functions.liquidity().call()
            
            # Generate simulated bids and asks
            bids = []
            asks = []
            
            # Distribute liquidity for bids
            for i in range(limit // 2):
                price = bid_price * (1 - i * 0.001)
                amount = total_liquidity / (limit // 2) / price
                bids.append([price, amount])
            
            # Distribute liquidity for asks
            for i in range(limit // 2):
                price = ask_price * (1 + i * 0.001)
                amount = total_liquidity / (limit // 2) / price
                asks.append([price, amount])
            
            # Sort in order book order (highest bid and lowest ask first)
            bids.sort(reverse=True)
            asks.sort()
            
            return self._standardize_response({
                "symbol": symbol,
                "pool_address": pool_address,
                "bids": bids,
                "asks": asks,
                "current_tick": current_tick
            }, "order_book")
        except Exception as e:
            logger.error(f"Uniswap error getting order book for {symbol}: {e}")
            return {"error": str(e)}
    
    async def get_historical_klines(self, symbol: str, interval: str, 
                                   start_time: Optional[int] = None, 
                                   end_time: Optional[int] = None, 
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical price data using The Graph API for Uniswap.
        
        Args:
            symbol (str): The trading pair symbol.
            interval (str): Kline interval (e.g., '1h', '1d').
            start_time (int, optional): Start time in milliseconds.
            end_time (int, optional): End time in milliseconds.
            limit (int): Maximum number of data points to return.
            
        Returns:
            List[Dict[str, Any]]: List of historical price points.
        """
        await self._handle_rate_limits()
        try:
            # Parse symbol to get token addresses
            base_token, quote_token = self._parse_trading_pair(symbol)
            
            # Convert interval to seconds
            interval_seconds = self._interval_to_seconds(interval)
            
            # Set default time range if not provided
            if end_time is None:
                end_time = int(time.time() * 1000)
            if start_time is None:
                start_time = end_time - (interval_seconds * 1000 * limit)
            
            # Convert milliseconds to seconds for Graph API
            start_timestamp = start_time // 1000
            end_timestamp = end_time // 1000
            
            # Use The Graph API to fetch historical data
            query = '''
            {
              poolHourDatas(
                where: {
                  pool: "%s",
                  periodStartUnix_gte: %d,
                  periodStartUnix_lte: %d
                }
                orderBy: periodStartUnix
                orderDirection: asc
                first: %d
              ) {
                periodStartUnix
                token0Price
                token1Price
                tvlUSD
                volumeUSD
                txCount
              }
            }
            ''' % (
                await self._get_pool_address(base_token, quote_token),
                start_timestamp,
                end_timestamp,
                limit
            )
            
            # Make request to The Graph API for Uniswap V3
            async with self.session.post(
                'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
                json={"query": query}
            ) as response:
                data = await response.json()
                
                if "errors" in data:
                    logger.error(f"The Graph API error: {data['errors']}")
                    return {"error": str(data['errors'])}
                
                pool_data = data.get('data', {}).get('poolHourDatas', [])
                
                # Convert to standard kline format
                formatted_data = []
                for entry in pool_data:
                    # Use token0Price or token1Price based on token order in the pool
                    price = float(entry['token0Price'])
                    formatted_kline = {
                        "open_time": int(entry['periodStartUnix']) * 1000,
                        "close_time": (int(entry['periodStartUnix']) + interval_seconds) * 1000,
                        "open": price,
                        "high": price * 1.005,  # Simulated high
                        "low": price * 0.995,  # Simulated low
                        "close": price,
                        "volume": float(entry['volumeUSD']),
                        "quote_asset_volume": float(entry['volumeUSD']),
                        "number_of_trades": int(entry['txCount']),
                        "tvl": float(entry['tvlUSD'])
                    }
                    formatted_data.append(formatted_kline)
                
                return self._standardize_response(formatted_data, "klines")
        except Exception as e:
            logger.error(f"Uniswap error getting historical data for {symbol}: {e}")
            return {"error": str(e)}
    
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                         quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place a swap on Uniswap V3.
        
        Note: In DEXes, there are no traditional limit orders. This function implements swaps.
        
        Args:
            symbol (str): The trading pair symbol.
            side (OrderSide): Whether to buy or sell.
            order_type (OrderType): Only MARKET is supported for Uniswap.
            quantity (float): Amount to swap.
            price (float, optional): Not used for market orders.
            
        Returns:
            Dict[str, Any]: Dictionary containing transaction information.
        """
        if not self.wallet_address or not self.private_key:
            return {"error": "Wallet address and private key are required for trading"}
        
        if order_type != OrderType.MARKET:
            return {"error": "Only market orders (swaps) are supported on Uniswap"}
        
        await self._handle_rate_limits()
        try:
            # Parse symbol to get token addresses
            base_token, quote_token = self._parse_trading_pair(symbol)
            
            # Determine input and output tokens based on side
            if side == OrderSide.BUY:
                token_in = quote_token
                token_out = base_token
                # For buys, quantity is in the base token (what we want to receive)
                # We need to estimate how much quote token to send
                amount_out = int(quantity * 10 ** 18)  # Assuming 18 decimals
                # TODO: Implement exact calculation based on token decimals
            else:  # SELL
                token_in = base_token
                token_out = quote_token
                # For sells, quantity is in the base token (what we want to sell)
                amount_in = int(quantity * 10 ** 18)  # Assuming 18 decimals
                # TODO: Implement exact calculation based on token decimals
            
            # Construct swap transaction (simplified)
            # Note: In a real implementation, you would use the Uniswap Router contract
            # to perform the swap with proper slippage protection
            
            # This is a placeholder for demonstration purposes
            # In a real implementation, you would:
            # 1. Approve the router to spend your tokens
            # 2. Estimate gas
            # 3. Calculate minimum amount out with slippage
            # 4. Call exactInputSingle or exactOutputSingle on the router
            
            return {
                "error": "Real trading functionality requires a full Ethereum transaction implementation.",
                "note": "This would involve approving tokens, calculating slippage, and executing a swap transaction.",
                "details": {
                    "operation": "swap",
                    "token_in": token_in,
                    "token_out": token_out,
                    "amount": quantity,
                    "side": side.value
                }
            }
        except Exception as e:
            logger.error(f"Uniswap error placing order: {e}")
            return {"error": str(e)}
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information, including token balances.
        
        Returns:
            Dict[str, Any]: Dictionary containing account information.
        """
        if not self.wallet_address:
            return {"error": "Wallet address is required for account information"}
        
        await self._handle_rate_limits()
        try:
            balances = []
            
            # Get ETH balance
            eth_balance = self.web3.eth.get_balance(self.wallet_address)
            balances.append({
                "asset": "ETH",
                "free": self.web3.from_wei(eth_balance, 'ether'),
                "locked": 0,  # No concept of locked funds in Ethereum
                "address": "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"  # Common representation for native ETH
            })
            
            # Get balances for common tokens
            for symbol, address in self.token_addresses.items():
                try:
                    token_contract = self.web3.eth.contract(
                        address=self.web3.to_checksum_address(address),
                        abi=ERC20_ABI
                    )
                    
                    token_balance = token_contract.functions.balanceOf(
                        self.web3.to_checksum_address(self.wallet_address)
                    ).call()
                    
                    decimals = token_contract.functions.decimals().call()
                    
                    if token_balance > 0:
                        balances.append({
                            "asset": symbol,
                            "free": token_balance / (10 ** decimals),
                            "locked": 0,
                            "address": address
                        })
                except Exception as e:
                    logger.warning(f"Failed to get balance for {symbol}: {e}")
            
            account_info = {
                "address": self.wallet_address,
                "balances": balances,
                "network": "Goerli" if self.testnet else "Ethereum Mainnet"
            }
            
            return self._standardize_response(account_info, "account")
        except Exception as e:
            logger.error(f"Uniswap error getting account info: {e}")
            return {"error": str(e)}
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get information about Uniswap V3, including supported tokens and pools.
        
        Returns:
            Dict[str, Any]: Dictionary containing exchange information.
        """
        await self._handle_rate_limits()
        try:
            # Query The Graph for top Uniswap pools
            query = '''
            {
              pools(
                orderBy: totalValueLockedUSD
                orderDirection: desc
                first: 20
              ) {
                id
                token0 {
                  id
                  symbol
                  name
                  decimals
                }
                token1 {
                  id
                  symbol
                  name
                  decimals
                }
                feeTier
                liquidity
                totalValueLockedUSD
                volumeUSD
              }
            }
            '''
            
            async with self.session.post(
                'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
                json={"query": query}
            ) as response:
                data = await response.json()
                
                if "errors" in data:
                    logger.error(f"The Graph API error: {data['errors']}")
                    return {"error": str(data['errors'])}
                
                pools = data.get('data', {}).get('pools', [])
                
                # Format the pool data
                trading_pairs = []
                for pool in pools:
                    token0 = pool['token0']
                    token1 = pool['token1']
                    
                    pair_info = {
                        "symbol": f"{token0['symbol']}/{token1['symbol']}",
                        "pool_address": pool['id'],
                        "fee_tier": int(pool['feeTier']),
                        "base_asset": token0['symbol'],
                        "quote_asset": token1['symbol'],
                        "base_asset_address": token0['id'],
                        "quote_asset_address": token1['id'],
                        "liquidity": pool['liquidity'],
                        "tvl_usd": float(pool['totalValueLockedUSD']),
                        "volume_usd": float(pool['volumeUSD']),
                        "base_decimals": int(token0['decimals']),
                        "quote_decimals": int(token1['decimals'])
                    }
                    
                    trading_pairs.append(pair_info)
                
                exchange_info = {
                    "name": "Uniswap V3",
                    "network": "Goerli" if self.testnet else "Ethereum Mainnet",
                    "factory_address": self.factory_address,
                    "router_address": self.router_address,
                    "trading_pairs": trading_pairs
                }
                
                return self._standardize_response(exchange_info, "exchange_info")
        except Exception as e:
            logger.error(f"Uniswap error getting exchange info: {e}")
            return {"error": str(e)}
    
    async def close(self) -> None:
        """Close the connection to Uniswap."""
        if self.session:
            await self.session.close()
    
    async def _get_pool_address(self, token0: str, token1: str, fee_tier: int = 3000) -> str:
        """
        Get the pool address for a pair of tokens.
        
        Args:
            token0 (str): First token address.
            token1 (str): Second token address.
            fee_tier (int): Fee tier in basis points (default: 3000 = 0.3%).
            
        Returns:
            str: Pool address.
        """
        # Ensure token0 < token1 (as required by Uniswap)
        if token0.lower() > token1.lower():
            token0, token1 = token1, token0
        
        pool_address = self.factory_contract.functions.getPool(
            self.web3.to_checksum_address(token0),
            self.web3.to_checksum_address(token1),
            fee_tier
        ).call()
        
        if pool_address == "0x0000000000000000000000000000000000000000":
            # Try other fee tiers if pool not found
            for fee in [500, 10000]:  # 0.05%, 1%
                pool_address = self.factory_contract.functions.getPool(
                    self.web3.to_checksum_address(token0),
                    self.web3.to_checksum_address(token1),
                    fee
                ).call()
                if pool_address != "0x0000000000000000000000000000000000000000":
                    break
        
        if pool_address == "0x0000000000000000000000000000000000000000":
            raise ValueError(f"Pool not found for {token0}/{token1} with any fee tier")
        
        return pool_address
    
    def _parse_trading_pair(self, symbol: str) -> tuple:
        """
        Parse a trading pair symbol into token addresses.
        
        Args:
            symbol (str): The trading pair symbol, e.g., 'ETH/USDC'.
            
        Returns:
            tuple: (base_token_address, quote_token_address)
        """
        parts = symbol.split('/')
        if len(parts) != 2:
            raise ValueError(f"Invalid trading pair format: {symbol}. Expected format: BASE/QUOTE")
        
        base_symbol = parts[0].upper()
        quote_symbol = parts[1].upper()
        
        base_address = self.token_addresses.get(base_symbol)
        quote_address = self.token_addresses.get(quote_symbol)
        
        if not base_address:
            raise ValueError(f"Unknown base token: {base_symbol}")
        if not quote_address:
            raise ValueError(f"Unknown quote token: {quote_symbol}")
        
        return base_address, quote_address
    
    def _interval_to_seconds(self, interval: str) -> int:
        """
        Convert an interval string to seconds.
        
        Args:
            interval (str): Interval string (e.g., '1h', '1d').
            
        Returns:
            int: Interval in seconds.
        """
        unit = interval[-1]
        value = int(interval[:-1])
        
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 60 * 60
        elif unit == 'd':
            return value * 60 * 60 * 24
        elif unit == 'w':
            return value * 60 * 60 * 24 * 7
        else:
            raise ValueError(f"Unknown interval unit: {unit}")
    
    def _standardize_response(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """
        Standardize the response data from Uniswap to have a consistent format.
        
        Args:
            data (Dict[str, Any]): The raw response data.
            data_type (str): The type of data.
            
        Returns:
            Dict[str, Any]: Standardized data.
        """
        standardized = {
            "exchange": self.name,
            "timestamp": int(time.time() * 1000),
            "data_type": data_type,
        }
        
        # Add type-specific data
        if isinstance(data, dict):
            standardized.update(data)
        else:
            standardized["data"] = data
        
        return standardized

# Example usage
async def example_usage():
    connector = UniswapV3Connector(
        web3_provider="https://mainnet.infura.io/v3/your-infura-key"
    )
    
    if await connector.initialize():
        # Get price of ETH/USDC
        ticker = await connector.get_ticker("ETH/USDC")
        if "error" not in ticker:
            print(f"ETH/USDC price: ${ticker['price']}")
        
        # Get account info if wallet address is provided
        if connector.wallet_address:
            account = await connector.get_account_info()
            if "error" not in account:
                print(f"Account balances: {account['balances']}")
        
        # Close connection
        await connector.close()

if __name__ == "__main__":
    asyncio.run(example_usage())
