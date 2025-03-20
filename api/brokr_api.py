# api/brokr_api.py

import asyncio
import logging
import json
import os
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import BROKR components
from data_ingestion.exchange_manager import ExchangeManager
from security.token_security import TokenSecurity
from trading_strategy.strategies.mean_reversion import MeanReversionStrategy
from trading_strategy.strategies.momentum import MomentumStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("brokr_api")

# Initialize FastAPI app
app = FastAPI(
    title="BROKR API",
    description="API for the BROKR cryptocurrency trading bot framework",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Pydantic Models -----

class TokenCheck(BaseModel):
    token_address: str
    chain_id: int = 1  # Default to Ethereum mainnet

class OrderRequest(BaseModel):
    symbol: str
    side: str  # "BUY" or "SELL"
    order_type: str = "MARKET"  # Default to market order
    quantity: float
    price: Optional[float] = None  # Required for limit orders
    exchange_id: Optional[str] = None  # Use default exchange if not specified

class ExchangeConfig(BaseModel):
    type: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    web3_provider: Optional[str] = None
    wallet_address: Optional[str] = None
    private_key: Optional[str] = None
    testnet: bool = True

class SecuritySettings(BaseModel):
    min_sol_sniffer_score: int = Field(80, ge=0, le=100)
    check_liquidity_lock: bool = True
    check_bundled_supply: bool = True

# ----- Global State -----

# In a real application, you'd use a proper database
# For simplicity, we'll use in-memory state
exchange_manager = None
token_security = None
running_tasks = {}

# ----- Dependency Functions -----

async def get_exchange_manager():
    """Get the exchange manager singleton."""
    global exchange_manager
    if exchange_manager is None:
        exchange_manager = ExchangeManager()
        
        # Load config if exists
        config_path = os.path.join(os.path.dirname(__file__), "../config/exchanges.json")
        if os.path.exists(config_path):
            exchange_manager.load_config(config_path)
        
        # Initialize exchanges
        await exchange_manager.initialize_all()
    
    return exchange_manager

async def get_token_security():
    """Get the token security singleton."""
    global token_security
    if token_security is None:
        token_security = TokenSecurity()
        
        # Load config if exists
        config_path = os.path.join(os.path.dirname(__file__), "../config/token_security.json")
        if os.path.exists(config_path):
            token_security.load_config(config_path)
    
    return token_security

# ----- Background Task Functions -----

async def run_scan_tokens(token_addresses: List[str], chain_id: int = 1, task_id: str = None):
    """Background task to scan multiple tokens."""
    try:
        token_security = await get_token_security()
        results = await token_security.scan_tokens(token_addresses, chain_id)
        running_tasks[task_id] = {
            "status": "completed",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in token scan task: {e}")
        running_tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ----- API Routes -----

@app.get("/")
async def root():
    """Root endpoint showing API info."""
    return {
        "name": "BROKR API",
        "version": "0.1.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

# --- Exchange Management Routes ---

@app.get("/exchanges", tags=["Exchanges"])
async def list_exchanges(manager: ExchangeManager = Depends(get_exchange_manager)):
    """List all configured exchanges."""
    return manager.list_exchanges()

@app.post("/exchanges/{exchange_id}", tags=["Exchanges"])
async def add_exchange(
    exchange_id: str, 
    config: ExchangeConfig,
    manager: ExchangeManager = Depends(get_exchange_manager)
):
    """Add a new exchange."""
    try:
        if config.type.lower() == "binance":
            manager.add_binance_exchange(
                exchange_id,
                api_key=config.api_key or "",
                api_secret=config.api_secret or "",
                testnet=config.testnet
            )
        elif config.type.lower() == "uniswap":
            manager.add_uniswap_exchange(
                exchange_id,
                web3_provider=config.web3_provider or "",
                wallet_address=config.wallet_address or "",
                private_key=config.private_key or "",
                testnet=config.testnet
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported exchange type: {config.type}")
        
        # Initialize the new exchange
        exchange = manager.get_exchange(exchange_id)
        if exchange:
            success = await exchange.initialize()
            if not success:
                return JSONResponse(
                    status_code=207,  # Multi-Status
                    content={
                        "message": f"Exchange {exchange_id} added but initialization failed",
                        "exchange_id": exchange_id,
                        "initialized": False
                    }
                )
        
        # Save config
        config_path = os.path.join(os.path.dirname(__file__), "../config/exchanges.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        manager.save_config(config_path)
        
        return {
            "message": f"Exchange {exchange_id} added successfully",
            "exchange_id": exchange_id,
            "initialized": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding exchange: {str(e)}")

@app.delete("/exchanges/{exchange_id}", tags=["Exchanges"])
async def remove_exchange(
    exchange_id: str,
    manager: ExchangeManager = Depends(get_exchange_manager)
):
    """Remove an exchange."""
    try:
        # Close the exchange connection first
        exchange = manager.get_exchange(exchange_id)
        if exchange:
            await exchange.close()
        
        # Remove the exchange
        success = manager.remove_exchange(exchange_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Exchange {exchange_id} not found")
        
        # Save updated config
        config_path = os.path.join(os.path.dirname(__file__), "../config/exchanges.json")
        manager.save_config(config_path)
        
        return {"message": f"Exchange {exchange_id} removed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing exchange: {str(e)}")

@app.get("/exchanges/{exchange_id}/info", tags=["Exchanges"])
async def get_exchange_info(
    exchange_id: str,
    manager: ExchangeManager = Depends(get_exchange_manager)
):
    """Get information about an exchange."""
    try:
        result = await manager.get_exchange_info(exchange_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting exchange info: {str(e)}")

# --- Market Data Routes ---

@app.get("/market/ticker/{symbol}", tags=["Market Data"])
async def get_ticker(
    symbol: str,
    exchange_id: Optional[str] = None,
    manager: ExchangeManager = Depends(get_exchange_manager)
):
    """Get ticker information for a symbol."""
    try:
        result = await manager.get_ticker(symbol, exchange_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting ticker: {str(e)}")

@app.get("/market/orderbook/{symbol}", tags=["Market Data"])
async def get_order_book(
    symbol: str,
    limit: int = Query(100, ge=1, le=5000),
    exchange_id: Optional[str] = None,
    manager: ExchangeManager = Depends(get_exchange_manager)
):
    """Get order book for a symbol."""
    try:
        result = await manager.get_order_book(symbol, limit, exchange_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting order book: {str(e)}")

@app.get("/market/klines/{symbol}", tags=["Market Data"])
async def get_klines(
    symbol: str,
    interval: str = "1h",
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = Query(500, ge=1, le=1000),
    exchange_id: Optional[str] = None,
    manager: ExchangeManager = Depends(get_exchange_manager)
):
    """Get historical klines/candlesticks for a symbol."""
    try:
        result = await manager.get_historical_klines(
            symbol, interval, start_time, end_time, limit, exchange_id
        )
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting klines: {str(e)}")

@app.get("/market/arbitrage/{symbol}", tags=["Market Data"])
async def get_arbitrage_opportunities(
    symbol: str,
    min_profit: float = Query(1.0, ge=0.0),
    manager: ExchangeManager = Depends(get_exchange_manager)
):
    """Find arbitrage opportunities for a symbol across exchanges."""
    try:
        opportunities = await manager.get_arbitrage_opportunities(symbol, min_profit)
        return {
            "symbol": symbol,
            "min_profit_percent": min_profit,
            "opportunities": opportunities,
            "count": len(opportunities)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding arbitrage opportunities: {str(e)}")

# --- Trading Routes ---

@app.post("/trading/order", tags=["Trading"])
async def place_order(
    order: OrderRequest,
    manager: ExchangeManager = Depends(get_exchange_manager)
):
    """Place an order on an exchange."""
    try:
        result = await manager.place_order(
            order.symbol,
            order.side,
            order.order_type,
            order.quantity,
            order.price,
            order.exchange_id
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error placing order: {str(e)}")

@app.get("/trading/account", tags=["Trading"])
async def get_account_info(
    exchange_id: Optional[str] = None,
    manager: ExchangeManager = Depends(get_exchange_manager)
):
    """Get account information from an exchange."""
    try:
        result = await manager.get_account_info(exchange_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting account info: {str(e)}")

# --- Security Routes ---

@app.post("/security/token/check", tags=["Security"])
async def check_token_security(
    token: TokenCheck,
    security: TokenSecurity = Depends(get_token_security)
):
    """Check a token's security."""
    try:
        result = await security.check_token_security(token.token_address, token.chain_id)
        is_safe, reasons = security.is_token_safe(result)
        
        return {
            "token_address": token.token_address,
            "chain_id": token.chain_id,
            "is_safe": is_safe,
            "reasons": reasons,
            "details": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking token security: {str(e)}")

@app.post("/security/token/scan", tags=["Security"])
async def scan_tokens(
    token_addresses: List[str],
    chain_id: int = 1,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    security: TokenSecurity = Depends(get_token_security)
):
    """Scan multiple tokens for security issues (async task)."""
    try:
        task_id = f"scan_{int(time.time())}"
        
        # Start the scan in the background
        running_tasks[task_id] = {
            "status": "running",
            "token_addresses": token_addresses,
            "chain_id": chain_id,
            "timestamp": datetime.now().isoformat()
        }
        
        background_tasks.add_task(
            run_scan_tokens, token_addresses, chain_id, task_id
        )
        
        return {
            "task_id": task_id,
            "status": "running",
            "token_count": len(token_addresses),
            "message": "Token scan started in the background"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting token scan: {str(e)}")

@app.get("/security/token/scan/{task_id}", tags=["Security"])
async def get_scan_result(task_id: str):
    """Get the result of a token scan task."""
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return running_tasks[task_id]

@app.post("/security/token/blacklist/{token_address}", tags=["Security"])
async def add_to_token_blacklist(
    token_address: str,
    security: TokenSecurity = Depends(get_token_security)
):
    """Add a token to the blacklist."""
    try:
        added = security.add_to_token_blacklist(token_address)
        
        # Save updated config
        config_path = os.path.join(os.path.dirname(__file__), "../config/token_security.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        security.save_config(config_path)
        
        return {
            "token_address": token_address,
            "added": added,
            "message": f"Token {'added to' if added else 'already in'} blacklist"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding token to blacklist: {str(e)}")

@app.delete("/security/token/blacklist/{token_address}", tags=["Security"])
async def remove_from_token_blacklist(
    token_address: str,
    security: TokenSecurity = Depends(get_token_security)
):
    """Remove a token from the blacklist."""
    try:
        removed = security.remove_from_token_blacklist(token_address)
        
        # Save updated config
        config_path = os.path.join(os.path.dirname(__file__), "../config/token_security.json")
        security.save_config(config_path)
        
        return {
            "token_address": token_address,
            "removed": removed,
            "message": f"Token {'removed from' if removed else 'not found in'} blacklist"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing token from blacklist: {str(e)}")

@app.get("/security/token/blacklist", tags=["Security"])
async def get_token_blacklist(
    security: TokenSecurity = Depends(get_token_security)
):
    """Get the list of blacklisted tokens."""
    try:
        return {
            "token_blacklist": security.get_token_blacklist(),
            "count": len(security.token_blacklist)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting token blacklist: {str(e)}")

@app.post("/security/developer/blacklist/{developer_address}", tags=["Security"])
async def add_to_dev_blacklist(
    developer_address: str,
    security: TokenSecurity = Depends(get_token_security)
):
    """Add a developer to the blacklist."""
    try:
        added = security.add_to_dev_blacklist(developer_address)
        
        # Save updated config
        config_path = os.path.join(os.path.dirname(__file__), "../config/token_security.json")
        security.save_config(config_path)
        
        return {
            "developer_address": developer_address,
            "added": added,
            "message": f"Developer {'added to' if added else 'already in'} blacklist"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding developer to blacklist: {str(e)}")

@app.delete("/security/developer/blacklist/{developer_address}", tags=["Security"])
async def remove_from_dev_blacklist(
    developer_address: str,
    security: TokenSecurity = Depends(get_token_security)
):
    """Remove a developer from the blacklist."""
    try:
        removed = security.remove_from_dev_blacklist(developer_address)
        
        # Save updated config
        config_path = os.path.join(os.path.dirname(__file__), "../config/token_security.json")
        security.save_config(config_path)
        
        return {
            "developer_address": developer_address,
            "removed": removed,
            "message": f"Developer {'removed from' if removed else 'not found in'} blacklist"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing developer from blacklist: {str(e)}")

@app.get("/security/developer/blacklist", tags=["Security"])
async def get_dev_blacklist(
    security: TokenSecurity = Depends(get_token_security)
):
    """Get the list of blacklisted developers."""
    try:
        return {
            "dev_blacklist": security.get_dev_blacklist(),
            "count": len(security.dev_blacklist)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting developer blacklist: {str(e)}")

@app.put("/security/settings", tags=["Security"])
async def update_security_settings(
    settings: SecuritySettings,
    security: TokenSecurity = Depends(get_token_security)
):
    """Update security settings."""
    try:
        security.set_min_sol_sniffer_score(settings.min_sol_sniffer_score)
        security.set_check_liquidity_lock(settings.check_liquidity_lock)
        security.set_check_bundled_supply(settings.check_bundled_supply)
        
        # Save updated config
        config_path = os.path.join(os.path.dirname(__file__), "../config/token_security.json")
        security.save_config(config_path)
        
        return {
            "message": "Security settings updated successfully",
            "settings": {
                "min_sol_sniffer_score": security.min_sol_sniffer_score,
                "check_liquidity_lock": security.check_liquidity_lock,
                "check_bundled_supply": security.check_bundled_supply
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating security settings: {str(e)}")

@app.get("/security/settings", tags=["Security"])
async def get_security_settings(
    security: TokenSecurity = Depends(get_token_security)
):
    """Get current security settings."""
    try:
        return {
            "min_sol_sniffer_score": security.min_sol_sniffer_score,
            "check_liquidity_lock": security.check_liquidity_lock,
            "check_bundled_supply": security.check_bundled_supply
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting security settings: {str(e)}")

# --- Shutdown hook ---

@app.on_event("shutdown")
async def shutdown_event():
    """Close all connections on shutdown."""
    global exchange_manager
    if exchange_manager:
        await exchange_manager.close_all()

# --- Main Entry Point ---

def start_api_server(port: int = 8000, development: bool = True):
    """Start the API server."""
    uvicorn.run(
        "api.brokr_api:app",
        host="0.0.0.0" if not development else "127.0.0.1",
        port=port,
        reload=development,
        log_level="info"
    )

if __name__ == "__main__":
    start_api_server()
