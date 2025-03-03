# integration_test.py

import asyncio
import logging

# --- Import configuration and logging setup
from config import initialize_app

# --- Import Data Ingestion Modules
from data_ingestion.data_manager import DataManager

# --- Import Trading Strategy Modules
from trading_strategy.strategy_manager import StrategyManager
from trading_strategy.strategies.momentum import MomentumStrategy
from trading_strategy.strategies.mean_reversion import MeanReversionStrategy

# --- Import Trading Execution Modules
from trading_execution.order_manager import OrderManager

# --- Import Analysis & Reporting Modules
from analysis.data_analyzer import DataAnalyzer
from analysis.report_generator import ReportGenerator


# Dummy connector to simulate data ingestion without calling a real API.
class DummyExchangeAPIConnector:
    async def fetch_market_data(self, endpoint: str, params: dict = None):
        # Simulated market data (as strings/numbers) matching expected format.
        return {
            "ticker": {"symbol": "BTCUSD", "price": "10000"},
            "timestamp": "2023-10-15T10:00:00Z"
        }


# Dummy exchange interface to simulate order execution.
class DummyExchangeInterface:
    async def create_order(self, order_type: str, symbol: str, side: str, quantity: float, price: float = None):
        return {"order_id": "dummy_order_001", "status": "created"}

    async def cancel_order(self, order_id: str, symbol: str):
        return {"order_id": order_id, "status": "cancelled"}

    async def get_order_status(self, order_id: str, symbol: str):
        return {"order_id": order_id, "status": "filled"}


async def integration_flow():
    # --- Step 1: Initialize configuration and logging.
    settings = initialize_app()
    logger = logging.getLogger(__name__)
    logger.info("Starting integration test...")

    # --- Step 2: Data Ingestion (using a dummy connector).
    dummy_connector = DummyExchangeAPIConnector()
    dm = DataManager([dummy_connector])
    ingestion_results = await dm.fetch_all_market_data()
    logger.info(f"Data Ingestion Results: {ingestion_results}")

    # --- Step 3: Trading Strategy Evaluation.
    # For strategy evaluation, we simulate a market data input with historical prices.
    sample_market_data = {
        "historical_prices": [10000, 10050, 10020, 10080, 10100],
        "current_price": 10120,
    }
    # Set up strategy manager and register strategies.
    strategy_manager = StrategyManager()
    strategy_manager.register_strategy(MomentumStrategy(lookback_period=3))
    strategy_manager.register_strategy(MeanReversionStrategy(threshold=0.02, window_size=3))
    signals = strategy_manager.evaluate_strategies(sample_market_data)
    logger.info(f"Trading Signals: {signals}")

    # --- Step 4: Trading Execution.
    # If any strategy returns a "buy" signal, simulate placing an order.
    if any(signal == "buy" for signal in signals.values()):
        dummy_exchange = DummyExchangeInterface()
        order_manager = OrderManager(dummy_exchange)
        order_response = await order_manager.place_order(side="buy", symbol="BTCUSD", quantity=0.01)
        logger.info(f"Order Response: {order_response}")
    else:
        logger.info("No 'buy' signal received; skipping order placement.")

    # --- Step 5: Data Analysis and Reporting.
    analyzer = DataAnalyzer()
    # For analysis, we can use the ingestion results (which is a list of normalized records).
    analysis_results = analyzer.analyze_data(ingestion_results)
    logger.info(f"Analysis Results: {analysis_results}")

    report_gen = ReportGenerator()
    report_gen.generate_text_report(analysis_results)

    logger.info("Integration test completed.")


def main():
    asyncio.run(integration_flow())


if __name__ == "__main__":
    main()
