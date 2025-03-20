# BROKR - Cryptocurrency Trading Bot Framework

BROKR is a cryptocurrency trading bot framework that enables the development, testing, and deployment of automated trading strategies.

## Features

- **Strategy Development**: Abstract base class for creating customizable trading strategies
- **Exchange API Integration**: Connect to cryptocurrency exchanges (Binance support included)
- **Backtesting Engine**: Test strategies against historical data
- **Data Visualization**: Comprehensive visualization tools for strategy analysis
- **Performance Analytics**: Calculate ROI, max drawdown, and other performance metrics

## Getting Started

### Prerequisites

- Python 3.9+
- Git

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd BROKR
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up API keys:
   Copy the `.env.example` file to `.env` and add your exchange API keys:
   ```
   cp .env.example .env
   ```

### Running Tests

```
pytest -xvs tests/
```

### Command Line Interface

BROKR includes a command-line interface for easy access to its functionality:

```
# Show available commands
python cli.py --help

# Show information about available strategies
python cli.py info

# Run a backtest with default parameters
python cli.py backtest

# Run a backtest with custom parameters
python cli.py backtest --strategy momentum --days 365 --capital 50000 --threshold 0.02 --lookback 15
```

### Direct API Usage

You can also run backtests directly from Python code:

```
python strategy_backtest.py
```

## Project Structure

```
BROKR/
├── analysis/                # Analysis and reporting tools
├── data_ingestion/          # Data retrieval from exchanges
├── trading_strategy/        # Trading strategy implementations
│   └── strategies/          # Individual strategy implementations
├── trading_execution/       # Order execution and management
├── tests/                   # Test suite
├── .env                     # Environment variables (API keys, etc.)
├── requirements.txt         # Project dependencies
└── strategy_backtest.py     # Backtest runner script
```

## Available Strategies

- **Mean Reversion**: Trades based on price deviations from the moving average
- **Momentum**: Trades based on price momentum and trend direction

## Adding a New Strategy

1. Create a new file in `trading_strategy/strategies/`
2. Extend the `BaseStrategy` class
3. Implement the `calculate_signal` method
4. Add tests for your strategy

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## See Also

For a detailed implementation overview, see [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md).
