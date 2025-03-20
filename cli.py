#!/usr/bin/env python
# cli.py - Command line interface for BROKR

import argparse
import os
import sys
from datetime import datetime, timedelta
import json

# Import strategies
try:
    from trading_strategy.strategies.mean_reversion import MeanReversionStrategy
    from trading_strategy.strategies.momentum import MomentumStrategy
    from strategy_backtest import StrategyBacktester, generate_sample_historical_data
    HAS_STRATEGIES = True
except ImportError:
    HAS_STRATEGIES = False
    print("Warning: Strategy modules not found. Not all commands will be available.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BROKR - Cryptocurrency Trading Bot CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run a backtest on historical data')
    backtest_parser.add_argument('--strategy', choices=['mean_reversion', 'momentum'], default='mean_reversion',
                                help='Trading strategy to test')
    backtest_parser.add_argument('--days', type=int, default=180, help='Number of days of historical data')
    backtest_parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    backtest_parser.add_argument('--output', default='backtest_results', help='Output directory for results')
    
    # Parameters specific to Mean Reversion strategy
    backtest_parser.add_argument('--threshold', type=float, default=0.03, 
                                help='Price deviation threshold (for Mean Reversion)')
    backtest_parser.add_argument('--window', type=int, default=14, 
                                help='Window size for moving average (for Mean Reversion)')
    
    # Parameters specific to Momentum strategy
    backtest_parser.add_argument('--lookback', type=int, default=10,
                                help='Lookback period (for Momentum)')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show information about available strategies')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show the version of BROKR')
    
    return parser.parse_args()


def run_backtest(args):
    """Run backtest with provided arguments."""
    if not HAS_STRATEGIES:
        print("Error: Strategy modules not available. Cannot run backtest.")
        return 1
    
    print(f"Running backtest with {args.strategy} strategy...")
    print(f"- Initial capital: ${args.capital:,.2f}")
    print(f"- Historical data: {args.days} days")
    
    # Create the strategy
    if args.strategy == 'mean_reversion':
        print(f"- Using Mean Reversion with threshold={args.threshold}, window={args.window}")
        strategy = MeanReversionStrategy(threshold=args.threshold, window_size=args.window)
    else:  # momentum
        print(f"- Using Momentum with lookback={args.lookback}, threshold={args.threshold}")
        strategy = MomentumStrategy(lookback_period=args.lookback, threshold=args.threshold)
    
    # Generate historical data
    print("\nGenerating sample historical data...")
    historical_data = generate_sample_historical_data(days=args.days)
    
    # Run backtest
    backtester = StrategyBacktester(strategy, initial_capital=args.capital)
    results = backtester.run_backtest(historical_data)
    
    # Generate output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output}/{args.strategy}_{timestamp}"
    output_files = backtester.generate_report(results, output_dir)
    
    # Save configuration
    config = {
        'strategy': args.strategy,
        'initial_capital': args.capital,
        'days': args.days,
        'timestamp': timestamp,
        'parameters': {
            'threshold': args.threshold,
            'window': args.window if args.strategy == 'mean_reversion' else None,
            'lookback': args.lookback if args.strategy == 'momentum' else None
        }
    }
    
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nBacktest complete!")
    print(f"Results saved to {output_dir}:")
    for file in output_files:
        print(f"- {file}")
        
    print(f"ROI: {results['roi']:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Number of Trades: {len(results['trades'])}")
    
    return 0


def show_info():
    """Display information about available strategies."""
    print("BROKR - Available Trading Strategies")
    print("\n1. Mean Reversion Strategy")
    print("   Description: Trades based on price deviations from the moving average")
    print("   Parameters:")
    print("     - threshold: Percentage deviation from MA to trigger signals (default: 0.05)")
    print("     - window_size: Number of periods for moving average calculation (default: 10)")
    print("   Signals:")
    print("     - BUY when price < MA * (1 - threshold)")
    print("     - SELL when price > MA * (1 + threshold)")
    print("     - HOLD when price is near the MA")
    
    print("\n2. Momentum Strategy")
    print("   Description: Trades based on price momentum and trend direction")
    print("   Parameters:")
    print("     - lookback_period: Number of periods to calculate momentum (default: 10)")
    print("     - threshold: Sensitivity of the momentum signal (default: 0.01)")
    print("   Signals:")
    print("     - BUY when price > previous average * (1 + threshold)")
    print("     - SELL when price < previous average * (1 - threshold)")
    print("     - HOLD when price change is within the threshold")
    
    return 0


def show_version():
    """Display the current version of BROKR."""
    print("BROKR v0.1.0")
    print("Cryptocurrency Trading Bot Framework")
    return 0


def main():
    """Main entry point for the CLI."""
    args = parse_arguments()
    
    if args.command == 'backtest':
        return run_backtest(args)
    elif args.command == 'info':
        return show_info()
    elif args.command == 'version':
        return show_version()
    else:
        print("Error: Please specify a command. Use --help for available commands.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
