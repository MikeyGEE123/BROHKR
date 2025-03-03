# analysis/report_generator.py

import logging

logger = logging.getLogger(__name__)

# Try importing matplotlib; if not available, we skip plotting functions.
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class ReportGenerator:
    """
    Generates textual reports and visual plots from market analysis data.
    """

    def generate_text_report(self, analysis_results: dict):
        """
        Generates and prints a textual report from the analysis results.

        Args:
            analysis_results (dict): Analysis results from DataAnalyzer.
        """
        print("====== Market Analysis Report ======")
        for symbol, stats in analysis_results.items():
            print(f"\nSymbol: {symbol}")
            print(f"  Average Price     : {stats.get('average'):.2f}")
            print(f"  Minimum Price     : {stats.get('min'):.2f}")
            print(f"  Maximum Price     : {stats.get('max'):.2f}")
            print(f"  Volatility (Std)  : {stats.get('volatility'):.2f}")
            moving_avg = stats.get("moving_average")
            if moving_avg:
                formatted_ma = ", ".join([f"{ma:.2f}" for ma in moving_avg])
                print(f"  Moving Averages   : {formatted_ma}")
            else:
                print("  Moving Averages   : Not enough data")
        print("====================================")

    def generate_plots(self, analysis_results: dict):
        """
        Generates plots for each symbol using matplotlib (if available).

        Args:
            analysis_results (dict): Analysis results from DataAnalyzer.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib is not available. Plotting is skipped.")
            return

        for symbol, stats in analysis_results.items():
            data_points = stats.get("data_points")
            if not data_points:
                continue

            plt.figure(figsize=(8, 4))
            plt.plot(data_points, marker='o', linestyle='-', label=f'{symbol} Prices')
            
            moving_avg = stats.get("moving_average")
            if moving_avg:
                # Align moving average plot with proper x values.
                start_index = len(data_points) - len(moving_avg)
                plt.plot(range(start_index, len(data_points)), moving_avg,
                         marker='x', linestyle='--', label='Moving Average')
            
            plt.title(f"Price Trend for {symbol}")
            plt.xlabel("Data Point Index")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.show()

# Example usage for testing purposes.
if __name__ == "__main__":
    # Simulated analysis results for demonstration.
    sample_analysis = {
        "BTCUSD": {
            "average": 10100.0,
            "min": 10000.0,
            "max": 10200.0,
            "volatility": 100.0,
            "moving_average": [10033.33, 10100.0],
            "data_points": [10000, 10100, 10200],
        },
        "ETHUSD": {
            "average": 305.0,
            "min": 300.0,
            "max": 310.0,
            "volatility": 5.0,
            "moving_average": [301.67, 305.0],
            "data_points": [300, 305, 310],
        }
    }
    report_gen = ReportGenerator()
    report_gen.generate_text_report(sample_analysis)
    
    # Uncomment below to generate plots if matplotlib is installed.
    # report_gen.generate_plots(sample_analysis)
